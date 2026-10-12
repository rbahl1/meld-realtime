"""words versus tone: jensen-shannon divergence between the text and audio
posteriors, the flag, the direction rule, and the keyphrase slot filler
(design 1.3 step 6, 1.9, 6.2 WP5). no model, no parser."""

from __future__ import annotations

import re

import numpy as np

from meld_rt.labels import EMOTIONS, EMO_INDEX, VALENCE, sv_probs_to_meld7

DIRECTIONS = ("none", "tone_more_negative", "tone_more_positive", "tone_surprised", "mixed")
SURPRISE_AUDIO_MIN = 0.4   # tone_surprised: audio surprise above this ...
SURPRISE_TEXT_MAX = 0.2    # ... and text surprise below this
VALENCE_MARGIN = 0.1       # |valence(audio) - valence(text)| at or below this has no clear sign
MAX_KEYPHRASE_WORDS = 4
# attached in front of the chosen phrase when it directly precedes it and the
# total stays within MAX_KEYPHRASE_WORDS, so "the GR-6 system" rather than "GR-6 system"
LEAD_WORDS = frozenset("the a an my your our their his its".split())

_VALENCE_VEC = np.array([VALENCE[e] for e in EMOTIONS], dtype=np.float64)


def _normalize(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64).ravel(), 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(p.shape, 1.0 / p.size)


def jsd(p, q) -> float:
    """jensen-shannon divergence in bits: 0 for equal distributions, 1 for
    disjoint supports, so it is already in [0, 1]; clipped for rounding."""
    p, q = _normalize(p), _normalize(q)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {q.shape}")
    m = 0.5 * (p + q)

    def kl(a: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))

    return float(min(1.0, max(0.0, 0.5 * kl(p) + 0.5 * kl(q))))


def valence(p7) -> float:
    """expected valence of a 7-way posterior in MELD order, in [-1, 1]."""
    return float(np.dot(_normalize(p7), _VALENCE_VEC))


def audio7(p_audio) -> np.ndarray:
    """a 9-way sensevoice posterior renormalized over the seven named classes
    (MELD order); a 7-way MELD-order vector passes through normalized."""
    p = np.asarray(p_audio, dtype=np.float64).ravel()
    if p.size == 9:
        return _normalize(sv_probs_to_meld7(p))
    if p.size == 7:
        return _normalize(p)
    raise ValueError(f"audio posterior must have 9 or 7 entries, got {p.size}")


def direction(p_text7, p_audio7, flag: bool) -> str:
    """the direction rule. surprise is checked before valence because surprise
    has valence 0 and would otherwise read as a valence drop against happy words."""
    if not flag:
        return "none"
    pt, pa = _normalize(p_text7), _normalize(p_audio7)
    i = EMO_INDEX["surprise"]
    if pa[i] > SURPRISE_AUDIO_MIN and pt[i] < SURPRISE_TEXT_MAX:
        return "tone_surprised"
    dv = valence(pa) - valence(pt)
    if dv < -VALENCE_MARGIN:
        return "tone_more_negative"
    if dv > VALENCE_MARGIN:
        return "tone_more_positive"
    return "mixed"


def discrepancy(p_text, p_audio9, flag_threshold: float) -> tuple[float, bool, str]:
    """(jsd, flag, direction) between the 7-way text posterior and the audio
    posterior (9-way sensevoice or 7-way MELD order). flag is jsd > threshold."""
    pt, pa = _normalize(p_text), audio7(p_audio9)
    d = jsd(pt, pa)
    flag = bool(d > flag_threshold)
    return d, flag, direction(pt, pa, flag)


# function words, fillers and discourse verbs; anything else is content
STOPWORDS = frozenset("""
a an the and or but nor so yet if then than that this these those there here
i me my mine myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs themselves
who whom whose which what where when why how whatever whoever
am is are was were be been being have has had having do does did doing done
will would shall should can could may might must ought
to of in on at by for from with without about into onto over under up down out off
through across along around between among before after during until while
as per via upon within
not no nor never none nothing nobody nowhere neither
all any both each every few more most much many other another some such only own same
very too also just really quite rather pretty almost enough still even ever again
yes yeah yep nope okay ok oh ah um uh hmm huh hey well right sure like
know knew think thought mean meant guess say said says see saw look wanna gonna gotta kinda sorta
let lets tell told
i'm i've i'd i'll you're you've you'd you'll he's she's it's we're we've we'd we'll they're they've they'd they'll
isn't aren't wasn't weren't don't doesn't didn't won't wouldn't can't couldn't shouldn't mustn't
haven't hasn't hadn't that's there's here's what's where's who's how's let's
im ive id ill youre youve youd youll hes shes its were weve wed well theyre theyve theyd theyll
isnt arent wasnt werent dont doesnt didnt wont wouldnt cant couldnt shouldnt havent hasnt hadnt thats
""".split())

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")


def _is_stop(w: str) -> bool:
    lw = w.lower().replace("’", "'")
    return lw in STOPWORDS or (len(lw) == 1 and not lw.isdigit())


def keyphrase(text: str) -> str | None:
    """longest run of content words (at most 4) after stopword removal, as the
    exact substring of the input. runs break at stopwords and at punctuation.
    ties: more characters, then earliest. a leading article or possessive is
    kept when the phrase stays within 4 words. None when nothing is left."""
    if not text:
        return None
    runs: list[list[re.Match]] = []
    cur: list[re.Match] = []
    prev_end: int | None = None
    matches = list(_WORD.finditer(text))
    for m in matches:
        sep = text[prev_end:m.start()] if prev_end is not None else ""
        prev_end = m.end()
        if _is_stop(m.group(0)):
            if cur:
                runs.append(cur)
            cur = []
            continue
        if cur and sep.strip() != "":
            runs.append(cur)
            cur = []
        cur.append(m)
    if cur:
        runs.append(cur)
    best = None
    best_key = None
    for run in runs:
        n = min(len(run), MAX_KEYPHRASE_WORDS)
        for start in range(len(run) - n + 1):
            a, b = run[start], run[start + n - 1]
            key = (n, b.end() - a.start(), -a.start())
            if best_key is None or key > best_key:
                best_key, best = key, (a, b, n)
    if best is None:
        return None
    a, b, n = best
    i = matches.index(a)
    if n < MAX_KEYPHRASE_WORDS and i > 0:
        lead = matches[i - 1]
        if lead.group(0).lower() in LEAD_WORDS and text[lead.end():a.start()].strip() == "":
            a = lead
    return text[a.start():b.end()]
