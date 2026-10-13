"""streaming reply generator (spec 4): a small instruct LLM in plain onnxruntime
with a kv-cache loop, greedy and deterministic, prompted with the evidence of
one final state.

- build_messages(state, low_conf) turns a state dict into a ChatML prompt, the
  grounding list (state fields put into the prompt, in the selector's format)
  and the emotion classes the reply may name.
- ReplyGenerator.generate streams whole words through on_delta. two masks act
  on the logits: the lexicon mask (whole-word tokens of every class not in
  `allowed`, with and without a leading space, lower, capitalized and upper case)
  and the register mask (tokens with "!", an ascii control byte such as a
  newline or a tab, or a non-ascii byte, and every special token but the two
  stops). a word guard covers the emotion words the tokenizer splits into
  pieces: when a finished word is a lexicon word of a class not in `allowed`,
  the loop rolls the kv cache back to the word's first token, bans that token at
  that position and decodes again. guard phrases (the notes copied into the
  reply) and runs of transcript words are rolled back the same way, on a budget
  of their own. a word is sent to on_delta only after it passed, so nothing sent
  is taken back; when a budget is spent, the reply ends at the last word that
  passed (stop_reason guard).
- the system prompt is the same for every reply, so its kv cache is computed
  once and reused; only the user turn is prefilled per reply.
- check_reply(text, allowed) checks the final text (spec 4, response.checks).

no torch; onnxruntime and tokenizers are imported when a generator is built."""

from __future__ import annotations

import functools
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meld_rt.labels import EMOTIONS
from meld_rt.response.lexicon import LEXICON, emotion_words_in, forbidden_classes

ROOT = Path(__file__).resolve().parents[2]
LLM_ROOT = ROOT / "models" / "llm"

# hf onnx-community candidates (spec 4), pinned revisions. files land in
# models/llm/<name>/ with the onnx files under onnx/, as in the hf repo
CANDIDATES = {
    "Qwen2.5-0.5B-Instruct": "22942cb7d7ba4cc81bb4673549ca4d4614469b5e",
    "Qwen2.5-1.5B-Instruct": "d54957e3af1be04546436e874a60fe386041b244",
}
CANDIDATE_FILES = ("model_int8.onnx", "model_q4.onnx")
SIDE_FILES = ("tokenizer.json", "config.json", "generation_config.json")
# the deployed file. tools/bench_generator.py applies the choice rule of spec 4
# and prints the two values to put here after the final bench (phase 2)
DEFAULT_MODEL = "Qwen2.5-0.5B-Instruct"
DEFAULT_ONNX = "model_int8.onnx"
DEFAULT_MODEL_DIR = LLM_ROOT / DEFAULT_MODEL

MAX_NEW_TOKENS = 40
MAX_REPLY_WORDS = 20            # the prompt's limit, checked by check_reply
MODALITY_MIN_P = 0.4            # a modality's top class enters `allowed` at this probability
MAX_TRANSCRIPT_WORDS = 60       # longer transcripts are cut in the prompt
MAX_BACKTRACKS = 6              # rollbacks per reply in each budget: lexicon words; guard phrases and transcript echoes
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
META_WORDS = frozenset({"text", "audio", "posterior", "posteriors", "model", "models", "probability", "probabilities"})
# prompt phrases that a reply should never repeat (the last two: the system prompt copied)
ECHO_PHRASES = ("overall reading", "you may name", "any feeling", "example reply", "example notes", "the notes", "the person",
                "low confidence", "medium confidence", "high confidence", "desk robot", "each message gives")
# word runs the decoder rolls back like a forbidden emotion word: the start of the
# notes copied into the reply. a run that could still grow into one is held back from
# on_delta, so phrases that start with a common first word ("you may name") are left
# to check_reply. a copy of the system prompt ("desk robot") is left to check_reply too:
# rolled back, the 0.5B model rewords the copy ("You are a small robot. A person just
# spoke to you ...") instead of answering (notes_C section 8)
GUARD_PHRASES = tuple(tuple(p.split()) for p in (
    "overall reading", "example reply", "example notes", "the notes", "the person", "low confidence", "medium confidence",
    "high confidence", "they said", "the words sound", "the words suggest", "the words give", "the words were",
    "the voice sounds", "the voice suggests", "the voice gives", "the face looks", "the face suggests", "the face gives",
    "no face was seen"))
# a run of this many consecutive transcript words in the reply is rolled back like a guard
# phrase (the prompt says not to repeat their words); 0 turns it off
ECHO_NGRAM = 4
ASR_DEGRADED = ("asr_failed", "low_asr_conf")

DIRECTION_VOICE = {
    "tone_more_negative": "The voice sounds more negative than the words.",
    "tone_more_positive": "The voice sounds more positive than the words.",
    "tone_surprised": "The voice sounds more surprised than the words.",
    "mixed": "The voice and the words do not match.",
}
DIRECTION_FACE = {
    "face_more_negative": "The face looks more negative than the words.",
    "face_more_positive": "The face looks more positive than the words.",
    "face_surprised": "The face looks more surprised than the words.",
    "mixed": "The face and the words do not match.",
}
# a modality line and its direction sentence read different posteriors: the voice line the raw
# sensevoice mass, the words-voice direction the posterior renormalized over the seven classes; the
# face line the calibrated face posterior, the words-face direction the raw fer posterior. when the
# line says no clear emotion and a direction is flagged, the two would contradict each other.
# GAP_STYLE "lean" makes them one sentence; "direction_posterior" reads the line from the posterior
# the direction comes from (the lean sentence when that is still unclear)
VOICE_LEAN = {
    "tone_more_negative": "The voice gives no clear emotion, but it leans more negative than the words.",
    "tone_more_positive": "The voice gives no clear emotion, but it leans more positive than the words.",
    "tone_surprised": "The voice gives no clear emotion, but it leans toward surprise more than the words do.",
    "mixed": "The voice gives no clear emotion, and it does not quite match the words.",
}
FACE_LEAN = {
    "face_more_negative": "The face gives no clear emotion, but it looks more negative than the words.",
    "face_more_positive": "The face gives no clear emotion, but it looks more positive than the words.",
    "face_surprised": "The face gives no clear emotion, but it looks more surprised than the words.",
    "mixed": "The face gives no clear emotion, and it does not quite match the words.",
}
GAP_STYLES = ("lean", "direction_posterior")
GAP_STYLE = "lean"              # chosen on dev (notes_C section 8)


# ---------------------------------------------------------------- prompt

def _get(d: dict | None, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def _top(p: dict | None) -> tuple[str, float] | None:
    """(class, probability) of the largest MELD class in an emotion dict; keys
    outside the seven classes (the voice's other and unknown) count as mass
    but never win."""
    if not p:
        return None
    vals = [(float(p.get(e, 0.0)), -i, e) for i, e in enumerate(EMOTIONS)]
    v, _, e = max(vals)
    return e, v


PHRASES = {  # subject: (no clear emotion, neutral, another class)
    "words": ("The words give no clear emotion.", "The words sound neutral.", "The words suggest {}."),
    "voice": ("The voice gives no clear emotion.", "The voice sounds neutral.", "The voice suggests {}."),
    "face": ("The face gives no clear emotion.", "The face looks neutral.", "The face suggests {}."),
}


def _modality_line(subject: str, top: tuple[str, float] | None) -> str:
    unclear, neutral, other = PHRASES[subject]
    if top is None or top[1] < MODALITY_MIN_P:
        return unclear
    return neutral if top[0] == "neutral" else other.format(top[0])


_ASCII = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": "'", "\u201d": "'", '"': "'", "\u2014": " - ",
                        "\u2013": " - ", "\u2026": "..."})


def _transcript(text: str) -> str:
    """the transcript as it goes into the prompt: curly quotes and dashes to ascii,
    double quotes to single (the prompt quotes it), cut at MAX_TRANSCRIPT_WORDS."""
    words = str(text or "").translate(_ASCII).split()
    if len(words) > MAX_TRANSCRIPT_WORDS:
        words = words[:MAX_TRANSCRIPT_WORDS] + ["..."]
    return " ".join(words)


@functools.lru_cache(maxsize=4)
def _manifest_thresholds(path: str) -> dict:
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {"core": m.get("thresholds") or {}, "vision": (m.get("vision") or {}).get("thresholds") or {}}


def default_high_conf(vision_mode: bool, manifest: Path | None = None) -> float:
    """the high_conf the service's selector uses: the vision-mode thresholds of
    heads/manifest.json in vision mode, else the core ones, else the selector's
    default 0.70."""
    t = _manifest_thresholds(str(manifest or ROOT / "heads" / "manifest.json"))
    for sec in (("vision", "core") if vision_mode else ("core",)):
        v = (t.get(sec) or {}).get("high_conf")
        if v is not None:
            return float(v)
    return 0.70


def _renormalized(p: dict | None) -> dict | None:
    """an emotion dict renormalized over the seven MELD classes (the voice's other and
    unknown dropped), as discrepancy.audio7 does for the words-voice direction."""
    if not p:
        return None
    tot = sum(float(p.get(e, 0.0)) for e in EMOTIONS)
    return {e: float(p.get(e, 0.0)) / tot for e in EMOTIONS} if tot > 0 else None


def evidence(state: dict, low_conf: float, high_conf: float,
             gap_style: str = GAP_STYLE) -> tuple[list[str], list[str], list[str], bool]:
    """(prompt lines, grounding, allowed classes, ask) for one state dict."""
    if gap_style not in GAP_STYLES:
        raise ValueError(f"gap_style must be one of {GAP_STYLES}")
    lines: list[str] = []
    grounding: list[str] = []
    degraded = [d for d in (state.get("degraded") or []) if d in ASR_DEGRADED]
    said = _transcript(_get(state, "asr.text", ""))
    caught = bool(said) and not degraded
    conf = float(_get(state, "fused.confidence", 0.0))
    label = str(_get(state, "fused.label", "neutral"))
    calibrated = bool(_get(state, "fused.calibrated", False))
    low = conf < low_conf or not caught

    if caught:
        lines.append(f'They said: "{said}"')
        grounding.append(f"asr.text='{said}'")
    else:
        lines.append("The words were not caught.")
        grounding.append(f"degraded={degraded[0]}" if degraded else "asr.text=''")

    tops = {}
    if caught:
        tops["text"] = _top(_get(state, "text.emotion"))
        lines.append(_modality_line("words", tops["text"]))
        if tops["text"]:
            grounding.append(f"text.emotion={tops['text'][0]}@{tops['text'][1]:.2f}")
    def flagged(flag: str, path: str, sentences: dict) -> str | None:
        d = str(_get(state, path, "none"))
        return d if caught and _get(state, flag, False) and d in sentences else None

    def modality(subject: str, key: str, p: dict | None, direction: str | None, path: str, p_direction: dict | None,
                 path_direction: str, lean: dict) -> bool:
        """the modality's line and grounding; True when the direction went into the line (lean)."""
        tops[key] = _top(p)
        if direction and gap_style == "direction_posterior":
            top_d = _top(p_direction)
            if top_d is not None and top_d[1] >= MODALITY_MIN_P:
                tops[key], path = top_d, path_direction
        line = _modality_line(subject, tops[key])
        merged = direction is not None and line == PHRASES[subject][0]
        lines.append(lean[direction] if merged else line)
        if tops[key]:
            grounding.append(f"{path}={tops[key][0]}@{tops[key][1]:.2f}")
        return merged

    voice_d = flagged("discrepancy.flag", "discrepancy.direction", DIRECTION_VOICE)
    audio = _get(state, "audio.emotion")
    voice_in_line = modality("voice", "audio", audio, voice_d, "audio.emotion", _renormalized(audio), "audio.emotion7", VOICE_LEAN)
    vision = state.get("vision")
    face_d = face_in_line = None
    if isinstance(vision, dict) and vision.get("present"):
        face_d = flagged("discrepancy.face_flag", "discrepancy.face_direction", DIRECTION_FACE)
        face_in_line = modality("face", "vision", vision.get("emotion"), face_d, "vision.emotion", vision.get("raw"), "vision.raw",
                                FACE_LEAN)
    else:
        lines.append("No face was seen.")
        if isinstance(vision, dict):
            grounding.append("vision.present=false")

    for d, in_line, sentences, path in ((voice_d, voice_in_line, DIRECTION_VOICE, "discrepancy.direction"),
                                        (face_d, face_in_line, DIRECTION_FACE, "discrepancy.face_direction")):
        if d:
            if not in_line:
                lines.append(sentences[d])
            grounding.append(f"{path}={d}")

    if low:
        lines.append("Overall reading: unclear, low confidence.")
        if caught:
            grounding.insert(0, f"fused.confidence={conf:.2f}<{low_conf:.2f}")
        allowed: list[str] = []
    else:
        word = "high" if conf >= high_conf and calibrated else "medium"
        lines.append(f"Overall reading: {label}, {word} confidence.")
        grounding.insert(0, f"fused.label={label}@{conf:.2f}")
        cls = {label} | {t[0] for t in tops.values() if t is not None and t[1] >= MODALITY_MIN_P}
        allowed = [e for e in EMOTIONS if e in cls]
    named = [e for e in allowed if LEXICON[e]]
    if not caught:
        lines.append("Ask them to say it again, and do not name any feeling.")
    elif low:
        lines.append("Reply with one short question about what they said, and do not name any feeling.")
    elif named:
        which = "this feeling" if len(named) == 1 else "these feelings"
        lines.append(f"Reply to them in one or two short sentences. You may name only {which}: {', '.join(named)}.")
    else:
        lines.append("Reply to them in one or two short sentences, and do not name any feeling.")
    return lines, grounding, allowed, low


def _example(transcript: str, words: dict | None, voice: dict, face: dict | None, label: str, conf: float,
             reply: str, **disc) -> tuple[str, str, list[str]]:
    st = {"asr": {"text": transcript}, "text": {"emotion": words or {}}, "audio": {"emotion": voice},
          "vision": {"present": face is not None, "emotion": face or {}},
          "fused": {"label": label, "confidence": conf, "calibrated": True}, "discrepancy": disc, "degraded": []}
    lines, _, allowed, _ = evidence(st, 0.47, 0.73)
    return "\n".join(lines), reply, allowed


# the few-shot examples, rendered by the same function as a live state. a reply shares no run of
# ECHO_NGRAM words with its transcript (the prompt says not to repeat their words; tested). example
# 1's reply was chosen on dev among two that follow that rule (notes_C section 8)
EXAMPLES = [
    _example("I got the job at the bakery.", {"joy": 0.81}, {"joy": 0.52, "unknown": 0.4}, {"joy": 0.77}, "joy", 0.86,
             "You sound happy about the new job, and you look it too."),
    _example("It went fine, I guess.", {"neutral": 0.62}, {"neutral": 0.1, "unknown": 0.85}, {"sadness": 0.58}, "sadness", 0.55,
             "You say it went fine, but you look a little sad. What happened?",
             face_flag=True, face_direction="face_more_negative"),
    _example("so that is the plan then", {"neutral": 0.31}, {"neutral": 0.12, "unknown": 0.8}, None, "neutral", 0.38,
             "What do you want to happen next?"),
]

RULES = (
    "You are a small desk robot. A person has just spoken to you, and you heard them and maybe saw their face. "
    "Each message gives notes on what they said and how they seemed. Answer the person, not the notes. "
    "Reply in one or two short sentences, at most 20 words in all. Speak to them as \"you\". "
    "Talk to them, not about them, and do not repeat their words back to them. "
    "Use only what the notes say. Do not use exclamation marks or emoji. "
    "Do not mention the notes, models, numbers or probabilities, and do not use the words \"text\", \"audio\" or \"posterior\". "
    "Name a feeling only when the notes allow it, and follow the last line of the notes."
)
# where the examples go: "system" (inside the system prompt, spec 4) or "turns"
# (as earlier user and assistant turns). both are fixed, so both sit in the cached prefix
EXAMPLE_PLACES = ("system", "turns")
EXAMPLES_IN = "system"


def system_prompt(examples_in: str = EXAMPLES_IN) -> str:
    if examples_in == "turns":
        return RULES
    return RULES + "\n\n" + "\n\n".join(f"Example notes:\n{ev}\nExample reply: {rep}" for ev, rep, _ in EXAMPLES)


SYSTEM_PROMPT = system_prompt()


def build_messages(state: dict, low_conf: float, high_conf: float | None = None, examples_in: str = EXAMPLES_IN,
                   gap_style: str = GAP_STYLE) -> tuple[list[dict], list[str], list[str]]:
    """(chat messages, grounding, allowed) for one state dict (spec 4). allowed is
    empty when the fused confidence is below low_conf or the words were not
    caught; the prompt then asks a question and names no feeling. high_conf
    defaults to the selector's threshold (default_high_conf)."""
    if examples_in not in EXAMPLE_PLACES:
        raise ValueError(f"examples_in must be one of {EXAMPLE_PLACES}")
    if high_conf is None:
        high_conf = default_high_conf("vision" in state)
    lines, grounding, allowed, _ = evidence(state, float(low_conf), float(high_conf), gap_style)
    messages = [{"role": "system", "content": system_prompt(examples_in)}]
    if examples_in == "turns":
        for ev, rep, _ in EXAMPLES:
            messages += [{"role": "user", "content": ev}, {"role": "assistant", "content": rep}]
    messages.append({"role": "user", "content": "\n".join(lines)})
    return messages, grounding, allowed


def transcript_ngrams(messages: list[dict], n: int) -> tuple[tuple[str, ...], ...]:
    """every run of n consecutive lowercase words of the transcript quoted on the first
    line of the last user turn (`They said: "..."`); empty when n is 0 or there is none."""
    first = str(messages[-1].get("content", "")).split("\n", 1)[0] if messages else ""
    m = re.match(r'They said: "(.*)"$', first)
    if n <= 0 or m is None:
        return ()
    w = [x.lower() for x in _WORDS.findall(m.group(1))]
    return tuple(dict.fromkeys(tuple(w[i:i + n]) for i in range(len(w) - n + 1)))


def chatml(messages: list[dict]) -> tuple[str, str]:
    """(cached prefix, per-reply rest) of the ChatML prompt: every turn but the
    last, then the last turn and the open assistant turn."""
    if not messages or messages[0]["role"] != "system":
        raise ValueError("the first message must be the system prompt")

    def turn(m: dict) -> str:
        return f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"

    return "".join(turn(m) for m in messages[:-1]), turn(messages[-1]) + "<|im_start|>assistant\n"


# ---------------------------------------------------------------- checks

_WORDS = re.compile(r"[A-Za-z]+")


def check_reply(text: str, allowed) -> dict:
    """checks of a final reply (spec 4): word count against the prompt's limit,
    exclamation mark, emotion words of classes not in allowed, empty. also the
    meta words the prompt forbids, phrases copied from the prompt, non-ascii
    characters and non-printable ones (control bytes, line breaks, tabs). ok is
    all of them."""
    t = str(text or "").strip()
    bad = set(forbidden_classes(allowed))
    forbidden = [w for w, c in emotion_words_in(t) if c in bad]
    meta = [w for w in _WORDS.findall(t) if w.lower() in META_WORDS]
    low = " ".join(_WORDS.findall(t.lower()))
    echo = [p for p in ECHO_PHRASES if f" {p} " in f" {low} "]
    n = len(t.split())
    out = {"n_words": n, "over_length": n > MAX_REPLY_WORDS, "exclamation": "!" in t, "forbidden_words": forbidden,
           "empty": not t, "meta_words": meta, "prompt_echo": echo, "non_ascii": not t.isascii(),
           "non_printable": not t.isprintable()}
    out["ok"] = not (out["over_length"] or out["exclamation"] or forbidden or out["empty"] or meta or echo or out["non_ascii"]
                     or out["non_printable"])
    return out


# ---------------------------------------------------------------- token masks

def _byte_decoder() -> dict[str, int]:
    """inverse of the byte-level bpe alphabet (gpt-2 bytes_to_unicode)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def variants(word: str) -> list[str]:
    """the spellings a mask covers: lower, capitalized, upper, each with and without a leading space."""
    forms = dict.fromkeys([word, word.capitalize(), word.upper()])
    return [p + f for f in forms for p in ("", " ")]


@dataclass
class TokenMasks:
    by_class: dict[str, np.ndarray]     # whole-word token ids per class
    register: np.ndarray                 # ids masked in every reply
    stop: np.ndarray                     # stop token ids
    n_split_variants: int                # variants that are not one token (left to the word guard)


def token_masks(tok, n_logits: int) -> TokenMasks:
    """precompute the mask ids from a tokenizers.Tokenizer (spec 4)."""
    by_class: dict[str, np.ndarray] = {}
    n_split = 0
    for c, words in LEXICON.items():
        ids = set()
        for w in words:
            for v in variants(w):
                enc = tok.encode(v, add_special_tokens=False).ids
                if len(enc) == 1:
                    ids.add(enc[0])
                else:
                    n_split += 1
        by_class[c] = np.array(sorted(ids), dtype=np.int64)
    dec = _byte_decoder()
    vocab = tok.get_vocab(with_added_tokens=True)
    special = {t.content for t in tok.get_added_tokens_decoder().values()} if hasattr(tok, "get_added_tokens_decoder") else set()
    stop = [vocab[s] for s in STOP_TOKENS if s in vocab]
    reg = np.zeros(n_logits, dtype=bool)
    reg[len(vocab):] = True                  # padding rows of the output layer
    for s, i in vocab.items():
        if i >= n_logits:
            continue
        if s in special or any(ch not in dec for ch in s):
            reg[i] = i not in stop
            continue
        b = bytes(dec[ch] for ch in s)
        # "!", every ascii control byte (newline, carriage return and tab among them, 0x7f too) and non-ascii bytes
        if b"!" in b or any(x < 0x20 or x >= 0x7f for x in b):
            reg[i] = True
    return TokenMasks(by_class, np.flatnonzero(reg).astype(np.int64), np.array(stop, dtype=np.int64), n_split)


# ---------------------------------------------------------------- generator

@dataclass
class GenResult:
    text: str
    n_tokens: int
    first_token_ms: float
    total_ms: float
    stop_reason: str                      # eos | max_tokens | deadline | guard (a violation with no rollback left)
    masked_steps: int                     # steps where the lexicon mask changed the greedy token
    prompt_tokens: int = 0                # tokens prefilled for this reply (the cached system prompt not counted)
    steps: int = 0                        # tokens chosen, including those a rollback undid
    backtracks: int = 0                   # rollbacks, both budgets
    register_masked_steps: int = 0        # steps where the register mask changed the greedy token
    deltas: list[str] = field(default_factory=list)
    word_backtracks: int = 0              # rollbacks of a lexicon word (the rest: guard phrases and transcript echoes)
    decode_passes: int = 0                # forward passes after the first token (rolled-back tokens and the end token's included)
    rework_passes: int = 0                # of those, the passes of tokens a rollback undid


def resolve_model(model_dir: str | Path | None = None, onnx_file: str | None = None) -> tuple[Path, Path]:
    """(onnx path, folder with tokenizer.json). model_dir is a model folder
    (models/llm/<name>, onnx files in it or under onnx/) or an .onnx file."""
    p = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    if p.suffix == ".onnx":
        onnx_path = p
        tok_dir = p.parent.parent if p.parent.name == "onnx" else p.parent
    else:
        name = onnx_file or DEFAULT_ONNX
        onnx_path = next((q for q in (p / "onnx" / name, p / name) if q.exists()), p / "onnx" / name)
        tok_dir = p
    if not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} not found (tools/fetch_models.py)")
    if not (tok_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"{tok_dir / 'tokenizer.json'} not found")
    return onnx_path, tok_dir


def last_position_model(path: Path) -> bytes | None:
    """the graph with the lm head applied to the last position only, serialized; None when
    the logits do not come from a MatMul over an activation or the model is too large to
    serialize. the prefill then skips the lm head on every other prompt position (the
    0.5B lm head is 151936 x 896), and the last position's logits are the same."""
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(str(path), load_external_data=False)
    g = m.graph
    if any(t.data_location == TensorProto.EXTERNAL for t in g.initializer):
        return None
    prod = {o: n for n in g.node for o in n.output}
    lm = prod.get("logits")
    if lm is None or lm.op_type != "MatMul" or lm.input[0] in {t.name for t in g.initializer}:
        return None
    g.initializer.extend([helper.make_tensor("last_pos_starts", TensorProto.INT64, [1], [-1]),
                          helper.make_tensor("last_pos_ends", TensorProto.INT64, [1], [2 ** 62]),
                          helper.make_tensor("last_pos_axes", TensorProto.INT64, [1], [1])])
    cut = helper.make_node("Slice", [lm.input[0], "last_pos_starts", "last_pos_ends", "last_pos_axes"], ["last_pos_hidden"],
                           name="last_pos_slice")
    lm.input[0] = "last_pos_hidden"
    g.node.insert(list(g.node).index(lm), cut)
    try:
        return m.SerializeToString()
    except ValueError:      # over the 2 GB protobuf limit
        return None


class ReplyGenerator:
    """greedy kv-cache decoding of one onnx causal LM. not thread safe: one
    generate at a time."""

    def __init__(self, model_dir: str | Path | None = None, threads: int = 2, max_new_tokens: int = MAX_NEW_TOKENS,
                 onnx_file: str | None = None, max_backtracks: int = MAX_BACKTRACKS, last_logits: bool = True,
                 echo_ngram: int = ECHO_NGRAM):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.onnx_path, self.tok_dir = resolve_model(model_dir, onnx_file)
        self.name = f"{self.tok_dir.name}/{self.onnx_path.name}"
        self.threads, self.max_new_tokens, self.max_backtracks = int(threads), int(max_new_tokens), int(max_backtracks)
        self.echo_ngram = int(echo_ngram)
        self.tok = Tokenizer.from_file(str(self.tok_dir / "tokenizer.json"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # the int8 exports dequantize the uint8 embedding for the lm head on every call (Transpose then
        # DequantizeLinear of a constant); with the qdq transformers off, constant folding does it once
        # at load, about twice as fast per step. the session then holds the float lm head. on the int8
        # files this changes which graph rewrites run, so the logits are not the same: they move by up
        # to about 2.5 (the default rewrites themselves move them by about 6 against the unoptimized
        # graph), and a greedy reply can change (notes decision 13). the q4 files are unaffected
        so.add_session_config_entry("session.disable_quant_qdq", "1")
        t0 = time.perf_counter()
        src = last_position_model(self.onnx_path) if last_logits else None
        self.last_position_logits = src is not None
        self.sess = ort.InferenceSession(src if src is not None else str(self.onnx_path), so, providers=["CPUExecutionProvider"])
        del src
        self.load_ms = (time.perf_counter() - t0) * 1000
        self._read_graph()
        self.masks = token_masks(self.tok, self.n_logits)
        self._prefix: dict[str, tuple[int, list[np.ndarray], np.ndarray]] = {}

    def _read_graph(self) -> None:
        """input and output names from the graph, not guessed."""
        ins = {i.name: i for i in self.sess.get_inputs()}
        outs = [o.name for o in self.sess.get_outputs()]
        for need in ("input_ids", "attention_mask"):
            if need not in ins:
                raise ValueError(f"{self.onnx_path.name} has no input {need}: {sorted(ins)[:5]}")
        self.has_position_ids = "position_ids" in ins
        pat = re.compile(r"past_key_values\.(\d+)\.(key|value)$")
        past = sorted((int(m.group(1)), m.group(2) == "value", n) for n in ins if (m := pat.match(n)))
        if not past or len(past) % 2:
            raise ValueError(f"{self.onnx_path.name}: no past_key_values inputs")
        self.past_names = [n for _, _, n in past]
        present = {(int(m.group(1)), m.group(2)): n for n in outs if (m := re.match(r"present\.(\d+)\.(key|value)$", n))}
        self.present_names = [present[(i, "value" if v else "key")] for i, v, _ in past]
        if "logits" not in outs:
            raise ValueError(f"{self.onnx_path.name} has no logits output")
        self.out_names = ["logits"] + self.present_names
        shape = ins[self.past_names[0]].shape          # [batch, kv heads, past, head dim]
        self.kv_heads, self.head_dim = int(shape[1]), int(shape[3])
        self.kv_dtype = np.float16 if "float16" in ins[self.past_names[0]].type else np.float32
        lshape = next(o for o in self.sess.get_outputs() if o.name == "logits").shape
        self.n_logits = int(lshape[-1]) if isinstance(lshape[-1], int) else self.tok.get_vocab_size(with_added_tokens=True)
        self.n_layers = len(self.past_names) // 2

    def _empty_past(self) -> list[np.ndarray]:
        return [np.zeros((1, self.kv_heads, 0, self.head_dim), dtype=self.kv_dtype) for _ in self.past_names]

    def _run(self, ids: list[int], past: list[np.ndarray], pos0: int) -> tuple[np.ndarray, list[np.ndarray]]:
        n = len(ids)
        feed = {"input_ids": np.asarray(ids, dtype=np.int64)[None], "attention_mask": np.ones((1, pos0 + n), dtype=np.int64)}
        if self.has_position_ids:
            feed["position_ids"] = np.arange(pos0, pos0 + n, dtype=np.int64)[None]
        feed.update(zip(self.past_names, past))
        out = self.sess.run(self.out_names, feed)
        return np.asarray(out[0][0, -1], dtype=np.float32), out[1:]

    def _system(self, prefix: str) -> tuple[int, list[np.ndarray], np.ndarray]:
        """(length, kv cache, ids) of the fixed prompt prefix, computed once per text."""
        if prefix not in self._prefix:
            ids = self.tok.encode(prefix, add_special_tokens=False).ids
            _, past = self._run(ids, self._empty_past(), 0)
            self._prefix = {prefix: (len(ids), past, np.asarray(ids))}
        return self._prefix[prefix]

    def prompt_ids(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """(prefix ids, rest ids); their concatenation is the tokenization of the whole prompt."""
        prefix, rest = chatml(messages)
        return self.tok.encode(prefix, add_special_tokens=False).ids, self.tok.encode(rest, add_special_tokens=False).ids

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def generate(self, messages: list[dict], allowed, on_delta=None, deadline_s: float | None = None) -> GenResult:
        """greedy reply to the messages. on_delta gets each finished word with
        its leading space, then once the rest; an exception it raises stops the
        generation and propagates."""
        t0 = time.perf_counter()
        emit = on_delta or (lambda _w: None)
        prefix, rest = chatml(messages)
        n_sys, past, _ = self._system(prefix)
        rest_ids = self.tok.encode(rest, add_special_tokens=False).ids
        logits, past = self._run(rest_ids, past, n_sys)
        n_prompt = n_sys + len(rest_ids)
        bad_classes = forbidden_classes(allowed)
        lex = np.concatenate([self.masks.by_class[c] for c in bad_classes]) if bad_classes else np.zeros(0, np.int64)
        lex_set = set(lex.tolist())
        bad_words = {w for c in bad_classes for w in LEXICON[c]}
        guard = GUARD_PHRASES + transcript_ngrams(messages, self.echo_ngram)
        stop = set(self.masks.stop.tolist())

        ids: list[int] = []
        hist: list[np.ndarray] = []          # hist[k]: the logits that chose ids[k]
        bans: dict[int, set[int]] = {}
        ends: list[int] = []                 # ends[k]: len of the decoded text after ids[:k+1]
        sent = 0                             # chars of the text already sent; 0 until the first word goes out
        deltas: list[str] = []
        first_ms = None
        masked = reg_masked = steps = passes = rework = 0
        used = {"word": 0, "phrase": 0}      # rollbacks per budget: lexicon words; guard phrases and echoes
        stop_reason = "max_tokens"
        text = ""

        def tok_at(char: int) -> int:
            return next(k for k, e in enumerate(ends) if e > char)

        def violation(final: bool) -> tuple[int, str] | None:
            """(token index to roll back to, budget) for the first unsent finished word that
            is a lexicon word of a class not allowed ("word"), or the first word of a
            finished guard phrase or transcript echo ("phrase"); None when there is none."""
            ms = list(_WORDS.finditer(text, sent))
            done = len(ms) if final or not ms or ms[-1].end() < len(text) else len(ms) - 1
            low = [m.group(0).lower() for m in ms]
            for i in range(done):
                if low[i] in bad_words:
                    return tok_at(ms[i].start()), "word"
                for ph in guard:
                    if i + len(ph) <= done and tuple(low[i:i + len(ph)]) == ph:
                        return tok_at(ms[i].start()), "phrase"
            return None

        def hold_from() -> int:
            """first char of a word run that could still grow into a guard phrase; words
            from there on wait, so a rollback never has to take back a sent word."""
            ms = list(_WORDS.finditer(text, sent))
            low = [m.group(0).lower() for m in ms]
            open_last = bool(ms) and ms[-1].end() >= len(text)
            for i in range(len(ms)):
                rest = low[i:]
                for ph in guard:
                    n = len(rest)
                    if n > len(ph) or tuple(rest[:-1]) != ph[:n - 1]:
                        continue
                    if (open_last and ph[n - 1].startswith(rest[-1])) or (rest[-1] == ph[n - 1] and n < len(ph)):
                        return ms[i].start()
            return len(text)

        def rollback(a: int, budget: str, pending: int = 0) -> bool:
            """undo ids[a:] and ban ids[a] at position a. pending: undone tokens whose forward
            pass has not run yet (the token just appended), not counted as rework."""
            nonlocal ids, hist, ends, logits, past, text, rework
            start = ends[a - 1] if a else 0
            if used[budget] >= self.max_backtracks or start < sent:
                return False
            used[budget] += 1
            rework += len(ids) - a - pending
            bans.setdefault(a, set()).add(ids[a])
            for k in [k for k in bans if k > a]:
                del bans[k]
            ids, ends, logits = ids[:a], ends[:a], hist[a]
            hist = hist[:a]
            past = [np.ascontiguousarray(p[:, :, : n_prompt + a]) for p in past]
            text = self._decode(ids)
            return True

        def cut(a: int) -> None:
            """no rollback left: drop the word at token a and what follows, keep what was sent."""
            nonlocal text
            text = text[: max(sent, ends[a - 1] if a else 0)]

        while True:
            k = len(ids)
            if k >= self.max_new_tokens:
                break
            if deadline_s is not None and time.perf_counter() - t0 > deadline_s:
                stop_reason = "deadline"
                break
            x = logits.copy()
            raw = int(np.argmax(x))
            x[self.masks.register] = -np.inf
            after_reg = int(np.argmax(x))
            if lex.size:
                x[lex] = -np.inf
            if k in bans:
                x[list(bans[k])] = -np.inf
            tok = int(np.argmax(x))
            steps += 1
            reg_masked += int(raw != after_reg)
            masked += int(after_reg != tok and after_reg in lex_set)
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
            hist.append(logits)
            if tok in stop:
                v = violation(final=True)
                if v is not None and rollback(*v):
                    continue
                hist.pop()
                stop_reason = "eos"
                if v is not None:
                    cut(v[0])
                    stop_reason = "guard"
                break
            ids.append(tok)
            text = self._decode(ids)
            ends.append(len(text))
            v = violation(final=False)
            if v is not None:
                if rollback(v[0], v[1], pending=1):
                    continue
                # no rollback left: end the reply at the last word that passed, as at the token budget
                cut(v[0])
                stop_reason = "guard"
                break
            # send every finished word: a word ends at a whitespace that follows a non-space. leading
            # whitespace is skipped for the first word, and sent stays 0 until that word goes out, so
            # the first word can still be rolled back
            start = sent if deltas else len(text) - len(text.lstrip())
            hold = hold_from()
            for i in range(start + 1, hold):
                if text[i].isspace() and not text[i - 1].isspace():
                    deltas.append(text[start:i])
                    start = sent = i
                    emit(deltas[-1])
            logits, past = self._run([tok], past, n_prompt + k)
            passes += 1

        if stop_reason in ("max_tokens", "deadline"):
            v = violation(final=True)
            if v is not None:
                cut(v[0])                    # no token budget left to decode again
        start = sent if deltas else len(text) - len(text.lstrip())
        tail = text[start:].rstrip()
        if tail:
            deltas.append(tail)
            emit(tail)
        reply = "".join(deltas).strip()
        return GenResult(reply, len(ids), first_ms if first_ms is not None else 0.0, (time.perf_counter() - t0) * 1000,
                         stop_reason, masked, prompt_tokens=len(rest_ids), steps=steps, backtracks=sum(used.values()),
                         register_masked_steps=reg_masked, deltas=deltas, word_backtracks=used["word"], decode_passes=passes,
                         rework_passes=rework)
