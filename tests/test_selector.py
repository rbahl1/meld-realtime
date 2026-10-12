"""reply bank shape and selector rules (WP5, design 1.9)."""

import time

import numpy as np
import yaml

from meld_rt.labels import EMOTIONS, SV_EMO_TOKENS, SV_TO_MELD
from meld_rt.state import Discrepancy, State, audio_emotion_dict, emotion7_dict
from meld_rt.discrepancy import DIRECTIONS, audio7, discrepancy, keyphrase
from meld_rt.response import BANK_PATH
from meld_rt.response.selector import (CELL_DIRECTIONS, MAX_SLOT_TEMPLATE_WORDS, MAX_WORDS, STYLES,
                                       ResponseSelector, cell_key, load_bank, word_count)

JSD_FLAG = 0.2   # stand-in until dev picks heads/manifest.json thresholds.jsd_flag
DRAFT_MARK = "draft: every line to be edited by me before submission"


def sv9(**meld_probs):
    """9-way vector in sensevoice token order from MELD names."""
    name = dict(SV_TO_MELD, OTHER="other", EMO_UNKNOWN="unknown")
    v = np.array([meld_probs.get(name[t], 0.0) for t in SV_EMO_TOKENS], dtype=np.float64)
    return v / v.sum()


def meld7(**probs):
    v = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float64)
    return v / v.sum()


def poe(p_text7, p_audio9):
    """the fixed fallback fusion of design 1.6: p ~ p_text * p_audio7^0.5."""
    f = np.asarray(p_text7) * audio7(p_audio9) ** 0.5
    return f / f.sum()


def make_state(text7, audio9, transcript, utt_id=1, events=None, conf=None, direction=None, calibrated=True):
    """calibrated=True stands for a stacked fusion; False is the poe_uncalibrated fallback."""
    st = State.empty("s1", utt_id)
    st.asr.text = transcript
    st.text.emotion = emotion7_dict(text7)
    st.text.keyphrase = keyphrase(transcript)
    st.audio.emotion = audio_emotion_dict(audio9)
    if events:
        st.audio.events.update(events)
    f = poe(text7, audio9)
    i = int(np.argmax(f))
    st.fused.emotion = emotion7_dict(f)
    st.fused.label = EMOTIONS[i]
    st.fused.confidence = float(f[i]) if conf is None else conf
    st.fused.mode = "stacked" if calibrated else "poe_uncalibrated"
    st.fused.calibrated = calibrated
    j, flag, d = discrepancy(text7, audio9, JSD_FLAG)
    st.discrepancy = Discrepancy(j, flag, d if direction is None else direction)
    return st


def peaked(label, p=0.7):
    rest = (1 - p) / 6
    return meld7(**{e: (p if e == label else rest) for e in EMOTIONS})


def peaked9(label, p=0.7):
    rest = (1 - p) / 6
    return sv9(**{e: (p if e == label else rest) for e in EMOTIONS})


def plain_state(label, direction, conf, transcript="the GR-6 system failed again", utt_id=1, events=None, calibrated=True):
    st = make_state(peaked(label), peaked9(label), transcript, utt_id=utt_id, events=events, calibrated=calibrated)
    st.fused.label, st.fused.confidence = label, conf
    st.discrepancy.direction = direction
    st.discrepancy.flag = direction != "none"
    return st


# the draft marker on the bank's first line and the two assertions on it below go together: remove both when the bank is rewritten
def test_bank_shape():
    raw_text = BANK_PATH.read_text(encoding="utf-8")
    assert raw_text.splitlines()[0].strip("# ").startswith(DRAFT_MARK)
    raw = yaml.safe_load(raw_text)
    assert f"draft: {raw['draft']}" == DRAFT_MARK
    cells = raw["cells"]
    assert len(cells) == 28 and set(cells) == {f"{l}|{d}" for l in EMOTIONS for d in CELL_DIRECTIONS}
    ids = []
    all_lines = []
    for key, items in cells.items():
        assert len(items) >= 3, key
        styles = {it["style"] for it in items}
        assert styles <= set(STYLES) and "ask" in styles and "acknowledge" in styles, key
        if not key.endswith("|none"):
            assert "name_the_gap" in styles, key
        assert sum("{keyphrase}" not in it["text"] for it in items) >= 2, key
        ids += [it["id"] for it in items]
        all_lines += [it["text"] for it in items]
    assert len(raw["hold_back"]) == 6 and all("{" not in it["text"] for it in raw["hold_back"])
    assert len(raw["event_prefix"]["laughter"]) == 2 and len(raw["event_prefix"]["cry"]) == 2
    ids += [it["id"] for it in raw["hold_back"]] + [it["id"] for it in raw.get("asr_failed", [])]
    assert len(ids) == len(set(ids))
    all_lines += [it["text"] for it in raw["hold_back"]] + [it["text"] for it in raw.get("asr_failed", [])]
    all_lines += raw["event_prefix"]["laughter"] + raw["event_prefix"]["cry"]
    for line in all_lines:
        assert 1 <= word_count(line) <= MAX_WORDS, line
        assert "!" not in line and line.isascii() and line.isprintable(), line
        if "{keyphrase}" in line:
            assert word_count(line) <= MAX_SLOT_TEMPLATE_WORDS, line
    bank = load_bank(BANK_PATH)
    assert len(bank["cells"]) == 28 and len(bank["hold_back"]) == 6


def test_low_confidence_always_holds_back():
    sel = ResponseSelector(BANK_PATH)
    for label in EMOTIONS:
        for direction in DIRECTIONS:
            for conf in (0.0, 0.1, 0.3, 0.449):
                r = sel.select(plain_state(label, direction, conf), [])
                assert r.style == "hold_back" and r.template_id.startswith("hold_back_"), (label, direction, conf)
                assert "{" not in r.text and r.grounding[0].startswith("fused.confidence=")
    r = sel.select(plain_state("anger", "none", 0.45), [])
    assert r.style != "hold_back"


CASES = [
    ("well that went great", "joy", "anger"),
    ("the GR-6 system failed again", "anger", "joy"),
    ("I guess we will see tomorrow", "neutral", "anger"),
    ("my brother called last night", "neutral", "joy"),
    ("they moved the deadline to friday", "neutral", "surprise"),
    ("I got the apartment", "joy", "sadness"),
    ("the dog is sick again", "sadness", "joy"),
    ("she left without a word", "sadness", "surprise"),
    ("you promised me the report", "anger", "surprise"),
    ("the brakes made that noise again", "fear", "joy"),
    ("there was hair in the soup", "disgust", "joy"),
    ("he showed up two hours late", "anger", "joy"),
    ("the test results came back", "neutral", "anger"),
    ("I did not see that coming", "surprise", "anger"),
    ("the whole office got a bonus", "surprise", "joy"),
    ("the landlord raised the rent", "disgust", "surprise"),
    ("we are out of coffee", "sadness", "anger"),
    ("the kids are finally asleep", "joy", "surprise"),
    ("someone keyed my car", "anger", "sadness"),
    ("the meeting is at nine", "neutral", "fear"),
]


def test_changing_only_the_audio_posterior_changes_the_line():
    sel = ResponseSelector(BANK_PATH)
    changed = 0
    for i, (transcript, label, contrast) in enumerate(CASES):
        a = make_state(peaked(label), peaked9(label), transcript, utt_id=i)
        b = make_state(peaked(label), peaked9(contrast), transcript, utt_id=i)
        assert a.text == b.text and a.asr == b.asr
        ra, rb = sel.select(a, []), sel.select(b, [])
        changed += ra.text != rb.text
    assert len(CASES) == 20 and changed >= 15, changed


def test_keyphrase_is_a_substring_and_fills_the_slot():
    sel = ResponseSelector(BANK_PATH)
    transcripts = ["well that went great", "the GR-6 system failed again", "my brother called last night",
                   "there was hair in the soup", "I think the landlord raised the rent, honestly"]
    for i, t in enumerate(transcripts):
        for label in EMOTIONS:
            st = plain_state(label, "none", 0.9, transcript=t, utt_id=i)
            r = sel.select(st, [])
            kp = st.text.keyphrase
            assert kp and kp in t
            assert "{" not in r.text
            if any(g.startswith("text.keyphrase=") for g in r.grounding):
                assert kp in r.text and f"text.keyphrase='{kp}'" in r.grounding
    # the slot is filled from asr.text when the state carries no keyphrase
    st = plain_state("joy", "none", 0.9, transcript="well that went great")
    st.text.keyphrase = None
    r = sel.select(st, [])
    assert "went great" in r.text and "text.keyphrase='went great'" in r.grounding
    # no content words: a slot-free line is chosen
    st = plain_state("joy", "none", 0.9, transcript="you know what I mean")
    assert st.text.keyphrase is None
    r = sel.select(st, [])
    assert "{" not in r.text and not any(g.startswith("text.keyphrase") for g in r.grounding)


def test_no_repeat_within_five_turns():
    sel = ResponseSelector(BANK_PATH, no_repeat_turns=5)
    for label, direction, conf in (("anger", "tone_more_negative", 0.9), ("joy", "none", 0.55),
                                   ("sadness", "none", 0.2), ("fear", "tone_surprised", 0.75)):
        recent = []
        for turn in range(12):
            r = sel.select(plain_state(label, direction, conf, utt_id=turn), recent)
            recent.append(r.template_id)
        for k in range(len(recent) - 4):
            assert len(set(recent[k:k + 5])) == 5, (label, direction, recent)
    # a constant utt_id still rotates through the recency penalty
    recent = []
    for _ in range(8):
        recent.append(sel.select(plain_state("disgust", "tone_more_positive", 0.9, utt_id=3), recent).template_id)
    assert all(len(set(recent[k:k + 5])) == 5 for k in range(len(recent) - 4))


def test_deterministic():
    a, b = ResponseSelector(BANK_PATH, seed=0), ResponseSelector(BANK_PATH, seed=0)
    for i, (t, label, contrast) in enumerate(CASES):
        st = make_state(peaked(label), peaked9(contrast), t, utt_id=i)
        assert a.select(st, ["anger_neg_01"]) == b.select(st, ["anger_neg_01"])


def test_bucket_style_preference():
    sel = ResponseSelector(BANK_PATH)
    assert sel.bucket(0.449) == "low" and sel.bucket(0.45) == "mid" and sel.bucket(0.699) == "mid" and sel.bucket(0.7) == "high"
    for label in EMOTIONS:
        assert sel.select(plain_state(label, "none", 0.55), []).style == "ask"
        assert sel.select(plain_state(label, "mixed", 0.55), []).style == "ask"
        assert sel.select(plain_state(label, "none", 0.8), []).style == "acknowledge"
        for direction in ("tone_more_negative", "tone_more_positive", "tone_surprised"):
            assert sel.select(plain_state(label, direction, 0.55), []).style == "ask"
            assert sel.select(plain_state(label, direction, 0.8), []).style == "name_the_gap"
    # mixed shares the none cell
    r = sel.select(plain_state("joy", "mixed", 0.8), [])
    assert "_none_" in r.template_id and cell_key("joy", "mixed") == "joy|none"


def test_uncalibrated_state_never_reaches_the_high_bucket():
    """fused.calibrated false (poe_uncalibrated, or poe without a fusion head): the
    thresholds were chosen for a calibrated posterior, so the bucket is capped at
    mid and the line asks instead of asserting."""
    sel = ResponseSelector(BANK_PATH)
    assert sel.bucket(0.9) == "high" and sel.bucket(0.9, calibrated=False) == "mid"
    assert sel.bucket(0.5, calibrated=False) == "mid" and sel.bucket(0.3, calibrated=False) == "low"
    for label in EMOTIONS:
        for direction in CELL_DIRECTIONS:
            r = sel.select(plain_state(label, direction, 0.95, calibrated=False), [])
            assert r.style == "ask" and "rule=uncalibrated_cap" in r.grounding, (label, direction, r)
            assert sel.select(plain_state(label, direction, 0.95, calibrated=True), []).style != "ask"
            r = sel.select(plain_state(label, direction, 0.55, calibrated=False), [])
            assert r.style == "ask" and "rule=uncalibrated_cap" not in r.grounding, (label, direction, r)
    r = sel.select(plain_state("anger", "none", 0.2, calibrated=False), [])
    assert r.style == "hold_back" and "rule=uncalibrated_cap" not in r.grounding
    r = sel.select(plain_state("joy", "tone_more_negative", 0.8, calibrated=False), [])
    assert r.grounding[:3] == ["fused.label=joy@0.80", "discrepancy.direction=tone_more_negative", "rule=uncalibrated_cap"]
    # the cap does not stop the no-repeat rule: a used-up cell still falls back to hold_back
    recent = []
    for turn in range(12):
        recent.append(sel.select(plain_state("fear", "tone_surprised", 0.9, utt_id=turn, calibrated=False), recent).template_id)
    assert all(len(set(recent[k:k + 5])) == 5 for k in range(len(recent) - 4))


def test_grounding_format():
    sel = ResponseSelector(BANK_PATH)
    st = plain_state("anger", "tone_more_negative", 0.62, transcript="the GR-6 system failed again")
    r = sel.select(st, [])
    assert r.grounding[0] == "fused.label=anger@0.62" and r.grounding[1] == "discrepancy.direction=tone_more_negative"
    assert r.template_id.startswith("anger_neg_") and r.style in STYLES


def test_event_prefix():
    bank = load_bank(BANK_PATH)
    sel = ResponseSelector(BANK_PATH)
    st = plain_state("joy", "none", 0.9, events={"laughter": 0.8})
    r = sel.select(st, [])
    assert any(r.text.startswith(p + " ") for p in bank["event_prefix"]["laughter"])
    assert "audio.events.laughter=0.80" in r.grounding
    st = plain_state("sadness", "none", 0.9, events={"cry": 0.6, "laughter": 0.3})
    r = sel.select(st, [])
    assert any(r.text.startswith(p + " ") for p in bank["event_prefix"]["cry"]) and "audio.events.cry=0.60" in r.grounding
    st = plain_state("sadness", "none", 0.9, events={"cry": 0.5, "laughter": 0.49})
    r = sel.select(st, [])
    assert not any(g.startswith("audio.events") for g in r.grounding)
    st = plain_state("joy", "none", 0.9, events={"cry": 0.7, "laughter": 0.9})
    assert "audio.events.laughter=0.90" in sel.select(st, []).grounding
    # the prefix also applies to hold_back lines, and both prefix clauses get used
    seen = set()
    for i in range(6):
        r = sel.select(plain_state("joy", "none", 0.2, utt_id=i, events={"laughter": 0.9}), [])
        seen.add(r.text.split(" ")[0:3] == bank["event_prefix"]["laughter"][0].split(" ")[0:3])
    assert seen == {True, False}


def test_asr_failure_line():
    sel = ResponseSelector(BANK_PATH)
    st = State.empty("s1", 1)
    st.degraded = ["asr_failed"]
    r = sel.select(st, [])
    assert r.template_id.startswith("asr_failed_") and r.style == "hold_back" and r.grounding == ["degraded=asr_failed"]
    st.degraded = ["low_asr_conf"]
    assert sel.select(st, []).grounding == ["degraded=low_asr_conf"]


def test_select_is_fast():
    sel = ResponseSelector(BANK_PATH)
    st = plain_state("anger", "tone_more_negative", 0.9)
    t0 = time.perf_counter()
    for i in range(1000):
        sel.select(st, ["anger_neg_01", "anger_neg_02"])
    assert (time.perf_counter() - t0) / 1000 < 0.005
