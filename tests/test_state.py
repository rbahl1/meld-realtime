"""state schema round trip, jsd, direction rule, keyphrase (WP5)."""

import json

import numpy as np

from meld_rt.labels import EMOTIONS, EVENT_KEYS, SV_EMO_TOKENS, SV_EVENT_TOKENS
from meld_rt.state import (AUDIO_EMOTION_KEYS, State, audio_emotion_dict, audio_emotion_vec9,
                           emotion7_dict, emotion7_vec, events_dict)
from meld_rt.discrepancy import audio7, direction, discrepancy, jsd, keyphrase, valence

# the literal of design 1.8 (its dict key orders differ from the label orders on purpose)
DESIGN_18 = """
{"kind": "final", "stream_id": "s1", "utt_id": 7, "seq_range": [1200, 1356],
 "t_media_start": 10.02, "t_media_end": 12.40,
 "t_speech_end_capture_ns": 171000000000, "t_service_recv_ns": 171000000400, "t_emitted_ns": 171000000590,
 "window": "full", "continued_from": null,
 "asr": {"text": "well that went great", "conf": 0.91, "lang": "en", "tokens_per_s": 2.9},
 "audio": {"emotion": {"joy": 0.05, "sadness": 0.06, "anger": 0.52, "neutral": 0.20, "fear": 0.02, "disgust": 0.03, "surprise": 0.02, "other": 0.01, "unknown": 0.09},
           "events": {"speech": 0.97, "laughter": 0.01, "bgm": 0.01, "applause": 0.0, "cry": 0.0, "sneeze": 0.0, "breath": 0.0, "cough": 0.0, "sing": 0.0, "speech_noise": 0.01, "event_unk": 0.0},
           "prosody": {"f0_median_st": 44.1, "f0_range_st": 9.3, "intensity_std_db": 6.2, "rate_tps": 2.9},
           "encoder_states_used": true},
 "text": {"partial": null, "emotion": {"neutral": 0.3, "surprise": 0.05, "fear": 0.02, "sadness": 0.05, "joy": 0.5, "disgust": 0.03, "anger": 0.05}, "context_k": 2, "keyphrase": "went great"},
 "fused": {"emotion": {"neutral": 0.22, "surprise": 0.04, "fear": 0.02, "sadness": 0.06, "joy": 0.08, "disgust": 0.03, "anger": 0.55},
           "label": "anger", "confidence": 0.55, "sentiment": "negative", "mode": "stacked", "calibrated": true},
 "discrepancy": {"jsd": 0.41, "flag": true, "direction": "tone_more_negative"},
 "dialogue": {"turn_index": 7, "prev_label": "neutral", "emotion_changed": true, "trajectory": ["neutral", "neutral", "joy", "neutral", "anger"]},
 "timing_ms": {"vad_tail": 400, "frontend": 6, "sensevoice": 95, "prosody": 14, "text_encoder": 17, "heads": 1, "service_total": 190, "provisional_in_flight": 0},
 "degraded": []}
"""

TOP_KEYS = ["kind", "stream_id", "utt_id", "seq_range", "t_media_start", "t_media_end",
            "t_speech_end_capture_ns", "t_service_recv_ns", "t_emitted_ns", "window", "continued_from",
            "asr", "audio", "text", "fused", "discrepancy", "dialogue", "timing_ms", "degraded"]


def sv9(**meld_probs):
    """9-way vector in sensevoice token order from MELD names (+ other, unknown)."""
    name = {"HAPPY": "joy", "SAD": "sadness", "ANGRY": "anger", "NEUTRAL": "neutral", "FEARFUL": "fear",
            "DISGUSTED": "disgust", "SURPRISED": "surprise", "OTHER": "other", "EMO_UNKNOWN": "unknown"}
    v = np.array([meld_probs.get(name[t], 0.0) for t in SV_EMO_TOKENS], dtype=np.float64)
    return v / v.sum()


def meld7(**probs):
    v = np.array([probs.get(e, 0.0) for e in EMOTIONS], dtype=np.float64)
    return v / v.sum()


def test_event_key_orders_are_aligned():
    assert [t.lower() for t in SV_EVENT_TOKENS] == EVENT_KEYS


def test_design_example_round_trip_and_key_order():
    st = State.from_json(DESIGN_18)
    assert st.fused.label == "anger" and st.discrepancy.jsd == 0.41 and st.audio.emotion["anger"] == 0.52
    assert st.dialogue.trajectory[-1] == "anger" and st.continued_from is None and st.text.partial is None
    j = st.to_json()
    assert State.from_json(j) == st
    assert State.from_json(State.from_json(j).to_json()) == st
    d = json.loads(j)
    assert list(d) == TOP_KEYS and "stability" not in d
    assert list(d["text"]["emotion"]) == EMOTIONS and list(d["fused"]["emotion"]) == EMOTIONS
    assert list(d["audio"]["emotion"]) == AUDIO_EMOTION_KEYS and list(d["audio"]["events"]) == EVENT_KEYS
    assert set(d["asr"]) == {"text", "conf", "lang", "tokens_per_s"}
    assert set(d["audio"]) == {"emotion", "events", "prosody", "encoder_states_used"}
    assert set(d["text"]) == {"partial", "emotion", "context_k", "keyphrase"}
    assert set(d["fused"]) == {"emotion", "label", "confidence", "sentiment", "mode", "calibrated"}
    assert set(d["discrepancy"]) == {"jsd", "flag", "direction"}
    assert set(d["dialogue"]) == {"turn_index", "prev_label", "emotion_changed", "trajectory"}
    assert json.loads(DESIGN_18)["timing_ms"] == d["timing_ms"]


def test_empty_states_round_trip():
    for kind in ("final", "provisional"):
        st = State.empty("s1", 7, kind=kind)
        assert st.stream_id == "s1" and st.utt_id == 7 and st.kind == kind
        for dct in (st.text.emotion, st.fused.emotion, st.audio.emotion):
            assert abs(sum(dct.values()) - 1) < 1e-9
        assert set(st.audio.events) == set(EVENT_KEYS) and all(v == 0 for v in st.audio.events.values())
        j = st.to_json()
        assert State.from_json(j) == st
        assert ("stability" in json.loads(j)) == (kind == "provisional")
    st = State.empty("s1", 8, kind="provisional")
    st.stability = 0.8
    st.text.partial = "well that"
    st.window = "tail"
    assert State.from_json(st.to_json()) == st and json.loads(st.to_json())["stability"] == 0.8


def test_vector_dict_helpers_invert():
    rng = np.random.default_rng(0)
    p9 = rng.random(9)
    p9 /= p9.sum()
    d = audio_emotion_dict(p9)
    assert list(d) == AUDIO_EMOTION_KEYS
    assert np.allclose(audio_emotion_vec9(d), p9)
    assert d["joy"] == p9[SV_EMO_TOKENS.index("HAPPY")] and d["unknown"] == p9[-1]
    p7 = rng.random(7)
    assert np.allclose(emotion7_vec(emotion7_dict(p7)), p7)
    assert list(events_dict(rng.random(11))) == EVENT_KEYS


def test_from_json_tolerates_unknown_and_missing_keys():
    d = json.loads(DESIGN_18)
    del d["dialogue"]
    d["future_field"] = 1
    d["asr"]["future_sub"] = 2
    st = State.from_dict(d)
    assert st.dialogue.turn_index == 0 and st.dialogue.trajectory == []
    assert st.asr.text == "well that went great"
    assert State.from_json(st.to_json()) == st


def test_numpy_values_serialize():
    st = State.empty("s1", 1)
    st.text.emotion = emotion7_dict(np.full(7, 1 / 7, dtype=np.float32))
    st.timing_ms["sensevoice"] = np.float32(95.5)
    st.seq_range = (3, 9)
    d = json.loads(st.to_json())
    assert abs(d["timing_ms"]["sensevoice"] - 95.5) < 1e-6 and d["seq_range"] == [3, 9]


def test_jsd_endpoints_and_symmetry():
    for p in (meld7(neutral=1), meld7(joy=0.5, anger=0.5), np.full(7, 1 / 7)):
        assert jsd(p, p) == 0.0
    assert jsd([1, 0, 0], [0, 1, 0]) == 1.0
    assert jsd(meld7(joy=1), meld7(anger=1)) == 1.0
    a, b = meld7(joy=0.6, neutral=0.4), meld7(anger=0.7, fear=0.3)
    assert jsd(a, b) == jsd(b, a) and 0 < jsd(a, b) <= 1
    # closed form: jsd([.5,.5],[1,0]) = .25*log2(2/3) + .25 + .5*log2(4/3)
    assert abs(jsd([0.5, 0.5], [1, 0]) - (0.25 * np.log2(2 / 3) + 0.25 + 0.5 * np.log2(4 / 3))) < 1e-12
    assert jsd([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_audio_renormalization():
    p9 = sv9(anger=0.5, neutral=0.2, other=0.1, unknown=0.2)
    p7 = audio7(p9)
    assert abs(p7.sum() - 1) < 1e-12 and abs(p7[EMOTIONS.index("anger")] - 0.5 / 0.7) < 1e-12
    assert discrepancy(meld7(neutral=1), p9, 0.1)[0] == discrepancy(meld7(neutral=1), p7, 0.1)[0]


def test_direction_rule_cases():
    text_joy = meld7(joy=0.8, neutral=0.2)
    # not flagged: identical posteriors
    assert discrepancy(text_joy, sv9(joy=0.8, neutral=0.2), 0.1) == (0.0, False, "none")
    # flagged only strictly above the threshold
    j = jsd(text_joy, meld7(joy=0.6, neutral=0.4))
    assert discrepancy(text_joy, meld7(joy=0.6, neutral=0.4), j)[1] is False
    assert discrepancy(text_joy, meld7(joy=0.6, neutral=0.4), j - 1e-9)[1] is True
    # valence sign
    assert discrepancy(text_joy, sv9(anger=0.7, neutral=0.3), 0.2)[1:] == (True, "tone_more_negative")
    assert discrepancy(meld7(sadness=0.8, neutral=0.2), sv9(joy=0.7, neutral=0.3), 0.2)[1:] == (True, "tone_more_positive")
    # surprise rule: audio surprise > 0.4 and text surprise < 0.2, checked before valence
    assert discrepancy(meld7(neutral=1), sv9(surprise=0.6, neutral=0.4), 0.2)[2] == "tone_surprised"
    assert discrepancy(text_joy, sv9(surprise=0.5, anger=0.5), 0.2)[2] == "tone_surprised"
    # surprise rule off when the text already says surprise; equal valence -> mixed
    assert discrepancy(meld7(surprise=0.3, neutral=0.7), sv9(surprise=0.9, neutral=0.1), 0.2)[2] == "mixed"
    assert discrepancy(meld7(neutral=1), sv9(joy=0.5, anger=0.5), 0.2)[2] == "mixed"
    # same valence, different class -> mixed
    assert discrepancy(meld7(sadness=1), sv9(anger=1), 0.2)[2] == "mixed"
    assert direction(text_joy, text_joy, False) == "none"
    assert valence(meld7(joy=1)) == 1 and valence(meld7(anger=1)) == -1 and valence(meld7(surprise=1)) == 0
    # the 1.8 example itself
    ex = json.loads(DESIGN_18)
    pt = [ex["text"]["emotion"][e] for e in EMOTIONS]
    pa = [ex["audio"]["emotion"][e] for e in EMOTIONS]
    assert discrepancy(pt, pa, 0.2)[2] == "tone_more_negative"


def test_keyphrase_rules():
    cases = {
        "well that went great": "went great",
        "I think the GR-6 system is broken again, honestly": "the GR-6 system",
        "the giant purple spotted river monster": "purple spotted river monster",
        "Sure, fine. Whatever you say, boss": "fine",
        "I’m fine with the new schedule": "the new schedule",
        "room 404 is locked": "room 404",
        "my brother called": "my brother called",
        "my brother called last night": "brother called last night",
        "I told her everything": "everything",
        "the giant purple spotted river": "giant purple spotted river",
        "You know what I mean": None,
        "": None,
        "um so yeah": None,
    }
    for text, want in cases.items():
        got = keyphrase(text)
        assert got == want, (text, got)
        if got is not None:
            assert got in text and len(got.split()) <= 4
    # a stopword inside a run splits it; punctuation splits it too
    assert keyphrase("red car and blue truck") in ("red car", "blue truck")
    assert keyphrase("red car, blue truck") == "blue truck" or keyphrase("red car, blue truck") == "red car"
