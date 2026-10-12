"""per-utterance interaction state (design 1.8): one dataclass per nested
object, json in and out. emotion dicts are keyed in labels.EMOTIONS order,
audio events in labels.EVENT_KEYS order, whatever order the input had."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

from meld_rt.labels import EMOTIONS, EVENT_KEYS, SV_EMO_TOKENS, SV_TO_MELD

AUDIO_EMOTION_KEYS = EMOTIONS + ["other", "unknown"]
PROSODY_KEYS = ["f0_median_st", "f0_range_st", "intensity_std_db", "rate_tps"]
TIMING_KEYS = ["vad_tail", "frontend", "sensevoice", "prosody", "text_encoder", "heads",
               "service_total", "provisional_in_flight"]
KINDS = ("provisional", "final")
WINDOWS = ("full", "tail")
FUSION_MODES = ("stacked", "poe", "poe_uncalibrated", "text_only")


def _uniform(keys: list[str]) -> dict[str, float]:
    return {k: 1.0 / len(keys) for k in keys}


def _zeros(keys: list[str]) -> dict[str, float]:
    return {k: 0.0 for k in keys}


def _ordered(d: dict, keys: list[str]) -> dict:
    """known keys first in canonical order, unknown keys after in their own order."""
    out = {k: d[k] for k in keys if k in d}
    out.update({k: v for k, v in d.items() if k not in out})
    return out


def emotion7_dict(p7) -> dict[str, float]:
    """7-way posterior in MELD index order -> dict in labels.EMOTIONS order."""
    return {e: float(p7[i]) for i, e in enumerate(EMOTIONS)}


def emotion7_vec(d: dict[str, float]) -> list[float]:
    """inverse of emotion7_dict."""
    return [float(d[e]) for e in EMOTIONS]


def audio_emotion_dict(p9) -> dict[str, float]:
    """9-way sensevoice posterior in SV token order -> the 7 MELD keys in
    EMOTIONS order, then other and unknown."""
    by_tok = {t: float(p9[i]) for i, t in enumerate(SV_EMO_TOKENS)}
    out = {e: by_tok[next(t for t, m in SV_TO_MELD.items() if m == e)] for e in EMOTIONS}
    out["other"] = by_tok["OTHER"]
    out["unknown"] = by_tok["EMO_UNKNOWN"]
    return out


def audio_emotion_vec9(d: dict[str, float]) -> list[float]:
    """inverse of audio_emotion_dict: back to the 9-way SV token order."""
    name = dict(SV_TO_MELD, OTHER="other", EMO_UNKNOWN="unknown")
    return [float(d[name[t]]) for t in SV_EMO_TOKENS]


def events_dict(p11) -> dict[str, float]:
    """11-way sensevoice event posterior (SV token order) -> dict in EVENT_KEYS order.
    the two orders are index-aligned (tests/test_state.py asserts it)."""
    return {k: float(p11[i]) for i, k in enumerate(EVENT_KEYS)}


def prosody_dict(v4) -> dict[str, float]:
    return {k: float(v4[i]) for i, k in enumerate(PROSODY_KEYS)}


@dataclass
class Asr:
    text: str = ""
    conf: float = 0.0
    lang: str = "en"
    tokens_per_s: float = 0.0


@dataclass
class Audio:
    emotion: dict[str, float] = field(default_factory=lambda: _uniform(AUDIO_EMOTION_KEYS))
    events: dict[str, float] = field(default_factory=lambda: _zeros(EVENT_KEYS))
    prosody: dict[str, float] = field(default_factory=lambda: _zeros(PROSODY_KEYS))
    encoder_states_used: bool = False


@dataclass
class Text:
    partial: str | None = None
    emotion: dict[str, float] = field(default_factory=lambda: _uniform(EMOTIONS))
    context_k: int = 0
    keyphrase: str | None = None


@dataclass
class Fused:
    emotion: dict[str, float] = field(default_factory=lambda: _uniform(EMOTIONS))
    label: str = "neutral"
    confidence: float = 0.0
    sentiment: str = "neutral"
    mode: str = "text_only"
    calibrated: bool = False


@dataclass
class Discrepancy:
    jsd: float = 0.0
    flag: bool = False
    direction: str = "none"


@dataclass
class Dialogue:
    turn_index: int = 0
    prev_label: str | None = None
    emotion_changed: bool = False
    trajectory: list[str] = field(default_factory=list)


_NESTED = {"asr": Asr, "audio": Audio, "text": Text, "fused": Fused,
           "discrepancy": Discrepancy, "dialogue": Dialogue}


@dataclass
class State:
    """the state message of design 1.8. `stability` exists on provisionals only
    (1 minus the jsd to the previous provisional) and is left out of the json
    of a final state."""

    kind: str = "final"
    stream_id: str = ""
    utt_id: int = 0
    seq_range: list[int] = field(default_factory=lambda: [0, 0])
    t_media_start: float = 0.0
    t_media_end: float = 0.0
    t_speech_end_capture_ns: int = 0
    t_service_recv_ns: int = 0
    t_emitted_ns: int = 0
    window: str = "full"
    continued_from: int | None = None
    asr: Asr = field(default_factory=Asr)
    audio: Audio = field(default_factory=Audio)
    text: Text = field(default_factory=Text)
    fused: Fused = field(default_factory=Fused)
    discrepancy: Discrepancy = field(default_factory=Discrepancy)
    dialogue: Dialogue = field(default_factory=Dialogue)
    timing_ms: dict[str, float] = field(default_factory=lambda: _zeros(TIMING_KEYS))
    degraded: list[str] = field(default_factory=list)
    stability: float | None = None

    @classmethod
    def empty(cls, stream_id: str, utt_id: int, kind: str = "final") -> "State":
        """a valid state with uniform posteriors, zero events and no claims."""
        return cls(kind=kind, stream_id=stream_id, utt_id=utt_id)

    def canonical(self) -> "State":
        """reorder the dict fields in place to the label orders; returns self."""
        self.audio.emotion = _ordered(self.audio.emotion, AUDIO_EMOTION_KEYS)
        self.audio.events = _ordered(self.audio.events, EVENT_KEYS)
        self.audio.prosody = _ordered(self.audio.prosody, PROSODY_KEYS)
        self.text.emotion = _ordered(self.text.emotion, EMOTIONS)
        self.fused.emotion = _ordered(self.fused.emotion, EMOTIONS)
        self.timing_ms = _ordered(self.timing_ms, TIMING_KEYS)
        self.seq_range = list(self.seq_range)
        return self

    def to_dict(self) -> dict:
        self.canonical()
        d = asdict(self)
        if self.kind != "provisional" and self.stability is None:
            del d["stability"]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), default=_json_default)

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        """unknown keys are ignored, missing keys take the defaults."""
        kw = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name in _NESTED and isinstance(v, dict):
                v = _build(_NESTED[f.name], v)
            kw[f.name] = v
        return cls(**kw).canonical()

    @classmethod
    def from_json(cls, s: str | bytes) -> "State":
        return cls.from_dict(json.loads(s))


def _build(klass, d: dict):
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in d.items() if k in names})


def _json_default(o):
    # numpy scalars and arrays that slipped into a dict field
    if hasattr(o, "item") and not hasattr(o, "__len__"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not json serializable: {type(o).__name__}")
