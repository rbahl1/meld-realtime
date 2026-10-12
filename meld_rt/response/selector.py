"""picks one line from the bank by (fused label, discrepancy direction), the
confidence bucket, a recency penalty and a keyphrase slot. pure function of
(state, recent ids, seed): no randomness at run time (design 1.9).

buckets: low (< low_conf) takes a hold_back line, mid prefers ask, high
(>= high_conf) prefers acknowledge or name_the_gap. the thresholds are chosen on
dev for a calibrated posterior, so a state with fused.calibrated false (the
poe_uncalibrated fallback, or poe without a fusion head) is capped at mid: it
can ask but never asserts, and the grounding says rule=uncalibrated_cap when
the cap changed the bucket."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from meld_rt.discrepancy import keyphrase
from meld_rt.labels import EMOTIONS
from meld_rt.state import State

STYLES = ("acknowledge", "ask", "name_the_gap")
CELL_DIRECTIONS = ("none", "tone_more_negative", "tone_more_positive", "tone_surprised")
MAX_WORDS = 12
MAX_SLOT_TEMPLATE_WORDS = 8     # so a 4-word keyphrase keeps the filled line at 12
SLOT = "{keyphrase}"
EVENT_PREFIX_MIN = 0.5
EVENT_PREFIX_KEYS = ("laughter", "cry")
ASR_DEGRADED = ("asr_failed", "low_asr_conf")


@dataclass
class Response:
    text: str
    template_id: str
    style: str
    grounding: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Line:
    id: str
    style: str
    text: str

    @property
    def has_slot(self) -> bool:
        return SLOT in self.text


def cell_key(label: str, direction: str) -> str:
    """mixed shares the none cell (design 1.9)."""
    return f"{label}|{'none' if direction == 'mixed' else direction}"


def word_count(text: str) -> int:
    return len(text.split())


def _check_line(line: Line, where: str) -> None:
    if word_count(line.text) > MAX_WORDS:
        raise ValueError(f"{where} {line.id}: more than {MAX_WORDS} words")
    if line.has_slot and word_count(line.text) > MAX_SLOT_TEMPLATE_WORDS:
        raise ValueError(f"{where} {line.id}: slot template longer than {MAX_SLOT_TEMPLATE_WORDS} words")
    if "!" in line.text:
        raise ValueError(f"{where} {line.id}: exclamation mark")


def _lines(items: list, style: str | None, where: str, seen: set[str]) -> list[Line]:
    out = []
    for it in items:
        line = Line(str(it["id"]), style or str(it["style"]), str(it["text"]))
        if line.style not in STYLES and style is None:
            raise ValueError(f"{where} {line.id}: unknown style {line.style}")
        if line.id in seen:
            raise ValueError(f"duplicate template id {line.id}")
        seen.add(line.id)
        _check_line(line, where)
        out.append(line)
    return out


def load_bank(path: Path) -> dict:
    """parse and validate bank.yaml: 28 cells with at least 3 variants each,
    hold_back lines, asr_failed lines, two event prefix lists."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seen: set[str] = set()
    cells: dict[str, list[Line]] = {}
    for label in EMOTIONS:
        for d in CELL_DIRECTIONS:
            key = f"{label}|{d}"
            items = raw.get("cells", {}).get(key)
            if not items or len(items) < 3:
                raise ValueError(f"cell {key} needs at least 3 variants")
            cells[key] = _lines(items, None, f"cell {key}", seen)
    hold = _lines(raw.get("hold_back", []), "hold_back", "hold_back", seen)
    asr = _lines(raw.get("asr_failed", []), "hold_back", "asr_failed", seen) or hold
    if not hold:
        raise ValueError("hold_back is empty")
    prefix = {k: [str(s) for s in raw.get("event_prefix", {}).get(k, [])] for k in EVENT_PREFIX_KEYS}
    for k, v in prefix.items():
        if not v:
            raise ValueError(f"event_prefix.{k} is empty")
        for s in v:
            if "!" in s or word_count(s) > MAX_WORDS:
                raise ValueError(f"event_prefix.{k}: bad clause {s!r}")
    return {"cells": cells, "hold_back": hold, "asr_failed": asr, "event_prefix": prefix,
            "draft": raw.get("draft")}


def _style_order(bucket: str, flagged: bool) -> list[str]:
    if bucket == "high":
        return ["name_the_gap", "acknowledge", "ask"] if flagged else ["acknowledge", "ask", "name_the_gap"]
    return ["ask", "name_the_gap", "acknowledge"] if flagged else ["ask", "acknowledge", "name_the_gap"]


class ResponseSelector:
    def __init__(self, bank_path: Path | None = None, low_conf: float = 0.45, high_conf: float = 0.70,
                 no_repeat_turns: int = 5, seed: int = 0):
        if bank_path is None:
            bank_path = Path(__file__).with_name("bank.yaml")
        bank = load_bank(bank_path)
        self.cells: dict[str, list[Line]] = bank["cells"]
        self.hold_back: list[Line] = bank["hold_back"]
        self.asr_failed: list[Line] = bank["asr_failed"]
        self.event_prefix: dict[str, list[str]] = bank["event_prefix"]
        self.low_conf = float(low_conf)
        self.high_conf = float(high_conf)
        self.no_repeat_turns = int(no_repeat_turns)
        self.seed = int(seed)
        smallest = min(len(v) for v in self.cells.values())
        if smallest + len(self.hold_back) <= self.no_repeat_turns:
            raise ValueError("bank too small for the no-repeat window")

    def bucket(self, confidence: float, calibrated: bool = True) -> str:
        """low / mid / high by the thresholds; an uncalibrated confidence never reaches high."""
        if confidence < self.low_conf:
            return "low"
        return "high" if confidence >= self.high_conf and calibrated else "mid"

    def _rot(self, utt_id: int, key: str) -> int:
        # deterministic tie-break that rotates with the utterance id
        return zlib.crc32(f"{self.seed}:{utt_id}:{key}".encode())

    def _pick(self, cands: list[Line], score, utt_id: int) -> Line:
        return min(cands, key=lambda l: (score(l), self._rot(utt_id, l.id)))

    def select(self, state: State, recent_template_ids: list[str] | None = None) -> Response:
        recent = set(list(recent_template_ids or [])[-self.no_repeat_turns:]) if self.no_repeat_turns > 0 else set()
        conf = float(state.fused.confidence)
        calibrated = bool(state.fused.calibrated)
        degraded = [d for d in state.degraded if d in ASR_DEGRADED]
        kp: str | None = None
        grounding: list[str] = []

        if degraded:
            pool, prefs = self.asr_failed, []
            grounding.append(f"degraded={degraded[0]}")
        elif self.bucket(conf, calibrated) == "low":
            pool, prefs = self.hold_back, []
            grounding.append(f"fused.confidence={conf:.2f}<{self.low_conf:.2f}")
        else:
            label, direction = state.fused.label, state.discrepancy.direction
            pool = self.cells[cell_key(label, direction)]
            prefs = _style_order(self.bucket(conf, calibrated), direction != "none")
            kp = state.text.keyphrase or keyphrase(state.asr.text or state.text.partial or "") or None
            grounding += [f"fused.label={label}@{conf:.2f}", f"discrepancy.direction={direction}"]
            if not calibrated and conf >= self.high_conf:
                grounding.append("rule=uncalibrated_cap")

        cands = [l for l in pool if kp or not l.has_slot]

        def score(l: Line) -> int:
            s = (prefs.index(l.style) if l.style in prefs else len(prefs)) * 10
            s -= 1 if l.has_slot else 0
            s += 100 if l.id in recent else 0
            return s

        chosen = self._pick(cands, score, state.utt_id)
        if chosen.id in recent and pool is not self.hold_back:
            # the cell is used up inside the window: say something that claims nothing
            fresh = [l for l in self.hold_back if l.id not in recent]
            if fresh:
                chosen = self._pick(fresh, lambda l: 0, state.utt_id)
                grounding.append("rule=no_repeat")

        text = chosen.text
        if chosen.has_slot:
            text = text.replace(SLOT, kp)
            grounding.append(f"text.keyphrase='{kp}'")

        ev = state.audio.events
        hits = [(float(ev.get(k, 0.0)), k) for k in EVENT_PREFIX_KEYS if float(ev.get(k, 0.0)) > EVENT_PREFIX_MIN]
        if hits:
            p, k = max(hits)
            clauses = self.event_prefix[k]
            text = f"{clauses[self._rot(state.utt_id, k) % len(clauses)]} {text}"
            grounding.append(f"audio.events.{k}={p:.2f}")

        return Response(text=text, template_id=chosen.id, style=chosen.style, grounding=grounding)
