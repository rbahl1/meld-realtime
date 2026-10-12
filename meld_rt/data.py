"""meld csv access: utterances, cache keys, clip paths, context windows, class weights.

reads the declare-lab MELD text files (train/dev/test_sent_emo.csv). no audio is
touched here; clip_path only names where a clip lives.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meld_rt import labels

SPLITS = ("train", "dev", "test")
MELD_DIR = Path(__file__).resolve().parent.parent / "meld"

# csv column names as shipped by declare-lab
_COLS = ("Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID", "StartTime", "EndTime")


@dataclass(frozen=True)
class Utt:
    split: str
    dialogue_id: int
    utterance_id: int
    speaker: str
    text: str
    emotion: str
    sentiment: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def parse_time(s: str) -> float:
    """subtitle timestamp 'hh:mm:ss,ms' -> seconds. the hour field is sometimes one digit."""
    h, m, rest = s.strip().split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def csv_path(split: str, meld_dir: Path = MELD_DIR) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return Path(meld_dir) / f"{split}_sent_emo.csv"


def load_split(split: str, meld_dir: Path = MELD_DIR) -> list[Utt]:
    """all utterances of a split, sorted by (dialogue_id, utterance_id). text is kept verbatim (utf-8, curly quotes)."""
    path = csv_path(split, meld_dir)
    out: list[Utt] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for r in reader:
            emo, sent = r["Emotion"].strip(), r["Sentiment"].strip()
            if emo not in labels.EMO_INDEX:
                raise ValueError(f"{path}: unknown emotion {emo!r}")
            if sent not in labels.SENT_INDEX:
                raise ValueError(f"{path}: unknown sentiment {sent!r}")
            out.append(Utt(
                split=split,
                dialogue_id=int(r["Dialogue_ID"]),
                utterance_id=int(r["Utterance_ID"]),
                speaker=r["Speaker"],
                text=r["Utterance"],
                emotion=emo,
                sentiment=sent,
                start_s=parse_time(r["StartTime"]),
                end_s=parse_time(r["EndTime"]),
            ))
    out.sort(key=lambda u: (u.dialogue_id, u.utterance_id))
    keys = [key(u) for u in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate (dialogue_id, utterance_id)")
    return out


def key(u: Utt) -> str:
    """feature cache key, 'd_u'."""
    return f"{u.dialogue_id}_{u.utterance_id}"


def clip_path(u: Utt, audio_dir: Path) -> Path:
    return Path(audio_dir) / u.split / f"dia{u.dialogue_id}_utt{u.utterance_id}.wav"


def dialogues(utts: Iterable[Utt]) -> list[list[Utt]]:
    """group by (split, dialogue_id) in order of first appearance; each group in utterance_id order."""
    groups: dict[tuple[str, int], list[Utt]] = {}
    for u in utts:
        groups.setdefault((u.split, u.dialogue_id), []).append(u)
    return [sorted(g, key=lambda u: u.utterance_id) for g in groups.values()]


def context_rows(utts: Sequence[Utt], k: int, same_speaker_only: bool = False) -> list[tuple[Utt, list[Utt]]]:
    """(utterance, previous k utterances of the same dialogue, oldest first), in input order.

    previous means the rows before it in utterance_id order, whatever the id gaps
    (72 train dialogues have gaps). with same_speaker_only, only earlier rows by
    the same speaker count. nothing after the utterance is ever included.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    ctx: dict[tuple[str, int, int], list[Utt]] = {}
    for dia in dialogues(utts):
        for i, u in enumerate(dia):
            prev = dia[:i]
            if same_speaker_only:
                prev = [p for p in prev if p.speaker == u.speaker]
            ctx[(u.split, u.dialogue_id, u.utterance_id)] = prev[len(prev) - k:] if k else []
    return [(u, ctx[(u.split, u.dialogue_id, u.utterance_id)]) for u in utts]


def emotion_ids(utts: Iterable[Utt]) -> np.ndarray:
    return np.array([labels.EMO_INDEX[u.emotion] for u in utts], dtype=np.int64)


def sentiment_ids(utts: Iterable[Utt]) -> np.ndarray:
    return np.array([labels.SENT_INDEX[u.sentiment] for u in utts], dtype=np.int64)


def emotion_counts(utts: Iterable[Utt]) -> np.ndarray:
    """per-class counts in labels.EMOTIONS order."""
    return np.bincount(emotion_ids(utts), minlength=len(labels.EMOTIONS)).astype(np.int64)


def class_weights(scheme: str, train: Sequence[Utt] | None = None, meld_dir: Path = MELD_DIR) -> np.ndarray:
    """emotion class weights in labels.EMOTIONS order.

    "meld_readme": the weights from the MELD README. "balanced": n / (n_classes * count_c)
    from the train split (sklearn's convention), loaded from meld_dir unless train is given.
    "none": ones.
    """
    n_cls = len(labels.EMOTIONS)
    if scheme == "meld_readme":
        return np.array(labels.MELD_README_WEIGHTS, dtype=np.float64)
    if scheme == "none":
        return np.ones(n_cls, dtype=np.float64)
    if scheme == "balanced":
        counts = emotion_counts(train if train is not None else load_split("train", meld_dir))
        if (counts == 0).any():
            raise ValueError("balanced weights need every class present in train")
        return counts.sum() / (n_cls * counts.astype(np.float64))
    raise ValueError(f"unknown class weight scheme {scheme!r}")
