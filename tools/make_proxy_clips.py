#!/usr/bin/env python
"""synthesize MELD utterances with piper (en_US-lessac-medium) through sherpa-onnx
into 16 khz mono pcm16 wavs: OUT/{split}_{i:04d}.wav (i from 1) and OUT/manifest.json,
a list of {file, key, text, label, sentiment, speaker, duration_s}. the clips
exercise the audio path and the protocol end to end; they are never reported as
a result (tts is affectless; sensevoice labels it neutral). deterministic: the
first --n utterances of the split in (dialogue_id, utterance_id) order that have
at least --min-words words with a letter in them.

usage: python tools/make_proxy_clips.py --n 100 --split dev --out proxy/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt.data import key, load_split

SR = 16000
PIPER_DIR = ROOT / "models/piper/en_US-lessac-medium"
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", "\x92": "'"}


def clean_text(text: str) -> str:
    """ascii quotes and dashes, no control characters, single spaces; espeak reads the rest."""
    for a, b in _QUOTES.items():
        text = text.replace(a, b)
    text = "".join(ch for ch in text if ch.isprintable())
    return re.sub(r"\s+", " ", text).strip()


def has_words(text: str, min_words: int) -> bool:
    return sum(1 for w in text.split() if re.search(r"[A-Za-z]", w)) >= min_words


def load_tts(piper_dir: Path, threads: int):
    import sherpa_onnx

    voice = next(piper_dir.glob("*.onnx"))
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(voice), lexicon="", tokens=str(piper_dir / "tokens.txt"),
                                                      data_dir=str(piper_dir / "espeak-ng-data")),
            num_threads=threads, debug=False, provider="cpu"),
        rule_fsts="", max_num_sentences=1)
    if not cfg.validate():
        raise RuntimeError(f"piper config invalid for {piper_dir}")
    return sherpa_onnx.OfflineTts(cfg), voice.name


def synth(tts, text: str) -> np.ndarray:
    """float32 at 16 khz."""
    from scipy.signal import resample_poly

    a = tts.generate(text, sid=0, speed=1.0)
    x = np.asarray(a.samples, dtype=np.float32)
    if a.sample_rate != SR:
        g = gcd(SR, a.sample_rate)
        x = resample_poly(x, SR // g, a.sample_rate // g).astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="piper proxy clips for MELD utterances")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--out", type=Path, default=ROOT / "proxy")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--piper", type=Path, default=PIPER_DIR)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--min-words", type=int, default=2)
    a = ap.parse_args(argv)

    utts = [u for u in load_split(a.split, a.meld) if has_words(clean_text(u.text), a.min_words)][:a.n]
    if len(utts) < a.n:
        print(f"only {len(utts)} usable utterances in {a.split}", file=sys.stderr)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tts, voice = load_tts(a.piper, a.threads)
    t_load = time.perf_counter() - t0
    rows = []
    total_s = 0.0
    t1 = time.perf_counter()
    for i, u in enumerate(utts, 1):
        text = clean_text(u.text)
        x = synth(tts, text)
        name = f"{a.split}_{i:04d}.wav"
        sf.write(str(a.out / name), (x * 32767).round().astype(np.int16), SR, subtype="PCM_16")
        dur = round(len(x) / SR, 3)
        total_s += dur
        rows.append({"file": name, "key": key(u), "text": u.text, "tts_text": text, "label": u.emotion,
                     "sentiment": u.sentiment, "speaker": u.speaker, "duration_s": dur})
        if i % 20 == 0 or i == len(utts):
            print(f"{i}/{len(utts)} clips, {total_s:.1f} s of audio, {time.perf_counter() - t1:.1f} s wall", flush=True)
    t_synth = time.perf_counter() - t1
    (a.out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    meta = {"split": a.split, "n": len(rows), "voice": voice, "sr": SR, "tts": "piper via sherpa-onnx",
            "load_s": round(t_load, 2), "synth_wall_s": round(t_synth, 2), "audio_s": round(total_s, 2),
            "rtf": round(t_synth / total_s, 4) if total_s else None,
            "duration_mean_s": round(total_s / len(rows), 3) if rows else None,
            "duration_max_s": max((r["duration_s"] for r in rows), default=None),
            "labels": {lab: sum(1 for r in rows if r["label"] == lab) for lab in sorted({r["label"] for r in rows})}}
    (a.out / "manifest_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
