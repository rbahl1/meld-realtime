"""embed every utterance of one MELD split with the frozen text expert.

writes cache/text_{split}.npy, float32 (n, 768) in load_split order (dialogue_id,
utterance_id), cache/text_{split}_keys.json with the "d_u" key per row, and
cache/text_{split}_meta.json with the wall time. resumable: a finished output
with the right row count is skipped, a partial run continues from its checkpoint.

usage: python tools/embed_text.py --split dev --out cache/text_dev.npy [--threads 2]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt.experts.text import HIDDEN, TextExpert  # noqa: E402

CHECKPOINT_EVERY = 500


def load_utterances(split: str, meld_dir: Path) -> list[tuple[str, str]]:
    """(key, text) per utterance in load_split order; uses meld_rt.data when present."""
    try:
        from meld_rt.data import key, load_split
    except ImportError:
        return load_utterances_csv(split, meld_dir)
    return [(key(u), u.text) for u in load_split(split, meld_dir)]


def load_utterances_csv(split: str, meld_dir: Path) -> list[tuple[str, str]]:
    """same order and keys as meld_rt.data.load_split, read straight from the csv."""
    rows = []
    with (meld_dir / f"{split}_sent_emo.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Dialogue_ID"]), int(r["Utterance_ID"]), r["Utterance"]))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [(f"{d}_{u}", text) for d, u, text in rows]


def keys_path(out: Path) -> Path:
    return out.with_name(out.stem + "_keys.json")


def meta_path(out: Path) -> Path:
    return out.with_name(out.stem + "_meta.json")


def partial_path(out: Path) -> Path:
    return out.with_name(out.stem + ".partial.npz")


def finished(out: Path, n: int) -> bool:
    if not out.exists() or not keys_path(out).exists():
        return False
    arr = np.load(out, mmap_mode="r")
    return arr.shape == (n, HIDDEN) and arr.dtype == np.float32


def load_partial(out: Path, n: int) -> tuple[np.ndarray, int]:
    emb = np.zeros((n, HIDDEN), dtype=np.float32)
    p = partial_path(out)
    if p.exists():
        with np.load(p) as z:
            if z["emb"].shape == (n, HIDDEN):
                done = int(z["done"])
                emb[:done] = z["emb"][:done]
                return emb, done
    return emb, 0


def save_partial(out: Path, emb: np.ndarray, done: int) -> None:
    tmp = partial_path(out).with_suffix(".tmp.npz")
    np.savez(tmp, emb=emb, done=done)
    tmp.replace(partial_path(out))


def embed_split(split: str, out: Path, meld_dir: Path, onnx: Path, tokenizer_dir: Path,
                threads: int, max_tokens: int) -> dict:
    utts = load_utterances(split, meld_dir)
    n = len(utts)
    out.parent.mkdir(parents=True, exist_ok=True)
    if finished(out, n):
        print(f"{split}: {out} already has {n} rows, skipping")
        return json.loads(meta_path(out).read_text()) if meta_path(out).exists() else {"split": split, "n": n}

    t_load = time.perf_counter()
    expert = TextExpert(onnx, tokenizer_dir, threads=threads, max_tokens=max_tokens)
    load_s = time.perf_counter() - t_load
    emb, done = load_partial(out, n)
    if done:
        print(f"{split}: resuming at row {done}/{n}")
    t0 = time.perf_counter()
    n_trunc = 0
    for i in range(done, n):
        enc = expert.tokenizer.encode(utts[i][1])
        n_trunc += bool(enc.overflowing)
        emb[i] = expert.encode_ids(enc.ids)
        if (i + 1) % CHECKPOINT_EVERY == 0 and i + 1 < n:
            save_partial(out, emb, i + 1)
            print(f"{split}: {i + 1}/{n} rows, {(time.perf_counter() - t0) / (i + 1 - done) * 1000:.1f} ms per row", flush=True)
    wall_s = time.perf_counter() - t0

    np.save(out, emb)
    keys_path(out).write_text(json.dumps([k for k, _ in utts]))
    partial_path(out).unlink(missing_ok=True)
    meta = {"split": split, "n": n, "dim": HIDDEN, "threads": threads, "max_tokens": max_tokens,
            "rows_embedded_this_run": n - done, "n_truncated_this_run": int(n_trunc),
            "wall_s": round(wall_s, 2), "ms_per_row": round(wall_s / max(1, n - done) * 1000, 2),
            "session_load_s": round(load_s, 2), "onnx": str(onnx), "tokenizer_dir": str(tokenizer_dir),
            "loader": "meld_rt.data" if "meld_rt.data" in sys.modules else "csv"}
    meta_path(out).write_text(json.dumps(meta, indent=1))
    print(f"{split}: {n} rows -> {out} in {wall_s:.1f} s ({meta['ms_per_row']} ms per row, "
          f"{threads} threads, {n_trunc} truncated)")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--out", type=Path, default=None, help="default cache/text_{split}.npy")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--onnx", type=Path, default=ROOT / "models" / "roberta-base-11.onnx")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "assets" / "roberta_tokenizer")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args(argv)
    out = args.out or ROOT / "cache" / f"text_{args.split}.npy"
    embed_split(args.split, out, args.meld, args.onnx, args.tokenizer, args.threads, args.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
