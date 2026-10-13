"""embed every utterance of one MELD split with the frozen text expert.

writes cache/text_{split}.npy, float32 (n, 768) in load_split order (dialogue_id,
utterance_id), cache/text_{split}_keys.json with the "d_u" key per row, and
cache/text_{split}_meta.json with the wall time. resumable: a finished output
with the right row count is skipped, a partial run continues from its checkpoint.

--from-asr embeds the sensevoice transcripts of cache/audio_{split}.npz instead of
the gold csv text: field asr_text (default) or a prefix field such as asr_text_p2,
one row per cached clip in the audio cache's order, written to
cache/text_asr_{split}.npy (text_asr_{split}_p2.npy, ...). an empty transcript gets
the expert's empty-text embedding (mean over <s> and </s>). the meta file records
the field and the audio cache's sha256, and a finished output from another cache
or field is recomputed.

usage: python tools/embed_text.py --split dev --out cache/text_dev.npy [--threads 2]
       python tools/embed_text.py --split test --from-asr [asr_text_p2] [--audio-cache cache/audio_test.npz] [--threads 2]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def load_asr_utterances(audio_cache: Path, field: str = "asr_text") -> list[tuple[str, str]]:
    """(key, sensevoice transcript) per cached clip, in the audio cache's row order."""
    if field != "asr_text" and not field.startswith("asr_text_p"):
        raise ValueError(f"--from-asr takes asr_text or a prefix field such as asr_text_p2, got {field!r}")
    with np.load(audio_cache, allow_pickle=False) as z:
        if field not in z.files:
            have = sorted(f for f in z.files if f.startswith("asr_text"))
            raise ValueError(f"{audio_cache} has no field {field!r} (has {have})")
        return [(str(k), str(t)) for k, t in zip(z["keys"], z[field])]


def asr_out_name(split: str, field: str) -> str:
    """asr_text -> text_asr_test.npy, asr_text_p2 -> text_asr_test_p2.npy"""
    return f"text_asr_{split}{field[len('asr_text'):]}.npy"


def asr_source(audio_cache: Path, field: str) -> dict:
    h = hashlib.sha256()
    with open(audio_cache, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"file": str(audio_cache), "field": field, "sha256": h.hexdigest()}


def keys_path(out: Path) -> Path:
    return out.with_name(out.stem + "_keys.json")


def meta_path(out: Path) -> Path:
    return out.with_name(out.stem + "_meta.json")


def partial_path(out: Path) -> Path:
    return out.with_name(out.stem + ".partial.npz")


def finished(out: Path, n: int, source: dict | None = None) -> bool:
    """right row count, and with a source the same field and audio cache hash in the meta file."""
    if not out.exists() or not keys_path(out).exists():
        return False
    arr = np.load(out, mmap_mode="r")
    if arr.shape != (n, HIDDEN) or arr.dtype != np.float32:
        return False
    if source is None:
        return True
    old = json.loads(meta_path(out).read_text(encoding="utf-8")).get("source", {}) if meta_path(out).exists() else {}
    return all(old.get(f) == source[f] for f in ("field", "sha256"))


def load_partial(out: Path, n: int, tag: str = "") -> tuple[np.ndarray, int]:
    """rows embedded so far. a checkpoint written for another source (tag) starts over."""
    emb = np.zeros((n, HIDDEN), dtype=np.float32)
    p = partial_path(out)
    if p.exists():
        with np.load(p) as z:
            old_tag = str(z["tag"]) if "tag" in z.files else ""
            if z["emb"].shape == (n, HIDDEN) and old_tag == tag:
                done = int(z["done"])
                emb[:done] = z["emb"][:done]
                return emb, done
    return emb, 0


def save_partial(out: Path, emb: np.ndarray, done: int, tag: str = "") -> None:
    tmp = partial_path(out).with_suffix(".tmp.npz")
    np.savez(tmp, emb=emb, done=done, tag=np.array(tag))
    tmp.replace(partial_path(out))


def embed_split(split: str, out: Path, meld_dir: Path, onnx: Path, tokenizer_dir: Path,
                threads: int, max_tokens: int) -> dict:
    return embed_rows(split, load_utterances(split, meld_dir), out, onnx, tokenizer_dir, threads, max_tokens)


def embed_asr(split: str, audio_cache: Path, field: str, out: Path, onnx: Path, tokenizer_dir: Path,
              threads: int, max_tokens: int) -> dict:
    utts = load_asr_utterances(audio_cache, field)
    return embed_rows(split, utts, out, onnx, tokenizer_dir, threads, max_tokens,
                      source=asr_source(audio_cache, field), n_empty=sum(1 for _, t in utts if not t))


def embed_rows(split: str, utts: list[tuple[str, str]], out: Path, onnx: Path, tokenizer_dir: Path,
               threads: int, max_tokens: int, source: dict | None = None, n_empty: int | None = None) -> dict:
    n = len(utts)
    out.parent.mkdir(parents=True, exist_ok=True)
    if finished(out, n, source):
        print(f"{split}: {out} already has {n} rows, skipping")
        return json.loads(meta_path(out).read_text(encoding="utf-8")) if meta_path(out).exists() else {"split": split, "n": n}
    tag = "" if source is None else f"{source['field']}:{source['sha256']}"

    t_load = time.perf_counter()
    expert = TextExpert(onnx, tokenizer_dir, threads=threads, max_tokens=max_tokens)
    load_s = time.perf_counter() - t_load
    emb, done = load_partial(out, n, tag)
    if done:
        print(f"{split}: resuming at row {done}/{n}")
    t0 = time.perf_counter()
    n_trunc = 0
    for i in range(done, n):
        enc = expert.tokenizer.encode(utts[i][1])
        n_trunc += bool(enc.overflowing)
        emb[i] = expert.encode_ids(enc.ids)
        if (i + 1) % CHECKPOINT_EVERY == 0 and i + 1 < n:
            save_partial(out, emb, i + 1, tag)
            print(f"{split}: {i + 1}/{n} rows, {(time.perf_counter() - t0) / (i + 1 - done) * 1000:.1f} ms per row", flush=True)
    wall_s = time.perf_counter() - t0

    np.save(out, emb)
    keys_path(out).write_text(json.dumps([k for k, _ in utts]), encoding="utf-8", newline="\n")
    partial_path(out).unlink(missing_ok=True)
    meta = {"split": split, "n": n, "dim": HIDDEN, "threads": threads, "max_tokens": max_tokens,
            "rows_embedded_this_run": n - done, "n_truncated_this_run": int(n_trunc),
            "wall_s": round(wall_s, 2), "ms_per_row": round(wall_s / max(1, n - done) * 1000, 2),
            "session_load_s": round(load_s, 2), "onnx": str(onnx), "tokenizer_dir": str(tokenizer_dir),
            "loader": "meld_rt.data" if "meld_rt.data" in sys.modules else "csv"}
    if source is not None:
        meta.update(loader="audio cache", source=source, n_empty_text=n_empty)
    meta_path(out).write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
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
    ap.add_argument("--from-asr", nargs="?", const="asr_text", default=None, metavar="FIELD",
                    help="embed the sensevoice transcripts of the audio cache instead of the gold text "
                         "(asr_text when FIELD is left out, or asr_text_p2, asr_text_p4)")
    ap.add_argument("--audio-cache", type=Path, default=None, help="with --from-asr, default cache/audio_{split}.npz")
    args = ap.parse_args(argv)
    if args.from_asr is None:
        out = args.out or ROOT / "cache" / f"text_{args.split}.npy"
        embed_split(args.split, out, args.meld, args.onnx, args.tokenizer, args.threads, args.max_tokens)
        return 0
    audio_cache = args.audio_cache or ROOT / "cache" / f"audio_{args.split}.npz"
    out = args.out or audio_cache.parent / asr_out_name(args.split, args.from_asr)
    embed_asr(args.split, audio_cache, args.from_asr, out, args.onnx, args.tokenizer, args.threads, args.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
