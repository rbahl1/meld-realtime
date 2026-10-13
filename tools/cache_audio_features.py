"""run the service's sensevoice expert and prosody on every clip of one MELD split
and write the audio feature cache that tools/train_heads.py and tools/evaluate.py read.

cache/audio_{split}.npz, one row per clip, in csv order (load_split) or manifest order:
  keys (str "d_u"), emo_probs (n, 9), event_probs (n, 11), enc_pooled (n, 1024,
  zeros without --encoder-states), prosody (n, 4), asr_text (str), asr_conf (n,),
  tokens_per_s (n,), duration_s (n,), and the same eight arrays with suffix _p2 / _p4
  for the first 2 s / 4 s of the clip (a clip no longer than the prefix repeats its
  full-clip values). every array is row aligned, so readers can reorder or subset by key.
cache/audio_{split}_meta.json: settings, timing, and the clips skipped because the
  wav is missing (count + keys) or could not be read (key + error).
cache/wer_{split}.json: corpus wer of asr_text against the gold csv text (the
  manifest's text with --manifest), jiwer after lowercasing and stripping punctuation.

same path as the service: the expert comes from meld_rt.service.build_sensevoice,
samples are int16 / 32768 as the service's ingest makes them, and prosody is
prosody_scalars on the same samples with the expert's tokens/s. with
--encoder-states the expert runs the edited graph (model_enc.int8.onnx), as the
service does when its audio calibrator needs encoder states.

resumable: a checkpoint (audio_{split}.partial.npz) is written every 500 clips and on
ctrl-c. a rerun with the same settings continues from it.

usage: python tools/cache_audio_features.py --split dev --audio audio/ --out cache/audio_dev.npz --prefixes 2,4
           [--encoder-states] [--threads 4] [--limit N] [--manifest proxy/manifest.json]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt.data import SPLITS, clip_path, key, load_split  # noqa: E402
from meld_rt.experts import prosody  # noqa: E402
from meld_rt.service import ServiceConfig, build_sensevoice  # noqa: E402

SR = 16000
CHECKPOINT_EVERY = 500
PROGRESS_EVERY = 100
DIMS = {"emo_probs": 9, "event_probs": 11, "enc_pooled": 1024, "prosody": 4}
FIELDS = ("emo_probs", "event_probs", "enc_pooled", "prosody", "asr_text", "asr_conf", "tokens_per_s", "duration_s")
APOSTROPHES = "'\u2019\u2018`\u00b4\x91\x92"


# ---------------------------------------------------------------- work list and audio

def work_list(split: str, audio: Path, meld: Path, manifest: Path | None) -> list[tuple[str, Path, str]]:
    """(key, wav path, gold text) per row, in csv order or manifest order."""
    if manifest is None:
        return [(key(u), clip_path(u, audio), u.text) for u in load_split(split, meld)]
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    return [(str(r["key"]), manifest.parent / r["file"], str(r["text"])) for r in rows]


def read_clip(path: Path) -> np.ndarray:
    """float32 samples scaled as the service's ingest does it (int16 / 32768)."""
    x, sr = sf.read(str(path), dtype="int16", always_2d=True)
    if sr != SR or x.shape[1] != 1:
        raise ValueError(f"{sr} hz, {x.shape[1]} channels, expected 16000 hz mono (tools/extract_audio.py)")
    return x[:, 0].astype(np.float32) / 32768.0


def suffix(p: float) -> str:
    return f"_p{p:g}"


def clip_features(sv, pcm: np.ndarray) -> dict:
    r = sv.run(pcm)
    enc = r.enc_pooled if r.enc_pooled is not None else np.zeros(DIMS["enc_pooled"], np.float32)
    return {"emo_probs": np.asarray(r.emo_probs, np.float32), "event_probs": np.asarray(r.event_probs, np.float32),
            "enc_pooled": np.asarray(enc, np.float32), "prosody": prosody.prosody_scalars(pcm, r.tokens_per_s),
            "asr_text": r.text, "asr_conf": float(r.asr_conf), "tokens_per_s": float(r.tokens_per_s),
            "duration_s": len(pcm) / SR}


def row_features(sv, pcm: np.ndarray, prefixes: list[float]) -> tuple[dict, int]:
    """(full clip and prefix features, number of expert runs)."""
    full = clip_features(sv, pcm)
    out, runs = dict(full), 1
    for p in prefixes:
        n = round(p * SR)
        if len(pcm) > n:
            f = clip_features(sv, pcm[:n])
            runs += 1
        else:
            f = full
        out.update({k + suffix(p): v for k, v in f.items()})
    return out, runs


def all_fields(prefixes: list[float]) -> list[str]:
    return list(FIELDS) + [f + suffix(p) for p in prefixes for f in FIELDS]


def base_of(field: str) -> str:
    """emo_probs_p2 -> emo_probs (tokens_per_s has a "_p" of its own, so split at the last one)."""
    return field if field in FIELDS else field.rsplit("_p", 1)[0]


def stack(keys: list[str], rows: dict[str, dict], fields: list[str]) -> dict[str, np.ndarray]:
    out = {"keys": np.array(keys, dtype=str)}
    for f in fields:
        base = base_of(f)
        vals = [rows[k][f] for k in keys]
        if base == "asr_text":
            out[f] = np.array(vals, dtype=str)
        elif base in DIMS:
            out[f] = np.stack(vals).astype(np.float32) if vals else np.zeros((0, DIMS[base]), np.float32)
        else:
            out[f] = np.array(vals, dtype=np.float32)
    return out


# ---------------------------------------------------------------- checkpoint

def partial_path(out: Path) -> Path:
    return out.with_name(out.stem + ".partial.npz")


def meta_path(out: Path) -> Path:
    return out.with_name(out.stem + "_meta.json")


def save_npz(path: Path, arrays: dict, compressed: bool) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def load_partial(out: Path, config: str, fields: list[str]) -> dict[str, dict]:
    p = partial_path(out)
    if not p.exists():
        return {}
    with np.load(p, allow_pickle=False) as z:
        if str(z["config"]) != config:
            raise SystemExit(f"{p} was written with other settings:\n  {z['config']}\nnow:\n  {config}\n"
                             "delete it to start over, or rerun with the old settings")
        keys = [str(k) for k in z["keys"]]
        cols = {f: z[f] for f in fields}
    rows = {}
    for i, k in enumerate(keys):
        rows[k] = {f: (str(cols[f][i]) if base_of(f) == "asr_text" else cols[f][i]) for f in fields}
    return rows


def save_partial(out: Path, order: list[str], rows: dict, fields: list[str], config: str) -> None:
    done = [k for k in order if k in rows]
    save_npz(partial_path(out), {**stack(done, rows, fields), "config": np.array(config)}, compressed=False)


# ---------------------------------------------------------------- wer

def normalize(text: str) -> str:
    """lowercase, apostrophes removed (he's -> hes), other punctuation, symbols and
    control characters (MELD has cp1252 leftovers such as \\x92) become a space."""
    t = text.lower()
    for ch in APOSTROPHES:
        t = t.replace(ch, "")
    t = "".join(" " if unicodedata.category(c)[0] in "PS" or unicodedata.category(c) == "Cc" else c for c in t)
    return " ".join(t.split())


def corpus_wer(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    kept = [(r, h) for r, h in pairs if r]
    rec = {"n": len(kept), "n_empty_reference_excluded": len(pairs) - len(kept),
           "n_empty_hypothesis": sum(1 for _, h in kept if not h)}
    if not kept:
        return {**rec, "wer": None}
    o = jiwer.process_words([r for r, _ in kept], [h for _, h in kept])
    n_ref = o.substitutions + o.deletions + o.hits
    return {**rec, "wer": round(float(o.wer), 6), "reference_words": int(n_ref), "substitutions": int(o.substitutions),
            "deletions": int(o.deletions), "insertions": int(o.insertions), "hits": int(o.hits)}


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--audio", type=Path, default=ROOT / "audio", help="tools/extract_audio.py output folder")
    ap.add_argument("--out", type=Path, default=None, help="default cache/audio_{split}.npz")
    ap.add_argument("--prefixes", default="2,4", help="prefix lengths in seconds, comma separated ('' for none)")
    ap.add_argument("--encoder-states", action="store_true", help="also pool the encoder states (edited graph)")
    ap.add_argument("--threads", type=int, default=4, help="sensevoice intra-op threads")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows")
    ap.add_argument("--manifest", type=Path, default=None, help="make_proxy_clips.py manifest: clips, keys and gold text")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    a = ap.parse_args(argv)
    out = a.out or ROOT / "cache" / f"audio_{a.split}.npz"
    prefixes = sorted({float(p) for p in a.prefixes.split(",") if p.strip()})
    if any(p <= 0 for p in prefixes):
        ap.error("prefixes must be positive")
    fields = all_fields(prefixes)

    work = work_list(a.split, a.audio, a.meld, a.manifest)
    if a.limit is not None:
        work = work[:a.limit]
    order = [k for k, _, _ in work]
    if len(set(order)) != len(order):
        raise SystemExit("duplicate keys in the clip list")
    out.parent.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    cfg = ServiceConfig(models=a.models, threads=a.threads)
    try:
        sv = build_sensevoice(cfg, with_encoder_states=a.encoder_states)
    except RuntimeError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    load_s = time.perf_counter() - t_load
    config = json.dumps({"split": a.split, "prefixes": prefixes, "encoder_states": bool(a.encoder_states),
                         "model": sv.model_path.name, "model_bytes": sv.model_path.stat().st_size,
                         "prosody": "praat" if prosody.HAVE_PARSELMOUTH else "zeros (no parselmouth)",
                         "source": str((a.manifest or a.audio).resolve())}, sort_keys=True)
    wanted = set(order)
    rows = {k: v for k, v in load_partial(out, config, fields).items() if k in wanted}
    resumed = len(rows)
    if resumed:
        print(f"{a.split}: resuming with {resumed} clips from {partial_path(out)}", flush=True)

    missing, failed = [], {}
    runs = done_now = 0
    audio_s = 0.0
    t0 = time.perf_counter()
    try:
        for i, (k, path, _) in enumerate(work, 1):
            if k in rows:
                continue
            if not path.exists():
                missing.append(k)
                continue
            try:
                pcm = read_clip(path)
                rows[k], n = row_features(sv, pcm, prefixes)
            except Exception as e:  # noqa: BLE001 - one bad clip is recorded and skipped
                failed[k] = f"{type(e).__name__}: {e}"
                print(f"{a.split}: {k} failed: {failed[k]}", file=sys.stderr, flush=True)
                continue
            runs += n
            done_now += 1
            audio_s += len(pcm) / SR
            if done_now % CHECKPOINT_EVERY == 0:
                save_partial(out, order, rows, fields, config)
            if done_now % PROGRESS_EVERY == 0:
                el = time.perf_counter() - t0
                left = len(work) - i
                print(f"{a.split}: {i}/{len(work)} rows, {el / done_now * 1000:.0f} ms per clip, "
                      f"about {el / done_now * left / 60:.1f} min left", flush=True)
    except KeyboardInterrupt:
        save_partial(out, order, rows, fields, config)
        print(f"\n{a.split}: interrupted, {len(rows)} clips saved to {partial_path(out)}", file=sys.stderr)
        return 130
    wall_s = time.perf_counter() - t0

    keys = [k for k in order if k in rows]
    save_npz(out, stack(keys, rows, fields), compressed=True)
    partial_path(out).unlink(missing_ok=True)

    gold = {k: text for k, _, text in work}
    wer = {"split": a.split, "gold": "manifest text" if a.manifest else "csv Utterance",
           "normalization": "lowercase, apostrophes removed, other punctuation, symbols and control characters "
                            "replaced by a space, whitespace collapsed",
           **corpus_wer([gold[k] for k in keys], [rows[k]["asr_text"] for k in keys])}
    wer["jiwer"] = importlib.metadata.version("jiwer")
    wer_path = out.parent / f"wer_{a.split}.json"
    wer_path.write_text(json.dumps(wer, indent=1) + "\n", encoding="utf-8", newline="\n")

    meta = {"split": a.split, "rows": len(work), "cached": len(keys),
            "missing": {"count": len(missing), "keys": missing},
            "failed": {"count": len(failed), "errors": failed},
            "prefixes_s": prefixes, "encoder_states": bool(a.encoder_states), "model": str(sv.model_path),
            "prosody": json.loads(config)["prosody"], "threads": a.threads, "limit": a.limit,
            "manifest": str(a.manifest) if a.manifest else None, "audio": None if a.manifest else str(a.audio),
            "resumed_with": resumed, "clips_this_run": done_now, "expert_runs_this_run": runs,
            "audio_s_this_run": round(audio_s, 2), "wall_s": round(wall_s, 2),
            "ms_per_clip": round(wall_s / done_now * 1000, 1) if done_now else None,
            "session_load_s": round(load_s, 2), "wer_file": wer_path.name,
            "command": "python tools/cache_audio_features.py " + " ".join(sys.argv[1:] if argv is None else argv)}
    meta_path(out).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.split}: {len(keys)}/{len(work)} clips -> {out} ({len(missing)} missing, {len(failed)} unreadable, "
          f"{done_now} run now in {wall_s:.1f} s), wer {wer['wer']} over {wer['n']} -> {wer_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
