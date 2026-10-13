#!/usr/bin/env python
"""per-frame face features of the MELD clips, read from MELD.Raw.tar.gz as a
stream (the archive can still be downloading: --follow waits for the bytes
until the file reaches --size).

for every clip of a split (diaX_uttY.mp4 inside the inner train/dev/test
tarballs; a name with a prefix such as final_videos_test... is used only when
the plain name is missing), ffmpeg decodes the video at --fps frames per second
scaled to 640 px wide (at most --max-s seconds), and meld_rt.experts.vision
gives each frame's faces: box and landmarks (divided by the frame width and
height), yunet score, track id, the 7 fer logits (zoo order), the mouth and
face motion, and the 512-d fer embedding on every other frame. frame k is at
t = k / fps from the start of the clip.

shards: cache/video_parts/{split}_{n:04d}.npz, one per --shard clips, so a
rerun skips the clips already written. --merge joins the shards of a split
into cache/video_{split}.npz with these arrays:

  keys (n_clips,) str 'd_u', n_frames, frame_w, frame_h (n_clips,)
  f_clip, f_frame, f_track (n_faces,) int, f_box (n_faces, 4), f_lmk (n_faces, 10),
  f_score, f_mouth, f_face (n_faces,), f_logits (n_faces, 7), f_has_emb (n_faces,) bool
  e_row (n_emb,) index into the face rows, e_vec (n_emb, 512) float16
  fps, meta (json string: models, sha256, settings, failures)

--keep-dir keeps the mp4 files of the --keep-split splits (default test) for
the replay demo and bench. they are never committed.

usage:
  python -I tools/extract_video_features.py --archive MELD.Raw.tar.gz --follow --size 10878146150 \
      --staging /data/staging --keep-dir /data/test_mp4 --jobs 4
  python -I tools/extract_video_features.py --merge --split train,dev,test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLIP_RE = re.compile(r"(?P<prefix>[A-Za-z0-9_-]*?)dia(?P<d>\d+)_utt(?P<u>\d+)\.mp4", re.IGNORECASE)
YUNET = ROOT / "models" / "vision" / "face_detection_yunet_2023mar.onnx"
FER = ROOT / "models" / "vision" / "facial_expression_recognition_mobilefacenet_2022july.onnx"
PARTS = ROOT / "cache" / "video_parts"
WIDTH = 640


class FollowFile:
    """a file that is still being written: read() waits for more bytes until the
    file has `size` bytes (or `stall_s` passes with no growth)."""

    def __init__(self, path: Path, size: int, stall_s: float = 900.0):
        self.f = open(path, "rb")
        self.path = Path(path)
        self.size = int(size)
        self.stall_s = stall_s
        self.pos = 0

    def read(self, n: int = -1) -> bytes:
        want = n if n is not None and n >= 0 else self.size - self.pos
        out = bytearray()
        last_growth = time.monotonic()
        while len(out) < want and self.pos < self.size:
            chunk = self.f.read(want - len(out))
            if chunk:
                out += chunk
                self.pos += len(chunk)
                last_growth = time.monotonic()
                continue
            if time.monotonic() - last_growth > self.stall_s:
                raise OSError(f"{self.path} stopped growing at {self.pos} of {self.size} bytes")
            time.sleep(0.5)
        return bytes(out)

    def close(self) -> None:
        self.f.close()


def split_for(name: str) -> str | None:
    base = Path(name).name.lower()
    for s in ("train", "dev", "test"):
        if base.startswith(s) and base.endswith((".tar.gz", ".tgz")):
            return s
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# worker ---------------------------------------------------------------------
_EXPERT = None


def _init_worker() -> None:
    global _EXPERT
    from meld_rt.experts.vision import VisionExpert
    _EXPERT = VisionExpert(YUNET, FER, threads=1)


def probe(path: Path) -> tuple[int, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
    w, h = (int(v) for v in r.stdout.strip().split(",")[:2])
    return w, h


def decode(path: Path, fps: float, max_s: float) -> tuple[np.ndarray, int, int]:
    w, h = probe(path)
    H = int(round(h * WIDTH / w / 2.0)) * 2
    r = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-t", str(max_s), "-i", str(path),
                        "-vf", f"fps={fps},scale={WIDTH}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace")[-600:])
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = a.size // (H * WIDTH * 3)
    return a[: n * H * WIDTH * 3].reshape(n, H, WIDTH, 3), w, h


def clip_features(path: str, key: str, fps: float, max_s: float, keep_to: str | None) -> dict:
    """runs in a worker: decode, faces per frame, then delete or keep the mp4."""
    p = Path(path)
    try:
        frames, w, h = decode(p, fps, max_s)
        _EXPERT.reset()
        rows = []
        for k, fr in enumerate(frames):
            H, W = fr.shape[:2]
            for f in _EXPERT.process(fr):
                rows.append((k, f.track, f.box / [W, H, W, H], (f.landmarks / [W, H]).reshape(-1), f.score,
                             f.mouth_motion, f.face_motion, f.logits, f.embed if k % 2 == 0 else None))
        out = {"key": key, "n_frames": len(frames), "w": w, "h": h, "rows": rows, "error": None}
    except Exception as e:  # noqa: BLE001 - reported per clip
        out = {"key": key, "n_frames": 0, "w": 0, "h": 0, "rows": [], "error": f"{type(e).__name__}: {e}"[:600]}
    if keep_to:
        dst = Path(keep_to) / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), dst)
    else:
        p.unlink(missing_ok=True)
    return out


# shards ---------------------------------------------------------------------
def done_keys(split: str) -> set[str]:
    out: set[str] = set()
    for f in sorted(PARTS.glob(f"{split}_*.npz")):
        with np.load(f, allow_pickle=False) as z:
            out |= set(z["keys"].tolist())
    return out


def write_shard(split: str, recs: list[dict], fps: float) -> Path:
    PARTS.mkdir(parents=True, exist_ok=True)
    n = len(list(PARTS.glob(f"{split}_*.npz")))
    path = PARTS / f"{split}_{n:04d}.npz"
    keys, nf, fw, fh, errs = [], [], [], [], {}
    cols = {c: [] for c in ("clip", "frame", "track", "box", "lmk", "score", "mouth", "face", "logits", "has_emb")}
    e_row, e_vec = [], []
    for i, r in enumerate(recs):
        keys.append(r["key"]); nf.append(r["n_frames"]); fw.append(r["w"]); fh.append(r["h"])
        if r["error"]:
            errs[r["key"]] = r["error"]
        for (k, tr, box, lmk, sc, mm, fm, lg, emb) in r["rows"]:
            if emb is not None:
                e_row.append(len(cols["clip"]))
                e_vec.append(np.asarray(emb, dtype=np.float16))
            cols["clip"].append(i); cols["frame"].append(k); cols["track"].append(tr)
            cols["box"].append(box); cols["lmk"].append(lmk); cols["score"].append(sc)
            cols["mouth"].append(mm); cols["face"].append(fm); cols["logits"].append(lg)
            cols["has_emb"].append(emb is not None)
    nface = len(cols["clip"])
    np.savez(path, keys=np.array(keys), n_frames=np.array(nf, np.int32), frame_w=np.array(fw, np.int32),
             frame_h=np.array(fh, np.int32),
             f_clip=np.array(cols["clip"], np.int32), f_frame=np.array(cols["frame"], np.int32),
             f_track=np.array(cols["track"], np.int32),
             f_box=np.array(cols["box"], np.float32).reshape(nface, 4),
             f_lmk=np.array(cols["lmk"], np.float32).reshape(nface, 10),
             f_score=np.array(cols["score"], np.float32), f_mouth=np.array(cols["mouth"], np.float32),
             f_face=np.array(cols["face"], np.float32),
             f_logits=np.array(cols["logits"], np.float32).reshape(nface, 7),
             f_has_emb=np.array(cols["has_emb"], bool),
             e_row=np.array(e_row, np.int64), e_vec=np.array(e_vec, np.float16).reshape(len(e_vec), 512),
             fps=np.float32(fps), errors=np.array(json.dumps(errs)))
    return path


def merge(split: str, out_dir: Path) -> Path:
    parts = sorted(PARTS.glob(f"{split}_*.npz"))
    if not parts:
        raise SystemExit(f"no shards for {split} in {PARTS}")
    acc: dict[str, list] = {}
    errors: dict[str, str] = {}
    clip_off = row_off = 0
    fps = None
    for p in parts:
        with np.load(p, allow_pickle=False) as z:
            fps = float(z["fps"])
            errors.update(json.loads(str(z["errors"])))
            for name in z.files:
                if name in ("fps", "errors"):
                    continue
                v = z[name]
                if name == "f_clip":
                    v = v + clip_off
                if name == "e_row":
                    v = v + row_off
                acc.setdefault(name, []).append(v)
            clip_off += len(z["keys"])
            row_off += len(z["f_clip"])
    arrays = {k: np.concatenate(v) for k, v in acc.items()}
    keys = arrays["keys"].tolist()
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{split}: duplicate clip keys across shards")
    meta = {
        "split": split, "fps": fps, "width": WIDTH, "clips": len(keys), "face_rows": int(len(arrays["f_clip"])),
        "embeddings": int(len(arrays["e_row"])), "errors": errors,
        "yunet": {"file": YUNET.name, "sha256": sha256(YUNET)}, "fer": {"file": FER.name, "sha256": sha256(FER)},
        "shards": [p.name for p in parts],
    }
    out = Path(out_dir) / f"video_{split}.npz"
    np.savez(out, fps=np.float32(fps), meta=np.array(json.dumps(meta)), **arrays)
    print(f"{split}: {len(keys)} clips, {meta['face_rows']} face rows, {len(errors)} errors -> {out}", flush=True)
    return out


# driver ---------------------------------------------------------------------
def run(args) -> None:
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    keep_splits = set(s for s in args.keep_split.split(",") if s)
    src = FollowFile(Path(args.archive), args.size) if args.follow else open(args.archive, "rb")
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool, \
            tarfile.open(fileobj=src, mode="r|gz") as outer:
        for m in outer:
            split = split_for(m.name) if m.isfile() else None
            if split is None or split not in args.splits:
                print(f"skip outer member {m.name}", flush=True)
                continue
            have = done_keys(split)
            keep_to = str(Path(args.keep_dir) / split) if (args.keep_dir and split in keep_splits) else None
            pending, recs, prefixed, seen = set(), [], {}, set(have)
            n_sub = 0

            def drain(block_until: int) -> None:
                while len(pending) > block_until:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        pending.discard(fut)
                        recs.append(fut.result())
                        if len(recs) >= args.shard:
                            write_shard(split, recs, args.fps)
                            print(f"{split}: shard written, {n_sub} submitted, {time.monotonic() - t0:.0f} s",
                                  flush=True)
                            recs.clear()

            def submit(path: Path, key: str) -> None:
                nonlocal n_sub
                pending.add(pool.submit(clip_features, str(path), key, args.fps, args.max_s, keep_to))
                n_sub += 1
                drain(args.jobs * 4)

            inner = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
            for im in inner:
                name = Path(im.name).name
                if not im.isfile() or name.startswith("._"):
                    continue
                mm = CLIP_RE.fullmatch(name)
                if not mm:
                    continue
                key = f"{int(mm['d'])}_{int(mm['u'])}"
                if key in seen:
                    continue
                dst = staging / split / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "wb") as out:
                    shutil.copyfileobj(inner.extractfile(im), out, 1 << 20)
                if mm["prefix"]:
                    prefixed.setdefault(key, dst)
                    continue
                seen.add(key)
                old = prefixed.pop(key, None)
                if old is not None:
                    old.unlink(missing_ok=True)
                submit(dst, key)
            for key, path in prefixed.items():
                if key not in seen:
                    seen.add(key)
                    submit(path, key)
            drain(0)
            if recs:
                write_shard(split, recs, args.fps)
            print(f"== {split}: {n_sub} clips this run, {len(have)} from earlier shards, "
                  f"{time.monotonic() - t0:.0f} s", flush=True)
    src.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--follow", action="store_true", help="the archive is still downloading")
    ap.add_argument("--size", type=int, default=10878146150)
    ap.add_argument("--staging", type=Path, default=Path("/tmp/meld_video_staging"))
    ap.add_argument("--keep-dir", type=Path, default=None)
    ap.add_argument("--keep-split", default="test")
    ap.add_argument("--split", default="train,dev,test")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--merge", action="store_true", help="join the shards into cache/video_{split}.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "cache")
    a = ap.parse_args(argv)
    a.splits = [s for s in a.split.split(",") if s]
    if a.merge:
        for s in a.splits:
            merge(s, a.out)
        return 0
    if a.archive is None:
        ap.error("--archive is required unless --merge")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
