#!/usr/bin/env python
"""stream MELD.Raw.tar.gz (outer tar of train/dev/test .tar.gz) into 16 kHz wavs
without unpacking the 10 GB archive to disk. this is how the build container
converted the meld audio (README, "How the audio results were produced").

each inner archive is read as a stream; its mp4 members go to a small staging
folder in chunks, and tools/extract_audio.py (run with uv from REPO) converts
each chunk with --delete-mp4. the last chunk of a split leaves an exact
audio/missing.json. the archive itself is left in place.

usage: python tools/stream_meld_raw.py ARCHIVE STAGING REPO [--chunk 400] [--jobs 4]
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

SPLIT_OF = {"train": "train", "dev": "dev", "test": "test"}


def split_for(name: str) -> str | None:
    base = Path(name).name.lower()
    for k, v in SPLIT_OF.items():
        if base.startswith(k) and base.endswith((".tar.gz", ".tgz")):
            return v
    return None


def convert_chunk(repo: Path, staging: Path, split: str, jobs: int) -> None:
    cmd = ["uv", "run", "python", "tools/extract_audio.py", "--raw", str(staging), "--out", "audio/",
           "--split", split, "--jobs", str(jobs), "--delete-mp4"]
    r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False)
    last = [l for l in r.stdout.splitlines() if l.startswith(f"{split}:")]
    print(last[-1] if last else r.stdout[-300:], flush=True)
    if r.returncode not in (0,):
        print(r.stderr[-1500:], file=sys.stderr, flush=True)
        raise SystemExit(f"extract_audio failed for {split} (rc {r.returncode})")


def run(archive: Path, staging: Path, repo: Path, chunk: int, jobs: int) -> None:
    seen = {}
    with tarfile.open(archive, mode="r|gz") as outer:
        for m in outer:
            split = split_for(m.name) if m.isfile() else None
            if split is None:
                print(f"skip outer member {m.name}", flush=True)
                continue
            root = staging / split
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True)
            n = pending = 0
            inner_f = outer.extractfile(m)
            with tarfile.open(fileobj=inner_f, mode="r|gz") as inner:
                for im in inner:
                    nm = im.name
                    if not im.isfile() or not nm.lower().endswith(".mp4") or Path(nm).name.startswith("._"):
                        continue
                    dst = root / nm.lstrip("./")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src = inner.extractfile(im)
                    with open(dst, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
                    n += 1
                    pending += 1
                    if pending >= chunk:
                        convert_chunk(repo, root, split, jobs)
                        pending = 0
            if pending or n == 0:
                convert_chunk(repo, root, split, jobs)
            seen[split] = n
            shutil.rmtree(root, ignore_errors=True)
            print(f"== {split}: {n} mp4 members streamed from {m.name}", flush=True)
    print(f"done: {seen}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("staging", type=Path)
    ap.add_argument("repo", type=Path)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    run(a.archive, a.staging, a.repo, a.chunk, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
