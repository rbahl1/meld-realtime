#!/usr/bin/env python
"""download the external model files (urllib only), verify sha256 against the
table below, extract the needed archive members, write the computed hashes to
results/provenance_hashes.json. idempotent: files whose hash already matches are
not touched. --check verifies and writes the json without downloading.

hashes in TABLE were computed with sha256sum on the files present in the build
container on 2026-10-12; the two sherpa-onnx tar.bz2 archives were not kept, so
their members are verified instead of the archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SV_DIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
PIPER_DIR = "vits-piper-en_US-lessac-medium"


@dataclass
class Asset:
    name: str
    url: str
    dest: str  # relative to --dest (or to --dl-dir when dl_dir is True)
    sha256: str | None = None  # None: existence only
    archive: bool = False
    members: dict[str, str] = field(default_factory=dict)  # archive member -> dest-relative path
    member_sha256: dict[str, str] = field(default_factory=dict)  # dest-relative path -> sha256
    member_dirs: dict[str, str] = field(default_factory=dict)  # archive dir prefix -> dest-relative dir
    dl_dir: bool = False


TABLE = [
    Asset("silero_vad.onnx", f"{SHERPA}/asr-models/silero_vad.onnx", "silero_vad.onnx",
          sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"),
    Asset("sensevoice", f"{SHERPA}/asr-models/{SV_DIR}.tar.bz2", "sensevoice", archive=True,
          members={f"{SV_DIR}/model.int8.onnx": "sensevoice/model.int8.onnx",
                   f"{SV_DIR}/tokens.txt": "sensevoice/tokens.txt"},
          member_dirs={f"{SV_DIR}/test_wavs/": "sensevoice/test_wavs/"},
          member_sha256={"sensevoice/model.int8.onnx": "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
                         "sensevoice/tokens.txt": "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"}),
    Asset("piper", f"{SHERPA}/tts-models/{PIPER_DIR}.tar.bz2", "piper/en_US-lessac-medium", archive=True,
          members={f"{PIPER_DIR}/en_US-lessac-medium.onnx": "piper/en_US-lessac-medium/en_US-lessac-medium.onnx",
                   f"{PIPER_DIR}/en_US-lessac-medium.onnx.json": "piper/en_US-lessac-medium/en_US-lessac-medium.onnx.json",
                   f"{PIPER_DIR}/tokens.txt": "piper/en_US-lessac-medium/tokens.txt",
                   f"{PIPER_DIR}/MODEL_CARD": "piper/en_US-lessac-medium/MODEL_CARD"},
          member_dirs={f"{PIPER_DIR}/espeak-ng-data/": "piper/en_US-lessac-medium/espeak-ng-data/"},
          member_sha256={"piper/en_US-lessac-medium/en_US-lessac-medium.onnx": "4ba07d8549906668ee855fd9abf9faf66c5db74742712ff026a159f7277fca9f",
                         "piper/en_US-lessac-medium/en_US-lessac-medium.onnx.json": "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
                         "piper/en_US-lessac-medium/tokens.txt": "87c8ef66eae5473ed0cc0366b3964c736ca6c5f676c979522ea31234e47430b9"}),
    Asset("roberta-base-11.onnx",
          "https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/roberta/model/roberta-base-11.onnx",
          "roberta-base-11.onnx", sha256="ad476a33a4b227f6e6b2e1c7192df1b61640657f26b390857ab943de66236c0b"),
    Asset("en_core_web_trf wheel",
          "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
          "en_core_web_trf-3.8.0-py3-none-any.whl", dl_dir=True,
          sha256="272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dst.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", file=sys.stderr)
    print(f"\r  {dst.name}: {done / 1e6:.0f} MB in {time.perf_counter() - t0:.0f} s", file=sys.stderr)
    tmp.replace(dst)


def extract(archive: Path, asset: Asset, dest: Path) -> None:
    with tarfile.open(archive, "r:bz2") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name.removeprefix("./")
            target = None
            if name in asset.members:
                target = dest / asset.members[name]
            else:
                for prefix, out_dir in asset.member_dirs.items():
                    if name.startswith(prefix):
                        target = dest / out_dir / name[len(prefix):]
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def expected_files(asset: Asset, dest: Path, dl_dir: Path) -> dict[Path, str | None]:
    """dest-relative files this asset must provide, with their expected sha256 (None: existence)."""
    if asset.archive:
        files = {dest / rel: asset.member_sha256.get(rel) for rel in asset.members.values()}
        for out_dir in asset.member_dirs.values():
            files[dest / out_dir] = None
        return files
    base = dl_dir if asset.dl_dir else dest
    return {base / asset.dest: asset.sha256}


def status(asset: Asset, dest: Path, dl_dir: Path) -> tuple[bool, list[dict]]:
    ok = True
    rows = []
    for path, expected in expected_files(asset, dest, dl_dir).items():
        row = {"asset": asset.name, "url": asset.url, "path": str(path.relative_to(ROOT)) if path.is_absolute() and ROOT in path.parents else str(path),
               "exists": path.exists(), "expected_sha256": expected}
        if path.is_dir():
            row["n_files"] = sum(1 for _ in path.rglob("*") if _.is_file())
            row["ok"] = row["n_files"] > 0
        elif path.exists():
            row["bytes"] = path.stat().st_size
            row["sha256"] = sha256_of(path)
            row["ok"] = expected is None or row["sha256"] == expected
        else:
            row["ok"] = False
        ok = ok and row["ok"]
        rows.append(row)
    return ok, rows


def fetch(asset: Asset, dest: Path, dl_dir: Path) -> None:
    if asset.archive:
        with tempfile.TemporaryDirectory(dir=dest) as td:
            arc = Path(td) / Path(asset.url).name
            download(asset.url, arc)
            extract(arc, asset, dest)
    else:
        base = dl_dir if asset.dl_dir else dest
        download(asset.url, base / asset.dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=ROOT / "models")
    ap.add_argument("--dl-dir", type=Path, default=ROOT / "verify_dl", help="where the spacy wheel goes")
    ap.add_argument("--check", action="store_true", help="verify only, never download")
    ap.add_argument("--provenance", type=Path, default=ROOT / "results/provenance_hashes.json")
    ap.add_argument("--only", nargs="*", help="asset names to restrict to")
    args = ap.parse_args()
    dest, dl_dir = args.dest.resolve(), args.dl_dir.resolve()

    all_ok = True
    report = []
    for asset in TABLE:
        if args.only and asset.name not in args.only:
            continue
        ok, rows = status(asset, dest, dl_dir)
        if not ok and not args.check:
            print(f"fetching {asset.name}", file=sys.stderr)
            try:
                fetch(asset, dest, dl_dir)
            except (OSError, tarfile.TarError) as e:  # keep going so one blocked host does not hide the others
                print(f"  failed: {e}", file=sys.stderr)
            ok, rows = status(asset, dest, dl_dir)
        for r in rows:
            mark = "ok " if r["ok"] else "BAD"
            extra = f"{r.get('n_files')} files" if "n_files" in r else (r.get("sha256") or "missing")
            print(f"{mark} {r['path']}  {extra}")
        all_ok = all_ok and ok
        report.extend(rows)

    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with open(args.provenance, "w") as f:
        json.dump({"generated_by": "tools/fetch_models.py", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "all_ok": all_ok, "files": report}, f, indent=1)
    print(f"{'all verified' if all_ok else 'some files missing or mismatched'}; wrote {args.provenance}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
