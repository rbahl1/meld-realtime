#!/usr/bin/env python
"""download the external model files (urllib only), verify sha256 against the
table below, extract the needed archive members, write the computed hashes to
results/provenance_hashes.json. idempotent: files whose hash already matches are
not touched. --check verifies and writes the json without downloading.

hashes in TABLE were computed with sha256sum on the files present in the build
container on 2026-10-12; the two sherpa-onnx tar.bz2 archives were not kept, so
their members are verified instead of the archives. the hugging face files
(vision models, LLM) are pinned to a revision, and their hashes equal the lfs
sha256 the hub lists for that revision.

groups: core (the text and audio path), vision (YuNet, FER), llm (the deployed
reply generator: meld_rt.response.generator DEFAULT_MODEL / DEFAULT_ONNX and the
tokenizer and config files) and llm_candidates (the other candidate files of
spec 4, about 3 GB more, for tools/bench_generator.py). the default is core, as
before the extension; --groups core,vision,llm adds what --vision --reply llm
needs, --groups all every candidate.

the provenance file keeps one row per file of every group: rows of assets this
run did not check (other groups, or assets outside --only) are carried over
from the existing file unchanged, so a core run rewrites the core rows and
keeps the vision and llm rows. all_ok in the file is over every row in it; the
exit status is over the rows checked in this run."""
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
sys.path.insert(0, str(ROOT))
from meld_rt.response import generator as G  # noqa: E402

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
HF = "https://huggingface.co"
YUNET_REV = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"
FER_REV = "b473270576eabddaacffd08adb4bed27b3fc54e6"
GROUPS = ("core", "vision", "llm", "llm_candidates")
DEFAULT_GROUPS = "core"
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
    group: str = "core"
    licence: str | None = None


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
    Asset("yunet", f"{HF}/opencv/face_detection_yunet/resolve/{YUNET_REV}/face_detection_yunet_2023mar.onnx",
          "vision/face_detection_yunet_2023mar.onnx", group="vision", licence="MIT (opencv/face_detection_yunet LICENSE)",
          sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    Asset("fer", f"{HF}/opencv/facial_expression_recognition/resolve/{FER_REV}/facial_expression_recognition_mobilefacenet_2022july.onnx",
          "vision/facial_expression_recognition_mobilefacenet_2022july.onnx", group="vision",
          licence="Apache-2.0 (opencv/facial_expression_recognition README)",
          sha256="4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9"),
]

# spec 4 candidates (hf onnx-community, pinned). the onnx-community cards name no
# licence; the base model cards Qwen/Qwen2.5-0.5B-Instruct (7ae557604adf67be50417f59c2c2f167def9a775)
# and Qwen/Qwen2.5-1.5B-Instruct (989aa7980e4cf806f80c7fef2b1adb7bc71aa306) say apache-2.0
LLM_SHA256 = {
    "Qwen2.5-0.5B-Instruct": {
        "onnx/model_int8.onnx": "41834041ab1b29eff9fc592f1a29a1844133aea35832ea9fa91682be13016100",
        "onnx/model_q4.onnx": "09235a3b1c135cd04ef570e5053b5c079e028078a2cc5f76ba6a251e91bf3296",
        "tokenizer.json": "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
        "config.json": "777e01f0fbb3346eb229cb6fb278ed6533c1e4dcb9ebf4bed0f6e94ef17fa1b5",
        "generation_config.json": "f7e7ce458658b2d40d9eb213b91b77a8bf698845ab89360976722d7ac46928a3",
    },
    "Qwen2.5-1.5B-Instruct": {
        "onnx/model_int8.onnx": "ee8364373c6fb35644c67fd8127cbee6c3d98ac889f8bb32ea4ac04a29650787",
        "onnx/model_q4.onnx": "70c24509f760fed9a8f50391165c46a87660197028a42c4547763f1c173ddea6",
        "tokenizer.json": "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
        "config.json": "215eb99c4955b0c42ea9f6e0980d922c228950c5dbc09bde6dc451fbba4d21f3",
        "generation_config.json": "f7e7ce458658b2d40d9eb213b91b77a8bf698845ab89360976722d7ac46928a3",
    },
}


def llm_assets() -> list[Asset]:
    """the candidate files; the deployed model's onnx file and side files are group llm."""
    out = []
    for model, files in LLM_SHA256.items():
        rev = G.CANDIDATES[model]
        for rel, sha in files.items():
            deployed = model == G.DEFAULT_MODEL and (rel in G.SIDE_FILES or rel == f"onnx/{G.DEFAULT_ONNX}")
            out.append(Asset(f"llm:{model}/{rel}", f"{HF}/onnx-community/{model}/resolve/{rev}/{rel}", f"llm/{model}/{rel}",
                             sha256=sha, group="llm" if deployed else "llm_candidates",
                             licence=f"Apache-2.0 (base model card Qwen/{model})"))
    return out


TABLE += llm_assets()


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
        if asset.group != "core":
            row["group"] = asset.group
        if asset.licence:
            row["licence"] = asset.licence
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


def merged_rows(checked: dict[str, list[dict]], old_path: Path) -> list[dict]:
    """rows in TABLE order: this run's rows for the assets it checked, the existing file's
    rows for the others (none when the file is missing or unreadable)."""
    old: dict[str, list[dict]] = {}
    try:
        for r in json.loads(old_path.read_text()).get("files", []):
            old.setdefault(r["asset"], []).append(r)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        old = {}
    out = []
    for asset in TABLE:
        out.extend(checked[asset.name] if asset.name in checked else old.get(asset.name, []))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=ROOT / "models")
    ap.add_argument("--dl-dir", type=Path, default=ROOT / "verify_dl", help="where the spacy wheel goes")
    ap.add_argument("--check", action="store_true", help="verify only, never download")
    ap.add_argument("--provenance", type=Path, default=ROOT / "results/provenance_hashes.json")
    ap.add_argument("--only", nargs="*", help="asset names to restrict to")
    ap.add_argument("--groups", default=DEFAULT_GROUPS, help=f"comma list of {', '.join(GROUPS)}, or all")
    args = ap.parse_args(argv)
    groups = set(GROUPS) if args.groups == "all" else set(args.groups.split(","))
    if groups - set(GROUPS):
        ap.error(f"unknown groups {sorted(groups - set(GROUPS))}")
    dest, dl_dir = args.dest.resolve(), args.dl_dir.resolve()

    all_ok = True
    checked: dict[str, list[dict]] = {}
    for asset in TABLE:
        if args.only and asset.name not in args.only:
            continue
        if not args.only and asset.group not in groups:
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
        checked[asset.name] = rows

    report = merged_rows(checked, args.provenance)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with open(args.provenance, "w") as f:
        json.dump({"generated_by": "tools/fetch_models.py", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "all_ok": all(r.get("ok") for r in report), "files": report}, f, indent=1)
    print(f"{'all verified' if all_ok else 'some files missing or mismatched'}; wrote {args.provenance}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
