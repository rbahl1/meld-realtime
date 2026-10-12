"""convert the MELD clips to 16 khz mono pcm16 wavs at audio/{split}/dia{d}_utt{u}.wav
(meld_rt.data.clip_path), one per csv row of meld_rt.data.load_split.

clips are found by file name, diaX_uttY.mp4 or diaX_uttY.wav, anywhere under --raw:
- the umich tarball once its inner train/dev/test tarballs are extracted:
  train/train_splits/, dev/dev_splits_complete/, test/output_repeated_splits_test/,
  also accepted without the train/ dev/ test/ level and under --raw/MELD.Raw/
- any other tree (ex. a hugging face mirror). the split comes from the nearest
  directory (or file name prefix) that says train, dev / val / validation or test.
  files with no split in their path are used only when one split is requested.
the same name exists in every split (dia0_utt0 is in train, dev and test), so a
file is never used for a split its path does not name.

conversion: ffmpeg -ac 1 -ar 16000 to pcm_s16le on the first audio stream, written
to a temporary name and renamed when ffmpeg succeeded and the wav reads back as
16 khz mono. existing outputs are skipped, so a rerun only does what is left.
--delete-mp4 removes a source mp4 once its wav exists.

audio/missing.json: per split, the csv rows with no clip or a failed conversion
(with the tail of ffmpeg's stderr), the total count, and how the clips were found.
a rerun for some splits keeps the entries of the others.

usage: python tools/extract_audio.py --raw /data/MELD.Raw --out audio/ [--split train,dev,test] [--delete-mp4] [--jobs 2]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt.data import SPLITS, Utt, clip_path, key, load_split  # noqa: E402

SR = 16000
# umich layout, relative to --raw or --raw/MELD.Raw
KNOWN_DIRS = {
    "train": ("train/train_splits", "train_splits"),
    "dev": ("dev/dev_splits_complete", "dev_splits_complete"),
    "test": ("test/output_repeated_splits_test", "output_repeated_splits_test"),
}
SPLIT_WORDS = {"train": {"train", "training"}, "dev": {"dev", "val", "valid", "validation"}, "test": {"test", "testing"}}
# an optional prefix is allowed (ex. final_videos_testdia0_utt0.mp4), a plain name wins over it
CLIP_RE = re.compile(r"(?P<prefix>[A-Za-z0-9_-]*?)dia(?P<d>\d+)_utt(?P<u>\d+)\.(?P<ext>mp4|wav)", re.IGNORECASE)
TIMEOUT_S = 300.0
TAIL_LINES = 12
TAIL_CHARS = 1500


def split_of(parts: list[str]) -> str | None:
    """the split named by the nearest path part that names exactly one split."""
    for part in reversed(parts):
        words = set(re.split(r"[^a-z0-9]+", part.lower()))
        hits = [s for s, w in SPLIT_WORDS.items() if words & w]
        if len(hits) == 1:
            return hits[0]
    return None


def index_tree(raw: Path, exclude: Path | None = None) -> tuple[dict, dict]:
    """(candidates, tarballs). candidates[(split or None, d, u)] is a list of
    (rank, path, how), lower rank first. tarballs[split] lists unextracted
    {split}.tar.gz files, for the hint when a split has no clips. the exclude
    folder (the output) is not searched."""
    raw = raw.resolve()
    skip = os.path.realpath(exclude) if exclude is not None else None
    known = {s: {(raw / b / d).resolve() for b in (".", "MELD.Raw") for d in dirs} for s, dirs in KNOWN_DIRS.items()}
    cands: dict = {}
    tarballs: dict[str, list[str]] = {}
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(raw, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen or real == skip:  # symlink loop, or the output folder
            dirnames[:] = []
            continue
        seen.add(real)
        d = Path(dirpath)
        rel_dirs = list(d.relative_to(raw).parts)
        dir_split = split_of(rel_dirs)
        known_split = next((s for s, dirs in known.items() if Path(real) in dirs), None)
        for name in filenames:
            low = name.lower()
            for s in SPLITS:
                if low in (f"{s}.tar.gz", f"{s}.tgz", f"{s}.tar"):
                    tarballs.setdefault(s, []).append(str(d / name))
            if name.startswith("._"):  # macos resource forks shipped inside the tarball
                continue
            m = CLIP_RE.fullmatch(name)
            if not m:
                continue
            prefix, ext = m["prefix"], m["ext"].lower()
            split = split_of(rel_dirs + [prefix]) if prefix else dir_split
            in_known = split is not None and split == known_split
            how = "layout" if in_known else ("search" if split else "unsplit")
            rank = (0 if in_known else 1, 1 if prefix else 0, 0 if ext == "mp4" else 1, len(rel_dirs), str(d / name))
            cands.setdefault((split, int(m["d"]), int(m["u"])), []).append((rank, d / name, how))
    for v in cands.values():
        v.sort(key=lambda t: t[0])
    return cands, tarballs


def pick(cands: dict, split: str, u: Utt, allow_unsplit: bool) -> tuple[Path, str, int] | None:
    """(source, how it was found, number of candidates) for one csv row."""
    c = cands.get((split, u.dialogue_id, u.utterance_id))
    if not c and allow_unsplit:
        c = cands.get((None, u.dialogue_id, u.utterance_id))
    if not c:
        return None
    return c[0][1], c[0][2], len(c)


def tail(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(lines[-TAIL_LINES:])[-TAIL_CHARS:]


def wav_ok(path: Path) -> str | None:
    """None when the file reads as 16 khz mono with at least one sample, else why not."""
    try:
        info = sf.info(str(path))
    except RuntimeError as e:  # soundfile's LibsndfileError is a RuntimeError
        return f"unreadable output: {e}"
    if info.samplerate != SR or info.channels != 1:
        return f"output is {info.samplerate} hz, {info.channels} channels"
    if info.frames == 0:
        return "output has no samples"
    return None


def convert(ffmpeg: str, src: Path, dst: Path, timeout: float = TIMEOUT_S) -> str | None:
    """ffmpeg src -> dst (16 khz mono pcm16). None on success, else the error tail."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
           "-map_metadata", "-1", "-fflags", "+bitexact", "-f", "wav", str(tmp)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return f"ffmpeg timed out after {timeout:.0f} s"
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        return tail(p.stderr) or f"ffmpeg exited with {p.returncode}"
    bad = wav_ok(tmp)
    if bad:
        tmp.unlink(missing_ok=True)
        return bad + ("\n" + tail(p.stderr) if p.stderr.strip() else "")
    os.replace(tmp, dst)
    return None


def delete_source(src: Path) -> bool:
    if src.suffix.lower() != ".mp4":
        return False
    src.unlink(missing_ok=True)
    return True


def extract_split(split: str, utts: list[Utt], cands: dict, tarballs: dict, out: Path, ffmpeg: str, jobs: int,
                  delete_mp4: bool, allow_unsplit: bool) -> dict:
    t0 = time.perf_counter()
    found_by = {"layout": 0, "search": 0, "unsplit": 0}
    no_clip, todo, failed = [], [], {}
    existing = deleted = dup = 0
    for u in utts:
        dst = clip_path(u, out)
        hit = pick(cands, split, u, allow_unsplit)
        if dst.exists():
            existing += 1
            if delete_mp4 and hit and wav_ok(dst) is None:
                deleted += delete_source(hit[0])
            continue
        if hit is None:
            no_clip.append(key(u))
            continue
        src, how, n = hit
        found_by[how] += 1
        dup += n > 1
        todo.append((u, src, dst))

    converted = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(convert, ffmpeg, src, dst): (u, src) for u, src, dst in todo}
        for i, f in enumerate(as_completed(futs), 1):
            u, src = futs[f]
            err = f.result()
            if err is None:
                converted += 1
                if delete_mp4:
                    deleted += delete_source(src)
            else:
                failed[key(u)] = {"src": str(src), "error": err}
            if i % 500 == 0:
                print(f"{split}: {i}/{len(todo)} converted, {time.perf_counter() - t0:.0f} s", flush=True)

    keys = sorted(no_clip + list(failed), key=lambda k: tuple(int(x) for x in k.split("_")))
    rep = {"rows": len(utts), "existing": existing, "converted": converted, "count": len(keys), "keys": keys,
           "no_clip": no_clip, "failed": dict(sorted(failed.items())), "found_by": found_by,
           "rows_with_several_candidates": dup, "deleted_mp4": deleted, "wall_s": round(time.perf_counter() - t0, 1)}
    if existing + converted == 0 and tarballs.get(split):
        rep["hint"] = f"no {split} clips found, extract {tarballs[split][0]} first (tar -xzf FILE -C its folder)"
    return rep


def ffmpeg_version(ffmpeg: str) -> str:
    try:
        return subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=30, check=False).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, required=True, help="folder holding the extracted clips (any depth)")
    ap.add_argument("--out", type=Path, default=ROOT / "audio")
    ap.add_argument("--split", default="train,dev,test", help="comma separated subset of train,dev,test")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld", help="folder with the MELD csv files")
    ap.add_argument("--delete-mp4", action="store_true", help="remove each source mp4 once its wav exists")
    ap.add_argument("--jobs", type=int, default=2, help="ffmpeg processes in parallel")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    a = ap.parse_args(argv)
    splits = [s.strip() for s in a.split.split(",") if s.strip()]
    bad = [s for s in splits if s not in SPLITS]
    if bad or not splits:
        ap.error(f"--split takes a subset of {','.join(SPLITS)}, got {a.split!r}")
    ffmpeg = shutil.which(a.ffmpeg)
    if ffmpeg is None:
        print(f"ffmpeg not found ({a.ffmpeg}), install it or pass --ffmpeg PATH", file=sys.stderr)
        return 2
    if not a.raw.is_dir():
        print(f"--raw {a.raw} is not a directory", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    cands, tarballs = index_tree(a.raw, exclude=a.out)
    n_files = sum(len(v) for v in cands.values())
    print(f"indexed {n_files} clip files under {a.raw} in {time.perf_counter() - t0:.1f} s", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    report_path = a.out / "missing.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report = {}
    per_split = report.get("splits", {}) if isinstance(report.get("splits"), dict) else {}
    empty = []
    for split in splits:
        utts = load_split(split, a.meld)
        rep = extract_split(split, utts, cands, tarballs, a.out, ffmpeg, a.jobs, a.delete_mp4, allow_unsplit=len(splits) == 1)
        per_split[split] = rep
        print(f"{split}: {rep['rows']} rows, {rep['converted']} converted, {rep['existing']} already there, "
              f"{len(rep['no_clip'])} without a clip, {len(rep['failed'])} failed, {rep['deleted_mp4']} mp4 deleted "
              f"({rep['wall_s']} s)", flush=True)
        if "hint" in rep:
            print(f"{split}: {rep['hint']}", file=sys.stderr)
        if rep["existing"] + rep["converted"] == 0:
            empty.append(split)

    ordered = {s: per_split[s] for s in SPLITS if s in per_split}
    out = {"count": sum(r["count"] for r in ordered.values()), "keys": {s: r["keys"] for s, r in ordered.items()},
           "splits": ordered, "raw": str(a.raw), "ffmpeg": ffmpeg_version(ffmpeg),
           "command": "python tools/extract_audio.py " + " ".join(sys.argv[1:] if argv is None else argv)}
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, report_path)
    print(f"{out['count']} rows without audio -> {report_path}")
    if empty:
        print(f"no clips at all for {','.join(empty)}: check --raw", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
