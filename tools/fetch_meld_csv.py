"""fetch the three MELD annotation csvs from the dataset's github repo.

the csvs are not committed here: they carry the utterance text of a tv show,
so the repo only keeps statistics derived from them (results/data_stats.md).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/"
FILES = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]


def fetch(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = dest / name
        if out.exists() and out.stat().st_size > 0:
            print(f"present  {out} ({out.stat().st_size} bytes)")
            continue
        with urllib.request.urlopen(BASE + name, timeout=60) as r:
            data = r.read()
        out.write_bytes(data)
        print(f"fetched  {out} ({len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:12]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dest", default="meld", type=Path)
    a = p.parse_args(argv)
    return fetch(a.dest)


if __name__ == "__main__":
    sys.exit(main())
