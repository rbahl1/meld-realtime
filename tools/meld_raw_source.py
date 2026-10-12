#!/usr/bin/env python
"""record where MELD.Raw.tar.gz comes from (PROVENANCE.md, MELD.Raw): asks the
hugging face hub for the revision that main points to and the dataset card's
licence, and reads the resolve headers of the archive at main and at the pinned
revision (x-repo-commit, x-linked-size, x-linked-etag, which is the lfs sha256).
writes results/meld_raw_source.json. nothing is downloaded.

the build container fetched resolve/main and compared the byte count of the
download with x-linked-size before it unpacked anything; it deleted the archive
after extraction, so the sha256 of the download itself was not computed.

usage: python tools/meld_raw_source.py [--revision 9abc51ee...] [--out results/meld_raw_source.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "declare-lab/MELD"
FILE = "MELD.Raw.tar.gz"
PINNED = "9abc51ee7903424ffb971297608aa6d3d0de3bfa"
HUB = "https://huggingface.co"
HEADERS = ("x-repo-commit", "x-linked-size", "x-linked-etag")
# what the build container did with the archive (see the module docstring)
CONTAINER_DOWNLOAD = {
    "url": f"{HUB}/datasets/{REPO}/resolve/main/{FILE}",
    "check": "byte count of the download (stat -c %s) compared with x-linked-size before extraction, equal",
    "sha256_of_download": "not computed, the archive was deleted after extraction",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # keep the hub's own headers, do not follow to the cdn
        return None


def resolve_headers(url: str, timeout: float = 60) -> dict:
    """the hub headers of a resolve url, read from the response before any redirect."""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method="HEAD")
    try:
        with opener.open(req, timeout=timeout) as r:
            h, status = r.headers, r.status
    except urllib.error.HTTPError as e:  # the 302 to the cdn lands here
        h, status = e.headers, e.code
    out = {"url": url, "status": status}
    for k in HEADERS:
        v = h.get(k)
        out[k] = v.strip('"') if v else None
    if out["x-linked-size"] is not None:
        out["x-linked-size"] = int(out["x-linked-size"])
    return out


def revision_info(revision: str, timeout: float = 60) -> dict:
    with urllib.request.urlopen(f"{HUB}/api/datasets/{REPO}/revision/{revision}", timeout=timeout) as r:
        d = json.load(r)
    return {"sha": d.get("sha"), "last_modified": d.get("lastModified"),
            "card_license": (d.get("cardData") or {}).get("license"),
            "files": [s.get("rfilename") for s in d.get("siblings", [])]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="record the hub revision, size, lfs sha256 and licence of MELD.Raw.tar.gz")
    ap.add_argument("--revision", default=PINNED, help="the revision the documents name")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "meld_raw_source.json")
    a = ap.parse_args(argv)
    main_rev = revision_info("main")
    pinned_rev = revision_info(a.revision)
    at_main = resolve_headers(f"{HUB}/datasets/{REPO}/resolve/main/{FILE}")
    at_pin = resolve_headers(f"{HUB}/datasets/{REPO}/resolve/{a.revision}/{FILE}")
    same = (at_main["x-linked-etag"] is not None and at_main["x-linked-etag"] == at_pin["x-linked-etag"]
            and at_main["x-linked-size"] == at_pin["x-linked-size"])
    out = {"generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python tools/meld_raw_source.py" + ("" if not (argv if argv is not None else sys.argv[1:]) else
                                                            " " + " ".join(argv if argv is not None else sys.argv[1:])),
           "repo": REPO, "repo_type": "dataset", "file": FILE,
           "pinned_revision": a.revision, "main": main_rev, "pinned": pinned_rev,
           "resolve_main": at_main, "resolve_pinned": at_pin,
           "main_serves_pinned_revision": at_main["x-repo-commit"] == a.revision,
           "same_file_at_main_and_pinned": same,
           "container_download": CONTAINER_DOWNLOAD}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: main is {main_rev['sha']}, {FILE} {at_pin['x-linked-size']} bytes, "
          f"lfs sha256 {at_pin['x-linked-etag']}, licence {pinned_rev['card_license']}, same file at main: {same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
