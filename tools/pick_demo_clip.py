#!/usr/bin/env python
"""the clip demo_tav.sh streams, by the rule written before the choice: a MELD
test clip whose chosen face is present in at least 80 % of the frames, whose
SenseVoice transcript passes the asr gate, on which the text head (on that
transcript) is wrong and the three-way fusion (on the same transcript) is
right; the lowest dialogue id, then the lowest utterance id.

reads cache/vision_post_test.npz (tools/train_vision.py) and the test csv,
writes results/demo_clip.json with the chosen key, the count of clips that
meet the rule and the first ten of them in order.

usage: python tools/pick_demo_clip.py [--cache cache/] [--out results/demo_clip.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt import labels  # noqa: E402
from meld_rt.data import key, load_split  # noqa: E402

MIN_PRESENCE = 0.8


def pick(post: dict, utts: dict) -> tuple[list[str], dict]:
    keys = [str(k) for k in post["keys"]]
    gold = np.array([labels.EMOTIONS.index(utts[k].emotion) for k in keys])
    text = np.asarray(post["p_text_live"]).argmax(axis=1)
    fused3 = np.asarray(post["fused3_live"]).argmax(axis=1)
    degraded = np.asarray(post["asr_degraded"]).astype(bool)
    ok = (np.asarray(post["presence"]) >= MIN_PRESENCE) & ~degraded & (text != gold) & (fused3 == gold)
    order = sorted(np.flatnonzero(ok), key=lambda i: (utts[keys[i]].dialogue_id, utts[keys[i]].utterance_id))
    rows = [{"key": keys[i], "gold": labels.EMOTIONS[gold[i]], "text_live": labels.EMOTIONS[text[i]],
             "fused3_live": labels.EMOTIONS[fused3[i]], "presence": round(float(post["presence"][i]), 4),
             "mp4": f"dia{utts[keys[i]].dialogue_id}_utt{utts[keys[i]].utterance_id}.mp4"} for i in order]
    return [r["key"] for r in rows], {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "demo_clip.json")
    a = ap.parse_args(argv)
    with np.load(a.cache / "vision_post_test.npz", allow_pickle=False) as z:
        post = {k: z[k] for k in z.files}
    utts = {key(u): u for u in load_split("test")}
    chosen, info = pick(post, utts)
    out = {
        "rule": ("test clip; chosen face present in at least 80 % of the frames; SenseVoice transcript passes the asr gate; "
                 "text head on that transcript wrong; three-way fusion on that transcript right; "
                 "lowest dialogue id, then lowest utterance id"),
        "min_presence": MIN_PRESENCE,
        "n_clips_meeting_rule": len(chosen),
        "n_test_clips": int(len(post["keys"])),
        "chosen": info["rows"][0] if chosen else None,
        "first_ten": info["rows"][:10],
        "command": "python tools/pick_demo_clip.py",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(chosen)} clips meet the rule, chosen {out['chosen']['key'] if chosen else None} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
