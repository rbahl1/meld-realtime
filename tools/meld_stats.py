"""split statistics of the MELD csvs as markdown tables.

python tools/meld_stats.py --meld meld/ --out results/data_stats.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from meld_rt import labels
from meld_rt.data import SPLITS, Utt, context_rows, csv_path, dialogues, load_split

LONG_S = (5.0, 8.0, 15.0)


def split_stats(utts: list[Utt]) -> dict:
    """every number of one table row set; plain python floats so it can be dumped."""
    n = len(utts)
    dur = np.array([u.duration_s for u in utts])
    words = np.array([len(u.text.split()) for u in utts])
    pairs = [(c[-1], u) for u, c in context_rows(utts, 1, False) if c]
    emo = {e: sum(u.emotion == e for u in utts) for e in labels.EMOTIONS}
    sent = {s: sum(u.sentiment == s for u in utts) for s in labels.SENTIMENTS}
    return {
        "utterances": n,
        "dialogues": len(dialogues(utts)),
        "speakers": len({u.speaker for u in utts}),
        "pairs": len(pairs),
        "speaker_change_pct": 100.0 * np.mean([a.speaker != b.speaker for a, b in pairs]) if pairs else float("nan"),
        "emotion_change_pct": 100.0 * np.mean([a.emotion != b.emotion for a, b in pairs]) if pairs else float("nan"),
        "emotion": emo,
        "sentiment": sent,
        "dur_median": float(np.median(dur)),
        "dur_mean": float(dur.mean()),
        "dur_p90": float(np.percentile(dur, 90)),
        "dur_p95": float(np.percentile(dur, 95)),
        "dur_max": float(dur.max()),
        "dur_le_pct": {t: float(100.0 * (dur <= t).mean()) for t in LONG_S},
        "over_15_s": int((dur > 15.0).sum()),
        "non_positive": int((dur <= 0).sum()),
        "words_mean": float(words.mean()),
        "words_median": float(np.median(words)),
        "words_p95": float(np.percentile(words, 95)),
        "words_max": int(words.max()),
        "anomalies": sorted(
            [u for u in utts if u.duration_s <= 0 or u.duration_s > 60.0],
            key=lambda u: (u.dialogue_id, u.utterance_id),
        ),
    }


def _row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list]) -> str:
    return "\n".join([_row(header), _row(["---"] * len(header))] + [_row(r) for r in rows])


def _pct(num: int, den: int) -> str:
    return f"{num} ({100.0 * num / den:.1f} %)"


def render(stats: dict[str, dict], inputs: list[tuple[str, int, str]], cmd: str) -> str:
    names = list(stats)
    out = ["# MELD CSV statistics", ""]
    out.append(f"Computed from the CSV files below by `{cmd}`. No audio was used.")
    out.append("")
    out.append(_table(["file", "bytes", "sha256"], [[f"`{n}`", b, f"`{h}`"] for n, b, h in inputs]))
    out.append("")
    out.append("Definitions:")
    out.append("")
    out.append("- Duration is EndTime minus StartTime of the subtitle timestamps.")
    out.append("- Percentiles use numpy linear interpolation.")
    out.append("- Words are the whitespace-separated tokens of the Utterance column.")
    out.append("- Consecutive pairs are adjacent rows of the same dialogue in Utterance_ID order. Utterance ids have "
               "gaps in some dialogues, so a pair need not have consecutive ids. The speaker and emotion change rates "
               "are over these pairs.")
    out.append("- The `all` rows pool the three splits.")
    out.append("")
    out.append("## Counts")
    out.append("")
    out.append(_table(
        ["split", "utterances", "dialogues", "speakers", "consecutive pairs", "speaker change %", "emotion change %"],
        [[s, st["utterances"], st["dialogues"], st["speakers"], st["pairs"],
          f"{st['speaker_change_pct']:.1f}", f"{st['emotion_change_pct']:.1f}"] for s, st in stats.items()],
    ))
    out.append("")
    out.append("## Emotion classes (count, share)")
    out.append("")
    out.append(_table(
        ["split"] + labels.EMOTIONS,
        [[s] + [_pct(st["emotion"][e], st["utterances"]) for e in labels.EMOTIONS] for s, st in stats.items()],
    ))
    out.append("")
    out.append("## Sentiment classes (count, share)")
    out.append("")
    out.append(_table(
        ["split"] + labels.SENTIMENTS,
        [[s] + [_pct(st["sentiment"][c], st["utterances"]) for c in labels.SENTIMENTS] for s, st in stats.items()],
    ))
    out.append("")
    out.append("## Subtitle durations (s)")
    out.append("")
    out.append(_table(
        ["split", "median", "mean", "p90", "p95", "max"] + [f"<= {t:g} s %" for t in LONG_S] + ["> 15 s clips", "non-positive"],
        [[s, f"{st['dur_median']:.2f}", f"{st['dur_mean']:.2f}", f"{st['dur_p90']:.2f}", f"{st['dur_p95']:.2f}",
          f"{st['dur_max']:.2f}"] + [f"{st['dur_le_pct'][t]:.1f}" for t in LONG_S] + [st["over_15_s"], st["non_positive"]]
         for s, st in stats.items()],
    ))
    out.append("")
    out.append("## Words per utterance")
    out.append("")
    out.append(_table(
        ["split", "mean", "median", "p95", "max"],
        [[s, f"{st['words_mean']:.2f}", f"{st['words_median']:.0f}", f"{st['words_p95']:.0f}", st["words_max"]]
         for s, st in stats.items()],
    ))
    out.append("")
    out.append("## Timestamp anomalies (duration <= 0 or > 60 s)")
    out.append("")
    rows = []
    for s in names:
        if s == "all":
            continue
        for u in stats[s]["anomalies"]:
            rows.append([s, u.dialogue_id, u.utterance_id, f"{u.start_s:.3f}", f"{u.end_s:.3f}", f"{u.duration_s:.3f}",
                         len(u.text.split())])
    out.append(_table(["split", "dialogue_id", "utterance_id", "start s", "end s", "duration s", "words"], rows)
               if rows else "none")
    out.append("")
    out.append(f"The MELD README gives these emotion class weights, in the index order {', '.join(labels.EMOTIONS)}: "
               f"`{labels.MELD_README_WEIGHTS}`.")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meld", type=Path, default=Path("meld"))
    ap.add_argument("--out", type=Path, default=Path("results/data_stats.md"))
    args = ap.parse_args(argv)

    inputs = []
    stats: dict[str, dict] = {}
    pooled: list[Utt] = []
    for split in SPLITS:
        p = csv_path(split, args.meld)
        data = p.read_bytes()
        inputs.append((p.name, len(data), hashlib.sha256(data).hexdigest()))
        utts = load_split(split, args.meld)
        stats[split] = split_stats(utts)
        pooled.extend(utts)
    stats["all"] = split_stats(pooled)

    cmd = "python tools/meld_stats.py " + " ".join(sys.argv[1:] if argv is None else argv)
    text = render(stats, inputs, cmd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
