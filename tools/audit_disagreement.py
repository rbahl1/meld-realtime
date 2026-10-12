"""the disagreement audit (design 1.9, 1.11): a person listens to MELD test clips that
the discrepancy rule flags and marks which branch was right.

sample: flags every test row that has a cached clip the way the service does (jsd
between the text head's posterior on the gold text and the raw sensevoice posterior,
threshold from heads/manifest.json or the service default), draws --n flagged rows
with a fixed seed and writes a csv with empty columns for the listener. the
audio_label column comes from the audio calibrator when heads/audio_cal.npz exists,
else from the raw sensevoice posterior. the csv holds MELD utterance text, so it
goes to cache/ (not committed).

tally: reads the filled csv and writes counts only to results/disagreement_audit.md.

usage: python tools/audit_disagreement.py sample [--n 50] [--threshold T] [--out cache/audit_disagreement.csv]
       python tools/audit_disagreement.py tally [--csv cache/audit_disagreement.csv] [--out results/disagreement_audit.md]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from meld_rt import heads, labels  # noqa: E402
from meld_rt.data import clip_path, key, load_split  # noqa: E402
from meld_rt.discrepancy import audio7, discrepancy  # noqa: E402
from meld_rt.service import ServiceConfig  # noqa: E402

BRANCHES = ("text", "audio", "both", "neither")
LAUGHTER = ("speaker", "audience", "none")
LISTENER_COLUMNS = ("right_branch", "laughter_source", "note")
LAUGHTER_IDX = labels.SV_EVENT_TOKENS.index("Laughter")


def flagged(p_text: np.ndarray, p_audio: np.ndarray, threshold: float) -> list[tuple[int, float, str]]:
    """(row, jsd, direction) of every row whose text and audio posteriors disagree past the threshold."""
    out = []
    for i in range(len(p_text)):
        d, flag, direction = discrepancy(p_text[i], p_audio[i], threshold)
        if flag:
            out.append((i, d, direction))
    return out


def draw(rows: list, n: int, seed: int) -> list:
    """n rows without replacement, in their original order; all of them when there are fewer."""
    if len(rows) <= n:
        return list(rows)
    pick = np.sort(np.random.default_rng(seed).choice(len(rows), size=n, replace=False))
    return [rows[i] for i in pick]


def tally(rows: list[dict]) -> dict:
    """counts of the listener's marks, in total and by direction. unmarked rows are counted apart."""
    marked = [r for r in rows if r.get("right_branch", "").strip() in BRANCHES]
    by_dir: dict[str, Counter] = {}
    for r in marked:
        by_dir.setdefault(r["direction"], Counter())[r["right_branch"].strip()] += 1
    laugh = Counter(r.get("laughter_source", "").strip() for r in marked if r.get("laughter_source", "").strip() in LAUGHTER)
    return {"rows": len(rows), "marked": len(marked), "right_branch": Counter(r["right_branch"].strip() for r in marked),
            "by_direction": by_dir, "laughter_source": laugh, "thresholds": sorted({r.get("threshold", "") for r in rows} - {""})}


def render_tally(t: dict, csv_name: str) -> str:
    L = ["# Disagreement audit", "",
         (f"Counted by `python tools/audit_disagreement.py tally` from `{csv_name}`. A person listened to each flagged clip "
          f"and marked which branch was right. {t['marked']} of {t['rows']} rows are marked. "
          f"The flag threshold was {', '.join(t['thresholds']) or 'not recorded'}."), "",
         "| direction | " + " | ".join(BRANCHES) + " |", "|---|" + "---|" * len(BRANCHES)]
    for d, c in sorted(t["by_direction"].items()):
        L.append(f"| {d} | " + " | ".join(str(c.get(b, 0)) for b in BRANCHES) + " |")
    L.append("| all | " + " | ".join(str(t["right_branch"].get(b, 0)) for b in BRANCHES) + " |")
    L += ["", "| laughter source | " + " | ".join(LAUGHTER) + " |", "|---|" + "---|" * len(LAUGHTER),
          "| marked rows | " + " | ".join(str(t["laughter_source"].get(s, 0)) for s in LAUGHTER) + " |", ""]
    return "\n".join(L)


def sample(a: argparse.Namespace) -> int:
    from train_heads import read_audio_cache, load_text_cache, text_matrix

    test = load_split("test", a.meld)
    ra = read_audio_cache(a.cache, "test", test)
    if ra is None:
        print(f"no {a.cache / 'audio_test.npz'}: run tools/cache_audio_features.py --split test first", file=sys.stderr)
        return 2
    aud, keep, _ = ra
    text_head = heads.TextContextHead.load(a.heads / "text_ctx.npz")
    emb = load_text_cache(a.cache, "test", test)
    p_text = np.exp(text_head.predict_log_proba_batch(text_matrix(emb, test, text_head.k)))[keep]
    # the flag uses the raw sensevoice posterior, as the service and the dev threshold do (design 1.6)
    p_audio_flag = np.stack([audio7(p) for p in aud["emo_probs"]])
    cal_path = a.heads / "audio_cal.npz"
    if cal_path.exists():
        cal = heads.AudioCalibrator.load(cal_path)
        from train_heads import audio_matrix
        p_audio = np.exp(cal.predict_log_proba_batch(audio_matrix(aud, cal.uses_encoder_states)))
        audio_source = "heads/audio_cal.npz"
    else:
        p_audio = p_audio_flag
        audio_source = "sensevoice posterior over its seven named classes"
    manifest = json.loads((a.heads / "manifest.json").read_text()) if (a.heads / "manifest.json").exists() else {}
    thr = (manifest.get("thresholds") or {}).get("jsd_flag")
    threshold = a.threshold if a.threshold is not None else (float(thr) if isinstance(thr, (int, float)) else ServiceConfig().jsd_flag)
    rows = draw(flagged(p_text, p_audio_flag, threshold), a.n, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "wav", "speaker", "utterance", "gold", "text_label", "audio_label", "jsd", "threshold", "direction",
                    "p_laughter", *LISTENER_COLUMNS])
        for i, d, direction in rows:
            u = test[keep[i]]
            w.writerow([key(u), str(clip_path(u, a.audio)), u.speaker, u.text, u.emotion,
                        labels.EMOTIONS[int(p_text[i].argmax())], labels.EMOTIONS[int(p_audio[i].argmax())],
                        f"{d:.4f}", f"{threshold:g}", direction, f"{float(aud['event_probs'][i][LAUGHTER_IDX]):.4f}", "", "", ""])
    print(f"{len(rows)} flagged rows (threshold {threshold} on the raw sensevoice posterior, audio label from {audio_source}) -> {a.out}. "
          f"fill right_branch ({' / '.join(BRANCHES)}) and laughter_source ({' / '.join(LAUGHTER)}), then run tally.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="sample flagged MELD test clips for a listening audit, or count the marks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threshold", type=float, default=None, help="jsd flag threshold (default: the manifest's, else the service default)")
    s.add_argument("--cache", type=Path, default=ROOT / "cache")
    s.add_argument("--heads", type=Path, default=ROOT / "heads")
    s.add_argument("--meld", type=Path, default=ROOT / "meld")
    s.add_argument("--audio", type=Path, default=ROOT / "audio")
    s.add_argument("--out", type=Path, default=ROOT / "cache" / "audit_disagreement.csv")
    t = sub.add_parser("tally")
    t.add_argument("--csv", type=Path, default=ROOT / "cache" / "audit_disagreement.csv")
    t.add_argument("--out", type=Path, default=ROOT / "results" / "disagreement_audit.md")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        return sample(a)
    with a.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    md = render_tally(tally(rows), a.csv.name)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
