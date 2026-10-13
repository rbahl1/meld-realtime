"""evaluate the three-input extension on MELD test (extension spec 2.4) with the heads
of tools/train_vision.py. same metrics and bootstrap as tools/evaluate.py (its scored()):
weighted f1 (primary), macro f1, accuracy, per-class f1, confusion, 95 % bootstrap
interval over test dialogues (1000 resamples, seed 0), ece. every choice was made on
dev by train_vision.py; test is read once per condition.

conditions: vision_raw_fer (argmax of the averaged fer posterior of the chosen track),
vision_cal (deployed calibrator), both with the face coverage (no face: uniform
posterior, label neutral); fused3_stacked (gold text) and live_asr_fused3 (sensevoice
transcripts with the service's asr gate, as live_asr_text in evaluate.py); the same on
the clips with a face; the face-disagreement subset (clips with a face whose jsd between
the text posterior and the raw face posterior is above vision.jsd_face_flag); sentiment3.
the deployed two-way rows (fused_stacked, live_asr_text) and the text head are recomputed
as references. paired bootstrap over test dialogues of the weighted-f1 difference fused3
minus the two-way fusion (paired_bootstrap_diff), gold and live, all clips and face clips.

claim rule (fixed before the run): the face adds to text and audio only if the paired
95 % interval of fused3 minus the two-way fusion on gold-text test lies above zero.

usage: python tools/evaluate_vision.py --cache cache/ --heads heads/ --out results/RESULTS_vision.md results/metrics_vision.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate import N_RESAMPLES, SEED, brief, pending, scored  # noqa: E402
from train_heads import file_record, groups_of  # noqa: E402
from train_vision import deployed_posteriors, envelopes, load_video  # noqa: E402

from meld_rt import labels, metrics  # noqa: E402
from meld_rt.data import emotion_ids, load_split, sentiment_ids  # noqa: E402

CLAIM_RULE = ("the face adds to text and audio only if the paired 95 % interval of fused3_stacked minus fused_stacked "
              "(weighted f1, gold-text test, all clips with audio) lies above zero")

CONDITIONS = [
    ("vision_raw_fer", "emotion", "argmax of the averaged fer posterior of the chosen track; no face: uniform, label neutral; all test rows"),
    ("vision_cal", "emotion", "deployed vision calibrator (heads/vision_cal.npz); no face: uniform, label neutral; all test rows"),
    ("text_best_k", "emotion", "reference: deployed text head, gold text, rows with audio"),
    ("fused_stacked", "emotion", "reference: deployed two-way fusion (heads/fusion.npz), gold text"),
    ("fused3_stacked", "emotion", "fusion3 (heads/fusion3.npz) over gold-text, audio and face posteriors"),
    ("live_asr_text", "emotion", "reference: two-way fusion on the sensevoice transcripts with the service's asr gate"),
    ("live_asr_fused3", "emotion", "fusion3 on the sensevoice transcripts with the service's asr gate"),
    ("text_best_k_face_rows", "emotion", "reference: text head, gold text, clips with a face"),
    ("fused_stacked_face_rows", "emotion", "reference: two-way fusion, gold text, clips with a face"),
    ("fused3_stacked_face_rows", "emotion", "fusion3, gold text, clips with a face"),
    ("live_asr_text_face_rows", "emotion", "reference: two-way fusion, transcripts with the asr gate, clips with a face"),
    ("live_asr_fused3_face_rows", "emotion", "fusion3, transcripts with the asr gate, clips with a face"),
    ("face_disagreement_subset", "emotion", "fusion3 on the clips with a face whose words-face jsd is above vision.jsd_face_flag"),
    ("sentiment3", "sentiment", "sentiment over the 29 fusion3 features (heads/sentiment3.npz), gold text"),
]
NEW_ROWS = ("vision_raw_fer", "vision_cal", "fused3_stacked", "live_asr_fused3", "fused3_stacked_face_rows",
            "live_asr_fused3_face_rows", "face_disagreement_subset")
COMPARISONS = [  # name, a, b, rows
    ("gold_all", "fused3_stacked", "fused_stacked", "all"),
    ("live_all", "live_asr_fused3", "live_asr_text", "all"),
    ("gold_face_rows", "fused3_stacked", "fused_stacked", "face"),
    ("live_face_rows", "live_asr_fused3", "live_asr_text", "face"),
    ("gold_face_disagreement", "fused3_stacked", "fused_stacked", "disagree"),
    ("gold_face_disagreement_vs_text", "fused3_stacked", "text_best_k", "disagree"),
]


def paired_bootstrap_diff(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, groups: np.ndarray, n_classes: int,
                          n: int = 1000, seed: int = 0) -> dict:
    """metric(a) - metric(b) for weighted f1, macro f1 and accuracy when whole groups (dialogues) are
    resampled with replacement, both predictions on the same resample (the resamples of
    metrics.bootstrap_by_group with the same seed). returns the point difference, the 95 %
    percentile interval, whether it excludes zero and the share of resamples at or below zero."""
    y, pred_a, pred_b = (np.asarray(v, dtype=np.int64) for v in (y, pred_a, pred_b))
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = len(uniq)
    ca = np.zeros((G, n_classes, n_classes), dtype=np.int64)
    cb = np.zeros_like(ca)
    np.add.at(ca, (inv, y, pred_a), 1)
    np.add.at(cb, (inv, y, pred_b), 1)
    fns = {"weighted_f1": metrics.weighted_f1, "macro_f1": metrics.macro_f1, "accuracy": metrics.accuracy}
    point = {m: fn(ca.sum(axis=0)) - fn(cb.sum(axis=0)) for m, fn in fns.items()}
    rng = np.random.default_rng(seed)
    d = {m: np.zeros(n) for m in fns}
    for r in range(n):
        pick = rng.integers(0, G, size=G)
        A, B = ca[pick].sum(axis=0), cb[pick].sum(axis=0)
        for m, fn in fns.items():
            d[m][r] = fn(A) - fn(B)
    ci = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in d.items()}
    return {"n": len(y), "n_groups": int(G), "n_resamples": int(n), "seed": int(seed),
            "diff": {m: round(float(v), 6) for m, v in point.items()},
            "ci95": {m: [round(lo, 6), round(hi, 6)] for m, (lo, hi) in ci.items()},
            "excludes_zero": {m: bool(lo > 0 or hi < 0) for m, (lo, hi) in ci.items()},
            "share_at_or_below_zero": {m: round(float((v <= 0).mean()), 6) for m, v in d.items()}}


def run(cache: Path, heads_dir: Path, meld: Path, audio_dir: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((heads_dir / "manifest.json").read_text())
    vis = manifest.get("vision")
    if not vis or vis.get("status") != "trained":
        raise SystemExit(f"{heads_dir / 'manifest.json'} has no trained vision section (run tools/train_vision.py)")
    test = load_split("test", meld)
    y_all, ys_all, g_all = emotion_ids(test), sentiment_ids(test), groups_of(test)
    video = load_video(cache, "test")
    env = envelopes(test, audio_dir, cache, "test")
    P = deployed_posteriors("test", test, cache, heads_dir, manifest, video, env, live=True)
    pres_all = P["present"]
    cov = round(float(pres_all.mean()), 6)
    R: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    def face_part(y, p, m) -> dict:
        return {"n_face": int(m.sum()), "face_rows": brief(y[m], p[m]) if m.any() else None,
                "no_face_rows": brief(y[~m], p[~m]) if (~m).any() else None}

    for name in ("vision_raw_fer", "vision_cal"):
        p = P["p_face_raw"] if name == "vision_raw_fer" else P["p_face"]
        note = f"rule {P['rule']}, " + ("raw fer, no calibration" if name == "vision_raw_fer" else f"calibrator {P['variant']}")
        R[name] = scored(name, "emotion", y_all, p, g_all, note + f", face coverage {100 * cov:.2f} %", face_coverage=cov,
                         **face_part(y_all, p, pres_all))
    keep = P["keep"]
    if not len(keep):
        raise SystemExit("no test row has a cached audio clip (cache/audio_test.npz)")
    y, ys, g, pres = y_all[keep], ys_all[keep], g_all[keep], pres_all[keep]
    cov_k = round(float(pres.mean()), 6)
    rows = {"all": np.ones(len(keep), bool), "face": pres}
    thr = float(vis["jsd_face_flag"])
    rows["disagree"] = pres & (P["jsd_face"] > thr)
    probs = {"text_best_k": P["p_text"], "fused_stacked": P["fused2"], "fused3_stacked": P["fused3"]}
    if "fused3_live" in P:
        probs.update(live_asr_text=P["fused2_live"], live_asr_fused3=P["fused3_live"])
    for name, p in probs.items():
        preds[name] = p.argmax(axis=1)
        extra = {"face_coverage": cov_k} if name in ("fused3_stacked", "live_asr_fused3") else {}
        if name == "live_asr_fused3":
            extra.update(without_gate=brief(y, P["fused3_live_without_gate"]), **P["gate"])
        R[name] = scored(name, "emotion", y, p, g, f"{len(keep)} test rows with audio", **extra, **face_part(y, p, pres))
        fr = rows["face"]
        if fr.any():
            R[f"{name}_face_rows"] = scored(f"{name}_face_rows", "emotion", y[fr], p[fr], g[fr], "clips with a face")
    m = rows["disagree"]
    if m.any() and (~m & pres).any():
        def part(sel: np.ndarray) -> dict:
            return {"n": int(sel.sum()), **{b: brief(y[sel], probs[b][sel]) for b in ("text_best_k", "fused_stacked", "fused3_stacked")}}

        R["face_disagreement_subset"] = scored(
            "face_disagreement_subset", "emotion", y[m], P["fused3"][m], g[m],
            f"{int(m.sum())} of {int(pres.sum())} clips with a face have jsd above {thr} (vision.jsd_face_flag, dev)",
            jsd_face_flag=thr, n_face_rows=int(pres.sum()), n_subset=int(m.sum()), share_of_face_rows=round(float(m.sum() / pres.sum()), 6),
            subset=part(m), complement=part(pres & ~m))
    if "sent3" in P:
        R["sentiment3"] = scored("sentiment3", "sentiment", ys, P["sent3"], g, "29 fusion3 features, gold text")
    out = {}
    for name, task, desc in CONDITIONS:
        r = R.get(name) or pending(name, task, "pending (inputs missing: live rows need cache/text_asr_test.npy, the subset needs flagged and unflagged face clips)")
        r["description"] = desc
        out[name] = r
    C = {}
    for name, a, b, rws in COMPARISONS:
        sel = rows[rws]
        if a not in preds or b not in preds or not sel.any():
            C[name] = {"status": "pending", "a": a, "b": b, "rows": rws}
            continue
        C[name] = {"a": a, "b": b, "rows": rws, **paired_bootstrap_diff(y[sel], preds[a][sel], preds[b][sel], g[sel],
                                                                       len(labels.EMOTIONS), N_RESAMPLES, SEED)}
    lo = C["gold_all"]["ci95"]["weighted_f1"][0] if "ci95" in C["gold_all"] else None
    claim = {"rule": CLAIM_RULE, "gold_all_ci95_weighted_f1": C["gold_all"].get("ci95", {}).get("weighted_f1"),
             "face_adds": bool(lo is not None and lo > 0)}
    inputs = {}
    for p in [cache / "video_test.npz", cache / "envelope_test.npz", cache / "text_test.npy", cache / "audio_test.npz",
              cache / "text_asr_test.npy", meld / "test_sent_emo.csv"]:
        if p.exists():
            inputs[f"{p.parent.name}/{p.name}"] = file_record(p)
    for f in ("text_ctx.npz", "audio_cal.npz", "fusion.npz", "vision_cal.npz", "fusion3.npz", "sentiment3.npz"):
        if (heads_dir / f).exists():
            inputs[f"heads/{f}"] = file_record(heads_dir / f)
    meta = {"inputs": inputs, "claim": claim, "comparisons": C,
            "vision": {k: vis.get(k) for k in ("rule", "variant", "jsd_face_flag", "thresholds", "face_coverage", "rule_disagreement")},
            "dev": {"rules": {r: {"dev": v["dev"], "dev_face_rows": v["dev_face_rows"], "fallback_rate": v.get("fallback_rate")}
                              for r, v in (vis.get("rules") or {}).items()},
                    "calibrator": {v: (vis.get("calibrator") or {}).get(v, {}).get("dev") for v in ("A", "B")},
                    "fusion3": (vis.get("fusion3") or {}).get("dev"),
                    "two_way": (vis.get("fusion3") or {}).get("two_way_dev_recomputed"),
                    "sentiment3": (vis.get("sentiment3") or {}).get("dev")},
            "test_face_coverage": {"all_rows": cov, "rows_with_audio": cov_k}, "n_test_rows": len(test), "n_rows_with_audio": len(keep)}
    return out, C, meta


# ---------------------------------------------------------------- rendering (tables only)

def pct(v) -> str:
    return "" if v is None else f"{100 * v:.2f}"


def interval(lo_hi) -> str:
    return "" if not lo_hi else f"[{100 * lo_hi[0]:.2f}, {100 * lo_hi[1]:.2f}]"


def render(results: dict, meta: dict, cmd: str) -> str:
    L = ["# Three inputs on MELD test", "", f"`{cmd}`", "",
         "## Inputs", "", "| file | bytes | sha256 |", "| --- | --- | --- |"]
    L += [f"| `{k}` | {v['bytes']} | `{v['sha256']}` |" for k, v in meta["inputs"].items()]
    v = meta["vision"]
    th = v.get("thresholds") or {}
    L += ["", "| setting | value |", "| --- | --- |", f"| speaker rule | {v.get('rule')} |", f"| calibrator | {v.get('variant')} |",
          f"| jsd_face_flag | {v.get('jsd_face_flag')} |",
          f"| vision-mode thresholds | low_conf {th.get('low_conf')}, high_conf {th.get('high_conf')} ({th.get('source')}) |",
          f"| face coverage, test rows | {pct(meta['test_face_coverage']['all_rows'])} |", ""]
    for task, title in (("emotion", "Emotion (7 classes)"), ("sentiment", "Sentiment (3 classes)")):
        L += [f"## {title}", "", "| condition | n | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | face coverage |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in results.items():
            if r["task"] != task:
                continue
            if r["status"] != "done":
                L.append(f"| {name} | | {r['status']} | | | | | |")
                continue
            L.append(f"| {name} | {r['n']} | {pct(r['weighted_f1'])} | {interval(r['ci95']['weighted_f1'])} | {pct(r['macro_f1'])} | "
                     f"{pct(r['accuracy'])} | {pct(r['ece'])} | {pct(r.get('face_coverage'))} |")
        L.append("")
    L += ["## Paired differences (weighted F1, a minus b)", "",
          f"Paired bootstrap over test dialogues, {N_RESAMPLES} resamples, seed {SEED}.", "",
          "| comparison | a | b | rows | n | difference | 95 % CI | excludes zero | share at or below zero |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, c in meta["comparisons"].items():
        if "ci95" not in c:
            L.append(f"| {name} | {c['a']} | {c['b']} | {c['rows']} | | pending | | | |")
            continue
        L.append(f"| {name} | {c['a']} | {c['b']} | {c['rows']} | {c['n']} | {pct(c['diff']['weighted_f1'])} | "
                 f"{interval(c['ci95']['weighted_f1'])} | {'yes' if c['excludes_zero']['weighted_f1'] else 'no'} | "
                 f"{pct(c['share_at_or_below_zero']['weighted_f1'])} |")
    cl = meta["claim"]
    L += ["", "| claim rule | interval | face adds |", "| --- | --- | --- |",
          f"| {cl['rule']} | {interval(cl['gold_all_ci95_weighted_f1'])} | {'yes' if cl['face_adds'] else 'no'} |", ""]
    r = results.get("face_disagreement_subset", {})
    if r.get("status") == "done":
        L += ["## Face-disagreement subset", "", f"jsd_face_flag {r['jsd_face_flag']}, {r['n_subset']} of {r['n_face_rows']} clips with a face.", "",
              "| rows | n | text weighted F1 | text accuracy | two-way weighted F1 | two-way accuracy | fused3 weighted F1 | fused3 accuracy |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for part in ("subset", "complement"):
            q = r[part]
            L.append(f"| {part} | {q['n']} | " + " | ".join(f"{pct(q[b]['weighted_f1'])} | {pct(q[b]['accuracy'])}"
                                                         for b in ("text_best_k", "fused_stacked", "fused3_stacked")) + " |")
        L.append("")
    done = [r for n, r in results.items() if r["status"] == "done" and r["task"] == "emotion"]
    if done:
        L += ["## Per-class F1", "", "| condition | n | " + " | ".join(labels.EMOTIONS) + " |", "| --- | --- |" + " --- |" * len(labels.EMOTIONS)]
        L += [f"| {r['condition']} | {r['n']} | " + " | ".join(pct(r["per_class_f1"][c]) for c in labels.EMOTIONS) + " |" for r in done]
        L.append("")
    dev = meta["dev"]
    if dev["rules"]:
        L += ["## Speaker rules on dev", "", "| rule | dev weighted F1, all rows | dev weighted F1, face rows | fallback rate, dev face rows | chosen |",
              "| --- | --- | --- | --- | --- |"]
        for rule, q in dev["rules"].items():
            fb = (q.get("fallback_rate") or {}).get("dev")
            L.append(f"| {rule} | {pct(q['dev']['weighted_f1'])} | {pct(q['dev_face_rows']['weighted_f1'])} | {pct(fb)} | "
                     f"{'yes' if rule == v.get('rule') else ''} |")
        L += ["", "| dev | weighted F1 | macro F1 | accuracy |", "| --- | --- | --- | --- |"]
        for name, q in (("calibrator A", dev["calibrator"].get("A")), ("calibrator B", dev["calibrator"].get("B")),
                        ("two-way fusion", dev["two_way"]), ("fusion3", dev["fusion3"]), ("sentiment3", dev["sentiment3"])):
            if q:
                L.append(f"| {name} | {pct(q['weighted_f1'])} | {pct(q['macro_f1'])} | {pct(q['accuracy'])} |")
        L.append("")
    for r in [r for r in results.values() if r["status"] == "done" and r["condition"] in NEW_ROWS + ("sentiment3",)]:
        classes = labels.EMOTIONS if r["task"] == "emotion" else labels.SENTIMENTS
        L += [f"### Confusion matrix of {r['condition']} (rows true, columns predicted)", "",
              "| true \\ pred | " + " | ".join(classes) + " |", "| --- |" + " --- |" * len(classes)]
        L += [f"| {c} | " + " | ".join(str(x) for x in row) + " |" for c, row in zip(classes, r["confusion"])]
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--audio", type=Path, default=ROOT / "audio")
    ap.add_argument("--out", type=Path, nargs=2, default=[ROOT / "results" / "RESULTS_vision.md", ROOT / "results" / "metrics_vision.json"],
                    metavar=("RESULTS_MD", "METRICS_JSON"))
    args = ap.parse_args(argv)
    import threadpoolctl

    cmd = "python tools/evaluate_vision.py " + " ".join(sys.argv[1:] if argv is None else argv)
    t0 = time.perf_counter()
    with threadpoolctl.threadpool_limits(2):
        results, comps, meta = run(args.cache, args.heads, args.meld, args.audio)
    md, js = args.out
    md.parent.mkdir(parents=True, exist_ok=True)
    js.parent.mkdir(parents=True, exist_ok=True)
    payload = {"command": cmd, "split": "test", "n_resamples": N_RESAMPLES, "seed": SEED, "wall_s": round(time.perf_counter() - t0, 1),
               **meta, "conditions": results}
    js.write_text(json.dumps(payload, indent=1))
    md.write_text(render(results, meta, cmd))
    for name, r in results.items():
        status = f"wF1 {100 * r['weighted_f1']:.2f}  n {r['n']}" if r["status"] == "done" else r["status"]
        print(f"{name:28s} {status}")
    c = comps.get("gold_all", {})
    if "ci95" in c:
        print(f"fused3 - two-way, gold: {100 * c['diff']['weighted_f1']:+.2f} {interval(c['ci95']['weighted_f1'])}; face adds: {meta['claim']['face_adds']}")
    print(f"wrote {md} and {js} in {payload['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
