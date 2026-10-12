"""render results/01_text_only.md from heads/manifest.json and results/metrics.json.

the dev grid (k x class-weight scheme, weighted and macro f1), the chosen
configuration, the test rows of the text conditions and two sentences of
observation derived from those numbers. nothing here is typed by hand.

usage: python tools/report_text_only.py --manifest heads/manifest.json --metrics results/metrics.json --out results/01_text_only.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_ROWS = ["majority", "text_k0", "text_best_k", "text_runtime_regime"]
SENT_ROWS = ["sentiment_text_only"]


def pct(v: float) -> str:
    return f"{100 * v:.2f}"


def grid_table(dev: dict) -> list[str]:
    """k x class-weight cells, each showing its best learning rate; then every grid point."""
    ks = sorted({r["k"] for r in dev.values()})
    schemes = []
    for r in dev.values():
        if r["class_weight_scheme"] not in schemes:
            schemes.append(r["class_weight_scheme"])
    L = ["| k \\ class weights | " + " | ".join(f"{s} (wF1 / mF1)" for s in schemes) + " |", "| --- |" + " --- |" * len(schemes)]
    for k in ks:
        cells = []
        for s in schemes:
            r = max((r for r in dev.values() if r["k"] == k and r["class_weight_scheme"] == s), key=lambda r: r["weighted_f1"])
            cells.append(f"{pct(r['weighted_f1'])} / {pct(r['macro_f1'])} (lr {r.get('learning_rate_init', '')}, ep {r['best_epoch'] + 1}/{r['epochs']})")
        L.append(f"| k = {k} | " + " | ".join(cells) + " |")
    L += ["", "All grid points:", "", "| grid point | weighted F1 | macro F1 | accuracy | best epoch / run | wall s |",
          "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {n} | {pct(r['weighted_f1'])} | {pct(r['macro_f1'])} | {pct(r['accuracy'])} | {r['best_epoch'] + 1} / {r['epochs']} | {r['wall_s']} |"
          for n, r in dev.items()]
    return L


def observations(man: dict, met: dict) -> list[str]:
    dev = man["dev_scores"]
    best = max(dev, key=lambda n: dev[n]["weighted_f1"])
    best_macro = max(dev, key=lambda n: dev[n]["macro_f1"])
    b, bm = dev[best], dev[best_macro]
    readme = [r for r in dev.values() if r["class_weight_scheme"] == "meld_readme"]
    s1 = (f"On dev the grid point {best} (k = {b['k']}, class weights `{b['class_weight_scheme']}`) won on weighted F1 with "
          f"{pct(b['weighted_f1'])}; its macro F1 is {pct(b['macro_f1'])}, against {pct(bm['macro_f1'])} for the best macro-F1 point "
          f"{best_macro} (whose weighted F1 is {pct(bm['weighted_f1'])}), so the chosen scheme "
          f"{'costs' if bm['macro_f1'] > b['macro_f1'] else 'gives up nothing in'} "
          f"{abs(100 * (bm['macro_f1'] - b['macro_f1'])):.2f} macro-F1 points for {100 * (b['weighted_f1'] - bm['weighted_f1']):+.2f} weighted-F1 points")
    if readme:
        rb = max(readme, key=lambda r: r["weighted_f1"])
        s1 += (f"; the MELD README weights reach at best {pct(rb['weighted_f1'])} / {pct(rb['macro_f1'])} (weighted / macro), "
               f"{100 * (rb['weighted_f1'] - b['weighted_f1']):+.2f} and {100 * (rb['macro_f1'] - b['macro_f1']):+.2f} points against the chosen point")
    s1 += "."
    by_k = {}
    for r in dev.values():
        by_k[r["k"]] = max(by_k.get(r["k"], 0.0), r["weighted_f1"])
    kbest = max(by_k, key=by_k.get)
    others = ", ".join(f"k = {k} reaches {pct(v)}" for k, v in sorted(by_k.items()) if k != kbest)
    cond = met["conditions"]
    if kbest == 0:
        s2 = f"Context did not help on dev: {others} against {pct(by_k[0])} at k = 0, so the deployed head sees the current utterance only"
    else:
        s2 = f"Context helped on dev: k = {kbest} reaches {pct(by_k[kbest])} while {others}, so the deployed head uses k = {kbest}"
    if all(cond.get(n, {}).get("status") == "done" for n in ("text_k0", "text_best_k", "text_runtime_regime")):
        t0, tb, tr = cond["text_k0"], cond["text_best_k"], cond["text_runtime_regime"]
        lo, hi = tb["ci95"]["weighted_f1"]
        pub = {p["row"]: p["weighted_f1"] for p in met.get("published", [])}
        rows = [("text-CNN", "MELD paper Table 11, text-CNN, text"), ("bcLSTM", "MELD paper Table 11, bcLSTM, text"),
                ("DialogueRNN", "MELD paper Table 11, DialogueRNN, text"),
                ("EmoBERTa without context", "EmoBERTa Table 2, RoBERTa-large, no past or future utterances")]
        refs = "; ".join(f"{name} {pub[row]:.2f}" for name, row in rows if row in pub)
        s2 += (f"; on test the deployed head scores {pct(tb['weighted_f1'])} weighted F1 (95 % CI [{pct(lo)}, {pct(hi)}]) against "
               f"{pct(t0['weighted_f1'])} for k = 0 and {pct(tr['weighted_f1'])} with the same-speaker runtime-regime window, "
               f"next to the published text-only rows {refs}")
    s2 += "."
    return [s1, s2]


def render(man: dict, met: dict, cmd: str) -> str:
    sel = man["selected"]["text_ctx"]
    cal, thr = sel["calibration_dev"], man["thresholds"]
    cond = met["conditions"]
    intro = (f"Rendered by `{cmd}` from `heads/manifest.json` (written by `{man['command']}`) and `results/metrics.json` "
             f"(written by `{met['command']}`). Frozen RoBERTa-base mean-pooled embeddings (`cache/text_{{split}}.npy`, WP2), "
             f"TextContextHead MLP {man['config']['text_ctx']['mlp']['hidden']} hidden units, early stopping on dev weighted F1, "
             f"scikit-learn {man['sklearn_version']}, {man['threads']} threads, seed {man['seed']}. Metrics in percent.")
    L = ["# Experiment 01: text-only heads", "", intro, "", "## Dev grid (weighted F1 / macro F1, best epoch / epochs run)", ""]
    L += grid_table(man["dev_scores"])
    chosen = (f"- grid point `{sel['grid_point']}`: k = {sel['k']}, class weights `{sel['class_weight_scheme']}`, "
              f"dev weighted F1 {pct(sel['dev']['weighted_f1'])}, macro F1 {pct(sel['dev']['macro_f1'])}, accuracy {pct(sel['dev']['accuracy'])}; "
              f"{sel['n_params']} parameters in `heads/{sel['file']}`")
    pending = man.get("thresholds_pending", {}).get("jsd_flag")
    thresholds = (f"- selector thresholds low_conf = {thr['low_conf']}, high_conf = {thr['high_conf']} ({man['thresholds_source']}); "
                  f"jsd_flag = {thr['jsd_flag']}" + (f" ({pending})" if pending else ""))
    L += ["", f"Training wall time for the grid: {man['wall_time_s']['text_ctx']} s; total run {man['wall_time_s']['total']} s.", "",
          "## Chosen configuration (dev)", "", chosen,
          f"- temperature {cal['temperature']} (dev NLL {cal['nll_before']} -> {cal['nll_after']}, dev ECE {pct(cal['ece_before'])} -> {pct(cal['ece_after'])})",
          thresholds,
          f"- k = 0 head kept for the k = 0 row: `{man['selected']['text_ctx_k0']['grid_point']}` in `heads/{man['selected']['text_ctx_k0']['file']}`"]
    sent = man.get("sentiment", {})
    if sent:
        L.append(f"- sentiment head ({sent.get('variant')}): logistic regression over {sent.get('input')}, C = {sent.get('C')}, "
                 f"class weights `{sent.get('class_weight_scheme')}`, dev weighted F1 {pct(sent['dev']['weighted_f1'])}, "
                 f"temperature {sent['calibration_dev']['temperature']}")
    L += ["", "## Test (one run per condition)", "",
          "| condition | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | note |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in TEXT_ROWS + SENT_ROWS:
        r = cond.get(name)
        if not r:
            continue
        if r["status"] == "done":
            lo, hi = r["ci95"]["weighted_f1"]
            L.append(f"| {name} | {pct(r['weighted_f1'])} | [{pct(lo)}, {pct(hi)}] | {pct(r['macro_f1'])} | {pct(r['accuracy'])} | {pct(r['ece'])} | {r.get('note', '')} |")
        else:
            L.append(f"| {name} | {r['status']} | | | | | |")
    emo = [r for r in (cond.get(n) for n in TEXT_ROWS) if r and r["status"] == "done"]
    if emo:
        classes = list(emo[0]["per_class_f1"])
        L += ["", "### Per-class F1 (test)", "", "| condition | " + " | ".join(classes) + " |", "| --- |" + " --- |" * len(classes)]
        L += [f"| {r['condition']} | " + " | ".join(pct(r["per_class_f1"][c]) for c in classes) + " |" for r in emo]
    pub = [p for p in met.get("published", []) if p["task"] == "emotion" and "text" in p["row"] and "audio" not in p["row"] and p["weighted_f1"]]
    if pub:
        L += ["", "### Published text-only rows (test weighted F1, from the sources named in RESULTS.md)", ""]
        L += [f"- {p['row']}: {p['weighted_f1']:.2f}" for p in pub]
    L += ["", "## Observations", ""] + observations(man, met) + [""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=ROOT / "heads" / "manifest.json")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "01_text_only.md")
    args = ap.parse_args(argv)
    man, met = json.loads(args.manifest.read_text()), json.loads(args.metrics.read_text())
    cmd = "python tools/report_text_only.py " + " ".join(sys.argv[1:] if argv is None else argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(man, met, cmd))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
