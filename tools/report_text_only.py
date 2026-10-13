"""render results/01_text_only.md from heads/manifest.json and results/metrics.json.

the dev grid (k x class-weight scheme, weighted and macro f1), the chosen
configuration, the test rows of the text conditions and short sentences of
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


def _points(d: float) -> str:
    """a difference in points, written as a magnitude with a direction word."""
    return f"{abs(100 * d):.2f} points {'below' if d < 0 else 'above'}"


def observations(man: dict, met: dict) -> list[str]:
    """short sentences, each one claim, every number read from the manifest or the metrics."""
    dev = man["dev_scores"]
    best = max(dev, key=lambda n: dev[n]["weighted_f1"])
    best_macro = max(dev, key=lambda n: dev[n]["macro_f1"])
    b, bm = dev[best], dev[best_macro]
    out = [f"On dev, the grid point `{best}` (k = {b['k']}, class weights `{b['class_weight_scheme']}`) has the best weighted F1, "
           f"{pct(b['weighted_f1'])}, with a macro F1 of {pct(b['macro_f1'])}."]
    if best_macro != best:
        out.append(f"The best macro F1 on dev is {pct(bm['macro_f1'])}, at `{best_macro}`, whose weighted F1 is {pct(bm['weighted_f1'])}. "
                   f"Choosing by weighted F1 costs {abs(100 * (bm['macro_f1'] - b['macro_f1'])):.2f} macro-F1 points "
                   f"and gains {abs(100 * (b['weighted_f1'] - bm['weighted_f1'])):.2f} weighted-F1 points.")
    readme = [r for r in dev.values() if r["class_weight_scheme"] == "meld_readme"]
    if readme and b["class_weight_scheme"] != "meld_readme":
        rb = max(readme, key=lambda r: r["weighted_f1"])
        dw, dm = rb["weighted_f1"] - b["weighted_f1"], rb["macro_f1"] - b["macro_f1"]
        gap = (f"{abs(100 * dw):.2f} and {abs(100 * dm):.2f} points {'below' if dw < 0 else 'above'}" if (dw < 0) == (dm < 0)
               else f"{_points(dw)} and {_points(dm)}")
        out.append(f"The MELD README class weights reach at best {pct(rb['weighted_f1'])} weighted F1 and {pct(rb['macro_f1'])} macro F1, "
                   f"{gap} the chosen point.")
    by_k = {}
    for r in dev.values():
        by_k[r["k"]] = max(by_k.get(r["k"], 0.0), r["weighted_f1"])
    kbest = max(by_k, key=by_k.get)
    others = " and ".join(f"{pct(v)} at k = {k}" for k, v in sorted(by_k.items()) if k != kbest)
    if kbest == 0:
        out.append(f"Context did not help on dev: the best points reach {others}, against {pct(by_k[0])} at k = 0. "
                   "The deployed head sees the current utterance only.")
    else:
        out.append(f"Context helped on dev: k = {kbest} reaches {pct(by_k[kbest])}, against {others}. The deployed head uses k = {kbest}.")
    cond = met["conditions"]
    if all(cond.get(n, {}).get("status") == "done" for n in ("text_k0", "text_best_k", "text_runtime_regime")):
        t0, tb, tr = cond["text_k0"], cond["text_best_k"], cond["text_runtime_regime"]
        lo, hi = tb["ci95"]["weighted_f1"]
        out.append(f"On test, the deployed head scores {pct(tb['weighted_f1'])} weighted F1 (95 % CI [{pct(lo)}, {pct(hi)}]). "
                   f"The k = 0 head scores {pct(t0['weighted_f1'])}, and the deployed head with the same-speaker runtime window scores {pct(tr['weighted_f1'])}.")
        pub = {p["row"]: p["weighted_f1"] for p in met.get("published", [])}
        rows = [("text-CNN", "MELD paper Table 11, text-CNN, text"), ("bcLSTM", "MELD paper Table 11, bcLSTM, text"),
                ("DialogueRNN", "MELD paper Table 11, DialogueRNN, text")]
        refs = [f"{name} {pub[row]:.2f}" for name, row in rows if row in pub]
        emo = "EmoBERTa Table 2, RoBERTa-large, no past or future utterances"
        if refs:
            out.append(f"The published text-only rows of the MELD paper (Table 11) are {', '.join(refs[:-1])} and {refs[-1]}."
                       if len(refs) > 1 else f"The published text-only row of the MELD paper (Table 11) is {refs[0]}.")
        if emo in pub:
            out.append(f"EmoBERTa, a fine-tuned RoBERTa-large without context, reports {pub[emo]:.2f} (Table 2).")
    return out


def render(man: dict, met: dict, cmd: str) -> str:
    sel = man["selected"]["text_ctx"]
    cal, thr = sel["calibration_dev"], man["thresholds"]
    cond = met["conditions"]
    intro = (f"Rendered by `{cmd}` from `heads/manifest.json` (written by `{man['command']}`) and `results/metrics.json` "
             f"(written by `{met['command']}`). The inputs are frozen RoBERTa-base embeddings, mean-pooled, in `cache/text_{{split}}.npy` "
             f"(written by `tools/embed_text.py`). The TextContextHead is an MLP with {man['config']['text_ctx']['mlp']['hidden']} hidden units "
             f"and early stopping on dev weighted F1 (scikit-learn {man['sklearn_version']}, {man['threads']} threads, seed {man['seed']}). "
             "Metrics are in percent.")
    L = ["# Experiment 01, text-only heads", "", intro, "", "## Dev grid (weighted F1 / macro F1, best epoch / epochs run)", ""]
    L += grid_table(man["dev_scores"])
    chosen = (f"- Grid point `{sel['grid_point']}`: k = {sel['k']}, class weights `{sel['class_weight_scheme']}`, "
              f"dev weighted F1 {pct(sel['dev']['weighted_f1'])}, macro F1 {pct(sel['dev']['macro_f1'])}, accuracy {pct(sel['dev']['accuracy'])}, "
              f"{sel['n_params']} parameters in `heads/{sel['file']}`.")
    pending = man.get("thresholds_pending", {}).get("jsd_flag")
    thresholds = (f"- Selector thresholds: low_conf = {thr['low_conf']} and high_conf = {thr['high_conf']} ({man['thresholds_source']}). "
                  f"jsd_flag = {thr['jsd_flag']}" + (f" ({pending}), so the service uses its `--jsd-flag` default." if pending else "."))
    L += ["", f"Training the grid took {man['wall_time_s']['text_ctx']} s. The whole run took {man['wall_time_s']['total']} s.", "",
          "## Chosen configuration (dev)", "", chosen,
          (f"- Temperature {cal['temperature']}: dev NLL from {cal['nll_before']} to {cal['nll_after']}, "
           f"dev ECE from {pct(cal['ece_before'])} to {pct(cal['ece_after'])}."),
          thresholds,
          f"- The k = 0 row uses `{man['selected']['text_ctx_k0']['grid_point']}` in `heads/{man['selected']['text_ctx_k0']['file']}`."]
    sent = man.get("sentiment", {})
    if sent:
        L.append(f"- Sentiment head ({sent.get('variant')}): logistic regression over {sent.get('input')}, C = {sent.get('C')}, "
                 f"class weights `{sent.get('class_weight_scheme')}`, dev weighted F1 {pct(sent['dev']['weighted_f1'])}, "
                 f"temperature {sent['calibration_dev']['temperature']}.")
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
        L += ["", "### Published text-only rows (test weighted F1, sources in results/RESULTS.md)", ""]
        L += [f"- {p['row']}: {p['weighted_f1']:.2f}" for p in pub]
    L += ["", "## Observations", ""] + observations(man, met) + [""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=ROOT / "heads" / "manifest.json")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "01_text_only.md")
    args = ap.parse_args(argv)
    man, met = json.loads(args.manifest.read_text(encoding="utf-8")), json.loads(args.metrics.read_text(encoding="utf-8"))
    cmd = "python tools/report_text_only.py " + " ".join(sys.argv[1:] if argv is None else argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(man, met, cmd), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
