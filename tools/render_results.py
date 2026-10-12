"""render RESULTS.md at the repository root from results/metrics.json (written by
tools/evaluate.py), heads/manifest.json (written by tools/train_heads.py) and the
published rows that evaluate.py embeds in results/metrics.json: the headline
tables, per-class f1, the published rows with their sources, the pending rows
with what each needs, the dev choices and a few sentences of observation
computed from the numbers. nothing in the output is typed by hand.

usage: python tools/render_results.py [--metrics results/metrics.json] [--manifest heads/manifest.json] [--out RESULTS.md]
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from meld_rt import labels, metrics  # noqa: E402

# per-class f1 of the MELD paper's best text+audio row; evaluate.py keeps it next to its published rows
from evaluate import DialogueRNN_PER_CLASS  # noqa: E402

ECE_BINS = inspect.signature(metrics.ece).parameters["n_bins"].default
TEXT_ROWS = {"cnn": "MELD paper Table 11, text-CNN, text", "bclstm": "MELD paper Table 11, bcLSTM, text",
             "drnn": "MELD paper Table 11, DialogueRNN, text", "drnn_ta": "MELD paper Table 11, DialogueRNN, text+audio",
             "emoberta": "EmoBERTa Table 2, RoBERTa-large, no past or future utterances",
             "s_drnn_t": "MELD paper Table 13, DialogueRNN, text", "s_drnn_ta": "MELD paper Table 13, DialogueRNN, text+audio"}
NO_CODE = "no code yet: tools/evaluate.py has no branch that scores this row"
# what each pending row waits for, read off tools/evaluate.py and tools/train_heads.py
NEEDS = {
    "audio_raw_tag": "`cache/audio_test.npz` (MELD audio through `tools/extract_audio.py` and `tools/cache_audio_features.py`)",
    "audio_cal_A": "`cache/audio_test.npz` and `heads/audio_cal_A.npz` (`tools/train_heads.py` without `--text-only`, on the train and dev audio caches)",
    "audio_cal_B": "as audio_cal_A, with caches written with `--encoder-states` so that `heads/audio_cal_B.npz` exists",
    "fused_stacked": "`cache/audio_test.npz`, `heads/audio_cal.npz` and `heads/fusion.npz` (`tools/train_heads.py` without `--text-only`)",
    "fused_poe": "`cache/audio_test.npz` and `heads/audio_cal.npz`, with the weight from `heads/manifest.json`",
    "fused_poe_uncalibrated": "`cache/audio_test.npz` only (the raw SenseVoice posterior at w = 0.5)",
    "ablation_no_events": "MELD audio, heads trained without these features, and " + NO_CODE,
    "ablation_no_prosody": "MELD audio, heads trained without these features, and " + NO_CODE,
    "ablation_no_events_no_prosody": "MELD audio, heads trained without these features, and " + NO_CODE,
    "live_asr_text": ("MELD audio, text embeddings of the SenseVoice test transcripts (`tools/embed_text.py` embeds the CSV text only), and "
                      + NO_CODE),
    "disagreement_subset": "MELD audio, the jsd threshold that `tools/train_heads.py` picks on dev once audio exists, and " + NO_CODE,
    "early_commit": "MELD audio (the 2 s and 4 s prefix features are in `cache/audio_test.npz` once cached), and " + NO_CODE,
    "sentiment_fused": "`cache/audio_test.npz`, `heads/audio_cal.npz` and a fused `heads/sentiment.npz` (`tools/train_heads.py` without `--text-only`)",
}


def pct(v: float | None) -> str:
    return "" if v is None else f"{100 * v:.2f}"


def what(r: dict) -> str:
    """the condition's description without its design-section number."""
    return re.sub(r"^\d+\.\s*", "", r.get("description", ""))


def published(met: dict) -> list[dict]:
    """published rows with "same" resolved to the source of the row above."""
    out, last = [], ""
    for p in met.get("published", []):
        src = last if p["source"] == "same" else p["source"]
        last = src
        out.append({**p, "source": src})
    return out


def headline(met: dict, task: str) -> list[str]:
    rows = [r for r in met["conditions"].values() if r["task"] == task and r["status"] == "done"]
    if not rows:
        return ["No condition of this task is scored yet.", ""]
    L = ["| condition | what it is | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["ci95"]["weighted_f1"]
        note = f" ({r['n_no_audio']} test rows without a clip left out)" if r.get("n_no_audio") else ""
        L.append(f"| `{r['condition']}` | {what(r)}{note} | {r['n']} | {pct(r['weighted_f1'])} [{pct(lo)}, {pct(hi)}] | "
                 f"{pct(r['macro_f1'])} | {pct(r['accuracy'])} | {pct(r['ece'])} |")
    same = [r for r in rows if "text_best_k_same_rows_weighted_f1" in r]
    if same:
        L += ["", "Text head on the same rows as each audio or fused row: "
              + ", ".join(f"`{r['condition']}` {pct(r['text_best_k_same_rows_weighted_f1'])}" for r in same) + "."]
    return L + [""]


def per_class(met: dict) -> list[str]:
    rows = [r for r in met["conditions"].values() if r["task"] == "emotion" and r["status"] == "done"]
    if not rows:
        return []
    L = ["| condition | " + " | ".join(labels.EMOTIONS) + " |", "|---|" + "---|" * len(labels.EMOTIONS)]
    for r in rows:
        L.append(f"| `{r['condition']}` | " + " | ".join(pct(r["per_class_f1"][c]) for c in labels.EMOTIONS) + " |")
    L.append("| DialogueRNN text+audio (MELD paper, Table 11) | " + " | ".join(f"{DialogueRNN_PER_CLASS[c]:.2f}" for c in labels.EMOTIONS) + " |")
    support = [sum(row) for row in rows[0]["confusion"]]
    L.append("| test utterances in the class | " + " | ".join(str(n) for n in support) + " |")
    return L + [""]


def pending(met: dict) -> list[str]:
    rows = [r for r in met["conditions"].values() if r["status"] != "done"]
    if not rows:
        return ["None.", ""]
    L = ["| condition | what it measures | what it needs |", "|---|---|---|"]
    L += [f"| `{r['condition']}` | {what(r)} | {NEEDS.get(r['condition'], r['status'])} |" for r in rows]
    return L + [""]


def dev_choices(man: dict) -> list[str]:
    sel = man["selected"]["text_ctx"]
    thr = man["thresholds"]
    sent = man.get("sentiment", {})
    L = [f"- Text head: grid point `{sel['grid_point']}` (k = {sel['k']}, class weights `{sel['class_weight_scheme']}`), "
         f"dev weighted F1 {pct(sel['dev']['weighted_f1'])}, temperature {sel['calibration_dev']['temperature']}.",
         f"- Selector thresholds from {man.get('thresholds_source')}: low_conf {thr['low_conf']}, high_conf {thr['high_conf']}."]
    if thr.get("jsd_flag") is None:
        L.append(f"- Discrepancy flag threshold: not chosen yet ({man.get('thresholds_pending', {}).get('jsd_flag', 'pending')}). "
                 "The service uses its `--jsd-flag` default until then.")
    else:
        L.append(f"- Discrepancy flag threshold from dev: {thr['jsd_flag']}.")
    if sent:
        L.append(f"- Sentiment head ({sent.get('variant')}): C = {sent.get('C')}, class weights `{sent.get('class_weight_scheme')}`, "
                 f"dev weighted F1 {pct(sent['dev']['weighted_f1'])}, temperature {sent['calibration_dev']['temperature']}.")
    L.append(f"- Audio and fusion stage: {man.get('audio_fusion', {}).get('status')}.")
    return L + [""]


def observations(met: dict) -> list[str]:
    """three to six sentences, one claim each, computed from the numbers; a sentence whose inputs are missing is skipped."""
    c = met["conditions"]
    pub = {p["row"]: p["weighted_f1"] for p in met.get("published", [])}
    done = lambda n: c.get(n, {}).get("status") == "done"  # noqa: E731
    out = []
    if done("text_best_k"):
        tb = c["text_best_k"]
        lo, hi = tb["ci95"]["weighted_f1"]
        s = f"The deployed text head scores {pct(tb['weighted_f1'])} weighted F1 on test (95 % CI [{pct(lo)}, {pct(hi)}])"
        out.append(s + (f", against {pct(c['majority']['weighted_f1'])} for the majority floor." if done("majority") else "."))
        text_pub = [pub[v] for k, v in TEXT_ROWS.items() if v in pub and k in ("cnn", "bclstm", "drnn")]
        if text_pub:
            best = max(text_pub)
            if 100 * lo > best:
                rel = "below the lower end of that interval"
            elif 100 * hi < best:
                rel = "above the upper end of that interval"
            else:
                rel = "inside that interval"
            out.append(f"The best text-only row of the MELD paper, {best:.2f} (Table 11), is {rel}.")
        if TEXT_ROWS["emoberta"] in pub:
            e = pub[TEXT_ROWS["emoberta"]]
            rel = "above" if e > 100 * hi else ("below" if e < 100 * lo else "inside")
            out.append(f"Fine-tuned RoBERTa-large without context reaches {e:.2f} in EmoBERTa (Table 2), "
                       + ("inside the interval." if rel == "inside" else f"{rel} its {'upper' if rel == 'above' else 'lower'} end."))
        support = [sum(row) for row in tb["confusion"]]
        f1 = tb["per_class_f1"]
        low2 = sorted(labels.EMOTIONS, key=lambda e: f1[e])[:2]
        rare2 = sorted(labels.EMOTIONS, key=lambda e: support[labels.EMOTIONS.index(e)])[:2]
        n2 = [support[labels.EMOTIONS.index(e)] for e in low2]
        out.append(f"Macro F1 is {pct(tb['macro_f1'])}, and the two lowest per-class scores are {low2[0]} ({pct(f1[low2[0]])}) and "
                   f"{low2[1]} ({pct(f1[low2[1]])}), "
                   + ("the two classes with the fewest test utterances " if set(low2) == set(rare2) else "classes with ")
                   + f"({n2[0]} and {n2[1]}).")
    if done("sentiment_text_only") and TEXT_ROWS["s_drnn_t"] in pub and TEXT_ROWS["s_drnn_ta"] in pub:
        out.append(f"The text-only sentiment head scores {pct(c['sentiment_text_only']['weighted_f1'])}, next to "
                   f"{pub[TEXT_ROWS['s_drnn_t']]:.2f} for DialogueRNN on text and {pub[TEXT_ROWS['s_drnn_ta']]:.2f} on text and audio (MELD paper, Table 13).")
    n_pending = sum(r["status"] != "done" for r in c.values())
    if n_pending:
        out.append(f"Of the {len(c)} conditions, {n_pending} are pending, and the table above says what each one needs.")
    if done("text_best_k") and done("text_runtime_regime"):
        d = c["text_runtime_regime"]["weighted_f1"] - c["text_best_k"]["weighted_f1"]
        out.append(f"Restricting the context to the same speaker changes the score by {abs(100 * d):.2f} points.")
    return out[:6]


def render(met: dict, man: dict, cmd: str, metrics_path: str, manifest_path: str) -> str:
    intro = (f"Rendered by `{cmd}` from `{metrics_path}` (written by `{met['command']}`) and `{manifest_path}` "
             f"(written by `{man['command']}`). The full report with the confusion matrices is [results/RESULTS.md](results/RESULTS.md), "
             "and the dev grid of the text head is in [results/01_text_only.md](results/01_text_only.md). Model selection, temperature and thresholds were chosen on dev. "
             "The test split was read once per condition. Weighted F1 is the primary metric, as in the published rows. "
             f"The intervals are 95 % percentile bootstrap intervals over test dialogues ({met['n_resamples']} resamples, seed {met['seed']}). "
             f"ECE uses {ECE_BINS} equal-width confidence bins. All metrics are in percent.")
    L = ["# Results", "", intro, "", "## Emotion on MELD test (7 classes)", ""] + headline(met, "emotion")
    L += ["## Sentiment on MELD test (3 classes)", ""] + headline(met, "sentiment")
    L += ["## Per-class F1 (emotion, test)", ""] + per_class(met)
    L += ["## Published rows", "", "| row | task | weighted F1 | source |", "|---|---|---|---|"]
    L += [f"| {p['row']} | {p['task']} | {'' if p['weighted_f1'] is None else f'{p['weighted_f1']:.2f}'} | {p['source']} |" for p in published(met)]
    L += ["", "## Pending rows", ""] + pending(met)
    L += ["## Choices made on dev (`heads/manifest.json`)", ""] + dev_choices(man)
    L += ["## Observations", ""] + [" ".join(observations(met)), ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="render RESULTS.md from results/metrics.json and heads/manifest.json")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "heads" / "manifest.json")
    ap.add_argument("--out", type=Path, default=ROOT / "RESULTS.md")
    a = ap.parse_args(argv)
    met = json.loads(a.metrics.read_text(encoding="utf-8"))
    man = json.loads(a.manifest.read_text(encoding="utf-8"))
    cmd = "python tools/render_results.py" + ("" if not (argv if argv is not None else sys.argv[1:]) else " " + " ".join(argv if argv is not None else sys.argv[1:]))

    def rel(p: Path) -> str:
        p = p.resolve()
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

    a.out.write_text(render(met, man, cmd, rel(a.metrics), rel(a.manifest)), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
