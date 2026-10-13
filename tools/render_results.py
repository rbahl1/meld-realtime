"""render RESULTS.md at the repository root from results/metrics.json (written by
tools/evaluate.py), heads/manifest.json (written by tools/train_heads.py) and the
published rows that evaluate.py embeds in results/metrics.json: the main tables,
gold text next to the live asr transcripts, the ablations, the disagreement
subset, early commit, per-class f1, the published rows with their sources, the
pending rows with what each needs (only while a row is pending), the dev choices,
and sentences of observation computed from the numbers, the negative ones
included. a sentence whose inputs are missing is skipped. nothing in the output
is typed by hand.

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
SENSEVOICE_ROW = "FunAudioLLM Table 8, SenseVoice-S on MELD"
# what each pending row waits for, read off tools/evaluate.py and tools/train_heads.py
NEEDS = {
    "audio_raw_tag": "`cache/audio_test.npz` (MELD audio through `tools/extract_audio.py` and `tools/cache_audio_features.py`)",
    "audio_cal_A": "`cache/audio_test.npz` and `heads/audio_cal_A.npz` (`tools/train_heads.py` without `--text-only`, on the train and dev audio caches)",
    "audio_cal_B": "as audio_cal_A, with caches written with `--encoder-states` so that `heads/audio_cal_B.npz` exists",
    "fused_stacked": "`cache/audio_test.npz`, `heads/audio_cal.npz` and `heads/fusion.npz` (`tools/train_heads.py` without `--text-only`)",
    "fused_poe": "`cache/audio_test.npz` and `heads/audio_cal.npz`, with the weight from `heads/manifest.json`",
    "fused_poe_uncalibrated": "`cache/audio_test.npz` only (the raw SenseVoice posterior at w = 0.5)",
    "ablation_no_events": "the ablation heads (`tools/train_heads.py --ablation`)",
    "ablation_no_prosody": "the ablation heads (`tools/train_heads.py --ablation`)",
    "ablation_no_events_no_prosody": "the ablation heads (`tools/train_heads.py --ablation`)",
    "live_asr_text_only": "`cache/text_asr_test.npy` (`tools/embed_text.py --split test --from-asr`)",
    "live_asr_text": "`cache/text_asr_test.npy` (`tools/embed_text.py --split test --from-asr`) and the fusion head",
    "disagreement_subset": "`cache/audio_test.npz` and the jsd threshold that `tools/train_heads.py` picks on dev",
    "early_commit_audio_2s": "the 2 s prefix features of `cache/audio_test.npz` (`--prefixes 2,4`)",
    "early_commit_audio_4s": "the 4 s prefix features of `cache/audio_test.npz` (`--prefixes 2,4`)",
    "early_commit_fused_2s": "the 2 s prefix features and `cache/text_asr_test_p2.npy` (`tools/embed_text.py --split test --from-asr asr_text_p2`)",
    "early_commit_fused_4s": "the 4 s prefix features and `cache/text_asr_test_p4.npy` (`tools/embed_text.py --split test --from-asr asr_text_p4`)",
    "sentiment_text_only": ("`heads/sentiment_text_only.npz` (`tools/train_heads.py --sentiment-text-only`, next to the fused "
                            "`heads/sentiment.npz` that the service uses), or a text-only `heads/sentiment.npz` (`tools/train_heads.py --text-only`)"),
    "sentiment_fused": "`cache/audio_test.npz`, `heads/audio_cal.npz` and a fused `heads/sentiment.npz` (`tools/train_heads.py` without `--text-only`)",
}
TABLE_HEAD = ["| condition | what it is | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE |", "|---|---|---|---|---|---|---|"]


def pct(v: float | None) -> str:
    return "" if v is None else f"{100 * v:.2f}"


def what(r: dict) -> str:
    """the condition's description without its design-section number."""
    return sentences(re.sub(r"^\d+\.\s*", "", r.get("description", "")))


def group_of(name: str) -> str:
    if name.startswith("ablation_"):
        return "ablation"
    if name.startswith("live_asr"):
        return "live"
    if name == "disagreement_subset":
        return "disagreement"
    if name.startswith("early_commit"):
        return "early"
    return "main"


def done(c: dict, name: str) -> bool:
    return c.get(name, {}).get("status") == "done"


def ci(r: dict) -> tuple[float, float]:
    lo, hi = r["ci95"]["weighted_f1"]
    return lo, hi


def wci(r: dict) -> str:
    lo, hi = ci(r)
    return f"{pct(r['weighted_f1'])} [{pct(lo)}, {pct(hi)}]"


def shown(v: float) -> float:
    """a score as the tables print it, in percent with two decimals."""
    return float(pct(v))


def pts(a: float, b: float) -> str:
    """the absolute difference of two scores in percentage points, taken from the values as printed,
    so a reader who subtracts the printed numbers gets the same gap."""
    return f"{abs(shown(a) - shown(b)):.2f}"


def side(a: float, b: float) -> str:
    a, b = shown(a), shown(b)
    return "above" if a > b else ("below" if a < b else "level with")


def sentences(cell: str) -> str:
    """a table cell from evaluate.py with "; " between clauses, written as sentences: a clause
    inside parentheses is joined with a comma, any other starts a new sentence."""
    out, depth, i = [], 0, 0
    while i < len(cell):
        ch = cell[i]
        depth += ch == "("
        depth -= ch == ")"
        if cell.startswith("; ", i):
            if depth > 0:
                out.append(", ")
            else:
                out.append(". ")
                if i + 2 < len(cell):
                    out.append(cell[i + 2].upper())
                    i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def inside(v: float, r: dict) -> bool:
    lo, hi = ci(r)
    return lo <= v <= hi


def published(met: dict) -> list[dict]:
    """published rows with "same" resolved to the source of the row above."""
    out, last = [], ""
    for p in met.get("published", []):
        src = last if p["source"] == "same" else p["source"]
        last = src
        out.append({**p, "source": src})
    return out


def table_row(r: dict) -> str:
    note = f" ({r['n_no_audio']} test rows without a clip left out)" if r.get("n_no_audio") else ""
    return (f"| `{r['condition']}` | {what(r)}{note} | {r['n']} | {wci(r)} | "
            f"{pct(r['macro_f1'])} | {pct(r['accuracy'])} | {pct(r['ece'])} |")


def headline(met: dict, task: str, group: str = "main") -> list[str]:
    rows = [r for r in met["conditions"].values() if r["task"] == task and r["status"] == "done" and group_of(r["condition"]) == group]
    if not rows:
        return ["No condition of this task is scored yet.", ""]
    L = TABLE_HEAD + [table_row(r) for r in rows]
    same = [r for r in rows if "text_best_k_same_rows_weighted_f1" in r]
    if same:
        L += ["", "Text head on the same rows as each audio or fused row: "
              + ", ".join(f"`{r['condition']}` {pct(r['text_best_k_same_rows_weighted_f1'])}" for r in same) + "."]
    return L + [""]


def main_notes(met: dict) -> list[str]:
    """sentences under the main emotion table."""
    c = met["conditions"]
    out = []
    if done(c, "audio_raw_tag") and done(c, "audio_cal_A"):
        raw, cal = c["audio_raw_tag"], c["audio_cal_A"]
        out.append(f"Calibrator A lowers the ECE of the audio posterior from {pct(raw['ece'])} to {pct(cal['ece'])}. "
                   f"Its weighted F1 is {pct(cal['weighted_f1'])}, {pts(cal['weighted_f1'], raw['weighted_f1'])} points "
                   f"{side(cal['weighted_f1'], raw['weighted_f1'])} the raw SenseVoice tag.")
    if done(c, "audio_cal_A") and done(c, "audio_cal_B"):
        a, b = c["audio_cal_A"]["weighted_f1"], c["audio_cal_B"]["weighted_f1"]
        out.append(f"Calibrator B, which adds the encoder states, scores {pct(b)} against {pct(a)} for calibrator A.")
    if done(c, "fused_stacked") and done(c, "fused_poe") and done(c, "fused_poe_uncalibrated"):
        out.append(f"The product of experts scores {pct(c['fused_poe']['weighted_f1'])} at the dev weight and "
                   f"{pct(c['fused_poe_uncalibrated']['weighted_f1'])} at the fixed weight, against {pct(c['fused_stacked']['weighted_f1'])} "
                   f"for the stacked fusion. Its ECE is {pct(c['fused_poe']['ece'])} and {pct(c['fused_poe_uncalibrated']['ece'])}, "
                   f"against {pct(c['fused_stacked']['ece'])}.")
    return [" ".join(out), ""] if out else []


def live_asr(met: dict) -> list[str]:
    c = met["conditions"]
    pairs = [("text_best_k", "live_asr_text_only"), ("fused_stacked", "live_asr_text")]
    pairs = [(g, a) for g, a in pairs if done(c, g) and done(c, a)]
    if not pairs:
        return ["No live ASR row is scored yet.", ""]
    L = ["| condition | text input | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE |", "|---|---|---|---|---|---|---|"]
    for g, a in pairs:
        for name, src in ((g, "gold CSV text"), (a, "SenseVoice transcripts")):
            r = c[name]
            L.append(f"| `{name}` | {src} | {r['n']} | {wci(r)} | {pct(r['macro_f1'])} | {pct(r['accuracy'])} | {pct(r['ece'])} |")
    r = c[pairs[0][1]]
    L += ["", f"The transcripts have a corpus WER of {r['wer']} over {r['wer_n']} test utterances (`cache/wer_test.json`). "
          f"The rows apply the service's ASR gate: {r['n_empty_transcripts']} empty transcripts and {r['n_low_asr_conf']} transcripts "
          f"with an ASR confidence below {r['min_asr_conf']} get a uniform posterior and the label neutral. "
          "Without the gate the rows score " + " and ".join(f"{pct(c[a]['without_gate']['weighted_f1'])} (`{a}`)" for _, a in pairs) + ". "
          "The heads were trained on gold text. A text head trained on SenseVoice transcripts of train and dev was not built."]
    who = {"live_asr_text_only": "the text head", "live_asr_text": "the stacked fusion"}
    parts = [f"{who[a]} scores {pct(c[a]['weighted_f1'])} against {pct(c[g]['weighted_f1'])} on gold text "
             f"({pts(c[a]['weighted_f1'], c[g]['weighted_f1'])} points {side(c[a]['weighted_f1'], c[g]['weighted_f1'])})" for g, a in pairs]
    return L + ["", "On the transcripts, " + ", and ".join(parts) + ".", ""]


def ablations(met: dict, man: dict) -> list[str]:
    c = met["conditions"]
    rows = [r for r in c.values() if group_of(r["condition"]) == "ablation" and r["status"] == "done"]
    if not rows or not done(c, "fused_stacked"):
        return ["No ablation is scored yet.", ""]
    ref = c["fused_stacked"]
    ab = man.get("ablations", {})
    var = ab.get("variants", {})
    refm = ab.get("reference", {})
    L = [("| condition | dropped | calibrator inputs | fusion inputs | weighted F1 [95 % CI] | macro F1 | accuracy | ECE | "
         "audio-only weighted F1 | dev weighted F1 of the fusion |"), "|---|---|---|---|---|---|---|---|---|---|"]
    if refm and done(c, "audio_cal_A"):
        L.append(f"| `fused_stacked` (deployed) | nothing | {refm['audio_cal']['input_dim']} | {refm['fusion']['input_dim']} | {wci(ref)} | "
                 f"{pct(ref['macro_f1'])} | {pct(ref['accuracy'])} | {pct(ref['ece'])} | {pct(c['audio_cal_A']['weighted_f1'])} | "
                 f"{pct(refm['fusion']['dev']['weighted_f1'])} |")
    for r in rows:
        v = var.get(r["condition"].removeprefix("ablation_"), {})
        dev = pct(v["fusion"]["dev"]["weighted_f1"]) if v else ""
        L.append(f"| `{r['condition']}` | {' and '.join(r.get('dropped', []))} | {r.get('audio_cal_input_dim', '')} | "
                 f"{r.get('fusion_input_dim', '')} | {wci(r)} | {pct(r['macro_f1'])} | {pct(r['accuracy'])} | {pct(r['ece'])} | "
                 f"{pct(r['audio_only']['weighted_f1'])} | {dev} |")
    L += ["", (f"The ablation heads were trained by `{ab.get('command', 'tools/train_heads.py --ablation')}` with the same folds, grids and "
          "dev selection as the deployed heads. Without events the fusion also loses the laughter probability. The deployed heads were "
          "not changed, and the reference heads that the same run retrains match the deployed files (`reproduces_deployed` in "
          "`heads/manifest.json`).")]
    obs = []
    for r in rows:
        x = r["weighted_f1"]
        obs.append(f"Without {' and '.join(r.get('dropped', []))}, the fusion scores {pct(x)}, {pts(x, ref['weighted_f1'])} points "
                   f"{side(x, ref['weighted_f1'])} the deployed fusion and {'inside' if inside(x, ref) else 'outside'} its interval.")
    if all(inside(r["weighted_f1"], ref) for r in rows):
        obs.append("No ablation leaves the deployed fusion's interval, so the event and prosody inputs add nothing that this test set can measure.")
    if refm and var and all(v["fusion"]["dev"]["weighted_f1"] >= refm["fusion"]["dev"]["weighted_f1"] for v in var.values()):
        obs.append("On dev, every ablated fusion also scores at least as high as the deployed one.")
    return L + ["", " ".join(obs), ""]


def disagreement(met: dict) -> list[str]:
    c = met["conditions"]
    if not done(c, "disagreement_subset"):
        return ["The disagreement subset is not scored yet.", ""]
    r = c["disagreement_subset"]
    L = ["| rows | n | text weighted F1 | audio weighted F1 | fused weighted F1 | text accuracy | audio accuracy | fused accuracy |",
         "|---|---|---|---|---|---|---|---|"]
    for part in ("subset", "complement"):
        p = r[part]
        L.append(f"| {part} | {p['n']} | {pct(p['text']['weighted_f1'])} | {pct(p['audio_cal']['weighted_f1'])} | "
                 f"{pct(p['fused_stacked']['weighted_f1'])} | {pct(p['text']['accuracy'])} | {pct(p['audio_cal']['accuracy'])} | "
                 f"{pct(p['fused_stacked']['accuracy'])} |")
    L += ["", (f"`disagreement_subset` holds the {r['n_subset']} of {r['n_rows']} test utterances ({pct(r['share'])} %) whose JSD between "
          f"the gold-text posterior and the raw SenseVoice posterior is above {r['jsd_flag']} (`thresholds.jsd_flag`, the dev threshold). "
          f"The stacked fusion scores {wci(r)} weighted F1 on it. The audio column is the deployed calibrator.")]
    s, comp = r["subset"], r["complement"]
    ft, tt = s["fused_stacked"]["weighted_f1"], s["text"]["weighted_f1"]
    obs = [f"On the subset, the fusion scores {pct(ft)} against {pct(tt)} for the text head alone, {pts(ft, tt)} points {side(ft, tt)} it."]
    fc, tc = comp["fused_stacked"]["weighted_f1"], comp["text"]["weighted_f1"]
    obs.append(f"On the complement it scores {pct(fc)} against {pct(tc)}, {pts(fc, tc)} points {side(fc, tc)}.")
    if ft <= tt:
        obs.append("So the fusion shows no gain over the text head where the words and the voice disagree, and the decision rule in "
                   "[docs/decisions.md](docs/decisions.md) (Fusion) takes that claim out of the write-up.")
    return L + ["", " ".join(obs), ""]


def early_commit(met: dict) -> list[str]:
    c = met["conditions"]
    rows = [r for r in c.values() if group_of(r["condition"]) == "early" and r["status"] == "done"]
    if not rows:
        return ["No early-commit row is scored yet.", ""]
    L = [("| condition | branch | prefix | weighted F1 [95 % CI] | full-clip weighted F1 | label agreement with the full clip | "
         "clips longer than the prefix | their prefix weighted F1 | their full-clip weighted F1 | their label agreement |"),
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cut = r["cut_rows"]
        L.append(f"| `{r['condition']}` | {r['branch']} | {r['prefix_s']:g} s | {wci(r)} | {pct(r['full_clip']['weighted_f1'])} | "
                 f"{pct(r['label_agreement_with_full_clip'])} | {cut['n']} | {pct(cut['prefix']['weighted_f1'])} | "
                 f"{pct(cut['full_clip']['weighted_f1'])} | {pct(cut['label_agreement_with_full_clip'])} |")
    L += ["", ("These are offline prefix experiments, not the service's provisional states. A clip that is not longer than the prefix keeps "
          "its full-clip values. The fused rows take the text posterior from the prefix transcript, and their full-clip reference is the "
          "stacked fusion on the full transcript without the ASR gate. Agreement is in percent of clips. "
          "[results/RESULTS.md](results/RESULTS.md) also scores the audio rows with the inputs a provisional state uses.")]
    obs = []
    for r in rows:
        cut = r["cut_rows"]
        a, b = cut["prefix"]["weighted_f1"], cut["full_clip"]["weighted_f1"]
        obs.append(f"On the {cut['n']} clips longer than {r['prefix_s']:g} s, the {r['branch']} label from the prefix scores {pct(a)} "
                   f"against {pct(b)} on the full clip ({pts(a, b)} points {side(a, b)}) and matches the full-clip label on "
                   f"{pct(cut['label_agreement_with_full_clip'])} % of them.")
    return L + ["", " ".join(obs), ""]


def sentiment_notes(met: dict) -> list[str]:
    c = met["conditions"]
    pub = {p["row"]: p["weighted_f1"] for p in met.get("published", [])}
    out = []
    paper = TEXT_ROWS["s_drnn_t"] in pub and TEXT_ROWS["s_drnn_ta"] in pub
    ref = (f"{pub[TEXT_ROWS['s_drnn_t']]:.2f} for DialogueRNN on text and {pub[TEXT_ROWS['s_drnn_ta']]:.2f} on text and audio "
           "(MELD paper, Table 13)") if paper else ""
    if done(c, "sentiment_text_only") and done(c, "sentiment_fused"):
        t, f = c["sentiment_text_only"], c["sentiment_fused"]
        if paper:
            out.append(f"The text-only sentiment head scores {pct(t['weighted_f1'])} and the fused one {pct(f['weighted_f1'])}, next to {ref}.")
        (tlo, thi), (flo, fhi) = ci(t), ci(f)
        out.append(f"The fused head scores {pts(f['weighted_f1'], t['weighted_f1'])} points {side(f['weighted_f1'], t['weighted_f1'])} "
                   "the text-only head, and the two intervals " + ("overlap." if flo <= thi and tlo <= fhi else "do not overlap."))
    else:
        for name, label in (("sentiment_text_only", "text-only"), ("sentiment_fused", "fused")):
            if done(c, name) and paper:
                out.append(f"The {label} sentiment head scores {pct(c[name]['weighted_f1'])}, next to {ref}.")
    for name in ("sentiment_text_only", "sentiment_fused"):
        if name in c and not done(c, name):
            out.append(f"`{name}` is not scored in this run (see the pending rows).")
    return [" ".join(out), ""] if out else []


def per_class(met: dict) -> list[str]:
    rows = [r for r in met["conditions"].values()
            if r["task"] == "emotion" and r["status"] == "done" and group_of(r["condition"]) != "disagreement"]
    if not rows:
        return []
    L = ["| condition | " + " | ".join(labels.EMOTIONS) + " |", "|---|" + "---|" * len(labels.EMOTIONS)]
    for r in rows:
        L.append(f"| `{r['condition']}` | " + " | ".join(pct(r["per_class_f1"][c]) for c in labels.EMOTIONS) + " |")
    L.append("| DialogueRNN text+audio (MELD paper, Table 11) | " + " | ".join(f"{DialogueRNN_PER_CLASS[c]:.2f}" for c in labels.EMOTIONS) + " |")
    support = [sum(row) for row in rows[0]["confusion"]]
    L.append("| test utterances in the class | " + " | ".join(str(n) for n in support) + " |")
    L += ["", "The disagreement subset covers only part of test, so its per-class F1 is in [results/RESULTS.md](results/RESULTS.md)."]
    c = met["conditions"]
    obs = []
    if done(c, "text_best_k"):
        tb = c["text_best_k"]
        f1 = tb["per_class_f1"]
        low2 = sorted(labels.EMOTIONS, key=lambda e: f1[e])[:2]
        rare2 = sorted(labels.EMOTIONS, key=lambda e: support[labels.EMOTIONS.index(e)])[:2]
        n2 = [support[labels.EMOTIONS.index(e)] for e in low2]
        obs.append(f"The text head's macro F1 is {pct(tb['macro_f1'])}, and its two lowest per-class scores are {low2[0]} ({pct(f1[low2[0]])}) "
                   f"and {low2[1]} ({pct(f1[low2[1]])}), "
                   + ("the two classes with the fewest test utterances " if set(low2) == set(rare2) else "classes with ")
                   + f"({n2[0]} and {n2[1]}).")
        if done(c, "fused_stacked"):
            fs = c["fused_stacked"]
            worse = [e for e in labels.EMOTIONS if fs["per_class_f1"][e] < f1[e]]
            s = (f"The stacked fusion's macro F1 is {pct(fs['macro_f1'])}, {pts(fs['macro_f1'], tb['macro_f1'])} points "
                 f"{side(fs['macro_f1'], tb['macro_f1'])} the text head")
            if worse:
                items = [f"{e} ({pct(fs['per_class_f1'][e])} against {pct(f1[e])})" for e in worse]
                s += ", and it scores lower on " + (", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0])
            obs.append(s + ".")
    return L + ["", " ".join(obs), ""] if obs else L + [""]


def published_notes(met: dict) -> list[str]:
    c = met["conditions"]
    pub = {p["row"]: p["weighted_f1"] for p in met.get("published", [])}
    out = []
    if done(c, "text_best_k"):
        tb = c["text_best_k"]
        lo, hi = ci(tb)
        text_pub = [pub[v] for k, v in TEXT_ROWS.items() if v in pub and k in ("cnn", "bclstm", "drnn")]
        if text_pub:
            best = max(text_pub)
            rel = ("below the lower end of" if 100 * lo > best else "above the upper end of" if 100 * hi < best else "inside")
            out.append(f"The best text-only row of the MELD paper, {best:.2f} (Table 11), is {rel} the text head's interval.")
        if TEXT_ROWS["emoberta"] in pub:
            e = pub[TEXT_ROWS["emoberta"]]
            rel = "above the upper end of" if e > 100 * hi else ("below the lower end of" if e < 100 * lo else "inside")
            out.append(f"Fine-tuned RoBERTa-large without context reaches {e:.2f} in EmoBERTa (Table 2), {rel} that interval.")
    if done(c, "fused_stacked") and TEXT_ROWS["drnn_ta"] in pub:
        fs = c["fused_stacked"]
        lo, hi = ci(fs)
        e = pub[TEXT_ROWS["drnn_ta"]]
        rel = "above the upper end of" if e > 100 * hi else ("below the lower end of" if e < 100 * lo else "inside")
        out.append(f"DialogueRNN on text and audio, {e:.2f} (Table 11), is {rel} the stacked fusion's interval.")
    sv = next((v for k, v in pub.items() if k.startswith(SENSEVOICE_ROW) and v is not None), None)
    if done(c, "audio_raw_tag") and sv is not None:
        r = c["audio_raw_tag"]
        lo, hi = ci(r)
        rel = "above the upper end of" if sv > 100 * hi else ("below the lower end of" if sv < 100 * lo else "inside")
        out.append(f"The FunAudioLLM report gives SenseVoice-S {sv:.2f} weighted F1 on MELD (Table 8), {rel} the interval of the raw "
                   f"SenseVoice tag measured here ({wci(r)}).")
        if "abstain_rate" in r:
            out.append(f"On {pct(r['abstain_rate'])} % of the test clips, more than half of the tag's emotion mass is on OTHER or EMO_UNKNOWN.")
        out.append("The report does not state its MELD protocol (the split, the class mapping, how OTHER and unknown are scored) "
                   "in enough detail to reproduce it.")
    return [" ".join(out), ""] if out else []


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
    af = man.get("audio_fusion", {})
    L = [(f"- Text head: grid point `{sel['grid_point']}` (k = {sel['k']}, class weights `{sel['class_weight_scheme']}`), "
         f"dev weighted F1 {pct(sel['dev']['weighted_f1'])}, temperature {sel['calibration_dev']['temperature']}.")]
    trained = af.get("status") == "trained"
    if trained:
        var = af["audio_cal"]["variant"]
        other = "B" if var == "A" else "A"
        a, b = af[f"audio_cal_{var}"], af.get(f"audio_cal_{other}")
        L.append(f"- Audio calibrator: variant {var} ({a['input_dim']} inputs, C = {a['C']}, class weights `{a['class_weight_scheme']}`), "
                 f"dev weighted F1 {pct(a['dev']['weighted_f1'])}"
                 + (f" against {pct(b['dev']['weighted_f1'])} for variant {other} ({b['input_dim']} inputs)" if b else "")
                 + f", temperature {a['calibration_dev']['temperature']}.")
        fu, poe = af["fusion"], af.get("poe", {})
        L.append(f"- Fusion head: C = {fu['C']}, class weights `{fu['class_weight_scheme']}`, trained on out-of-fold expert posteriors "
                 f"({af['cross_fit']['folds']} folds by dialogue), dev weighted F1 {pct(fu['dev']['weighted_f1'])}, "
                 f"temperature {fu['calibration_dev']['temperature']}.")
        w = poe.get("weight")
        pw = poe.get("dev_scores", {}).get(str(w))
        if pw:
            rule = ("The stacked fusion scores higher on dev, so it stays deployed (decision rule in [docs/decisions.md](docs/decisions.md), Fusion)."
                    if fu["dev"]["weighted_f1"] > pw["weighted_f1"] else
                    "The product of experts matches or beats the stacked fusion on dev, so the decision rule in "
                    "[docs/decisions.md](docs/decisions.md) (Fusion) calls for deploying it.")
            L.append(f"- Product of experts: weight {w} from the grid {', '.join(poe['dev_scores'])}, dev weighted F1 "
                     f"{pct(pw['weighted_f1'])}. {rule}")
    src = "the fusion head's" if trained and af["fusion"].get("thresholds", {}).get("low_conf") == thr["low_conf"] else "the text head's"
    th_text = sel.get("thresholds", {})
    L.append(f"- Selector thresholds in use: low_conf {thr['low_conf']}, high_conf {thr['high_conf']}, from {src} calibrated dev predictions"
             + (f" (the text head alone gives {th_text['low_conf']} and {th_text['high_conf']})." if trained and th_text else "."))
    if thr.get("jsd_flag") is None:
        L.append(f"- Discrepancy flag threshold: not chosen yet ({man.get('thresholds_pending', {}).get('jsd_flag', 'pending')}). "
                 "The service uses its `--jsd-flag` default until then.")
    else:
        q = man.get("config", {}).get("thresholds", {}).get("jsd_flag", {}).get("quantile")
        L.append(f"- Discrepancy flag threshold: {thr['jsd_flag']}"
                 + (f", the {q} quantile of the dev JSD between the text posterior and the raw SenseVoice posterior." if q is not None else "."))
    if sent:
        L.append(f"- Sentiment head ({sent.get('variant')}): C = {sent.get('C')}, class weights `{sent.get('class_weight_scheme')}`, "
                 f"dev weighted F1 {pct(sent['dev']['weighted_f1'])}, temperature {sent['calibration_dev']['temperature']}.")
    if trained:
        rows = af.get("rows", {})
        dropped = [f"{v['dropped']} in {s} (`{', '.join(v['dropped_keys'])}`)" for s, v in rows.items() if v.get("dropped")]
        if dropped:
            L.append("- CSV rows without an audio clip, left out of the audio stage: " + " and ".join(dropped) + ".")
    ab = man.get("ablations", {})
    if ab.get("variants") and ab.get("reference"):
        L.append("- Ablation heads: dev weighted F1 of the fusion " + f"{pct(ab['reference']['fusion']['dev']['weighted_f1'])} with every input, "
                 + ", ".join(f"{pct(v['fusion']['dev']['weighted_f1'])} without {' and '.join(v['dropped'])}" for v in ab["variants"].values())
                 + ".")
    L.append(f"- Audio and fusion stage: {af.get('status')}.")
    return L + [""]


def observations(met: dict) -> list[str]:
    """three to six sentences, one claim each, computed from the numbers; a sentence whose inputs are missing is skipped."""
    c = met["conditions"]
    pub = {p["row"]: p["weighted_f1"] for p in met.get("published", [])}
    out = []
    if done(c, "text_best_k"):
        tb = c["text_best_k"]
        lo, hi = ci(tb)
        s = f"The deployed text head scores {pct(tb['weighted_f1'])} weighted F1 on test (95 % CI [{pct(lo)}, {pct(hi)}])"
        out.append(s + (f", against {pct(c['majority']['weighted_f1'])} for the majority floor." if done(c, "majority") else "."))
    if done(c, "text_best_k") and done(c, "fused_stacked"):
        tb, fs = c["text_best_k"], c["fused_stacked"]
        (tlo, thi), (flo, fhi) = ci(tb), ci(fs)
        overlap = flo <= thi and tlo <= fhi
        s = (f"The deployed stacked fusion scores {wci(fs)}, {pts(fs['weighted_f1'], tb['weighted_f1'])} points "
             f"{side(fs['weighted_f1'], tb['weighted_f1'])} the text head")
        if overlap and inside(tb["weighted_f1"], fs):
            s += ", and the two intervals overlap, with the text head's score inside the fusion's interval."
        elif overlap:
            s += ", and the two intervals overlap."
        else:
            s += ", and the two intervals do not overlap."
        out.append(s)
    if done(c, "disagreement_subset"):
        r = c["disagreement_subset"]
        ft, tt = r["subset"]["fused_stacked"]["weighted_f1"], r["subset"]["text"]["weighted_f1"]
        s = (f"On the {r['n_subset']} test utterances whose JSD is above the dev threshold ({pct(r['share'])} %), the fusion scores "
             f"{pct(ft)} against {pct(tt)} for the text head alone")
        out.append(s + (", so it does not help where the words and the voice disagree." if ft <= tt else f", {pts(ft, tt)} points above."))
    if done(c, "live_asr_text_only") and done(c, "text_best_k"):
        a, g = c["live_asr_text_only"], c["text_best_k"]
        s = (f"On SenseVoice transcripts (corpus WER {a['wer']}), the text head scores {pct(a['weighted_f1'])} instead of "
             f"{pct(g['weighted_f1'])} on gold text")
        if done(c, "live_asr_text") and done(c, "fused_stacked"):
            s += f", and the fusion {pct(c['live_asr_text']['weighted_f1'])} instead of {pct(c['fused_stacked']['weighted_f1'])}"
        out.append(s + ".")
    if done(c, "ablation_no_events_no_prosody") and done(c, "fused_stacked"):
        x, ref = c["ablation_no_events_no_prosody"]["weighted_f1"], c["fused_stacked"]
        s = (f"Without the event and prosody inputs the fusion scores {pct(x)}, {pts(x, ref['weighted_f1'])} points "
             f"{side(x, ref['weighted_f1'])} the deployed one")
        out.append(s + (", inside its interval, so those inputs add nothing measurable here." if inside(x, ref) else ", outside its interval."))
    sv = next((v for k, v in pub.items() if k.startswith(SENSEVOICE_ROW) and v is not None), None)
    if done(c, "audio_raw_tag") and sv is not None:
        r = c["audio_raw_tag"]
        s = f"The raw SenseVoice tag scores {pct(r['weighted_f1'])} against the {sv:.2f} of the FunAudioLLM report (Table 8)"
        if "abstain_rate" in r:
            s += f", and on {pct(r['abstain_rate'])} % of the test clips most of its emotion mass is on OTHER or EMO_UNKNOWN"
        out.append(s + ".")
    if len(out) < 6 and done(c, "sentiment_text_only") and TEXT_ROWS["s_drnn_t"] in pub and TEXT_ROWS["s_drnn_ta"] in pub:
        out.append(f"The text-only sentiment head scores {pct(c['sentiment_text_only']['weighted_f1'])}, next to "
                   f"{pub[TEXT_ROWS['s_drnn_t']]:.2f} for DialogueRNN on text and {pub[TEXT_ROWS['s_drnn_ta']]:.2f} on text and audio "
                   "(MELD paper, Table 13).")
    n_pending = sum(r["status"] != "done" for r in c.values())
    if len(out) < 6 and n_pending:
        out.append(f"Of the {len(c)} conditions, {n_pending} are pending, and the pending table says what each one needs.")
    if len(out) < 6 and done(c, "text_best_k") and done(c, "text_runtime_regime"):
        out.append("Restricting the context to the same speaker changes the score by "
                   f"{pts(c['text_runtime_regime']['weighted_f1'], c['text_best_k']['weighted_f1'])} points.")
    return out[:6]


# -- extension: three inputs, commit policy, streamed reply (appended sections) ----

def above(lo_hi) -> str:
    return "yes" if lo_hi[0] > 0 else "no"


ACRONYMS = {"fer": "FER", "sensevoice": "SenseVoice", "asr": "ASR", "jsd": "JSD", "mlp": "MLP"}


def acr(text: str) -> str:
    """the acronyms and model names that the result files write in lower case, as the prose writes them."""
    return re.sub(r"\b(" + "|".join(ACRONYMS) + r")\b", lambda m: ACRONYMS[m.group(1)], text)


def three_inputs(vm: dict, met: dict, path: str) -> list[str]:
    """RESULTS.md section from results/metrics_vision.json (tools/evaluate_vision.py)."""
    c, comp, cl, v, dev = vm["conditions"], vm["comparisons"], vm["claim"], vm["vision"], vm["dev"]
    th = v.get("thresholds") or {}
    L = [(f"Rendered from `{path}` (written by `{vm['command']}`). The speaker rule (`{v['rule']}`), the face calibrator (variant "
          f"{v['variant']}), fusion3 (a logistic regression over 29 features built from the text, audio and face posteriors) and its thresholds were chosen on dev by "
          "`tools/train_vision.py` (`heads/manifest.json`, section `vision`). "
          f"A face was found in {pct(vm['test_face_coverage']['all_rows'])} % of the {vm['n_test_rows']} test clips. In `vision_raw_fer` and "
          "`vision_cal`, a clip without a face gets the uniform posterior and the label neutral. The test protocol and the intervals are those of the "
          f"main tables. The paired differences resample test dialogues ({vm['n_resamples']} resamples, seed {vm['seed']}) and score both "
          "conditions on each resample."), ""]
    obs = []
    if done(c, "vision_cal") and done(c, "vision_raw_fer") and done(c, "text_best_k"):
        obs.append(f"The face alone scores {pct(c['vision_cal']['weighted_f1'])} weighted F1 with the calibrator and "
                   f"{pct(c['vision_raw_fer']['weighted_f1'])} as the raw FER label, against {pct(c['text_best_k']['weighted_f1'])} for the text head.")
    g, lv = comp.get("gold_all", {}), comp.get("live_all", {})
    if done(c, "fused3_stacked") and done(c, "fused_stacked") and "ci95" in g:
        obs.append(f"Fusion3 scores {wci(c['fused3_stacked'])} on gold text against {pct(c['fused_stacked']['weighted_f1'])} for the two-way "
                   f"fusion, a paired difference of {pct(g['diff']['weighted_f1'])} points [{pct(g['ci95']['weighted_f1'][0])}, "
                   f"{pct(g['ci95']['weighted_f1'][1])}].")
    if done(c, "live_asr_fused3") and done(c, "live_asr_text") and "ci95" in lv:
        obs.append(f"On SenseVoice transcripts it scores {pct(c['live_asr_fused3']['weighted_f1'])} against "
                   f"{pct(c['live_asr_text']['weighted_f1'])}, a paired difference of {pct(lv['diff']['weighted_f1'])} points "
                   f"[{pct(lv['ci95']['weighted_f1'][0])}, {pct(lv['ci95']['weighted_f1'][1])}].")
    if cl.get("gold_all_ci95_weighted_f1"):
        obs.append("The claim rule was fixed before the run: the face adds to text and audio only if the paired 95 % interval of fusion3 "
                   "minus the two-way fusion on gold text lies above zero. "
                   + ("It does, so by this rule the face adds measured value here." if cl["face_adds"] else
                      "It does not, so the gain is not shown."))
    fl = (vm.get("label_flips") or {}).get("gold", {})
    if "changed" in fl:
        other = fl["changed"] - fl["wrong_to_right"] - fl["right_to_wrong"]
        obs.append(f"Fusion3 changes {fl['changed']} of the {fl['n']} gold-text labels of the two-way fusion: {fl['wrong_to_right']} from wrong "
                   f"to right, {fl['right_to_wrong']} from right to wrong and {other} from one wrong label to another.")
    L += [" ".join(obs), ""] if obs else []
    L += ["| condition | what it is | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE | face coverage |", "|---|---|---|---|---|---|---|---|"]
    for name, r in c.items():
        if r["status"] != "done":
            L.append(f"| `{name}` | {acr(what(r))} | | {r['status']} | | | | |")
            continue
        L.append(f"| `{name}` | {acr(what(r))} | {r['n']} | {wci(r)} | {pct(r['macro_f1'])} | {pct(r['accuracy'])} | {pct(r['ece'])} | "
                 f"{pct(r.get('face_coverage'))} |")
    if done(c, "sentiment3") and done(met["conditions"], "sentiment_fused"):
        L += ["", (f"`sentiment3` is a 3-class task. It scores {pct(c['sentiment3']['weighted_f1'])} against "
                   f"{pct(met['conditions']['sentiment_fused']['weighted_f1'])} for the deployed two-way sentiment head (`sentiment_fused`).")]
    L += ["", "### Paired differences (weighted F1, a minus b, in points)", "",
          "| comparison | a | b | rows | n | difference | 95 % CI | interval above zero |", "|---|---|---|---|---|---|---|---|"]
    for name, d in comp.items():
        if "ci95" not in d:
            L.append(f"| `{name}` | `{d['a']}` | `{d['b']}` | {d['rows']} | | pending | | |")
            continue
        lo, hi = d["ci95"]["weighted_f1"]
        L.append(f"| `{name}` | `{d['a']}` | `{d['b']}` | {d['rows']} | {d['n']} | {pct(d['diff']['weighted_f1'])} | [{pct(lo)}, {pct(hi)}] | "
                 f"{above((lo, hi))} |")
    flips = vm.get("label_flips") or {}
    if flips:
        L += ["", "### Label flips (fusion3 against the two-way fusion)", "",
              "| text input | n | labels changed | wrong to right | right to wrong | wrong to another wrong label |", "|---|---|---|---|---|---|"]
        for f in flips.values():
            if "changed" not in f:
                L.append(f"| {acr(f['text'])} | | pending | | | |")
                continue
            L.append(f"| {acr(f['text'])} | {f['n']} | {f['changed']} | {f['wrong_to_right']} | {f['right_to_wrong']} | "
                     f"{f['changed'] - f['wrong_to_right'] - f['right_to_wrong']} |")
    r = c.get("face_disagreement_subset", {})
    if r.get("status") == "done":
        L += ["", "### Face-disagreement subset", "",
              "| rows | n | text weighted F1 | two-way weighted F1 | fusion3 weighted F1 | text accuracy | two-way accuracy | fusion3 accuracy |",
              "|---|---|---|---|---|---|---|---|"]
        for part in ("subset", "complement"):
            q = r[part]
            L.append(f"| {part} | {q['n']} | " + " | ".join(pct(q[b]["weighted_f1"]) for b in ("text_best_k", "fused_stacked", "fused3_stacked"))
                     + " | " + " | ".join(pct(q[b]["accuracy"]) for b in ("text_best_k", "fused_stacked", "fused3_stacked")) + " |")
        s = r["subset"]
        text = (f"The subset holds the {r['n_subset']} of {r['n_face_rows']} test clips with a face ({pct(r['share_of_face_rows'])} %) whose JSD "
                f"between the gold-text posterior and the raw face posterior is above {r['jsd_face_flag']} (`vision.jsd_face_flag`, chosen on dev). "
                f"There, fusion3 scores {pct(s['fused3_stacked']['weighted_f1'])} against {pct(s['fused_stacked']['weighted_f1'])} for the two-way "
                f"fusion and {pct(s['text_best_k']['weighted_f1'])} for the text head.")
        pairs = [(who, comp[key]) for key, who in (("gold_face_disagreement", "the two-way fusion"),
                                                   ("gold_face_disagreement_vs_text", "the text head")) if "ci95" in comp.get(key, {})]

        def diff(d: dict) -> str:
            lo, hi = d["ci95"]["weighted_f1"]
            return f"{pct(d['diff']['weighted_f1'])} points [{pct(lo)}, {pct(hi)}]"

        def side_of_zero(d: dict) -> str:
            lo, hi = d["ci95"]["weighted_f1"]
            return "lies above zero" if lo > 0 else "lies below zero" if hi < 0 else "includes zero"

        if len(pairs) == 2 and all(side_of_zero(d) == "includes zero" for _, d in pairs):
            text += (f" The paired differences to {pairs[0][0]} ({diff(pairs[0][1])}) and to {pairs[1][0]} ({diff(pairs[1][1])}) both have "
                     "intervals that include zero.")
        else:
            for who, d in pairs:
                text += f" The paired difference to {who} is {diff(d)}, and its interval {side_of_zero(d)}."
        L += ["", text]
    if dev.get("rules"):
        L += ["", "### Choices made on dev (`heads/manifest.json`, section `vision`)", "",
              "| speaker rule | dev weighted F1, all rows | dev weighted F1, rows with a face | fallback rate on dev (%) | chosen |", "|---|---|---|---|---|"]
        for rule, q in dev["rules"].items():
            fb = (q.get("fallback_rate") or {}).get("dev")
            L.append(f"| `{rule}` | {pct(q['dev']['weighted_f1'])} | {pct(q['dev_face_rows']['weighted_f1'])} | {pct(fb)} | "
                     f"{'yes' if rule == v.get('rule') else ''} |")
        L.append("")
        dis = (v.get("rule_disagreement") or {}).get("dev") or {}
        if dis:
            L.append(f"- Of the {dis['n_face_clips']} dev clips with a face, {pct(dis['multi_track_share'])} % have more than one track, and on "
                     f"{pct(dis['any_rule_differs'])} % the four rules do not all choose the same track.")
        cal = dev.get("calibrator") or {}
        if cal.get("A") and cal.get("B"):
            other = "A" if v["variant"] == "B" else "B"
            L.append(f"- Face calibrator: variant {v['variant']}, dev weighted F1 {pct(cal[v['variant']]['weighted_f1'])} against "
                     f"{pct(cal[other]['weighted_f1'])} for variant {other}.")
        if dev.get("fusion3") and dev.get("two_way"):
            L.append(f"- Fusion3: dev weighted F1 {pct(dev['fusion3']['weighted_f1'])} against {pct(dev['two_way']['weighted_f1'])} for the "
                     "two-way fusion.")
        if dev.get("sentiment3"):
            L.append(f"- Sentiment3: dev weighted F1 {pct(dev['sentiment3']['weighted_f1'])}.")
        if th:
            L.append(f"- Selector thresholds in the vision mode: low_conf {th.get('low_conf')}, high_conf {th.get('high_conf')}. "
                     f"Words-face flag: JSD above {v.get('jsd_face_flag')}.")
    return L + [""]


def commit_policy(cp: dict, path: str) -> list[str]:
    """RESULTS.md section from results/commit_policy.json (tools/train_commit_policy.py)."""
    costs = [str(c) for c in cp["costs"]]
    dc = str(cp["deploy_c"])
    tr = cp["inputs"]["train"]
    L = [(f"Rendered from `{path}` (written by `{cp['command']}`). At each full-window provisional of an utterance "
          f"(t = {', '.join(str(t) for t in cp['grids']['fixed_T'][:-1])} and {cp['grids']['fixed_T'][-1]} s while the segment is open), "
          "the policy commits the fused label or waits. Without a commit, the label of the endpoint is committed. The reward is one for a "
          "correct label minus c times the commit time in seconds. The policy is trained by fitted Q-iteration on the train traces "
          f"({tr['ticks']} ticks of {tr['clips']} clips). The Q_wait model and the parameters of the three baselines were chosen on dev for "
          f"each c, and test was read once per method and c. The deployed policy uses c = {dc}.")]
    t = cp.get("test")
    obs = []
    if t and dc in t:
        m = t[dc]["methods"]
        f, e = m["fqi"], m["endpoint"]
        any_pair = next(iter(t[dc]["paired_fqi_minus"].values()))
        L[0] += (f" The intervals are 95 % bootstrap intervals over test dialogues ({any_pair['n_resamples']} resamples, seed "
                 f"{any_pair['seed']}).")
        obs.append(f"At c = {dc} the policy commits {pct(f['early_share'])} % of the test labels before the endpoint, at a mean of "
                   f"{f['mean_commit_s']:.3f} s after the segment start against {e['mean_commit_s']:.3f} s at the endpoint. Its accuracy is "
                   f"{pct(f['accuracy'])} against {pct(e['accuracy'])} at the endpoint, and its weighted F1 {pct(f['weighted_f1'])} against "
                   f"{pct(e['weighted_f1'])}.")
        fq = cp["train"]["costs"][dc]["fqi"]
        if not fq.get("converged"):
            obs.append(f"Its Q_wait model ({acr(fq['kind'])}) stopped at the cap of {cp['grids']['max_iterations']} iterations before the emit "
                       "decisions on the train ticks stopped changing.")
        names = {"endpoint": "the endpoint", "fixed": "the fixed time", "threshold": "the threshold"}
        parts = []
        for b in t[dc]["rl_adds_value_over"]:
            yes = [c for c in costs if t[c]["rl_adds_value_over"][b]]
            where = ("at every cost" if len(yes) == len(costs) else "at no cost" if not yes else
                     "at " + " and ".join(f"c = {c}" for c in yes) + " only")
            parts.append(f"over {names.get(b, b)} {where}")
        missing = [names.get(b, b) for b, v in t[dc]["rl_adds_value_over"].items() if not v]
        obs.append("The claim rule was fixed before the run: RL adds measured value over a baseline at a cost c only where the paired 95 % "
                   "interval of the test reward difference lies above zero. By that rule the policy adds measured value "
                   + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + ". "
                   + (f"At the deployed c = {dc} the gain over {' and over '.join(missing)} is not shown." if missing else
                      f"At the deployed c = {dc} it adds measured value over every baseline."))
        sens = {c: t[c].get("sensitivity_tick_latency") for c in costs}
        if all(sens.values()):
            changed = [f"c = {c} against the {b}" for c in costs for b, d in sens[c]["paired_fqi_minus"].items()
                       if d["above_zero"] != t[c]["rl_adds_value_over"][b]]
            obs.append(f"Charging every tick commit {sens[costs[0]]['latency_s']} s more, as the endpoint commit is charged for the computation "
                       "of the final, "
                       + ("changes none of these outcomes." if not changed else "changes the outcome at " + ", ".join(changed) + "."))
    L += ["", " ".join(obs), ""] if obs else [""]
    L += ["### Choices made on dev", "",
          "| c | Q_wait model | iterations | dev reward of the policy | fixed T | dev reward fixed | tau | dev reward threshold | dev reward endpoint |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, rc in cp["train"]["costs"].items():
        dv = cp["dev"][c]["methods"]
        f, b = rc["fqi"], rc["baselines"]
        L.append(f"| {c} | {acr(f['kind'])}, alpha {f['alpha']:g} | {f['n_iterations']} | {dv['fqi']['mean_reward']:.4f} | {b['fixed']['param']} s | "
                 f"{dv['fixed']['mean_reward']:.4f} | {b['threshold']['param']} | {dv['threshold']['mean_reward']:.4f} | "
                 f"{dv['endpoint']['mean_reward']:.4f} |")
    if t:
        L += ["", "### Test, every method and cost", "",
              "| c | method | mean reward [95 % CI] | accuracy | weighted F1 | mean commit s | committed before the endpoint |",
              "|---|---|---|---|---|---|---|"]
        for c in costs:
            for name, s in t[c]["methods"].items():
                lo, hi = s["mean_reward_ci95"]
                label = ("policy" if name == "fqi" else name if s["param"] is None else
                         f"fixed time ({s['param']} s)" if name == "fixed" else f"{name} ({s['param']})")
                L.append(f"| {c} | {label} | {s['mean_reward']:.4f} [{lo:.4f}, {hi:.4f}] | {pct(s['accuracy'])} | {pct(s['weighted_f1'])} | "
                         f"{s['mean_commit_s']:.3f} | {pct(s['early_share'])} % |")
        L += ["", "### Test, policy minus each baseline (paired bootstrap over dialogues)", "",
              "| c | baseline | reward difference | 95 % CI | RL adds measured value (claim rule) |", "|---|---|---|---|---|"]
        for c in costs:
            for b, d in t[c]["paired_fqi_minus"].items():
                L.append(f"| {c} | {b} | {d['diff']:.4f} | [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | "
                         f"{'yes' if t[c]['rl_adds_value_over'][b] else 'no'} |")
        L.append("")
        L.append("The full tables, the P(correct) grid and the sensitivity rows are in [results/commit_policy.md](results/commit_policy.md).")
    return L + [""]


def deployed_choice(gens: list[dict]) -> dict | None:
    """the choice rule over the files of several bench runs: of each run's choice that meets the targets, the
    largest model's (each run applies the same-model int8 / q4 rule to its own files)."""
    ok = [g["choice"] for g in gens if (g.get("choice") or {}).get("meets_targets")]
    if not ok:
        return None
    order = {cand["file"]: cand["params_order"] for g in gens for cand in g["choice"].get("candidates", [])}
    return max(ok, key=lambda ch: order.get(ch["deployed"], -1))


# the choice rule as first worded (docstring of tools/bench_generator.py): a decode rate of at least 15 tokens per
# second, read here as the final-reply rate (decode_tokens_per_s), and q4 only when it is faster by more than 20 %,
# read here on the full-reply p50
ORIGINAL_MIN_TOKENS_PER_S, ORIGINAL_Q4_FASTER = 15, 0.20


def original_wording_choice(gens: list[dict]) -> str | None:
    """a sentence on the file that the first wording of the choice rule picks from the same runs, or None."""
    rows = []
    for g in gens:
        ch = g.get("choice") or {}
        rule, th = ch.get("rule") or {}, str((ch.get("rule") or {}).get("threads"))
        for cand in ch.get("candidates", []):
            sm = ((g.get("results") or {}).get(cand["file"]) or {}).get(th, {}).get("summary") or {}
            if sm.get("decode_tokens_per_s") is None:
                return None
            rows.append({**cand, "tok_s": sm["decode_tokens_per_s"], "threads": th,
                         "ok": cand["first_token_p95"] <= rule["first_token_p95_max_ms"] and sm["decode_tokens_per_s"] >= ORIGINAL_MIN_TOKENS_PER_S})
    ok = [r for r in rows if r["ok"]]
    deployed = deployed_choice(gens)
    if not ok or deployed is None:
        return None
    top = [r for r in ok if r["params_order"] == max(q["params_order"] for q in ok)]
    i8, q4 = (next((r for r in top if kind in r["onnx"]), None) for kind in ("int8", "q4"))
    pick = i8 or top[0]
    if (i8 and q4 and i8["full_reply_p50"] / q4["full_reply_p50"] > 1 + ORIGINAL_Q4_FASTER
            and q4["check_pass_rate"] >= i8["check_pass_rate"]):
        pick = q4
    text = (f"The original wording asked for {ORIGINAL_MIN_TOKENS_PER_S} tokens per s and a q4 file faster by more than "
            f"{ORIGINAL_Q4_FASTER * 100:.0f} %")
    if pick["file"] != deployed["deployed"]:
        return text + f", and it would pick `{pick['file']}`."

    def kind(r: dict) -> str:
        return r["onnx"].removeprefix("model_").removesuffix(".onnx")

    models = list(dict.fromkeys(r["model"] for r in rows))
    rates = [(" and ".join(kind(r) for r in rows if r["model"] == m), m,
              " and ".join(f"{r['tok_s']}" for r in rows if r["model"] == m)) for m in models]
    text += (f", and it picks the same file: at {rows[0]['threads']} threads the {rates[0][0]} files of {rates[0][1]} write {rates[0][2]} "
             "final-reply tokens per s" + "".join(f", those of {m} {v}" for _, m, v in rates[1:]))
    if i8 and q4 and pick is i8:
        p50s = f"({q4['full_reply_p50']} against {i8['full_reply_p50']} ms)"
        text += (f", and the q4 file of {q4['model']} has the longer full-reply p50 {p50s}" if q4["full_reply_p50"] >= i8["full_reply_p50"] else
                 f", and the q4 file of {q4['model']} is not faster by more than {ORIGINAL_Q4_FASTER * 100:.0f} % on the full-reply p50 {p50s}")
    return text + "."


def streamed_reply(gens: list[tuple[str, dict]], replay: tuple[str, dict] | None) -> list[str]:
    """RESULTS.md section from the generator bench files (tools/bench_generator.py)."""
    g0 = gens[0][1]
    st = g0.get("states") or {}
    rule = (g0.get("choice") or {}).get("rule") or {}
    L = [(f"Rendered from {' and '.join(f'`{p}`' for p, _ in gens)} (written by `tools/bench_generator.py`, commands in the files). "
          f"Every run replies to the same {st.get('n')} test states, {st.get('n_gold')} with gold text and {st.get('n_live')} with SenseVoice "
          f"transcripts, drawn from `{st.get('source')}` with seed {st.get('seed')}. Decoding is greedy, with at most {g0.get('max_new_tokens')} "
          "new tokens. The decode rate counts every forward pass after the first token, and the check pass rate is the share of replies that "
          "pass `check_reply` (length, register, emotion words, prompt echo)."), "",
         "| file | threads | first token p50 / p95 ms | first word p50 / p95 ms | full reply p50 / p95 ms | decode passes per s | words per reply | check pass rate |",
         "|---|---|---|---|---|---|---|---|"]
    for _, g in gens:
        for f, by_t in g["results"].items():
            for th in sorted((k for k in by_t if k.isdigit()), key=int):
                sm = by_t[th]["summary"]
                L.append(f"| `{f}` | {th} | {sm['first_token_ms']['p50']} / {sm['first_token_ms']['p95']} | {sm['first_word_ms']['p50']} / "
                         f"{sm['first_word_ms']['p95']} | {sm['full_reply_ms']['p50']} / {sm['full_reply_ms']['p95']} | {sm['decode_passes_per_s']} | "
                         f"{sm['words_per_reply']['mean']} | {sm['check_pass_rate']} |")
    text = [(f"The choice rule was fixed before the bench on {st.get('n')} states. It deploys the largest model with a file whose first-token "
             f"p95 is at most {rule.get('first_token_p95_max_ms')} ms and whose decode rate is at least {rule.get('decode_min_passes_per_s')} "
             f"passes per s at {rule.get('threads')} threads. Among the files of one model it keeps int8 unless q4 is faster by more than "
             f"{rule.get('q4_same_length_speedup_over_int8')}x at the same reply length with a check pass rate no lower. These definitions of "
             "the decode rate and of a faster q4 were set after preliminary benches on 20 states (the docstring of `tools/bench_generator.py`).")]
    cands = [cand for _, g in gens for cand in (g.get("choice") or {}).get("candidates", [])]
    for model in dict.fromkeys(cand["model"] for cand in sorted(cands, key=lambda x: -x["params_order"])):
        files = [cand for cand in cands if cand["model"] == model]
        if not any(cand["meets_targets"] for cand in files):
            text.append(f"No file of {model} meets both targets (decode rates "
                        + " and ".join(f"{cand['decode_passes_per_s']}" for cand in files) + " passes per s).")
    ch = deployed_choice([g for _, g in gens])
    if ch:
        text.append(f"The deployed file is `{ch['deployed']}`. The reason recorded by its run: \"{ch['reason']}\".")
    orig = original_wording_choice([g for _, g in gens])
    if orig:
        text.append(orig)
    if replay:
        p, r = replay
        rep = (((r.get("results") or {}).get("vision") or {}).get("reply") or {})
        fd, ra = rep.get("first_delta_ms") or {}, (r.get("results") or {}).get("reply_after_final_ms") or {}
        if fd.get("n"):
            text.append(f"In the replay bench with video and `--reply llm` (`{p}`), the first streamed word is emitted {fd['p50']} / {fd['p95']} ms "
                        f"(p50 / p95) after the endpoint frame arrives, and the whole reply {ra.get('p50')} / {ra.get('p95')} ms after the final "
                        "state. [results/measurements.md](results/measurements.md) has the rest of that bench.")
    return L + ["", " ".join(text), ""]


def extension(met: dict, ext: dict) -> list[str]:
    """the three appended sections. a section whose file is missing says so."""
    L = ["## Text, voice and face on MELD test", ""]
    L += three_inputs(ext["vision"][1], met, ext["vision"][0]) if ext.get("vision") else ["`results/metrics_vision.json` does not exist yet.", ""]
    L += ["## Commit policy (fitted Q-iteration)", ""]
    L += commit_policy(ext["commit"][1], ext["commit"][0]) if ext.get("commit") else ["`results/commit_policy.json` does not exist yet.", ""]
    L += ["## Streamed reply", ""]
    L += streamed_reply(ext["generator"], ext.get("replay")) if ext.get("generator") else ["No generator bench file exists yet.", ""]
    return L


def render(met: dict, man: dict, cmd: str, metrics_path: str, manifest_path: str, ext: dict | None = None) -> str:
    writers = [f"`{x}`" for x in (man["command"], man.get("ablations", {}).get("command"), man.get("sentiment_text_only", {}).get("command")) if x]
    intro = (f"Rendered by `{cmd}` from `{metrics_path}` (written by `{met['command']}`) and `{manifest_path}` "
             "(written by " + (", ".join(writers[:-1]) + " and " + writers[-1] if len(writers) > 1 else writers[0]) + "). "
             "The full report with the confusion matrices is [results/RESULTS.md](results/RESULTS.md), "
             "and the dev grid of the text head is in [results/01_text_only.md](results/01_text_only.md). Model selection, temperature and thresholds were chosen on dev. "
             "The test split was read once per condition. Weighted F1 is the primary metric, as in the published rows. "
             f"The intervals are 95 % percentile bootstrap intervals over test dialogues ({met['n_resamples']} resamples, seed {met['seed']}). "
             f"ECE uses {ECE_BINS} equal-width confidence bins. All metrics are in percent.")
    L = ["# Results", "", intro, "", "## Observations", ""] + [" ".join(observations(met)), ""]
    L += ["## Emotion on MELD test (7 classes)", ""] + headline(met, "emotion") + main_notes(met)
    L += ["## Gold text and live ASR transcripts", ""] + live_asr(met)
    L += ["## Ablations", ""] + ablations(met, man)
    L += ["## Disagreement subset", ""] + disagreement(met)
    L += ["## Early commit", ""] + early_commit(met)
    L += ["## Sentiment on MELD test (3 classes)", ""] + headline(met, "sentiment") + sentiment_notes(met)
    L += ["## Per-class F1 (emotion, test)", ""] + per_class(met)
    L += ["## Published rows", "", "| row | task | weighted F1 | source |", "|---|---|---|---|"]
    L += [f"| {sentences(p['row'])} | {p['task']} | {'' if p['weighted_f1'] is None else f'{p['weighted_f1']:.2f}'} | {sentences(p['source'])} |"
          for p in published(met)]
    L += [""] + published_notes(met)
    if any(r["status"] != "done" for r in met["conditions"].values()):
        L += ["## Pending rows", ""] + pending(met)
    L += ["## Choices made on dev (`heads/manifest.json`)", ""] + dev_choices(man)
    if ext is not None:
        L += extension(met, ext)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="render RESULTS.md from results/metrics.json and heads/manifest.json")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "heads" / "manifest.json")
    ap.add_argument("--out", type=Path, default=ROOT / "RESULTS.md")
    ap.add_argument("--vision-metrics", type=Path, default=ROOT / "results" / "metrics_vision.json", help="tools/evaluate_vision.py output")
    ap.add_argument("--commit-policy", type=Path, default=ROOT / "results" / "commit_policy.json", help="tools/train_commit_policy.py output")
    ap.add_argument("--generator", type=Path, nargs="*", default=None,
                    help="tools/bench_generator.py outputs (default: results/generator_bench_*.json)")
    ap.add_argument("--replay-video", type=Path, default=ROOT / "results" / "replay_bench_vm_tav.json",
                    help="tools/replay_bench.py --video --reply llm output, for the streamed reply's live numbers")
    a = ap.parse_args(argv)
    met = json.loads(a.metrics.read_text(encoding="utf-8"))
    man = json.loads(a.manifest.read_text(encoding="utf-8"))
    cmd = "python tools/render_results.py" + ("" if not (argv if argv is not None else sys.argv[1:]) else " " + " ".join(argv if argv is not None else sys.argv[1:]))

    def rel(p: Path) -> str:
        p = p.resolve()
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

    def load(p: Path) -> tuple[str, dict] | None:
        return (rel(p), json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None

    gens = a.generator if a.generator is not None else sorted((ROOT / "results").glob("generator_bench_*.json"))
    ext = {"vision": load(a.vision_metrics), "commit": load(a.commit_policy), "generator": [x for x in map(load, gens) if x],
           "replay": load(a.replay_video)}
    a.out.write_text(render(met, man, cmd, rel(a.metrics), rel(a.manifest), ext), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
