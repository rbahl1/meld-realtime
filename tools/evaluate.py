"""evaluate on MELD test every condition of design 1.11 that the caches and heads
allow, and mark the rest pending. selection happened on dev in train_heads.py;
this script reads the test split once per condition and never touches dev.

now computable: majority floor, text only k=0, text only best k, text only with
the runtime-regime window (same-speaker previous utterances, flags 1, turn index
= own earlier utterances), sentiment (text_only head). with cache/audio_test.npz
and the audio heads: raw sensevoice tag, calibrator A/B, stacked fusion, poe,
poe_uncalibrated, fused sentiment. ablations, live asr text, disagreement subset
and early commit stay pending until their caches and heads exist.

metrics: weighted f1 (primary), macro f1, accuracy, per-class f1, confusion
matrix, 95 % bootstrap ci over dialogues (1000 resamples, seed 0), ece (10 bins).
published rows are copied from design section 0 with their sources.

usage: python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json [--conditions a b]
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

from meld_rt import heads, labels, metrics  # noqa: E402
from meld_rt.data import emotion_counts, emotion_ids, load_split, sentiment_ids  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from train_heads import audio_matrix, file_record, fusion_rows, groups_of, read_audio_cache, load_text_cache, text_matrix  # noqa: E402

N_RESAMPLES = 1000
SEED = 0
PENDING = "pending (needs MELD audio)"

# design section 0, as written there (test weighted f1 unless stated; percent)
PUBLISHED = [
    ("MELD paper Table 11, text-CNN, text", "emotion", 55.02, "Poria et al., ACL 2019, arXiv 1810.02508, Table 11"),
    ("MELD paper Table 11, bcLSTM, text", "emotion", 56.44, "same"),
    ("MELD paper Table 11, bcLSTM, audio", "emotion", 39.08, "same"),
    ("MELD paper Table 11, bcLSTM, text+audio", "emotion", 59.25, "same"),
    ("MELD paper Table 11, DialogueRNN, text", "emotion", 57.03, "same"),
    ("MELD paper Table 11, DialogueRNN, audio", "emotion", 41.79, "same"),
    ("MELD paper Table 11, DialogueRNN, text+audio", "emotion", 60.25, "same"),
    ("EmoBERTa Table 2, RoBERTa-large, no past or future utterances", "emotion", 63.46, "Kim and Vossen, arXiv 2108.12009, Table 2"),
    ("EmoBERTa Table 2, only past", "emotion", 64.55, "same"),
    ("EmoBERTa Table 2, past and future", "emotion", 65.61, "same"),
    ("EmoBERTa Table 2, without speaker names", "emotion", 65.07, "same"),
    ("EmoBERTa Table 2, COSMIC", "emotion", 65.21, "same"),
    ("EmoBERTa Table 2, DialogXL", "emotion", 62.41, "same"),
    ("FunAudioLLM Table 8, SenseVoice-S on MELD, no target-domain fine-tune (WF1; UA 50.8, WA 57.8, F1 44.6)", "emotion", 57.7,
     "arXiv 2407.04051, Table 8; the report does not list its SER training corpora, so overlap with MELD cannot be excluded"),
    ("FunAudioLLM Table 8, EmoBox reference row (UA 31.5, WA 51.9, F1 32.9)", "emotion", None, "arXiv 2407.04051, Table 8; no WF1 given"),
    ("MELD paper Table 13, text-CNN, text", "sentiment", 64.25, "Poria et al., ACL 2019, Table 13"),
    ("MELD paper Table 13, DialogueRNN, text", "sentiment", 66.10, "same"),
    ("MELD paper Table 13, DialogueRNN, audio", "sentiment", 49.61, "same"),
    ("MELD paper Table 13, DialogueRNN, text+audio", "sentiment", 67.56, "same"),
]

DialogueRNN_PER_CLASS = {"anger": 43.65, "disgust": 7.89, "fear": 11.68, "joy": 54.40, "neutral": 77.44, "sadness": 34.59, "surprise": 52.51}

# condition name, task, description (design 1.11 numbering)
CONDITIONS = [
    ("majority", "emotion", "1. majority class of train (neutral) for every test utterance; posterior = train class prior"),
    ("text_k0", "emotion", "2. text only, k = 0 (heads/text_ctx_k0.npz, gold transcripts)"),
    ("text_best_k", "emotion", "2. text only, best k on dev, full-dialogue window (heads/text_ctx.npz)"),
    ("text_runtime_regime", "emotion", "2. text only, best k, runtime-regime window: same-speaker previous utterances only, flags 1"),
    ("audio_raw_tag", "emotion", "3. audio only, raw sensevoice tag: argmax over the seven named tokens"),
    ("audio_cal_A", "emotion", "4. audio only, AudioCalibrator A (slots + events + prosody)"),
    ("audio_cal_B", "emotion", "4. audio only, AudioCalibrator B (+ encoder states)"),
    ("fused_stacked", "emotion", "5. fused, stacked logistic regression (deployed)"),
    ("fused_poe", "emotion", "5. fused, product of experts with the dev-chosen weight"),
    ("fused_poe_uncalibrated", "emotion", "5. fused, fixed poe_uncalibrated fallback (w = 0.5)"),
    ("ablation_no_events", "emotion", "6. fused without event features"),
    ("ablation_no_prosody", "emotion", "6. fused without prosody"),
    ("ablation_no_events_no_prosody", "emotion", "6. fused without events and prosody"),
    ("live_asr_text", "emotion", "7. fused with sensevoice transcripts instead of gold text, plus WER"),
    ("disagreement_subset", "emotion", "8. utterances with jsd above the dev threshold: size, per-branch accuracy, fused gain"),
    ("early_commit", "emotion", "9. audio-only and fused label on 2 s and 4 s prefixes versus the full clip"),
    ("sentiment_text_only", "sentiment", "10. sentiment, text-only head (logistic regression over the text context features)"),
    ("sentiment_fused", "sentiment", "10. sentiment, fused head over the 19 fusion features"),
]


def scored(name: str, task: str, y: np.ndarray, proba: np.ndarray, groups: np.ndarray, note: str = "", **extra) -> dict:
    classes = labels.EMOTIONS if task == "emotion" else labels.SENTIMENTS
    pred = proba.argmax(axis=1)
    cm = metrics.confusion(y, pred, len(classes))
    s = metrics.summary(cm)
    return {"condition": name, "task": task, "status": "done", "n": len(y), **{k: round(v, 6) for k, v in s.items()},
            "per_class_f1": {c: round(float(v), 6) for c, v in zip(classes, metrics.f1_per_class(cm))},
            "confusion": cm.tolist(), "ci95": metrics.bootstrap_by_group(y, pred, groups, len(classes), N_RESAMPLES, SEED),
            "ece": round(metrics.ece(proba, y), 6), "note": note, **extra}


def pending(name: str, task: str, why: str = PENDING) -> dict:
    return {"condition": name, "task": task, "status": why}


def head_or_none(cls, path: Path):
    return cls.load(path) if path.exists() else None


def run_conditions(cache: Path, heads_dir: Path, meld: Path, wanted: set[str] | None) -> tuple[dict, dict]:
    test = load_split("test", meld)
    train = load_split("train", meld)
    y, ys, groups = emotion_ids(test), sentiment_ids(test), groups_of(test)
    emb = load_text_cache(cache, "test", test)
    manifest = json.loads((heads_dir / "manifest.json").read_text()) if (heads_dir / "manifest.json").exists() else {}
    text_head = head_or_none(heads.TextContextHead, heads_dir / "text_ctx.npz")
    text_k0 = head_or_none(heads.TextContextHead, heads_dir / "text_ctx_k0.npz")
    sent_head = head_or_none(heads.SentimentHead, heads_dir / "sentiment.npz")
    # rows without a cached clip are left out of the audio and fused conditions; each of those rows
    # reports n_no_audio and the text head's weighted f1 on exactly the same rows, so the comparison stays fair
    ra = read_audio_cache(cache, "test", test)
    aud, keep, no_audio = (None, None, []) if ra is None else ra
    out: dict[str, dict] = {}
    lp_text = text_head.predict_log_proba_batch(text_matrix(emb, test, text_head.k)) if text_head else None
    inputs = {"cache/text_test.npy": file_record(cache / "text_test.npy"), "meld/test_sent_emo.csv": file_record(meld / "test_sent_emo.csv")}
    if aud is not None:
        inputs["cache/audio_test.npz"] = file_record(cache / "audio_test.npz")
    for p in sorted(heads_dir.glob("*.npz")):
        inputs[f"heads/{p.name}"] = file_record(p)

    for name, task, desc in CONDITIONS:
        if wanted and name not in wanted:
            continue
        t0 = time.perf_counter()
        r = None
        if name == "majority":
            prior = emotion_counts(train) / len(train)
            r = scored(name, task, y, np.tile(prior, (len(y), 1)), groups, "train prior as posterior; label = neutral")
        elif name == "text_k0" and text_k0 is not None:
            r = scored(name, task, y, np.exp(text_k0.predict_log_proba_batch(text_matrix(emb, test, 0))), groups,
                       f"class weights {text_k0.params['class_weight_scheme']}, T={text_k0.temperature:.3f}")
        elif name == "text_best_k" and text_head is not None:
            r = scored(name, task, y, np.exp(lp_text), groups,
                       f"k={text_head.k}, class weights {text_head.params['class_weight_scheme']}, T={text_head.temperature:.3f}")
        elif name == "text_runtime_regime" and text_head is not None:
            lp = text_head.predict_log_proba_batch(text_matrix(emb, test, text_head.k, same_speaker_only=True))
            note = "same head as text_best_k, same-speaker window" + (" (k=0: differs from text_best_k only through the turn-index flags)" if text_head.k == 0 else "")
            r = scored(name, task, y, np.exp(lp), groups, note)
        elif name == "sentiment_text_only" and sent_head is not None and sent_head.variant == "text_only":
            lp = sent_head.predict_log_proba_batch(text_matrix(emb, test, sent_head.k))
            r = scored(name, task, ys, np.exp(lp), groups, f"k={sent_head.k}, C={sent_head.params.get('C', '?')}, T={sent_head.temperature:.3f}")
        elif name == "sentiment_fused" and sent_head is not None and sent_head.variant == "fused" and aud is not None:
            r = _fused_sentiment(name, task, ys[keep], groups[keep], sent_head, heads_dir, _rows(lp_text, keep), aud)
            r = _audio_rows_note(r, no_audio, ys[keep], None)
        elif aud is not None and name in ("audio_raw_tag", "audio_cal_A", "audio_cal_B", "fused_stacked", "fused_poe", "fused_poe_uncalibrated"):
            r = _audio_condition(name, task, y[keep], groups[keep], aud, heads_dir, manifest, _rows(lp_text, keep))
            r = _audio_rows_note(r, no_audio, y[keep], _rows(lp_text, keep))
        if r is None:
            why = PENDING
            if name.startswith("ablation"):
                why = PENDING + "; also needs the ablation heads of experiment 02"
            elif name == "live_asr_text":
                why = PENDING + "; also needs cache/text_asr_test.npy (embed_text.py on the sensevoice transcripts)"
            elif name == "early_commit":
                why = PENDING + "; needs the _p2/_p4 prefix features of cache/audio_test.npz"
            elif name == "disagreement_subset":
                why = PENDING + "; needs the jsd threshold chosen on dev with audio"
            elif name in ("text_k0", "text_best_k", "text_runtime_regime", "sentiment_text_only"):
                why = "pending (run tools/train_heads.py --text-only first)"
            r = pending(name, task, why)
        r["description"] = desc
        r["wall_s"] = round(time.perf_counter() - t0, 2)
        out[name] = r
        status = f"wF1 {100 * r['weighted_f1']:.2f}  mF1 {100 * r['macro_f1']:.2f}  acc {100 * r['accuracy']:.2f}" if r["status"] == "done" else r["status"]
        print(f"{name:32s} {status}", flush=True)
    return out, {"inputs": inputs, "manifest": {k: manifest.get(k) for k in ("k", "class_weight_scheme", "temperature", "thresholds", "text_only", "sklearn_version")}}


def _rows(a, keep):
    return None if a is None else a[keep]


def _audio_rows_note(r, no_audio, y_rows, lp_text_rows):
    """mark a result computed on the rows that have audio: how many test rows had no clip, and the
    text head's weighted f1 on the same rows (none when every row has audio, the rows are then identical)."""
    if r is None or not no_audio:
        return r
    r["n_no_audio"] = len(no_audio)
    r["note"] = (r.get("note") or "") + f"; {len(no_audio)} test rows without a clip left out (see cache/audio_test_meta.json)"
    if lp_text_rows is not None:
        cm = metrics.confusion(y_rows, np.argmax(lp_text_rows, axis=1), len(labels.EMOTIONS))
        r["text_best_k_same_rows_weighted_f1"] = round(float(metrics.summary(cm)["weighted_f1"]), 6)
    return r


def _audio_condition(name, task, y, groups, aud, heads_dir, manifest, lp_text):
    if name == "audio_raw_tag":
        p7 = np.stack([heads.audio7(p) for p in aud["emo_probs"]])
        abstain = float(np.mean(aud["emo_probs"][:, [labels.SV_EMO_TOKENS.index("OTHER"), heads.UNKNOWN_IDX]].sum(axis=1) > 0.5))
        return scored(name, task, y, p7, groups, "argmax over the seven named tokens", abstain_rate=round(abstain, 6))
    if name in ("audio_cal_A", "audio_cal_B"):
        h = head_or_none(heads.AudioCalibrator, heads_dir / f"{name}.npz")
        if h is None:
            return None
        try:
            X = audio_matrix(aud, h.uses_encoder_states)
        except ValueError:
            return None
        return scored(name, task, y, np.exp(h.predict_log_proba_batch(X)), groups, f"T={h.temperature:.3f}")
    cal = head_or_none(heads.AudioCalibrator, heads_dir / "audio_cal.npz")
    if cal is None or lp_text is None:
        return None
    lp_audio = cal.predict_log_proba_batch(audio_matrix(aud, cal.uses_encoder_states))
    if name == "fused_stacked":
        fh = head_or_none(heads.Fusion, heads_dir / "fusion.npz")
        if fh is None:
            return None
        return scored(name, task, y, np.exp(fh.predict_log_proba_batch(fusion_rows(lp_text, lp_audio, aud))), groups, f"T={fh.temperature:.3f}")
    w = 0.5 if name == "fused_poe_uncalibrated" else float(manifest.get("audio_fusion", {}).get("poe", {}).get("weight", 0.5))
    p_audio = np.exp(lp_audio) if name == "fused_poe" else aud["emo_probs"]
    p = np.stack([heads.fuse_poe(np.exp(lp_text[i]), p_audio[i], w) for i in range(len(y))])
    return scored(name, task, y, p, groups, f"w={w}" + ("; raw sensevoice posterior, uncalibrated" if name == "fused_poe_uncalibrated" else "; calibrated audio posterior"))


def _fused_sentiment(name, task, ys, groups, sent_head, heads_dir, lp_text, aud):
    cal = head_or_none(heads.AudioCalibrator, heads_dir / "audio_cal.npz")
    if cal is None or lp_text is None:
        return None
    lp_audio = cal.predict_log_proba_batch(audio_matrix(aud, cal.uses_encoder_states))
    return scored(name, task, ys, np.exp(sent_head.predict_log_proba_batch(fusion_rows(lp_text, lp_audio, aud))), groups, f"T={sent_head.temperature:.3f}")


# ---------------------------------------------------------------- rendering

def pct(v) -> str:
    return "" if v is None else f"{100 * v:.2f}"


def ci(r: dict, m: str) -> str:
    lo, hi = r["ci95"][m]
    return f"[{100 * lo:.2f}, {100 * hi:.2f}]"


def render(results: dict, meta: dict, cmd: str) -> str:
    intro = (f"Written by `{cmd}`. Every number below was computed by that command from the inputs listed; "
             "model selection, temperature and thresholds were chosen on dev by `tools/train_heads.py` and the test split was read once per condition. "
             "Weighted F1 is the primary metric (as in the published rows); CIs are 95 % percentile bootstrap intervals over test dialogues "
             f"({N_RESAMPLES} resamples, seed {SEED}); ECE uses 10 equal-width confidence bins. All metrics in percent.")
    L = ["# Results on MELD test", "", intro, ""]
    L += ["## Inputs", "", "| file | bytes | sha256 |", "| --- | --- | --- |"]
    L += [f"| `{k}` | {v['bytes']} | `{v['sha256']}` |" for k, v in meta["inputs"].items()]
    m = meta["manifest"]
    deployed = (f"Deployed text head from `heads/manifest.json`: k = {m.get('k')}, class weights `{m.get('class_weight_scheme')}`, "
                f"temperature {m.get('temperature')}, thresholds {json.dumps(m.get('thresholds'))}, text_only = {m.get('text_only')}, "
                f"scikit-learn {m.get('sklearn_version')}.")
    L += ["", deployed, ""]
    for task, title in (("emotion", "Emotion (7 classes)"), ("sentiment", "Sentiment (3 classes)")):
        L += [f"## {title}", "", "| condition | status | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | note |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in results.items():
            if r["task"] != task:
                continue
            if r["status"] == "done":
                L.append(f"| {name} | done | {pct(r['weighted_f1'])} | {ci(r, 'weighted_f1')} | {pct(r['macro_f1'])} | "
                         f"{pct(r['accuracy'])} | {pct(r['ece'])} | {r.get('note', '')} |")
            else:
                L.append(f"| {name} | {r['status']} | | | | | | {r['description']} |")
        L += ["", f"### Published rows ({task})", "", "| row | weighted F1 | source |", "| --- | --- | --- |"]
        L += [f"| {row} | {'' if v is None else f'{v:.2f}'} | {src} |" for row, t, v, src in PUBLISHED if t == task]
        L.append("")
    done = [r for r in results.values() if r["status"] == "done" and r["task"] == "emotion"]
    if done:
        L += ["## Per-class F1 (emotion)", "", "| condition | " + " | ".join(labels.EMOTIONS) + " |", "| --- |" + " --- |" * len(labels.EMOTIONS)]
        L += [f"| {r['condition']} | " + " | ".join(pct(r["per_class_f1"][c]) for c in labels.EMOTIONS) + " |" for r in done]
        L.append("| DialogueRNN text+audio (MELD paper Table 11) | " + " | ".join(f"{DialogueRNN_PER_CLASS[c]:.2f}" for c in labels.EMOTIONS) + " |")
        L.append("")
    for r in [r for r in results.values() if r["status"] == "done"]:
        classes = labels.EMOTIONS if r["task"] == "emotion" else labels.SENTIMENTS
        L += [f"### Confusion matrix: {r['condition']} (rows true, columns predicted)", "",
              "| true \\ pred | " + " | ".join(classes) + " |", "| --- |" + " --- |" * len(classes)]
        L += [f"| {c} | " + " | ".join(str(v) for v in row) + " |" for c, row in zip(classes, r["confusion"])]
        if "abstain_rate" in r:
            L.append(f"\nAbstain rate (OTHER + EMO_UNKNOWN above half the mass): {pct(r['abstain_rate'])} %.")
        L.append("")
    runtime = ("- runtime-regime window: previous utterances restricted to the same speaker within the dialogue, same-speaker flags 1, "
               "turn index = number of that speaker's earlier utterances (what a single user at the lamp has said so far); "
               "the head is the one trained on the full-dialogue window.")
    majority = ("- majority floor: every test utterance labelled with the train majority class; its posterior is the train class prior, "
                "which is what the ECE column scores.")
    L += ["## Definitions", "", runtime, majority,
          "- pending rows wait for MELD audio (`tools/extract_audio.py`, `tools/cache_audio_features.py`) and the heads of experiment 02.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--out", type=Path, nargs=2, default=[ROOT / "results" / "RESULTS.md", ROOT / "results" / "metrics.json"],
                    metavar=("RESULTS_MD", "METRICS_JSON"))
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of condition names; default all")
    args = ap.parse_args(argv)
    import threadpoolctl

    cmd = "python tools/evaluate.py " + " ".join(sys.argv[1:] if argv is None else argv)
    t0 = time.perf_counter()
    with threadpoolctl.threadpool_limits(2):
        results, meta = run_conditions(args.cache, args.heads, args.meld, set(args.conditions) if args.conditions else None)
    md, js = args.out
    md.parent.mkdir(parents=True, exist_ok=True)
    js.parent.mkdir(parents=True, exist_ok=True)
    payload = {"command": cmd, "split": "test", "n_resamples": N_RESAMPLES, "seed": SEED, "wall_s": round(time.perf_counter() - t0, 1),
               **meta, "published": [{"row": r, "task": t, "weighted_f1": v, "source": s} for r, t, v, s in PUBLISHED],
               "conditions": results}
    js.write_text(json.dumps(payload, indent=1))
    md.write_text(render(results, meta, cmd))
    print(f"wrote {md} and {js} in {payload['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
