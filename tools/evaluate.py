"""evaluate on MELD test every condition of design 1.11 that the caches and heads
allow, and mark the rest pending. selection happened on dev in train_heads.py;
this script reads the test split once per condition and never touches dev.

now computable: majority floor, text only k=0, text only best k, text only with
the runtime-regime window (same-speaker previous utterances, flags 1, turn index
= own earlier utterances), sentiment from the text-only head
(heads/sentiment_text_only.npz from train_heads.py --sentiment-text-only, checked
against its manifest record, or a text-only heads/sentiment.npz from --text-only).
with cache/audio_test.npz
and the audio heads: raw sensevoice tag, calibrator A/B, stacked fusion, poe,
poe_uncalibrated, fused sentiment, the disagreement subset (needs the dev jsd
threshold) and the audio-only early-commit rows. with the ablation heads
(train_heads.py --ablation): the three ablations. with the transcript embeddings
(embed_text.py --from-asr, cache/text_asr_test{,_p2,_p4}.npy): the live asr rows
(with the service's asr gate) and the fused early-commit rows. the early-commit rows
are offline prefix experiments, not the service's provisional states. a row whose
inputs are missing stays pending.

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

from functools import cached_property  # noqa: E402

from meld_rt import heads, labels, metrics  # noqa: E402
from meld_rt.data import emotion_counts, emotion_ids, key, load_split, sentiment_ids  # noqa: E402
from meld_rt.discrepancy import discrepancy  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from train_heads import (SENTIMENT_TEXT_ONLY, audio_matrix, file_record, fusion_rows, groups_of, read_audio_cache,  # noqa: E402
                         load_text_cache, text_matrix)

N_RESAMPLES = 1000
SEED = 0
PENDING = "pending (needs MELD audio)"
SR = 16000  # sample rate of the cached clips (tools/cache_audio_features.py)
AUDIO_FIELDS = ("emo_probs", "event_probs", "enc_pooled", "prosody", "asr_text", "asr_conf", "tokens_per_s", "duration_s")
ABLATION_ROWS = ("ablation_no_events", "ablation_no_prosody", "ablation_no_events_no_prosody")
# the heads this script reads (with the ablation heads), recorded under "inputs". the extension heads in heads/
# (vision_cal, fusion3, sentiment3, commit_policy) are scored by evaluate_vision.py and train_commit_policy.py
EVAL_HEADS = ("text_ctx.npz", "text_ctx_k0.npz", "sentiment.npz", SENTIMENT_TEXT_ONLY, "audio_cal.npz", "audio_cal_A.npz",
              "audio_cal_B.npz", "fusion.npz")
EARLY_ROWS = {"early_commit_audio_2s": ("audio", 2), "early_commit_fused_2s": ("fused", 2),
              "early_commit_audio_4s": ("audio", 4), "early_commit_fused_4s": ("fused", 4)}

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
    ("ablation_no_events", "emotion", "6. fused without event features (calibrator A without the 11 events, fusion without p_laughter)"),
    ("ablation_no_prosody", "emotion", "6. fused without prosody (calibrator A without the 4 prosody inputs)"),
    ("ablation_no_events_no_prosody", "emotion", "6. fused without events and prosody (calibrator A on the 9 emotion slots, fusion without p_laughter)"),
    ("live_asr_text_only", "emotion", ("7. text only, deployed text head on the sensevoice transcripts instead of gold text, with the "
     "service's asr gate (empty transcript or asr_conf below min_asr_conf: uniform posterior, label neutral)")),
    ("live_asr_text", "emotion", ("7. fused, deployed stacked fusion with the text posterior from the sensevoice transcripts and the "
     "service's asr gate, plus WER")),
    ("disagreement_subset", "emotion", "8. utterances with jsd above the dev threshold: size, share, per-branch scores on the subset and its complement"),
    ("early_commit_audio_2s", "emotion", ("9. audio only, deployed calibrator on the first 2 s with the cached prefix prosody, versus the "
     "full clip (offline prefix experiment, not the service's provisional state)")),
    ("early_commit_fused_2s", "emotion", ("9. stacked fusion on the first 2 s (prefix audio and prefix transcript), versus the full clip "
     "(offline prefix experiment, a provisional state runs no text expert)")),
    ("early_commit_audio_4s", "emotion", ("9. audio only, deployed calibrator on the first 4 s with the cached prefix prosody, versus the "
     "full clip (offline prefix experiment, not the service's provisional state)")),
    ("early_commit_fused_4s", "emotion", ("9. stacked fusion on the first 4 s (prefix audio and prefix transcript), versus the full clip "
     "(offline prefix experiment, a provisional state runs no text expert)")),
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


def brief(y: np.ndarray, proba: np.ndarray) -> dict:
    """weighted f1, macro f1 and accuracy of the argmax, rounded as in scored()."""
    s = metrics.summary(metrics.confusion(y, np.asarray(proba).argmax(axis=1), len(labels.EMOTIONS)))
    return {k: round(v, 6) for k, v in s.items()}


def head_or_none(cls, path: Path):
    return cls.load(path) if path.exists() else None


def text_only_sentiment(heads_dir: Path, manifest: dict, sent_head) -> tuple:
    """(head, file) for the sentiment_text_only row, or (None, why it is pending). heads/sentiment_text_only.npz
    (train_heads.py --sentiment-text-only) when it exists and matches its manifest record, else heads/sentiment.npz
    when it is the text-only variant (--text-only)."""
    path = heads_dir / SENTIMENT_TEXT_ONLY
    if path.exists():
        rec = ((manifest.get("sentiment_text_only") or {}).get("files") or {}).get(path.name)
        if rec is not None and file_record(path) != rec:
            return None, (f"pending ({path.name} differs from heads/manifest.json, "
                          "rerun tools/train_heads.py --sentiment-text-only)")
        return heads.SentimentHead.load(path), f"heads/{path.name}"
    if sent_head is not None and sent_head.variant == "text_only":
        return sent_head, "heads/sentiment.npz"
    return None, "pending (run tools/train_heads.py --sentiment-text-only first)"


def run_conditions(cache: Path, heads_dir: Path, meld: Path, wanted: set[str] | None) -> tuple[dict, dict]:
    test = load_split("test", meld)
    train = load_split("train", meld)
    y, ys, groups = emotion_ids(test), sentiment_ids(test), groups_of(test)
    emb = load_text_cache(cache, "test", test)
    manifest = json.loads((heads_dir / "manifest.json").read_text()) if (heads_dir / "manifest.json").exists() else {}
    text_head = head_or_none(heads.TextContextHead, heads_dir / "text_ctx.npz")
    text_k0 = head_or_none(heads.TextContextHead, heads_dir / "text_ctx_k0.npz")
    sent_head = head_or_none(heads.SentimentHead, heads_dir / "sentiment.npz")
    sent_text, sent_text_file = text_only_sentiment(heads_dir, manifest, sent_head)
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
        if p.name in EVAL_HEADS or p.name.startswith("ablation_"):
            inputs[f"heads/{p.name}"] = file_record(p)
    for p in sorted(cache.glob("text_asr_test*.npy")):
        inputs[f"cache/{p.name}"] = file_record(p)
    ctx = None if aud is None else Experiments(cache, heads_dir, manifest, test, y[keep], groups[keep], aud, keep, text_head,
                                               _rows(lp_text, keep), inputs.get("cache/audio_test.npz"))

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
        elif name == "sentiment_text_only" and sent_text is not None:
            lp = sent_text.predict_log_proba_batch(text_matrix(emb, test, sent_text.k))
            r = scored(name, task, ys, np.exp(lp), groups,
                       f"{sent_text_file}, k={sent_text.k}, C={sent_text.params.get('C', '?')}, T={sent_text.temperature:.3f}")
        elif name == "sentiment_fused" and sent_head is not None and sent_head.variant == "fused" and aud is not None:
            r = _fused_sentiment(name, task, ys[keep], groups[keep], sent_head, heads_dir, _rows(lp_text, keep), aud)
            r = _audio_rows_note(r, no_audio, ys[keep], None)
        elif aud is not None and name in ("audio_raw_tag", "audio_cal_A", "audio_cal_B", "fused_stacked", "fused_poe", "fused_poe_uncalibrated"):
            r = _audio_condition(name, task, y[keep], groups[keep], aud, heads_dir, manifest, _rows(lp_text, keep))
            r = _audio_rows_note(r, no_audio, y[keep], _rows(lp_text, keep))
        elif ctx is not None and ctx.handles(name):
            r = ctx.run(name, task)
            if isinstance(r, str):
                r = pending(name, task, r)
            elif r["status"] == "done":
                # the disagreement row covers a subset of the rows, so the text head on "the same rows" would mislead there
                r = _audio_rows_note(r, no_audio, y[keep], None if name == "disagreement_subset" else _rows(lp_text, keep))
        if r is None:
            why = PENDING
            if name.startswith("ablation"):
                why = PENDING + "; also needs the ablation heads (tools/train_heads.py --ablation)"
            elif name.startswith("live_asr"):
                why = PENDING + "; also needs cache/text_asr_test.npy (tools/embed_text.py --from-asr)"
            elif name.startswith("early_commit"):
                why = PENDING + "; needs the _p2/_p4 prefix features of cache/audio_test.npz"
            elif name == "disagreement_subset":
                why = PENDING + "; needs the jsd threshold chosen on dev with audio"
            elif name in ("text_k0", "text_best_k", "text_runtime_regime"):
                why = "pending (run tools/train_heads.py --text-only first)"
            elif name == "sentiment_text_only":
                why = sent_text_file
            r = pending(name, task, why)
        r["description"] = desc
        r["wall_s"] = round(time.perf_counter() - t0, 2)
        out[name] = r
        status = f"wF1 {100 * r['weighted_f1']:.2f}  mF1 {100 * r['macro_f1']:.2f}  acc {100 * r['accuracy']:.2f}" if r["status"] == "done" else r["status"]
        print(f"{name:32s} {status}", flush=True)
    meta_manifest = {k: manifest.get(k) for k in ("k", "class_weight_scheme", "temperature", "thresholds", "text_only", "sklearn_version")}
    meta_manifest["audio_cal_variant"] = ((manifest.get("audio_fusion") or {}).get("audio_cal") or {}).get("variant")
    return out, {"inputs": inputs, "manifest": meta_manifest}


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
    if lp_text is None:
        return None
    if name == "fused_poe_uncalibrated":
        # the raw sensevoice posterior at w = 0.5: needs no audio head, as on the live path
        p = np.stack([heads.fuse_poe(np.exp(lp_text[i]), aud["emo_probs"][i], 0.5) for i in range(len(y))])
        return scored(name, task, y, p, groups, "w=0.5; raw sensevoice posterior, uncalibrated")
    cal = head_or_none(heads.AudioCalibrator, heads_dir / "audio_cal.npz")
    if cal is None:
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


# ---------------------------------------------------------------- experiments 6 to 9

def disagreement_mask(p_text: np.ndarray, emo_probs: np.ndarray, threshold: float) -> np.ndarray:
    """rows flagged by the service's discrepancy function (meld_rt.discrepancy.discrepancy) of
    the given 7-way text posterior against the raw 9-way sensevoice posterior, jsd above the threshold."""
    return np.array([discrepancy(p_text[i], emo_probs[i], threshold)[1] for i in range(len(p_text))], dtype=bool)


def asr_gate(asr_text: np.ndarray, asr_conf: np.ndarray, min_conf: float) -> tuple[np.ndarray, np.ndarray]:
    """(asr_failed, low_asr_conf) masks as Service._run_final sets them on a final state: an empty
    transcript fails, a nonempty one with asr_conf below min_conf is low."""
    failed = np.array([not str(t).strip() for t in asr_text], dtype=bool)
    return failed, ~failed & (np.asarray(asr_conf, dtype=np.float64) < min_conf)


def degrade(proba: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """the service's degraded final state on the given rows: a uniform posterior with the label
    neutral. neutral is EMOTIONS[0], so the argmax of the tie is the published label."""
    if labels.EMOTIONS[0] != "neutral":
        raise ValueError("degrade() needs neutral first in labels.EMOTIONS")
    out = np.array(proba, dtype=np.float64)
    out[rows] = 1.0 / out.shape[1]
    return out


def provisional_prosody(tokens_per_s: np.ndarray) -> np.ndarray:
    """(n, 4) prosody of a provisional state (Service._run_provisional): zeros except the speech rate."""
    p = np.zeros((len(tokens_per_s), 4), dtype=np.float32)
    p[:, 3] = tokens_per_s
    return p


def n_full_clip_values(duration_s: np.ndarray, prefix_s: float) -> np.ndarray:
    """mask of the clips no longer than the prefix: the cache repeats their full-clip values
    (cache_audio_features.py cuts a clip only when it has more than prefix * 16000 samples)."""
    return np.round(np.asarray(duration_s, dtype=np.float64) * SR) <= round(prefix_s * SR)


class Experiments:
    """experiments 6 to 9 of design 1.11 on the test rows that have a cached clip. the
    shared inputs (deployed audio posterior, transcript embeddings) are computed on
    first use. run() returns a scored row, or a string that says why the row is pending."""

    def __init__(self, cache: Path, heads_dir: Path, manifest: dict, test: list, y: np.ndarray, groups: np.ndarray,
                 aud: dict, keep: np.ndarray, text_head, lp_text: np.ndarray | None, audio_record: dict | None):
        self.cache, self.heads_dir, self.manifest, self.test = cache, heads_dir, manifest, test
        self.y, self.groups, self.aud, self.keep = y, groups, aud, keep
        self.text_head, self.lp_text, self.audio_record = text_head, lp_text, audio_record
        self._emb: dict[str, np.ndarray | None] = {}
        self._lp_asr: dict[str, np.ndarray | None] = {}

    @staticmethod
    def handles(name: str) -> bool:
        return name in ABLATION_ROWS or name in EARLY_ROWS or name in ("live_asr_text_only", "live_asr_text", "disagreement_subset")

    def run(self, name: str, task: str) -> dict | str:
        if name in ABLATION_ROWS:
            return self.ablation(name, task)
        if name.startswith("live_asr"):
            return self.live_asr(name, task)
        if name == "disagreement_subset":
            return self.disagreement(name, task)
        return self.early_commit(name, task, *EARLY_ROWS[name])

    # shared inputs
    @cached_property
    def cal(self):
        return head_or_none(heads.AudioCalibrator, self.heads_dir / "audio_cal.npz")

    @cached_property
    def fusion(self):
        return head_or_none(heads.Fusion, self.heads_dir / "fusion.npz")

    @cached_property
    def lp_audio(self) -> np.ndarray:
        return self.cal.predict_log_proba_batch(audio_matrix(self.aud, self.cal.uses_encoder_states))

    def fused(self, lp_text: np.ndarray, lp_audio: np.ndarray, d: dict) -> np.ndarray:
        return np.exp(self.fusion.predict_log_proba_batch(fusion_rows(lp_text, lp_audio, d)))

    @cached_property
    def gate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(asr_failed, low_asr_conf, min_asr_conf) of the full clips, with the service's default threshold."""
        from meld_rt.service import ServiceConfig

        min_conf = float(ServiceConfig().min_asr_conf)
        return (*asr_gate(self.aud["asr_text"], self.aud["asr_conf"], min_conf), min_conf)

    def asr_emb(self, suffix: str) -> np.ndarray | None:
        """(n_test, 768) transcript embeddings from embed_text.py --from-asr in csv order, None
        when the file is absent. rows without a clip stay zero, as an absent context slot is,
        and are never scored."""
        if suffix not in self._emb:
            path = self.cache / f"text_asr_test{suffix}.npy"
            kpath, mpath = path.with_name(path.stem + "_keys.json"), path.with_name(path.stem + "_meta.json")
            out = None
            if path.exists() and kpath.exists():
                keys = json.loads(kpath.read_text())
                if sorted(keys) != sorted(str(k) for k in self.aud["keys"]):
                    raise ValueError(f"{kpath.name} does not list the clips of cache/audio_test.npz")
                src = json.loads(mpath.read_text()).get("source", {}) if mpath.exists() else {}
                if src.get("field", f"asr_text{suffix}") != f"asr_text{suffix}" or (
                        src.get("sha256") and self.audio_record and src["sha256"] != self.audio_record["sha256"]):
                    raise ValueError(f"{path.name} was embedded from another field or audio cache, rerun tools/embed_text.py --from-asr")
                e = np.load(path)
                if e.shape != (len(keys), heads.TEXT_DIM):
                    raise ValueError(f"{path.name} has shape {e.shape}")
                row = {key(u): i for i, u in enumerate(self.test)}
                out = np.zeros((len(self.test), heads.TEXT_DIM), dtype=np.float32)
                out[[row[k] for k in keys]] = e
            self._emb[suffix] = out
        return self._emb[suffix]

    def lp_text_asr(self, suffix: str) -> np.ndarray | None:
        """deployed text head on the transcripts, scored rows only. a prefix row keeps the
        full transcripts of the earlier utterances as its context."""
        if suffix not in self._lp_asr:
            cur, full = self.asr_emb(suffix), self.asr_emb("")
            lp = None
            if cur is not None and full is not None and self.text_head is not None:
                lp = self.text_head.predict_log_proba_batch(text_matrix(cur, self.test, self.text_head.k, emb_prev=full))[self.keep]
            self._lp_asr[suffix] = lp
        return self._lp_asr[suffix]

    def view(self, prefix_s: float) -> dict | None:
        """the prefix columns of the audio cache under the full-clip names."""
        sfx = f"_p{prefix_s:g}"
        if any(f + sfx not in self.aud for f in AUDIO_FIELDS):
            return None
        return {"keys": self.aud["keys"], **{f: self.aud[f + sfx] for f in AUDIO_FIELDS}}

    # conditions
    def ablation(self, name: str, task: str) -> dict | str:
        sec = self.manifest.get("ablations") or {}
        rec = (sec.get("variants") or {}).get(name[len("ablation_"):])
        if rec is None:
            return "pending (run tools/train_heads.py --ablation)"
        if self.lp_text is None:
            return "pending (needs heads/text_ctx.npz)"
        paths = [self.heads_dir / rec[part]["file"] for part in ("audio_cal", "fusion")]
        for path in paths:
            if not path.exists() or file_record(path) != (sec.get("files") or {}).get(path.name):
                return f"pending ({path.name} is missing or differs from heads/manifest.json, rerun tools/train_heads.py --ablation)"
        cal, fh = (heads.NpzHead.load(path) for path in paths)
        lp_a = cal.predict_log_proba_batch(audio_matrix(self.aud, False)[:, np.asarray(cal.params["columns"], dtype=np.int64)])
        F = fusion_rows(self.lp_text, lp_a, self.aud)[:, np.asarray(fh.params["columns"], dtype=np.int64)]
        note = (f"calibrator A without {' and '.join(rec['dropped'])} ({cal.input_dim} inputs, T={cal.temperature:.3f}), "
                f"fusion with {fh.input_dim} inputs (T={fh.temperature:.3f})")
        return scored(name, task, self.y, np.exp(fh.predict_log_proba_batch(F)), self.groups, note, dropped=rec["dropped"],
                      audio_cal_input_dim=cal.input_dim, fusion_input_dim=fh.input_dim, audio_only=brief(self.y, np.exp(lp_a)))

    def live_asr(self, name: str, task: str) -> dict | str:
        lp = self.lp_text_asr("")
        if lp is None:
            return "pending (needs cache/text_asr_test.npy: python tools/embed_text.py --split test --from-asr)"
        if self.lp_text is None:
            return "pending (needs heads/text_ctx.npz)"
        wer_path = self.cache / "wer_test.json"
        wer = json.loads(wer_path.read_text()) if wer_path.exists() else {}
        failed, low, min_conf = self.gate
        bad = failed | low
        extra = {"wer": wer.get("wer"), "wer_n": wer.get("n"), "n_empty_transcripts": int(failed.sum()),
                 "n_low_asr_conf": int(low.sum()), "n_degraded": int(bad.sum()), "min_asr_conf": min_conf}
        note = (f"sensevoice transcripts, corpus WER {wer['wer']} over {wer['n']} utterances (cache/wer_test.json)" if "wer" in wer
                else "sensevoice transcripts, no cache/wer_test.json")
        note += (f", service asr gate: {int(failed.sum())} empty transcripts (asr_failed) and {int(low.sum())} with asr_conf below "
                 f"{min_conf:g} (low_asr_conf) get a uniform posterior and the label neutral, heads trained on gold text")
        # without_gate: every row through the heads, the empty transcripts with the expert's empty-text embedding
        if name == "live_asr_text_only":
            p = np.exp(lp)
            return scored(name, task, self.y, degrade(p, bad), self.groups, note, **extra, without_gate=brief(self.y, p),
                          gold_text_same_rows=brief(self.y, np.exp(self.lp_text)))
        if self.cal is None or self.fusion is None:
            return "pending (needs heads/audio_cal.npz and heads/fusion.npz)"
        p = self.fused(lp, self.lp_audio, self.aud)
        return scored(name, task, self.y, degrade(p, bad), self.groups, f"T={self.fusion.temperature:.3f}, " + note, **extra,
                      without_gate=brief(self.y, p), gold_text_same_rows=brief(self.y, self.fused(self.lp_text, self.lp_audio, self.aud)))

    def disagreement(self, name: str, task: str) -> dict | str:
        thr = (self.manifest.get("thresholds") or {}).get("jsd_flag")
        if thr is None:
            return "pending (needs thresholds.jsd_flag in heads/manifest.json, chosen on dev with audio)"
        if self.lp_text is None or self.cal is None or self.fusion is None:
            return "pending (needs heads/text_ctx.npz, heads/audio_cal.npz and heads/fusion.npz)"
        p_text, p_audio = np.exp(self.lp_text), np.exp(self.lp_audio)
        p_fused = self.fused(self.lp_text, self.lp_audio, self.aud)
        m = disagreement_mask(p_text, self.aud["emo_probs"], float(thr))
        if not m.any() or m.all():
            return f"pending ({int(m.sum())} of {len(m)} rows flagged at jsd_flag {thr}, no subset to compare)"

        def part(rows: np.ndarray) -> dict:
            return {"n": int(rows.sum()), "text": brief(self.y[rows], p_text[rows]), "audio_cal": brief(self.y[rows], p_audio[rows]),
                    "fused_stacked": brief(self.y[rows], p_fused[rows])}

        note = (f"stacked fusion on the {int(m.sum())} of {len(m)} rows with jsd above {thr} (thresholds.jsd_flag, chosen on dev), "
                "jsd of the gold-text posterior against the raw sensevoice posterior")
        return scored(name, task, self.y[m], p_fused[m], self.groups[m], note, jsd_flag=float(thr), n_rows=len(m),
                      n_subset=int(m.sum()), n_complement=int((~m).sum()), share=round(float(m.mean()), 6),
                      subset=part(m), complement=part(~m))

    def early_commit(self, name: str, task: str, branch: str, prefix_s: float) -> dict | str:
        view = self.view(prefix_s)
        if view is None:
            return f"pending (needs the _p{prefix_s:g} prefix features of cache/audio_test.npz)"
        if self.cal is None:
            return "pending (needs heads/audio_cal.npz)"
        lp_a = self.cal.predict_log_proba_batch(audio_matrix(view, self.cal.uses_encoder_states))
        short = n_full_clip_values(self.aud["duration_s"], prefix_s)
        cut = ~short
        extra = {}
        if branch == "audio":
            prefix, full = np.exp(lp_a), np.exp(self.lp_audio)
            ref = "deployed calibrator (heads/audio_cal.npz) on the full clip"
            # the calibrator inputs of a provisional state: the same prefix audio, prosody zero except the speech rate
            prov = dict(view, prosody=provisional_prosody(view["tokens_per_s"]))
            p_prov = np.exp(self.cal.predict_log_proba_batch(audio_matrix(prov, self.cal.uses_encoder_states)))
            extra["provisional_inputs"] = {**brief(self.y, p_prov), "cut": brief(self.y[cut], p_prov[cut]) if cut.any() else None}
        else:
            lp_tp, lp_tf = self.lp_text_asr(f"_p{prefix_s:g}"), self.lp_text_asr("")
            if lp_tp is None or lp_tf is None or self.fusion is None:
                return (f"pending (needs cache/text_asr_test_p{prefix_s:g}.npy and cache/text_asr_test.npy from "
                        f"tools/embed_text.py --from-asr, and heads/fusion.npz)")
            prefix, full = self.fused(lp_tp, lp_a, view), self.fused(lp_tf, self.lp_audio, self.aud)
            ref = "stacked fusion on the full clip and its transcript, without the service's asr gate (live_asr_text without_gate)"
        agree = prefix.argmax(axis=1) == full.argmax(axis=1)
        cut_rows = {"n": int(cut.sum())}
        if cut.any():
            cut_rows.update(prefix=brief(self.y[cut], prefix[cut]), full_clip=brief(self.y[cut], full[cut]),
                            label_agreement_with_full_clip=round(float(agree[cut].mean()), 6))
        note = (f"offline prefix experiment: first {prefix_s:g} s, {int(short.sum())} of {len(short)} clips are not longer than "
                f"{prefix_s:g} s and use their full-clip values")
        return scored(name, task, self.y, prefix, self.groups, note, branch=branch, prefix_s=float(prefix_s),
                      n_full_clip_values=int(short.sum()), full_clip={**brief(self.y, full), "reference": ref},
                      label_agreement_with_full_clip=round(float(agree.mean()), 6), cut_rows=cut_rows, **extra)


# ---------------------------------------------------------------- rendering

def pct(v) -> str:
    return "" if v is None else f"{100 * v:.2f}"


def ci(r: dict, m: str) -> str:
    lo, hi = r["ci95"][m]
    return f"[{100 * lo:.2f}, {100 * hi:.2f}]"


def render(results: dict, meta: dict, cmd: str) -> str:
    intro = (f"Written by `{cmd}`. Every number below was computed by that command from the inputs listed. "
             "Model selection, temperature and thresholds were chosen on dev by `tools/train_heads.py`, and the test split was read once per condition. "
             "Weighted F1 is the primary metric, as in the published rows. The CIs are 95 % percentile bootstrap intervals over test dialogues "
             f"({N_RESAMPLES} resamples, seed {SEED}). ECE uses 10 equal-width confidence bins. All metrics are in percent.")
    L = ["# Results on MELD test", "", intro, ""]
    L += ["## Inputs", "", "| file | bytes | sha256 |", "| --- | --- | --- |"]
    L += [f"| `{k}` | {v['bytes']} | `{v['sha256']}` |" for k, v in meta["inputs"].items()]
    m = meta["manifest"]
    deployed = (f"Deployed text head, from `heads/manifest.json`: k = {m.get('k')}, class weights `{m.get('class_weight_scheme')}`, "
                f"temperature {m.get('temperature')}, thresholds `{json.dumps(m.get('thresholds'))}`, text_only = {m.get('text_only')}, "
                f"scikit-learn {m.get('sklearn_version')}.")
    L += ["", deployed, ""]
    for task, title in (("emotion", "Emotion (7 classes)"), ("sentiment", "Sentiment (3 classes)")):
        L += [f"## {title}", "", "| condition | status | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | note |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in results.items():
            if r["task"] != task:
                continue
            if r["status"] == "done":
                note = r.get("note", "")
                if "text_best_k_same_rows_weighted_f1" in r:
                    note += f", text head on the same rows {pct(r['text_best_k_same_rows_weighted_f1'])}"
                L.append(f"| {name} | done | {pct(r['weighted_f1'])} | {ci(r, 'weighted_f1')} | {pct(r['macro_f1'])} | "
                         f"{pct(r['accuracy'])} | {pct(r['ece'])} | {note} |")
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
        L += [f"### Confusion matrix of {r['condition']} (rows true, columns predicted)", "",
              "| true \\ pred | " + " | ".join(classes) + " |", "| --- |" + " --- |" * len(classes)]
        L += [f"| {c} | " + " | ".join(str(v) for v in row) + " |" for c, row in zip(classes, r["confusion"])]
        if "abstain_rate" in r:
            L.append(f"\nAbstain rate (OTHER + EMO_UNKNOWN above half the mass): {pct(r['abstain_rate'])} %.")
        L.append("")
    runtime = ("- Runtime-regime window: the previous utterances are restricted to the same speaker within the dialogue, the same-speaker "
               "flags are 1, and the turn index counts that speaker's earlier utterances (what a single user in front of the robot has said so far). "
               "The head is the one trained on the full-dialogue window.")
    majority = ("- Majority floor: every test utterance gets the train majority class. Its posterior is the train class prior, "
                "which is what the ECE column scores.")
    no_clip = ("- Rows without a clip: the audio and fused conditions leave out test rows that have no cached clip. The note says how many, "
               "and `results/metrics.json` gives the text head's weighted F1 on the same rows (`text_best_k_same_rows_weighted_f1`).")
    pending_rows = ("- Pending rows: a row stays pending until its inputs exist, and its status names the missing input. The audio and fused "
                    "rows need MELD audio (`tools/extract_audio.py`, `tools/cache_audio_features.py`, then `tools/train_heads.py` without "
                    "`--text-only`). The ablation rows also need `tools/train_heads.py --ablation`, and the live ASR and fused early-commit rows "
                    "need the transcript embeddings (`tools/embed_text.py --from-asr`).")
    L += render_experiments(results, meta)
    L += ["## Definitions", "", runtime, majority, no_clip]
    if any(r["status"] != "done" for r in results.values()):
        L.append(pending_rows)
    L.append("")
    return "\n".join(L)


def render_experiments(results: dict, meta: dict) -> list[str]:
    """tables for experiments 6 to 9 from the extra fields of their rows. a table appears once its rows are done."""
    done = {n: r for n, r in results.items() if r["status"] == "done"}
    L: list[str] = []
    ab = [n for n in ABLATION_ROWS if n in done]
    if ab:
        L += ["## Ablations", "",
              ("Each ablation refits calibrator A without the named inputs, and a fusion head over its posterior, with the same folds, grids "
               "and dev selection as the deployed heads (`tools/train_heads.py --ablation`). Without events the fusion head also loses "
               "p_laughter. Audio only is the ablated calibrator by itself."), "",
              "| condition | calibrator inputs | fusion inputs | fused weighted F1 | fused accuracy | audio-only weighted F1 | audio-only accuracy |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        if meta["manifest"].get("audio_cal_variant") == "A" and "fused_stacked" in done and "audio_cal_A" in done:
            f, a = done["fused_stacked"], done["audio_cal_A"]
            L.append(f"| deployed (fused_stacked, audio_cal_A) | {heads.AUDIO_SLOT_DIM} | {heads.FUSION_DIM} | {pct(f['weighted_f1'])} | "
                     f"{pct(f['accuracy'])} | {pct(a['weighted_f1'])} | {pct(a['accuracy'])} |")
        for n in ab:
            r = done[n]
            L.append(f"| {n} | {r['audio_cal_input_dim']} | {r['fusion_input_dim']} | {pct(r['weighted_f1'])} | {pct(r['accuracy'])} | "
                     f"{pct(r['audio_only']['weighted_f1'])} | {pct(r['audio_only']['accuracy'])} |")
        L.append("")
    live = [n for n in ("live_asr_text_only", "live_asr_text") if n in done]
    if live:
        r0 = done[live[0]]
        wer = (f"The SenseVoice transcripts of test have a corpus WER of {r0['wer']} over {r0['wer_n']} utterances (`cache/wer_test.json`). "
               if r0.get("wer") is not None else "")
        L += ["## Live ASR text", "",
              wer + (f"The rows apply the service's gate on a final state. {r0['n_empty_transcripts']} transcripts are empty (asr_failed) "
                     f"and {r0['n_low_asr_conf']} have an ASR confidence below {r0['min_asr_conf']:g} (low_asr_conf). These "
                     f"{r0['n_degraded']} rows get a uniform posterior and the label neutral, as `Service._run_final` publishes them. "
                     "The without-gate columns score every row through the heads, with the text expert's empty-text embedding for "
                     "the empty transcripts. The heads are the deployed ones, trained on gold text. The gold columns score the same heads "
                     "on the gold text of the same rows."), "",
              ("| condition | weighted F1 | accuracy | without gate weighted F1 | without gate accuracy | gold text weighted F1 | "
               "gold text accuracy |"), "| --- |" + " --- |" * 6]
        for n in live:
            r, w, g = done[n], done[n]["without_gate"], done[n]["gold_text_same_rows"]
            L.append(f"| {n} | {pct(r['weighted_f1'])} | {pct(r['accuracy'])} | {pct(w['weighted_f1'])} | {pct(w['accuracy'])} | "
                     f"{pct(g['weighted_f1'])} | {pct(g['accuracy'])} |")
        L.append("")
    if "disagreement_subset" in done:
        r = done["disagreement_subset"]
        L += ["## Disagreement subset", "",
              (f"{r['n_subset']} of {r['n_rows']} test utterances ({pct(r['share'])} %) have a JSD above {r['jsd_flag']} between the text "
               "posterior and the raw SenseVoice posterior. The JSD uses the service's discrepancy function (`meld_rt.discrepancy`) and "
               "the dev threshold (`thresholds.jsd_flag` in `heads/manifest.json`). The text posterior is the gold-text one with the "
               "full-dialogue window. The service computes it on the ASR transcript with its own stream context, and only for a "
               "transcript that passes its ASR gate."), "",
              "| rows | n | text weighted F1 | text accuracy | audio weighted F1 | audio accuracy | fused weighted F1 | fused accuracy |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for part in ("subset", "complement"):
            q = r[part]
            L.append(f"| {part} | {q['n']} | " + " | ".join(f"{pct(q[b]['weighted_f1'])} | {pct(q[b]['accuracy'])}"
                                                         for b in ("text", "audio_cal", "fused_stacked")) + " |")
        L.append("")
    early = [n for n in EARLY_ROWS if n in done]
    if early:
        L += ["## Early commit", "",
              ("These rows are offline prefix experiments, not the service's provisional states. Each row scores the first 2 s or 4 s "
               "of every clip with the full calibrator inputs of the prefix, including its cached prosody. A clip that is not longer than "
               "the prefix keeps its full-clip values, as the cache stores them. The fused rows take the text posterior from the prefix "
               "transcript. The service runs no text expert on a provisional state. The full-clip reference is the same pipeline on the "
               "whole clip: the deployed calibrator for audio, and the stacked fusion with the full transcript for fused, without the "
               "service's ASR gate (the without-gate columns of live_asr_text). Agreement is the share of clips whose prefix label "
               "equals the full-clip label. The cut columns cover only the clips longer than the prefix."), "",
              ("| condition | clips with full-clip values | weighted F1 | full clip weighted F1 | accuracy | full clip accuracy | agreement | "
               "cut clips | cut weighted F1 | cut full clip weighted F1 |"), "| --- |" + " --- |" * 9]
        for n in early:
            r, c = done[n], done[n]["cut_rows"]
            cut = (f"{pct(c['prefix']['weighted_f1'])} | {pct(c['full_clip']['weighted_f1'])}" if c["n"] else " | ")
            L.append(f"| {n} | {r['n_full_clip_values']} | {pct(r['weighted_f1'])} | {pct(r['full_clip']['weighted_f1'])} | "
                     f"{pct(r['accuracy'])} | {pct(r['full_clip']['accuracy'])} | {pct(r['label_agreement_with_full_clip'])} | {c['n']} | {cut} |")
        L.append("")
        prov = [n for n in early if "provisional_inputs" in done[n]]
        if prov:
            L += [("A provisional state runs the deployed calibrator with the prosody set to zero except the speech rate "
                   "(`Service._run_provisional`). The next table scores the audio rows again with those inputs on the same prefixes."), "",
                  "| condition | weighted F1 | accuracy | cut weighted F1 | cut accuracy |", "| --- | --- | --- | --- | --- |"]
            for n in prov:
                q = done[n]["provisional_inputs"]
                c = q["cut"] or {}
                L.append(f"| {n}, provisional inputs | {pct(q['weighted_f1'])} | {pct(q['accuracy'])} | {pct(c.get('weighted_f1'))} | "
                         f"{pct(c.get('accuracy'))} |")
            L.append("")
    return L


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
