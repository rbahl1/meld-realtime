"""train_heads.py and evaluate.py end to end on csv subsets with synthetic text
and audio caches: the audio and fusion stage, the ablations (--ablation), the
text-only sentiment head next to the deployed ones (--sentiment-text-only) and the
conditions of experiments 6 to 9 (live asr text, disagreement subset, early commit).
the rows of experiments 7 and 9 are recomputed here from the caches and heads."""

import csv
import json
import shutil
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import embed_text  # noqa: E402
import evaluate  # noqa: E402
import train_heads  # noqa: E402

from meld_rt import heads, labels, metrics  # noqa: E402
from meld_rt.data import MELD_DIR, dialogues, emotion_ids, key, load_split, sentiment_ids  # noqa: E402
from meld_rt.discrepancy import discrepancy  # noqa: E402
from meld_rt.service import ServiceConfig  # noqa: E402

COLS = ["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"]
SV_OF = {m: labels.SV_EMO_TOKENS.index(t) for t, m in labels.SV_TO_MELD.items()}
DURATIONS = (1.5, 2.0, 3.0, 4.0, 5.5)  # cycled over the rows: some clips end before each prefix
SYNTH_WER = 0.25  # written to the synthetic cache/wer_test.json
ABLATION_DIMS = {"no_events": (13, 18), "no_prosody": (20, 19), "no_events_no_prosody": (9, 18)}
NEW_ROWS = (*evaluate.ABLATION_ROWS, "live_asr_text_only", "live_asr_text", "disagreement_subset", *evaluate.EARLY_ROWS)


def summary3(y: np.ndarray, p: np.ndarray) -> dict:
    """weighted f1, macro f1 and accuracy of the argmax, rounded as evaluate.py rounds them."""
    return {k: round(v, 6) for k, v in metrics.summary(metrics.confusion(y, p.argmax(axis=1), 7)).items()}


def assert_scored_as(r: dict, y: np.ndarray, p: np.ndarray) -> None:
    """the row was scored on posterior p: same confusion matrix and the same ece."""
    assert r["confusion"] == metrics.confusion(y, p.argmax(axis=1), 7).tolist(), r["condition"]
    assert r["ece"] == round(metrics.ece(p, y), 6), r["condition"]


def fused_proba(fh, lp_text: np.ndarray, lp_audio: np.ndarray, d: dict) -> np.ndarray:
    return np.exp(fh.predict_log_proba_batch(train_heads.fusion_rows(lp_text, lp_audio, d)))


def subset(split: str, n_dialogues: int):
    """the first dialogues of a split, extended until every emotion appears."""
    dias = dialogues(load_split(split, MELD_DIR))
    out, k = [], n_dialogues
    while k <= len(dias):
        out = [u for d in dias[:k] for u in d]
        if len({u.emotion for u in out}) == len(labels.EMOTIONS):
            return out
        k += 5
    raise AssertionError("not every emotion present")


def write_csv(utts, path: Path) -> None:
    def ts(s: float) -> str:
        return f"00:00:{s:06.3f}".replace(".", ",")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i, u in enumerate(utts):
            w.writerow({"Sr No.": i, "Utterance": u.text, "Speaker": u.speaker, "Emotion": u.emotion, "Sentiment": u.sentiment,
                        "Dialogue_ID": u.dialogue_id, "Utterance_ID": u.utterance_id, "Season": 1, "Episode": 1,
                        "StartTime": ts(u.start_s % 60), "EndTime": ts(u.end_s % 60)})


def synth_caches(utts, split: str, cache: Path, rng, centers_t, centers_e) -> None:
    y = emotion_ids(utts)
    n = len(utts)
    emb = (centers_t[y] + 1.5 * rng.standard_normal((n, 768))).astype(np.float32)
    np.save(cache / f"text_{split}.npy", emb)
    (cache / f"text_{split}_keys.json").write_text(json.dumps([key(u) for u in utts]))
    logit9 = rng.standard_normal((n, 9))
    logit9[np.arange(n), [SV_OF[labels.EMOTIONS[c]] for c in y]] += 2.0
    emo = np.exp(logit9) / np.exp(logit9).sum(1, keepdims=True)
    ev = rng.uniform(size=(n, 11))
    ev /= ev.sum(1, keepdims=True)
    full = {"emo_probs": emo.astype(np.float32), "event_probs": ev.astype(np.float32),
            "enc_pooled": (centers_e[y] + rng.standard_normal((n, 1024))).astype(np.float32),
            "prosody": rng.standard_normal((n, 4)).astype(np.float32), "asr_text": np.array([u.text for u in utts]),
            "asr_conf": rng.uniform(0.5, 1.0, n).astype(np.float32), "tokens_per_s": np.full(n, 3.0, np.float32),
            "duration_s": np.resize(np.array(DURATIONS, np.float32), n)}
    full["asr_text"][0] = ""  # sensevoice heard nothing: asr_failed in the service
    full["asr_conf"][1] = 0.2  # below ServiceConfig.min_asr_conf: low_asr_conf in the service
    # prefix columns as cache_audio_features.py writes them, from a generator of their own so the
    # full-clip columns above stay as they were: a clip no longer than the prefix repeats its values
    prng = np.random.default_rng(100 + len(split))
    prefixes = {}
    for p in (2, 4):
        cut = full["duration_s"] > p
        pf = {f: v.copy() for f, v in full.items()}
        for f in ("emo_probs", "event_probs"):
            noisy = full[f] * prng.uniform(0.5, 1.5, full[f].shape)
            pf[f][cut] = (noisy / noisy.sum(1, keepdims=True))[cut]
        pf["prosody"][cut] += prng.standard_normal((int(cut.sum()), 4)).astype(np.float32)
        pf["enc_pooled"][cut] += prng.standard_normal((int(cut.sum()), 1024)).astype(np.float32)
        # every third cut clip opens like the previous clip, so some prefix labels differ from the full clip
        swap = cut & (np.arange(n) % 3 == 0)
        for f in ("emo_probs", "event_probs", "enc_pooled", "prosody"):
            pf[f][swap] = np.roll(full[f], 1, axis=0)[swap]
        pf["asr_text"] = np.array([" ".join(t.split()[:2]) if c else t for t, c in zip(full["asr_text"], cut)])
        pf["asr_conf"][cut] *= 0.9
        pf["duration_s"][cut] = p
        prefixes.update({f"{f}_p{p}": v for f, v in pf.items()})
    np.savez(cache / f"audio_{split}.npz", keys=np.array([key(u) for u in utts]), **full, **prefixes)
    if split == "test":
        synth_asr_embeddings(cache, emb, full["duration_s"], prng)


def synth_asr_embeddings(cache: Path, gold: np.ndarray, duration_s: np.ndarray, rng) -> None:
    """text_asr_test{,_p2,_p4}.npy as embed_text.py --from-asr writes them: the gold embedding
    plus noise for the transcripts, every fourth row with the previous row's gold embedding (a
    misrecognized transcript, so asr text scores below gold text), more noise for the prefix
    transcripts of the cut clips."""
    keys = [str(k) for k in np.load(cache / "audio_test.npz")["keys"]]
    full = gold + 0.5 * rng.standard_normal(gold.shape).astype(np.float32)
    swap = np.arange(len(gold)) % 4 == 1
    full[swap] = np.roll(gold, 1, axis=0)[swap]
    for sfx, e in (("", full), *((f"_p{p}", np.where((duration_s > p)[:, None], gold + 1.5 * rng.standard_normal(gold.shape), full))
                                  for p in (2, 4))):
        out = cache / f"text_asr_test{sfx}.npy"
        np.save(out, e.astype(np.float32))
        embed_text.keys_path(out).write_text(json.dumps(keys))
        embed_text.meta_path(out).write_text(json.dumps({"source": embed_text.asr_source(cache / "audio_test.npz", f"asr_text{sfx}")}))
    (cache / "wer_test.json").write_text(json.dumps({"split": "test", "n": len(keys), "wer": SYNTH_WER}))


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    meld, cache, out = root / "meld", root / "cache", root / "heads"
    for d in (meld, cache, out):
        d.mkdir()
    rng = np.random.default_rng(0)
    centers_t, centers_e = 2.0 * rng.standard_normal((7, 768)), 2.0 * rng.standard_normal((7, 1024))
    for split, n in (("train", 40), ("dev", 12), ("test", 15)):
        utts = subset(split, n)
        write_csv(utts, meld / f"{split}_sent_emo.csv")
        synth_caches(utts, split, cache, rng, centers_t, centers_e)
    cfg = yaml.safe_load((ROOT / "configs" / "heads.yaml").read_text(encoding="utf-8"))
    cfg["text_ctx"]["grid"] = {"k": [0, 1], "class_weight": ["none", "balanced"], "learning_rate_init": [0.001]}
    cfg["text_ctx"]["mlp"].update({"max_epochs": 6, "patience": 3})
    cfg["lr"]["C"] = [0.1, 1]
    cfg["fusion"]["cross_fit_folds"] = 3
    (root / "heads.yaml").write_text(yaml.safe_dump(cfg))
    return root, meld, cache, out


def test_train_full_stage_and_evaluate(workspace):
    root, meld, cache, out = workspace
    assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]) == 0
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for name in ("text_ctx.npz", "text_ctx_k0.npz", "audio_cal_A.npz", "audio_cal_B.npz", "audio_cal.npz", "fusion.npz", "sentiment.npz"):
        assert (out / name).exists(), name
    assert man["audio_fusion"]["status"] == "trained" and man["audio_rows_dropped"] == {"train": 0, "dev": 0}
    assert man["audio_fusion"]["cross_fit"]["folds"] == 3
    assert sum(man["audio_fusion"]["cross_fit"]["fold_sizes"]) == len(load_split("train", meld))
    assert 0.0 < man["thresholds"]["jsd_flag"] <= 1.0 and man["thresholds_pending"] == {}
    assert man["sentiment"]["variant"] == "fused"
    assert heads.Fusion.load(out / "fusion.npz").input_dim == 19
    assert heads.SentimentHead.load(out / "sentiment.npz").input_dim == 19
    cal = heads.AudioCalibrator.load(out / "audio_cal.npz")
    assert cal.input_dim in (24, 1048)
    assert heads.TextContextHead.load(out / "text_ctx.npz").k in (0, 1)
    # synthetic signal is strong: audio and fused heads beat the majority rate on dev
    assert man["audio_fusion"]["fusion"]["dev"]["weighted_f1"] > 0.5

    md, js = root / "RESULTS.md", root / "metrics.json"
    assert evaluate.main(["--cache", str(cache), "--heads", str(out), "--meld", str(meld), "--out", str(md), str(js)]) == 0
    met = json.loads(js.read_text(encoding="utf-8"))
    done = {n for n, r in met["conditions"].items() if r["status"] == "done"}
    assert {"majority", "text_k0", "text_best_k", "text_runtime_regime", "audio_raw_tag", "audio_cal_A", "audio_cal_B",
            "fused_stacked", "fused_poe", "fused_poe_uncalibrated", "sentiment_fused"} <= done
    assert "sentiment_text_only" not in done
    # before --ablation the ablation rows wait for their heads, the rows of experiments 7 to 9 are done
    assert met["conditions"]["ablation_no_events"]["status"] == "pending (run tools/train_heads.py --ablation)"
    assert {"live_asr_text_only", "live_asr_text", "disagreement_subset", *evaluate.EARLY_ROWS} <= done
    r = met["conditions"]["fused_stacked"]
    assert r["n"] == len(load_split("test", meld)) and len(r["confusion"]) == 7 and 0 <= r["ece"] <= 1
    assert "abstain_rate" in met["conditions"]["audio_raw_tag"]
    text = md.read_text(encoding="utf-8")
    assert "| fused_stacked | done |" in text and "pending" in text


def test_ablation_heads_and_experiment_rows(workspace):
    """--ablation writes heads of the expected sizes next to the deployed ones and only adds a manifest
    section. evaluate.py then scores every row of experiments 6 to 9, and the earlier rows do not move."""
    root, meld, cache, out = workspace
    args = ["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]
    if not (out / "fusion.npz").exists():
        assert train_heads.main(args) == 0
    md0, js0 = root / "RESULTS_before.md", root / "metrics_before.json"
    assert evaluate.main(["--cache", str(cache), "--heads", str(out), "--meld", str(meld), "--out", str(md0), str(js0)]) == 0
    before = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    deployed = {p.name: p.read_bytes() for p in out.glob("*.npz") if not p.name.startswith("ablation_")}

    assert train_heads.main([*args, "--ablation"]) == 0
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    ab = man.pop("ablations")
    assert man == before
    assert {p.name: p.read_bytes() for p in out.glob("*.npz") if not p.name.startswith("ablation_")} == deployed
    assert ab["status"] == "trained" and ab["cross_fit"]["fold_sizes"] == before["audio_fusion"]["cross_fit"]["fold_sizes"]
    assert ab["reference"]["reproduces_deployed"]["audio_cal_A.npz"] is True
    assert set(ab["files"]) == {f"ablation_{n}_{part}.npz" for n in ABLATION_DIMS for part in ("audio_cal", "fusion")}
    for name, (n_cal, n_fus) in ABLATION_DIMS.items():
        cal = heads.NpzHead.load(out / f"ablation_{name}_audio_cal.npz")
        fh = heads.NpzHead.load(out / f"ablation_{name}_fusion.npz")
        assert cal.input_dim == n_cal == len(cal.params["columns"]) == ab["variants"][name]["audio_cal"]["input_dim"], name
        assert fh.input_dim == n_fus == len(fh.params["columns"]) == ab["variants"][name]["fusion"]["input_dim"], name
        assert cal.n_classes == fh.n_classes == 7 and 0.0 <= ab["variants"][name]["fusion"]["dev"]["weighted_f1"] <= 1.0
    assert heads.NpzHead.load(out / "ablation_no_events_audio_cal.npz").params["columns"].tolist() == [*range(9), 20, 21, 22, 23]
    assert train_heads.FUSION_LAUGHTER_COL not in heads.NpzHead.load(out / "ablation_no_events_fusion.npz").params["columns"].tolist()
    assert heads.NpzHead.load(out / "ablation_no_prosody_fusion.npz").params["columns"].tolist() == list(range(19))

    md, js = root / "RESULTS_exp.md", root / "metrics_exp.json"
    assert evaluate.main(["--cache", str(cache), "--heads", str(out), "--meld", str(meld), "--out", str(md), str(js)]) == 0
    met = json.loads(js.read_text(encoding="utf-8"))["conditions"]
    old = json.loads(js0.read_text(encoding="utf-8"))["conditions"]
    for name, r in old.items():
        if r["status"] == "done":
            assert {k: v for k, v in met[name].items() if k != "wall_s"} == {k: v for k, v in r.items() if k != "wall_s"}, name
    test = load_split("test", meld)
    n = len(test)
    for name in NEW_ROWS:
        r = met[name]
        assert r["status"] == "done", (name, r["status"])
        assert len(r["confusion"]) == 7 and sum(map(sum, r["confusion"])) == r["n"], name
        assert 0.0 <= r["weighted_f1"] <= 1.0 and 0.0 <= r["accuracy"] <= 1.0 and 0.0 <= r["ece"] <= 1.0, name
        assert r["ci95"]["weighted_f1"][0] <= r["ci95"]["weighted_f1"][1] and "n_no_audio" not in r, name
        assert r["n"] == (r["n_subset"] if name == "disagreement_subset" else n), name
    for name, (n_cal, n_fus) in ABLATION_DIMS.items():
        r = met[f"ablation_{name}"]
        assert (r["audio_cal_input_dim"], r["fusion_input_dim"]) == (n_cal, n_fus) and 0.0 <= r["audio_only"]["weighted_f1"] <= 1.0

    # live asr, recomputed: the text head on the transcript embeddings, then the service's gate (an empty
    # transcript or asr_conf below min_asr_conf gives a uniform posterior, label neutral). gold text on the same rows beside it
    z = dict(np.load(cache / "audio_test.npz"))
    y = emotion_ids(test)
    th = heads.TextContextHead.load(out / "text_ctx.npz")
    cal = heads.AudioCalibrator.load(out / "audio_cal.npz")
    fh = heads.Fusion.load(out / "fusion.npz")
    asr_emb = np.load(cache / "text_asr_test.npy")
    lp_asr = th.predict_log_proba_batch(train_heads.text_matrix(asr_emb, test, th.k))
    lp_gold = th.predict_log_proba_batch(train_heads.text_matrix(np.load(cache / "text_test.npy"), test, th.k))
    lp_audio = cal.predict_log_proba_batch(train_heads.audio_matrix(z, cal.uses_encoder_states))
    failed = np.array([not str(t).strip() for t in z["asr_text"]])
    low = ~failed & (z["asr_conf"] < ServiceConfig().min_asr_conf)
    bad = failed | low
    assert failed.sum() >= 1 and low.sum() >= 1

    def gated(p: np.ndarray) -> np.ndarray:
        g = np.array(p, dtype=np.float64)
        g[bad] = 1.0 / 7
        return g

    live = {"live_asr_text_only": (np.exp(lp_asr), np.exp(lp_gold)),
            "live_asr_text": (fused_proba(fh, lp_asr, lp_audio, z), fused_proba(fh, lp_gold, lp_audio, z))}
    for name, (p, p_gold) in live.items():
        r = met[name]
        assert_scored_as(r, y, gated(p))
        assert (gated(p)[bad].argmax(axis=1) == labels.EMOTIONS.index("neutral")).all()
        assert r["without_gate"] == summary3(y, p) and r["gold_text_same_rows"] == summary3(y, p_gold)
        assert (r["n_empty_transcripts"], r["n_low_asr_conf"], r["n_degraded"]) == (failed.sum(), low.sum(), bad.sum())
        assert r["wer"] == SYNTH_WER and f"WER {SYNTH_WER}" in r["note"]
        # the check can tell transcripts from gold text: gold through the same gate gives other labels
        assert r["confusion"] != metrics.confusion(y, gated(p_gold).argmax(axis=1), 7).tolist(), name
    assert met["live_asr_text_only"]["gold_text_same_rows"]["weighted_f1"] == met["text_best_k"]["weighted_f1"]
    assert met["live_asr_text"]["gold_text_same_rows"]["weighted_f1"] == met["fused_stacked"]["weighted_f1"]

    # disagreement: subset and complement partition the rows, and the subset is what meld_rt.discrepancy flags
    d = met["disagreement_subset"]
    assert d["n_subset"] + d["n_complement"] == d["n_rows"] == n and d["subset"]["n"] == d["n_subset"] == d["n"]
    assert d["complement"]["n"] == d["n_complement"] and d["share"] == round(d["n_subset"] / n, 6)
    assert d["jsd_flag"] == man["thresholds"]["jsd_flag"]
    th = heads.TextContextHead.load(out / "text_ctx.npz")
    p_text = np.exp(th.predict_log_proba_batch(train_heads.text_matrix(np.load(cache / "text_test.npy"), test, th.k)))
    emo = np.load(cache / "audio_test.npz")["emo_probs"]
    flags = [discrepancy(p_text[i], emo[i], d["jsd_flag"])[1] for i in range(n)]
    assert evaluate.disagreement_mask(p_text, emo, d["jsd_flag"]).tolist() == flags and sum(flags) == d["n_subset"]
    for part in ("subset", "complement"):
        assert {"text", "audio_cal", "fused_stacked"} <= set(d[part])
        assert all(0.0 <= d[part][b][m] <= 1.0 for b in ("text", "audio_cal", "fused_stacked") for m in ("weighted_f1", "accuracy"))

    # early commit, recomputed from the _p2/_p4 columns and prefix transcripts. a prefix row keeps the full
    # transcripts of the earlier utterances as context. short clips keep their full-clip values, so their labels agree
    dur = z["duration_s"]
    variant = man["audio_fusion"]["audio_cal"]["variant"]
    full = {"audio": np.exp(lp_audio), "fused": live["live_asr_text"][0]}
    for p in (2, 4):
        v = {"keys": z["keys"], **{f: z[f"{f}_p{p}"] for f in evaluate.AUDIO_FIELDS}}
        lp_ap = cal.predict_log_proba_batch(train_heads.audio_matrix(v, cal.uses_encoder_states))
        lp_tp = th.predict_log_proba_batch(train_heads.text_matrix(np.load(cache / f"text_asr_test_p{p}.npy"), test, th.k, emb_prev=asr_emb))
        prefix = {"audio": np.exp(lp_ap), "fused": fused_proba(fh, lp_tp, lp_ap, v)}
        short = dur <= p
        for branch in ("audio", "fused"):
            r = met[f"early_commit_{branch}_{p}s"]
            assert r["prefix_s"] == p and r["branch"] == branch and r["n_full_clip_values"] == int(short.sum())
            assert r["cut_rows"]["n"] == n - r["n_full_clip_values"] > 0
            assert_scored_as(r, y, prefix[branch])
            assert {m: x for m, x in r["full_clip"].items() if m != "reference"} == summary3(y, full[branch])
            assert r["cut_rows"]["prefix"] == summary3(y[~short], prefix[branch][~short])
            agree = prefix[branch].argmax(axis=1) == full[branch].argmax(axis=1)
            # and the check can tell prefix from full clip: some cut clips change their label
            assert agree[short].all() and not agree[~short].all(), r["condition"]
            # from the row alone: every clip that keeps its full-clip values agrees with the full clip
            n_agree = round(r["label_agreement_with_full_clip"] * n)
            n_agree_cut = round(r["cut_rows"]["label_agreement_with_full_clip"] * r["cut_rows"]["n"])
            assert n_agree - n_agree_cut == r["n_full_clip_values"] and n_agree == int(agree.sum()), r["condition"]
        assert met[f"early_commit_audio_{p}s"]["full_clip"]["weighted_f1"] == met["audio_cal_" + variant]["weighted_f1"]
        assert met[f"early_commit_fused_{p}s"]["full_clip"]["weighted_f1"] == met["live_asr_text"]["without_gate"]["weighted_f1"]
        # the calibrator inputs of a provisional state: the same prefix audio, prosody zero except the speech rate
        pv = dict(v, prosody=np.c_[np.zeros((n, 3), np.float32), v["tokens_per_s"]].astype(np.float32))
        p_prov = np.exp(cal.predict_log_proba_batch(train_heads.audio_matrix(pv, cal.uses_encoder_states)))
        assert met[f"early_commit_audio_{p}s"]["provisional_inputs"] == {**summary3(y, p_prov), "cut": summary3(y[~short], p_prov[~short])}
        assert "provisional_inputs" not in met[f"early_commit_fused_{p}s"]
    text = md.read_text(encoding="utf-8")
    for title in ("## Ablations", "## Live ASR text", "## Disagreement subset", "## Early commit"):
        assert title in text
    assert "| ablation_no_events | done |" in text and "| early_commit_fused_4s | done |" in text


def test_prefix_text_rows_keep_full_context(workspace):
    """Experiments.lp_text_asr with a k=1 head that returns its input: a prefix row has its own prefix
    transcript in the current slot and the full transcript of the previous utterance in the context slot."""
    _, meld, cache, out = workspace
    test = load_split("test", meld)
    aud, keep, _ = train_heads.read_audio_cache(cache, "test", test)

    class Echo:
        k = 1

        @staticmethod
        def predict_log_proba_batch(X):
            return X

    ex = evaluate.Experiments(cache, out, {}, test, emotion_ids(test)[keep], train_heads.groups_of(test)[keep], aud, keep, Echo(),
                              None, train_heads.file_record(cache / "audio_test.npz"))
    full, pre = np.load(cache / "text_asr_test.npy"), np.load(cache / "text_asr_test_p2.npy")
    X = ex.lp_text_asr("_p2")
    row = {key(u): i for i, u in enumerate(test)}
    D = heads.TEXT_DIM
    n_differ = 0
    for dia in dialogues(test):
        for prev, u in pairwise(dia):
            i, j = row[key(u)], row[key(prev)]
            assert np.array_equal(X[i, :D], pre[i]) and np.array_equal(X[i, D:2 * D], full[j])
            n_differ += not np.array_equal(pre[j], full[j])
    assert n_differ > 0  # some previous utterance has a prefix transcript that differs from its full one


def test_early_commit_provisional_inputs_with_calibrator_a(workspace, tmp_path):
    """the provisional-input scores of the audio early-commit rows, recomputed. calibrator A stands in for
    the deployed one here: with B the encoder states make the synthetic posteriors ignore prosody."""
    root, meld, cache, out = workspace
    if not (out / "fusion.npz").exists():
        assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]) == 0
    heads_a = tmp_path / "heads_a"
    heads_a.mkdir()
    for f in out.iterdir():
        if f.is_file():
            (heads_a / f.name).write_bytes(f.read_bytes())
    (heads_a / "audio_cal.npz").write_bytes((out / "audio_cal_A.npz").read_bytes())
    md, js = tmp_path / "RESULTS.md", tmp_path / "metrics.json"
    assert evaluate.main(["--cache", str(cache), "--heads", str(heads_a), "--meld", str(meld), "--out", str(md), str(js),
                          "--conditions", "early_commit_audio_2s", "early_commit_audio_4s"]) == 0
    met = json.loads(js.read_text(encoding="utf-8"))["conditions"]
    test = load_split("test", meld)
    y = emotion_ids(test)
    z = dict(np.load(cache / "audio_test.npz"))
    cal = heads.AudioCalibrator.load(out / "audio_cal_A.npz")
    for p in (2, 4):
        v = {"keys": z["keys"], **{f: z[f"{f}_p{p}"] for f in evaluate.AUDIO_FIELDS}}
        cut = z["duration_s"] > p
        prefix = np.exp(cal.predict_log_proba_batch(train_heads.audio_matrix(v, False)))
        pv = dict(v, prosody=np.c_[np.zeros((len(y), 3), np.float32), v["tokens_per_s"]].astype(np.float32))
        prov = np.exp(cal.predict_log_proba_batch(train_heads.audio_matrix(pv, False)))
        r = met[f"early_commit_audio_{p}s"]
        assert_scored_as(r, y, prefix)
        assert r["provisional_inputs"] == {**summary3(y, prov), "cut": summary3(y[cut], prov[cut])}
        assert (prov.argmax(axis=1) != prefix.argmax(axis=1)).any()  # under A, zero prosody moves some labels


def test_train_text_only_on_subset(workspace):
    root, meld, cache, _ = workspace
    out = root / "heads_text_only"
    assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld), "--text-only"]) == 0
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["text_only"] and man["sentiment"]["variant"] == "text_only" and man["thresholds"]["jsd_flag"] is None
    assert man["audio_rows_dropped"] is None
    assert man["thresholds_pending"] == {"jsd_flag": "needs MELD audio"}
    sent = heads.SentimentHead.load(out / "sentiment.npz")
    assert sent.variant == "text_only" and sent.input_dim == heads.TextContextHead.input_dim_for(sent.k)
    assert float(sent.params["C"]) in (0.1, 1.0)
    assert not (out / "fusion.npz").exists()


def test_sentiment_text_only_mode_leaves_the_deployed_heads(workspace, tmp_path):
    """--sentiment-text-only writes heads/sentiment_text_only.npz and a "sentiment_text_only" manifest section,
    with the function, config, seed and k of the deployed text head. every other head file and manifest entry stays
    byte for byte. evaluate.py then scores sentiment_text_only from that file, and no other row moves."""
    root, meld, cache, out = workspace
    if not (out / "fusion.npz").exists():
        assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]) == 0
    hd = tmp_path / "heads"
    shutil.copytree(out, hd)  # a copy, so the workspace heads stay as the other tests expect them
    args = ["--cache", str(cache), "--out", str(hd), "--config", str(root / "heads.yaml"), "--meld", str(meld)]
    md0, js0 = tmp_path / "RESULTS_before.md", tmp_path / "metrics_before.json"
    assert evaluate.main(["--cache", str(cache), "--heads", str(hd), "--meld", str(meld), "--out", str(md0), str(js0)]) == 0
    text_before = (hd / "manifest.json").read_text(encoding="utf-8")
    before = json.loads(text_before)
    assert before["sentiment"]["variant"] == "fused" and "sentiment_text_only" not in before
    others = {p.name: p.read_bytes() for p in hd.iterdir() if p.name != "manifest.json"}

    # another config is refused before anything is written
    cfg = yaml.safe_load((root / "heads.yaml").read_text(encoding="utf-8"))
    (tmp_path / "other.yaml").write_text(yaml.safe_dump({**cfg, "seed": 1}))
    with pytest.raises(SystemExit, match="not the config"):
        train_heads.main([*args[:4], "--config", str(tmp_path / "other.yaml"), *args[6:], "--sentiment-text-only"])
    assert (hd / "manifest.json").read_text(encoding="utf-8") == text_before and not (hd / "sentiment_text_only.npz").exists()

    assert train_heads.main([*args, "--sentiment-text-only"]) == 0
    text_after = (hd / "manifest.json").read_text(encoding="utf-8")
    man = json.loads(text_after)
    sec = man.pop("sentiment_text_only")
    assert man == before and list(man) == list(before)
    assert text_after.startswith(text_before.rstrip("}\n"))  # the earlier entries keep their text, the section comes last
    assert {p.name: p.read_bytes() for p in hd.iterdir() if p.name not in ("manifest.json", "sentiment_text_only.npz")} == others
    k = before["selected"]["text_ctx"]["k"]
    assert sec["variant"] == "text_only" and sec["file"] == "sentiment_text_only.npz" and sec["text_head"]["k"] == k
    assert sec["seed"] == before["seed"] and sec["command"].endswith("--sentiment-text-only")
    assert sec["files"] == {"sentiment_text_only.npz": train_heads.file_record(hd / "sentiment_text_only.npz")}
    h = heads.SentimentHead.load(hd / "sentiment_text_only.npz")
    assert h.variant == "text_only" and h.k == k and h.input_dim == heads.TextContextHead.input_dim_for(k)
    # the same head that --text-only fits as heads/sentiment.npz
    utts = {s: load_split(s, meld) for s in ("train", "dev")}
    emb = {s: train_heads.load_text_cache(cache, s, utts[s]) for s in ("train", "dev")}
    ref = train_heads.train_sentiment_text_only(tmp_path, cfg, utts, emb, k, cfg["seed"])
    assert train_heads.same_weights(h, tmp_path / "sentiment.npz") and (ref["C"], ref["dev"]) == (sec["C"], sec["dev"])

    md, js = tmp_path / "RESULTS.md", tmp_path / "metrics.json"
    assert evaluate.main(["--cache", str(cache), "--heads", str(hd), "--meld", str(meld), "--out", str(md), str(js)]) == 0
    met, old = json.loads(js.read_text(encoding="utf-8")), json.loads(js0.read_text(encoding="utf-8"))
    assert old["conditions"]["sentiment_text_only"]["status"].startswith("pending")
    r = met["conditions"]["sentiment_text_only"]
    assert r["status"] == "done" and r["note"].startswith("heads/sentiment_text_only.npz, k=")
    test = load_split("test", meld)
    ys = sentiment_ids(test)
    p = np.exp(h.predict_log_proba_batch(train_heads.text_matrix(np.load(cache / "text_test.npy"), test, k)))
    assert r["confusion"] == metrics.confusion(ys, p.argmax(axis=1), 3).tolist() and r["ece"] == round(metrics.ece(p, ys), 6)
    for name, r0 in old["conditions"].items():
        if name != "sentiment_text_only":
            assert {x: v for x, v in met["conditions"][name].items() if x != "wall_s"} == {x: v for x, v in r0.items() if x != "wall_s"}, name
    assert [x for x in met["inputs"] if x not in old["inputs"]] == ["heads/sentiment_text_only.npz"]
    assert "| sentiment_text_only | done |" in md.read_text(encoding="utf-8")

    # a file that no longer matches its manifest record is not scored
    (hd / "sentiment_text_only.npz").write_bytes((tmp_path / "sentiment.npz").read_bytes() + b"\0")
    assert evaluate.main(["--cache", str(cache), "--heads", str(hd), "--meld", str(meld), "--out", str(md), str(js),
                          "--conditions", "sentiment_text_only"]) == 0
    assert "differs from heads/manifest.json" in json.loads(js.read_text(encoding="utf-8"))["conditions"]["sentiment_text_only"]["status"]


def test_audio_stage_drops_rows_without_a_clip(workspace):
    """MELD.Raw lacks a few clips: the audio stage trains on the rest and counts the
    dropped rows per split. the cache rows are shuffled and its encoder states are
    zeros (no --encoder-states), so calibrator B is skipped."""
    root, meld, cache, _ = workspace
    drop_cache, out = root / "cache_drop", root / "heads_drop"
    drop_cache.mkdir()
    rng = np.random.default_rng(1)
    dropped = {}
    for split, n_drop in (("train", 3), ("dev", 2)):
        for name in (f"text_{split}.npy", f"text_{split}_keys.json"):
            (drop_cache / name).write_bytes((cache / name).read_bytes())
        with np.load(cache / f"audio_{split}.npz") as z:
            d = {k: z[k] for k in z.files}
        n = len(d["keys"])
        gone = set(rng.choice(n, n_drop, replace=False).tolist())
        rows = rng.permutation([i for i in range(n) if i not in gone])
        d = {k: v[rows] for k, v in d.items()}
        d["enc_pooled"] = np.zeros_like(d["enc_pooled"])
        np.savez(drop_cache / f"audio_{split}.npz", **d)
        dropped[split] = [str(k) for i, k in enumerate(np.load(cache / f"audio_{split}.npz")["keys"]) if i in gone]
    cfg = yaml.safe_load((root / "heads.yaml").read_text(encoding="utf-8"))
    cfg["text_ctx"]["grid"] = {"k": [0], "class_weight": ["none"], "learning_rate_init": [0.001]}
    cfg["lr"]["C"] = [1]
    (root / "heads_drop.yaml").write_text(yaml.safe_dump(cfg))
    assert train_heads.main(["--cache", str(drop_cache), "--out", str(out), "--config", str(root / "heads_drop.yaml"),
                             "--meld", str(meld)]) == 0
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    af = man["audio_fusion"]
    assert man["audio_rows_dropped"] == {"train": 3, "dev": 2} and af["status"] == "trained"
    for split in ("train", "dev"):
        assert af["rows"][split]["dropped_keys"] == dropped[split]
        assert af["rows"][split]["cached"] == len(load_split(split, meld)) - len(dropped[split])
    assert sum(af["cross_fit"]["fold_sizes"]) == len(load_split("train", meld)) - 3
    assert af["audio_cal_B"]["status"].startswith("skipped") and "all zero" in af["audio_cal_B"]["status"]
    assert af["audio_cal"]["variant"] == "A" and heads.AudioCalibrator.load(out / "audio_cal.npz").input_dim == 24
    assert (out / "fusion.npz").exists() and man["sentiment"]["variant"] == "fused"
    # with A deployed, the ablation code path with nothing dropped refits exactly the deployed A and fusion heads
    assert train_heads.main(["--cache", str(drop_cache), "--out", str(out), "--config", str(root / "heads_drop.yaml"),
                             "--meld", str(meld), "--ablation"]) == 0
    ab = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["ablations"]
    assert ab["reference"]["reproduces_deployed"] == {"audio_cal_A.npz": True, "fusion.npz": True}
    assert ab["rows"] == {s: af["rows"][s]["cached"] for s in ("train", "dev")}


def test_evaluate_leaves_out_test_rows_without_a_clip(workspace):
    """a few test clips missing: audio and fused rows are scored on the rows that have a clip,
    say how many were left out, and carry the text head's f1 on those same rows."""
    root, meld, cache, out = workspace
    if not (out / "fusion.npz").exists():
        assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]) == 0
    ev_cache = root / "cache_eval_drop"
    ev_cache.mkdir(exist_ok=True)
    for name in ("text_test.npy", "text_test_keys.json"):
        (ev_cache / name).write_bytes((cache / name).read_bytes())
    with np.load(cache / "audio_test.npz") as z:
        d = {k: z[k] for k in z.files}
    n = len(d["keys"])
    np.savez(ev_cache / "audio_test.npz", **{k: v[2:] for k, v in d.items()})
    md, js = root / "RESULTS_drop.md", root / "metrics_drop.json"
    assert evaluate.main(["--cache", str(ev_cache), "--heads", str(out), "--meld", str(meld), "--out", str(md), str(js)]) == 0
    met = json.loads(js.read_text(encoding="utf-8"))["conditions"]
    for name in ("audio_raw_tag", "fused_stacked", "fused_poe_uncalibrated", "sentiment_fused"):
        r = met[name]
        assert r["status"] == "done" and r["n"] == n - 2 and r["n_no_audio"] == 2, name
        assert "left out" in r["note"]
    assert 0.0 <= met["fused_stacked"]["text_best_k_same_rows_weighted_f1"] <= 1.0
    assert met["text_best_k"]["n"] == n and "n_no_audio" not in met["text_best_k"]
    # the rows of experiments 8 and 9 that need no transcript embeddings run on the same rows, the rest wait for them
    for name in ("early_commit_audio_2s", "early_commit_audio_4s"):
        assert met[name]["status"] == "done" and met[name]["n"] == n - 2 and met[name]["n_no_audio"] == 2, name
    d = met["disagreement_subset"]
    assert d["n_subset"] + d["n_complement"] == d["n_rows"] == n - 2 and d["n_no_audio"] == 2
    assert "text_best_k_same_rows_weighted_f1" not in d
    for name in ("live_asr_text_only", "live_asr_text", "early_commit_fused_2s"):
        assert met[name]["status"].startswith("pending") and "text_asr_test" in met[name]["status"], name


def test_audit_sample_and_tally(workspace, tmp_path):
    """the disagreement audit draws flagged test rows scored as the service scores them,
    and the tally counts the listener's marks without copying any utterance text."""
    import audit_disagreement as audit

    root, meld, cache, out = workspace
    if not (out / "fusion.npz").exists():
        assert train_heads.main(["--cache", str(cache), "--out", str(out), "--config", str(root / "heads.yaml"), "--meld", str(meld)]) == 0
    sheet = tmp_path / "audit.csv"
    # the flag uses the raw sensevoice posterior, as the service does, which is also what the
    # manifest's jsd threshold was chosen on. the calibrator only gives the audio_label column
    no_cal = tmp_path / "heads_no_cal"
    no_cal.mkdir()
    for name in ("text_ctx.npz", "manifest.json"):
        (no_cal / name).write_bytes((out / name).read_bytes())
    thr = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["thresholds"]["jsd_flag"]
    assert audit.main(["sample", "--n", "5", "--cache", str(cache), "--heads", str(no_cal), "--meld", str(meld),
                       "--audio", str(tmp_path), "--out", str(sheet)]) == 0
    with sheet.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert 0 < len(rows) <= 5 and all(r["direction"] != "none" and float(r["jsd"]) > thr - 1e-4 for r in rows)
    assert all(r[c] == "" for r in rows for c in audit.LISTENER_COLUMNS)
    with_cal = tmp_path / "audit_cal.csv"
    assert (out / "audio_cal.npz").exists()
    assert audit.main(["sample", "--n", "5", "--cache", str(cache), "--heads", str(out), "--meld", str(meld),
                       "--audio", str(tmp_path), "--out", str(with_cal)]) == 0
    with with_cal.open(newline="", encoding="utf-8") as f:
        rows_cal = list(csv.DictReader(f))
    assert [(r["key"], r["jsd"], r["direction"]) for r in rows_cal] == [(r["key"], r["jsd"], r["direction"]) for r in rows]
    for r, mark in zip(rows, ["text", "audio", "both", "neither", "text"]):
        r["right_branch"], r["laughter_source"] = mark, "none"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = tmp_path / "audit.md"
    assert audit.main(["tally", "--csv", str(sheet), "--out", str(md)]) == 0
    text = md.read_text(encoding="utf-8")
    assert f"{len(rows)} of {len(rows)} rows are marked" in text and f"threshold was {thr:g}" in text
    assert not any(r["utterance"] in text for r in rows if len(r["utterance"]) > 12)


def test_audit_draw_is_seeded_and_keeps_order():
    import audit_disagreement as audit

    rows = list(range(100))
    a, b = audit.draw(rows, 10, 0), audit.draw(rows, 10, 0)
    assert a == b and a == sorted(a) and len(set(a)) == 10
    assert audit.draw(rows[:3], 10, 0) == [0, 1, 2]
