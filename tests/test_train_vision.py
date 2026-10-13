"""train_vision.py and evaluate_vision.py end to end on csv subsets with synthetic text,
audio and video caches and synthetic wavs: the speaker rule chosen on dev, the folds and
out-of-fold rows of train_heads.py, the fold heads, the per-clip posteriors, the manifest
section, the test conditions and the paired bootstrap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import evaluate_vision  # noqa: E402
import train_heads  # noqa: E402
import train_vision  # noqa: E402
from test_train_pipeline import subset, synth_caches, write_csv  # noqa: E402

from meld_rt import heads, labels, metrics  # noqa: E402
from meld_rt import vision_features as vf  # noqa: E402
from meld_rt.data import clip_path, emotion_ids, key, load_split  # noqa: E402
from meld_rt.experts.vision import FER_CLASSES, FER_TO_MELD  # noqa: E402

FPS, N_FRAMES, DUR, SR = 10.0, 20, 2.0, 16000
FER_OF = {m: FER_CLASSES.index(f) for f, m in FER_TO_MELD.items()}


def no_face(i: int, emotion: str) -> bool:
    return emotion == "neutral" and i % 7 == 3


def not_cached(i: int, emotion: str) -> bool:
    return emotion == "neutral" and i % 11 == 5


def synth_video(utts, split: str, cache: Path, audio_dir: Path, seed: int) -> None:
    """a wav per clip with an on/off speech envelope, and cache/video_{split}.npz in the merged layout:
    1 to 3 tracks per clip. the speaker's fer logits point at the label and its mouth motion follows the
    envelope; the other tracks are bigger, uninformative and move their mouths more, so only the asd
    rule finds the speaker. some neutral clips have no face, some are missing from the cache."""
    rng = np.random.default_rng(seed)
    (audio_dir / split).mkdir(parents=True, exist_ok=True)
    keys, nf, cols = [], [], {c: [] for c in ("clip", "frame", "track", "box", "lmk", "score", "mouth", "face", "logits", "has_emb")}
    e_row, e_vec = [], []
    t = np.arange(int(DUR * SR)) / SR
    for i, u in enumerate(utts):
        on = rng.uniform(size=8) > 0.4
        amp = np.where(on[np.minimum((t / 0.25).astype(int), 7)], 0.3, 0.01)
        x = np.clip(amp * rng.normal(0, 1, len(t)), -1, 1).astype(np.float32)
        sf.write(audio_dir / split / f"dia{u.dialogue_id}_utt{u.utterance_id}.wav", x, SR, subtype="PCM_16")
        if not_cached(i, u.emotion):
            continue
        ci = len(keys)
        keys.append(key(u))
        nf.append(N_FRAMES)
        if no_face(i, u.emotion):
            continue
        et, ev = vf.audio_envelope(x)
        n_tr = 1 + i % 3
        speaker = i % n_tr
        for k in range(N_FRAMES):
            env_k = float(np.interp(k / FPS, et, ev))
            for tr in range(n_tr):
                is_sp = tr == speaker
                lg = rng.normal(0, 1, 7)
                if is_sp:
                    lg[FER_OF[u.emotion]] += 3.0
                w = 0.15 if is_sp else 0.25 + 0.05 * tr
                cols["clip"].append(ci); cols["frame"].append(k); cols["track"].append(tr)
                cols["box"].append([0.1 + 0.3 * tr, 0.2, w, w]); cols["lmk"].append(rng.uniform(0, 1, 10))
                cols["score"].append(rng.uniform(0.7, 0.95))
                mouth = 0.02 * (env_k + 12) + rng.normal(0, 0.005) if is_sp else 0.4 + rng.normal(0, 0.02)
                cols["mouth"].append(np.nan if k == 0 else mouth)
                cols["face"].append(np.nan if k == 0 else 0.01)
                cols["logits"].append(lg)
                cols["has_emb"].append(k % 2 == 0)
                if k % 2 == 0:
                    e_row.append(len(cols["clip"]) - 1)
                    e_vec.append(rng.normal(0, 1, 512))
    n = len(cols["clip"])
    np.savez(cache / f"video_{split}.npz", keys=np.array(keys), n_frames=np.array(nf, np.int32),
             frame_w=np.full(len(keys), 1280, np.int32), frame_h=np.full(len(keys), 720, np.int32),
             f_clip=np.array(cols["clip"], np.int32), f_frame=np.array(cols["frame"], np.int32), f_track=np.array(cols["track"], np.int32),
             f_box=np.array(cols["box"], np.float32).reshape(n, 4), f_lmk=np.array(cols["lmk"], np.float32).reshape(n, 10),
             f_score=np.array(cols["score"], np.float32), f_mouth=np.array(cols["mouth"], np.float32),
             f_face=np.array(cols["face"], np.float32), f_logits=np.array(cols["logits"], np.float32).reshape(n, 7),
             f_has_emb=np.array(cols["has_emb"]), e_row=np.array(e_row, np.int64),
             e_vec=np.array(e_vec, np.float16).reshape(len(e_vec), 512), fps=np.float32(FPS),
             meta=np.array(json.dumps({"split": split, "errors": {}})))


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    """csv subsets, synthetic caches and wavs, the deployed two-way heads of train_heads.py, then train_vision.py."""
    root = tmp_path_factory.mktemp("vision")
    meld, cache, out, audio = root / "meld", root / "cache", root / "heads", root / "audio"
    for d in (meld, cache, out, audio):
        d.mkdir()
    rng = np.random.default_rng(0)
    centers_t, centers_e = 2.0 * rng.standard_normal((7, 768)), 2.0 * rng.standard_normal((7, 1024))
    for split, n in (("train", 40), ("dev", 12), ("test", 15)):
        utts = subset(split, n)
        write_csv(utts, meld / f"{split}_sent_emo.csv")
        synth_caches(utts, split, cache, rng, centers_t, centers_e)
        synth_video(utts, split, cache, audio, {"train": 1, "dev": 2, "test": 3}[split])
    cfg = yaml.safe_load((ROOT / "configs" / "heads.yaml").read_text(encoding="utf-8"))
    cfg["text_ctx"]["grid"] = {"k": [0], "class_weight": ["none", "balanced"], "learning_rate_init": [0.001]}
    cfg["text_ctx"]["mlp"].update({"max_epochs": 6, "patience": 3})
    cfg["lr"]["C"] = [0.1, 1]
    cfg["fusion"]["cross_fit_folds"] = 3
    (root / "heads.yaml").write_text(yaml.safe_dump(cfg))
    base = ["--cache", str(cache), "--config", str(root / "heads.yaml"), "--meld", str(meld)]
    assert train_heads.main([*base, "--out", str(out)]) == 0
    before = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    deployed = {p.name: p.read_bytes() for p in out.glob("*.npz")}
    assert train_vision.main([*base, "--heads", str(out), "--audio", str(audio)]) == 0
    return {"root": root, "meld": meld, "cache": cache, "out": out, "audio": audio, "cfg": cfg, "before": before,
            "deployed": deployed, "manifest": json.loads((out / "manifest.json").read_text(encoding="utf-8"))}


def test_heads_and_manifest_section(ws):
    out, man, before = ws["out"], dict(ws["manifest"]), ws["before"]
    vis = man.pop("vision")
    assert man == before                                   # every other key as it was
    assert {p.name: p.read_bytes() for p in out.glob("*.npz") if p.name in ws["deployed"]} == ws["deployed"]
    vcal = heads.VisionCalibrator.load(out / "vision_cal.npz")
    f3, s3 = heads.Fusion3.load(out / "fusion3.npz"), heads.Sentiment3.load(out / "sentiment3.npz")
    assert vcal.input_dim == (528 if vis["variant"] == "B" else 16) and vcal.rule == vis["rule"] and vcal.variant == vis["variant"]
    assert f3.input_dim == s3.input_dim == 29 and f3.n_classes == 7 and s3.n_classes == 3
    assert vis["status"] == "trained" and set(vis["rules"]) == set(vf.RULES)
    # the synthetic speaker is found only by the asd rule, and the rule is the dev argmax
    assert vis["rule"] == "asd" == max(vf.RULES, key=lambda r: vis["rules"][r]["dev"]["weighted_f1"])
    assert vis["rules"]["asd"]["dev"]["weighted_f1"] > vis["rules"]["largest"]["dev"]["weighted_f1"]
    assert vis["calibrator"]["variant"] == vis["variant"] and set(vis["calibrator"]) >= {"A", "B"}
    better = "B" if vis["calibrator"]["B"]["dev"]["weighted_f1"] > vis["calibrator"]["A"]["dev"]["weighted_f1"] else "A"
    assert vis["variant"] == better
    assert 0 < vis["face_coverage"]["train"] < 1 and 0 < vis["face_coverage"]["dev"] <= 1
    dis = vis["rule_disagreement"]["dev"]
    assert 0 < dis["largest|asd"] <= 1 and dis["n_face_clips"] > 0
    assert vis["video_cache"]["train"]["csv_rows_without_clip"] > 0
    assert set(vis["thresholds"]) >= {"low_conf", "high_conf", "source"} and 0 <= vis["jsd_face_flag"] <= 1
    assert set(vis["files"]) == {"vision_cal.npz", "fusion3.npz", "sentiment3.npz"}
    assert vis["fusion3"]["two_way_dev_manifest"] == before["audio_fusion"]["fusion"]["dev"]["weighted_f1"]
    assert vis["fusion3"]["two_way_dev_recomputed"]["weighted_f1"] == pytest.approx(before["audio_fusion"]["fusion"]["dev"]["weighted_f1"], abs=1e-4)
    assert vis["command"].startswith("python tools/train_vision.py")


def test_folds_are_train_heads_folds(ws):
    vis, before = ws["manifest"]["vision"], ws["before"]
    cf = vis["cross_fit"]
    assert cf["fold_sizes"] == before["audio_fusion"]["cross_fit"]["fold_sizes"] and cf["same_fold_sizes_as_train_heads"]
    assert cf["reproduces_deployed_fusion"] is True          # the two-way fusion refit on these rows is heads/fusion.npz
    assert cf["oof_train_scores"]["text"]["weighted_f1"] == pytest.approx(before["audio_fusion"]["cross_fit"]["oof_text_train_scores"]["weighted_f1"], abs=1e-4)
    # fold_of_train.json against train_heads.cross_fit's own assignment on the audio rows
    train = load_split("train", ws["meld"])
    _, keep, _ = train_heads.read_audio_cache(ws["cache"], "train", train)
    kept = [train[i] for i in keep]
    y = emotion_ids(kept)
    _, fold_id, _ = train_heads.cross_fit(lambda X, yy: heads.NpzHead(heads.head_params(
        heads.lr_params(np.zeros((7, 1)), np.zeros(7)), np.zeros(1), np.ones(1), labels.EMOTIONS)),
        np.zeros((len(kept), 1)), y, train_heads.groups_of(kept), ws["cfg"]["fusion"]["cross_fit_folds"], ws["cfg"]["seed"])
    fold_of = json.loads((ws["cache"] / "folds" / "fold_of_train.json").read_text(encoding="utf-8"))
    assert [fold_of[key(u)] for u in kept] == fold_id.tolist()


def test_fold_heads_reproduce_the_out_of_fold_rows(ws):
    cache, meld = ws["cache"], ws["meld"]
    post = dict(np.load(cache / "vision_post_train.npz"))
    meta = json.loads(str(post["meta"]))
    assert meta["out_of_fold"] and meta["rule"] == ws["manifest"]["vision"]["rule"]
    train = load_split("train", meld)
    assert [str(k) for k in post["keys"]] == [key(u) for u in train]
    d, keep, _ = train_heads.read_audio_cache(cache, "train", train)
    emb = train_heads.load_text_cache(cache, "train", train)
    n_folds = ws["cfg"]["fusion"]["cross_fit_folds"]
    fold = post["fold"][keep]
    assert set(fold.tolist()) == set(range(n_folds)) and (post["fold"] >= 0).sum() == len(keep)
    video = vf.load_video_cache("train", cache)
    env_cache = cache / "envelope_train.npz"
    stamp = env_cache.stat().st_mtime_ns
    env = vf.load_envelopes([key(u) for u in train], [clip_path(u, ws["audio"]) for u in train], env_cache)
    assert env_cache.stat().st_mtime_ns == stamp     # the wavs did not change: read from the cache
    rule = meta["rule"]
    T = train_vision.evidence_tables(video, env, train, (rule,))[rule]
    for f in range(n_folds):
        rows = keep[fold == f]
        ht = heads.TextContextHead.load(cache / "folds" / f"fold{f}_text_ctx.npz")
        ha = heads.AudioCalibrator.load(cache / "folds" / f"fold{f}_audio_cal.npz")
        hv = heads.VisionCalibrator.load(cache / "folds" / f"fold{f}_vision_cal.npz")
        h3 = heads.Fusion3.load(cache / "folds" / f"fold{f}_fusion3.npz")
        assert ht.temperature == ha.temperature == hv.temperature == 1.0 and int(h3.params["fold"]) == f
        lp_t = ht.predict_log_proba_batch(train_heads.text_matrix(emb, train, ht.k)[rows])
        assert np.allclose(np.exp(lp_t), post["p_text"][rows], atol=1e-6)
        lp_a = ha.predict_log_proba_batch(train_heads.audio_matrix(d, ha.uses_encoder_states)[np.isin(keep, rows)])
        assert np.allclose(np.exp(lp_a), post["p_audio"][rows], atol=1e-6)
        X = train_vision.features_of(T, "B" if hv.uses_embedding else "A")[rows]
        p_v = train_vision.face_proba(hv, X, T["present"][rows])
        assert np.allclose(p_v, post["p_face"][rows], atol=1e-6)
        sel = np.isin(keep, rows)
        F3 = train_vision.fusion3_rows(lp_t, lp_a, np.log(np.clip(p_v, heads.EPS, 1)), {k: v[sel] for k, v in d.items()},
                                       T["present"][rows], T["presence"][rows])
        assert np.allclose(np.exp(h3.predict_log_proba_batch(F3)), post["fused3"][rows], atol=1e-5)
    # rows without a face carry the uniform face posterior, the rows without audio are nan
    absent = keep[~post["present"][keep]]
    assert len(absent) and np.allclose(post["p_face"][absent], 1 / 7)
    assert np.isnan(post["fused3"][~post["has_audio"]]).all()


def test_dev_and_test_posteriors(ws):
    cache, out = ws["cache"], ws["out"]
    vis = ws["manifest"]["vision"]
    dev = dict(np.load(cache / "vision_post_dev.npz"))
    test = dict(np.load(cache / "vision_post_test.npz"))
    assert not json.loads(str(dev["meta"]))["out_of_fold"]
    for z in (dev, test):
        ok = z["has_audio"]
        assert np.allclose(z["fused3"][ok].sum(axis=1), 1, atol=1e-5) and np.allclose(z["p_face"].sum(axis=1), 1, atol=1e-5)
        assert np.allclose(z["p_face"][~z["present"]], 1 / 7) and (z["track"][~z["present"]] == -1).all()
    assert {"p_text_live", "fused3_live", "fused2_live", "asr_degraded"} <= set(test)
    deg = test["asr_degraded"]
    assert deg.any() and np.allclose(test["fused3_live"][deg], 1 / 7)
    # the jsd threshold is the dev quantile over the dev rows with a face
    m = dev["has_audio"] & dev["present"]
    q = ws["cfg"]["thresholds"]["jsd_flag"]["quantile"]
    assert vis["jsd_face_flag"] == pytest.approx(float(np.quantile(dev["jsd_face"][m], q)), abs=1e-4)
    # fusion3 on test, recomputed from the saved heads and the caches, equals the posterior file
    utts = load_split("test", ws["meld"])
    th, ca = heads.TextContextHead.load(out / "text_ctx.npz"), heads.AudioCalibrator.load(out / "audio_cal.npz")
    vcal, f3 = heads.VisionCalibrator.load(out / "vision_cal.npz"), heads.Fusion3.load(out / "fusion3.npz")
    d, keep, _ = train_heads.read_audio_cache(cache, "test", utts)
    lp_t = th.predict_log_proba_batch(train_heads.text_matrix(train_heads.load_text_cache(cache, "test", utts), utts, th.k))[keep]
    lp_a = ca.predict_log_proba_batch(train_heads.audio_matrix(d, ca.uses_encoder_states))
    video = vf.load_video_cache("test", cache)
    env = vf.load_envelopes([key(u) for u in utts], [ws["audio"] / "test" / f"dia{u.dialogue_id}_utt{u.utterance_id}.wav" for u in utts])
    ev = [vf.face_evidence(vf.clip_rows(video, key(utts[i])), *env[key(utts[i])], vis["rule"]) for i in keep]
    p_v = np.stack([vf.calibrated_face(vcal, e) for e in ev])
    assert np.allclose(p_v, test["p_face"][keep], atol=1e-6)
    F3 = np.stack([vf.build_fusion3_features(lp_t[j], lp_a[j], np.log(np.clip(p_v[j], heads.EPS, 1)), d["asr_conf"][j],
                                             d["event_probs"][j][heads.LAUGHTER_IDX], d["emo_probs"][j][heads.UNKNOWN_IDX],
                                             e.present, e.presence) for j, e in enumerate(ev)])
    assert np.allclose(np.exp(f3.predict_log_proba_batch(F3)), test["fused3"][keep], atol=1e-5)


def test_evaluate_vision(ws):
    root = ws["root"]
    md, js = root / "RESULTS_vision.md", root / "metrics_vision.json"
    args = ["--cache", str(ws["cache"]), "--heads", str(ws["out"]), "--meld", str(ws["meld"]), "--audio", str(ws["audio"]), "--out", str(md), str(js)]
    assert evaluate_vision.main(args) == 0
    met = json.loads(js.read_text(encoding="utf-8"))
    C = met["conditions"]
    test = load_split("test", ws["meld"])
    y = emotion_ids(test)
    for name in ("vision_raw_fer", "vision_cal", "fused3_stacked", "fused_stacked", "live_asr_fused3", "live_asr_text",
                 "fused3_stacked_face_rows", "live_asr_fused3_face_rows", "sentiment3"):
        r = C[name]
        assert r["status"] == "done", (name, r["status"])
        assert sum(map(sum, r["confusion"])) == r["n"] and 0 <= r["weighted_f1"] <= 1, name
    assert C["vision_raw_fer"]["n"] == len(test) and 0 < C["vision_cal"]["face_coverage"] <= 1
    # the rows are scored on the posterior file's predictions
    z = dict(np.load(ws["cache"] / "vision_post_test.npz"))
    ok = z["has_audio"]
    assert C["fused3_stacked"]["confusion"] == metrics.confusion(y[ok], z["fused3"][ok].argmax(axis=1), 7).tolist()
    assert C["live_asr_fused3"]["confusion"] == metrics.confusion(y[ok], z["fused3_live"][ok].argmax(axis=1), 7).tolist()
    assert C["fused_stacked"]["confusion"] == metrics.confusion(y[ok], z["fused2"][ok].argmax(axis=1), 7).tolist()
    # the two-way reference equals evaluate.py's fused_stacked row on the same caches and heads
    import evaluate
    ref, _ = evaluate.run_conditions(ws["cache"], ws["out"], ws["meld"], {"fused_stacked", "live_asr_text"})
    for name in ("fused_stacked", "live_asr_text"):
        assert C[name]["confusion"] == ref[name]["confusion"] and C[name]["ci95"] == ref[name]["ci95"], name
    comp = met["comparisons"]
    g = comp["gold_all"]
    assert g["n"] == int(ok.sum()) and g["ci95"]["weighted_f1"][0] <= g["diff"]["weighted_f1"] <= g["ci95"]["weighted_f1"][1]
    assert g["diff"]["weighted_f1"] == pytest.approx(C["fused3_stacked"]["weighted_f1"] - C["fused_stacked"]["weighted_f1"], abs=1e-5)
    assert met["claim"]["face_adds"] == (g["ci95"]["weighted_f1"][0] > 0)
    assert comp["gold_face_rows"]["n"] == C["fused3_stacked_face_rows"]["n"]
    # label flips of fused3 against the two-way fusion, from the posterior file's predictions
    for name, a, b in (("gold", "fused3", "fused2"), ("live", "fused3_live", "fused2_live")):
        pa, pb, yy = z[a][ok].argmax(axis=1), z[b][ok].argmax(axis=1), y[ok]
        f = met["label_flips"][name]
        assert (f["n"], f["changed"], f["wrong_to_right"], f["right_to_wrong"]) == (
            len(yy), int((pa != pb).sum()), int(((pa != pb) & (pa == yy)).sum()), int(((pa != pb) & (pb == yy)).sum())), name
    text = md.read_text(encoding="utf-8")
    for header in ("## Emotion (7 classes)", "## Paired differences", "## Label flips", "## Per-class F1", "## Speaker rules on dev"):
        assert header in text
    assert "| asd |" in text


def test_label_flips():
    y = np.array([0, 1, 2, 3, 4, 5])
    a = np.array([0, 1, 0, 3, 5, 6])
    b = np.array([0, 2, 2, 3, 4, 0])
    # changed at 1 (b wrong, a right), 2 (b right, a wrong), 4 (b right, a wrong), 5 (both wrong)
    assert evaluate_vision.label_flips(y, a, b) == {"n": 6, "changed": 4, "wrong_to_right": 1, "right_to_wrong": 2}


def test_paired_bootstrap_diff():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 7, n)
    groups = np.repeat(np.arange(60), 5)
    a = np.where(rng.uniform(size=n) < 0.6, y, rng.integers(0, 7, n))
    same = evaluate_vision.paired_bootstrap_diff(y, a, a, groups, 7, 200, 0)
    assert same["diff"]["weighted_f1"] == 0 and same["ci95"]["weighted_f1"] == [0, 0] and not same["excludes_zero"]["weighted_f1"]
    assert same["share_at_or_below_zero"]["weighted_f1"] == 1.0
    # b predicts a class that never occurs: its f1 is 0 on every resample, so the difference interval is
    # the bootstrap interval of a alone, resample for resample (same seed, same draws as metrics.bootstrap_by_group)
    y6 = np.minimum(y, 5)
    a6 = np.minimum(a, 5)
    b = np.full(n, 6)
    d = evaluate_vision.paired_bootstrap_diff(y6, a6, b, groups, 7, 300, 0)
    ref = metrics.bootstrap_by_group(y6, a6, groups, 7, 300, 0)
    assert np.allclose(d["ci95"]["weighted_f1"], ref["weighted_f1"], atol=1e-6) and d["excludes_zero"]["weighted_f1"]
    assert d["diff"]["accuracy"] == pytest.approx(metrics.accuracy(metrics.confusion(y6, a6, 7)), abs=1e-6)
    assert d["n_groups"] == 60 and d["n_resamples"] == 300
