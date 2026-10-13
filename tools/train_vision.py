"""train the vision heads of the three-input extension (extension spec 2.3) from
the text, audio and video caches, the clip wavs (audio envelopes) and the
deployed text and audio heads and manifest of tools/train_heads.py.

1. speaker rule: for each rule (largest, present, asd, mouth) calibrator A is fit
   on the train rows with a face (lr_grid over C x class weights, chosen on the dev
   rows with a face) and scored on all dev rows, rows without a face getting the
   uniform posterior (label neutral). the rule with the best dev weighted f1 is kept.
2. calibrator A (16 inputs) against B (528, plus the mean fer embedding) with that
   rule, temperature on dev; the better dev weighted f1 (all dev rows) is deployed
   as heads/vision_cal.npz (A on a tie).
3. out-of-fold expert posteriors on train: text and audio with the deployed
   selections and vision with the deployed calibrator's choice, through
   train_heads.cross_fit with train_heads' rows, groups, folds and seed (asserted:
   the two-way fusion refit on these rows must equal heads/fusion.npz). fusion3 and
   sentiment3 by lr_grid over the 29 features of meld_rt.vision_features, temperature on dev.
4. selector thresholds of the vision mode from fusion3's calibrated dev posteriors.
5. jsd_face_flag: the 0.8 quantile of the dev jsd between the gold-text posterior and
   the raw face posterior (averaged fer, chosen track) on the dev rows with a face.
6. fold heads for the commit-policy traces: cache/folds/fold{k}_{text_ctx,audio_cal,
   vision_cal,fusion3}.npz (expert heads as fitted in the cross-fit, so fold k's heads on
   fold k's rows give exactly the out-of-fold rows; fusion3 refit on the other folds'
   out-of-fold rows with the deployed choice, temperature on dev) and cache/folds/fold_of_train.json.
7. per-clip posteriors cache/vision_post_{split}.npz (train out of fold; dev and test
   with the deployed heads; test also with the sensevoice transcripts and the asr gate).

writes heads/vision_cal.npz, heads/fusion3.npz, heads/sentiment3.npz and a "vision"
section in heads/manifest.json (every other key stays as it is). test labels are never read.

usage: python tools/train_vision.py --cache cache/ --heads heads/ [--out heads/] [--audio audio/] [--post-splits train,dev,test]
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
from collections.abc import Callable  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from train_heads import (  # noqa: E402
    audio_fitter,
    audio_matrix,
    calibrate,
    choose_thresholds,
    cross_fit,
    file_record,
    fusion_rows,
    groups_of,
    load_text_cache,
    lr_grid,
    read_audio_cache,
    same_weights,
    text_fitter,
    text_matrix,
)

from meld_rt import heads, labels, metrics  # noqa: E402
from meld_rt import vision_features as vf  # noqa: E402
from meld_rt.data import Utt, clip_path, emotion_ids, key, load_split, sentiment_ids  # noqa: E402
from meld_rt.data import class_weights as emotion_class_weights  # noqa: E402
from meld_rt.discrepancy import jsd  # noqa: E402

N_EMO = len(labels.EMOTIONS)
N_SENT = len(labels.SENTIMENTS)
PRESENT_COL = vf.DIM_A - 1      # the 'present' flag, last of features A (and of A inside B)
UNIFORM_LOG = float(np.log(1.0 / N_EMO))
EMPTY_ENV = (np.zeros(0), np.zeros(0))


# ---------------------------------------------------------------- inputs

def need(path: Path, hint: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{path} is missing ({hint})")
    return path


def load_video(cache: Path, split: str) -> dict:
    need(cache / f"video_{split}.npz", "tools/extract_video_features.py --merge")
    return vf.load_video_cache(split, cache)


def envelopes(utts: list[Utt], audio_dir: Path, cache: Path, split: str) -> dict:
    """key -> (t, log rms) of every csv row's wav, cached in cache/envelope_{split}.npz."""
    t0 = time.perf_counter()
    env = vf.load_envelopes([key(u) for u in utts], [clip_path(u, audio_dir) for u in utts], cache / f"envelope_{split}.npz")
    n_missing = sum(1 for v in env.values() if not len(v[0]))
    print(f"{split}: envelopes of {len(env)} clips ({n_missing} without a wav) in {time.perf_counter() - t0:.1f} s", flush=True)
    return env


def evidence_tables(video: dict, env: dict, utts: list[Utt], rules=vf.RULES) -> dict[str, dict]:
    """per rule and csv row: features A, the chosen track's mean embedding, the raw face
    posterior and the evidence scalars. the whole clip is the window."""
    n = len(utts)
    T = {r: {"XA": np.zeros((n, vf.DIM_A), np.float32), "emb": np.zeros((n, vf.EMBED_DIM), np.float32),
             "p_raw": np.full((n, N_EMO), 1.0 / N_EMO), "present": np.zeros(n, bool), "track": np.full(n, -1, np.int64),
             "presence": np.zeros(n), "n_tracks": np.zeros(n, np.int64), "asd_corr": np.zeros(n),
             "n_frames": np.zeros(n, np.int64), "fallback": np.zeros(n, bool)} for r in rules}
    in_cache = np.zeros(n, bool)
    for i, u in enumerate(utts):
        k = key(u)
        in_cache[i] = k in video["index"]
        rows = vf.clip_rows(video, k)
        et, ev = env.get(k, EMPTY_ENV)
        for r in rules:
            e = vf.face_evidence(rows, et, ev, r)
            t = T[r]
            t["XA"][i], t["emb"][i], t["p_raw"][i] = vf.vision_features_A(e), e.emb, e.p_face
            t["present"][i], t["track"][i] = e.present, -1 if e.track is None else e.track
            t["presence"][i], t["n_tracks"][i], t["asd_corr"][i] = e.presence, e.n_tracks, e.asd_corr
            t["n_frames"][i], t["fallback"][i] = e.n_frames, e.fallback
    for r in rules:
        T[r]["in_cache"] = in_cache
    return T


def features_of(T: dict, variant: str) -> np.ndarray:
    return T["XA"] if variant == "A" else np.concatenate([T["XA"], T["emb"]], axis=1)


def face_proba(h: heads.NpzHead, X: np.ndarray, present: np.ndarray) -> np.ndarray:
    """calibrated face posterior; uniform on the rows without a face."""
    p = np.full((len(X), N_EMO), 1.0 / N_EMO)
    if present.any():
        p[present] = np.exp(h.predict_log_proba_batch(X[present]))
    return p


def fusion3_rows(lp_text: np.ndarray, lp_audio: np.ndarray, lp_face: np.ndarray, d: dict, present: np.ndarray,
                 presence: np.ndarray) -> np.ndarray:
    """the 29 fusion3 features per row; asr_conf, p_laughter and p_unknown as train_heads.fusion_rows takes them."""
    return np.stack([vf.build_fusion3_features(lp_text[i], lp_audio[i], lp_face[i], d["asr_conf"][i],
                                               d["event_probs"][i][heads.LAUGHTER_IDX], d["emo_probs"][i][heads.UNKNOWN_IDX],
                                               bool(present[i]), float(presence[i]))
                     for i in range(len(lp_text))]).astype(np.float32)


def summary(y: np.ndarray, proba: np.ndarray) -> dict:
    n_cls = proba.shape[1]
    return {k: round(v, 4) for k, v in metrics.summary(metrics.confusion(y, proba.argmax(axis=1), n_cls)).items()}


# ---------------------------------------------------------------- fitting

def fit_vision_grid(X: dict, y: dict, present: dict, w_fn: Callable[[str], np.ndarray], cfg: dict, seed: int,
                    schemes: list[str], **meta) -> tuple[heads.VisionCalibrator, dict]:
    """lr_grid on the train rows with a face (chosen on the dev rows with a face), temperature
    on the dev rows with a face, then scored on all dev rows (no face: uniform)."""
    tr, dv = present["train"], present["dev"]
    h, choice, scores = lr_grid(X["train"][tr], y["train"][tr], X["dev"][dv], y["dev"][dv], labels.EMOTIONS, cfg["lr"]["C"],
                                schemes, w_fn, cfg["lr"], seed, **meta)
    h = heads.VisionCalibrator(h.params)
    cal = calibrate(h, X["dev"][dv], y["dev"][dv])
    p = face_proba(h, X["dev"], dv)
    return h, {**choice, "input_dim": h.input_dim, "dev": summary(y["dev"], p), "dev_face_rows": summary(y["dev"][dv], p[dv]),
               "dev_scores_face_rows": scores, "calibration_dev": cal}


def vision_fitter(choice: dict, w: np.ndarray, cfg: dict, seed: int, **meta) -> Callable[[np.ndarray, np.ndarray], heads.NpzHead]:
    """refits the deployed calibrator's choice on a fold's rows with a face (the present flag of the features)."""
    fit = audio_fitter(choice, w, cfg, seed)

    def fit_vision(X: np.ndarray, yy: np.ndarray) -> heads.NpzHead:
        m = X[:, PRESENT_COL] > 0.5
        p = dict(fit(X[m], yy[m]).params)
        p.update({k: np.array(v) for k, v in meta.items()})
        return heads.VisionCalibrator(p)

    return fit_vision


def capture(fit_fn: Callable, store: list) -> Callable:
    """keeps every head cross_fit fits, in fold order."""

    def f(X: np.ndarray, yy: np.ndarray):
        h = fit_fn(X, yy)
        store.append(h)
        return h

    return f


def rule_disagreement(T: dict, rules=vf.RULES) -> dict:
    """over the clips with a face: the share whose chosen track differs, per pair of rules,
    and the share with more than one track."""
    present = T[rules[0]]["present"]
    if not present.any():
        return {"n_face_clips": 0}
    out = {"n_face_clips": int(present.sum()),
           "multi_track_share": round(float((T[rules[0]]["n_tracks"][present] > 1).mean()), 4),
           "any_rule_differs": round(float(np.mean([len({int(T[r]["track"][i]) for r in rules}) > 1
                                                    for i in np.flatnonzero(present)])), 4)}
    for a, b in combinations(rules, 2):
        out[f"{a}|{b}"] = round(float((T[a]["track"][present] != T[b]["track"][present]).mean()), 4)
    return out


# ---------------------------------------------------------------- deployed posteriors (also used by evaluate_vision.py)

def live_text(cache: Path, heads_dir: Path, manifest: dict, utts: list[Utt], aud: dict, keep: np.ndarray,
              text_head: heads.TextContextHead) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """deployed text head on the sensevoice transcripts of test (evaluate.Experiments.lp_text_asr) and the
    service's asr gate (empty or asr_conf below min_asr_conf). (None, None, {}) without cache/text_asr_test.npy."""
    import evaluate

    ex = evaluate.Experiments(cache, heads_dir, manifest, utts, None, None, aud, keep, text_head, None,
                              file_record(cache / "audio_test.npz") if (cache / "audio_test.npz").exists() else None)
    lp = ex.lp_text_asr("")
    if lp is None:
        return None, None, {}
    failed, low, min_conf = ex.gate
    return lp, failed | low, {"n_empty_transcripts": int(failed.sum()), "n_low_asr_conf": int(low.sum()),
                              "n_degraded": int((failed | low).sum()), "min_asr_conf": min_conf}


def deployed_posteriors(split: str, utts: list[Utt], cache: Path, heads_dir: Path, manifest: dict, video: dict, env: dict,
                        live: bool = False, vision_dir: Path | None = None) -> dict:
    """posteriors of the deployed heads on a split. per csv row: the face evidence of the manifest rule,
    the raw and calibrated face posteriors, features A. per row with a cached audio clip ('keep'):
    text (gold), audio, two-way fusion, fusion3, sentiment3, and with live=True the same on the
    sensevoice transcripts with the asr gate (degraded rows uniform, as evaluate.degrade).
    text, audio and two-way heads come from heads_dir, the vision heads from vision_dir (default heads_dir)."""
    import evaluate

    vision_dir = vision_dir or heads_dir
    rule = manifest["vision"]["rule"]
    text_head = heads.TextContextHead.load(heads_dir / "text_ctx.npz")
    cal_a = heads.AudioCalibrator.load(heads_dir / "audio_cal.npz")
    fh2 = heads.Fusion.load(heads_dir / "fusion.npz")
    vcal = heads.VisionCalibrator.load(vision_dir / "vision_cal.npz")
    f3 = heads.Fusion3.load(vision_dir / "fusion3.npz")
    s3 = heads.Sentiment3.load(vision_dir / "sentiment3.npz") if (vision_dir / "sentiment3.npz").exists() else None
    T = evidence_tables(video, env, utts, (rule,))[rule]
    Xv = features_of(T, "B" if vcal.uses_embedding else "A")
    p_face = face_proba(vcal, Xv, T["present"])
    P = {"keys": [key(u) for u in utts], "rule": rule, "variant": "B" if vcal.uses_embedding else "A",
         **{k: T[k] for k in ("present", "track", "presence", "n_tracks", "asd_corr", "n_frames", "fallback", "in_cache")},
         "p_face_raw": T["p_raw"], "p_face": p_face, "feats_A": T["XA"]}
    emb = load_text_cache(cache, split, utts)
    lp_text_all = text_head.predict_log_proba_batch(text_matrix(emb, utts, text_head.k))
    r = read_audio_cache(cache, split, utts)
    if r is None:
        P["keep"] = np.zeros(0, np.int64)
        return P
    aud, keep, dropped = r
    P.update(keep=keep, no_audio=dropped, aud=aud)
    lp_t, lp_a = lp_text_all[keep], cal_a.predict_log_proba_batch(audio_matrix(aud, cal_a.uses_encoder_states))
    lp_f = np.log(np.clip(p_face[keep], heads.EPS, 1.0))
    pres, presence = T["present"][keep], T["presence"][keep]
    F3 = fusion3_rows(lp_t, lp_a, lp_f, aud, pres, presence)
    P.update(p_text=np.exp(lp_t), p_audio=np.exp(lp_a), F3=F3, fused3=np.exp(f3.predict_log_proba_batch(F3)),
             fused2=np.exp(fh2.predict_log_proba_batch(fusion_rows(lp_t, lp_a, aud))))
    if s3 is not None:
        P["sent3"] = np.exp(s3.predict_log_proba_batch(F3))
    P["jsd_face"] = np.array([jsd(P["p_text"][i], T["p_raw"][keep][i]) if pres[i] else 0.0 for i in range(len(keep))])
    if live:
        lp_live, bad, gate = live_text(cache, heads_dir, manifest, utts, aud, keep, text_head)
        if lp_live is not None:
            F3l = fusion3_rows(lp_live, lp_a, lp_f, aud, pres, presence)
            P.update(p_text_live=evaluate.degrade(np.exp(lp_live), bad), asr_degraded=bad, gate=gate,
                     fused3_live=evaluate.degrade(np.exp(f3.predict_log_proba_batch(F3l)), bad),
                     fused2_live=evaluate.degrade(np.exp(fh2.predict_log_proba_batch(fusion_rows(lp_live, lp_a, aud))), bad),
                     fused3_live_without_gate=np.exp(f3.predict_log_proba_batch(F3l)))
    return P


def write_post(path: Path, P: dict, split: str, oof: bool, extra_meta: dict | None = None) -> dict:
    """cache/vision_post_{split}.npz: csv-row arrays; rows without a cached audio clip are nan
    in the audio-dependent fields."""
    n = len(P["keys"])
    keep = P.get("keep", np.zeros(0, np.int64))

    def scatter(a: np.ndarray) -> np.ndarray:
        out = np.full((n, *a.shape[1:]), np.nan, dtype=np.float32)
        out[keep] = a
        return out

    has_audio = np.zeros(n, bool)
    has_audio[keep] = True
    arrays = {"keys": np.array(P["keys"]), "has_audio": has_audio, "in_video_cache": P["in_cache"], "present": P["present"],
              "track": P["track"], "n_frames": P["n_frames"], "presence": P["presence"].astype(np.float32),
              "n_tracks": P["n_tracks"], "asd_corr": P["asd_corr"].astype(np.float32), "fallback": P["fallback"],
              "p_face_raw": P["p_face_raw"].astype(np.float32), "p_face": P["p_face"].astype(np.float32),
              "feats_A": P["feats_A"].astype(np.float32), "fold": P.get("fold", np.full(n, -1, np.int64))}
    for f in ("p_text", "p_audio", "fused3", "fused2", "sent3", "jsd_face", "p_text_live", "fused3_live", "fused2_live"):
        if f in P:
            arrays[f] = scatter(np.asarray(P[f]))
    if "asr_degraded" in P:
        arrays["asr_degraded"] = scatter(P["asr_degraded"].astype(np.float32)) == 1.0
    meta = {"split": split, "rule": P["rule"], "variant": P["variant"], "out_of_fold": oof, "classes": labels.EMOTIONS,
            "fields": {
                "p_face_raw": "mean per-frame fer posterior of the chosen track, MELD order (uniform without a face)",
                "p_face": "calibrated face posterior" + (" (out of fold)" if oof else " (deployed heads/vision_cal.npz)"),
                "feats_A": "vision features A (meld_rt.vision_features.FEATURES_A)",
                "p_text": "text posterior, gold text" + (", out of fold" if oof else ""),
                "fused3": "fusion3 posterior" + (", fold fusion3 on out-of-fold rows" if oof else ", gold text"),
                "fused2": "deployed two-way fusion, gold text (absent for train)",
                "jsd_face": "jsd of p_text against p_face_raw (0 without a face)",
                "*_live": "sensevoice transcripts with the service's asr gate (test only)",
                "fold": "train fold of the cross-fit (-1 elsewhere)"},
            **(extra_meta or {})}
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return {"file": path.name, "rows": n, **file_record(path)}


# ---------------------------------------------------------------- main training

def train(args: argparse.Namespace, cfg: dict, seed: int) -> dict:
    t_all = time.perf_counter()
    heads_dir, out, cache = args.heads, args.out, args.cache
    manifest = json.loads(need(heads_dir / "manifest.json", "run tools/train_heads.py first").read_text(encoding="utf-8"))
    af = manifest.get("audio_fusion") or {}
    if af.get("status") != "trained":
        raise SystemExit("heads/manifest.json has no trained audio stage (run tools/train_heads.py without --text-only)")
    for f in ("text_ctx.npz", "audio_cal.npz", "fusion.npz"):
        need(heads_dir / f, "run tools/train_heads.py first")
    vcfg = cfg.get("vision") or {}
    v_schemes = list(vcfg.get("class_weight", cfg["audio_cal"]["class_weight"]))
    n_folds = int(cfg["fusion"]["cross_fit_folds"])
    q_face = float(vcfg.get("jsd_face_quantile", cfg["thresholds"]["jsd_flag"]["quantile"]))
    out.mkdir(parents=True, exist_ok=True)

    utts = {s: load_split(s, args.meld) for s in ("train", "dev")}
    y_all = {s: emotion_ids(utts[s]) for s in utts}
    video = {s: load_video(cache, s) for s in utts}
    env = {s: envelopes(utts[s], args.audio, cache, s) for s in utts}
    t0 = time.perf_counter()
    T = {s: evidence_tables(video[s], env[s], utts[s]) for s in utts}
    print(f"face evidence, {len(vf.RULES)} rules: {time.perf_counter() - t0:.1f} s", flush=True)
    rep: dict = {"status": "trained", "rules_tried": list(vf.RULES), "selection": "dev weighted f1, all dev rows (no face: uniform, label neutral)"}
    rep["video_cache"] = {s: {"clips": len(video[s]["keys"]), "csv_rows": len(utts[s]),
                              "csv_rows_without_clip": int((~T[s]["largest"]["in_cache"]).sum()),
                              "errors": len((video[s]["meta_json"] or {}).get("errors", {})),
                              "fps": video[s]["fps_value"]} for s in utts}
    rep["face_coverage"] = {s: round(float(T[s]["largest"]["present"].mean()), 4) for s in utts}
    rep["rule_disagreement"] = {s: rule_disagreement(T[s]) for s in utts}
    present = {r: {s: T[s][r]["present"] for s in utts} for r in vf.RULES}
    if not present["largest"]["train"].any() or not present["largest"]["dev"].any():
        raise SystemExit("no train or dev clip has a face")
    face_train = [u for u, p in zip(utts["train"], present["largest"]["train"]) if p]
    missing = set(labels.EMOTIONS) - {u.emotion for u in face_train}
    if missing:
        raise SystemExit(f"no train clip with a face for {sorted(missing)}")
    w_face = lambda scheme: emotion_class_weights(scheme, face_train)  # noqa: E731

    # 1. speaker rule (calibrator A per rule)
    rule_heads, rule_rec = {}, {}
    for r in vf.RULES:
        X = {s: T[s][r]["XA"] for s in utts}
        h, rec = fit_vision_grid(X, y_all, present[r], w_face, cfg, seed, v_schemes, variant="A", rule=r)
        rec["fallback_rate"] = {s: round(float(T[s][r]["fallback"][T[s][r]["present"]].mean()), 4) for s in utts}
        rule_heads[r], rule_rec[r] = h, rec
        print(f"rule {r}: dev weighted f1 {rec['dev']['weighted_f1']:.4f} (face rows {rec['dev_face_rows']['weighted_f1']:.4f}), "
              f"C={rec['C']}, cw={rec['class_weight_scheme']}", flush=True)
    rule = max(vf.RULES, key=lambda r: rule_rec[r]["dev"]["weighted_f1"])  # ties: the earlier rule in RULES
    rep["rules"] = rule_rec
    rep["rule"] = rule

    # 2. calibrator A against B with that rule
    cal = {"A": rule_heads[rule]}
    cal_rec = {"A": rule_rec[rule]}
    XB = {s: features_of(T[s][rule], "B") for s in utts}
    t0 = time.perf_counter()
    hB, recB = fit_vision_grid(XB, y_all, present[rule], w_face, cfg, seed, v_schemes, variant="B", rule=rule)
    recB["wall_s"] = round(time.perf_counter() - t0, 1)
    cal["B"], cal_rec["B"] = hB, recB
    variant = "B" if recB["dev"]["weighted_f1"] > cal_rec["A"]["dev"]["weighted_f1"] else "A"
    vcal = cal[variant]
    vcal.save(out / "vision_cal.npz")
    print(f"calibrator A {cal_rec['A']['dev']['weighted_f1']:.4f} B {recB['dev']['weighted_f1']:.4f} on dev: deploy {variant}", flush=True)
    rep["calibrator"] = {"A": cal_rec["A"], "B": recB, "variant": variant, "file": "vision_cal.npz"}
    rep["variant"] = variant
    rep["file"] = "vision_cal.npz"

    # 3. out-of-fold expert posteriors on train_heads' rows and folds
    loaded = {s: read_audio_cache(cache, s, utts[s]) for s in utts}
    if any(v is None for v in loaded.values()):
        raise SystemExit(f"needs {cache / 'audio_train.npz'} and {cache / 'audio_dev.npz'}")
    aud = {s: loaded[s][0] for s in utts}
    keep = {s: loaded[s][1] for s in utts}
    y = {s: y_all[s][keep[s]] for s in utts}
    ys = {s: sentiment_ids(utts[s])[keep[s]] for s in utts}
    train_kept = [utts["train"][i] for i in keep["train"]]
    w_emo = lambda scheme: emotion_class_weights(scheme, train_kept)  # noqa: E731
    groups = groups_of(train_kept)
    text_sel = manifest["selected"]["text_ctx"]
    k = int(text_sel["k"])
    emb = {s: load_text_cache(cache, s, utts[s]) for s in utts}
    Xt = {s: text_matrix(emb[s], utts[s], k)[keep[s]] for s in utts}
    a_variant = af["audio_cal"]["variant"]
    choice_a = af[f"audio_cal_{a_variant}"]
    Xa = {s: audio_matrix(aud[s], a_variant == "B") for s in utts}
    Xv = {s: features_of(T[s][rule], variant)[keep[s]] for s in utts}
    pres = {s: T[s][rule]["present"][keep[s]] for s in utts}
    presence = {s: T[s][rule]["presence"][keep[s]] for s in utts}
    choice_v = {"C": rep["calibrator"][variant]["C"], "class_weight_scheme": rep["calibrator"][variant]["class_weight_scheme"]}
    fit_text = text_fitter(text_sel, cfg, w_emo(text_sel["class_weight_scheme"]), Xt["dev"], y["dev"], seed)
    fit_audio = audio_fitter(choice_a, w_emo(choice_a["class_weight_scheme"]), cfg, seed)
    fit_vis = vision_fitter(choice_v, w_face(choice_v["class_weight_scheme"]), cfg, seed, variant=variant, rule=rule)
    fold_heads: dict[str, list] = {"text_ctx": [], "audio_cal": [], "vision_cal": []}
    t0 = time.perf_counter()
    oof_t, fold_id, _ = cross_fit(capture(fit_text, fold_heads["text_ctx"]), Xt["train"], y["train"], groups, n_folds, seed)
    oof_a, fold_a, _ = cross_fit(capture(fit_audio, fold_heads["audio_cal"]), Xa["train"], y["train"], groups, n_folds, seed)
    oof_v, fold_v, _ = cross_fit(capture(fit_vis, fold_heads["vision_cal"]), Xv["train"], y["train"], groups, n_folds, seed)
    if not (np.array_equal(fold_id, fold_a) and np.array_equal(fold_id, fold_v)):
        raise AssertionError("cross-fit folds differ between experts")
    oof_v[~pres["train"]] = UNIFORM_LOG
    fold_sizes = np.bincount(fold_id, minlength=n_folds).tolist()
    ref_sizes = (af.get("cross_fit") or {}).get("fold_sizes")
    print(f"cross-fit: {time.perf_counter() - t0:.1f} s, fold sizes {fold_sizes} (train_heads {ref_sizes})", flush=True)

    deployed = {"text": heads.TextContextHead.load(heads_dir / "text_ctx.npz"), "audio": heads.AudioCalibrator.load(heads_dir / "audio_cal.npz"),
                "fusion": heads.Fusion.load(heads_dir / "fusion.npz")}
    lp_t_dev = deployed["text"].predict_log_proba_batch(Xt["dev"])
    lp_a_dev = deployed["audio"].predict_log_proba_batch(Xa["dev"])
    lp_v_dev = np.log(np.clip(face_proba(vcal, Xv["dev"], pres["dev"]), heads.EPS, 1.0))
    # the same rows and folds as train_heads: its two-way fusion, refit here, must come out identical
    F2 = {"train": fusion_rows(oof_t, oof_a, aud["train"]), "dev": fusion_rows(lp_t_dev, lp_a_dev, aud["dev"])}
    fh2, _, _ = lr_grid(F2["train"], y["train"], F2["dev"], y["dev"], labels.EMOTIONS, cfg["lr"]["C"], cfg["fusion"]["class_weight"],
                        w_emo, cfg["lr"], seed)
    fh2 = heads.Fusion(fh2.params)
    calibrate(fh2, F2["dev"], y["dev"])
    reproduces = same_weights(fh2, heads_dir / "fusion.npz")
    print(f"two-way fusion refit on these rows equals heads/fusion.npz: {reproduces}", flush=True)
    two_way_dev = summary(y["dev"], np.exp(deployed["fusion"].predict_log_proba_batch(F2["dev"])))
    rep["cross_fit"] = {
        "folds": n_folds, "seed": seed, "fold_sizes": fold_sizes, "train_heads_fold_sizes": ref_sizes,
        "same_fold_sizes_as_train_heads": ref_sizes == fold_sizes, "reproduces_deployed_fusion": reproduces,
        "rows": {s: len(keep[s]) for s in utts},
        "oof_train_scores": {"text": summary(y["train"], np.exp(oof_t)), "audio": summary(y["train"], np.exp(oof_a)),
                             "vision": summary(y["train"], np.exp(oof_v)),
                             "vision_face_rows": summary(y["train"][pres["train"]], np.exp(oof_v[pres["train"]]))},
        "vision_choice": choice_v}

    # fusion3 and sentiment3
    F3 = {"train": fusion3_rows(oof_t, oof_a, oof_v, aud["train"], pres["train"], presence["train"]),
          "dev": fusion3_rows(lp_t_dev, lp_a_dev, lp_v_dev, aud["dev"], pres["dev"], presence["dev"])}
    f3, choice3, scores3 = lr_grid(F3["train"], y["train"], F3["dev"], y["dev"], labels.EMOTIONS, cfg["lr"]["C"],
                                   cfg["fusion"]["class_weight"], w_emo, cfg["lr"], seed, variant="tav")
    f3 = heads.Fusion3(f3.params)
    cal3 = calibrate(f3, F3["dev"], y["dev"])
    f3.save(out / "fusion3.npz")
    p3_dev = np.exp(f3.predict_log_proba_batch(F3["dev"]))
    dev3 = summary(y["dev"], p3_dev)
    two_way_manifest = (af.get("fusion") or {}).get("dev", {}).get("weighted_f1")
    face_dev = pres["dev"]
    rep["fusion3"] = {**choice3, "file": "fusion3.npz", "dev": dev3, "dev_scores": scores3, "calibration_dev": cal3,
                      "two_way_dev_manifest": two_way_manifest, "two_way_dev_recomputed": two_way_dev,
                      "dev_face_rows": {"n": int(face_dev.sum()), "fusion3": summary(y["dev"][face_dev], p3_dev[face_dev]),
                                        "two_way": summary(y["dev"][face_dev], np.exp(deployed["fusion"].predict_log_proba_batch(F2["dev"][face_dev])))}}
    print(f"fusion3: dev weighted f1 {dev3['weighted_f1']:.4f} against two-way {two_way_dev['weighted_f1']:.4f} "
          f"(manifest {two_way_manifest}), C={choice3['C']}, cw={choice3['class_weight_scheme']}", flush=True)
    counts = np.bincount(ys["train"], minlength=N_SENT).astype(np.float64)
    s3, choice_s, scores_s = lr_grid(F3["train"], ys["train"], F3["dev"], ys["dev"], labels.SENTIMENTS, cfg["lr"]["C"],
                                     cfg["sentiment"]["class_weight"],
                                     lambda sch: counts.sum() / (N_SENT * counts) if sch == "balanced" else np.ones(N_SENT),
                                     cfg["lr"], seed, variant="tav")
    s3 = heads.Sentiment3(s3.params)
    cal_s = calibrate(s3, F3["dev"], ys["dev"])
    s3.save(out / "sentiment3.npz")
    rep["sentiment3"] = {**choice_s, "file": "sentiment3.npz", "dev": summary(ys["dev"], np.exp(s3.predict_log_proba_batch(F3["dev"]))),
                         "dev_scores": scores_s, "calibration_dev": cal_s,
                         "two_way_dev_manifest": (manifest.get("sentiment") or {}).get("dev", {}).get("weighted_f1")}

    # 4. thresholds, 5. face jsd threshold
    rep["thresholds"] = choose_thresholds(p3_dev, y["dev"], cfg["thresholds"])
    p_raw_dev = T["dev"][rule]["p_raw"][keep["dev"]]
    jsd_dev = np.array([jsd(np.exp(lp_t_dev[i]), p_raw_dev[i]) for i in np.flatnonzero(face_dev)])
    rep["jsd_face_flag"] = round(float(np.quantile(jsd_dev, q_face)), 4)
    rep["jsd_face"] = {"rule": f"dev quantile {q_face} of jsd(text posterior on gold text, raw face posterior of the chosen track)",
                       "n_dev_face_rows": int(face_dev.sum()), "quantile": q_face,
                       "dev_jsd_percentiles": {str(q): round(float(np.quantile(jsd_dev, q)), 4) for q in (0.1, 0.5, 0.9)}}

    # 6. fold heads
    folds_dir = cache / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    fold_files = {}
    fit_f3 = audio_fitter(choice3, w_emo(choice3["class_weight_scheme"]), cfg, seed)
    oof_f3 = np.zeros((len(keep["train"]), N_EMO))
    for f in range(n_folds):
        tr, te = np.flatnonzero(fold_id != f), np.flatnonzero(fold_id == f)
        ht = fold_heads["text_ctx"][f]
        ht = heads.TextContextHead({**ht.params, "k": np.array(k), "variant": np.array("fold"), "fold": np.array(f)})
        ha = heads.AudioCalibrator({**fold_heads["audio_cal"][f].params, "variant": np.array(a_variant), "fold": np.array(f)})
        hv = heads.VisionCalibrator({**fold_heads["vision_cal"][f].params, "fold": np.array(f)})
        h3 = heads.Fusion3({**fit_f3(F3["train"][tr], y["train"][tr]).params, "variant": np.array("tav"), "fold": np.array(f)})
        calibrate(h3, F3["dev"], y["dev"])
        oof_f3[te] = np.exp(h3.predict_log_proba_batch(F3["train"][te]))
        for name, h in (("text_ctx", ht), ("audio_cal", ha), ("vision_cal", hv), ("fusion3", h3)):
            path = folds_dir / f"fold{f}_{name}.npz"
            h.save(path)
            fold_files[path.name] = file_record(path)
    fold_of_dia = {int(g): int(f) for g, f in zip(groups, fold_id)}
    fold_of_train = {key(u): fold_of_dia[u.dialogue_id] for u in utts["train"] if u.dialogue_id in fold_of_dia}
    (folds_dir / "fold_of_train.json").write_text(json.dumps(fold_of_train), encoding="utf-8", newline="\n")
    rep["folds"] = {"dir": str(folds_dir), "files": fold_files, "fold_of_train": "fold_of_train.json",
                    "n_keys": len(fold_of_train), "fusion3_oof_train": summary(y["train"], oof_f3),
                    "note": ("expert heads as fitted in the cross-fit (temperature 1, as the out-of-fold rows were made); "
                             "fusion3 per fold refit with the deployed choice on the other folds' out-of-fold rows, "
                             "temperature on dev with the deployed experts' dev rows")}

    rep["files"] = {p: file_record(out / p) for p in ("vision_cal.npz", "fusion3.npz", "sentiment3.npz")}
    rep["inputs"] = {f"video_{s}.npz": file_record(Path(video[s]["path"])) for s in utts}
    rep["envelope"] = {"hop_s": vf.HOP_S, "floor": vf.LOG_RMS_FLOOR, "files": [f"envelope_{s}.npz" for s in utts]}
    rep["features"] = {"A": vf.FEATURES_A, "B": "A plus the chosen track's mean fer embedding (512)", "fusion3": vf.FEATURES_FUSION3}
    rep["config"] = {"class_weight": v_schemes, "C": cfg["lr"]["C"], "jsd_face_quantile": q_face, "seed": seed}

    # manifest: the vision section, every other key as it was
    manifest["vision"] = rep
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")

    # 7. per-clip posteriors
    posts = {}
    tr_n = len(utts["train"])
    P = {"keys": [key(u) for u in utts["train"]], "rule": rule, "variant": variant, "keep": keep["train"],
         **{k2: T["train"][rule][k2] for k2 in ("present", "track", "presence", "n_tracks", "asd_corr", "n_frames", "fallback", "in_cache")},
         "p_face_raw": T["train"][rule]["p_raw"], "feats_A": T["train"][rule]["XA"],
         "p_text": np.exp(oof_t), "p_audio": np.exp(oof_a), "fused3": oof_f3}
    p_face_tr = np.full((tr_n, N_EMO), np.nan)
    p_face_tr[keep["train"]] = np.exp(oof_v)
    P["p_face"] = p_face_tr
    fold_rows = np.full(tr_n, -1, np.int64)
    fold_rows[keep["train"]] = fold_id
    P["fold"] = fold_rows
    posts["train"] = write_post(cache / "vision_post_train.npz", P, "train", True)
    for s in [s for s in args.post_splits if s in ("dev", "test")]:
        if s == "test" and not (cache / "video_test.npz").exists():
            posts["test"] = "skipped (no cache/video_test.npz)"
            continue
        su = utts[s] if s in utts else load_split(s, args.meld)
        vid = video[s] if s in video else load_video(cache, s)
        en = env[s] if s in env else envelopes(su, args.audio, cache, s)
        Ps = deployed_posteriors(s, su, cache, heads_dir, manifest, vid, en, live=(s == "test"), vision_dir=out)
        posts[s] = write_post(cache / f"vision_post_{s}.npz", Ps, s, False, {"gate": Ps.get("gate")})
    manifest["vision"]["posteriors"] = posts
    rep["wall_time_s"] = round(time.perf_counter() - t_all, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "cache", help="text, audio and video caches; folds/ and vision_post_* go here")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads", help="deployed heads and manifest.json of tools/train_heads.py")
    ap.add_argument("--out", type=Path, default=None, help="where the new heads and the manifest go (default: --heads)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "heads.yaml")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--audio", type=Path, default=ROOT / "audio", help="clip wavs, audio/{split}/diaX_uttY.wav")
    ap.add_argument("--post-splits", default="train,dev,test", help="splits to write cache/vision_post_{split}.npz for")
    args = ap.parse_args(argv)
    args.out = args.out or args.heads
    args.post_splits = [s for s in args.post_splits.split(",") if s]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))
    import threadpoolctl

    with threadpoolctl.threadpool_limits(int(cfg.get("threads", 2))):
        rep = train(args, cfg, seed)
    m = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    m["vision"]["command"] = "python tools/train_vision.py " + " ".join(sys.argv[1:] if argv is None else argv)
    (args.out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8", newline="\n")
    print(f"vision: rule {rep['rule']}, calibrator {rep['variant']}, fusion3 dev weighted f1 {rep['fusion3']['dev']['weighted_f1']:.4f}, "
          f"jsd_face_flag {rep['jsd_face_flag']}, {rep['wall_time_s']} s -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
