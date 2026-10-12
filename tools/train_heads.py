"""train the heads (design 1.6) from the feature caches and export them as .npz.

stage 1, text: TextContextHead over the k x class-weight grid of configs/heads.yaml,
mlp with early stopping on dev, chosen by dev weighted f1 (macro f1 recorded),
temperature scaled on dev, selector thresholds chosen on dev. writes
heads/text_ctx.npz (best), heads/text_ctx_k0.npz (best at k=0) and, with
--text-only, heads/sentiment.npz as a logistic regression over the same text
features (variant "text_only") so the service has a sentiment output before
MELD audio exists.

stage 2, audio and fusion: needs cache/audio_{split}.npz from tools/cache_audio_features.py.
audio calibrator A/B, 5-fold cross-fitting by dialogue of both expert heads,
fusion and sentiment logistic regressions over the out-of-fold posteriors,
product-of-experts weight, jsd threshold. skipped (and said so in the manifest)
when the caches are absent or --text-only is given. csv rows with no cached clip
are dropped from this stage; manifest.json audio_rows_dropped counts them per split.

usage: python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml [--text-only]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt import heads, labels, metrics  # noqa: E402
from meld_rt.data import Utt, context_rows, dialogues, emotion_ids, key, load_split, sentiment_ids  # noqa: E402
from meld_rt.data import class_weights as emotion_class_weights  # noqa: E402

N_EMO = len(labels.EMOTIONS)
N_SENT = len(labels.SENTIMENTS)
MIN_BUCKET = 20  # dev predictions needed before a threshold rule may fire


# ---------------------------------------------------------------- caches and features

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def load_text_cache(cache: Path, split: str, utts: list[Utt]) -> np.ndarray:
    """(n, 768) float32 in load_split order; the keys file must match the csv order."""
    emb = np.load(cache / f"text_{split}.npy")
    keys = json.loads((cache / f"text_{split}_keys.json").read_text())
    want = [key(u) for u in utts]
    if keys != want:
        raise ValueError(f"cache/text_{split}_keys.json does not match the csv order")
    if emb.shape != (len(utts), heads.TEXT_DIM):
        raise ValueError(f"text_{split}.npy has shape {emb.shape}")
    return emb.astype(np.float32)


def text_matrix(emb: np.ndarray, utts: list[Utt], k: int, same_speaker_only: bool = False) -> np.ndarray:
    """(n, 768*(k+1)+4) context features for a split. turn_index is the position in
    the dialogue; in the runtime regime it is the number of the same speaker's
    earlier utterances (what one user at the lamp has said so far)."""
    row = {key(u): i for i, u in enumerate(utts)}
    turn = {}
    for dia in dialogues(utts):
        seen: dict[str, int] = {}
        for i, u in enumerate(dia):
            turn[key(u)] = seen.get(u.speaker, 0) if same_speaker_only else i
            seen[u.speaker] = seen.get(u.speaker, 0) + 1
    X = np.zeros((len(utts), heads.TextContextHead.input_dim_for(k)), dtype=np.float32)
    for i, (u, prev) in enumerate(context_rows(utts, k, same_speaker_only)):
        X[i] = heads.TextContextHead.features(emb[row[key(u)]], [emb[row[key(p)]] for p in prev],
                                              [p.speaker == u.speaker for p in prev], turn[key(u)], k)
    return X


def standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, dtype=np.float64)
    scale = X.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def groups_of(utts: list[Utt]) -> np.ndarray:
    return np.array([u.dialogue_id for u in utts], dtype=np.int64)


# ---------------------------------------------------------------- fitting

def fit_mlp(Xtr: np.ndarray, ytr: np.ndarray, sw: np.ndarray, Xdev: np.ndarray, ydev: np.ndarray,
            cfg: dict, seed: int, n_classes: int, criterion: str = "weighted_f1") -> tuple[dict, dict]:
    """one hidden layer mlp, one partial_fit call per epoch, early stopping on the dev
    criterion, best epoch restored. returns (model arrays, record)."""
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(int(cfg["hidden"]),), activation=cfg["activation"], alpha=float(cfg["alpha"]),
                        batch_size=int(cfg["batch_size"]), learning_rate_init=float(cfg["learning_rate_init"]),
                        random_state=seed, max_iter=1)
    best, best_epoch, bad, history = None, -1, 0, []
    t0 = time.perf_counter()
    for ep in range(int(cfg["max_epochs"])):
        mlp.partial_fit(Xtr, ytr, sample_weight=sw, classes=np.arange(n_classes))
        s = metrics.summary(metrics.confusion(ydev, mlp.predict_proba(Xdev).argmax(axis=1), n_classes))
        history.append({"epoch": ep, "loss": float(mlp.loss_), **{k: round(v, 4) for k, v in s.items()}})
        if best is None or s[criterion] > best[criterion]:
            best, best_epoch, bad = s, ep, 0
            arrays = heads.mlp_params([c.copy() for c in mlp.coefs_], [b.copy() for b in mlp.intercepts_], cfg["activation"])
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    rec = {**best, "best_epoch": best_epoch, "epochs": len(history), "wall_s": round(time.perf_counter() - t0, 1),
           "history": history}
    return arrays, rec


def fit_lr(X: np.ndarray, y: np.ndarray, C: float, sw: np.ndarray | None, seed: int, max_iter: int) -> dict:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=float(C), max_iter=int(max_iter), random_state=seed)
    lr.fit(X, y, sample_weight=sw)
    return heads.lr_params(lr.coef_, lr.intercept_)


def head_from(arrays: dict, mean: np.ndarray, scale: np.ndarray, classes: list[str], cls=heads.NpzHead, **meta) -> heads.NpzHead:
    return cls(heads.head_params(arrays, mean, scale, classes, **meta))


def lr_grid(Xtr: np.ndarray, ytr: np.ndarray, Xdev: np.ndarray, ydev: np.ndarray, classes: list[str],
            Cs: list[float], schemes: list[str], weights: Callable[[str], np.ndarray], cfg_lr: dict, seed: int,
            criterion: str = "weighted_f1", **meta) -> tuple[heads.NpzHead, dict, dict]:
    """logistic regression over C x class-weight scheme chosen on dev. returns
    (best head without temperature, its choice, dev scores per grid point)."""
    mean, scale = standardizer(Xtr)
    Xs = (Xtr - mean) / scale
    scores, best = {}, None
    for scheme in schemes:
        w = weights(scheme)
        sw = None if scheme == "none" else w[ytr]
        for C in Cs:
            t0 = time.perf_counter()
            arrays = fit_lr(Xs, ytr, C, sw, seed, cfg_lr["max_iter"])
            h = head_from(arrays, mean, scale, classes, C=float(C), class_weight_scheme=scheme, **meta)
            s = metrics.summary(metrics.confusion(ydev, h.predict_log_proba_batch(Xdev).argmax(axis=1), len(classes)))
            name = f"C={C},cw={scheme}"
            scores[name] = {**{k: round(v, 4) for k, v in s.items()}, "wall_s": round(time.perf_counter() - t0, 1)}
            if best is None or s[criterion] > best[2][criterion]:
                best = (h, {"C": float(C), "class_weight_scheme": scheme}, s)
    return best[0], best[1], scores


def calibrate(h: heads.NpzHead, Xdev: np.ndarray, ydev: np.ndarray) -> dict:
    """fit the temperature on dev in place; report nll and ece before and after."""
    z = h.logits(Xdev) * h.temperature
    before = {"nll": float(-heads.log_softmax(z)[np.arange(len(ydev)), ydev].mean()), "ece": metrics.ece(heads.softmax(z), ydev)}
    T = heads.fit_temperature(z, ydev)
    h.temperature = T
    lp = heads.log_softmax(z / T)
    after = {"nll": float(-lp[np.arange(len(ydev)), ydev].mean()), "ece": metrics.ece(np.exp(lp), ydev)}
    return {"temperature": round(T, 4), "nll_before": round(before["nll"], 4), "nll_after": round(after["nll"], 4),
            "ece_before": round(before["ece"], 4), "ece_after": round(after["ece"], 4)}


def choose_thresholds(proba: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    """accuracy as a monotone function of confidence (isotonic regression of
    correctness on the argmax confidence, dev). low_conf: the smallest grid value
    at which that accuracy reaches low_conf.target_accuracy (below it a label claim
    is more often wrong than right); high_conf: the same for high_conf.target_accuracy.
    fallbacks when a target is never reached or low >= high."""
    from sklearn.isotonic import IsotonicRegression

    conf, correct = proba.max(axis=1), (proba.argmax(axis=1) == y).astype(np.float64)
    step = float(cfg["grid_step"])
    grid = np.round(np.arange(step, 1.0, step), 4)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(conf, correct)
    acc = iso.predict(grid)

    def first(target: float) -> float | None:
        hit = np.flatnonzero(acc >= target)
        return float(grid[hit[0]]) if hit.size else None

    low, high = first(float(cfg["low_conf"]["target_accuracy"])), first(float(cfg["high_conf"]["target_accuracy"]))
    out = {"low_conf": low, "high_conf": high, "source": "dev",
           "dev_accuracy_at": {str(t): round(float(a), 4) for t, a in zip(grid[9::10], acc[9::10])}}
    if low is None or high is None or low >= high:
        out = {"low_conf": float(cfg["low_conf"]["fallback"]), "high_conf": float(cfg["high_conf"]["fallback"]),
               "source": "fallback", "dev_rule_values": {"low_conf": low, "high_conf": high}, "dev_accuracy_at": out["dev_accuracy_at"]}
    return out


def cross_fit(fit_fn: Callable[[np.ndarray, np.ndarray], heads.NpzHead], X: np.ndarray, y: np.ndarray,
              groups: np.ndarray, n_folds: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """out-of-fold log-posteriors. groups are assigned to folds whole (shuffled with
    seed), each fold's head is fit on the other folds only. returns
    (oof_logp (n, C), fold_id (n,), train rows per fold) so a test can assert the ids."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_group = {uniq[gi]: rank % n_folds for rank, gi in enumerate(perm)}
    fold_id = np.array([fold_of_group[g] for g in groups], dtype=np.int64)
    oof, train_rows = None, []
    for f in range(n_folds):
        tr, te = np.flatnonzero(fold_id != f), np.flatnonzero(fold_id == f)
        if np.intersect1d(tr, te).size:
            raise AssertionError("fold overlap")
        h = fit_fn(X[tr], y[tr])
        lp = h.predict_log_proba_batch(X[te])
        if oof is None:
            oof = np.zeros((len(y), lp.shape[1]), dtype=np.float64)
        oof[te] = lp
        train_rows.append(tr)
    return oof, fold_id, train_rows


# ---------------------------------------------------------------- stage 1: text

def train_text_stage(cache: Path, out: Path, cfg: dict, utts: dict[str, list[Utt]], emb: dict[str, np.ndarray],
                     seed: int) -> dict:
    tcfg = cfg["text_ctx"]
    crit = tcfg.get("criterion", "weighted_f1")
    y = {s: emotion_ids(utts[s]) for s in ("train", "dev")}
    results, fitted = {}, {}
    t_stage = time.perf_counter()
    for k in tcfg["grid"]["k"]:
        Xtr, Xdev = text_matrix(emb["train"], utts["train"], k), text_matrix(emb["dev"], utts["dev"], k)
        mean, scale = standardizer(Xtr)
        Xs, Xd = (Xtr - mean) / scale, (Xdev - mean) / scale
        for scheme in tcfg["grid"]["class_weight"]:
            w = emotion_class_weights(scheme, utts["train"])
            for lr in tcfg["grid"].get("learning_rate_init", [tcfg["mlp"]["learning_rate_init"]]):
                mlp_cfg = {**tcfg["mlp"], "learning_rate_init": lr}
                arrays, rec = fit_mlp(Xs, y["train"], w[y["train"]], Xd, y["dev"], mlp_cfg, seed, N_EMO, crit)
                name = f"k={k},cw={scheme},lr={lr}"
                results[name] = {"k": k, "class_weight_scheme": scheme, "learning_rate_init": float(lr), **rec}
                fitted[name] = (arrays, mean, scale, Xdev)
                print(f"{name}: dev weighted f1 {rec['weighted_f1']:.4f} macro f1 {rec['macro_f1']:.4f} "
                      f"acc {rec['accuracy']:.4f} best epoch {rec['best_epoch']} ({rec['epochs']} run, {rec['wall_s']} s)", flush=True)

    def export(name: str, path: Path) -> dict:
        arrays, mean, scale, Xdev = fitted[name]
        r = results[name]
        h = head_from(arrays, mean, scale, labels.EMOTIONS, heads.TextContextHead, k=r["k"], variant="full_dialogue",
                      class_weight_scheme=r["class_weight_scheme"], learning_rate_init=r["learning_rate_init"])
        cal = calibrate(h, Xdev, y["dev"])
        thr = choose_thresholds(np.exp(h.predict_log_proba_batch(Xdev)), y["dev"], cfg["thresholds"])
        h.save(path)
        return {"grid_point": name, "k": r["k"], "class_weight_scheme": r["class_weight_scheme"],
                "learning_rate_init": r["learning_rate_init"], "file": path.name,
                "dev": {m: r[m] for m in ("weighted_f1", "macro_f1", "accuracy")}, "calibration_dev": cal, "thresholds": thr,
                "n_params": int(sum(int(np.asarray(v).size) for kk, v in arrays.items() if kk in ("W1", "b1", "W2", "b2")))}

    best = max(results, key=lambda n: results[n][crit])
    best_k0 = max((n for n in results if results[n]["k"] == 0), key=lambda n: results[n][crit])
    sel = {"text_ctx": export(best, out / "text_ctx.npz"), "text_ctx_k0": export(best_k0, out / "text_ctx_k0.npz")}
    fields = ("k", "class_weight_scheme", "learning_rate_init", "weighted_f1", "macro_f1", "accuracy", "best_epoch", "epochs", "wall_s")
    grid = {n: {m: r[m] for m in fields} for n, r in results.items()}
    return {"criterion": crit, "dev_scores": grid, "histories": {n: r["history"] for n, r in results.items()},
            "selected": sel, "wall_time_s": round(time.perf_counter() - t_stage, 1)}


def train_sentiment_text_only(out: Path, cfg: dict, utts: dict[str, list[Utt]], emb: dict[str, np.ndarray],
                              k: int, seed: int) -> dict:
    """logistic regression over the chosen k's text features, MELD sentiment labels."""
    t0 = time.perf_counter()
    Xtr, Xdev = text_matrix(emb["train"], utts["train"], k), text_matrix(emb["dev"], utts["dev"], k)
    ytr, ydev = sentiment_ids(utts["train"]), sentiment_ids(utts["dev"])
    counts = np.bincount(ytr, minlength=N_SENT).astype(np.float64)

    def weights(scheme: str) -> np.ndarray:
        return counts.sum() / (N_SENT * counts) if scheme == "balanced" else np.ones(N_SENT)

    h, choice, scores = lr_grid(Xtr, ytr, Xdev, ydev, labels.SENTIMENTS, cfg["lr"]["C"], cfg["sentiment"]["class_weight"],
                                weights, cfg["lr"], seed, variant="text_only", k=k)
    h = heads.SentimentHead(h.params)
    cal = calibrate(h, Xdev, ydev)
    h.save(out / "sentiment.npz")
    dev = metrics.summary(metrics.confusion(ydev, h.predict_log_proba_batch(Xdev).argmax(axis=1), N_SENT))
    for name, s in scores.items():
        print(f"sentiment {name}: dev weighted f1 {s['weighted_f1']:.4f} macro f1 {s['macro_f1']:.4f}", flush=True)
    return {"variant": "text_only", "input": f"TextContextHead features, k={k} (768*(k+1)+4)", "file": "sentiment.npz",
            **choice, "dev": {m: round(v, 4) for m, v in dev.items()}, "dev_scores": scores, "calibration_dev": cal,
            "wall_time_s": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------- stage 2: audio and fusion

def read_audio_cache(cache: Path, split: str, utts: list[Utt]) -> tuple[dict, np.ndarray, list[str]] | None:
    """(cache rows in csv order for the csv rows that have a cached clip, their
    indices into utts, keys of the csv rows without one). None when the file is absent."""
    path = cache / f"audio_{split}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    pos = {str(k): i for i, k in enumerate(d["keys"])}
    want = [key(u) for u in utts]
    keep = np.array([i for i, k in enumerate(want) if k in pos], dtype=np.int64)
    dropped = [k for k in want if k not in pos]
    order = np.array([pos[want[i]] for i in keep], dtype=np.int64)
    if not np.array_equal(order, np.arange(len(d["keys"]))):
        d = {k: v[order] for k, v in d.items()}
    return d, keep, dropped


def load_audio_cache(cache: Path, split: str, utts: list[Utt]) -> dict | None:
    """every csv row must have a cached clip (evaluate.py reads test this way)."""
    r = read_audio_cache(cache, split, utts)
    if r is None:
        return None
    d, _, dropped = r
    if dropped:
        raise ValueError(f"audio_{split}.npz lacks {len(dropped)} clips (first {dropped[:3]})")
    return d


def audio_matrix(d: dict, with_enc: bool) -> np.ndarray:
    enc = d.get("enc_pooled") if with_enc else None
    if with_enc and (enc is None or enc.shape[1] != heads.ENC_POOLED_DIM or not np.isfinite(enc).all()):
        raise ValueError("encoder states missing or malformed in the audio cache")
    if with_enc and not enc.any():
        raise ValueError("encoder states are all zero (cache written without --encoder-states)")
    return np.stack([heads.AudioCalibrator.features(d["emo_probs"][i], d["event_probs"][i], d["prosody"][i],
                                                    None if enc is None else enc[i]) for i in range(len(d["keys"]))])


def fusion_rows(logp_text: np.ndarray, logp_audio: np.ndarray, d: dict) -> np.ndarray:
    return np.stack([heads.build_fusion_features(logp_text[i], logp_audio[i], d["asr_conf"][i],
                                                 d["event_probs"][i][heads.LAUGHTER_IDX], d["emo_probs"][i][heads.UNKNOWN_IDX])
                     for i in range(len(logp_text))])


def train_audio_stage(cache: Path, out: Path, cfg: dict, utts: dict[str, list[Utt]], emb: dict[str, np.ndarray],
                      text_sel: dict, seed: int) -> dict:
    """audio calibrator A/B, cross-fitted expert posteriors, fusion, poe weight,
    fused sentiment, jsd threshold. runs only when cache/audio_{train,dev}.npz exist.
    csv rows without a cached clip (missing or unreadable in MELD.Raw) are dropped
    from this stage. text context features are still built over the full dialogues."""
    t_stage = time.perf_counter()
    loaded = {s: read_audio_cache(cache, s, utts[s]) for s in ("train", "dev")}
    if loaded["train"] is None or loaded["dev"] is None:
        return {"status": "pending (needs MELD audio)", "needs": [str(cache / f"audio_{s}.npz") for s in ("train", "dev")]}
    aud = {s: loaded[s][0] for s in loaded}
    keep = {s: loaded[s][1] for s in loaded}
    rows = {s: {"csv": len(utts[s]), "cached": len(keep[s]), "dropped": len(loaded[s][2]), "dropped_keys": loaded[s][2][:50]}
            for s in loaded}
    rep: dict = {"rows": rows, "rows_dropped": {s: rows[s]["dropped"] for s in rows}}
    for s, r in rows.items():
        if r["dropped"]:
            print(f"audio stage: {s} drops {r['dropped']} of {r['csv']} csv rows without a cached clip "
                  f"(first {r['dropped_keys'][:3]})", flush=True)
    if not all(len(keep[s]) for s in keep):
        return {"status": "no csv row has a cached clip", **rep}
    y = {s: emotion_ids(utts[s])[keep[s]] for s in ("train", "dev")}
    ys = {s: sentiment_ids(utts[s])[keep[s]] for s in ("train", "dev")}
    train_kept = [utts["train"][i] for i in keep["train"]]
    w_emo = lambda scheme: emotion_class_weights(scheme, train_kept)  # noqa: E731

    # audio calibrator A (24) and B (1048), chosen on dev
    cal_heads, cal_X = {}, {}
    for variant in cfg["audio_cal"]["variants"]:
        with_enc = variant == "B"
        try:
            Xtr, Xdev = audio_matrix(aud["train"], with_enc), audio_matrix(aud["dev"], with_enc)
        except ValueError as e:
            rep[f"audio_cal_{variant}"] = {"status": f"skipped: {e}"}
            continue
        h, choice, scores = lr_grid(Xtr, y["train"], Xdev, y["dev"], labels.EMOTIONS, cfg["lr"]["C"], cfg["audio_cal"]["class_weight"],
                                    w_emo, cfg["lr"], seed, variant=variant)
        h = heads.AudioCalibrator(h.params)
        cal = calibrate(h, Xdev, y["dev"])
        h.save(out / f"audio_cal_{variant}.npz")
        cal_heads[variant], cal_X[variant] = h, (Xtr, Xdev)
        dev = metrics.summary(metrics.confusion(y["dev"], h.predict_log_proba_batch(Xdev).argmax(axis=1), N_EMO))
        rep[f"audio_cal_{variant}"] = {**choice, "dev": {m: round(v, 4) for m, v in dev.items()}, "dev_scores": scores,
                                       "calibration_dev": cal, "input_dim": h.input_dim}
    if not cal_heads:
        return {"status": "audio calibrator could not be trained", **rep}
    best_variant = max(cal_heads, key=lambda v: rep[f"audio_cal_{v}"]["dev"]["weighted_f1"])
    cal_heads[best_variant].save(out / "audio_cal.npz")
    rep["audio_cal"] = {"variant": best_variant, "file": "audio_cal.npz"}

    # cross-fitted expert posteriors on train (never in-sample), full-train heads on dev
    k = int(text_sel["k"])
    tcfg = cfg["text_ctx"]
    Xt = {s: text_matrix(emb[s], utts[s], k)[keep[s]] for s in ("train", "dev")}
    Xa = {"train": cal_X[best_variant][0], "dev": cal_X[best_variant][1]}
    groups = groups_of(train_kept)
    w = w_emo(text_sel["class_weight_scheme"])
    mlp_cfg = {**tcfg["mlp"], "learning_rate_init": text_sel.get("learning_rate_init", tcfg["mlp"]["learning_rate_init"])}

    def fit_text(X: np.ndarray, yy: np.ndarray) -> heads.NpzHead:
        mean, scale = standardizer(X)
        arrays, _ = fit_mlp((X - mean) / scale, yy, w[yy], (Xt["dev"] - mean) / scale, y["dev"], mlp_cfg, seed, N_EMO)
        return head_from(arrays, mean, scale, labels.EMOTIONS)

    choice_a = rep[f"audio_cal_{best_variant}"]
    wa = w_emo(choice_a["class_weight_scheme"])

    def fit_audio(X: np.ndarray, yy: np.ndarray) -> heads.NpzHead:
        mean, scale = standardizer(X)
        sw = None if choice_a["class_weight_scheme"] == "none" else wa[yy]
        return head_from(fit_lr((X - mean) / scale, yy, choice_a["C"], sw, seed, cfg["lr"]["max_iter"]), mean, scale, labels.EMOTIONS)

    n_folds = int(cfg["fusion"]["cross_fit_folds"])
    oof_t, fold_id, _ = cross_fit(fit_text, Xt["train"], y["train"], groups, n_folds, seed)
    oof_a, fold_id_a, _ = cross_fit(fit_audio, Xa["train"], y["train"], groups, n_folds, seed)
    assert np.array_equal(fold_id, fold_id_a)
    text_head = heads.TextContextHead.load(out / "text_ctx.npz")
    lp_t_dev, lp_a_dev = text_head.predict_log_proba_batch(Xt["dev"]), cal_heads[best_variant].predict_log_proba_batch(Xa["dev"])
    F = {"train": fusion_rows(oof_t, oof_a, aud["train"]), "dev": fusion_rows(lp_t_dev, lp_a_dev, aud["dev"])}
    rep["cross_fit"] = {"folds": n_folds, "fold_sizes": np.bincount(fold_id, minlength=n_folds).tolist(),
                        "oof_text_train_scores": metrics.summary(metrics.confusion(y["train"], oof_t.argmax(axis=1), N_EMO)),
                        "oof_audio_train_scores": metrics.summary(metrics.confusion(y["train"], oof_a.argmax(axis=1), N_EMO))}

    # fusion lr, poe weight, fused sentiment, thresholds
    fh, choice, scores = lr_grid(F["train"], y["train"], F["dev"], y["dev"], labels.EMOTIONS, cfg["lr"]["C"], cfg["fusion"]["class_weight"],
                                 w_emo, cfg["lr"], seed)
    fh = heads.Fusion(fh.params)
    cal = calibrate(fh, F["dev"], y["dev"])
    fh.save(out / "fusion.npz")
    p_dev = np.exp(fh.predict_log_proba_batch(F["dev"]))
    dev = metrics.summary(metrics.confusion(y["dev"], p_dev.argmax(axis=1), N_EMO))
    jsd_dev = np.array([_jsd(np.exp(lp_t_dev[i]), heads.audio7(aud["dev"]["emo_probs"][i])) for i in range(len(y["dev"]))])
    thr = choose_thresholds(p_dev, y["dev"], cfg["thresholds"])
    thr["jsd_flag"] = round(float(np.quantile(jsd_dev, float(cfg["thresholds"]["jsd_flag"]["quantile"]))), 4)
    rep["fusion"] = {**choice, "dev": {m: round(v, 4) for m, v in dev.items()}, "dev_scores": scores, "calibration_dev": cal,
                     "thresholds": thr, "file": "fusion.npz"}
    poe = {}
    for wgt in cfg["fusion"]["poe_weight_grid"]:
        pred = np.array([heads.fuse_poe(np.exp(lp_t_dev[i]), aud["dev"]["emo_probs"][i], wgt).argmax() for i in range(len(y["dev"]))])
        poe[str(wgt)] = {m: round(v, 4) for m, v in metrics.summary(metrics.confusion(y["dev"], pred, N_EMO)).items()}
    rep["poe"] = {"dev_scores": poe, "weight": float(max(poe, key=lambda s: poe[s]["weighted_f1"]))}
    counts = np.bincount(ys["train"], minlength=N_SENT).astype(np.float64)
    sh, choice, scores = lr_grid(F["train"], ys["train"], F["dev"], ys["dev"], labels.SENTIMENTS, cfg["lr"]["C"], cfg["sentiment"]["class_weight"],
                                 lambda s: counts.sum() / (N_SENT * counts) if s == "balanced" else np.ones(N_SENT), cfg["lr"], seed,
                                 variant="fused")
    sh = heads.SentimentHead(sh.params)
    cal = calibrate(sh, F["dev"], ys["dev"])
    sh.save(out / "sentiment.npz")
    dev = metrics.summary(metrics.confusion(ys["dev"], sh.predict_log_proba_batch(F["dev"]).argmax(axis=1), N_SENT))
    rep["sentiment"] = {"variant": "fused", "file": "sentiment.npz", **choice, "dev": {m: round(v, 4) for m, v in dev.items()},
                        "dev_scores": scores, "calibration_dev": cal}
    rep["status"] = "trained"
    rep["wall_time_s"] = round(time.perf_counter() - t_stage, 1)
    return rep


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a: float(np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0])))  # noqa: E731
    return min(1.0, max(0.0, 0.5 * kl(p) + 0.5 * kl(q)))


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--out", type=Path, default=ROOT / "heads")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "heads.yaml")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--text-only", action="store_true", help="skip the audio and fusion stage; write a text-only sentiment head")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg.get("seed", 0))
    import sklearn
    import threadpoolctl

    t_all = time.perf_counter()
    with threadpoolctl.threadpool_limits(int(cfg.get("threads", 2))):
        utts = {s: load_split(s, args.meld) for s in ("train", "dev")}
        emb = {s: load_text_cache(args.cache, s, utts[s]) for s in ("train", "dev")}
        args.out.mkdir(parents=True, exist_ok=True)
        text = train_text_stage(args.cache, args.out, cfg, utts, emb, seed)
        sel = text["selected"]["text_ctx"]
        stages = {"text_ctx": text}
        if args.text_only:
            stages["sentiment"] = train_sentiment_text_only(args.out, cfg, utts, emb, sel["k"], seed)
            stages["audio_fusion"] = {"status": "pending (needs MELD audio)", "reason": "--text-only"}
        else:
            stages["audio_fusion"] = train_audio_stage(args.cache, args.out, cfg, utts, emb, sel, seed)
            if "sentiment" in stages["audio_fusion"]:
                stages["sentiment"] = stages["audio_fusion"]["sentiment"]
            else:
                stages["sentiment"] = train_sentiment_text_only(args.out, cfg, utts, emb, sel["k"], seed)
    thresholds = dict(sel["thresholds"])
    thresholds.setdefault("jsd_flag", None)
    if stages["audio_fusion"].get("status") == "trained":
        thresholds = stages["audio_fusion"]["fusion"]["thresholds"]
    manifest = {
        "text_only": bool(args.text_only), "k": sel["k"], "class_weight_scheme": sel["class_weight_scheme"], "C": None,
        "temperature": sel["calibration_dev"]["temperature"],
        "thresholds": {"low_conf": thresholds["low_conf"], "high_conf": thresholds["high_conf"], "jsd_flag": thresholds.get("jsd_flag")},
        "thresholds_source": thresholds.get("source"),
        "thresholds_pending": {} if thresholds.get("jsd_flag") is not None else {"jsd_flag": "needs MELD audio"},
        "dev_scores": text["dev_scores"], "selected": text["selected"], "sentiment": stages["sentiment"],
        "audio_fusion": stages["audio_fusion"],
        # per split, csv rows the audio stage dropped because they have no cached clip (None when the stage did not run)
        "audio_rows_dropped": stages["audio_fusion"].get("rows_dropped"),
        "histories": text["histories"],
        "files": {p.name: file_record(p) for p in sorted(args.out.glob("*.npz"))},
        "inputs": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): file_record(p)
                   for p in [args.cache / f"text_{s}.npy" for s in ("train", "dev")] + [args.meld / f"{s}_sent_emo.csv" for s in ("train", "dev")]},
        "config": cfg, "seed": seed, "threads": int(cfg.get("threads", 2)),
        "wall_time_s": {"text_ctx": text["wall_time_s"], "sentiment": stages["sentiment"].get("wall_time_s"),
                        "audio_fusion": stages["audio_fusion"].get("wall_time_s"), "total": round(time.perf_counter() - t_all, 1)},
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "python": platform.python_version(),
        "command": "python tools/train_heads.py " + " ".join(sys.argv[1:] if argv is None else argv),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"selected {sel['grid_point']}: dev weighted f1 {sel['dev']['weighted_f1']:.4f}, T={sel['calibration_dev']['temperature']}, "
          f"thresholds {manifest['thresholds']}; total {manifest['wall_time_s']['total']} s -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
