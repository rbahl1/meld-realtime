"""fit the commit policy (extension spec 5) by fitted q-iteration on the prefix traces of
tools/build_commit_traces.py, compare it with three baselines tuned on dev, score every
method once on test, and export heads/commit_policy.npz (c = 0.05).

decision: at each full-window provisional tick (t = 1, 2, 3, 4 s while the vad segment is
open, t < duration + the vad tail) EMIT the fused label now or WAIT; with no emit the label of the endpoint
(slot END of the traces, at t_end) is committed. reward r = 1[label == gold] - c * t_commit, c in {0.02, 0.05, 0.1, 0.2}
per second (deployed c = 0.05: one second of waiting must buy five points of accuracy).

fitted q-iteration (one model across ticks, t and k among the features of meld_rt.commit):
- Q_emit(s) = P(correct | s) - c * t_s. P(correct | s) is a logistic regression on the
  standardized tick features of the train ticks, target 1[argmax fused == gold], with C
  chosen by the log loss of the dev ticks. it does not depend on c.
- Q_wait(s_k) is regressed on max(Q_emit(s_k+1), Q_wait(s_k+1)), or on the endpoint reward
  when s_k is the clip's last tick. the first iteration has no Q_wait (the next tick's
  Q_emit alone); iterations stop when the emit decisions on the train ticks stop changing,
  at most 10.
- Q_wait family and regularization chosen per c on dev mean reward: ridge regression
  (alpha 0.1, 1, 10, 100) or a one-hidden-layer relu mlp with 16 units (alpha 1e-4 to 1e-1,
  adam, warm-started across iterations); ties go to ridge, then to stronger regularization.
- policy: EMIT when Q_emit >= Q_wait, evaluated with the numpy forward pass of
  meld_rt.commit.CommitPolicy, the object the service loads.
baselines, tuned on dev for every c: endpoint; fixed time T in {1, 2, 3, 4} s, the tick times (the
endpoint when the clip has no tick at T); threshold tau on the fused max probability over 0.30 to
0.95 in steps of 0.05 (the first tick at or above it, else the endpoint). ties go to the
earlier T and the lower tau.

test, read once per method and c: mean reward with a 95 % bootstrap interval over dialogues,
accuracy, weighted and macro f1 of the committed labels, mean commit time, the share
committed before the endpoint; paired bootstrap over test dialogues (1000 resamples, seed 0,
the resamples of metrics.bootstrap_by_group) of the mean reward difference fqi minus each
baseline. claim rule, fixed before the run: RL adds measured value over a baseline at a c
only where that paired interval lies above zero; otherwise the numbers are reported and the
gain is not shown. a sensitivity block charges every tick commit the p50 final latency of
the bench (a provisional's own compute is not in t_commit); it changes no choice.

writes heads/commit_policy.npz (the deployed c), results/commit_policy.json and
results/commit_policy.md (tables only). --skip-test writes no test number. traces that are
not in vision mode, or were built with --limit, are refused for the repository's heads/ and
results/ (testing only: pass both elsewhere).

usage: python tools/train_commit_policy.py [--traces cache/] [--heads heads/] [--results results/] [--skip-test]
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
import warnings  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_commit_traces import END, TICKS_S, load_traces  # noqa: E402
from train_heads import file_record, standardizer  # noqa: E402

from meld_rt import labels, metrics  # noqa: E402
from meld_rt.commit import FEATURES, CommitPolicy, tick_features, tick_sequence  # noqa: E402

N_EMO = len(labels.EMOTIONS)
N_TICKS = len(TICKS_S)
COSTS = (0.02, 0.05, 0.1, 0.2)
DEPLOY_C = 0.05
P_C_GRID = (0.01, 0.1, 1.0, 10.0)
RIDGE_ALPHAS = (100.0, 10.0, 1.0, 0.1)          # tie order: stronger regularization first
MLP_ALPHAS = (0.1, 0.01, 0.001, 0.0001)
MLP_HIDDEN = 16
FIXED_T = tuple(int(t) for t in TICKS_S)       # 1, 2, 3, 4 s: the tick times
TAUS = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))   # 0.30 .. 0.95
MAX_ITER = 10
N_RESAMPLES = 1000
SEED = 0
METHODS = ("fqi", "endpoint", "fixed", "threshold")


# ---------------------------------------------------------------- episodes

@dataclass
class Episodes:
    """one split of traces: clips, their tick features and what committing at each slot gives."""
    keys: np.ndarray
    dialogue: np.ndarray
    gold: np.ndarray
    n_ticks: np.ndarray
    t: np.ndarray          # (n, N_TICKS + 1) commit time per slot, slot END the endpoint
    label: np.ndarray      # (n, N_TICKS + 1) argmax of the fused posterior per slot
    maxp: np.ndarray       # (n, N_TICKS) fused max probability at the ticks (nan where no tick)
    X: np.ndarray          # (n, N_TICKS, 14) tick features (nan where no tick)
    meta: dict

    @property
    def n(self) -> int:
        return len(self.gold)

    @property
    def valid(self) -> np.ndarray:
        return np.arange(N_TICKS)[None, :] < self.n_ticks[:, None]

    @property
    def correct(self) -> np.ndarray:
        return self.label == self.gold[:, None]

    def rows(self) -> tuple[np.ndarray, np.ndarray]:
        """(clip, slot) of every tick row."""
        return np.nonzero(self.valid)


def episodes(d: dict) -> Episodes:
    """features of every tick through meld_rt.commit (tick_sequence, tick_features), as the service builds them."""
    n = len(d["keys"])
    nt = np.asarray(d["n_ticks"], np.int64)
    X = np.full((n, N_TICKS, len(FEATURES)), np.nan)
    for i in range(n):
        k = int(nt[i])
        if k == 0:
            continue
        seq = tick_sequence(d["t"][i, :k], d["fused"][i, :k], d["p_text"][i, :k], d["p_audio"][i, :k], d["p_face"][i, :k],
                            d["face_present"][i, :k], d["asr_conf"][i, :k], d["n_words"][i, :k])
        X[i, :k] = np.stack([tick_features(s) for s in seq])
    fused = np.nan_to_num(np.asarray(d["fused"], np.float64), nan=0.0)
    label = fused.argmax(axis=2)
    maxp = np.where(np.arange(N_TICKS)[None, :] < nt[:, None], fused[:, :N_TICKS].max(axis=2), np.nan)
    return Episodes(np.asarray(d["keys"]), np.asarray(d["dialogue"], np.int64), np.asarray(d["gold"], np.int64), nt,
                    np.asarray(d["t"], np.float64), label, maxp, X, d.get("meta", {}))


def rollout(ep: Episodes, emit: np.ndarray, c: float) -> dict:
    """commit at the first tick with emit (only ticks the clip has), else at the endpoint."""
    e = np.asarray(emit, bool) & ep.valid
    slot = np.where(e.any(axis=1), e.argmax(axis=1), END)
    i = np.arange(ep.n)
    label, t = ep.label[i, slot], ep.t[i, slot]
    correct = label == ep.gold
    return {"slot": slot, "label": label, "t": t, "correct": correct, "reward": correct.astype(np.float64) - c * t}


def baseline_emit(ep: Episodes, kind: str, param=None) -> np.ndarray:
    if kind == "endpoint":
        return np.zeros((ep.n, N_TICKS), bool)
    if kind == "fixed":
        return np.repeat((np.arange(N_TICKS) == FIXED_T.index(int(param)))[None, :], ep.n, axis=0)
    if kind == "threshold":
        return np.nan_to_num(ep.maxp, nan=-1.0) >= float(param) - 1e-12
    raise ValueError(kind)


def policy_emit(pol: CommitPolicy, ep: Episodes) -> np.ndarray:
    ci, sj = ep.rows()
    emit = np.zeros((ep.n, N_TICKS), bool)
    if len(ci):
        X = ep.X[ci, sj]
        emit[ci, sj] = pol.q_emit(X) >= pol.q_wait(X)
    return emit


# ---------------------------------------------------------------- models

def fit_p_correct(Z: np.ndarray, y: np.ndarray, C: float, seed: int) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=float(C), max_iter=2000, random_state=seed)
    lr.fit(Z, y.astype(np.int64))
    return lr.coef_[0].astype(np.float64), float(lr.intercept_[0])


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class QWait:
    """the Q_wait regressor of one candidate (ridge or mlp), refit at every fqi iteration."""

    def __init__(self, kind: str, alpha: float, seed: int):
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor

        self.kind, self.alpha = kind, float(alpha)
        if kind == "ridge":
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(MLP_HIDDEN,), activation="relu", alpha=self.alpha, batch_size=256,
                                      learning_rate_init=1e-3, max_iter=200, tol=1e-5, n_iter_no_change=10, random_state=seed,
                                      warm_start=True)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Z, y)

    def export(self) -> dict:
        m = self.model
        if self.kind == "ridge":
            return {"kind": "ridge", "w": np.asarray(m.coef_, np.float64).reshape(-1), "b": float(np.asarray(m.intercept_).reshape(()))}
        return {"kind": "mlp", "W1": np.asarray(m.coefs_[0], np.float64), "b1": np.asarray(m.intercepts_[0], np.float64),
                "W2": np.asarray(m.coefs_[1], np.float64)[:, 0], "b2": float(m.intercepts_[1][0])}


def candidates() -> list[tuple[str, float]]:
    return [("ridge", a) for a in RIDGE_ALPHAS] + [("mlp", a) for a in MLP_ALPHAS]


def fqi(tr: Episodes, mean: np.ndarray, scale: np.ndarray, p_w: np.ndarray, p_b: float, c: float, kind: str, alpha: float,
        seed: int, dev: Episodes | None = None) -> tuple[CommitPolicy, dict]:
    """fitted q-iteration on the train ticks; returns the policy and its iteration record."""
    ci, sj = tr.rows()
    X = tr.X[ci, sj]
    Z = (X - mean) / scale
    # the next tick of each row, or -1 when the row is the clip's last tick (then the endpoint reward)
    row_of = -np.ones((tr.n, N_TICKS), np.int64)
    row_of[ci, sj] = np.arange(len(ci))
    nxt = np.where(sj + 1 < tr.n_ticks[ci], row_of[ci, np.minimum(sj + 1, N_TICKS - 1)], -1)
    r_end = tr.correct[ci, END].astype(np.float64) - c * tr.t[ci, END]
    no_wait = {"kind": "ridge", "w": np.zeros(len(FEATURES)), "b": 0.0}
    qe = CommitPolicy.build(mean, scale, p_w, p_b, c, no_wait, "q_emit").q_emit(X)
    qw = np.full(len(ci), -np.inf)
    model = QWait(kind, alpha, seed)
    prev_policy, rec = None, {"iterations": [], "converged": False}
    pol = None
    for it in range(1, MAX_ITER + 1):
        v_next = np.where(nxt >= 0, np.maximum(qe[np.maximum(nxt, 0)], qw[np.maximum(nxt, 0)]), r_end)
        model.fit(Z, v_next)
        pol = CommitPolicy.build(mean, scale, p_w, p_b, c, model.export(), f"fqi_{kind}_c{c:g}")
        qw = pol.q_wait(X)
        policy = qe >= qw
        step = {"iteration": it, "emit_share_train_ticks": round(float(policy.mean()), 4),
                "train_reward": round(float(rollout(tr, _scatter(tr, ci, sj, policy), c)["reward"].mean()), 6)}
        if dev is not None:
            step["dev_reward"] = round(float(rollout(dev, policy_emit(pol, dev), c)["reward"].mean()), 6)
        if prev_policy is not None:
            step["changed"] = int((policy != prev_policy).sum())
        rec["iterations"].append(step)
        if prev_policy is not None and np.array_equal(policy, prev_policy):
            rec["converged"] = True
            break
        prev_policy = policy
    rec["n_iterations"] = len(rec["iterations"])
    return pol, rec


def _scatter(ep: Episodes, ci: np.ndarray, sj: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros((ep.n, N_TICKS), bool)
    out[ci, sj] = values
    return out


# ---------------------------------------------------------------- statistics

def bootstrap_mean(values: np.ndarray, groups: np.ndarray, n: int = N_RESAMPLES, seed: int = SEED) -> list[float]:
    """95 % percentile interval of the mean over clips when whole dialogues are resampled (the
    draws of metrics.bootstrap_by_group)."""
    return _boot(np.asarray(values, np.float64), groups, n, seed)[0]


def paired_bootstrap_reward(r_a: np.ndarray, r_b: np.ndarray, groups: np.ndarray, n: int = N_RESAMPLES, seed: int = SEED) -> dict:
    """mean(r_a - r_b) over clips with whole dialogues resampled with replacement, both methods on the
    same resample: the point difference, the 95 % percentile interval, whether it excludes zero, whether
    it lies above zero, and the share of resamples at or below zero."""
    d = np.asarray(r_a, np.float64) - np.asarray(r_b, np.float64)
    (lo, hi), draws, G = _boot(d, groups, n, seed)
    return {"n": len(d), "n_groups": G, "n_resamples": int(n), "seed": int(seed), "diff": round(float(d.mean()), 6),
            "ci95": [round(lo, 6), round(hi, 6)], "excludes_zero": bool(lo > 0 or hi < 0), "above_zero": bool(lo > 0),
            "share_at_or_below_zero": round(float((draws <= 0).mean()), 6)}


def _boot(v: np.ndarray, groups: np.ndarray, n: int, seed: int) -> tuple[list[float], np.ndarray, int]:
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = len(uniq)
    s = np.bincount(inv, weights=v, minlength=G)
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.zeros(n)
    for r in range(n):
        pick = rng.integers(0, G, size=G)
        draws[r] = s[pick].sum() / cnt[pick].sum()
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], draws, G


def method_summary(ep: Episodes, ro: dict, with_ci: bool) -> dict:
    cm = metrics.confusion(ep.gold, ro["label"], N_EMO)
    s = metrics.summary(cm)
    out = {"n": ep.n, "mean_reward": round(float(ro["reward"].mean()), 6), "accuracy": round(s["accuracy"], 6),
           "weighted_f1": round(s["weighted_f1"], 6), "macro_f1": round(s["macro_f1"], 6),
           "mean_commit_s": round(float(ro["t"].mean()), 4), "early_share": round(float((ro["slot"] < END).mean()), 6),
           "commit_slot_counts": {("end" if j == END else f"t{TICKS_S[j]:g}"): int((ro["slot"] == j).sum()) for j in range(END + 1)}}
    if with_ci:
        out["mean_reward_ci95"] = [round(v, 6) for v in bootstrap_mean(ro["reward"], ep.dialogue)]
        out["ci95"] = metrics.bootstrap_by_group(ep.gold, ro["label"], ep.dialogue, N_EMO, N_RESAMPLES, SEED)
    return out


# ---------------------------------------------------------------- training

def train(tr: Episodes, dev: Episodes, costs: tuple[float, ...], seed: int, log=print) -> dict:
    """every choice on dev: P(correct) C, then per c the Q_wait candidate and the baselines' parameters."""
    ci, sj = tr.rows()
    Xtr = tr.X[ci, sj]
    if not len(Xtr):
        raise SystemExit("the train traces have no tick")
    mean, scale = standardizer(Xtr)
    ytr = tr.correct[ci, sj]
    di, dj = dev.rows()
    Xdv, ydv = dev.X[di, dj], dev.correct[di, dj]
    pgrid, best = {}, None
    for C in P_C_GRID:
        w, b = fit_p_correct((Xtr - mean) / scale, ytr, C, seed)
        p = 1.0 / (1.0 + np.exp(-(((Xdv - mean) / scale) @ w + b)))
        ll = log_loss(p, ydv.astype(np.float64))
        pgrid[f"C={C:g}"] = {"dev_log_loss": round(ll, 6), "dev_brier": round(float(np.mean((p - ydv) ** 2)), 6)}
        if best is None or ll < best[0]:
            best = (ll, C, w, b)
    _, C, p_w, p_b = best
    log(f"P(correct): C={C:g}, dev log loss {best[0]:.4f} (train ticks {len(Xtr)}, dev ticks {len(Xdv)}, "
        f"base rate train {ytr.mean():.3f} dev {ydv.mean():.3f})")
    rep = {"p_correct": {"C": C, "grid": pgrid, "train_ticks": len(Xtr), "dev_ticks": len(Xdv),
                         "train_correct_rate": round(float(ytr.mean()), 6), "dev_correct_rate": round(float(ydv.mean()), 6),
                         "weights": dict(zip(FEATURES, np.round(p_w, 6).tolist())), "intercept": round(p_b, 6)},
           "standardizer": {"mean": dict(zip(FEATURES, np.round(mean, 6).tolist())), "scale": dict(zip(FEATURES, np.round(scale, 6).tolist()))},
           "costs": {}}
    policies = {}
    for c in costs:
        t0 = time.perf_counter()
        grid, chosen = {}, None
        for kind, alpha in candidates():
            pol, rec = fqi(tr, mean, scale, p_w, p_b, c, kind, alpha, seed, dev)
            r = float(rollout(dev, policy_emit(pol, dev), c)["reward"].mean())
            grid[f"{kind},alpha={alpha:g}"] = {"dev_reward": round(r, 6), "n_iterations": rec["n_iterations"], "converged": rec["converged"],
                                               "iterations": rec["iterations"]}
            if chosen is None or r > chosen[0] + 1e-12:
                chosen = (r, kind, alpha, pol, rec)
        r, kind, alpha, pol, rec = chosen
        base = {}
        for kind_b, params in (("fixed", FIXED_T), ("threshold", TAUS)):
            scores = {str(p): float(rollout(dev, baseline_emit(dev, kind_b, p), c)["reward"].mean()) for p in params}
            pick = max(params, key=lambda p: (scores[str(p)], -params.index(p)))
            base[kind_b] = {"param": pick, "dev_grid": {k: round(v, 6) for k, v in scores.items()}}
        base["endpoint"] = {"param": None}
        policies[c] = pol
        rep["costs"][str(c)] = {"fqi": {"kind": kind, "alpha": alpha, "dev_reward": round(r, 6), "n_iterations": rec["n_iterations"],
                                        "converged": rec["converged"], "grid": grid, "name": pol.name},
                                "baselines": base, "wall_s": round(time.perf_counter() - t0, 1)}
        log(f"c={c:g}: fqi {kind} alpha={alpha:g} dev reward {r:.4f} ({rec['n_iterations']} iterations), fixed T={base['fixed']['param']}, "
            f"tau={base['threshold']['param']} ({time.perf_counter() - t0:.1f} s)")
    return {"report": rep, "policies": policies, "mean": mean, "scale": scale, "p_w": p_w, "p_b": p_b}


def method_emit(name: str, ep: Episodes, c: float, rep_c: dict, pol: CommitPolicy) -> np.ndarray:
    if name == "fqi":
        return policy_emit(pol, ep)
    return baseline_emit(ep, name, rep_c["baselines"][name]["param"])


def evaluate(ep: Episodes, costs, rep: dict, policies: dict, with_ci: bool, latency_s: float | None = None) -> dict:
    """per c: every method's summary and the paired bootstrap of fqi minus each baseline."""
    out = {}
    for c in costs:
        rc = rep["costs"][str(c)]
        ro = {m: rollout(ep, method_emit(m, ep, c, rc, policies[c]), c) for m in METHODS}
        res = {"methods": {m: {"param": rc["fqi"]["name"] if m == "fqi" else rc["baselines"][m]["param"],
                               **method_summary(ep, ro[m], with_ci)} for m in METHODS}}
        if with_ci:
            res["paired_fqi_minus"] = {m: paired_bootstrap_reward(ro["fqi"]["reward"], ro[m]["reward"], ep.dialogue)
                                       for m in METHODS if m != "fqi"}
            res["rl_adds_value_over"] = {m: res["paired_fqi_minus"][m]["above_zero"] for m in res["paired_fqi_minus"]}
            if latency_s is not None:
                adj = {m: ro[m]["reward"] - c * latency_s * (ro[m]["slot"] < END) for m in METHODS}
                res["sensitivity_tick_latency"] = {
                    "latency_s": latency_s, "mean_reward": {m: round(float(adj[m].mean()), 6) for m in METHODS},
                    "paired_fqi_minus": {m: paired_bootstrap_reward(adj["fqi"], adj[m], ep.dialogue) for m in METHODS if m != "fqi"}}
        out[str(c)] = res
    return out


# ---------------------------------------------------------------- rendering

def pct(v) -> str:
    return "" if v is None else f"{100 * v:.2f}"


def render(res: dict) -> str:
    L = ["# Commit policy", "", f"Written by `{res['command']}`.", ""]
    L += ["## Inputs", "", "| file | bytes | sha256 | clips | ticks | mode |", "| --- | --- | --- | --- | --- | --- |"]
    for s, r in res["inputs"].items():
        L.append(f"| `{r['file']}` | {r['bytes']} | `{r['sha256']}` | {r['clips']} | {r['ticks']} | {r['mode']} |")
    p = res["train"]["p_correct"]
    L += ["", "## P(correct) model (dev log loss)", "", "| C | dev log loss | dev Brier |", "| --- | --- | --- |"]
    L += [f"| {k[2:]}{' (chosen)' if float(k[2:]) == p['C'] else ''} | {v['dev_log_loss']:.4f} | {v['dev_brier']:.4f} |"
          for k, v in p["grid"].items()]
    L += ["", "## Choices made on dev", "",
          "| c | fqi Q_wait | dev reward fqi | iterations | fixed T | dev reward fixed | tau | dev reward threshold | dev reward endpoint |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c, rc in res["train"]["costs"].items():
        dv = res["dev"][c]["methods"]
        f, b = rc["fqi"], rc["baselines"]
        L.append(f"| {c} | {f['kind']} alpha {f['alpha']:g} | {dv['fqi']['mean_reward']:.4f} | {f['n_iterations']} | "
                 f"{b['fixed']['param']} s | {dv['fixed']['mean_reward']:.4f} | {b['threshold']['param']} | "
                 f"{dv['threshold']['mean_reward']:.4f} | {dv['endpoint']['mean_reward']:.4f} |")
    for split in ("dev", "test"):
        if split not in res:
            continue
        L += ["", f"## {split.capitalize()}: every method and cost", "",
              "| c | method | mean reward | 95 % CI | accuracy | weighted F1 | macro F1 | mean commit s | committed before the endpoint |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c, rc in res[split].items():
            for m, s in rc["methods"].items():
                ci = s.get("mean_reward_ci95")
                name = m if s["param"] is None else f"{m} ({s['param']})"
                L.append(f"| {c} | {name} | {s['mean_reward']:.4f} | {'' if ci is None else f'[{ci[0]:.4f}, {ci[1]:.4f}]'} | "
                         f"{pct(s['accuracy'])} | {pct(s['weighted_f1'])} | {pct(s['macro_f1'])} | {s['mean_commit_s']:.3f} | "
                         f"{pct(s['early_share'])} % |")
        if split == "test":
            L += ["", "## Test: fqi minus each baseline (paired bootstrap over dialogues)", "",
                  "| c | baseline | reward difference | 95 % CI | interval above zero |", "| --- | --- | --- | --- | --- |"]
            for c, rc in res[split].items():
                for m, d in rc.get("paired_fqi_minus", {}).items():
                    L.append(f"| {c} | {m} | {d['diff']:.4f} | [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | {'yes' if d['above_zero'] else 'no'} |")
            sens = [(c, rc["sensitivity_tick_latency"]) for c, rc in res[split].items() if "sensitivity_tick_latency" in rc]
            if sens:
                L += ["", f"## Test: tick commits charged {sens[0][1]['latency_s']} s more (sensitivity, no choice changed)", "",
                      "| c | baseline | reward difference | 95 % CI | interval above zero |", "| --- | --- | --- | --- | --- |"]
                for c, s in sens:
                    for m, d in s["paired_fqi_minus"].items():
                        L.append(f"| {c} | {m} | {d['diff']:.4f} | [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | {'yes' if d['above_zero'] else 'no'} |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def trace_record(path: Path, ep: Episodes) -> dict:
    return {"file": str(path), **file_record(path), "clips": ep.n, "ticks": int(ep.valid.sum()), "mode": ep.meta.get("mode")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", type=Path, default=ROOT / "cache", help="folder with commit_traces_{train,dev,test}.npz")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads", help="where commit_policy.npz goes")
    ap.add_argument("--results", type=Path, default=ROOT / "results", help="where commit_policy.json and .md go")
    ap.add_argument("--costs", default=",".join(f"{c:g}" for c in COSTS))
    ap.add_argument("--deploy-c", type=float, default=DEPLOY_C)
    ap.add_argument("--skip-test", action="store_true", help="no test traces read, no test number written")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    costs = tuple(float(c) for c in args.costs.split(","))
    if args.deploy_c not in costs:
        ap.error("--deploy-c must be one of --costs")
    import threadpoolctl

    t_all = time.perf_counter()
    cmd = "python tools/train_commit_policy.py " + " ".join(sys.argv[1:] if argv is None else argv)
    paths = {s: args.traces / f"commit_traces_{s}.npz" for s in ("train", "dev", "test")}
    eps = {s: episodes(load_traces(paths[s])) for s in ("train", "dev")}
    modes = {s: eps[s].meta.get("mode") for s in eps}
    if len(set(modes.values())) != 1:
        raise SystemExit(f"train and dev traces were built in different modes: {modes}")
    repo_out = {args.heads.resolve(), args.results.resolve()} & {(ROOT / "heads").resolve(), (ROOT / "results").resolve()}
    if modes["train"] != "vision" and repo_out:
        # the service consults the policy on fusion3 posteriors only, so a policy fitted on two-way traces is never deployed
        raise SystemExit(f"traces in mode {modes['train']!r}: pass --heads and --results outside heads/ and results/ (testing only)")
    _refuse_limited(eps, repo_out)
    with threadpoolctl.threadpool_limits(2):
        fit = train(eps["train"], eps["dev"], costs, args.seed)
        rep = fit["report"]
        res = {"command": cmd, "costs": list(costs), "deploy_c": args.deploy_c, "seed": args.seed, "features": list(FEATURES),
               "grids": {"p_correct_C": list(P_C_GRID), "ridge_alpha": list(RIDGE_ALPHAS), "mlp_alpha": list(MLP_ALPHAS),
                         "mlp_hidden": MLP_HIDDEN, "fixed_T": list(FIXED_T), "tau": list(TAUS), "max_iterations": MAX_ITER},
               "inputs": {s: trace_record(paths[s], eps[s]) for s in eps}, "train": rep,
               "dev": evaluate(eps["dev"], costs, rep, fit["policies"], with_ci=False),
               "claim_rule": ("RL adds measured value over a baseline at a cost c only where the paired 95 % interval of the test "
                              "reward difference fqi minus that baseline lies above zero")}
        if not args.skip_test:
            if not paths["test"].exists():
                raise SystemExit(f"{paths['test']} is missing (tools/build_commit_traces.py --splits test), or pass --skip-test")
            eps["test"] = episodes(load_traces(paths["test"]))
            if eps["test"].meta.get("mode") != modes["train"]:
                raise SystemExit("test traces were built in another mode than train and dev")
            _refuse_limited({"test": eps["test"]}, repo_out)
            res["inputs"]["test"] = trace_record(paths["test"], eps["test"])
            lat = (eps["test"].meta.get("endpoint") or {}).get("final_p50_s")
            res["test"] = evaluate(eps["test"], costs, rep, fit["policies"], with_ci=True, latency_s=lat)
            for c in costs:
                m = res["test"][str(c)]
                print(f"test c={c:g}: " + ", ".join(f"{k} {v['mean_reward']:.4f}" for k, v in m["methods"].items())
                      + "; fqi minus " + ", ".join(f"{k} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                                                   for k, d in m["paired_fqi_minus"].items()), flush=True)
    c = args.deploy_c
    pol = fit["policies"][c]
    summary = {"c": c, "fqi": {k: v for k, v in rep["costs"][str(c)]["fqi"].items() if k != "grid"},
               "p_correct_C": rep["p_correct"]["C"], "dev": res["dev"][str(c)]["methods"]["fqi"],
               "test": res["test"][str(c)]["methods"]["fqi"] if "test" in res else None,
               "test_paired_fqi_minus": res["test"][str(c)]["paired_fqi_minus"] if "test" in res else None,
               "traces_mode": modes["train"], "command": cmd}
    deployed = CommitPolicy.build(fit["mean"], fit["scale"], fit["p_w"], fit["p_b"], c, _qw_of(pol), pol.name, summary)
    args.heads.mkdir(parents=True, exist_ok=True)
    out_npz = args.heads / "commit_policy.npz"
    deployed.save(out_npz)
    res["export"] = {"file": str(out_npz), **file_record(out_npz), "c": c, "name": pol.name}
    res["wall_s"] = round(time.perf_counter() - t_all, 1)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "commit_policy.json").write_text(json.dumps(res, indent=1))
    (args.results / "commit_policy.md").write_text(render(res))
    print(f"commit policy c={c:g} ({pol.name}) -> {out_npz}; {args.results / 'commit_policy.json'}, {args.results / 'commit_policy.md'} "
          f"in {res['wall_s']} s")
    return 0


def _refuse_limited(eps: dict, repo_out: set) -> None:
    """a subset (build_commit_traces.py --limit) never becomes the deployed policy or the reported result."""
    limited = {s: e.meta.get("limit") for s, e in eps.items() if e.meta.get("limit") is not None}
    if limited and repo_out:
        raise SystemExit(f"traces built with --limit {limited}: pass --heads and --results outside heads/ and results/ (testing only)")


def _qw_of(pol: CommitPolicy) -> dict:
    if pol.qw_kind == "ridge":
        return {"kind": "ridge", "w": pol.qw_w, "b": pol.qw_b}
    return {"kind": "mlp", "W1": pol.qw_W1, "b1": pol.qw_b1, "W2": pol.qw_W2, "b2": pol.qw_b2}


if __name__ == "__main__":
    sys.exit(main())
