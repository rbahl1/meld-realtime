"""commit policy (extension spec 5): when to commit the emotion label while the
person is still talking. at each full-window provisional tick (1 Hz in the first
4 s of a segment, ticks in the endpoint silence included) the policy either EMITs
the fused label now or WAITs for the next tick; at the endpoint the final is committed. the reward is
1[label == gold] - c * t_commit (seconds since the segment start).

the policy is fitted offline by fitted q-iteration (tools/train_commit_policy.py)
on the prefix traces of tools/build_commit_traces.py:
  Q_emit(s) = P(correct | s) - c * t_s   (logistic regression, no c dependence)
  Q_wait(s) = ridge regression or a one-hidden-layer mlp (16 units) on the backed-up
              value max(Q_emit(s'), Q_wait(s')) of the next tick, or the endpoint reward
  EMIT when Q_emit >= Q_wait.
both models read the same standardized tick features. this module holds what the
training and the service share: the tick features, the tick bookkeeping and the
forward pass of heads/commit_policy.npz. numpy only (jsd from meld_rt.discrepancy,
which is numpy only too).

tick (the service's dict, spec 3.4): t_s, k, fused, prev_fused (or None),
same_label_run, p_text, p_audio, p_face (7 each, MELD order), face_present,
asr_conf, n_words.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from meld_rt.discrepancy import jsd

N_EMO = 7
EPS = 1e-12
UNIFORM_TOL = 1e-6
FEATURES = ("max_p", "margin", "entropy_bits", "t_s", "k", "stability", "same_label_run", "max_p_text", "max_p_audio",
            "max_p_face", "face_present", "agreement", "n_words_div10", "asr_conf")
T_COL = FEATURES.index("t_s")
QW_KINDS = ("ridge", "mlp")
VERSION = 1


def _norm(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 0.0, None)
    if p.size != N_EMO:
        raise ValueError(f"expected a 7-way posterior, got {p.size} values")
    s = p.sum()
    return p / s if s > 0 else np.full(N_EMO, 1.0 / N_EMO)


def is_uniform(p) -> bool:
    """a uniform posterior (gated text, no face) has no argmax worth counting."""
    p = _norm(p)
    return bool(p.max() - p.min() < UNIFORM_TOL)


def entropy_bits(p) -> float:
    p = _norm(p)
    nz = p > EPS
    return float(-(p[nz] * np.log2(p[nz])).sum())


def tick_features(tick: dict) -> np.ndarray:
    """(14,) float64 in FEATURES order. stability is 1 - jsd(prev_fused, fused) (bits), 1 at
    the first tick. agreement counts the modalities whose argmax equals the fused argmax; a
    uniform posterior (text gated by the asr confidence, no face) and an absent face do not count."""
    fused = _norm(tick["fused"])
    srt = np.sort(fused)
    lab = int(np.argmax(fused))
    prev = tick.get("prev_fused")
    stability = 1.0 if prev is None else 1.0 - jsd(prev, fused)
    face_present = bool(tick.get("face_present", False))
    agree = 0
    for name in ("p_text", "p_audio", "p_face"):
        p = _norm(tick[name])
        if is_uniform(p) or (name == "p_face" and not face_present):
            continue
        agree += int(np.argmax(p)) == lab
    return np.array([srt[-1], srt[-1] - srt[-2], entropy_bits(fused), float(tick["t_s"]), float(tick["k"]), stability,
                     float(tick["same_label_run"]), _norm(tick["p_text"]).max(), _norm(tick["p_audio"]).max(),
                     _norm(tick["p_face"]).max(), float(face_present), float(agree), float(tick.get("n_words", 0)) / 10.0,
                     float(tick.get("asr_conf", 0.0))], dtype=np.float64)


def tick_sequence(t_s, fused, p_text, p_audio, p_face, face_present, asr_conf, n_words, hz: float = 1.0) -> list[dict]:
    """the tick dicts of one utterance in order, with prev_fused and same_label_run kept as
    the service keeps them (meld_rt.service: 1 at the first tick, +1 while the fused argmax
    stays), k = max(1, round(t_s * hz))."""
    out: list[dict] = []
    for j in range(len(t_s)):
        f = _norm(fused[j])
        prev = out[-1] if out else None
        run = prev["same_label_run"] + 1 if prev is not None and int(np.argmax(prev["fused"])) == int(np.argmax(f)) else 1
        out.append({"t_s": float(t_s[j]), "k": max(1, round(float(t_s[j]) * hz)), "fused": f,
                    "prev_fused": prev["fused"] if prev is not None else None, "same_label_run": run,
                    "p_text": _norm(p_text[j]), "p_audio": _norm(p_audio[j]), "p_face": _norm(p_face[j]),
                    "face_present": bool(face_present[j]), "asr_conf": float(asr_conf[j]), "n_words": int(n_words[j])})
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class CommitPolicy:
    """EMIT / WAIT at a provisional tick from heads/commit_policy.npz.

    file keys: feature_names, mean, scale (standardizer), p_w, p_b (P(correct) logistic
    regression), c, qw_kind ("ridge": qw_w, qw_b; "mlp": qw_W1, qw_b1, qw_W2, qw_b2, relu),
    name, version, summary (json: choices and dev / test numbers)."""

    def __init__(self, params: dict):
        names = [str(n) for n in np.asarray(params["feature_names"]).reshape(-1)]
        if tuple(names) != FEATURES:
            raise ValueError(f"policy features {names} differ from meld_rt.commit.FEATURES")
        self.params = dict(params)
        self.feature_names = names
        self.mean = np.asarray(params["mean"], dtype=np.float64).reshape(-1)
        self.scale = np.asarray(params["scale"], dtype=np.float64).reshape(-1)
        self.p_w = np.asarray(params["p_w"], dtype=np.float64).reshape(-1)
        self.p_b = float(np.asarray(params["p_b"]).reshape(()))
        self.c = float(np.asarray(params["c"]).reshape(()))
        self.qw_kind = str(np.asarray(params["qw_kind"]).reshape(()))
        n = len(FEATURES)
        if self.mean.shape != (n,) or self.scale.shape != (n,) or self.p_w.shape != (n,):
            raise ValueError("standardizer or P(correct) weights do not match the features")
        if self.qw_kind == "ridge":
            self.qw_w = np.asarray(params["qw_w"], dtype=np.float64).reshape(-1)
            self.qw_b = float(np.asarray(params["qw_b"]).reshape(()))
            if self.qw_w.shape != (n,):
                raise ValueError("ridge weights do not match the features")
        elif self.qw_kind == "mlp":
            self.qw_W1 = np.asarray(params["qw_W1"], dtype=np.float64)
            self.qw_b1 = np.asarray(params["qw_b1"], dtype=np.float64).reshape(-1)
            self.qw_W2 = np.asarray(params["qw_W2"], dtype=np.float64).reshape(-1)
            self.qw_b2 = float(np.asarray(params["qw_b2"]).reshape(()))
            if self.qw_W1.shape != (n, self.qw_b1.size) or self.qw_W2.shape != (self.qw_b1.size,):
                raise ValueError("mlp weights do not match the features")
        else:
            raise ValueError(f"qw_kind must be one of {QW_KINDS}, got {self.qw_kind!r}")
        self.name = str(np.asarray(params.get("name", f"fqi_{self.qw_kind}_c{self.c:g}")).reshape(()))
        raw = params.get("summary")
        self.summary = json.loads(str(np.asarray(raw).reshape(()))) if raw is not None else {}

    @classmethod
    def load(cls, path) -> CommitPolicy:
        with np.load(Path(path), allow_pickle=False) as z:
            params = {k: z[k] for k in z.files}
        return cls(params)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: np.asarray(v) for k, v in self.params.items()})

    @staticmethod
    def build(mean, scale, p_w, p_b, c: float, qw: dict, name: str, summary: dict | None = None) -> CommitPolicy:
        """qw: {"kind": "ridge", "w", "b"} or {"kind": "mlp", "W1", "b1", "W2", "b2"}."""
        p = {"feature_names": np.array(FEATURES), "mean": np.asarray(mean, np.float64), "scale": np.asarray(scale, np.float64),
             "p_w": np.asarray(p_w, np.float64).reshape(-1), "p_b": np.array(float(p_b)), "c": np.array(float(c)),
             "qw_kind": np.array(qw["kind"]), "name": np.array(name), "version": np.array(VERSION),
             "summary": np.array(json.dumps(summary or {}))}
        for k, v in qw.items():
            if k != "kind":
                p[f"qw_{k}"] = np.asarray(v, np.float64)
        return CommitPolicy(p)

    # -- forward ---------------------------------------------------------
    def features(self, tick: dict) -> np.ndarray:
        return tick_features(tick)

    def _x(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = X[None, :] if X.ndim == 1 else X
        if X.shape[1] != len(FEATURES):
            raise ValueError(f"expected {len(FEATURES)} features, got {X.shape[1]}")
        return X

    def standardize(self, X) -> np.ndarray:
        return (self._x(X) - self.mean) / self.scale

    def p_correct(self, X) -> np.ndarray:
        return _sigmoid(self.standardize(X) @ self.p_w + self.p_b)

    def q_emit(self, X) -> np.ndarray:
        X = self._x(X)
        return self.p_correct(X) - self.c * X[:, T_COL]

    def q_wait(self, X) -> np.ndarray:
        Z = self.standardize(X)
        if self.qw_kind == "ridge":
            return Z @ self.qw_w + self.qw_b
        return np.maximum(Z @ self.qw_W1 + self.qw_b1, 0.0) @ self.qw_W2 + self.qw_b2

    def decide(self, features) -> tuple[bool, float, float]:
        """(emit, q_emit, q_wait) for one feature vector; emit when q_emit >= q_wait."""
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        qe, qw = float(self.q_emit(x)[0]), float(self.q_wait(x)[0])
        return bool(qe >= qw), qe, qw
