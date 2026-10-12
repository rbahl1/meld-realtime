"""evaluation metrics in numpy (design 1.11): confusion matrix, per-class, macro
and weighted f1, accuracy, expected calibration error, and a bootstrap over
groups (dialogues). kept free of sklearn so a resample is one matrix sum."""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) counts, rows true, columns predicted."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


def macro_f1(cm: np.ndarray) -> float:
    """mean of the per-class f1 over the classes present in the truth or the
    predictions (sklearn's convention for labels seen in y_true or y_pred)."""
    present = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
    return float(f1_per_class(cm)[present].mean())


def weighted_f1(cm: np.ndarray) -> float:
    support = cm.sum(axis=1).astype(np.float64)
    return float((f1_per_class(cm) * support).sum() / support.sum())


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum())


def summary(cm: np.ndarray) -> dict:
    return {"weighted_f1": weighted_f1(cm), "macro_f1": macro_f1(cm), "accuracy": accuracy(cm)}


def ece(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """expected calibration error of the argmax confidence, equal-width bins on (0, 1]."""
    proba = np.asarray(proba, dtype=np.float64)
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y_true)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def bootstrap_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, n_classes: int,
                       n_resamples: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    """95 % percentile intervals of weighted f1, macro f1 and accuracy when whole
    groups (dialogues) are resampled with replacement."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    per_group = np.zeros((len(uniq), n_classes, n_classes), dtype=np.int64)
    for g in range(len(uniq)):
        m = inv == g
        per_group[g] = confusion(np.asarray(y_true)[m], np.asarray(y_pred)[m], n_classes)
    rng = np.random.default_rng(seed)
    stats = {"weighted_f1": [], "macro_f1": [], "accuracy": []}
    for _ in range(n_resamples):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        cm = per_group[pick].sum(axis=0)
        for name, fn in (("weighted_f1", weighted_f1), ("macro_f1", macro_f1), ("accuracy", accuracy)):
            stats[name].append(fn(cm))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}
