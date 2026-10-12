"""heads: numpy forward passes, training helpers, cross-fitting, fusion features, metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_heads import (  # noqa: E402
    choose_thresholds,
    cross_fit,
    fit_lr,
    fit_mlp,
    head_from,
    lr_grid,
    standardizer,
    text_matrix,
)

from meld_rt import heads, labels, metrics  # noqa: E402
from meld_rt.data import Utt  # noqa: E402

MLP_CFG = {"hidden": 32, "activation": "relu", "alpha": 1e-4, "batch_size": 32, "learning_rate_init": 1e-2,
           "max_epochs": 60, "patience": 8}
LR_CFG = {"max_iter": 500}


def synthetic(n: int, n_classes: int = 7, dim: int = 20, seed: int = 0, noise: float = 0.3):
    centers = np.random.default_rng(123).standard_normal((n_classes, dim)) * 2.0
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    X = (centers[y] + noise * rng.standard_normal((n, dim))).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def problem():
    Xtr, ytr = synthetic(200, seed=0)
    Xdev, ydev = synthetic(100, seed=1)
    Xte, yte = synthetic(100, seed=2)
    return Xtr, ytr, Xdev, ydev, Xte, yte


@pytest.fixture(scope="module")
def mlp_head(problem):
    Xtr, ytr, Xdev, ydev, _, _ = problem
    mean, scale = standardizer(Xtr)
    arrays, rec = fit_mlp((Xtr - mean) / scale, ytr, np.ones(len(ytr)), (Xdev - mean) / scale, ydev, MLP_CFG, 0, 7)
    return head_from(arrays, mean, scale, labels.EMOTIONS), rec


def test_synthetic_mlp_reaches_accuracy(mlp_head, problem):
    head, rec = mlp_head
    _, _, _, _, Xte, yte = problem
    acc = (head.predict_log_proba_batch(Xte).argmax(axis=1) == yte).mean()
    assert acc > 0.9
    assert rec["best_epoch"] >= 0 and rec["epochs"] <= MLP_CFG["max_epochs"]


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_numpy_forward_matches_sklearn(problem):
    from sklearn.neural_network import MLPClassifier

    Xtr, ytr, _, _, Xte, _ = problem
    mean, scale = standardizer(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(16,), random_state=0, max_iter=400).fit((Xtr - mean) / scale, ytr)
    head = head_from(heads.mlp_params(m.coefs_, m.intercepts_, "relu"), mean, scale, labels.EMOTIONS)
    np.testing.assert_allclose(np.exp(head.predict_log_proba_batch(Xte)), m.predict_proba((Xte - mean) / scale), atol=1e-5)


def test_log_proba_sums_to_one(mlp_head, problem):
    head, _ = mlp_head
    _, _, _, _, Xte, _ = problem
    lp = head.predict_log_proba(Xte[0])
    assert lp.shape == (7,)
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-9)
    head.temperature = 2.5
    assert np.exp(head.predict_log_proba(Xte[0])).sum() == pytest.approx(1.0, abs=1e-9)
    lr = head_from(fit_lr(Xte, np.arange(100) % 7, 1.0, None, 0, 200), np.zeros(20), np.ones(20), labels.EMOTIONS)
    assert np.exp(lr.predict_log_proba(Xte[3])).sum() == pytest.approx(1.0, abs=1e-9)


def test_save_load_round_trip(mlp_head, problem, tmp_path):
    head, _ = mlp_head
    _, _, _, _, Xte, _ = problem
    head.params["k"] = np.array(1)
    head.save(tmp_path / "h.npz")
    back = heads.NpzHead.load(tmp_path / "h.npz")
    assert back.kind == "mlp" and back.k == 1 and back.classes == labels.EMOTIONS
    np.testing.assert_allclose(back.predict_log_proba_batch(Xte), head.predict_log_proba_batch(Xte), atol=1e-6)


class _Memorizer(heads.NpzHead):
    """1-nn over its training rows: perfect in-sample, chance out of sample."""

    def __init__(self, X, y):  # noqa: D107
        self.X, self.y = X, y

    def predict_log_proba_batch(self, X):
        d = ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
        lp = np.full((len(X), 7), np.log(1e-6))
        lp[np.arange(len(X)), self.y[d.argmin(1)]] = 0.0
        return lp


def test_cross_fit_never_uses_in_fold_rows():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 5)).astype(np.float32)
    y = rng.integers(0, 7, n)  # labels independent of X
    groups = np.repeat(np.arange(30), 10)
    seen = []

    def fit(Xf, yf):
        seen.append(Xf)
        return _Memorizer(Xf, yf)

    oof, fold_id, train_rows = cross_fit(fit, X, y, groups, n_folds=5, seed=0)
    assert sorted(np.unique(fold_id)) == [0, 1, 2, 3, 4]
    for f, tr in enumerate(train_rows):
        te = np.flatnonzero(fold_id == f)
        assert np.intersect1d(tr, te).size == 0
        assert len(tr) + len(te) == n
        assert np.array_equal(seen[f], X[tr])
    # whole dialogues land in one fold
    assert all(len(np.unique(fold_id[groups == g])) == 1 for g in np.unique(groups))
    # the memorizer is perfect in sample; out of fold it is at chance on random labels
    oof_acc = (oof.argmax(1) == y).mean()
    assert oof_acc < 0.35
    in_sample = (_Memorizer(X, y).predict_log_proba_batch(X).argmax(1) == y).mean()
    assert in_sample == 1.0


def test_build_fusion_features_length_and_entropies():
    lt = heads.log_softmax(np.array([3.0, 0, 0, 0, 0, 0, 0]))
    la = np.full(7, np.log(1 / 7))
    f = heads.build_fusion_features(lt, la, 0.9, 0.1, 0.05)
    assert f.shape == (19,) and f.dtype == np.float32
    assert f[15] == pytest.approx(np.log2(7), abs=1e-5)  # uniform audio entropy in bits
    assert f[14] < f[15]
    assert list(f[16:]) == pytest.approx([0.9, 0.1, 0.05])
    with pytest.raises(ValueError):
        heads.build_fusion_features(lt[:6], la, 0, 0, 0)


def test_fuse_poe():
    p_text = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.05, 0.05])
    p9 = np.zeros(9)
    p9[labels.SV_EMO_TOKENS.index("ANGRY")] = 0.6
    p9[labels.SV_EMO_TOKENS.index("EMO_UNKNOWN")] = 0.4
    p = heads.fuse_poe(p_text, p9)
    assert p.shape == (7,) and p.sum() == pytest.approx(1.0, abs=1e-9)
    assert labels.EMOTIONS[p.argmax()] == "anger"
    assert heads.fuse_poe(p_text, p9, w=0.0) == pytest.approx(p_text)
    assert heads.fuse_poe(p_text, np.ones(7) / 7).argmax() == 4


def test_text_context_features_layout():
    k = 2
    u, p1, p2 = (np.full(768, v, dtype=np.float32) for v in (1.0, 2.0, 3.0))
    x = heads.TextContextHead.features(u, [p2, p1], [False, True], turn_index=5, k=k)
    assert x.shape == (768 * 3 + 4,)
    assert x[0] == 1.0 and x[768] == 2.0 and x[2 * 768] == 3.0  # t, t-1, t-2
    assert list(x[-4:]) == pytest.approx([1.0, 0.0, 0.25, 0.0])  # same1 (t-1), same2 (t-2), turn/20, is_first
    x0 = heads.TextContextHead.features(u, [], [], 0, k)
    assert not x0[768:3 * 768].any() and list(x0[-4:]) == [0.0, 0.0, 0.0, 1.0]
    assert heads.TextContextHead.features(u, [p1], [True], 1, 0).shape == (772,)


def test_text_matrix_windows():
    def utt(d, i, spk):
        return Utt("dev", d, i, spk, "t", "neutral", "neutral", 0.0, 1.0)

    utts = [utt(0, 0, "A"), utt(0, 1, "B"), utt(0, 3, "A"), utt(1, 0, "C")]
    emb = np.arange(1, 5, dtype=np.float32)[:, None] * np.ones((4, 768), dtype=np.float32)
    X = text_matrix(emb, utts, 2)
    assert X.shape == (4, 2308)
    assert X[2, 768] == 2.0 and X[2, 1536] == 1.0 and list(X[2, -4:]) == [0.0, 1.0, 0.1, 0.0]
    assert not X[3, 768:].any() or list(X[3, -4:]) == [0.0, 0.0, 0.0, 1.0]
    R = text_matrix(emb, utts, 2, same_speaker_only=True)
    assert R[2, 768] == 1.0 and not R[2, 1536:2304].any() and list(R[2, -4:]) == pytest.approx([1.0, 0.0, 0.05, 0.0])


def test_text_matrix_emb_prev():
    """early commit: the current slot reads emb (the prefix), the context slot reads emb_prev (the full earlier utterance)."""
    utts = [Utt("test", 0, 0, "A", "t", "neutral", "neutral", 0.0, 1.0), Utt("test", 0, 1, "B", "t", "neutral", "neutral", 1.0, 2.0)]
    emb = np.array([1.0, 2.0], dtype=np.float32)[:, None] * np.ones((2, 768), dtype=np.float32)
    X = text_matrix(emb, utts, 1, emb_prev=emb + 10)
    assert (X[1, :768] == 2.0).all() and (X[1, 768:1536] == 11.0).all()
    assert (X[0, :768] == 1.0).all() and not X[0, 768:1536].any()
    assert np.array_equal(text_matrix(emb, utts, 1, emb_prev=emb), text_matrix(emb, utts, 1))


def test_head_input_checks(tmp_path):
    p = heads.head_params(heads.lr_params(np.zeros((7, 24)), np.zeros(7)), np.zeros(24), np.ones(24), labels.EMOTIONS)
    assert heads.AudioCalibrator(p).input_dim == 24 and not heads.AudioCalibrator(p).uses_encoder_states
    with pytest.raises(ValueError):
        heads.Fusion(p)
    p19 = heads.head_params(heads.lr_params(np.zeros((3, 19)), np.zeros(3)), np.zeros(19), np.ones(19), labels.SENTIMENTS)
    assert heads.SentimentHead(p19).variant == "default"
    ptxt = heads.head_params(heads.lr_params(np.zeros((3, 772)), np.zeros(3)), np.zeros(772), np.ones(772), labels.SENTIMENTS,
                             variant="text_only", k=0)
    assert heads.SentimentHead(ptxt).input_dim == 772
    assert heads.AudioCalibrator.features(np.ones(9) / 9, np.zeros(11), np.zeros(4)).shape == (24,)
    assert heads.AudioCalibrator.features(np.ones(9) / 9, np.zeros(11), np.zeros(4), np.zeros(1024)).shape == (1048,)


def test_lr_grid_and_temperature(problem):
    Xtr, ytr, Xdev, ydev, Xte, yte = problem
    h, choice, scores = lr_grid(Xtr, ytr, Xdev, ydev, labels.EMOTIONS, [0.1, 1.0], ["none", "balanced"],
                                lambda s: np.ones(7), LR_CFG, 0)
    assert len(scores) == 4 and choice["C"] in (0.1, 1.0)
    assert (h.predict_log_proba_batch(Xte).argmax(1) == yte).mean() > 0.9


def test_fit_temperature_recovers_generating_temperature():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((20000, 7)) * 3.0
    for t_true in (0.5, 2.0):
        p = heads.softmax(z / t_true)
        y = (p.cumsum(1) < rng.uniform(size=(len(z), 1))).sum(1)
        assert heads.fit_temperature(z, y) == pytest.approx(t_true, rel=0.1)


def test_choose_thresholds():
    rng = np.random.default_rng(0)
    n = 2000
    conf = rng.uniform(0.2, 1.0, n)
    correct = rng.uniform(size=n) < conf  # calibrated by construction
    proba = np.zeros((n, 7))
    proba[:, 0] = conf
    proba[:, 1] = 1 - conf
    y = np.where(correct, 0, 1)
    cfg = {"grid_step": 0.01, "low_conf": {"target_accuracy": 0.5, "fallback": 0.45}, "high_conf": {"target_accuracy": 0.7, "fallback": 0.7},
           "jsd_flag": {}}
    t = choose_thresholds(proba, y, cfg)
    assert t["source"] == "dev" and 0.3 < t["low_conf"] < t["high_conf"] < 0.95


def test_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 7, 500), rng.integers(0, 7, 500)
    cm = metrics.confusion(y, p, 7)
    assert metrics.weighted_f1(cm) == pytest.approx(f1_score(y, p, average="weighted"))
    assert metrics.macro_f1(cm) == pytest.approx(f1_score(y, p, average="macro"))
    assert metrics.accuracy(cm) == pytest.approx(accuracy_score(y, p))
    np.testing.assert_allclose(metrics.f1_per_class(cm), f1_score(y, p, average=None))
    ci = metrics.bootstrap_by_group(y, p, np.repeat(np.arange(50), 10), 7, n_resamples=200, seed=0)
    assert ci["weighted_f1"][0] <= metrics.weighted_f1(cm) <= ci["weighted_f1"][1]
    proba = np.full((4, 7), 0.05)
    proba[:, 0] = 0.7
    assert metrics.ece(proba, np.array([0, 0, 0, 1])) == pytest.approx(0.05, abs=1e-9)
