"""numpy forward passes for the trained heads (design 1.5, 1.6, 6.2 WP4).

scikit-learn fits them in tools/train_heads.py; the service only loads the
.npz files written there. each file holds the standardizer (mean, scale), the
weights (W, b for logistic regression; W1, b1, W2, b2, activation for the mlp),
a temperature, the class names and a few metadata scalars.

feature layouts, all float32:
  TextContextHead   768*(k+1)+4  [u_t, u_t-1, u_t-2 (zeros when absent), same1, same2, turn_index/20, is_first]
  AudioCalibrator   24 or 1048   [log emo_probs(9), event_probs(11), prosody(4) [, enc_pooled(1024)]]
  Fusion            19           [log p_text(7), log p_audio(7), H_text, H_audio, asr_conf, p_laughter, p_unknown]
  SentimentHead     19 (fused) or the TextContextHead layout (variant "text_only")
  VisionCalibrator  16 or 528    [log p_face(7), max p, entropy, presence, log_area, n_tracks/4, asd_corr, score,
                                  mouth*10, present [, mean fer embedding(512)]] (meld_rt.vision_features)
  Fusion3           29           [log p_text(7), log p_audio(7), log p_face(7), H_text, H_audio, H_face, asr_conf,
                                  p_laughter, p_unknown, face_present, presence]
  Sentiment3        29           the Fusion3 layout, 3 classes
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from meld_rt.labels import EMOTIONS, SENTIMENTS, SV_EMO_TOKENS, SV_EVENT_TOKENS, sv_probs_to_meld7

TEXT_DIM = 768
N_FLAGS = 4
TURN_SCALE = 20.0
AUDIO_SLOT_DIM = len(SV_EMO_TOKENS) + len(SV_EVENT_TOKENS) + 4  # 24
ENC_POOLED_DIM = 1024
FUSION_DIM = 2 * len(EMOTIONS) + 5  # 19
VISION_A_DIM = 16
VISION_EMBED_DIM = 512
FUSION3_DIM = 3 * len(EMOTIONS) + 8  # 29
EPS = 1e-8
LAUGHTER_IDX = SV_EVENT_TOKENS.index("Laughter")
UNKNOWN_IDX = SV_EMO_TOKENS.index("EMO_UNKNOWN")


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = z.max(axis=-1, keepdims=True)
    s = np.log(np.exp(z - m).sum(axis=-1, keepdims=True))
    return z - m - s


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def entropy_bits(logp: np.ndarray) -> float:
    """entropy in bits of a distribution given as log-probabilities."""
    p = np.exp(np.asarray(logp, dtype=np.float64))
    return float(-(p * np.log2(np.clip(p, EPS, 1.0))).sum())


def _s(z: dict, name: str, default):
    """scalar metadata from an npz, with a default when the key is absent."""
    if name not in z:
        return default
    v = z[name]
    return v.item() if hasattr(v, "item") and v.shape == () else v


class NpzHead:
    """a standardizer followed by a linear or one-hidden-layer softmax model."""

    INPUT: ClassVar[int | tuple[int, ...] | None] = None  # nominal contract; the file says what it takes
    CLASSES: ClassVar[list[str]] = []

    def __init__(self, params: dict):
        self.kind = str(_s(params, "kind", "lr"))
        self.mean = np.asarray(params["mean"], dtype=np.float64)
        self.scale = np.asarray(params["scale"], dtype=np.float64)
        self.params = params
        self.temperature = float(_s(params, "temperature", 1.0))
        self.classes = [str(c) for c in params["classes"]] if "classes" in params else list(self.CLASSES)
        self.variant = str(_s(params, "variant", "default"))
        self.k = int(_s(params, "k", 0))
        if self.kind == "mlp":
            self.W1 = np.asarray(params["W1"], dtype=np.float64)
            self.b1 = np.asarray(params["b1"], dtype=np.float64)
            self.W2 = np.asarray(params["W2"], dtype=np.float64)
            self.b2 = np.asarray(params["b2"], dtype=np.float64)
            self.activation = str(_s(params, "activation", "relu"))
            if self.activation not in ("relu", "tanh", "identity", "logistic"):
                raise ValueError(f"unsupported activation {self.activation!r}")
            self.input_dim, self.n_classes = int(self.W1.shape[0]), int(self.W2.shape[1])
        elif self.kind == "lr":
            self.W = np.asarray(params["W"], dtype=np.float64)
            self.b = np.asarray(params["b"], dtype=np.float64)
            self.input_dim, self.n_classes = int(self.W.shape[0]), int(self.W.shape[1])
        else:
            raise ValueError(f"unknown head kind {self.kind!r}")
        if self.mean.shape != (self.input_dim,) or self.scale.shape != (self.input_dim,):
            raise ValueError("standardizer shape does not match the weights")
        if len(self.classes) != self.n_classes:
            raise ValueError(f"{self.n_classes} outputs but {len(self.classes)} class names")
        self.check()

    def check(self) -> None:
        """subclasses assert their input contract here."""

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        # written through to params so save() and the live object never disagree
        if not value > 0:
            raise ValueError("temperature must be positive")
        self._temperature = float(value)
        self.params["temperature"] = np.array(float(value))

    @classmethod
    def load(cls, path: Path) -> NpzHead:
        with np.load(Path(path), allow_pickle=False) as z:
            params = {k: z[k] for k in z.files}
        return cls(params)

    def save(self, path: Path) -> None:
        save_head(path, self.params)

    def logits(self, X: np.ndarray) -> np.ndarray:
        """(n, input_dim) -> (n, n_classes) pre-softmax scores, temperature applied."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} inputs, got {X.shape[1]}")
        X = (X - self.mean) / self.scale
        if self.kind == "lr":
            z = X @ self.W + self.b
        else:
            h = X @ self.W1 + self.b1
            if self.activation == "relu":
                h = np.maximum(h, 0.0)
            elif self.activation == "tanh":
                h = np.tanh(h)
            elif self.activation == "logistic":
                h = 1.0 / (1.0 + np.exp(-h))
            z = h @ self.W2 + self.b2
        return z / self.temperature

    def predict_log_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return log_softmax(self.logits(X))

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        """one feature vector -> (n_classes,) log-probabilities."""
        return self.predict_log_proba_batch(np.asarray(x).reshape(1, -1))[0]


class TextContextHead(NpzHead):
    """INPUT = 768*(k+1)+4; k is read from the file."""

    INPUT = None
    CLASSES = EMOTIONS

    @staticmethod
    def input_dim_for(k: int) -> int:
        return TEXT_DIM * (k + 1) + N_FLAGS

    def check(self) -> None:
        if self.input_dim != self.input_dim_for(self.k):
            raise ValueError(f"text head with k={self.k} must take {self.input_dim_for(self.k)} inputs, has {self.input_dim}")

    @staticmethod
    def features(u_t: np.ndarray, prev: list[np.ndarray], same: list[bool], turn_index: int, k: int) -> np.ndarray:
        """one row. prev holds up to k previous embeddings oldest first with their
        same-speaker flags; slot 1 is t-1, slot 2 is t-2; absent slots stay zero."""
        x = np.zeros(TextContextHead.input_dim_for(k), dtype=np.float32)
        x[:TEXT_DIM] = u_t
        flags = np.zeros(N_FLAGS, dtype=np.float32)
        for j in range(min(k, len(prev))):
            x[TEXT_DIM * (j + 1):TEXT_DIM * (j + 2)] = prev[-1 - j]
            if j < 2:
                flags[j] = float(bool(same[-1 - j]))
        flags[2] = turn_index / TURN_SCALE
        flags[3] = float(turn_index == 0)
        x[TEXT_DIM * (k + 1):] = flags
        return x


class AudioCalibrator(NpzHead):
    """INPUT 24 (slots, events, prosody) or 1048 (plus pooled encoder states)."""

    INPUT = (AUDIO_SLOT_DIM, AUDIO_SLOT_DIM + ENC_POOLED_DIM)
    CLASSES = EMOTIONS

    def check(self) -> None:
        if self.input_dim not in self.INPUT:
            raise ValueError(f"audio calibrator takes {self.INPUT} inputs, has {self.input_dim}")

    @property
    def uses_encoder_states(self) -> bool:
        return self.input_dim == AUDIO_SLOT_DIM + ENC_POOLED_DIM

    @staticmethod
    def features(emo_probs9: np.ndarray, event_probs11: np.ndarray, prosody4: np.ndarray,
                 enc_pooled: np.ndarray | None = None) -> np.ndarray:
        parts = [np.log(np.clip(np.asarray(emo_probs9, dtype=np.float64), EPS, 1.0)),
                 np.asarray(event_probs11, dtype=np.float64), np.asarray(prosody4, dtype=np.float64)]
        if enc_pooled is not None:
            parts.append(np.asarray(enc_pooled, dtype=np.float64))
        return np.concatenate(parts).astype(np.float32)


class Fusion(NpzHead):
    INPUT = FUSION_DIM
    CLASSES = EMOTIONS

    def check(self) -> None:
        if self.input_dim != FUSION_DIM:
            raise ValueError(f"fusion takes {FUSION_DIM} inputs, has {self.input_dim}")


class SentimentHead(NpzHead):
    """19 fused features, or the TextContextHead layout when variant == "text_only"
    (written by train_heads.py --text-only as sentiment.npz so sentiment exists before audio does,
    and by --sentiment-text-only as sentiment_text_only.npz for the text-only test row)."""

    INPUT = FUSION_DIM
    CLASSES = SENTIMENTS

    def check(self) -> None:
        if self.variant == "text_only":
            if self.input_dim != TextContextHead.input_dim_for(self.k):
                raise ValueError("text_only sentiment head must take the text context layout")
        elif self.input_dim != FUSION_DIM:
            raise ValueError(f"sentiment head takes {FUSION_DIM} inputs, has {self.input_dim}")


class VisionCalibrator(NpzHead):
    """INPUT 16 (face evidence, features A) or 528 (plus the mean fer embedding, features B).
    the file carries the speaker rule and the variant ("A" or "B") as metadata."""

    INPUT = (VISION_A_DIM, VISION_A_DIM + VISION_EMBED_DIM)
    CLASSES = EMOTIONS

    def check(self) -> None:
        if self.input_dim not in self.INPUT:
            raise ValueError(f"vision calibrator takes {self.INPUT} inputs, has {self.input_dim}")
        if self.n_classes != len(EMOTIONS):
            raise ValueError("vision calibrator must have 7 outputs")

    @property
    def uses_embedding(self) -> bool:
        return self.input_dim == VISION_A_DIM + VISION_EMBED_DIM

    @property
    def rule(self) -> str | None:
        return str(_s(self.params, "rule", "")) or None


class Fusion3(NpzHead):
    """text, audio and face posteriors, 29 inputs, 7 classes."""

    INPUT = FUSION3_DIM
    CLASSES = EMOTIONS

    def check(self) -> None:
        if self.input_dim != FUSION3_DIM or self.n_classes != len(EMOTIONS):
            raise ValueError(f"fusion3 takes {FUSION3_DIM} inputs and gives 7 outputs, has {self.input_dim} and {self.n_classes}")


class Sentiment3(NpzHead):
    """sentiment over the fusion3 layout, 29 inputs, 3 classes."""

    INPUT = FUSION3_DIM
    CLASSES = SENTIMENTS

    def check(self) -> None:
        if self.input_dim != FUSION3_DIM or self.n_classes != len(SENTIMENTS):
            raise ValueError(f"sentiment3 takes {FUSION3_DIM} inputs and gives 3 outputs, has {self.input_dim} and {self.n_classes}")


def build_fusion_features(logp_text: np.ndarray, logp_audio: np.ndarray, asr_conf: float,
                          p_laughter: float, p_unknown: float) -> np.ndarray:
    """(19,) float32; both log-posteriors are 7-way in MELD order."""
    lt = np.asarray(logp_text, dtype=np.float64).ravel()
    la = np.asarray(logp_audio, dtype=np.float64).ravel()
    if lt.size != len(EMOTIONS) or la.size != len(EMOTIONS):
        raise ValueError("fusion needs two 7-way log-posteriors")
    tail = [entropy_bits(lt), entropy_bits(la), float(asr_conf), float(p_laughter), float(p_unknown)]
    return np.concatenate([lt, la, tail]).astype(np.float32)


def audio7(p_audio) -> np.ndarray:
    """9-way sensevoice posterior renormalized over the seven named classes (MELD
    order); a 7-way vector passes through normalized."""
    p = np.clip(np.asarray(p_audio, dtype=np.float64).ravel(), 0.0, None)
    if p.size == 9:
        return np.asarray(sv_probs_to_meld7(p), dtype=np.float64)
    if p.size == 7:
        s = p.sum()
        return p / s if s > 0 else np.full(7, 1.0 / 7)
    raise ValueError(f"audio posterior must have 9 or 7 entries, got {p.size}")


def fuse_poe(p_text: np.ndarray, p_audio9: np.ndarray, w: float = 0.5) -> np.ndarray:
    """product of experts p ∝ p_text * p_audio7^w, the uncalibrated fallback fusion."""
    pt = np.clip(np.asarray(p_text, dtype=np.float64).ravel(), 0.0, None)
    if pt.size != len(EMOTIONS):
        raise ValueError("p_text must be 7-way")
    pa = audio7(p_audio9)
    logp = np.log(np.clip(pt, EPS, None)) + w * np.log(np.clip(pa, EPS, None))
    return softmax(logp)


def fit_temperature(logits: np.ndarray, y: np.ndarray, lo: float = 0.1, hi: float = 10.0, iters: int = 60) -> float:
    """temperature minimizing the nll of softmax(logits / T) on (logits, y) by golden
    section search on log T within [lo, hi]; 1.0 means no change."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)

    def nll(log_t: float) -> float:
        lp = log_softmax(z / np.exp(log_t))
        return float(-lp[np.arange(len(y)), y].mean())

    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return float(np.exp((a + b) / 2))


def mlp_params(coefs: list[np.ndarray], intercepts: list[np.ndarray], activation: str) -> dict:
    """arrays of a fitted sklearn MLPClassifier with one hidden layer."""
    if len(coefs) != 2:
        raise ValueError("expected exactly one hidden layer")
    return {"kind": np.array("mlp"), "activation": np.array(activation),
            "W1": np.asarray(coefs[0], dtype=np.float32), "b1": np.asarray(intercepts[0], dtype=np.float32),
            "W2": np.asarray(coefs[1], dtype=np.float32), "b2": np.asarray(intercepts[1], dtype=np.float32)}


def lr_params(coef: np.ndarray, intercept: np.ndarray) -> dict:
    """arrays of a fitted sklearn multinomial LogisticRegression (coef_ is (n_classes, n_features))."""
    coef = np.asarray(coef)
    if coef.shape[0] < 3:
        raise ValueError("binary logistic regression is not supported; use three or more classes")
    return {"kind": np.array("lr"), "W": coef.T.astype(np.float32), "b": np.asarray(intercept, dtype=np.float32)}


def head_params(model_params: dict, mean: np.ndarray, scale: np.ndarray, classes: list[str],
                temperature: float = 1.0, **meta) -> dict:
    """complete parameter dict for save_head: model arrays, standardizer, classes, temperature, metadata."""
    p = dict(model_params)
    p["mean"] = np.asarray(mean, dtype=np.float32)
    p["scale"] = np.asarray(scale, dtype=np.float32)
    p["classes"] = np.array(list(classes))
    p["temperature"] = np.array(float(temperature))
    for k, v in meta.items():
        p[k] = np.array(v)
    return p


def save_head(path: Path, params: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)
