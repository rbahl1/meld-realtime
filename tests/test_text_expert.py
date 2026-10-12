"""TextExpert over the zoo roberta-base graph: tokenization contract and embeddings."""

from pathlib import Path

import numpy as np
import pytest

from meld_rt.experts.text import BOS_ID, EOS_ID, HIDDEN, TextExpert

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "models" / "roberta-base-11.onnx"
TOK_DIR = ROOT / "assets" / "roberta_tokenizer"


@pytest.fixture(scope="module")
def expert() -> TextExpert:
    if not ONNX.exists():
        pytest.skip("models/roberta-base-11.onnx not on disk (tools/fetch_models.py)")
    return TextExpert(ONNX, TOK_DIR, threads=2, max_tokens=64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_tokenize_wraps_and_truncates(expert):
    assert expert.tokenize("Hello world") == [BOS_ID, 31414, 232, EOS_ID]
    ids = expert.tokenize(" ".join(["word"] * 100))
    assert len(ids) == 64 and ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert expert.tokenize("") == [BOS_ID, EOS_ID]


def test_encode_shape_finite(expert):
    v = expert.encode("You must've had your hands full.")
    assert v.shape == (HIDDEN,) and v.dtype == np.float32
    assert np.isfinite(v).all()
    assert np.abs(v).max() > 0


def test_hidden_states_shape(expert):
    ids = expert.tokenize("Hello world")
    h = expert.hidden_states(ids)
    assert h.shape == (len(ids), HIDDEN)


def test_different_sentences_differ(expert):
    a = expert.encode("I am so happy to see you!")
    b = expert.encode("Get out of my house.")
    assert cosine(a, b) < 0.99


def test_same_sentence_identical(expert):
    s = "also I was the point person on my company's transition from the KL-5 to GR-6 system."
    assert np.array_equal(expert.encode(s), expert.encode(s))


def test_empty_text_is_finite(expert):
    v = expert.encode("")
    assert v.shape == (HIDDEN,) and np.isfinite(v).all()


def test_mean_excludes_special_positions(expert):
    ids = expert.tokenize("Hello world")
    h = expert.hidden_states(ids)
    want = h[1:-1].mean(axis=0)
    assert np.allclose(expert.encode("Hello world"), want, atol=1e-6)
