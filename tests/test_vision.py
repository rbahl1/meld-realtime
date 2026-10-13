"""vision expert: class order, alignment, tracking and the opencv / onnxruntime parity of the fer graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meld_rt import labels
from meld_rt.experts import vision as V

ROOT = Path(__file__).resolve().parents[1]
YUNET = ROOT / "models/vision/face_detection_yunet_2023mar.onnx"
FER = ROOT / "models/vision/facial_expression_recognition_mobilefacenet_2022july.onnx"
have_models = pytest.mark.skipif(not (YUNET.exists() and FER.exists()), reason="vision models not fetched")
cv2 = pytest.importorskip("cv2")


def test_fer_to_meld_order():
    # one-hot fer logits land on the matching MELD class
    for i, name in enumerate(V.FER_CLASSES):
        z = np.full(7, -20.0)
        z[i] = 20.0
        p = V.posterior_meld(z)
        assert labels.EMOTIONS[int(np.argmax(p))] == V.FER_TO_MELD[name]
    assert sorted(V.MELD_FROM_FER) == list(range(7))


def test_posterior_meld_batch_rows_sum_to_one():
    p = V.posterior_meld(np.random.default_rng(0).normal(size=(5, 7)))
    assert p.shape == (5, 7)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_similarity_recovers_a_known_transform():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 200, size=(5, 2))
    s, a, t = 0.7, np.deg2rad(12.0), np.array([5.0, -3.0])
    R = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = src @ R.T + t
    m = V.similarity_from_points(src, dst)
    assert np.allclose(m[:, :2], R, atol=1e-9)
    assert np.allclose(m[:, 2], t, atol=1e-9)


def test_similarity_template_to_itself_is_identity():
    m = V.similarity_from_points(V.TEMPLATE, V.TEMPLATE)
    assert np.allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-9)


def test_similarity_never_reflects():
    src = V.TEMPLATE.copy()
    dst = src.copy()
    dst[:, 0] = 112 - dst[:, 0]   # mirror image: the best similarity must keep det > 0
    m = V.similarity_from_points(src, dst)
    assert np.linalg.det(m[:, :2]) > 0


def test_iou():
    assert V.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert V.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert V.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


@have_models
def test_fer_opencv_matches_onnxruntime():
    import onnx
    import onnxruntime as ort

    ve = V.VisionExpert(YUNET, FER, threads=1)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ve.graph_bytes, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(2)
    for _ in range(3):
        face = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
        logits, emb = ve.recognize(face)
        x = ((face[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        ref = sess.run([V.FER_LOGITS, V.FER_EMBED], {V.FER_INPUT: np.ascontiguousarray(x)})
        assert np.allclose(logits, ref[0].reshape(-1), atol=1e-4)
        assert np.allclose(emb, ref[1].reshape(-1), atol=1e-4)
        assert emb.shape == (V.EMBED_DIM,)
    assert onnx.load_from_string(ve.graph_bytes).graph.output[-1].name == V.FER_EMBED


@have_models
def test_blank_frame_has_no_faces_and_tracks_reset():
    ve = V.VisionExpert(YUNET, FER, threads=1)
    assert ve.process(np.zeros((360, 640, 3), np.uint8)) == []
    ve.reset()
    assert ve._next_track == 0


@have_models
def test_track_ids_persist_and_motion_is_measured(monkeypatch):
    """two frames with the same detection keep one track id; the second frame has a motion value."""
    ve = V.VisionExpert(YUNET, FER, threads=1)
    lm = V.TEMPLATE * 2.0 + 100.0
    row = np.concatenate([[100, 100, 224, 224], lm.reshape(-1), [0.9]])
    monkeypatch.setattr(ve, "detect", lambda frame: row[None, :].copy())
    rng = np.random.default_rng(3)
    f1 = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    f2 = f1.copy()
    f2[300:340, 160:270] = 255 - f2[300:340, 160:270]   # changes the mouth box of the aligned face
    a = ve.process(f1)
    b = ve.process(f2)
    assert len(a) == len(b) == 1
    assert a[0].track == b[0].track == 0
    assert np.isnan(a[0].mouth_motion)
    assert b[0].mouth_motion > b[0].face_motion > 0.0
