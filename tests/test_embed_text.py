"""tools/embed_text.py: output format, order, skip-when-finished and resume."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import embed_text  # noqa: E402

ONNX = ROOT / "models" / "roberta-base-11.onnx"
TOK_DIR = ROOT / "assets" / "roberta_tokenizer"
ROWS = [("0_0", "Hello world"), ("0_1", "You must've had your hands full."), ("1_0", "What?!"),
        ("1_1", " ".join(["word"] * 80)), ("2_0", "")]


def test_csv_loader_order_and_keys():
    utts = embed_text.load_utterances_csv("dev", ROOT / "meld")
    assert len(utts) == 1109
    keys = [k for k, _ in utts]
    assert keys[:2] == ["0_0", "0_1"]
    assert len(set(keys)) == len(keys)
    pairs = [tuple(int(x) for x in k.split("_")) for k in keys]
    assert pairs == sorted(pairs)


def test_embed_resume_and_skip(tmp_path, monkeypatch):
    if not ONNX.exists():
        pytest.skip("models/roberta-base-11.onnx not on disk")
    monkeypatch.setattr(embed_text, "load_utterances", lambda split, meld_dir: ROWS)
    monkeypatch.setattr(embed_text, "CHECKPOINT_EVERY", 2)
    out = tmp_path / "text_dev.npy"
    args = ("dev", out, ROOT / "meld", ONNX, TOK_DIR, 1, 64)

    # a partial checkpoint with the first two rows: those rows are kept, the rest computed
    emb0 = np.zeros((len(ROWS), 768), dtype=np.float32)
    emb0[:2] = 7.0
    embed_text.save_partial(out, emb0, 2)
    meta = embed_text.embed_split(*args)
    assert meta["rows_embedded_this_run"] == 3 and meta["n_truncated_this_run"] == 1
    assert not embed_text.partial_path(out).exists()
    arr = np.load(out)
    assert arr.shape == (5, 768) and arr.dtype == np.float32 and np.isfinite(arr).all()
    assert (arr[:2] == 7.0).all() and (arr[2:] != 7.0).any(axis=1).all()
    assert json.loads(embed_text.keys_path(out).read_text()) == [k for k, _ in ROWS]

    # a finished output is skipped untouched
    mtime = out.stat().st_mtime_ns
    embed_text.embed_split(*args)
    assert out.stat().st_mtime_ns == mtime
