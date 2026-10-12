"""sensevoice expert: transcript parity with sherpa-onnx on en.wav, posterior
rows, the edited graph with encoder_out, and one timed run."""
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from meld_rt import labels
from meld_rt.experts.sensevoice import (
    EMO_TOKENS,
    EVENT_TOKENS,
    N_PREFIX_ROWS,
    SenseVoiceExpert,
    encoder_model_path,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/sensevoice/model.int8.onnx"
TOKENS = ROOT / "models/sensevoice/tokens.txt"
WAV = ROOT / "models/sensevoice/test_wavs/en.wav"
THREADS = 2

pytestmark = pytest.mark.skipif(not (MODEL.exists() and TOKENS.exists() and WAV.exists()), reason="models not fetched")


def norm(s: str) -> str:
    return " ".join(s.split())


@pytest.fixture(scope="module")
def pcm() -> np.ndarray:
    x, sr = sf.read(str(WAV), dtype="float32")
    assert sr == 16000
    return x


@pytest.fixture(scope="module")
def expert() -> SenseVoiceExpert:
    return SenseVoiceExpert(MODEL, TOKENS, threads=THREADS)


@pytest.fixture(scope="module")
def result(expert, pcm):
    return expert.run(pcm)


@pytest.fixture(scope="module")
def edited_model(tmp_path_factory) -> Path:
    path = encoder_model_path(MODEL)
    if path.exists():
        return path
    spec = importlib.util.spec_from_file_location("edit_sensevoice_graph", ROOT / "tools/edit_sensevoice_graph.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("sv") / "model_enc.int8.onnx"
    mod.edit(MODEL, out)
    return out


def test_parity_with_sherpa_onnx(result):
    sherpa_onnx = pytest.importorskip("sherpa_onnx")
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=str(MODEL), tokens=str(TOKENS), num_threads=THREADS,
                                                         language="en", use_itn=False)
    x, sr = sf.read(str(WAV), dtype="float32")
    s = rec.create_stream()
    s.accept_waveform(sr, x)
    rec.decode_stream(s)
    ref = s.result
    assert norm(result.text) == norm(ref.text), (result.text, ref.text)
    assert ref.emotion == f"<|{EMO_TOKENS[int(result.emo_probs.argmax())]}|>"
    assert ref.event == f"<|{EVENT_TOKENS[int(result.event_probs.argmax())]}|>"
    assert ref.lang == f"<|{result.lang}|>"


def test_rows_and_shapes(result, expert, pcm):
    assert EMO_TOKENS[int(result.emo_probs.argmax())] == "NEUTRAL"
    assert result.emo_probs.shape == (9,) and abs(float(result.emo_probs.sum()) - 1.0) < 1e-4
    assert result.event_probs.shape == (11,) and abs(float(result.event_probs.sum()) - 1.0) < 1e-4
    assert EVENT_TOKENS[int(result.event_probs.argmax())] == "Speech"
    assert result.lang == "en"
    assert result.text and 0.0 < result.asr_conf <= 1.0
    assert result.tokens_per_s > 0
    assert result.enc_pooled is None
    assert result.n_frames == expert.frontend(pcm).shape[0]
    assert set(result.timing_ms) >= {"frontend", "model"}
    p7 = labels.sv_probs_to_meld7(result.emo_probs)
    assert abs(sum(p7) - 1.0) < 1e-6 and labels.EMOTIONS[int(np.argmax(p7))] == "neutral"


def test_short_and_silent_input_do_not_crash(expert):
    r = expert.run(np.zeros(800, np.float32))
    assert r.n_frames >= 1 and r.emo_probs.shape == (9,)
    r = expert.run(np.zeros(16000, np.float32))
    assert r.n_frames >= 1 and 0.0 <= r.asr_conf <= 1.0


def test_edited_graph_matches_and_pools(expert, pcm, edited_model):
    enc_expert = SenseVoiceExpert(edited_model, TOKENS, threads=THREADS, with_encoder_states=True)
    feats = expert.frontend(pcm)
    logits = expert.forward(feats)[0]
    logits_e, enc = enc_expert.forward(feats)
    assert logits.shape == logits_e.shape
    assert float(np.abs(logits - logits_e).max()) <= 1e-4
    assert enc.shape == (1, feats.shape[0] + N_PREFIX_ROWS, 512)
    r = enc_expert.run(pcm)
    assert r.enc_pooled is not None and r.enc_pooled.shape == (1024,) and r.enc_pooled.dtype == np.float32
    speech = enc[0, N_PREFIX_ROWS:]
    assert np.allclose(r.enc_pooled[:512], speech.mean(0), atol=1e-5)
    assert np.allclose(r.enc_pooled[512:], speech.max(0), atol=1e-5)
    assert norm(r.text) == norm(expert.run(pcm).text)
    # the base path resolves to the edited sibling when it exists
    if encoder_model_path(MODEL).exists():
        auto = SenseVoiceExpert(MODEL, TOKENS, threads=THREADS, with_encoder_states=True)
        assert auto.model_path == encoder_model_path(MODEL)


def test_latency_one_run(expert, pcm, capsys):
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        r = expert.run(pcm)
        times.append((time.perf_counter() - t0) * 1000)
    with capsys.disabled():
        print(f"\nsensevoice {len(pcm) / 16000:.2f} s clip, {THREADS} threads: median {statistics.median(times):.0f} ms "
              f"(frontend {r.timing_ms['frontend']:.0f} ms, model {r.timing_ms['model']:.0f} ms)")
    assert statistics.median(times) < 5000
