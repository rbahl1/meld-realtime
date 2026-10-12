"""vad segmenter on synthetic files built from en.wav: one start and one end at
the expected media times, the 15 s cap, speech_pcm growing during speech."""
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from meld_rt.experts.vad import FRAME, SR, VadEvent, VadSegmenter

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/silero_vad.onnx"
WAV = ROOT / "models/sensevoice/test_wavs/en.wav"
# continuous speech inside en.wav (silero p > 0.35 throughout; the clip has a 0.45 s pause at 3.39 s)
SPEECH_SLICE_S = (3.84, 6.46)
LEAD_S, TAIL_S = 1.0, 1.5
TOL_S = 0.1

pytestmark = pytest.mark.skipif(not (MODEL.exists() and WAV.exists()), reason="models not fetched")


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    pcm, sr = sf.read(str(WAV), dtype="int16")
    assert sr == SR
    return pcm[int(SPEECH_SLICE_S[0] * SR):int(SPEECH_SLICE_S[1] * SR)]


def push_all(vad: VadSegmenter, sig: np.ndarray) -> tuple[list[VadEvent], list[int]]:
    events, grow = [], []
    for i in range(len(sig) // FRAME):
        events += vad.push(sig[i * FRAME:(i + 1) * FRAME])
        grow.append(len(vad.speech_pcm()))
    return events, grow


def test_one_segment_at_expected_times(speech):
    sig = np.concatenate([np.zeros(int(LEAD_S * SR), np.int16), speech, np.zeros(int(TAIL_S * SR), np.int16)])
    vad = VadSegmenter(MODEL, threads=1)
    events, grow = push_all(vad, sig)
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"], kinds
    start, end = events
    onset = LEAD_S
    offset = LEAD_S + len(speech) / SR
    preroll_s = vad.preroll_frames * FRAME / SR
    assert abs(start.t_media - (onset - preroll_s)) <= TOL_S, (start.t_media, onset - preroll_s)
    assert abs(end.t_media - offset) <= TOL_S, (end.t_media, offset)
    assert start.pcm is None and not start.forced and not end.forced
    assert end.pcm.dtype == np.float32
    # pcm spans the segment start to the endpoint detection (speech + preroll + silence tail)
    tail_s = vad.silence_frames * FRAME / SR
    expected_len = (offset - start.t_media + tail_s) * SR
    assert abs(len(end.pcm) - expected_len) <= 2 * FRAME, (len(end.pcm), expected_len)
    assert np.abs(end.pcm).max() <= 1.0
    # media clock is cumulative samples
    assert abs(vad.t_media - (len(sig) // FRAME) * FRAME / SR) < 1e-9
    # speech_pcm grows frame by frame while the segment is open and is empty outside it
    active = [g for g in grow if g > 0]
    assert len(active) > 10
    assert all(b - a == FRAME for a, b in pairwise(active))
    assert max(grow) == len(end.pcm) - FRAME  # the last frame is added inside push before the end fires
    assert grow[-1] == 0


def test_cap_forces_endpoint_on_20s_file(speech):
    reps = int(np.ceil(20 * SR / len(speech)))
    sig = np.concatenate([np.tile(speech, reps)[:20 * SR], np.zeros(SR, np.int16)])
    vad = VadSegmenter(MODEL, max_segment_s=15.0, threads=1)
    events, _ = push_all(vad, sig)
    kinds = [e.kind for e in events]
    assert kinds[:3] == ["speech_start", "speech_end", "speech_start"], kinds
    first_end = events[1]
    assert first_end.forced is True
    assert abs(first_end.t_media - (events[0].t_media + 15.0)) <= FRAME / SR
    assert abs(len(first_end.pcm) - 15 * SR) <= FRAME
    # the next segment opens right where the forced one closed, so no audio is lost
    assert abs(events[2].t_media - first_end.t_media) <= FRAME / SR
    assert kinds[-1] == "speech_end" and events[-1].forced is False
    assert abs(events[-1].t_media - 20.0) <= TOL_S


def test_silence_only_emits_nothing_and_reset_clears():
    vad = VadSegmenter(MODEL, threads=1)
    events, grow = push_all(vad, np.zeros(2 * SR, np.int16))
    assert events == [] and max(grow) == 0
    assert vad.t_media == pytest.approx((2 * SR // FRAME) * FRAME / SR)
    vad.reset(keep_clock=False)
    assert vad.t_media == 0.0 and not vad.active
