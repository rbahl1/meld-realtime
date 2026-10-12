"""pause extraction and endpoint splits (tools/pause_stats.py): a synthetic
probability track, a speech-pause-speech signal built from en.wav slices, the
448 ms pause inside en.wav itself, and the json and table of a full run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import pause_stats as ps  # noqa: E402

MODEL = ROOT / "models/silero_vad.onnx"
WAV = ROOT / "models/sensevoice/test_wavs/en.wav"
SR = 16000
# continuous speech inside en.wav, as in tests/test_vad.py
SPEECH_SLICE_S = (3.84, 6.46)
LEAD_S, GAP_S, TAIL_S = 1.0, 0.6, 1.5
FRAME_S = ps.FRAME_S

needs_models = pytest.mark.skipif(not (MODEL.exists() and WAV.exists()), reason="models not fetched")


def test_extract_pauses_on_a_probability_track():
    # a short burst that does not open a segment, then speech, two pauses, speech, trailing silence
    p = [0.9] * 5 + [0.0] * 3 + [0.9] * 10 + [0.1] * 4 + [0.4] + [0.2] * 3 + [0.9] * 8 + [0.0] * 10
    # the segment opens on frame 15 (8th frame of the run that starts at 8); the 0.4 frame is not silence
    assert ps.extract_pauses(p, start_frames=8) == [(18, 4), (23, 3)]
    assert ps.extract_pauses([0.9] * 7 + [0.0] * 5, start_frames=8) == []        # never opens
    assert ps.extract_pauses([0.9] * 8 + [0.0] * 5, start_frames=8) == []        # only trailing silence
    assert ps.extract_pauses([0.9] * 8 + [0.34, 0.35], start_frames=8) == [(8, 1)]  # 0.35 is not below 0.35


def test_dist_uses_linear_percentiles():
    d = ps.dist([32.0, 64.0, 96.0, 448.0])
    assert d == {"n": 4, "p50": 80.0, "p90": 342.4, "p95": 395.2, "max": 448.0}
    assert ps.dist([]) == {"n": 0, "p50": None, "p90": None, "p95": None, "max": None}


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    pcm, sr = sf.read(str(WAV), dtype="int16")
    assert sr == SR
    return pcm[int(SPEECH_SLICE_S[0] * SR):int(SPEECH_SLICE_S[1] * SR)]


@pytest.fixture(scope="module")
def two_phrases(speech) -> np.ndarray:
    z = lambda s: np.zeros(int(s * SR), np.int16)  # noqa: E731
    return np.concatenate([z(LEAD_S), speech, z(GAP_S), speech])


@needs_models
def test_one_pause_of_the_inserted_length(speech, two_phrases):
    probe = ps.open_segmenter(MODEL)
    frames = ps.clip_frames(two_phrases, TAIL_S)
    pauses = ps.extract_pauses(ps.frame_probs(probe, frames), probe.start_frames, probe.threshold)
    assert len(pauses) == 1, pauses
    start, n = pauses[0]
    gap_start = LEAD_S + len(speech) / SR
    assert abs(start * FRAME_S - gap_start) <= 2 * FRAME_S, (start * FRAME_S, gap_start)
    assert abs(n * FRAME_S - GAP_S) <= 2 * FRAME_S, (n * FRAME_S, GAP_S)
    # a second pass gives the same track: frame_probs resets the rnn
    again = ps.extract_pauses(ps.frame_probs(probe, frames), probe.start_frames, probe.threshold)
    assert again == pauses


@needs_models
def test_split_follows_the_pause_and_the_tail(two_phrases):
    probe = ps.open_segmenter(MODEL)
    frames = ps.clip_frames(two_phrases, TAIL_S)
    (_, n), = ps.extract_pauses(ps.frame_probs(probe, frames), probe.start_frames, probe.threshold)
    for ms in ps.SETTINGS_MS:
        vad = ps.service_segmenter(MODEL, ms)
        got = ps.count_segments(vad, frames)
        split = n >= vad.silence_frames
        assert got == {"segments": 2 if split else 1, "forced": 0, "after_pause": 1 if split else 0}, (ms, n, got)
    assert ps.count_segments(ps.service_segmenter(MODEL, 300), frames)["segments"] == 2
    assert ps.count_segments(ps.service_segmenter(MODEL, 800), frames)["segments"] == 1


@needs_models
def test_en_wav_pause_splits_at_400_and_not_at_500():
    res = ps.analyze([WAV], MODEL, settings=(400, 500))
    clip, = res["clips"]
    assert clip["pauses_ms"] == [448.0] and clip["pause_starts_s"] == [3.392]
    by = {r["endpoint_silence_ms"]: r for r in res["settings"]}
    assert by[400]["added_latency_ms"] == 416.0 and by[400]["clips_split_by_pause"] == 1
    assert by[500]["added_latency_ms"] == 512.0 and by[500]["clips_split"] == 0


@needs_models
def test_main_writes_json_and_table(tmp_path, two_phrases, capsys):
    wav = tmp_path / "two_phrases.wav"
    sf.write(str(wav), two_phrases, SR, subtype="PCM_16")
    silence = tmp_path / "silence.wav"
    sf.write(str(silence), np.zeros(SR, np.int16), SR, subtype="PCM_16")
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"config": {"endpoint_silence_ms": 400}, "results": {"by_bucket": {"le5s": {
        "service_total_ms": {"n": 3, "p50": 100.5, "p95": 200.25}}}}}))
    out = tmp_path / "pause_stats_t.json"
    # the glob matches both files; the second pattern finds nothing new, so one group
    assert ps.main(["--wavs", str(tmp_path / "*.wav"), str(wav), "--label", "t", "--bench", str(bench),
                    "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["label"] == "t" and d["n_clips"] == 2 and d["n_with_speech"] == 1 and d["groups"] == {}
    assert d["pauses_ms"]["n"] == 1 and d["clip_max_pause_ms"]["n"] == 1
    rows = {r["endpoint_silence_ms"]: r for r in d["settings"]}
    assert [rows[ms]["added_latency_ms"] for ms in ps.SETTINGS_MS] == [320.0, 416.0, 512.0, 608.0, 800.0]
    assert rows[300]["pct_split"] == 100.0 and rows[800]["pct_split"] == 0.0
    assert rows[300]["wait_p50_ms"] == 420.5 and rows[300]["wait_p95_ms"] == 520.25
    table = capsys.readouterr().out
    assert "| 400 | 13 | 416.0 |" in table and "2 clips, 1 with speech, 1 internal pauses" in table


@needs_models
def test_groups_follow_the_wavs_patterns(tmp_path, two_phrases):
    wav = tmp_path / "two_phrases.wav"
    sf.write(str(wav), two_phrases, SR, subtype="PCM_16")
    silence = tmp_path / "silence.wav"
    sf.write(str(silence), np.zeros(SR, np.int16), SR, subtype="PCM_16")
    groups = ps.collect_wavs([f"{wav},{silence}", str(tmp_path / "*.wav")])
    assert groups == {str(wav): [wav], str(silence): [silence]}
    res = ps.analyze(groups, MODEL, settings=(300, 800))
    assert [c["group"] for c in res["clips"]] == [str(wav), str(silence)]
    g = res["groups"]
    assert g[str(wav)]["n_with_speech"] == 1 and g[str(silence)]["n_with_speech"] == 0
    assert [r["clips_split"] for r in g[str(wav)]["settings"]] == [1, 0]
    assert g[str(silence)]["settings"][0]["pct_split"] is None
    assert "| group |" in ps.markdown(res, "g")
