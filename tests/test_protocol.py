"""wire format: round trips, validation of bad input, pcm helpers, json of
numpy values. no sockets."""

import json

import numpy as np
import pytest

from meld_rt import protocol as P
from meld_rt.state import State


def test_constants():
    assert P.IN_ADDR == "tcp://127.0.0.1:5555" and P.OUT_ADDR == "tcp://127.0.0.1:5556"
    assert P.FRAME_SAMPLES == 512 and P.SR == 16000 and P.FRAME_BYTES == 1024
    assert P.FRAME_S == pytest.approx(0.032)
    assert set(P.IN_TOPICS) == {"audio", "text", "ctl"}
    assert set(P.OUT_TOPICS) == {"state.provisional", "state.final", "response", "health", "error"}


def test_audio_round_trip():
    pcm = (np.arange(P.FRAME_SAMPLES, dtype=np.int16) - 256)
    frames = P.audio_msg("s1", 7, 123456789, pcm.tobytes())
    assert [type(f) for f in frames] == [bytes, bytes, bytes] and frames[0] == b"audio"
    topic, hdr, payload = P.parse_in(frames)
    assert topic == "audio"
    assert hdr == {"stream_id": "s1", "seq": 7, "t_capture_ns": 123456789, "sr": 16000, "n_samples": 512, "fmt": "pcm16"}
    assert len(payload) == P.FRAME_BYTES
    np.testing.assert_array_equal(P.pcm16_from_bytes(payload), pcm)
    f32 = P.pcm16_to_f32(payload)
    assert f32.dtype == np.float32 and abs(f32[0] - (-256 / 32768)) < 1e-9


def test_text_and_ctl_round_trip():
    topic, hdr, payload = P.parse_in(P.text_msg("s1", 3, 42, "well that went great"))
    assert topic == "text" and payload == b""
    assert hdr["text"] == "well that went great" and hdr["speaker"] == "user" and hdr["seq"] == 3
    for cmd in P.CTL_CMDS:
        topic, hdr, _ = P.parse_in(P.ctl_msg("s1", cmd))
        assert topic == "ctl" and hdr == {"stream_id": "s1", "cmd": cmd}
    with pytest.raises(ValueError):
        P.ctl_msg("s1", "pause")


@pytest.mark.parametrize("frames", [
    [b"video", b"{}", b""],
    [b"audio"],
    [b"audio", b"not json", b""],
    [b"audio", b'{"stream_id": ""}', b""],
    [b"audio", b'{"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "n_samples": 512}', b"\x00" * 10],
    [b"audio", b'{"stream_id": "s1", "seq": "1", "t_capture_ns": 1, "n_samples": 1}', b"\x00\x00"],
    [b"audio", b'{"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "n_samples": 1, "fmt": "f32"}', b"\x00\x00"],
    [b"audio", b'{"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "n_samples": 1, "sr": 8000}', b"\x00\x00"],
    [b"text", b'{"stream_id": "s1"}', b""],
    [b"ctl", b'{"stream_id": "s1", "cmd": "pause"}', b""],
    [b"ctl", b'[1, 2]', b""],
])
def test_bad_input_raises_value_error(frames):
    with pytest.raises(ValueError):
        P.parse_in(frames)


def test_out_round_trip_with_numpy_values():
    payload = {"stream_id": "s1", "conf": np.float32(0.5), "vec": np.arange(3), "n": np.int64(4)}
    frames = P.out_msg("health", payload)
    assert frames[0] == b"health"
    topic, d = P.parse_out(frames)
    assert topic == "health" and d == {"stream_id": "s1", "conf": 0.5, "vec": [0, 1, 2], "n": 4}
    with pytest.raises(ValueError):
        P.out_msg("state", {})
    with pytest.raises(ValueError):
        P.parse_out([b"health", b"[]"])
    with pytest.raises(ValueError):
        P.parse_out([b"nope", b"{}"])


def test_state_travels_as_out_message():
    s = State.empty("s1", 4, "final")
    s.asr.text = "well that went great"
    topic, d = P.parse_out(P.out_msg("state.final", s.to_dict()))
    back = State.from_dict(d)
    assert topic == "state.final" and back.asr.text == s.asr.text and back.utt_id == 4
    assert json.loads(s.to_json()) == d


def test_pcm16_frames_pad_and_convert():
    x = np.random.default_rng(0).uniform(-1, 1, 1000).astype(np.float32)
    frames = P.pcm16_frames(x)
    assert len(frames) == 2 and all(f.shape == (512,) and f.dtype == np.int16 for f in frames)
    assert np.all(frames[1][1000 - 512:] == 0)
    assert abs(int(frames[0][0]) - round(float(x[0]) * 32767)) <= 1
    assert P.pcm16_frames(np.zeros(0, np.int16)) == []
    stereo = np.stack([x, -x], axis=1)
    assert np.abs(P.to_pcm16(stereo)).max() == 0
    assert P.to_pcm16(np.array([2.0, -2.0], np.float32)).tolist() == [32767, -32767]


def test_replay_reads_pcm16_wav_exactly(tmp_path):
    import soundfile as sf
    from meld_rt.clients.replay import load_wav

    x = (np.random.default_rng(0).uniform(-1, 1, 1600) * 32767).astype(np.int16)
    x[:3] = [32767, -32768, 1]
    p = tmp_path / "a.wav"
    sf.write(str(p), x, 16000, subtype="PCM_16")
    assert np.array_equal(load_wav(p), x)
