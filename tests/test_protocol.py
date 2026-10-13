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


# -- three-input extension (spec 3.1) ---------------------------------------

def test_extension_topics_are_kept_apart_from_the_core_sets():
    assert P.EXT_IN_TOPICS == ("video",) and set(P.EXT_OUT_TOPICS) == {"state.commit", "response.delta"}
    assert set(P.ALL_IN_TOPICS) == set(P.IN_TOPICS) | {"video"}
    assert set(P.ALL_OUT_TOPICS) == set(P.OUT_TOPICS) | {"state.commit", "response.delta"}
    for topic in P.EXT_OUT_TOPICS:
        t, d = P.parse_out(P.out_msg(topic, {"stream_id": "s1", "utt_id": 2}))
        assert t == topic and d["utt_id"] == 2


def test_video_round_trip_with_a_real_jpeg():
    pytest.importorskip("cv2")
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:, :32] = (0, 0, 255)
    msg = P.frame_msg("s1", 3, 987654321, 0.3, frame)
    assert msg[0] == b"video" and msg[2][:2] == P.JPEG_SOI
    topic, hdr, payload = P.parse_in(msg)
    assert topic == "video"
    assert hdr == {"stream_id": "s1", "seq": 3, "t_capture_ns": 987654321, "t_media": 0.3, "w": 64, "h": 48, "fmt": "jpeg"}
    back = P.decode_jpeg(payload)
    assert back.shape == frame.shape and abs(int(back[10, 5, 2]) - 255) < 8 and int(back[10, 50, 2]) < 8
    # an integer media time is accepted and comes back as float
    _, hdr, _ = P.parse_in(P.video_msg("s1", 0, 1, 2, 64, 48, payload))
    assert hdr["t_media"] == 2.0 and isinstance(hdr["t_media"], float)
    with pytest.raises(ValueError):
        P.decode_jpeg(b"\xff\xd8 not really a jpeg")


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


@pytest.mark.parametrize("hdr,payload", [
    ({"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "t_media": 0.1, "w": 64, "h": 48}, b"not a jpeg"),
    ({"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "t_media": 0.1, "w": 64, "h": 48}, b""),
    ({"stream_id": "s1", "seq": "1", "t_capture_ns": 1, "t_media": 0.1, "w": 64, "h": 48}, JPEG),
    ({"stream_id": "s1", "seq": True, "t_capture_ns": 1, "t_media": 0.1, "w": 64, "h": 48}, JPEG),
    ({"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "w": 64, "h": 48}, JPEG),
    ({"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "t_media": -0.1, "w": 64, "h": 48}, JPEG),
    ({"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "t_media": "0.1", "w": 64, "h": 48}, JPEG),
    ({"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "t_media": 0.1, "w": 0, "h": 48}, JPEG),
    ({"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "t_media": 0.1, "w": 64, "h": 48, "fmt": "png"}, JPEG),
    ({"seq": 1, "t_capture_ns": 1, "t_media": 0.1, "w": 64, "h": 48}, JPEG),
])
def test_bad_video_raises_value_error(hdr, payload):
    with pytest.raises(ValueError):
        P.parse_in([b"video", json.dumps(hdr).encode(), payload])


def test_nan_media_time_is_rejected():
    hdr = b'{"stream_id": "s1", "seq": 1, "t_capture_ns": 1, "t_media": NaN, "w": 64, "h": 48}'
    with pytest.raises(ValueError):
        P.parse_in([b"video", hdr, JPEG])
