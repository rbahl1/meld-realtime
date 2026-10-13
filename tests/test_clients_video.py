"""clients with video (spec 3.3) and the replay bench's --video path: decoding a
synthetic mp4 (ffmpeg test pattern and a sine tone, made in tmp), the media-time
schedule, real-time pacing, the camera sender's media clock, the monitor's
vision, commit and streamed-reply lines, and the bench's video streaming and
analysis. no service and no model here."""

from __future__ import annotations

import io
import itertools
import json
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import zmq

from meld_rt import protocol as P
from meld_rt.clients import connect_push, mic
from meld_rt.clients.mic import CameraSender, frames_lost
from meld_rt.clients.monitor import Monitor, fmt_line
from meld_rt.clients.replay import load_video, media_schedule, replay_video
from meld_rt.clients.replay import main as replay_main
from meld_rt.labels import EMOTIONS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import replay_bench as rb

cv2 = pytest.importorskip("cv2")
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed")


def make_mp4(path: Path, seconds: float, size: str = "320x240") -> Path:
    """test pattern at 25 fps and a 440 hz tone at 44.1 khz stereo, mpeg4 and aac."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", str(seconds), "-ac", "2",
           "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


@pytest.fixture(scope="module")
def mp4s(tmp_path_factory):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("mp4")
    return {"two": make_mp4(d / "dia1_utt0.mp4", 2.0), "short": make_mp4(d / "dia1_utt1.mp4", 0.6)}


class Pull:
    """a pull socket on a free port that keeps every message with its arrival time."""

    def __init__(self):
        self.sock = zmq.Context.instance().socket(zmq.PULL)
        self.sock.bind("tcp://127.0.0.1:*")
        self.addr = self.sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self.msgs: list[tuple[float, str, dict, bytes]] = []

    def drain(self, until_stop: bool = True, timeout_s: float = 10.0) -> list:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.sock.poll(50):
                topic, hdr, payload = P.parse_in(self.sock.recv_multipart())
                self.msgs.append((time.monotonic(), topic, hdr, payload))
                if until_stop and topic == "ctl" and hdr["cmd"] == "stop":
                    break
        return self.msgs

    def close(self):
        self.sock.close()


@pytest.fixture
def pull():
    p = Pull()
    yield p
    p.close()


# -- replay client -----------------------------------------------------------

def test_load_video_decodes_audio_and_frames(mp4s):
    clip = load_video(mp4s["two"])
    assert clip["pcm"].dtype == np.int16 and abs(len(clip["pcm"]) / P.SR - 2.0) < 0.1
    assert len(clip["jpegs"]) == 20 and clip["w"] == 640 and clip["h"] == 480 and clip["fps"] == 10.0
    f = P.decode_jpeg(clip["jpegs"][5])
    assert f.shape == (480, 640, 3) and f.std() > 10
    # the tone is there: rms well above silence
    assert np.sqrt(np.mean((clip["pcm"].astype(np.float64) / 32768) ** 2)) > 0.05


def test_media_schedule_orders_audio_and_video_by_media_time():
    sched = media_schedule(5, 2, 10.0)
    assert [(k, i) for _, k, i in sched] == [("audio", 0), ("video", 0), ("audio", 1), ("audio", 2), ("audio", 3),
                                             ("video", 1), ("audio", 4)]
    assert [t for t, _, _ in sched] == sorted(t for t, _, _ in sched)
    assert media_schedule(1, 1, 10.0, t0_media=3.0)[1][0] == 3.0


def test_replay_video_sends_both_streams_in_media_order(mp4s, pull):
    clip = load_video(mp4s["two"])
    push = connect_push(pull.addr)
    try:
        r = replay_video(push, clip, "s1", realtime=False, tail_s=0.2)
    finally:
        push.close()
    msgs = pull.drain()
    topics = [t for _, t, _, _ in msgs]
    assert topics[0] == "ctl" and msgs[0][2]["cmd"] == "start" and topics[-1] == "ctl" and msgs[-1][2]["cmd"] == "stop"
    audio = [h for _, t, h, _ in msgs if t == "audio"]
    video = [(h, p) for _, t, h, p in msgs if t == "video"]
    assert [h["seq"] for h in audio] == list(range(r["frames"] + r["tail_frames"]))
    assert [h["seq"] for h, _ in video] == list(range(20)) and r["video_frames"] == 20
    assert [h["t_media"] for h, _ in video] == pytest.approx([k / 10 for k in range(20)])
    assert all(h["w"] == 640 and h["h"] == 480 and p[:2] == P.JPEG_SOI for h, p in video)
    # media time never goes back: audio frame i starts at i * 32 ms
    media = [h["seq"] * P.FRAME_S if t == "audio" else h["t_media"] for _, t, h, _ in msgs if t in ("audio", "video")]
    assert media == sorted(media)
    # the silent tail carries no video
    assert max(h["t_media"] for h, _ in video) < r["frames"] * P.FRAME_S


def test_replay_video_realtime_pacing(mp4s, pull):
    clip = load_video(mp4s["short"])
    push = connect_push(pull.addr)
    t0 = time.monotonic()
    try:
        r = replay_video(push, clip, "s1", realtime=True, tail_s=0.0)
    finally:
        push.close()
    msgs = pull.drain()
    assert r["wall_s"] >= (r["frames"] - 1) * P.FRAME_S - 0.02
    for t_arr, topic, h, _ in msgs:
        if topic == "video":
            assert t_arr - t0 >= h["t_media"] - 0.02, (h["t_media"], t_arr - t0)


def test_replay_cli_with_video(mp4s, pull, capsys):
    rc = replay_main(["--video", str(mp4s["short"]), "--addr", pull.addr, "--tail-s", "0.1", "--stream-id", "cli"])
    assert rc == 0
    msgs = pull.drain()
    assert any(t == "video" for _, t, _, _ in msgs) and all(h["stream_id"] == "cli" for _, _, h, _ in msgs)
    assert "video frames (640x480 at 10 fps)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        replay_main(["--video", str(mp4s["short"]), "--text", "hello"])


# -- camera ------------------------------------------------------------------

class FakeCapture:
    def __init__(self, period_s=0.01, fail_after=None):
        self.period_s, self.n, self.released = period_s, 0, False
        self.fail_after = fail_after

    def read(self):
        time.sleep(self.period_s)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        f = np.zeros((48, 64, 3), np.uint8)
        f[:, :, 0] = self.n % 255
        return True, f

    def release(self):
        self.released = True


def test_camera_sender_stamps_frames_on_the_audio_media_clock(pull):
    cap = FakeCapture(0.01)
    cam = CameraSender(pull.addr, "mic", fps=10.0, capture=cap)
    cam.start()
    time.sleep(0.08)                    # frames read before the audio started are not sent
    a0 = P.now_ns()
    cam.set_audio0(a0)
    time.sleep(0.5)
    cam.stop()
    msgs = [(h, p) for _, t, h, p in pull.drain(until_stop=False, timeout_s=0.5) if t == "video"]
    assert cam.before_audio >= 3 and cap.released and cam.error is None
    assert 3 <= len(msgs) <= 7 and cam.sent == len(msgs)
    ts = [h["t_media"] for h, _ in msgs]
    assert ts[0] >= 0 and all(b - a >= 0.09 - 1e-9 for a, b in itertools.pairwise(ts))
    for h, _ in msgs:
        assert h["t_media"] == pytest.approx((h["t_capture_ns"] - a0) / 1e9, abs=1e-6)
        assert h["w"] == 64 and h["h"] == 48 and h["stream_id"] == "mic"
    assert [h["seq"] for h, _ in msgs] == list(range(len(msgs)))


def test_frames_lost_counts_whole_frames_from_the_wall_clock():
    t0 = 10**12
    assert frames_lost(t0 + int(0.517e9), t0, 6 * 512, 16000) == 10    # 192 ms received in 517 ms: 325 ms lost
    assert frames_lost(t0 + int(0.517e9), t0, 6 * 1536, 48000) == 10
    assert frames_lost(t0 + int(0.192e9), t0, 6 * 512, 16000) == 0
    assert frames_lost(t0 + int(0.150e9), t0, 6 * 512, 16000) == 0     # callback early: no negative skip


class FakeInputStream:
    """sounddevice.InputStream that calls the callback with silent blocks at scripted callback times
    (the patched protocol clock); `over` marks an input overflow before that block."""
    script: ClassVar[list] = []
    clock: ClassVar[list] = [0]

    def __init__(self, device, channels, samplerate, dtype, blocksize, callback):
        self.blocksize, self.callback = blocksize, callback

    class Status:
        def __init__(self, over):
            self.input_overflow = over

        def __bool__(self):
            return self.input_overflow

    def _run(self):
        for t_ns, over in self.script:
            self.clock[0] = t_ns
            self.callback(np.zeros((self.blocksize, 1), np.float32), self.blocksize, None, self.Status(over))
            time.sleep(0.002)

    def __enter__(self):
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.th.join(timeout=5)


@pytest.mark.parametrize("skip_lost", [True, False])
def test_mic_skips_the_sequence_numbers_of_audio_the_device_dropped(pull, monkeypatch, skip_lost):
    """with --camera (skip_lost), an input overflow that lost 325 ms skips 10 sequence numbers, so the
    service puts the next audio where the camera's wall-clock media time is; without it, as before."""
    t0, blk = 10**12, 32_000_000
    FakeInputStream.script = ([(t0 + k * blk, False) for k in range(1, 6)] + [(t0 + 16 * blk + 5_000_000, True)]
                              + [(t0 + k * blk + 5_000_000, False) for k in (17, 18)])
    monkeypatch.setattr(mic, "sd", types.SimpleNamespace(InputStream=FakeInputStream))
    monkeypatch.setattr(P, "now_ns", lambda: FakeInputStream.clock[0])
    push = connect_push(pull.addr)
    a0 = []
    try:
        sent = mic.capture(push, "mic", None, 16000, seconds=0.4, out=io.StringIO(), on_audio0=a0.append, skip_lost=skip_lost)
    finally:
        push.close()
    seqs = [h["seq"] for _, t, h, _ in pull.drain() if t == "audio"]
    assert sent == 8 and a0 == [t0]
    assert seqs == (list(range(5)) + [15, 16, 17] if skip_lost else list(range(8)))


def test_bench_waits_for_the_reply_thread():
    """wait_idle needs the reply thread idle too (--reply llm): a silent generation is still part of the run."""
    col = types.SimpleNamespace(last_non_health=time.monotonic() - 10)

    class Svc:
        replying = True

        def queue_s(self):
            return 0.0

        def reply_idle(self):
            return not self.replying

    svc = Svc()
    assert not rb.wait_idle(col, svc, 0.1, 0.3)
    svc.replying = False
    assert rb.wait_idle(col, svc, 0.1, 0.3)


def test_camera_sender_stops_after_failed_reads(pull):
    cam = CameraSender(pull.addr, "mic", capture=FakeCapture(0.001, fail_after=2), max_failures=5)
    cam.start()
    cam.join(timeout=5)
    assert not cam.is_alive() and cam.error == "5 failed reads"


# -- monitor -----------------------------------------------------------------

def vision_final() -> dict:
    p = {e: (0.7 if e == "joy" else 0.05) for e in EMOTIONS}
    return {"kind": "final", "stream_id": "s1", "utt_id": 4, "window": "full", "t_media_start": 0.5, "t_media_end": 3.1,
            "t_speech_end_capture_ns": 1, "asr": {"text": "that was fun", "conf": 0.9},
            "audio": {"emotion": {"neutral": 0.6}}, "text": {"emotion": p}, "fused": {"label": "joy", "confidence": 0.71, "mode": "stacked_tav"},
            "discrepancy": {"jsd": 0.1, "direction": "none", "face_jsd": 0.42, "face_flag": True, "face_direction": "face_more_negative"},
            "timing_ms": {"service_total": 180.0, "vision_wait": 0.2, "vision_aggregate": 0.9},
            "vision": {"emotion": p, "raw": p, "present": True, "presence": 0.95, "track": 1, "n_tracks": 2},
            "commit": {"label": "joy", "confidence": 0.66, "t_commit_s": 2.048, "matches_final": True, "source": "policy"},
            "degraded": [], "t_emitted_ns": 10_000_000}


def test_monitor_lines_for_vision_commit_and_streamed_reply(tmp_path):
    out = io.StringIO()
    mon = Monitor("tcp://127.0.0.1:1", tmp_path / "log.jsonl", out=out)
    try:
        mon.handle("state.commit", {"stream_id": "s1", "utt_id": 4, "label": "joy", "confidence": 0.66, "t_commit_s": 2.048,
                                    "q_emit": 0.7, "q_wait": 0.4, "policy": "commit_policy.npz", "t_emitted_ns": 5})
        mon.handle("state.final", vision_final())
        for i, w in enumerate(["You", " sound", " happy."]):
            mon.handle("response.delta", {"stream_id": "s1", "utt_id": 4, "seq": i, "text": w, "t_emitted_ns": 12_000_000 + i})
        mon.handle("health", {"stream_id": "s1", "queue_s": 0.0, "video_frames": 30, "dropped_video_frames": 0})
        mon.handle("response", {"stream_id": "s1", "utt_id": 4, "text": "You sound happy.", "template_id": "llm", "style": "generated",
                                "grounding": ["fused.label=joy@0.71"], "source": "llm", "fallback": None, "first_delta_ms": 210.0,
                                "gen_ms": 400.0, "n_tokens": 5, "checks": {"n_words": 3}, "t_emitted_ns": 30})
    finally:
        mon.close()
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("commit s1 utt=4 joy@0.66 t=2.048s q_emit=0.7")
    assert "face=joy@0.70" in lines[1] and "presence=0.95" in lines[1] and "face_jsd=0.42 face_more_negative" in lines[1]
    assert "commit=joy@0.66 t=2.048s policy matches_final=True" in lines[1] and "vis_wait=0.2ms" in lines[1]
    assert lines[2] == "reply~ s1 utt=4:You sound happy."
    assert lines[3].startswith("reply  s1 utt=4 'You sound happy.'") and "source=llm" in lines[3] and "first_delta=210.0ms" in lines[3]
    assert not any(line.startswith("health") for line in lines)     # held back while the reply line was open
    rows = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["topic"] for r in rows] == ["state.commit", "state.final"] + ["response.delta"] * 3 + ["health", "response"]
    assert rows[2]["after_final_ms"] == 2.0


def test_monitor_core_lines_are_unchanged():
    f = vision_final()
    for k in ("vision", "commit"):
        del f[k]
    for k in ("face_jsd", "face_flag", "face_direction"):
        del f["discrepancy"][k]
    line = fmt_line("state.final", f, None)
    assert "face=" not in line and "commit=" not in line and "vis_wait" not in line
    r = fmt_line("response", {"stream_id": "s1", "utt_id": 1, "text": "ok", "template_id": "a", "style": "ask", "grounding": []}, None)
    assert r == "reply  s1 utt=1 'ok' [a ask] grounding=[]"


# -- replay bench --video ----------------------------------------------------

def test_bench_streams_clips_with_video_on_one_media_clock(mp4s, pull):
    clips = [{"name": p.name, "path": str(p), "pcm": rb.decode_audio(p)} for p in (mp4s["two"], mp4s["short"])]
    push = connect_push(pull.addr)
    try:
        sent = rb.stream_clips_av(push, clips, "bench", realtime=False, tail_s=0.5, fps=10.0, ahead=1)
    finally:
        push.close()
    msgs = pull.drain()
    video = [h for _, t, h, _ in msgs if t == "video"]
    assert clips[0]["media_start"] == 0.0 and clips[1]["media_start"] == pytest.approx(clips[0]["media_end"] + 0.512, abs=0.04)
    assert [c["n_video_frames"] for c in clips] == [20, 6] and sent["video_frames"] == 26 and not sent["video_decode_errors"]
    second = [h["t_media"] for h in video[20:]]
    assert second == pytest.approx([clips[1]["media_start"] + k / 10 for k in range(6)])
    assert [h["seq"] for h in video] == list(range(26)) and "_frames" not in clips[0]
    audio = [h["seq"] for _, t, h, _ in msgs if t == "audio"]
    assert audio == list(range(sent["frames"]))


def test_bench_analyze_vision_on_synthetic_messages():
    def final(utt, t0, t1, present, commit, degraded=(), mode="stacked_tav"):
        return (0, "state.final", {"stream_id": "bench", "utt_id": utt, "t_media_start": t0, "t_media_end": t1,
                                   "fused": {"mode": mode}, "discrepancy": {"face_flag": present},
                                   "timing_ms": {"vad_tail": 512.0, "vision_wait": 1.0 * utt, "vision_aggregate": 0.5},
                                   "vision": {"present": present, "presence": 0.9 if present else 0.0, "n_tracks": 1 if present else 0},
                                   "commit": commit, "degraded": list(degraded)})

    msgs = [
        (0, "state.provisional", {"stream_id": "bench", "utt_id": 1, "window": "full", "fused": {"mode": "stacked_tav"},
                                  "timing_ms": {"text_encoder": 30.0, "vision_aggregate": 0.4}}),
        (0, "state.provisional", {"stream_id": "bench", "utt_id": 1, "window": "tail", "fused": {"mode": "poe"},
                                  "timing_ms": {"vision_aggregate": 0.3}}),
        (0, "state.commit", {"stream_id": "bench", "utt_id": 1, "label": "joy", "t_commit_s": 1.024}),
        final(1, 0.0, 3.0, True, {"source": "policy", "t_commit_s": 1.024, "matches_final": True}),
        final(2, 4.0, 5.0, False, {"source": "endpoint", "t_commit_s": 1.512, "matches_final": True}, ["no_face"]),
        (0, "response.delta", {"stream_id": "bench", "utt_id": 1}),
        (0, "response", {"stream_id": "bench", "utt_id": 1, "source": "llm", "first_delta_ms": 300.0, "gen_ms": 900.0, "n_tokens": 12}),
        (0, "response", {"stream_id": "bench", "utt_id": 2, "source": "bank", "fallback": "first_token_timeout", "first_delta_ms": 2100.0}),
        (0, "health", {"stream_id": "bench", "video_frames": 50, "dropped_video_frames": 1, "vision_frame_ms_mean": 12.5}),
        (0, "state.final", {"stream_id": "other", "utt_id": 9}),
    ]
    v = rb.analyze_vision(msgs, "bench")
    assert v["n_finals"] == 2 and v["face_present_finals"] == 1 and v["face_present_rate"] == 0.5
    assert v["degraded"] == {"no_face": 1} and v["vision_wait_ms"]["max"] == 2.0 and v["fused_modes"] == {"stacked_tav": 2}
    assert v["provisionals_full_stacked_tav"] == "1/1" and v["provisional_text_encoder_ms"]["p50"] == 30.0
    c = v["commits"]
    assert c["committed_before_endpoint"] == 1 and c["rate_before_endpoint"] == 0.5 and c["matches_final_rate"] == 1.0
    assert c["seconds_before_endpoint_commit"]["p50"] == pytest.approx(3.0 + 0.512 - 1.024, abs=0.01)
    rp = v["reply"]
    assert rp["sources"] == {"llm": 1, "bank": 1} and rp["fallbacks"] == {"first_token_timeout": 1} and rp["n_deltas"] == 1
    assert rp["first_delta_ms"]["max"] == 2100.0 and rp["n_tokens"]["p50"] == 12
    assert v["video"]["video_frames"] == 50 and v["video"]["vision_frame_ms_mean"] == 12.5


def test_prefetcher_reports_a_broken_clip(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    pre = rb.Prefetcher([{"name": "broken.mp4", "path": str(bad)}], fps=10.0, ahead=1)
    pre.start()
    item = pre.get(0, timeout=30)
    pre.stop()
    assert item["jpegs"] == [] and item["error"]
