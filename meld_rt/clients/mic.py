"""microphone client: sounddevice capture, pushed as 512-sample pcm16 frames
exactly like the replay client. the device is opened at 16 khz mono when it
allows it; otherwise at --samplerate and resampled. needs the `mic` extra
(sounddevice) and libportaudio2 on linux. the windows and macos wheels of
sounddevice bring portaudio with them.

windows: a wasapi device is opened with wasapi's sample rate converter
(auto_convert), which shared mode needs for a rate other than the mixer's. the
start line also names the host api and the input latency portaudio reports for
the stream (sounddevice opens it at its "high" latency). that is the audio
callback's latency in the clock note below, not measured on windows.

--camera N (spec 3.3) adds a capture thread on cv2.VideoCapture(N) at 640x480
that sends at most --fps jpeg frames per second on its own push socket. a frame
is stamped t_media = (t_capture_ns - t_audio0_ns) / 1e9, where t_audio0_ns is
the capture time of the first audio sample sent after `start` (the first
block's callback time minus the block length) and t_capture_ns the time read()
returned, both on this host's monotonic clock. audio and video then share the
media clock up to the difference of their capture latencies: read() returns
after the camera's exposure and transfer, so a frame's t_media is late by that
latency minus the audio callback's. frames captured before t_audio0 are not sent.
the camera's media time is wall time, so with --camera an input overflow (audio
the device dropped) skips sequence numbers for the frames lost, measured on the
same clock (frames_lost); the service in vision mode advances the audio media
clock over a sequence gap, and both streams stay on one clock. without --camera
the audio is sent as before. needs the `vision` extra (opencv).
--camera-backend picks opencv's capture api (default: opencv's own order, which
on windows tries media foundation before directshow). on windows the camera is
opened with media foundation's hardware transforms off
(OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 unless it is set), because opencv
says they may slow down camera probing, and the summary names the backend used.

usage: python -m meld_rt.clients.mic [--list-devices] [--device N] [--camera 0 [--fps 10] [--camera-backend dshow]]
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections.abc import Callable

import numpy as np

from meld_rt import protocol as P
from meld_rt.clients import connect_push, utf8_stdio
from meld_rt.clients.replay import send_frames

try:
    import sounddevice as sd
except (ImportError, OSError) as e:  # OSError: libportaudio missing
    sd = None
    _SD_ERROR = e

CAMERA_BACKENDS = ("any", "msmf", "dshow")

class Resampler:
    """stateful polyphase resampling of a block stream to 16 khz (scipy)."""

    def __init__(self, sr_in: int):
        from math import gcd

        g = gcd(P.SR, sr_in)
        self.up, self.down = P.SR // g, sr_in // g
        self.identity = sr_in == P.SR

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.identity:
            return x
        from scipy.signal import resample_poly

        return resample_poly(x, self.up, self.down).astype(np.float32)


class CameraSender(threading.Thread):
    """camera frames as video messages on the audio media clock (see the module docstring).
    `capture` replaces cv2.VideoCapture (an object with read() and release())."""

    def __init__(self, addr: str, stream_id: str, index: int = 0, fps: float = 10.0, width: int = 640,
                 height: int = 480, quality: int = P.JPEG_QUALITY, capture=None, max_failures: int = 50,
                 backend: str = "any"):
        super().__init__(name="camera", daemon=True)
        self.addr, self.stream_id, self.index = addr, stream_id, int(index)
        if backend not in CAMERA_BACKENDS:
            raise ValueError(f"camera backend {backend!r} is not one of {CAMERA_BACKENDS}")
        self.backend = backend
        self.backend_name: str | None = None
        self.period = 1.0 / float(fps)
        self.width, self.height, self.quality = int(width), int(height), int(quality)
        self.capture = capture
        self.max_failures = int(max_failures)
        self.t_audio0_ns: int | None = None
        self.sent = self.before_audio = self.failures = 0
        self.error: str | None = None
        self._halt = threading.Event()

    def set_audio0(self, t_audio0_ns: int) -> None:
        """the capture time of the first audio sample sent after start; frames are sent from then on."""
        self.t_audio0_ns = int(t_audio0_ns)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)

    def _open(self):
        if self.capture is not None:
            return self.capture
        import cv2

        if self.backend == "any":
            cap = cv2.VideoCapture(self.index)
        else:
            cap = cv2.VideoCapture(self.index, {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}[self.backend])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.index} did not open" + ("" if self.backend == "any" else f" with {self.backend}"))
        try:
            self.backend_name = cap.getBackendName()
        except (cv2.error, AttributeError):
            self.backend_name = None
        return cap

    def run(self) -> None:
        from meld_rt.clients import connect_push

        try:
            cap = self._open()
        except Exception as e:  # noqa: BLE001 - audio keeps going without the camera
            self.error = f"{type(e).__name__}: {e}"
            print(f"camera unavailable: {self.error}", file=sys.stderr, flush=True)
            return
        push = connect_push(self.addr)
        last, seq = None, 0
        try:
            while not self._halt.is_set():
                ok, frame = cap.read()
                t_cap = P.now_ns()
                if not ok or frame is None:
                    self.failures += 1
                    if self.failures >= self.max_failures:
                        self.error = f"{self.failures} failed reads"
                        break
                    time.sleep(0.01)
                    continue
                a0 = self.t_audio0_ns
                t_media = (t_cap - a0) / 1e9 if a0 is not None else -1.0
                if t_media < 0:
                    self.before_audio += 1
                    continue
                if last is not None and t_media - last < 0.9 * self.period:
                    continue
                push.send_multipart(P.frame_msg(self.stream_id, seq, t_cap, t_media, frame, self.quality))
                last, seq = t_media, seq + 1
                self.sent = seq
        finally:
            cap.release()
            push.close()


def frames_lost(t_block_ns: int, t_audio0_ns: int, samples_in: int, samplerate: int) -> int:
    """whole 16 khz frames the device dropped so far: the time from the first sample to the
    callback of the block just received, minus the samples received (that block included)."""
    lost_s = (int(t_block_ns) - int(t_audio0_ns)) / 1e9 - samples_in / samplerate
    return max(0, int(lost_s * P.SR) // P.FRAME_SAMPLES)


def host_api_name(device: int | None) -> str:
    """the portaudio host api of the input device (ex. MME, Windows WASAPI), '' when unknown."""
    try:
        return str(sd.query_hostapis(sd.query_devices(device, "input")["hostapi"])["name"])
    except Exception:  # noqa: BLE001 - only a label
        return ""


def stream_options(device: int | None, platform: str | None = None) -> tuple[dict, str]:
    """(extra InputStream keywords, host api name). windows: a wasapi input gets auto_convert, which
    shared mode needs when the rate differs from the mixer's. other platforms: ({}, '')."""
    if (platform or sys.platform) != "win32":
        return {}, ""
    api = host_api_name(device)
    if "WASAPI" in api.upper():
        return {"extra_settings": sd.WasapiSettings(auto_convert=True)}, api
    return {}, api


def capture(push, stream_id: str, device: int | None, samplerate: int, seconds: float, block_ms: float = 32.0,
            out=sys.stderr, on_audio0: Callable[[int], None] | None = None, skip_lost: bool = False) -> int:
    """stream from the device until ctrl-c or `seconds`; returns frames sent. on_audio0 gets
    the capture time (ns, monotonic) of the first audio sample sent after start. skip_lost
    (with --camera): after an input overflow, skip the sequence numbers of the frames lost."""
    q: queue.Queue = queue.Queue()
    overflows = 0

    def cb(indata, frames, t, status):
        nonlocal overflows
        over = bool(status and status.input_overflow)
        if over:
            overflows += 1
        q.put((P.now_ns(), indata[:, 0].copy(), over))

    rs = Resampler(samplerate)
    buf = np.zeros(0, np.float32)
    seq, sent, samples_in, skipped = 0, 0, 0, 0
    t_audio0 = None
    push.send_multipart(P.ctl_msg(stream_id, "start"))
    t_end = time.monotonic() + seconds if seconds > 0 else None
    extra, api = stream_options(device)
    with sd.InputStream(device=device, channels=1, samplerate=samplerate, dtype="float32",
                        blocksize=int(samplerate * block_ms / 1000), callback=cb, **extra) as stream:
        where = ""
        if sys.platform == "win32":
            where = f" host_api={api or '?'} input_latency={getattr(stream, 'latency', None)} s"
        print(f"capturing device={device} sr={samplerate}{where} -> {stream_id}; ctrl-c to stop", file=out, flush=True)
        try:
            while t_end is None or time.monotonic() < t_end:
                try:
                    t_cb, block, over = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if t_audio0 is None:
                    # the callback runs once the block is full: its first sample is a block earlier
                    t_audio0 = t_cb - int(len(block) / samplerate * 1e9)
                samples_in += len(block)
                if over and skip_lost:
                    # the device dropped audio before this block: its frames' sequence numbers go
                    # unused, so the service sees the gap and audio media time keeps up with the camera's
                    lost = frames_lost(t_cb, t_audio0, samples_in, samplerate)
                    if lost:
                        seq += lost
                        skipped += lost
                        samples_in += round(lost * P.FRAME_SAMPLES * samplerate / P.SR)
                buf = np.concatenate([buf, rs(block)])
                n = len(buf) // P.FRAME_SAMPLES * P.FRAME_SAMPLES
                if n:
                    frames = P.pcm16_frames(buf[:n])
                    buf = buf[n:]
                    seq = send_frames(push, frames, stream_id, seq0=seq)
                    if sent == 0 and on_audio0 is not None:
                        on_audio0(t_audio0)
                    sent += len(frames)
        except KeyboardInterrupt:
            pass
    push.send_multipart(P.ctl_msg(stream_id, "stop"))
    print(f"sent {sent} frames ({sent * P.FRAME_S:.1f} s), {overflows} input overflows"
          + (f", {skipped} lost frames skipped in seq" if skip_lost else ""), file=out, flush=True)
    return sent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="push microphone audio to the meld_rt service")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, default=None, help="sounddevice input index (default device when omitted)")
    ap.add_argument("--samplerate", type=int, default=P.SR, help="capture rate; resampled to 16 khz when different")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: until ctrl-c)")
    ap.add_argument("--stream-id", default="mic")
    ap.add_argument("--addr", default=P.IN_ADDR)
    ap.add_argument("--camera", type=int, default=None, metavar="N", help="also send frames of cv2.VideoCapture(N)")
    ap.add_argument("--fps", type=float, default=10.0, help="video frames per second with --camera")
    ap.add_argument("--camera-backend", choices=CAMERA_BACKENDS, default="any",
                    help="opencv capture api with --camera (any: opencv's own order)")
    a = ap.parse_args(argv)
    utf8_stdio()
    if sd is None:
        need = "the `mic` extra" + (" and libportaudio2" if sys.platform.startswith("linux") else "")
        print(f"sounddevice is not available ({_SD_ERROR}); install {need}", file=sys.stderr)
        return 2
    if a.list_devices:
        print(sd.query_devices())
        if sys.platform == "win32":
            # windows lists a microphone once per audio api. --samplerate takes the default rate shown here
            apis = [h["name"] for h in sd.query_hostapis()]
            for d in sd.query_devices():
                if d["max_input_channels"] > 0:
                    print(f"input {d['index']}: {apis[d['hostapi']]}, default {d['default_samplerate']:g} Hz")
        return 0
    push = connect_push(a.addr)
    cam = None
    if a.camera is not None:
        if sys.platform == "win32":
            # opencv: media foundation's hardware transforms "may slow down camera probing"
            os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
        cam = CameraSender(a.addr, a.stream_id, a.camera, fps=a.fps, backend=a.camera_backend)
        cam.start()
    try:
        capture(push, a.stream_id, a.device, a.samplerate, a.seconds, on_audio0=cam.set_audio0 if cam else None,
                skip_lost=cam is not None)
    finally:
        if cam is not None:
            cam.stop()
            print(f"camera: sent {cam.sent} frames, {cam.before_audio} before the audio started"
                  + (f", backend {cam.backend_name}" if sys.platform == "win32" else "")
                  + (f", error {cam.error}" if cam.error else ""), file=sys.stderr, flush=True)
        push.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
