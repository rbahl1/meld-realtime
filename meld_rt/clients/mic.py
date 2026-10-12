"""microphone client: sounddevice capture, pushed as 512-sample pcm16 frames
exactly like the replay client. the device is opened at 16 khz mono when it
allows it; otherwise at --samplerate and resampled. needs the `mic` extra
(sounddevice) and libportaudio2 on linux.

usage: python -m meld_rt.clients.mic [--list-devices] [--device N]
"""

from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np

from meld_rt import protocol as P
from meld_rt.clients import connect_push
from meld_rt.clients.replay import send_frames

try:
    import sounddevice as sd
except (ImportError, OSError) as e:  # OSError: libportaudio missing
    sd = None
    _SD_ERROR = e


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


def capture(push, stream_id: str, device: int | None, samplerate: int, seconds: float, block_ms: float = 32.0,
            out=sys.stderr) -> int:
    """stream from the device until ctrl-c or `seconds`; returns frames sent."""
    q: queue.Queue = queue.Queue()
    overflows = 0

    def cb(indata, frames, t, status):
        nonlocal overflows
        if status and status.input_overflow:
            overflows += 1
        q.put((P.now_ns(), indata[:, 0].copy()))

    rs = Resampler(samplerate)
    buf = np.zeros(0, np.float32)
    seq, sent = 0, 0
    push.send_multipart(P.ctl_msg(stream_id, "start"))
    t_end = time.monotonic() + seconds if seconds > 0 else None
    with sd.InputStream(device=device, channels=1, samplerate=samplerate, dtype="float32",
                        blocksize=int(samplerate * block_ms / 1000), callback=cb):
        print(f"capturing device={device} sr={samplerate} -> {stream_id}; ctrl-c to stop", file=out, flush=True)
        try:
            while t_end is None or time.monotonic() < t_end:
                try:
                    _, block = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                buf = np.concatenate([buf, rs(block)])
                n = len(buf) // P.FRAME_SAMPLES * P.FRAME_SAMPLES
                if n:
                    frames = P.pcm16_frames(buf[:n])
                    buf = buf[n:]
                    seq = send_frames(push, frames, stream_id, seq0=seq)
                    sent += len(frames)
        except KeyboardInterrupt:
            pass
    push.send_multipart(P.ctl_msg(stream_id, "stop"))
    print(f"sent {sent} frames ({sent * P.FRAME_S:.1f} s), {overflows} input overflows", file=out, flush=True)
    return sent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="push microphone audio to the meld_rt service")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, default=None, help="sounddevice input index (default device when omitted)")
    ap.add_argument("--samplerate", type=int, default=P.SR, help="capture rate; resampled to 16 khz when different")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: until ctrl-c)")
    ap.add_argument("--stream-id", default="mic")
    ap.add_argument("--addr", default=P.IN_ADDR)
    a = ap.parse_args(argv)
    if sd is None:
        print(f"sounddevice is not available ({_SD_ERROR}); install the `mic` extra and libportaudio2", file=sys.stderr)
        return 2
    if a.list_devices:
        print(sd.query_devices())
        return 0
    push = connect_push(a.addr)
    try:
        capture(push, a.stream_id, a.device, a.samplerate, a.seconds)
    finally:
        push.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
