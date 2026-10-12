"""replay client: push a wav as 512-sample pcm16 frames at real-time pace
(--realtime) or as fast as possible, or send one typed text message. this is
the headless path used for every measurement; the mic client shares
send_frames. a silent tail (--tail-s) lets the endpoint fire on clips that end
right after the last word.

usage: python -m meld_rt.clients.replay --wav FILE [--realtime] [--stream-id s1] [--text "typed utterance"]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from meld_rt import protocol as P
from meld_rt.clients import connect_push, wait_ready


def load_wav(path: Path) -> np.ndarray:
    """int16 mono at 16 khz; other rates are resampled (scipy), stereo is mixed."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != P.SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(P.SR, int(sr))
        x = resample_poly(x, P.SR // g, int(sr) // g).astype(np.float32)
    return P.to_pcm16(x)


def send_frames(push, frames: list[np.ndarray], stream_id: str, seq0: int = 0, realtime: bool = False,
                t0: float | None = None) -> int:
    """push frames with increasing seq; realtime paces them at 32 ms on this
    host's monotonic clock from t0. returns the next seq."""
    t0 = time.monotonic() if t0 is None else t0
    for i, f in enumerate(frames):
        if realtime:
            wait = t0 + i * P.FRAME_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        push.send_multipart(P.audio_msg(stream_id, seq0 + i, P.now_ns(), f.tobytes()))
    return seq0 + len(frames)


def replay(push, pcm16: np.ndarray, stream_id: str, realtime: bool, tail_s: float = 1.0,
           start: bool = True, stop: bool = True, seq0: int = 0) -> dict:
    """ctl start, the clip, tail_s of silence, ctl stop. returns a small summary."""
    frames = P.pcm16_frames(pcm16)
    tail = P.pcm16_frames(np.zeros(int(tail_s * P.SR), np.int16))
    if start:
        push.send_multipart(P.ctl_msg(stream_id, "start"))
    t0 = time.monotonic()
    seq = send_frames(push, frames + tail, stream_id, seq0=seq0, realtime=realtime, t0=t0)
    wall = time.monotonic() - t0
    if stop:
        push.send_multipart(P.ctl_msg(stream_id, "stop"))
    return {"frames": len(frames), "tail_frames": len(tail), "media_s": round(seq * P.FRAME_S, 3),
            "wall_s": round(wall, 3), "next_seq": seq}


def send_text(push, stream_id: str, text: str, seq: int = 0) -> None:
    push.send_multipart(P.text_msg(stream_id, seq, P.now_ns(), text))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="replay a wav (or send typed text) to the meld_rt service")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--text", help="send one typed utterance instead of audio")
    ap.add_argument("--realtime", action="store_true", help="pace frames at 32 ms; default is as fast as possible")
    ap.add_argument("--stream-id", default="s1")
    ap.add_argument("--addr", default=P.IN_ADDR)
    ap.add_argument("--tail-s", type=float, default=1.0, help="silence appended so the endpoint fires")
    ap.add_argument("--no-start", action="store_true", help="do not reset the media clock first")
    ap.add_argument("--no-stop", action="store_true", help="leave the stream open")
    ap.add_argument("--wait-ready", type=float, default=0.0, metavar="S",
                    help="wait up to S seconds for a health message with models_loaded on --out-addr before sending")
    ap.add_argument("--out-addr", default=P.OUT_ADDR)
    a = ap.parse_args(argv)
    if (a.wav is None) == (a.text is None):
        ap.error("give exactly one of --wav or --text")
    if a.wait_ready > 0 and wait_ready(a.out_addr, a.wait_ready) is None:
        print(f"no health message within {a.wait_ready} s on {a.out_addr}", file=sys.stderr)
        return 2
    push = connect_push(a.addr)
    try:
        if a.text is not None:
            send_text(push, a.stream_id, a.text)
            print(f"sent text to {a.stream_id}: {a.text!r}")
        else:
            pcm = load_wav(a.wav)
            r = replay(push, pcm, a.stream_id, a.realtime, tail_s=a.tail_s, start=not a.no_start, stop=not a.no_stop)
            print(f"sent {a.wav.name}: {r['frames']} frames + {r['tail_frames']} tail, media {r['media_s']} s, wall {r['wall_s']} s")
    finally:
        push.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
