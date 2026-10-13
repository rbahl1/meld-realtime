"""replay client: push a wav as 512-sample pcm16 frames at real-time pace
(--realtime) or as fast as possible, or send one typed text message. this is
the headless path used for every measurement; the mic client shares
send_frames. a silent tail (--tail-s) lets the endpoint fire on clips that end
right after the last word.

--video clip.mp4 (spec 3.3): ffmpeg decodes the audio (16 khz mono, as
tools/extract_audio.py) and the video (--video-fps, default 10, scaled to 640 px
wide, as tools/extract_video_features.py) of the same file. both go out on one
schedule ordered by media time: audio frame i at i * 32 ms, video frame k at
k / fps (its presentation time, the media clock of the clip's audio after
`start`), each frame jpeg-encoded at quality 80. with --realtime an item is sent
at t0 + its media time, otherwise as fast as possible in the same order. the
silent tail has no video frames.

usage: python -m meld_rt.clients.replay --wav FILE [--realtime] [--stream-id s1] [--text "typed utterance"]
       python -m meld_rt.clients.replay --video clip.mp4 --realtime [--video-fps 10]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from meld_rt import protocol as P
from meld_rt.clients import connect_push, wait_ready

VIDEO_FPS = 10.0
VIDEO_WIDTH = 640


def load_wav(path: Path) -> np.ndarray:
    """int16 mono at 16 khz; other rates are resampled (scipy), stereo is mixed. a 16 khz mono
    pcm16 file is read as int16 directly so the samples on the wire equal the file."""
    info = sf.info(str(path))
    if info.samplerate == P.SR and info.channels == 1 and info.subtype == "PCM_16":
        return sf.read(str(path), dtype="int16")[0]
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


# -- video (spec 3.3) --------------------------------------------------------

def probe_size(path: Path) -> tuple[int, int]:
    """(width, height) of the first video stream (ffprobe)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60, check=False)
    try:
        w, h = (int(v) for v in r.stdout.strip().split(",")[:2])
    except ValueError:
        raise RuntimeError(f"no video stream in {path}: {r.stderr.strip()[-300:]}") from None
    return w, h


def decode_frames(path: Path, fps: float = VIDEO_FPS, width: int = VIDEO_WIDTH) -> np.ndarray:
    """(n, h, width, 3) uint8 bgr frames at fps, frame k at k / fps; the height keeps
    the aspect ratio, rounded to an even number (tools/extract_video_features.py)."""
    w, h = probe_size(path)
    hh = round(h * width / w / 2.0) * 2
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
                        "-vf", f"fps={fps},scale={width}:{hh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True, timeout=600, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode the video of {path}: {r.stderr.decode(errors='replace')[-300:]}")
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = a.size // (hh * width * 3)
    return a[: n * hh * width * 3].reshape(n, hh, width, 3)


def decode_audio(path: Path) -> np.ndarray:
    """int16 mono 16 khz of the first audio stream, with the options of tools/extract_audio.py."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
                        "-ar", str(P.SR), "-c:a", "pcm_s16le", "-f", "s16le", "-"],
                       capture_output=True, timeout=600, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode the audio of {path}: {r.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(r.stdout, dtype="<i2").copy()


def load_video(path: Path, fps: float = VIDEO_FPS, width: int = VIDEO_WIDTH, quality: int = P.JPEG_QUALITY) -> dict:
    """pcm16 and the jpeg frames of one clip: {pcm, jpegs, w, h, fps, duration_s}."""
    pcm = decode_audio(path)
    frames = decode_frames(path, fps, width)
    jpegs = [P.encode_jpeg(f, quality) for f in frames]
    h, w = (frames.shape[1], frames.shape[2]) if len(frames) else (0, width)
    return {"pcm": pcm, "jpegs": jpegs, "w": int(w), "h": int(h), "fps": float(fps), "duration_s": round(len(pcm) / P.SR, 3)}


def media_schedule(n_audio: int, n_video: int, fps: float, t0_media: float = 0.0) -> list[tuple[float, str, int]]:
    """(t_media, kind, index) for n_audio audio frames and n_video video frames, ordered by
    media time; an audio frame goes first when both start at the same time."""
    items = [(t0_media + i * P.FRAME_S, 0, i) for i in range(n_audio)]
    items += [(t0_media + k / fps, 1, k) for k in range(n_video)]
    items.sort()
    return [(round(t, 6), "audio" if kind == 0 else "video", i) for t, kind, i in items]


def send_av(push, frames: list[np.ndarray], jpegs: list[bytes], w: int, h: int, fps: float, stream_id: str,
            realtime: bool = False, seq0: int = 0, vseq0: int = 0, t0_media: float = 0.0,
            t0: float | None = None) -> tuple[int, int]:
    """push audio frames and jpeg video frames on one media-time schedule from t0_media;
    realtime paces each item at t0 + (its media time - t0_media) on this host's monotonic
    clock. returns the next (audio seq, video seq)."""
    t0 = time.monotonic() if t0 is None else t0
    for t, kind, i in media_schedule(len(frames), len(jpegs), fps, t0_media):
        if realtime:
            wait = t0 + (t - t0_media) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        if kind == "audio":
            push.send_multipart(P.audio_msg(stream_id, seq0 + i, P.now_ns(), frames[i].tobytes()))
        else:
            push.send_multipart(P.video_msg(stream_id, vseq0 + i, P.now_ns(), t, w, h, jpegs[i]))
    return seq0 + len(frames), vseq0 + len(jpegs)


def replay_video(push, clip: dict, stream_id: str, realtime: bool, tail_s: float = 1.0, start: bool = True,
                 stop: bool = True) -> dict:
    """ctl start, the clip's audio and video on one schedule, tail_s of silence (audio only), ctl stop."""
    frames = P.pcm16_frames(clip["pcm"])
    tail = P.pcm16_frames(np.zeros(int(tail_s * P.SR), np.int16))
    if start:
        push.send_multipart(P.ctl_msg(stream_id, "start"))
    t0 = time.monotonic()
    seq, vseq = send_av(push, frames + tail, clip["jpegs"], clip["w"], clip["h"], clip["fps"], stream_id, realtime, t0=t0)
    wall = time.monotonic() - t0
    if stop:
        push.send_multipart(P.ctl_msg(stream_id, "stop"))
    return {"frames": len(frames), "tail_frames": len(tail), "video_frames": vseq, "media_s": round(seq * P.FRAME_S, 3),
            "wall_s": round(wall, 3), "next_seq": seq}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="replay a wav (or send typed text) to the meld_rt service")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--video", type=Path, help="an mp4: its audio and its frames on one media-time schedule")
    ap.add_argument("--video-fps", type=float, default=VIDEO_FPS, help="frames per second sent with --video")
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
    if sum(x is not None for x in (a.wav, a.text, a.video)) != 1:
        ap.error("give exactly one of --wav, --video or --text")
    if a.wait_ready > 0 and wait_ready(a.out_addr, a.wait_ready) is None:
        print(f"no health message within {a.wait_ready} s on {a.out_addr}", file=sys.stderr)
        return 2
    push = connect_push(a.addr)
    try:
        if a.text is not None:
            send_text(push, a.stream_id, a.text)
            print(f"sent text to {a.stream_id}: {a.text!r}")
        elif a.video is not None:
            clip = load_video(a.video, a.video_fps)
            r = replay_video(push, clip, a.stream_id, a.realtime, tail_s=a.tail_s, start=not a.no_start, stop=not a.no_stop)
            print(f"sent {a.video.name}: {r['frames']} audio frames + {r['tail_frames']} tail, {r['video_frames']} video frames "
                  f"({clip['w']}x{clip['h']} at {clip['fps']:g} fps), media {r['media_s']} s, wall {r['wall_s']} s")
        else:
            pcm = load_wav(a.wav)
            r = replay(push, pcm, a.stream_id, a.realtime, tail_s=a.tail_s, start=not a.no_start, stop=not a.no_stop)
            print(f"sent {a.wav.name}: {r['frames']} frames + {r['tail_frames']} tail, media {r['media_s']} s, wall {r['wall_s']} s")
    finally:
        push.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
