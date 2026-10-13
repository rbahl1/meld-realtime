"""wire format (design 1.4): zeromq multipart messages, topic names, the frame
size and the clocks. input is push (client) to pull (service), output is pub
(service) to sub (any consumer). input messages are [topic, header_json,
payload]; output messages are [topic, json]. audio payloads are little-endian
int16, 512 samples at 16 khz per message.

clocks: the media clock (cumulative samples / 16000 since `start`) is the only
clock inside state fields. t_capture_ns is the client's monotonic clock and
t_service_recv_ns / t_emitted_ns the service's; they are never subtracted from
each other unless client and service share a host (the monitor does that).
the host clock is time.monotonic_ns, except on windows: there python 3.12's
monotonic clock is GetTickCount64 (15.6 ms steps), so now_ns reads
time.perf_counter_ns (QueryPerformanceCounter, the same clock in every process).

extension (spec 3.1): a `video` input carries one jpeg frame stamped with its
media time t_media. the replay client stamps a frame with its presentation time
in the clip, whose audio starts at media 0 after `start`. the live client stamps
t_media = (t_capture_ns - t_audio0_ns) / 1e9, where t_audio0_ns is the capture
time of the first audio sample sent after `start`, both on the client's
monotonic clock; the two streams then share the media clock up to the
difference of their capture latencies (camera exposure to read() against the
audio block callback). outputs `state.commit` and `response.delta` belong to
the extension. IN_TOPICS / OUT_TOPICS stay the core sets; EXT_* hold the new
topics and parse_in / out_msg / parse_out accept both."""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

IN_ADDR = "tcp://127.0.0.1:5555"
OUT_ADDR = "tcp://127.0.0.1:5556"
FRAME_SAMPLES = 512
SR = 16000
FRAME_S = FRAME_SAMPLES / SR
FRAME_BYTES = 2 * FRAME_SAMPLES
FMT = "pcm16"

IN_TOPICS = ("audio", "text", "ctl")
OUT_TOPICS = ("state.provisional", "state.final", "response", "health", "error")
CTL_CMDS = ("start", "flush", "reset", "stop")
ERROR_CODES = ("audio_gap", "overloaded", "asr_failed", "text_failed", "bad_message", "vision_failed")

# three-input extension (spec 3.1, 3.4, 3.5)
EXT_IN_TOPICS = ("video",)
EXT_OUT_TOPICS = ("state.commit", "response.delta")
ALL_IN_TOPICS = IN_TOPICS + EXT_IN_TOPICS
ALL_OUT_TOPICS = OUT_TOPICS + EXT_OUT_TOPICS
VIDEO_FMT = "jpeg"
JPEG_QUALITY = 80
JPEG_SOI = b"\xff\xd8"


def host_clocks(platform: str = sys.platform):
    """(ns clock, s clock) of now_ns and now_s on this platform. windows takes the
    performance counter: python 3.12's monotonic clock there moves in 15.6 ms steps."""
    if platform == "win32":
        return time.perf_counter_ns, time.perf_counter
    return time.monotonic_ns, time.monotonic


_CLOCK_NS, _CLOCK_S = host_clocks()


def now_ns() -> int:
    """monotonic wall clock of this host, ns (shared by the processes on it)."""
    return _CLOCK_NS()


def now_s() -> float:
    """now_ns in seconds, for pacing against it."""
    return _CLOCK_S()


def _json_default(o):
    if hasattr(o, "item") and not hasattr(o, "__len__"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not json serializable: {type(o).__name__}")


def dumps(d: dict) -> bytes:
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"), default=_json_default).encode("utf-8")


# -- input messages --------------------------------------------------------

def audio_msg(stream_id: str, seq: int, t_capture_ns: int, pcm16: bytes) -> list[bytes]:
    """one frame of little-endian int16 samples."""
    pcm16 = bytes(pcm16)
    header = {"stream_id": str(stream_id), "seq": int(seq), "t_capture_ns": int(t_capture_ns),
              "sr": SR, "n_samples": len(pcm16) // 2, "fmt": FMT}
    return [b"audio", dumps(header), pcm16]


def text_msg(stream_id: str, seq: int, t_capture_ns: int, text: str, speaker: str = "user") -> list[bytes]:
    header = {"stream_id": str(stream_id), "seq": int(seq), "t_capture_ns": int(t_capture_ns),
              "text": str(text), "speaker": str(speaker)}
    return [b"text", dumps(header), b""]


def ctl_msg(stream_id: str, cmd: str) -> list[bytes]:
    if cmd not in CTL_CMDS:
        raise ValueError(f"unknown ctl command {cmd!r}")
    return [b"ctl", dumps({"stream_id": str(stream_id), "cmd": cmd}), b""]


def video_msg(stream_id: str, seq: int, t_capture_ns: int, t_media: float, w: int, h: int, jpeg: bytes) -> list[bytes]:
    """one jpeg frame of w x h pixels at media time t_media (s)."""
    header = {"stream_id": str(stream_id), "seq": int(seq), "t_capture_ns": int(t_capture_ns),
              "t_media": float(t_media), "w": int(w), "h": int(h), "fmt": VIDEO_FMT}
    return [b"video", dumps(header), bytes(jpeg)]


def encode_jpeg(frame_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """uint8 bgr frame -> jpeg bytes (opencv, the `vision` extra)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(frame_bgr), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("jpeg encoding failed")
    return buf.tobytes()


def decode_jpeg(payload: bytes) -> np.ndarray:
    """jpeg bytes -> uint8 bgr frame; ValueError when opencv cannot decode them."""
    import cv2

    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("payload is not a decodable jpeg")
    return frame


def frame_msg(stream_id: str, seq: int, t_capture_ns: int, t_media: float, frame_bgr: np.ndarray,
              quality: int = JPEG_QUALITY) -> list[bytes]:
    """video_msg of a bgr frame, jpeg-encoded at `quality`."""
    h, w = frame_bgr.shape[:2]
    return video_msg(stream_id, seq, t_capture_ns, t_media, w, h, encode_jpeg(frame_bgr, quality))


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_in(frames: list[bytes]) -> tuple[str, dict, bytes]:
    """validate an input message; ValueError names what is wrong (the service
    answers with error{code: bad_message})."""
    if not isinstance(frames, (list, tuple)) or len(frames) not in (2, 3):
        raise ValueError(f"expected 2 or 3 frames, got {len(frames) if hasattr(frames, '__len__') else type(frames)}")
    topic = bytes(frames[0]).decode("utf-8", "replace")
    if topic not in ALL_IN_TOPICS:
        raise ValueError(f"unknown topic {topic!r}")
    try:
        header = json.loads(bytes(frames[1]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"header is not json: {e}") from None
    if not isinstance(header, dict) or not isinstance(header.get("stream_id"), str) or not header["stream_id"]:
        raise ValueError("header needs a non-empty string stream_id")
    payload = bytes(frames[2]) if len(frames) == 3 else b""
    if topic == "audio":
        for k in ("seq", "t_capture_ns", "n_samples"):
            if not isinstance(header.get(k), int):
                raise ValueError(f"audio header needs integer {k}")  # noqa: TRY004
        if header.get("fmt", FMT) != FMT:
            raise ValueError(f"unsupported fmt {header.get('fmt')!r}")
        if header.get("sr", SR) != SR:
            raise ValueError(f"unsupported sr {header.get('sr')!r}")
        if len(payload) != 2 * header["n_samples"]:
            raise ValueError(f"payload has {len(payload)} bytes for n_samples={header['n_samples']}")
    elif topic == "text":
        if not isinstance(header.get("text"), str):
            raise ValueError("text header needs a string text")
        header.setdefault("seq", 0)
        header.setdefault("t_capture_ns", 0)
        header.setdefault("speaker", "user")
    elif topic == "video":
        for k in ("seq", "t_capture_ns", "w", "h"):
            if not _is_int(header.get(k)):
                raise ValueError(f"video header needs integer {k}")
        if header["w"] <= 0 or header["h"] <= 0:
            raise ValueError(f"video size must be positive, got {header['w']}x{header['h']}")
        t = header.get("t_media")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError(f"video header needs a finite t_media >= 0, got {t!r}")
        header["t_media"] = float(t)
        if header.get("fmt", VIDEO_FMT) != VIDEO_FMT:
            raise ValueError(f"unsupported video fmt {header.get('fmt')!r}")
        if not payload.startswith(JPEG_SOI):
            raise ValueError("video payload is not a jpeg")
    elif header.get("cmd") not in CTL_CMDS:
        raise ValueError(f"unknown ctl command {header.get('cmd')!r}")
    return topic, header, payload


# -- output messages -------------------------------------------------------

def out_msg(topic: str, payload: dict) -> list[bytes]:
    if topic not in ALL_OUT_TOPICS:
        raise ValueError(f"unknown output topic {topic!r}")
    return [topic.encode("utf-8"), dumps(payload)]


def parse_out(frames: list[bytes]) -> tuple[str, dict]:
    if len(frames) != 2:
        raise ValueError(f"expected 2 frames, got {len(frames)}")
    topic = bytes(frames[0]).decode("utf-8")
    if topic not in ALL_OUT_TOPICS:
        raise ValueError(f"unknown output topic {topic!r}")
    payload = json.loads(bytes(frames[1]).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("payload is not a json object")  # noqa: TRY004
    return topic, payload


# -- pcm helpers -----------------------------------------------------------

def to_pcm16(x: np.ndarray) -> np.ndarray:
    """int16 samples from int16 or float in [-1, 1] (clipped); stereo is mixed to mono."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.astype(np.float32).mean(axis=1) if x.dtype.kind == "f" else x.astype(np.float32).mean(axis=1) / 32768.0
    if x.dtype == np.int16:
        return x
    if x.dtype.kind == "f":
        return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    return x.astype(np.int16)


def pcm16_frames(pcm16: np.ndarray) -> list[np.ndarray]:
    """split into FRAME_SAMPLES-long int16 frames; the last one is zero-padded."""
    x = to_pcm16(pcm16)
    n = int(np.ceil(len(x) / FRAME_SAMPLES)) if len(x) else 0
    pad = np.zeros(n * FRAME_SAMPLES - len(x), np.int16)
    x = np.concatenate([x, pad]) if len(pad) else x
    return [x[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n)]


def pcm16_from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2")


def pcm16_to_f32(b: bytes) -> np.ndarray:
    return pcm16_from_bytes(b).astype(np.float32) / 32768.0
