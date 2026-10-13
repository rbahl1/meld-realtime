"""the service: one process, three threads (design 1.2-1.4, 1.8-1.10).

ingest: pull socket; runs silero vad on every frame, keeps the segment state
(speech start, forced cap, sequence gaps) and the bounded provisional policy
(at most provisional_hz, worker idle only, whole segment while it is at most
provisional_window_s long, then the trailing window at half the rate, audio
posterior only); hands jobs to the worker through a queue.
worker: front end + sensevoice, prosody, text expert, heads, discrepancy,
dialogue ring buffer, selector; serialized, so sensevoice may use every core.
publisher: pub socket; stamps t_emitted_ns, emits state.final then response
back to back, health at 1 hz, errors.

fusion mode auto: stacked when heads/fusion.npz exists, else poe (fixed product
of experts, reported as poe_uncalibrated while the expert heads are missing),
and text_only for text messages. a missing heads/text_ctx.npz gives a uniform
text posterior with degraded ["no_text_head"].

confidence: fused.confidence is what the selector buckets on. in
poe_uncalibrated mode it is the calibrated text head's max probability
(temperature applied), not the max of the fixed product, which sharpens two soft
posteriors (0.94 from text 0.42 and audio 0.46 on the en.wav demo); fused.emotion
stays the product and fused.label its argmax, calibrated stays false, and the
selector never buckets an uncalibrated state above mid (selector.py).

three-input extension (spec section 3), off unless --vision or --reply llm:
- vision: a fourth thread owns one VisionExpert per stream. ingest puts video
  messages on a bounded queue (VIDEO_QUEUE_MAX, the oldest frame is dropped and
  counted when it is full) and never decodes them. the vision thread decodes,
  skips frames closer than MIN_FRAME_GAP / video_fps_max in media time to the
  last processed frame of the stream, runs the expert and keeps a ring of face
  rows (last VISION_RING_S of media time). ingest keeps the audio envelope (log
  rms per received frame). a final waits at most vision_lag_ms for a processed
  frame at t_media_end - 0.1 s, or until the stream has no frame queued or in
  processing, then runs vision_features.face_evidence over its window, the
  vision calibrator, fusion3 (fused.mode stacked_tav) and sentiment3. full-window
  provisionals run on the segment grid (about 1, 2, 3 and 4 s at 1 hz, the ticks
  of the commit traces), add the text expert on the partial transcript and face
  evidence up to their end, fuse with fusion3 and feed the commit policy
  (state.commit on its first emit). a sequence gap advances the audio media
  clock over the missing frames, as the video's runs on.
- reply llm: the worker hands each final to a reply thread, which streams the
  generator's words as response.delta and ends with response; the bank line is
  the fallback, and a reply still streaming at llm_reply_timeout_s ends with the
  words sent. with --reply bank and the extension on, the bank line also goes
  out as one response.delta. the core path (no --vision, --reply bank)
  publishes exactly what it did before; a video message there gets the core's
  bad_message (once per stream id) and opens no stream.
package A (meld_rt.vision_features, the vision heads), C (the generator) and D
(meld_rt.commit) are imported lazily, and the service runs without them.

usage: python -m meld_rt.service --models models/ --heads heads/ --threads 4
       python -m meld_rt.service --vision --reply llm --preload
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import logging
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq

from meld_rt import protocol as P
from meld_rt.discrepancy import audio7, discrepancy, jsd, keyphrase
from meld_rt.labels import EMOTIONS, SENTIMENTS, SV_EMO_TOKENS, SV_EVENT_TOKENS, VALENCE
from meld_rt.state import (
    Commit,
    State,
    Vision,
    audio_emotion_dict,
    emotion7_dict,
    events_dict,
    prosody_dict,
)

ROOT = Path(__file__).resolve().parents[1]
FUSION_MODES = ("auto", "stacked", "poe", "text_only")
REPLY_MODES = ("bank", "llm")
N_RECENT_TEMPLATES = 5
TRAJECTORY_LEN = 5
WARMUP_S = 1.0
LAUGHTER_IDX = SV_EVENT_TOKENS.index("Laughter")
UNKNOWN_IDX = SV_EMO_TOKENS.index("EMO_UNKNOWN")
log = logging.getLogger("meld_rt.service")

# vision mode (spec 3.2)
VISION_RULES = ("largest", "present", "asd", "mouth")
DEFAULT_VISION_RULE = "largest"   # until the manifest names the dev-chosen rule
VIDEO_QUEUE_MAX = 4
VISION_RING_S = 60.0
VISION_TARGET_MARGIN_S = 0.1      # a final waits for a processed frame at t_media_end - this
MIN_FRAME_GAP = 0.75              # of 1 / video_fps_max: 10 fps frames with capture jitter still pass at 10 fps max
EMBED_DIM = 512
LOG_RMS_FLOOR = 1e-5              # vision_features.LOG_RMS_FLOOR, for the envelope without package A
GRID_EPS_S = 1e-6                 # float slack of the vision-mode provisional grid
REPLY_GRACE_S = 1.0               # the reply thread waits this long past the generator's own deadline
STUCK_JOIN_S = 5.0                # after a reply timed out, the generation gets this long to end before it counts as stuck
FACE_DIRECTION_OF = {"none": "none", "mixed": "mixed", "tone_more_negative": "face_more_negative",
                     "tone_more_positive": "face_more_positive", "tone_surprised": "face_surprised"}
VISION_HEADS = (("vision_cal", "VisionCalibrator"), ("fusion3", "Fusion3"), ("sentiment3", "Sentiment3"))


def _stream_id_of(frames: list[bytes]) -> str:
    """best effort stream id of a message that failed to parse, for the error topic."""
    try:
        hdr = json.loads(bytes(frames[1])) if len(frames) > 1 else {}
    except (ValueError, UnicodeDecodeError):
        return ""
    return str(hdr.get("stream_id", "")) if isinstance(hdr, dict) else ""


def _num(v, default: float) -> float:
    """a manifest value when it is a number (strings such as "pending" keep the default)."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float(default)


def poe_weight_from(manifest, default: float) -> float:
    """the dev-chosen product-of-experts weight, where train_heads.py writes it
    (audio_fusion.poe.weight); a top-level poe_weight is read too; else the default."""
    m = manifest if isinstance(manifest, dict) else {}
    w = ((m.get("audio_fusion") or {}).get("poe") or {}).get("weight", m.get("poe_weight"))
    return _num(w, default)


def face_discrepancy(p_text, p_face, threshold: float) -> tuple[float, bool, str]:
    """(jsd, flag, direction) of the words against the face: the words-tone rule of
    discrepancy.py with the face posterior in the place of the voice, so the
    directions read face_more_negative, face_more_positive, face_surprised, mixed, none."""
    d, flag, direction = discrepancy(p_text, p_face, threshold)
    return d, flag, FACE_DIRECTION_OF[direction]


def vision_rule_from(manifest, default: str = DEFAULT_VISION_RULE) -> str:
    """the dev-chosen speaker rule: manifest vision.rule (a name, or a dict with
    "chosen" / "name"), else vision.speaker_rule, else the default."""
    v = (manifest or {}).get("vision") if isinstance(manifest, dict) else None
    v = v if isinstance(v, dict) else {}
    r = v.get("rule", v.get("speaker_rule"))
    if isinstance(r, dict):
        r = r.get("chosen", r.get("name"))
    return r if isinstance(r, str) and r in VISION_RULES else default


def construct(cls, required: dict, optional: dict | None = None):
    """cls(**required, **the optional keys cls accepts): FaceRows gets frame_t only
    when package A declares it."""
    if dataclasses.is_dataclass(cls):
        names = {f.name for f in dataclasses.fields(cls)}
    else:
        try:
            names = set(inspect.signature(cls).parameters)
        except (TypeError, ValueError):
            names = set()
    kw = dict(required)
    kw.update({k: v for k, v in (optional or {}).items() if k in names})
    return cls(**kw)


def skip_media(vad, n_samples: int) -> None:
    """advance a VadSegmenter's media clock over samples that never arrived (a sequence
    gap). VadSegmenter.skip when the class has it, else its sample counter."""
    if n_samples <= 0:
        return
    fn = getattr(vad, "skip", None)
    if callable(fn):
        fn(int(n_samples))
    else:
        vad._n_samples += int(n_samples)


def reported_confidence(mode: str, fused: np.ndarray, p_text: np.ndarray) -> float:
    """the confidence the selector buckets on: the fused posterior's max, except in
    poe_uncalibrated mode, where it is the calibrated text head's probability of the
    fused label (temperature applied). the fixed product's max is not a probability of
    being right, and the text head's own max can belong to another class. a uniform
    text posterior (no text head, text_failed) gives 1/7, which the selector holds back on."""
    fused = np.asarray(fused, dtype=np.float64)
    if mode == "poe_uncalibrated":
        return float(np.asarray(p_text, dtype=np.float64)[int(np.argmax(fused))])
    return float(np.max(fused))


@dataclass
class ServiceConfig:
    models: Path = ROOT / "models"
    heads: Path = ROOT / "heads"
    in_addr: str = P.IN_ADDR          # "tcp://127.0.0.1:*" binds a free port (see Service.in_addr)
    out_addr: str = P.OUT_ADDR
    threads: int = 4                  # sensevoice intra-op threads; roberta uses text_threads, vad 1
    text_threads: int = 2
    endpoint_silence_ms: int = 500    # 16 frames, a 512 ms tail (decisions: end-pointing)
    min_speech_ms: int = 250
    preroll_ms: int = 250
    max_segment_s: float = 15.0
    provisional_hz: float = 1.0
    provisional_window_s: float = 4.0
    provisional_budget_ms: float = 0.0  # > 0: skip the next provisional when the last one ran longer than this
    fusion_mode: str = "auto"
    prosody: bool = True
    encoder_states: bool = False
    bank_path: Path | None = None
    low_conf: float = 0.45            # manifest thresholds override these three
    high_conf: float = 0.70
    jsd_flag: float = 0.2
    poe_weight: float = 0.5
    min_asr_conf: float = 0.3
    gap_ms: int = 500
    overloaded_s: float = 5.0
    stream_ttl_s: float = 60.0
    health_hz: float = 1.0
    dialogue_len: int = 8
    preload: bool = False             # default: models load on the first message
    tokenizer_dir: Path = ROOT / "assets/roberta_tokenizer"
    roberta: str = "roberta-base-11.onnx"
    sensevoice: str = "sensevoice/model.int8.onnx"
    sensevoice_tokens: str = "sensevoice/tokens.txt"
    vad: str = "silero_vad.onnx"
    provenance: Path = ROOT / "results/provenance_hashes.json"
    # three-input extension (spec 3.2, 3.4, 3.5)
    vision: bool = False
    video_fps_max: float = 10.0
    yunet: str = "vision/face_detection_yunet_2023mar.onnx"
    fer: str = "vision/facial_expression_recognition_mobilefacenet_2022july.onnx"
    vision_threads: int = 1           # opencv threads of the vision expert
    vision_lag_ms: float = 150.0      # longest wait of a final for the vision thread
    jsd_face_flag: float = 0.2        # manifest vision.jsd_face_flag overrides it
    reply: str = "bank"               # "bank" | "llm"
    llm_dir: Path | None = None       # None: the only model folder under models/llm/
    llm_threads: int = 2
    llm_first_token_timeout_s: float = 2.0
    llm_reply_timeout_s: float = 30.0  # the generator's deadline_s; the whole reply ends by then (plus REPLY_GRACE_S)
    llm_max_new_tokens: int = 40
    commit_policy: Path | None = None  # None: heads/commit_policy.npz when it exists (vision mode only)
    commit: bool = True               # --no-commit turns the commit policy off

    def __post_init__(self):
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
        if self.reply not in REPLY_MODES:
            raise ValueError(f"reply must be one of {REPLY_MODES}")
        if not self.video_fps_max > 0:
            raise ValueError("video_fps_max must be positive")
        if not self.llm_reply_timeout_s > 0:
            raise ValueError("llm_reply_timeout_s must be positive")
        for name in ("models", "heads", "tokenizer_dir", "provenance"):
            setattr(self, name, Path(getattr(self, name)))
        if self.bank_path is not None:
            self.bank_path = Path(self.bank_path)
        for name in ("llm_dir", "commit_policy"):
            if getattr(self, name) is not None:
                setattr(self, name, Path(getattr(self, name)))

    @property
    def extension(self) -> bool:
        """any part of the three-input extension is on."""
        return self.vision or self.reply == "llm"


def build_sensevoice(cfg: ServiceConfig, with_encoder_states: bool = False):
    """the sensevoice expert as the service runs it. tools/cache_audio_features.py
    builds its expert here too, so cached features and live ones come from one call."""
    from meld_rt.experts.sensevoice import SenseVoiceExpert

    return SenseVoiceExpert(cfg.models / cfg.sensevoice, cfg.models / cfg.sensevoice_tokens, threads=cfg.threads,
                            with_encoder_states=with_encoder_states)


@dataclass
class Job:
    kind: str                         # "final" | "provisional" | "text"
    stream_id: str
    utt_id: int
    pcm: np.ndarray | None
    text: str | None
    window: str
    seq_range: list[int]
    t_media_start: float
    t_media_end: float
    t_capture_ns: int
    t_recv_ns: int
    vad_tail_ms: float
    continued_from: int | None
    degraded: list[str]
    forced: bool = False
    t_enqueued_ns: int = 0
    provisional_running: bool = False  # a provisional was in flight when this job was queued
    cancelled: bool = False

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / P.SR if self.pcm is not None else 0.0


class JobQueue:
    """fifo with removal of cancelled provisionals and pending-seconds accounting."""

    def __init__(self):
        self._q: deque[Job] = deque()
        self._cv = threading.Condition()

    def put(self, job: Job) -> None:
        job.t_enqueued_ns = P.now_ns()
        with self._cv:
            self._q.append(job)
            self._cv.notify()

    def get(self, timeout: float) -> Job | None:
        with self._cv:
            if not self._cv.wait_for(lambda: bool(self._q), timeout):
                return None
            return self._q.popleft()

    def drop_provisionals(self, stream_id: str) -> int:
        with self._cv:
            keep = [j for j in self._q if not (j.kind == "provisional" and j.stream_id == stream_id)]
            n = len(self._q) - len(keep)
            self._q = deque(keep)
        return n

    def pending_audio_s(self) -> float:
        with self._cv:
            return float(sum(j.duration_s for j in self._q))

    def oldest_age_s(self, now_ns: int) -> float:
        """how long the oldest queued job has been waiting."""
        with self._cv:
            return (now_ns - self._q[0].t_enqueued_ns) / 1e9 if self._q else 0.0

    def stream_ids(self) -> set[str]:
        with self._cv:
            return {j.stream_id for j in self._q}

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)


@dataclass
class Turn:
    utt_id: int
    label: str
    text: str
    emb: np.ndarray | None


@dataclass
class Stream:
    stream_id: str
    vad: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_seq: int | None = None
    last_recv_ns: int = 0
    utt_id: int = 0
    seg_open: bool = False
    seg_utt_id: int = 0
    seg_t_start: float = 0.0
    seg_seq_start: int = 0
    continued_from: int | None = None
    next_prov_t: float = 0.0
    pending_degraded: list[str] = field(default_factory=list)
    dropped_provisionals: int = 0
    prev_prov_audio7: np.ndarray | None = None
    prev_prov_tav: np.ndarray | None = None  # vision mode: the previous fusion3 provisional posterior
    last_prov_ms: float = 0.0
    dialogue: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_ids: list[str] = field(default_factory=list)
    closed_ns: int | None = None
    n_finals: int = 0
    # vision mode (spec 3.2); the core path never touches these
    v_cv: threading.Condition = field(default_factory=threading.Condition)  # guards the v_* fields and env
    v_epoch: int = 0                  # bumped by ctl start: frames of an older epoch are skipped
    v_frames: deque = field(default_factory=deque)   # FaceFrame per processed frame, oldest first
    v_last_t: float = float("-inf")   # media time of the newest processed frame of this epoch
    v_pending: int = 0                # frames queued or being processed
    env: deque = field(default_factory=lambda: deque(maxlen=int(VISION_RING_S / P.FRAME_S)))  # (t, log rms) per audio frame
    video_frames: int = 0             # processed
    dropped_video_frames: int = 0     # dropped from the full queue
    skipped_video_frames: int = 0     # closer than the frame gap to the last processed frame, or stale after a start
    bad_video_frames: int = 0         # undecodable, or the expert raised
    vision_lag_finals: int = 0
    vision_ms_total: float = 0.0      # decode, expert and row building of the processed frames
    ticks: dict = field(default_factory=dict)        # utt_id -> commit ticks so far (worker only)
    commits: dict = field(default_factory=dict)      # utt_id -> the policy's first emit (worker only)


@dataclass
class VideoItem:
    st: Stream
    seq: int
    t_media: float
    t_capture_ns: int
    t_recv_ns: int
    w: int
    h: int
    payload: bytes
    epoch: int


class VideoQueue:
    """bounded fifo between ingest and the vision thread: put never blocks, and
    when the queue is full the oldest frame is dropped and returned."""

    def __init__(self, maxlen: int = VIDEO_QUEUE_MAX):
        self._q: deque[VideoItem] = deque()
        self._cv = threading.Condition()
        self.maxlen = int(maxlen)

    def put(self, item: VideoItem) -> VideoItem | None:
        with self._cv:
            dropped = self._q.popleft() if len(self._q) >= self.maxlen else None
            self._q.append(item)
            self._cv.notify()
        return dropped

    def get(self, timeout: float) -> VideoItem | None:
        with self._cv:
            if not self._cv.wait_for(lambda: bool(self._q), timeout):
                return None
            return self._q.popleft()

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)


@dataclass
class FaceFrame:
    """the faces of one processed frame, in the FaceRows layout of spec 2.1."""
    t: float
    track: np.ndarray                 # (n,) int
    area: np.ndarray                  # (n,) box w * h over frame w * h
    score: np.ndarray                 # (n,)
    logits: np.ndarray                # (n, 7) fer order
    mouth: np.ndarray                 # (n,) nan on a track's first frame
    face: np.ndarray                  # (n,)
    emb: np.ndarray                   # (n, 512) float32, nan rows without an embedding


def face_frame(t_media: float, frame_shape: tuple, faces: list) -> FaceFrame:
    """FaceFrame from VisionExpert.process output (meld_rt.experts.vision.Face)."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    n = len(faces)
    emb = np.full((n, EMBED_DIM), np.nan, np.float32)
    for i, f in enumerate(faces):
        if f.embed is not None:
            emb[i] = np.asarray(f.embed, np.float32).reshape(-1)[:EMBED_DIM]
    return FaceFrame(
        t=float(t_media),
        track=np.array([int(f.track) for f in faces], dtype=np.int64),
        area=np.array([float(f.box[2]) * float(f.box[3]) / float(w * h) for f in faces], dtype=np.float64),
        score=np.array([float(f.score) for f in faces], dtype=np.float64),
        logits=np.array([np.asarray(f.logits, np.float32).reshape(-1) for f in faces], dtype=np.float32).reshape(n, 7),
        mouth=np.array([float(f.mouth_motion) for f in faces], dtype=np.float64),
        face=np.array([float(f.face_motion) for f in faces], dtype=np.float64),
        emb=emb)


def face_rows(vf, frames: list[FaceFrame]):
    """vision_features.FaceRows over the face rows of these frames. frame_t (every
    processed frame, with or without a face) is passed when FaceRows declares it."""
    cat = (lambda xs, shape, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(shape, dt))
    required = {
        "t": cat([np.full(len(f.track), f.t) for f in frames], (0,), np.float64),
        "track": cat([f.track for f in frames], (0,), np.int64),
        "area": cat([f.area for f in frames], (0,), np.float64),
        "score": cat([f.score for f in frames], (0,), np.float64),
        "logits": cat([f.logits for f in frames], (0, 7), np.float32),
        "mouth": cat([f.mouth for f in frames], (0,), np.float64),
        "face": cat([f.face for f in frames], (0,), np.float64),
        "emb": cat([f.emb for f in frames], (0, EMBED_DIM), np.float32),
    }
    return construct(vf.FaceRows, required, {"frame_t": np.array([f.t for f in frames], dtype=np.float64)})


@dataclass
class VisionResult:
    """what a window's face evidence puts into the state and fusion3."""
    present: bool
    p_face: np.ndarray                # calibrated face posterior (uniform without a face)
    raw: np.ndarray                   # averaged fer posterior of the chosen track (uniform without a face)
    track: int | None = None
    n_frames: int = 0
    presence: float = 0.0
    n_tracks: int = 0
    asd_corr: float = 0.0
    calibrated: bool = False

    def block(self, rule: str) -> Vision:
        return Vision(emotion=emotion7_dict(self.p_face), raw=emotion7_dict(self.raw), present=bool(self.present),
                      track=self.track, rule=rule, n_frames=int(self.n_frames), presence=round(float(self.presence), 4),
                      n_tracks=int(self.n_tracks), asd_corr=round(float(self.asd_corr), 4))


@dataclass
class ReplyJob:
    stream_id: str
    utt_id: int
    state: dict                       # the final's state dict
    bank: object                      # the selector's Response, the fallback
    t_service_recv_ns: int
    t_enqueued_ns: int = 0


class ReplyQueue:
    """fifo of finals waiting for the reply thread. a job taken by get counts as in
    progress until done(), so idle() never reads true between the two."""

    def __init__(self):
        self._q: deque[ReplyJob] = deque()
        self._cv = threading.Condition()
        self._active = 0

    def put(self, job: ReplyJob) -> None:
        job.t_enqueued_ns = P.now_ns()
        with self._cv:
            self._q.append(job)
            self._cv.notify()

    def get(self, timeout: float) -> ReplyJob | None:
        with self._cv:
            if not self._cv.wait_for(lambda: bool(self._q), timeout):
                return None
            self._active += 1
            return self._q.popleft()

    def done(self) -> None:
        with self._cv:
            self._active = max(0, self._active - 1)

    def idle(self) -> bool:
        """no final waiting and no reply in progress."""
        with self._cv:
            return not self._q and self._active == 0

    def has_newer(self, stream_id: str) -> bool:
        """another final of the stream is waiting behind the one just taken."""
        with self._cv:
            return any(j.stream_id == stream_id for j in self._q)

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)


class _Cancelled(Exception):
    """raised from on_delta to stop a generation whose reply already fell back to the bank."""


def _gen_field(res, name: str, default=None):
    if res is None:
        return default
    if isinstance(res, dict):
        return res.get(name, default)
    return getattr(res, name, default)


def _num_or(v, default):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


class Service:
    """importable for in-process tests and the bench: start()/stop(), or run()
    which blocks until stop() is called from another thread or ctrl-c."""

    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self.ctx = zmq.Context.instance()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.models_loaded = threading.Event()
        self._load_requested = threading.Event()
        self._jobs = JobQueue()
        self._out: queue.Queue = queue.Queue()
        self._streams: dict[str, Stream] = {}
        self._streams_lock = threading.Lock()
        self._busy = False
        self._running: Job | None = None
        self._overloaded = False
        self._threads: list[threading.Thread] = []
        self._t0 = time.monotonic()
        self.in_addr = cfg.in_addr
        self.out_addr = cfg.out_addr
        self.mode = cfg.fusion_mode        # resolved at load
        self.sv = self.text = self.selector = None
        self.heads: dict = {}
        self.manifest: dict = {}
        self.model_versions: dict = {}
        self._pull = self._pub = None
        # three-input extension
        self.vf = None                     # meld_rt.vision_features (package A) when importable
        self.vision_rule = DEFAULT_VISION_RULE
        self.jsd_face_flag = cfg.jsd_face_flag
        self._video_q = VideoQueue(VIDEO_QUEUE_MAX)
        self._experts: dict[str, list] = {}   # stream_id -> [epoch, VisionExpert], vision thread only
        self._spare_expert = None
        self.vision_error: str | None = None
        self._envelope_fn = None
        self.commit_policy = None
        self.commit_policy_name: str | None = None
        self._reply_q = ReplyQueue()
        self.generator = None
        self._gen_mod = None
        self.gen_error: str | None = None
        self.llm_model: str | None = None
        self._warned: set[str] = set()
        self._video_refused: set[str] = set()  # stream ids told once that video needs --vision

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._pull = self.ctx.socket(zmq.PULL)
        self._pull.setsockopt(zmq.LINGER, 0)
        self._pull.setsockopt(zmq.RCVHWM, 0)
        self._pull.bind(self.cfg.in_addr)
        self.in_addr = self._pull.getsockopt_string(zmq.LAST_ENDPOINT)
        self._pub = self.ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 500)
        self._pub.setsockopt(zmq.SNDHWM, 0)
        self._pub.bind(self.cfg.out_addr)
        self.out_addr = self._pub.getsockopt_string(zmq.LAST_ENDPOINT)
        if self.cfg.vision:
            # before ingest runs: the audio envelope uses package A's function from the first frame on
            self.vf = self._import_optional("meld_rt.vision_features")
            self._envelope_fn = getattr(self.vf, "frame_log_rms", None)
        self.model_versions = self._versions()
        roles = [(self._ingest, "ingest"), (self._worker, "worker"), (self._publisher, "publisher")]
        if self.cfg.vision:
            roles.append((self._vision, "vision"))
        if self.cfg.reply == "llm":
            roles.append((self._replier, "reply"))
        self._threads = [threading.Thread(target=f, name=n, daemon=True) for f, n in roles]
        for t in self._threads:
            t.start()
        if self.cfg.preload:
            self._load_requested.set()
        self._ready.set()
        log.info("listening in=%s out=%s threads=%d mode=%s", self.in_addr, self.out_addr, self.cfg.threads, self.cfg.fusion_mode)

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """sockets bound and, with preload, models loaded."""
        ok = self._ready.wait(timeout)
        if ok and self.cfg.preload:
            ok = self.models_loaded.wait(timeout)
        return ok

    def run(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                self._stop.wait(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10.0)
        for s in (self._pull, self._pub):
            if s is not None:
                s.close()
        self._pull = self._pub = None

    # -- shared helpers --------------------------------------------------
    def _publish(self, topic: str, payload: dict) -> None:
        self._out.put((topic, payload))

    def _error(self, stream_id: str, code: str, msg: str) -> None:
        log.warning("%s %s: %s", stream_id, code, msg)
        self._publish("error", {"stream_id": stream_id, "code": code, "msg": msg})

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    def _import_optional(self, name: str):
        """a module another package writes, or None (logged) when it is missing or broken."""
        try:
            return importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - the service runs without the extension modules
            self._warn_once(f"import:{name}", "%s unavailable (%s: %s); its part of the extension is off", name,
                            type(e).__name__, e)
            return None

    def _stream(self, stream_id: str) -> Stream:
        with self._streams_lock:
            st = self._streams.get(stream_id)
            if st is None:
                from meld_rt.experts.vad import VadSegmenter
                c = self.cfg
                vad = VadSegmenter(c.models / c.vad, endpoint_silence_ms=c.endpoint_silence_ms, min_speech_ms=c.min_speech_ms,
                                   preroll_ms=c.preroll_ms, max_segment_s=c.max_segment_s, threads=1)
                st = Stream(stream_id, vad, dialogue=deque(maxlen=c.dialogue_len))
                self._streams[stream_id] = st
                log.info("stream %s opened", stream_id)
            st.closed_ns = None
            return st

    def _get_stream(self, stream_id: str) -> Stream:
        """the worker's view: never revives a stream that `stop` closed."""
        with self._streams_lock:
            st = self._streams.get(stream_id)
        return st if st is not None else self._stream(stream_id)

    def queue_s(self) -> float:
        """how far behind the worker is: the age of the oldest unfinished job (queued or running), seconds."""
        now = P.now_ns()
        running = self._running
        age = (now - running.t_enqueued_ns) / 1e9 if running is not None else 0.0
        return round(max(age, self._jobs.oldest_age_s(now)), 3)

    def queue_audio_s(self) -> float:
        """seconds of audio pending in the worker queue plus the running job."""
        running = self._running
        return self._jobs.pending_audio_s() + (running.duration_s if running is not None else 0.0)

    def reply_idle(self) -> bool:
        """no final waiting for the reply thread and no reply streaming (always true with --reply bank)."""
        return self._reply_q.idle()

    def _versions(self) -> dict:
        c = self.cfg
        by_path = {}
        if c.provenance.exists():
            try:
                for row in json.loads(c.provenance.read_text()).get("files", []) or []:
                    by_path[Path(row["path"]).name] = row.get("sha256", "")[:12]
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                pass
        out = {}
        files = [("vad", c.vad), ("sensevoice", c.sensevoice), ("roberta", c.roberta)]
        if c.vision:
            files += [("yunet", c.yunet), ("fer", c.fer)]
        for name, rel in files:
            p = c.models / rel
            out[name] = by_path.get(p.name) or (f"size:{p.stat().st_size}" if p.exists() else "missing")
        out["heads"] = sorted(p.name for p in c.heads.glob("*.npz")) if c.heads.exists() else []
        if c.reply == "llm":
            out["llm"] = self.llm_model or "missing"
        return out

    # -- ingest thread ---------------------------------------------------
    def _ingest(self) -> None:
        poller = zmq.Poller()
        poller.register(self._pull, zmq.POLLIN)
        next_chore = time.monotonic()
        while not self._stop.is_set():
            if poller.poll(50):
                frames = self._pull.recv_multipart()
                self._handle(frames, P.now_ns())
            if time.monotonic() >= next_chore:
                self._chores()
                next_chore = time.monotonic() + 0.25

    def _handle(self, frames: list[bytes], t_recv: int) -> None:
        try:
            topic, hdr, payload = P.parse_in(frames)
        except ValueError as e:
            self._error(_stream_id_of(frames), "bad_message", str(e))
            return
        if topic == "video" and not self.cfg.vision:
            # the core never took video: no stream, no model load, the bad_message the core sent for
            # an unknown topic, once per stream id (not once per frame)
            sid = hdr["stream_id"]
            if sid not in self._video_refused:
                self._video_refused.add(sid)
                self._error(sid, "bad_message", "unknown topic 'video' (the service runs without --vision; video is ignored)")
            return
        if not self.models_loaded.is_set():
            self._load_requested.set()
        st = self._stream(hdr["stream_id"])
        if topic == "audio":
            self._on_audio(st, hdr, payload, t_recv)
        elif topic == "video":
            self._on_video(st, hdr, payload, t_recv)
        elif topic == "text":
            self._on_text(st, hdr, t_recv)
        else:
            self._on_ctl(st, hdr["cmd"], t_recv)

    def _on_video(self, st: Stream, hdr: dict, payload: bytes, t_recv: int) -> None:
        """queue the frame for the vision thread; decoding happens there, so ingest never waits
        (vision mode only: _handle refuses video without --vision)."""
        with st.v_cv:
            st.v_pending += 1
            item = VideoItem(st, hdr["seq"], hdr["t_media"], hdr["t_capture_ns"], t_recv, hdr["w"], hdr["h"], payload, st.v_epoch)
        dropped = self._video_q.put(item)
        if dropped is not None:
            dst = dropped.st
            with dst.v_cv:
                dst.v_pending -= 1
                dst.dropped_video_frames += 1
                dst.v_cv.notify_all()

    def _log_rms(self, frame_i16: np.ndarray) -> float:
        """log rms of one audio frame through vision_features.frame_log_rms, the
        formula of the offline envelopes; the same formula here when package A is missing."""
        fn = self._envelope_fn
        if fn is not None:
            try:
                return float(fn(frame_i16))
            except Exception as e:  # noqa: BLE001
                self._warn_once("envelope", "frame_log_rms failed (%s); using log rms", e)
                self._envelope_fn = None
        x = frame_i16.astype(np.float64) / 32768.0
        return float(np.log(max(float(np.sqrt(np.mean(x * x))), LOG_RMS_FLOOR)))

    def _on_audio(self, st: Stream, hdr: dict, payload: bytes, t_recv: int) -> None:
        if hdr["n_samples"] != P.FRAME_SAMPLES:
            self._error(st.stream_id, "bad_message", f"audio frames must have {P.FRAME_SAMPLES} samples, got {hdr['n_samples']}")
            return
        seq = hdr["seq"]
        if st.last_seq is not None and seq > st.last_seq + 1:
            missing = seq - st.last_seq - 1
            gap_ms = missing * P.FRAME_S * 1000
            if gap_ms > self.cfg.gap_ms:
                self._gap(st, seq, hdr["t_capture_ns"], t_recv, gap_ms)
            if self.cfg.vision:
                # video media time runs on over the gap, so audio media time does too and the two stay
                # on one clock; the core keeps counting received samples only
                skip_media(st.vad, missing * P.FRAME_SAMPLES)
        st.last_seq, st.last_recv_ns = seq, t_recv
        frame = P.pcm16_from_bytes(payload)
        vad = st.vad
        if self.cfg.vision:
            # the envelope value sits at the frame centre on the media clock
            v = self._log_rms(frame)
            with st.v_cv:
                st.env.append((vad.t_media + P.FRAME_S / 2, v))
        for ev in vad.push(frame):
            if ev.kind == "speech_start":
                st.utt_id += 1
                st.seg_utt_id, st.seg_t_start, st.seg_open = st.utt_id, ev.t_media, True
                st.seg_seq_start = seq - round((vad.t_media - ev.t_media) / P.FRAME_S) + 1
                st.next_prov_t = ev.t_media + 1.0 / self.cfg.provisional_hz - (GRID_EPS_S if self.cfg.vision else 0.0)
                st.prev_prov_audio7, st.prev_prov_tav, st.last_prov_ms = None, None, 0.0
            else:
                tail = 0.0 if ev.forced else vad.silence_frames * P.FRAME_S * 1000
                self._enqueue_final(st, ev.pcm, seq, hdr["t_capture_ns"], t_recv, ev.t_media, tail, ev.forced)
                st.continued_from = st.seg_utt_id if ev.forced else None
        if st.seg_open:
            self._maybe_provisional(st, seq, hdr["t_capture_ns"], t_recv)

    def _enqueue_final(self, st: Stream, pcm: np.ndarray, seq: int, t_capture: int, t_recv: int,
                       t_media_end: float, vad_tail_ms: float, forced: bool) -> None:
        st.seg_open = False
        degraded = list(st.pending_degraded)
        st.pending_degraded = []
        self._jobs.drop_provisionals(st.stream_id)
        job = Job("final", st.stream_id, st.seg_utt_id, pcm, None, "full", [st.seg_seq_start, seq], st.seg_t_start,
                  t_media_end, t_capture, t_recv, vad_tail_ms, st.continued_from, degraded, forced=forced,
                  provisional_running=(self._running is not None and self._running.kind == "provisional"))
        st.continued_from = None
        self._jobs.put(job)

    def _maybe_provisional(self, st: Stream, seq: int, t_capture: int, t_recv: int) -> None:
        vad = st.vad
        t = vad.t_media
        if t < st.next_prov_t:
            return
        c = self.cfg
        speech_s = t - st.seg_t_start
        if c.vision:
            # full windows on the segment grid: tick k at the first frame end at or after k / provisional_hz
            # (1.024, 2.016, 3.008, 4.0 s at 1 hz), so the k-th tick is near k s as in package D's traces and
            # the 4th, at 4.0 s, is still a full window. the core schedule below adds a frame of drift per tick
            # (1.024, 2.048, 3.072, then 4.096 is a tail window).
            full = speech_s <= c.provisional_window_s + P.FRAME_S / 2
            if full:
                k = math.floor(speech_s * c.provisional_hz + GRID_EPS_S)
                st.next_prov_t = st.seg_t_start + (k + 1) / c.provisional_hz - GRID_EPS_S
            else:
                st.next_prov_t = t + 2.0 / c.provisional_hz
        else:
            full = speech_s <= c.provisional_window_s
            st.next_prov_t = t + (1.0 if full else 2.0) / c.provisional_hz
        over_budget = c.provisional_budget_ms > 0 and st.last_prov_ms > c.provisional_budget_ms
        if self._busy or len(self._jobs) > 0 or over_budget:
            st.dropped_provisionals += 1
            return
        pcm = vad.speech_pcm()
        if full:
            window, t_start, seq_start = "full", st.seg_t_start, st.seg_seq_start
        else:
            n = int(c.provisional_window_s * P.SR)
            pcm = pcm[-n:]
            window, t_start = "tail", t - len(pcm) / P.SR
            seq_start = seq - len(pcm) // P.FRAME_SAMPLES + 1
        self._jobs.put(Job("provisional", st.stream_id, st.seg_utt_id, pcm, None, window, [seq_start, seq], t_start, t,
                           t_capture, t_recv, 0.0, st.continued_from, list(st.pending_degraded)))

    def _gap(self, st: Stream, seq: int, t_capture: int, t_recv: int, gap_ms: float) -> None:
        vad = st.vad
        self._jobs.drop_provisionals(st.stream_id)
        if st.seg_open:
            pcm = vad.speech_pcm()
            st.pending_degraded = ["audio_gap"]
            if len(pcm) >= int(self.cfg.min_speech_ms / 1000 * P.SR):
                self._enqueue_final(st, pcm, st.last_seq if st.last_seq is not None else seq, t_capture, t_recv,
                                    vad.t_media, 0.0, True)
            st.seg_open = False
        vad.reset(keep_clock=True)
        st.continued_from = None
        if "audio_gap" not in st.pending_degraded:
            st.pending_degraded.append("audio_gap")
        self._error(st.stream_id, "audio_gap", f"sequence gap of {gap_ms:.0f} ms before seq {seq}")

    def _on_text(self, st: Stream, hdr: dict, t_recv: int) -> None:
        st.utt_id += 1
        t = st.vad.t_media
        seq = int(hdr.get("seq", 0))
        self._jobs.put(Job("text", st.stream_id, st.utt_id, None, hdr["text"], "full", [seq, seq], t, t,
                           int(hdr.get("t_capture_ns", 0)), t_recv, 0.0, None, ["no_audio"]))

    def _on_ctl(self, st: Stream, cmd: str, t_recv: int) -> None:
        vad = st.vad
        if cmd == "start":
            self._jobs.drop_provisionals(st.stream_id)
            vad.reset(keep_clock=False)
            st.last_seq, st.seg_open, st.continued_from, st.pending_degraded = None, False, None, []
            if self.cfg.vision:
                # new media clock: the face rows, the envelope and the tracks of the old one go
                with st.v_cv:
                    st.v_epoch += 1
                    st.v_frames.clear()
                    st.env.clear()
                    st.v_last_t = float("-inf")
                    st.v_cv.notify_all()
        elif cmd in ("flush", "stop"):
            if st.seg_open:
                pcm = vad.speech_pcm()
                vad.reset(keep_clock=True)
                if len(pcm) >= int(self.cfg.min_speech_ms / 1000 * P.SR):
                    self._enqueue_final(st, pcm, st.last_seq or 0, 0, t_recv, vad.t_media, 0.0, True)
                else:
                    st.seg_open = False
                    self._jobs.drop_provisionals(st.stream_id)
            if cmd == "stop":
                st.closed_ns = t_recv
        elif cmd == "reset":
            with st.lock:
                st.dialogue.clear()
                st.recent_ids = []
                st.n_finals = 0

    def _chores(self) -> None:
        qs = self.queue_s()
        if qs > self.cfg.overloaded_s and not self._overloaded:
            self._overloaded = True
            for sid in self._jobs.stream_ids():
                self._error(sid, "overloaded", f"worker is {qs:.1f} s behind ({self.queue_audio_s():.1f} s of audio queued)")
        elif qs < self.cfg.overloaded_s / 2:
            self._overloaded = False
        now = P.now_ns()
        ttl = int(self.cfg.stream_ttl_s * 1e9)
        with self._streams_lock:
            for sid in [s for s, st in self._streams.items() if st.closed_ns is not None and now - st.closed_ns > ttl]:
                del self._streams[sid]
                log.info("stream %s expired", sid)
            # the vision thread drops the experts of expired streams on its next idle turn

    # -- worker thread ---------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            if not self.models_loaded.is_set() and self._load_requested.is_set():
                try:
                    self._load()
                except Exception as e:
                    log.exception("model load failed")
                    self._error("", "asr_failed", f"model load failed: {e}")
                    self._load_requested.clear()
            job = self._jobs.get(0.1)
            if job is None or job.cancelled:
                continue
            self._running, self._busy = job, True
            try:
                if not self.models_loaded.is_set():
                    self._load()
                if job.kind == "provisional":
                    self._run_provisional(job)
                else:
                    self._run_final(job)
            except Exception as e:
                log.exception("job failed")
                self._error(job.stream_id, "asr_failed" if job.kind != "text" else "text_failed", f"{type(e).__name__}: {e}")
            finally:
                self._running, self._busy = None, False

    def _load(self) -> None:
        from meld_rt.experts.text import TextExpert
        from meld_rt.response.selector import ResponseSelector

        c = self.cfg
        t0 = time.perf_counter()
        self.heads = self._load_heads()
        need_enc = c.encoder_states or bool(self.heads.get("audio_cal") is not None and self.heads["audio_cal"].uses_encoder_states)
        try:
            self.sv = build_sensevoice(c, with_encoder_states=need_enc)
        except RuntimeError as e:
            if not need_enc:
                raise
            log.warning("encoder states unavailable (%s); running without them", e)
            if self.heads.get("audio_cal") is not None and self.heads["audio_cal"].uses_encoder_states:
                self.heads["audio_cal"] = None
            self.sv = build_sensevoice(c)
        try:
            self.text = TextExpert(c.models / c.roberta, c.tokenizer_dir, threads=c.text_threads)
        except Exception as e:  # noqa: BLE001 - the audio path keeps working without the text expert
            log.warning("text expert unavailable: %s", e)
            self.text = None
        th = self.manifest.get("thresholds", {}) if isinstance(self.manifest, dict) else {}
        self.low_conf = _num(th.get("low_conf"), c.low_conf)
        self.high_conf = _num(th.get("high_conf"), c.high_conf)
        self.jsd_flag = _num(th.get("jsd_flag"), c.jsd_flag)
        self.poe_weight = poe_weight_from(self.manifest, c.poe_weight)
        if c.vision:
            vm = self.manifest.get("vision") if isinstance(self.manifest.get("vision"), dict) else {}
            self.vision_rule = vision_rule_from(self.manifest)
            self.jsd_face_flag = _num(vm.get("jsd_face_flag"), c.jsd_face_flag)
            if self.heads.get("fusion3") is not None:
                # fusion3's calibrated dev predictions set these (spec 2.3 step 4)
                vth = vm.get("thresholds") if isinstance(vm.get("thresholds"), dict) else {}
                self.low_conf = _num(vth.get("low_conf"), self.low_conf)
                self.high_conf = _num(vth.get("high_conf"), self.high_conf)
        self.selector = ResponseSelector(c.bank_path, low_conf=self.low_conf, high_conf=self.high_conf,
                                         no_repeat_turns=N_RECENT_TEMPLATES)
        mode = c.fusion_mode
        if mode == "auto":
            mode = "stacked" if self.heads.get("fusion") is not None else "poe"
        elif mode == "stacked" and self.heads.get("fusion") is None:
            log.warning("fusion-mode stacked requested but heads/fusion.npz is missing; using poe")
            mode = "poe"
        self.mode = mode
        # warm-up: the first onnxruntime call of a session is slower than the rest
        self.sv.run(np.zeros(int(WARMUP_S * P.SR), np.float32))
        if self.text is not None:
            self.text.encode("hello")
        if c.vision:
            self._load_vision()
        if c.reply == "llm":
            self._load_generator()
        self.model_versions = self._versions()
        self.models_loaded.set()
        log.info("models loaded in %.1f s: mode=%s heads=%s text=%s enc=%s", time.perf_counter() - t0, mode,
                 [k for k, v in self.heads.items() if v is not None], self.text is not None, self.sv.with_encoder_states)
        if c.extension:
            log.info("extension: vision=%s rule=%s vision_features=%s expert=%s commit=%s reply=%s generator=%s",
                     c.vision, self.vision_rule, self.vf is not None, self.vision_error or "ok",
                     self.commit_policy_name, c.reply, self.generator_status())

    # -- extension loading -----------------------------------------------
    def new_vision_expert(self):
        """one VisionExpert (spec 1); tests replace this method on the instance."""
        from meld_rt.experts.vision import VisionExpert

        c = self.cfg
        return VisionExpert(c.models / c.yunet, c.models / c.fer, threads=c.vision_threads)

    def _load_vision(self) -> None:
        c = self.cfg
        if self.vf is None:
            self.vf = self._import_optional("meld_rt.vision_features")
            self._envelope_fn = getattr(self.vf, "frame_log_rms", None)
        try:
            # a spare expert, so the first stream's first frames are not spent loading the models
            self._spare_expert = self.new_vision_expert()
            self.vision_error = None
        except Exception as e:  # noqa: BLE001 - the audio path keeps working without vision
            self.vision_error = f"{type(e).__name__}: {e}"
            log.warning("vision expert unavailable: %s", self.vision_error)
            self._error("", "vision_failed", f"vision expert unavailable: {self.vision_error}")
        self.commit_policy, self.commit_policy_name = None, None
        if c.commit:
            path = c.commit_policy if c.commit_policy is not None else c.heads / "commit_policy.npz"
            if path.exists():
                mod = self._import_optional("meld_rt.commit")
                if mod is not None:
                    try:
                        self.commit_policy = mod.CommitPolicy.load(path)
                        self.commit_policy_name = str(getattr(self.commit_policy, "name", None) or path.name)
                    except Exception as e:  # noqa: BLE001
                        log.warning("commit policy %s unusable: %s", path, e)
            elif c.commit_policy is not None:
                log.warning("commit policy %s not found; no commit", path)

    def _llm_dir(self) -> Path | None:
        """--llm-dir, else the generator module's DEFAULT_MODEL_DIR, else the only
        model folder (one with a tokenizer.json) under models/llm/."""
        c = self.cfg
        if c.llm_dir is not None:
            return c.llm_dir
        d = getattr(self._gen_mod, "DEFAULT_MODEL_DIR", None)
        if d is not None and Path(d).exists():
            return Path(d)
        root = c.models / "llm"
        cands = sorted(p.parent for p in root.glob("*/tokenizer.json")) if root.exists() else []
        if len(cands) == 1:
            return cands[0]
        if cands:
            raise FileNotFoundError(f"{len(cands)} models under {root}; pass --llm-dir")
        raise FileNotFoundError(f"no model under {root}")

    def _load_generator(self) -> None:
        c = self.cfg
        self.generator, self.gen_error, self.llm_model = None, None, None
        self._gen_mod = self._import_optional("meld_rt.response.generator")
        if self._gen_mod is None:
            self.gen_error = "no_generator_module"
            return
        try:
            d = self._llm_dir()
            self.generator = self._gen_mod.ReplyGenerator(d, threads=c.llm_threads, max_new_tokens=c.llm_max_new_tokens)
            self.llm_model = Path(d).name
        except Exception as e:  # noqa: BLE001 - the bank line is the fallback
            self.gen_error = f"generator_unavailable: {type(e).__name__}: {e}"
            log.warning("reply generator unavailable: %s", e)
            return
        try:
            # the first onnxruntime calls are slower; one short generation before the first reply
            msgs, _, allowed = self._gen_mod.build_messages(State.empty("warmup", 0, vision_mode=c.vision).to_dict(),
                                                            self.low_conf, high_conf=self.high_conf)
            t = time.perf_counter()
            self.generator.generate(msgs, allowed, lambda _w: None, deadline_s=c.llm_reply_timeout_s)
            log.info("generator warm-up %.0f ms", (time.perf_counter() - t) * 1000)
        except Exception as e:  # noqa: BLE001
            log.warning("generator warm-up failed: %s", e)

    def _load_heads(self) -> dict:
        from meld_rt import heads as H
        from meld_rt.heads import (
            AudioCalibrator,
            Fusion,
            SentimentHead,
            TextContextHead,
        )

        d = self.cfg.heads
        out = {}
        named = [("text_ctx", TextContextHead), ("audio_cal", AudioCalibrator), ("fusion", Fusion), ("sentiment", SentimentHead)]
        if self.cfg.vision:
            # package A's classes when meld_rt.heads has them, else the generic npz head
            named += [(name, getattr(H, cls, H.NpzHead)) for name, cls in VISION_HEADS]
        for name, klass in named:
            p = d / f"{name}.npz"
            out[name] = None
            if p.exists():
                try:
                    out[name] = klass.load(p)
                except (OSError, ValueError, KeyError) as e:
                    log.warning("head %s unusable: %s", p, e)
        self.manifest = {}
        if (d / "manifest.json").exists():
            try:
                self.manifest = json.loads((d / "manifest.json").read_text())
            except json.JSONDecodeError as e:
                log.warning("manifest unreadable: %s", e)
        return out

    # -- pipeline --------------------------------------------------------
    def _base_state(self, job: Job, kind: str) -> State:
        s = State.empty(job.stream_id, job.utt_id, kind, vision_mode=self.cfg.vision)
        s.seq_range = list(job.seq_range)
        s.t_media_start, s.t_media_end = round(job.t_media_start, 3), round(job.t_media_end, 3)
        s.t_speech_end_capture_ns, s.t_service_recv_ns = int(job.t_capture_ns), int(job.t_recv_ns)
        s.window, s.continued_from = job.window, job.continued_from
        s.degraded = list(job.degraded)
        s.timing_ms["vad_tail"] = round(job.vad_tail_ms, 1)
        if job.provisional_running:
            s.timing_ms["provisional_in_flight"] = round(max(0, P.now_ns() - job.t_enqueued_ns) / 1e6, 1)
        return s

    def _audio_posterior7(self, r, prosody4: np.ndarray) -> tuple[np.ndarray, bool]:
        """(7-way audio posterior in MELD order, calibrated?)"""
        head = self.heads.get("audio_cal")
        if head is not None:
            from meld_rt.heads import AudioCalibrator
            x = AudioCalibrator.features(r.emo_probs, r.event_probs, prosody4, r.enc_pooled if head.uses_encoder_states else None)
            return np.exp(head.predict_log_proba(x)), True
        return audio7(r.emo_probs), False

    def _text_posterior7(self, st: Stream, emb: np.ndarray | None, degraded: list[str]) -> tuple[np.ndarray, int, np.ndarray | None]:
        """(7-way text posterior, context_k, text-head features or None)."""
        head = self.heads.get("text_ctx")
        if emb is None:
            return np.full(len(EMOTIONS), 1.0 / len(EMOTIONS)), 0, None
        if head is None:
            if "no_text_head" not in degraded:
                degraded.append("no_text_head")
            return np.full(len(EMOTIONS), 1.0 / len(EMOTIONS)), 0, None
        from meld_rt.heads import TextContextHead
        with st.lock:
            prev = [t.emb for t in st.dialogue if t.emb is not None][-head.k:] if head.k > 0 else []
            turn_index = st.n_finals
        x = TextContextHead.features(emb, prev, [True] * len(prev), turn_index, head.k)
        return np.exp(head.predict_log_proba(x)), head.k, x

    def _sentiment(self, fused7: np.ndarray, feats19: np.ndarray | None, text_x: np.ndarray | None,
                   feats29: np.ndarray | None = None) -> str:
        if feats29 is not None:
            h3 = self.heads.get("sentiment3")
            if h3 is not None and feats29.shape[0] == h3.input_dim:
                names = h3.classes if len(h3.classes) == len(SENTIMENTS) else SENTIMENTS
                return str(names[int(np.argmax(h3.predict_log_proba(feats29)))])
        head = self.heads.get("sentiment")
        if head is not None:
            x = text_x if head.variant == "text_only" else feats19
            if x is not None and x.shape[0] == head.input_dim:
                return SENTIMENTS[int(np.argmax(head.predict_log_proba(x)))]
        # rule until a sentiment head exists: the valence of the fused label
        v = VALENCE[EMOTIONS[int(np.argmax(fused7))]]
        return "positive" if v > 0 else "negative" if v < 0 else "neutral"

    def _run_final(self, job: Job) -> None:
        from meld_rt.experts import prosody as pros
        from meld_rt.heads import build_fusion_features, fuse_poe

        st = self._get_stream(job.stream_id)
        s = self._base_state(job, "final")
        tm = s.timing_ms
        degraded = s.degraded
        r = None
        p4 = np.zeros(4, np.float32)
        if job.kind == "text":
            s.asr.text, s.asr.conf, s.asr.lang = job.text or "", 1.0, "en"
            asr_ok = bool(s.asr.text.strip())
            if not asr_ok:
                degraded.append("asr_failed")
        else:
            try:
                r = self.sv.run(job.pcm)
                tm["frontend"] = r.timing_ms["frontend"]
                tm["sensevoice"] = round(r.timing_ms["model"] + r.timing_ms["decode"], 2)
            except Exception as e:  # noqa: BLE001 - design 1.4: an asr exception degrades the state
                degraded.append("asr_failed")
                self._error(job.stream_id, "asr_failed", f"{type(e).__name__}: {e}")
            if r is not None:
                s.asr.text, s.asr.conf, s.asr.lang = r.text, round(float(r.asr_conf), 4), r.lang
                s.asr.tokens_per_s = round(float(r.tokens_per_s), 3)
                s.audio.emotion, s.audio.events = audio_emotion_dict(r.emo_probs), events_dict(r.event_probs)
                s.audio.encoder_states_used = r.enc_pooled is not None
                if self.cfg.prosody:
                    p4 = pros.prosody_scalars(job.pcm, r.tokens_per_s)
                    tm["prosody"] = round(pros.last_timing_ms, 2)
                else:
                    p4[3] = r.tokens_per_s
                s.audio.prosody = prosody_dict(p4)
                if not r.text.strip():
                    degraded.append("asr_failed")
                elif r.asr_conf < self.cfg.min_asr_conf:
                    degraded.append("low_asr_conf")
            asr_ok = not any(d in ("asr_failed", "low_asr_conf") for d in degraded)

        emb = None
        if asr_ok and self.text is not None:
            t1 = time.perf_counter()
            try:
                emb = self.text.encode(s.asr.text)
            except Exception as e:  # noqa: BLE001 - design 1.4: a text-expert exception degrades the state
                degraded.append("text_failed")
                self._error(job.stream_id, "text_failed", f"{type(e).__name__}: {e}")
            tm["text_encoder"] = round((time.perf_counter() - t1) * 1000, 2)
        elif asr_ok:
            degraded.append("text_failed")

        vis = None
        if self.cfg.vision and job.kind != "text":
            # spec 3.2: wait for the vision thread, then the face evidence over the segment
            vis = self._vision_window(st, job.t_media_start, job.t_media_end, tm, degraded, wait=True)

        t2 = time.perf_counter()
        p_text, context_k, text_x = self._text_posterior7(st, emb, degraded)
        if r is not None:
            p_audio, audio_cal = self._audio_posterior7(r, p4)
        else:
            p_audio, audio_cal = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS)), False
        s.text.emotion, s.text.context_k = emotion7_dict(p_text), context_k
        text_cal = self.heads.get("text_ctx") is not None and emb is not None

        feats = feats3 = None
        if job.kind == "text" or self.mode == "text_only":
            fused, mode, calibrated = p_text, "text_only", text_cal
        elif "text_failed" in degraded:
            fused, mode, calibrated = p_audio, self.mode if self.mode == "stacked" else ("poe" if audio_cal else "poe_uncalibrated"), False
        else:
            if vis is not None:
                feats3 = self._fusion3_features(p_text, p_audio, vis, s.asr.conf, r, degraded)
            if feats3 is not None or self.mode == "stacked":
                feats = build_fusion_features(np.log(np.clip(p_text, 1e-8, 1)), np.log(np.clip(p_audio, 1e-8, 1)),
                                              s.asr.conf, float(r.event_probs[LAUGHTER_IDX]) if r is not None else 0.0,
                                              float(r.emo_probs[UNKNOWN_IDX]) if r is not None else 0.0)
            if feats3 is not None:
                fused, mode, calibrated = np.exp(self.heads["fusion3"].predict_log_proba(feats3)), "stacked_tav", True
            elif self.mode == "stacked":
                fused, mode, calibrated = np.exp(self.heads["fusion"].predict_log_proba(feats)), "stacked", True
            else:
                fused = fuse_poe(p_text, p_audio, self.poe_weight)
                mode, calibrated = ("poe" if (audio_cal and text_cal) else "poe_uncalibrated"), False
        if not asr_ok:
            fused = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS))
        i = int(np.argmax(fused))
        s.fused.emotion, s.fused.mode, s.fused.calibrated = emotion7_dict(fused), mode, bool(calibrated)
        s.fused.label = EMOTIONS[i] if asr_ok else "neutral"
        s.fused.confidence = round(reported_confidence(mode, fused, p_text), 4) if asr_ok else 0.0
        s.fused.sentiment = self._sentiment(fused, feats, text_x, feats3) if asr_ok else "neutral"
        if r is not None and asr_ok and job.kind != "text" and self.mode != "text_only":
            # the raw sensevoice posterior, the same one train_heads.py chose the jsd threshold on
            d, flag, direction = discrepancy(p_text, r.emo_probs, self.jsd_flag)
            s.discrepancy.jsd, s.discrepancy.flag, s.discrepancy.direction = round(d, 4), flag, direction
        s.text.keyphrase = keyphrase(s.asr.text) if asr_ok else None
        if vis is not None:
            s.vision = vis.block(self.vision_rule)
            if asr_ok and text_cal:
                self._set_face_discrepancy(s, p_text, vis)
        if self.cfg.vision and job.kind != "text":
            s.commit = self._final_commit(st, job, s)

        with st.lock:
            labels = [t.label for t in st.dialogue]
            s.dialogue.turn_index = st.n_finals
            s.dialogue.prev_label = labels[-1] if labels else None
            s.dialogue.emotion_changed = bool(labels) and labels[-1] != s.fused.label
            s.dialogue.trajectory = (labels + [s.fused.label])[-TRAJECTORY_LEN:]
            st.dialogue.append(Turn(job.utt_id, s.fused.label, s.asr.text, emb))
            st.n_finals += 1
            resp = self.selector.select(s, st.recent_ids)
            st.recent_ids = (st.recent_ids + [resp.template_id])[-N_RECENT_TEMPLATES:]
            st.prev_prov_audio7 = st.prev_prov_tav = None
        tm["heads"] = round((time.perf_counter() - t2) * 1000, 2)
        self._publish("state.final", s.to_dict())
        if self.cfg.reply == "llm":
            # spec 3.5: the reply thread streams it, so the next utterance is not blocked
            self._reply_q.put(ReplyJob(job.stream_id, job.utt_id, s.to_dict(), resp, int(job.t_recv_ns)))
            return
        if self.cfg.extension:
            # one delta with the whole line, so a consumer of deltas works in both reply modes
            self._publish("response.delta", {"stream_id": job.stream_id, "utt_id": job.utt_id, "seq": 0, "text": resp.text})
        self._publish("response", {"stream_id": job.stream_id, "utt_id": job.utt_id, "text": resp.text,
                                   "template_id": resp.template_id, "style": resp.style, "grounding": resp.grounding})

    def _run_provisional(self, job: Job) -> None:
        t_start = P.now_ns()
        st = self._get_stream(job.stream_id)
        s = self._base_state(job, "provisional")
        r = self.sv.run(job.pcm)
        s.timing_ms["frontend"] = r.timing_ms["frontend"]
        s.timing_ms["sensevoice"] = round(r.timing_ms["model"] + r.timing_ms["decode"], 2)
        s.asr.text = r.text if job.window == "full" else ""
        s.asr.conf, s.asr.lang, s.asr.tokens_per_s = round(float(r.asr_conf), 4), r.lang, round(float(r.tokens_per_s), 3)
        s.text.partial = r.text if job.window == "full" else None
        s.audio.emotion, s.audio.events = audio_emotion_dict(r.emo_probs), events_dict(r.event_probs)
        s.audio.prosody = prosody_dict([0.0, 0.0, 0.0, r.tokens_per_s])
        s.audio.encoder_states_used = r.enc_pooled is not None
        p_audio, audio_cal = self._audio_posterior7(r, np.array([0, 0, 0, r.tokens_per_s], np.float32))
        i = int(np.argmax(p_audio))
        s.fused.emotion, s.fused.label, s.fused.confidence = emotion7_dict(p_audio), EMOTIONS[i], round(float(p_audio[i]), 4)
        s.fused.mode = "text_only" if self.mode == "text_only" else ("poe" if audio_cal else "poe_uncalibrated")
        s.fused.sentiment = self._sentiment(p_audio, None, None)
        s.text.keyphrase = keyphrase(r.text) if s.text.partial else None
        fused, commit = p_audio, None
        if self.cfg.vision:
            fused, commit = self._provisional_tav(job, st, s, r, p_audio)
        with st.lock:
            labels = [t.label for t in st.dialogue]
            s.dialogue.turn_index, s.dialogue.prev_label = st.n_finals, (labels[-1] if labels else None)
            s.dialogue.trajectory = labels[-TRAJECTORY_LEN:]
            # stability: 1 - jsd to the previous provisional's posterior of the same kind. the core fuses
            # nothing here (the audio posterior); vision-mode full windows give fusion3's and tail windows
            # the audio one, and comparing across that switch would measure the change of model, not of evidence
            tav = s.fused.mode == "stacked_tav"
            prev = st.prev_prov_tav if tav else st.prev_prov_audio7
            st.prev_prov_audio7 = p_audio
            if tav:
                st.prev_prov_tav = fused
        s.stability = round(1.0 - jsd(prev, fused), 4) if prev is not None else 1.0
        st.last_prov_ms = r.timing_ms["total"] if not self.cfg.vision else (P.now_ns() - t_start) / 1e6
        self._publish("state.provisional", s.to_dict())
        if commit is not None:
            self._publish("state.commit", commit)

    # -- vision mode: evidence, fusion3, commit (spec 3.2, 3.4) ---------------
    def _vision_window(self, st: Stream, t0: float, t1: float, tm: dict, degraded: list[str], wait: bool) -> VisionResult:
        """face evidence over [t0, t1) of the media clock. a final (wait) first waits
        at most vision_lag_ms for a processed frame at t1 - 0.1 s, and stops waiting
        when the stream has no frame queued or in processing (the video ended, or
        none was sent): only a timeout adds vision_lag."""
        c = self.cfg
        uniform = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS))
        t_w = time.perf_counter()
        if wait and c.vision_lag_ms > 0:
            target = t1 - VISION_TARGET_MARGIN_S
            with st.v_cv:
                ok = st.v_cv.wait_for(lambda: st.v_last_t >= target or st.v_pending <= 0, timeout=c.vision_lag_ms / 1000)
                if not ok:
                    st.vision_lag_finals += 1
            if not ok:
                degraded.append("vision_lag")
        tm["vision_wait"] = round((time.perf_counter() - t_w) * 1000, 2)
        t_a = time.perf_counter()
        res = None
        try:
            res = self._face_evidence(st, t0, t1, degraded)
        except Exception as e:  # noqa: BLE001 - a vision bug degrades the state, the audio path goes on
            self._warn_once(f"evidence:{type(e).__name__}", "face evidence failed: %s: %s", type(e).__name__, e)
            degraded.append("vision_failed")
        if res is None:
            if "vision_failed" not in degraded:
                degraded.append("vision_unavailable")
            res = VisionResult(False, uniform, uniform)
        elif self.vision_error is not None and not res.present:
            degraded.append("vision_unavailable")
        elif not res.present:
            degraded.append("no_face")
        tm["vision_aggregate"] = round((time.perf_counter() - t_a) * 1000, 2)
        return res

    def _face_evidence(self, st: Stream, t0: float, t1: float, degraded: list[str]) -> VisionResult | None:
        vf = self.vf
        if vf is None:
            return None
        uniform = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS))
        with st.v_cv:
            frames = [f for f in st.v_frames if t0 <= f.t < t1]
            env = [e for e in st.env if t0 - 1.0 <= e[0] <= t1 + 1.0]
        env_t = np.array([e[0] for e in env], dtype=np.float64)
        env_v = np.array([e[1] for e in env], dtype=np.float64)
        ev = vf.face_evidence(face_rows(vf, frames), env_t, env_v, self.vision_rule, t0, t1)
        present = bool(ev.present)
        raw = np.asarray(ev.p_face, dtype=np.float64).reshape(-1) if present else uniform
        p_face, calibrated = raw, False
        if present:
            head = self.heads.get("vision_cal")
            if head is not None:
                x = np.asarray(vf.vision_features_A(ev), dtype=np.float32)
                if x.shape[0] != head.input_dim:
                    x = np.asarray(vf.vision_features_B(ev), dtype=np.float32)
                p_face, calibrated = np.exp(head.predict_log_proba(x)), True
            elif "no_vision_head" not in degraded:
                degraded.append("no_vision_head")
        asd = float(_num_or(getattr(ev, "asd_corr", 0.0), 0.0))
        track = getattr(ev, "track", None)
        return VisionResult(present, p_face, raw, track=int(track) if present and track is not None else None,
                            n_frames=int(_num_or(getattr(ev, "n_frames", 0), 0)),
                            presence=float(_num_or(getattr(ev, "presence", 0.0), 0.0)) if present else 0.0,
                            n_tracks=int(_num_or(getattr(ev, "n_tracks", 0), 0)),
                            asd_corr=asd if np.isfinite(asd) else 0.0, calibrated=calibrated)

    def _fusion3_features(self, p_text, p_audio, vis: VisionResult, asr_conf: float, r, degraded: list[str]) -> np.ndarray | None:
        """the 29 fusion3 inputs, or None when fusion3 or vision_features is missing
        (the two-way path then runs). no face: uniform p_face, flags 0."""
        if self.heads.get("fusion3") is None or self.vf is None:
            return None
        def lg(p):
            return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0))

        try:
            x = self.vf.build_fusion3_features(
                lg(p_text), lg(p_audio), lg(vis.p_face), float(asr_conf),
                float(r.event_probs[LAUGHTER_IDX]) if r is not None else 0.0,
                float(r.emo_probs[UNKNOWN_IDX]) if r is not None else 0.0,
                1.0 if vis.present else 0.0, float(vis.presence) if vis.present else 0.0)
            return np.asarray(x, dtype=np.float32).reshape(-1)
        except Exception as e:  # noqa: BLE001
            self._warn_once(f"fusion3:{type(e).__name__}", "fusion3 features failed: %s", e)
            if "vision_failed" not in degraded:
                degraded.append("vision_failed")
            return None

    def _set_face_discrepancy(self, s: State, p_text: np.ndarray, vis: VisionResult) -> None:
        """words against face, only with a face and a text posterior from the text head. the
        raw face posterior, the one train_vision.py chose jsd_face_flag on (as the words-tone
        jsd uses the raw sensevoice posterior)."""
        if not vis.present:
            return
        d, flag, direction = face_discrepancy(p_text, vis.raw, self.jsd_face_flag)
        s.discrepancy.face_jsd, s.discrepancy.face_flag, s.discrepancy.face_direction = round(d, 4), bool(flag), direction

    def _final_commit(self, st: Stream, job: Job, s: State) -> Commit | None:
        """the commit block (spec 3.4): the policy's first emit for this utterance, else the final."""
        entry = st.commits.pop(job.utt_id, None)
        for u in [u for u in st.ticks if u <= job.utt_id]:
            del st.ticks[u]
        for u in [u for u in st.commits if u < job.utt_id]:
            del st.commits[u]
        if self.commit_policy is None:
            return None
        if entry is not None:
            return Commit(label=entry["label"], confidence=entry["confidence"], t_commit_s=entry["t_commit_s"],
                          matches_final=entry["label"] == s.fused.label, source="policy")
        seg = job.t_media_end - job.t_media_start + job.vad_tail_ms / 1000.0
        return Commit(label=s.fused.label, confidence=s.fused.confidence, t_commit_s=round(seg, 3), matches_final=True,
                      source="endpoint")

    def _provisional_tav(self, job: Job, st: Stream, s: State, r, p_audio: np.ndarray) -> tuple[np.ndarray, dict | None]:
        """vision-mode additions to a provisional (spec 3.2): face evidence up to its
        end; on the full window also the text expert on the partial transcript,
        fusion3 over the three posteriors, both discrepancies and a commit tick.
        returns (fused posterior, state.commit payload or None)."""
        tm, degraded = s.timing_ms, s.degraded
        vis = self._vision_window(st, job.t_media_start, job.t_media_end, tm, degraded, wait=False)
        s.vision = vis.block(self.vision_rule)
        if job.window != "full":
            return p_audio, None
        p_text, emb = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS)), None
        if self.text is not None and r.text.strip() and r.asr_conf >= self.cfg.min_asr_conf:
            t1 = time.perf_counter()
            try:
                emb = self.text.encode(r.text)
            except Exception as e:  # noqa: BLE001
                degraded.append("text_failed")
                self._warn_once(f"prov_text:{type(e).__name__}", "text expert failed on a provisional: %s", e)
            tm["text_encoder"] = round((time.perf_counter() - t1) * 1000, 2)
        text_cal = False
        if emb is not None:
            p_text, k, _ = self._text_posterior7(st, emb, degraded)
            s.text.emotion, s.text.context_k = emotion7_dict(p_text), k
            text_cal = self.heads.get("text_ctx") is not None
        fused = p_audio
        feats3 = self._fusion3_features(p_text, p_audio, vis, r.asr_conf, r, degraded) if self.mode != "text_only" else None
        if feats3 is not None:
            fused = np.exp(self.heads["fusion3"].predict_log_proba(feats3))
            i = int(np.argmax(fused))
            s.fused.emotion, s.fused.label, s.fused.confidence = emotion7_dict(fused), EMOTIONS[i], round(float(fused[i]), 4)
            s.fused.mode, s.fused.calibrated = "stacked_tav", True
            s.fused.sentiment = self._sentiment(fused, None, None, feats3)
        if text_cal:
            d, flag, direction = discrepancy(p_text, r.emo_probs, self.jsd_flag)
            s.discrepancy.jsd, s.discrepancy.flag, s.discrepancy.direction = round(d, 4), flag, direction
            self._set_face_discrepancy(s, p_text, vis)
        commit = None
        if feats3 is not None:
            commit = self._commit_tick(st, job, fused, p_text, p_audio, vis, float(r.asr_conf), len(r.text.split()))
        return fused, commit

    def _commit_tick(self, st: Stream, job: Job, fused: np.ndarray, p_text: np.ndarray, p_audio: np.ndarray,
                     vis: VisionResult, asr_conf: float, n_words: int) -> dict | None:
        """one EMIT / WAIT decision of the commit policy (spec 3.4) at a full-window
        provisional; the state.commit payload on the utterance's first emit."""
        pol = self.commit_policy
        if pol is None or job.window != "full" or job.utt_id in st.commits:
            return None
        hist = st.ticks.setdefault(job.utt_id, [])
        t_s = round(job.t_media_end - job.t_media_start, 3)
        lab = int(np.argmax(fused))
        prev = hist[-1] if hist else None
        run = prev["same_label_run"] + 1 if prev is not None and int(np.argmax(prev["fused"])) == lab else 1
        tick = {"t_s": t_s, "k": max(1, round(t_s * self.cfg.provisional_hz)),
                "fused": np.asarray(fused, dtype=np.float64), "prev_fused": prev["fused"] if prev is not None else None,
                "same_label_run": run, "p_text": np.asarray(p_text, dtype=np.float64),
                "p_audio": np.asarray(p_audio, dtype=np.float64), "p_face": np.asarray(vis.p_face, dtype=np.float64),
                "face_present": bool(vis.present), "asr_conf": float(asr_conf), "n_words": int(n_words)}
        hist.append(tick)
        try:
            emit, q_emit, q_wait = pol.decide(pol.features(tick))
        except Exception as e:  # noqa: BLE001 - no commit from a broken policy, the final still comes
            self._warn_once(f"commit:{type(e).__name__}", "commit policy failed: %s: %s", type(e).__name__, e)
            return None
        if not emit:
            return None
        entry = {"label": EMOTIONS[lab], "confidence": round(float(fused[lab]), 4), "t_commit_s": t_s,
                 "q_emit": round(float(q_emit), 4), "q_wait": round(float(q_wait), 4)}
        st.commits[job.utt_id] = entry
        return {"stream_id": job.stream_id, "utt_id": job.utt_id, **entry, "policy": self.commit_policy_name,
                "t_service_recv_ns": int(job.t_recv_ns)}

    # -- vision thread ---------------------------------------------------
    def _vision(self) -> None:
        while not self._stop.is_set():
            item = self._video_q.get(0.1)
            if item is None:
                self._vision_idle()
                continue
            st = item.st
            try:
                self._vision_frame(item)
            except Exception as e:  # noqa: BLE001 - one bad frame never stops the thread
                with st.v_cv:
                    st.bad_video_frames += 1
                self._warn_once(f"vframe:{st.stream_id}:{type(e).__name__}", "video frame of %s failed: %s: %s",
                                st.stream_id, type(e).__name__, e)
            finally:
                with st.v_cv:
                    st.v_pending -= 1
                    st.v_cv.notify_all()

    def _vision_frame(self, item: VideoItem) -> None:
        st = item.st
        with st.v_cv:
            if item.epoch != st.v_epoch or item.t_media < st.v_last_t + MIN_FRAME_GAP / self.cfg.video_fps_max:
                st.skipped_video_frames += 1
                return
        t0 = time.perf_counter()
        frame = P.decode_jpeg(item.payload)
        faces = self._expert(st).process(frame)
        ff = face_frame(item.t_media, frame.shape, faces)
        with st.v_cv:
            if item.epoch != st.v_epoch:
                st.skipped_video_frames += 1
                return
            st.v_frames.append(ff)
            while st.v_frames and st.v_frames[0].t < ff.t - VISION_RING_S:
                st.v_frames.popleft()
            st.v_last_t = ff.t
            st.video_frames += 1
            st.vision_ms_total += (time.perf_counter() - t0) * 1000

    def _expert(self, st: Stream):
        """the stream's VisionExpert (the spare one first); its tracks reset with the media clock."""
        e = self._experts.get(st.stream_id)
        if e is None:
            ex, self._spare_expert = self._spare_expert, None
            if ex is None:
                ex = self.new_vision_expert()
            e = self._experts[st.stream_id] = [st.v_epoch, ex]
        elif e[0] != st.v_epoch:
            e[1].reset()
            e[0] = st.v_epoch
        return e[1]

    def _vision_idle(self) -> None:
        """drop the experts of expired streams, and keep one spare expert once the first is taken."""
        with self._streams_lock:
            live = set(self._streams)
        for sid in [s for s in self._experts if s not in live]:
            del self._experts[sid]
        if self._spare_expert is None and self._experts and self.vision_error is None and self.models_loaded.is_set():
            try:
                self._spare_expert = self.new_vision_expert()
            except Exception as e:  # noqa: BLE001
                self.vision_error = f"{type(e).__name__}: {e}"

    # -- reply thread (spec 3.5) -------------------------------------------
    def _replier(self) -> None:
        while not self._stop.is_set():
            job = self._reply_q.get(0.1)
            if job is None:
                continue
            try:
                if self._reply_q.has_newer(job.stream_id):
                    # a newer final of the stream waits: answer this one at once
                    self._reply_bank(job, "superseded")
                else:
                    self._reply_llm(job)
            except Exception as e:
                log.exception("reply failed")
                self._error(job.stream_id, "text_failed", f"reply failed: {type(e).__name__}: {e}")
            finally:
                self._reply_q.done()

    def _delta(self, job: ReplyJob, seq: int, text: str) -> dict:
        """publish one response.delta; returns its payload, which the publisher stamps with
        t_emitted_ns when it sends it (t_handoff_ns until then)."""
        d = {"stream_id": job.stream_id, "utt_id": job.utt_id, "seq": int(seq), "text": text}
        handoff = P.now_ns()
        self._publish("response.delta", d)
        return {"payload": d, "t_handoff_ns": handoff}

    @staticmethod
    def _sent_ns(first: dict | None) -> int | None:
        """emission time of a delta, or its hand-off time when the publisher has not sent it yet."""
        if first is None:
            return None
        return int(first["payload"].get("t_emitted_ns") or first["t_handoff_ns"])

    def _checks(self, text: str, allowed) -> dict | None:
        fn = getattr(self._gen_mod, "check_reply", None)
        if fn is None or allowed is None:
            return None
        try:
            return fn(text, allowed)
        except Exception as e:  # noqa: BLE001
            self._warn_once("check_reply", "check_reply failed: %s", e)
            return None

    def _reply_payload(self, job: ReplyJob, text: str, source: str, template_id: str, style: str, grounding,
                       fallback: str | None, checks, n_tokens: int, first: dict | None, gen_ms: float | None,
                       res=None) -> dict:
        t_first_ns = self._sent_ns(first)
        return {"stream_id": job.stream_id, "utt_id": job.utt_id, "text": text, "template_id": template_id, "style": style,
                "grounding": list(grounding or []), "source": source, "fallback": fallback, "model": self.llm_model,
                "checks": checks, "n_tokens": int(n_tokens or 0),
                "first_delta_ms": round((t_first_ns - job.t_service_recv_ns) / 1e6, 2) if t_first_ns else None,
                "gen_ms": round(float(gen_ms), 2) if gen_ms is not None else None,
                "stop_reason": _gen_field(res, "stop_reason"), "masked_steps": _gen_field(res, "masked_steps")}

    def _reply_bank(self, job: ReplyJob, fallback: str, allowed=None, gen_ms: float | None = None) -> None:
        """the selector's line as the reply: one delta, then response with source bank."""
        b = job.bank
        first = self._delta(job, 0, b.text)
        self._publish("response", self._reply_payload(job, b.text, "bank", b.template_id, b.style, b.grounding, fallback,
                                                      self._checks(b.text, allowed), 0, first, gen_ms))

    def _reply_llm(self, job: ReplyJob) -> None:
        """generate on a helper thread and stream its words; the bank line when the
        generator is missing, raises, or gives no word within llm_first_token_timeout_s.
        the generator gets llm_reply_timeout_s as its deadline, and the reply thread stops
        waiting REPLY_GRACE_S after that: the response then carries the words already
        streamed (fallback reply_timeout_after_<n>_words). two generations never run at
        once: a late one is stopped at its next word, and one that does not end is
        marked stuck (every later reply is a bank line)."""
        gen, gm, c = self.generator, self._gen_mod, self.cfg
        if gen is None:
            self._reply_bank(job, self.gen_error or "no_generator")
            return
        try:
            # the selector's thresholds, so the prompt's confidence word matches the selector's bucket
            messages, grounding, allowed = gm.build_messages(job.state, self.low_conf, high_conf=self.high_conf)
        except Exception as e:  # noqa: BLE001
            self._warn_once(f"build_messages:{type(e).__name__}", "build_messages failed: %s", e)
            self._reply_bank(job, f"build_messages_failed: {type(e).__name__}")
            return
        words: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        out: dict = {}

        def on_delta(w: str) -> None:
            if cancelled.is_set():
                raise _Cancelled
            words.put(("delta", w))

        def run() -> None:
            try:
                out["res"] = gen.generate(messages, allowed, on_delta, deadline_s=c.llm_reply_timeout_s)
            except _Cancelled:
                out["cancelled"] = True
            except Exception as e:  # noqa: BLE001
                out["error"] = e
            finally:
                words.put(("done", None))

        t0 = time.perf_counter()
        t_end = t0 + c.llm_reply_timeout_s + REPLY_GRACE_S
        th = threading.Thread(target=run, name="reply-gen", daemon=True)
        th.start()
        try:
            kind, w = words.get(timeout=min(c.llm_first_token_timeout_s, t_end - t0))
        except queue.Empty:
            kind, w = "timeout", None
        if kind != "delta":
            cancelled.set()
            if kind == "timeout":
                reason = "first_token_timeout"
            elif "error" in out:
                reason = f"generator_error: {type(out['error']).__name__}"
            else:
                reason = "empty_reply"
            self._reply_bank(job, reason, allowed, gen_ms=(time.perf_counter() - t0) * 1000)
            self._join_generation(th)
            return
        seq, parts, first = 0, [], None
        while kind == "delta":
            d = self._delta(job, seq, w)
            first = first or d
            parts.append(w)
            seq += 1
            try:
                kind, w = words.get(timeout=max(0.0, t_end - time.perf_counter()))
            except queue.Empty:
                kind, w = "timeout", None
        if kind == "timeout":
            # the generation stalled after its first word: answer with what was streamed, stop it at
            # its next word, and give it STUCK_JOIN_S to end
            cancelled.set()
            text = "".join(parts).strip()
            self._publish("response", self._reply_payload(job, text, "llm", "llm", "generated", grounding,
                                                          f"reply_timeout_after_{seq}_words", self._checks(text, allowed),
                                                          0, first, (time.perf_counter() - t0) * 1000))
            self._join_generation(th, STUCK_JOIN_S)
            return
        self._join_generation(th)
        gen_ms = (time.perf_counter() - t0) * 1000
        res = out.get("res")
        text = str(_gen_field(res, "text") or "".join(parts)).strip()
        fallback = f"generator_error_after_{seq}_words: {type(out['error']).__name__}" if "error" in out else None
        self._publish("response", self._reply_payload(job, text, "llm", "llm", "generated", grounding, fallback,
                                                      self._checks(text, allowed), _gen_field(res, "n_tokens", 0),
                                                      first, gen_ms, res))

    def generator_status(self) -> str | None:
        """the model name while the generator is in use, else why it is not (no_generator_module,
        generator_unavailable: ..., generator_stuck); None before the load and with --reply bank."""
        return self.llm_model if self.generator is not None else self.gen_error

    def _join_generation(self, th: threading.Thread, timeout_s: float = 30.0) -> None:
        th.join(timeout_s)
        if th.is_alive():
            # a generation that never ends: later replies use the bank line
            self.generator, self.gen_error = None, "generator_stuck"
            log.error("reply generator did not finish within %.0f s; replies fall back to the bank", timeout_s)

    # -- publisher thread ------------------------------------------------
    def _publisher(self) -> None:
        period = 1.0 / self.cfg.health_hz
        next_health = time.monotonic() + period
        while not (self._stop.is_set() and self._out.empty()):
            try:
                item = self._out.get(timeout=max(0.0, min(period, next_health - time.monotonic())))
            except queue.Empty:
                item = None
            if item is not None:
                self._send(*item)
            if time.monotonic() >= next_health:
                self._health()
                next_health = time.monotonic() + period

    def _send(self, topic: str, payload: dict) -> None:
        t = P.now_ns()
        payload["t_emitted_ns"] = t
        if topic in ("state.provisional", "state.final"):
            payload["timing_ms"]["service_total"] = round((t - int(payload.get("t_service_recv_ns", t))) / 1e6, 2)
        try:
            self._pub.send_multipart(P.out_msg(topic, payload), zmq.NOBLOCK)
        except zmq.ZMQError as e:  # pragma: no cover - only when a subscriber's queue is full
            log.warning("publish %s failed: %s", topic, e)

    def _health(self) -> None:
        import psutil
        rss = round(psutil.Process().memory_info().rss / 2**20, 1)
        with self._streams_lock:
            streams = list(self._streams.values())
        base = {"queue_s": self.queue_s(), "queue_audio_s": round(self.queue_audio_s(), 3), "fusion_mode": self.mode,
                "model_versions": self.model_versions,
                "rss_mb": rss, "models_loaded": self.models_loaded.is_set(), "uptime_s": round(time.monotonic() - self._t0, 1)}
        if self.cfg.extension:
            base.update({"vision": self.cfg.vision, "video_queue": len(self._video_q), "reply": self.cfg.reply,
                         "reply_queue": len(self._reply_q), "generator": self.generator_status(),
                         "commit_policy": self.commit_policy_name, "vision_error": self.vision_error})
        if not streams:
            self._send("health", {"stream_id": "", "dropped_provisionals": 0, **base})
        for st in streams:
            msg = {"stream_id": st.stream_id, "dropped_provisionals": st.dropped_provisionals, **base}
            if self.cfg.vision:
                with st.v_cv:
                    msg.update({"video_frames": st.video_frames, "dropped_video_frames": st.dropped_video_frames,
                                "skipped_video_frames": st.skipped_video_frames, "bad_video_frames": st.bad_video_frames,
                                "vision_lag_finals": st.vision_lag_finals,
                                "vision_frame_ms_mean": round(st.vision_ms_total / st.video_frames, 2) if st.video_frames else None})
            self._send("health", msg)


# -- cli -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    d = ServiceConfig()
    ap = argparse.ArgumentParser(description="meld_rt service: audio/text in over zeromq, state and reply out")
    ap.add_argument("--models", type=Path, default=d.models)
    ap.add_argument("--heads", type=Path, default=d.heads)
    ap.add_argument("--threads", type=int, default=d.threads, help="sensevoice intra-op threads")
    ap.add_argument("--text-threads", type=int, default=d.text_threads)
    ap.add_argument("--endpoint-silence-ms", type=int, default=d.endpoint_silence_ms)
    ap.add_argument("--max-segment-s", type=float, default=d.max_segment_s)
    ap.add_argument("--provisional-hz", type=float, default=d.provisional_hz)
    ap.add_argument("--provisional-window-s", type=float, default=d.provisional_window_s)
    ap.add_argument("--provisional-budget-ms", type=float, default=d.provisional_budget_ms,
                    help="skip the next provisional when the last one took longer than this (0: never)")
    ap.add_argument("--fusion-mode", choices=FUSION_MODES, default=d.fusion_mode)
    ap.add_argument("--no-prosody", action="store_true")
    ap.add_argument("--encoder-states", action="store_true", help="use the graph-edited model's encoder_out")
    ap.add_argument("--in-addr", default=d.in_addr)
    ap.add_argument("--out-addr", default=d.out_addr)
    ap.add_argument("--jsd-flag", type=float, default=d.jsd_flag)
    ap.add_argument("--preload", action="store_true", help="load models at start instead of on the first message")
    ap.add_argument("--log-level", default="info")
    g = ap.add_argument_group("three-input extension (spec 3)")
    g.add_argument("--vision", action="store_true", help="take video messages: faces, fusion3, commit policy")
    g.add_argument("--video-fps-max", type=float, default=d.video_fps_max, help="frames per second the vision thread processes per stream")
    g.add_argument("--yunet", default=d.yunet, help="face detector, relative to --models")
    g.add_argument("--fer", default=d.fer, help="facial expression model, relative to --models")
    g.add_argument("--vision-threads", type=int, default=d.vision_threads, help="opencv threads of the vision expert")
    g.add_argument("--vision-lag-ms", type=float, default=d.vision_lag_ms, help="longest wait of a final for the vision thread")
    g.add_argument("--jsd-face-flag", type=float, default=d.jsd_face_flag,
                   help="words-against-face threshold while the manifest has no vision.jsd_face_flag")
    g.add_argument("--reply", choices=REPLY_MODES, default=d.reply, help="bank lines, or the generator streamed as response.delta")
    g.add_argument("--llm-dir", type=Path, default=None, help="generator model folder (default: the only one under models/llm/)")
    g.add_argument("--llm-threads", type=int, default=d.llm_threads)
    g.add_argument("--llm-first-token-timeout-s", type=float, default=d.llm_first_token_timeout_s,
                   help="fall back to the bank line when no word comes within this")
    g.add_argument("--llm-reply-timeout-s", type=float, default=d.llm_reply_timeout_s,
                   help="the generator's deadline; a reply still streaming after it ends with the words sent so far")
    g.add_argument("--commit-policy", type=Path, default=None, help="default: heads/commit_policy.npz when it exists")
    g.add_argument("--no-commit", action="store_true", help="no commit policy")
    return ap


def config_from_args(a: argparse.Namespace) -> ServiceConfig:
    return ServiceConfig(models=a.models, heads=a.heads, threads=a.threads, text_threads=a.text_threads,
                         endpoint_silence_ms=a.endpoint_silence_ms, max_segment_s=a.max_segment_s,
                         provisional_hz=a.provisional_hz, provisional_window_s=a.provisional_window_s,
                         provisional_budget_ms=a.provisional_budget_ms,
                         fusion_mode=a.fusion_mode, prosody=not a.no_prosody, encoder_states=a.encoder_states,
                         in_addr=a.in_addr, out_addr=a.out_addr, jsd_flag=a.jsd_flag, preload=a.preload,
                         vision=a.vision, video_fps_max=a.video_fps_max, yunet=a.yunet, fer=a.fer,
                         vision_threads=a.vision_threads, vision_lag_ms=a.vision_lag_ms, jsd_face_flag=a.jsd_face_flag,
                         reply=a.reply, llm_dir=a.llm_dir, llm_threads=a.llm_threads,
                         llm_first_token_timeout_s=a.llm_first_token_timeout_s,
                         llm_reply_timeout_s=a.llm_reply_timeout_s, commit_policy=a.commit_policy,
                         commit=not a.no_commit)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname).1s %(name)s %(message)s", stream=sys.stderr)
    svc = Service(config_from_args(a))
    svc.start()
    print(f"service listening in={svc.in_addr} out={svc.out_addr}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
