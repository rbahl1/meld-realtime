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

usage: python -m meld_rt.service --models models/ --heads heads/ --threads 4
"""

from __future__ import annotations

import argparse
import json
import logging
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
from meld_rt.labels import EMOTIONS, SV_EMO_TOKENS, SV_EVENT_TOKENS, VALENCE
from meld_rt.state import (
    State,
    audio_emotion_dict,
    emotion7_dict,
    events_dict,
    prosody_dict,
)

ROOT = Path(__file__).resolve().parents[1]
FUSION_MODES = ("auto", "stacked", "poe", "text_only")
N_RECENT_TEMPLATES = 5
TRAJECTORY_LEN = 5
WARMUP_S = 1.0
LAUGHTER_IDX = SV_EVENT_TOKENS.index("Laughter")
UNKNOWN_IDX = SV_EMO_TOKENS.index("EMO_UNKNOWN")
log = logging.getLogger("meld_rt.service")


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
    endpoint_silence_ms: int = 400
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

    def __post_init__(self):
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
        for name in ("models", "heads", "tokenizer_dir", "provenance"):
            setattr(self, name, Path(getattr(self, name)))
        if self.bank_path is not None:
            self.bank_path = Path(self.bank_path)


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
    last_prov_ms: float = 0.0
    dialogue: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_ids: list[str] = field(default_factory=list)
    closed_ns: int | None = None
    n_finals: int = 0


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
        self.model_versions = self._versions()
        self._threads = [threading.Thread(target=f, name=n, daemon=True)
                         for f, n in ((self._ingest, "ingest"), (self._worker, "worker"), (self._publisher, "publisher"))]
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
        for name, rel in (("vad", c.vad), ("sensevoice", c.sensevoice), ("roberta", c.roberta)):
            p = c.models / rel
            out[name] = by_path.get(p.name) or (f"size:{p.stat().st_size}" if p.exists() else "missing")
        out["heads"] = sorted(p.name for p in c.heads.glob("*.npz")) if c.heads.exists() else []
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
        if not self.models_loaded.is_set():
            self._load_requested.set()
        st = self._stream(hdr["stream_id"])
        if topic == "audio":
            self._on_audio(st, hdr, payload, t_recv)
        elif topic == "text":
            self._on_text(st, hdr, t_recv)
        else:
            self._on_ctl(st, hdr["cmd"], t_recv)

    def _on_audio(self, st: Stream, hdr: dict, payload: bytes, t_recv: int) -> None:
        if hdr["n_samples"] != P.FRAME_SAMPLES:
            self._error(st.stream_id, "bad_message", f"audio frames must have {P.FRAME_SAMPLES} samples, got {hdr['n_samples']}")
            return
        seq = hdr["seq"]
        if st.last_seq is not None and seq > st.last_seq + 1:
            gap_ms = (seq - st.last_seq - 1) * P.FRAME_S * 1000
            if gap_ms > self.cfg.gap_ms:
                self._gap(st, seq, hdr["t_capture_ns"], t_recv, gap_ms)
        st.last_seq, st.last_recv_ns = seq, t_recv
        frame = P.pcm16_from_bytes(payload)
        vad = st.vad
        for ev in vad.push(frame):
            if ev.kind == "speech_start":
                st.utt_id += 1
                st.seg_utt_id, st.seg_t_start, st.seg_open = st.utt_id, ev.t_media, True
                st.seg_seq_start = seq - round((vad.t_media - ev.t_media) / P.FRAME_S) + 1
                st.next_prov_t = ev.t_media + 1.0 / self.cfg.provisional_hz
                st.prev_prov_audio7, st.last_prov_ms = None, 0.0
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
        self.model_versions = self._versions()
        self.models_loaded.set()
        log.info("models loaded in %.1f s: mode=%s heads=%s text=%s enc=%s", time.perf_counter() - t0, mode,
                 [k for k, v in self.heads.items() if v is not None], self.text is not None, self.sv.with_encoder_states)

    def _load_heads(self) -> dict:
        from meld_rt.heads import (
            AudioCalibrator,
            Fusion,
            SentimentHead,
            TextContextHead,
        )

        d = self.cfg.heads
        out = {}
        for name, klass in (("text_ctx", TextContextHead), ("audio_cal", AudioCalibrator), ("fusion", Fusion), ("sentiment", SentimentHead)):
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
        s = State.empty(job.stream_id, job.utt_id, kind)
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

    def _sentiment(self, fused7: np.ndarray, feats19: np.ndarray | None, text_x: np.ndarray | None) -> str:
        from meld_rt.labels import SENTIMENTS
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

        t2 = time.perf_counter()
        p_text, context_k, text_x = self._text_posterior7(st, emb, degraded)
        if r is not None:
            p_audio, audio_cal = self._audio_posterior7(r, p4)
        else:
            p_audio, audio_cal = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS)), False
        s.text.emotion, s.text.context_k = emotion7_dict(p_text), context_k
        text_cal = self.heads.get("text_ctx") is not None and emb is not None

        feats = None
        if job.kind == "text" or self.mode == "text_only":
            fused, mode, calibrated = p_text, "text_only", text_cal
        elif "text_failed" in degraded:
            fused, mode, calibrated = p_audio, self.mode if self.mode == "stacked" else ("poe" if audio_cal else "poe_uncalibrated"), False
        elif self.mode == "stacked":
            feats = build_fusion_features(np.log(np.clip(p_text, 1e-8, 1)), np.log(np.clip(p_audio, 1e-8, 1)),
                                          s.asr.conf, float(r.event_probs[LAUGHTER_IDX]) if r is not None else 0.0,
                                          float(r.emo_probs[UNKNOWN_IDX]) if r is not None else 0.0)
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
        s.fused.sentiment = self._sentiment(fused, feats, text_x) if asr_ok else "neutral"
        if r is not None and asr_ok and job.kind != "text" and self.mode != "text_only":
            # the raw sensevoice posterior, the same one train_heads.py chose the jsd threshold on
            d, flag, direction = discrepancy(p_text, r.emo_probs, self.jsd_flag)
            s.discrepancy.jsd, s.discrepancy.flag, s.discrepancy.direction = round(d, 4), flag, direction
        s.text.keyphrase = keyphrase(s.asr.text) if asr_ok else None

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
            st.prev_prov_audio7 = None
        tm["heads"] = round((time.perf_counter() - t2) * 1000, 2)
        self._publish("state.final", s.to_dict())
        self._publish("response", {"stream_id": job.stream_id, "utt_id": job.utt_id, "text": resp.text,
                                   "template_id": resp.template_id, "style": resp.style, "grounding": resp.grounding})

    def _run_provisional(self, job: Job) -> None:
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
        with st.lock:
            labels = [t.label for t in st.dialogue]
            s.dialogue.turn_index, s.dialogue.prev_label = st.n_finals, (labels[-1] if labels else None)
            s.dialogue.trajectory = labels[-TRAJECTORY_LEN:]
            prev = st.prev_prov_audio7
            st.prev_prov_audio7 = p_audio
        s.stability = round(1.0 - jsd(prev, p_audio), 4) if prev is not None else 1.0
        st.last_prov_ms = r.timing_ms["total"]
        self._publish("state.provisional", s.to_dict())

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
        if topic.startswith("state."):
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
        if not streams:
            self._send("health", {"stream_id": "", "dropped_provisionals": 0, **base})
        for st in streams:
            self._send("health", {"stream_id": st.stream_id, "dropped_provisionals": st.dropped_provisionals, **base})


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
    return ap


def config_from_args(a: argparse.Namespace) -> ServiceConfig:
    return ServiceConfig(models=a.models, heads=a.heads, threads=a.threads, text_threads=a.text_threads,
                         endpoint_silence_ms=a.endpoint_silence_ms, max_segment_s=a.max_segment_s,
                         provisional_hz=a.provisional_hz, provisional_window_s=a.provisional_window_s,
                         provisional_budget_ms=a.provisional_budget_ms,
                         fusion_mode=a.fusion_mode, prosody=not a.no_prosody, encoder_states=a.encoder_states,
                         in_addr=a.in_addr, out_addr=a.out_addr, jsd_flag=a.jsd_flag, preload=a.preload)


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
