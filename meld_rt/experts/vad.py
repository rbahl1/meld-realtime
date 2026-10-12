"""silero vad v4 end-pointing over 512-sample frames (16 khz).

media time is cumulative pushed samples / 16000. a segment opens after
`start_frames` consecutive frames with p > threshold (3 frames or min_speech_ms,
whichever is longer) and includes `preroll_ms` of audio before that run; it
closes after `endpoint_silence_ms` of p < 0.35 or at `max_segment_s` (forced).
`speech_start.t_media` is the first sample of the segment pcm;
`speech_end.t_media` is the end of the last frame with p >= 0.35 (the time the
speech stopped), while the returned pcm still carries the silence tail.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16000
FRAME = 512
FRAME_S = FRAME / SR
SILENCE_P = 0.35  # lower hysteresis used for the endpoint


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "speech_end"
    t_media: float
    pcm: np.ndarray | None  # float32 of the finished segment on speech_end
    forced: bool


class VadSegmenter:
    def __init__(self, model_path: Path | str, endpoint_silence_ms: int = 400, min_speech_ms: int = 250,
                 preroll_ms: int = 250, max_segment_s: float = 15.0, threshold: float = 0.5,
                 threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        names = [i.name for i in self.sess.get_inputs()]
        # k2-fsa v4 export: x, h, c -> prob, new_h, new_c. upstream v5 export: input, sr, state -> output, stateN
        self._style = "hc" if "h" in names and "c" in names else "state"
        if self._style == "state":
            shp = next(i.shape for i in self.sess.get_inputs() if i.name == "state")
            self._state_shape = tuple(int(d) for d in shp)
        self.threshold = float(threshold)
        self.start_frames = max(3, math.ceil(min_speech_ms / (FRAME_S * 1000)))
        self.silence_frames = max(1, math.ceil(endpoint_silence_ms / (FRAME_S * 1000)))
        self.preroll_frames = max(0, math.ceil(preroll_ms / (FRAME_S * 1000)))
        self.max_samples = int(max_segment_s * SR)
        self.last_prob = 0.0
        self._ring: deque[np.ndarray] = deque(maxlen=self.preroll_frames + self.start_frames)
        self._n_samples = 0
        self.reset()

    # -- state -----------------------------------------------------------
    def reset(self, keep_clock: bool = True) -> None:
        """drop the active segment and the recurrent state; keep_clock=False also zeroes media time."""
        self._active = False
        self._seg: list[np.ndarray] = []
        self._run_speech = 0
        self._run_silence = 0
        self._ring.clear()
        self._reset_rnn()
        if not keep_clock:
            self._n_samples = 0

    def _reset_rnn(self) -> None:
        if self._style == "hc":
            self._h = np.zeros((2, 1, 64), np.float32)
            self._c = np.zeros((2, 1, 64), np.float32)
        else:
            self._state = np.zeros(self._state_shape, np.float32)

    @property
    def t_media(self) -> float:
        return self._n_samples / SR

    @property
    def active(self) -> bool:
        return self._active

    # -- model -----------------------------------------------------------
    def _prob(self, frame: np.ndarray) -> float:
        x = frame[None, :]
        if self._style == "hc":
            p, self._h, self._c = self.sess.run(None, {"x": x, "h": self._h, "c": self._c})
        else:
            p, self._state = self.sess.run(None, {"input": x, "sr": np.array(SR, np.int64), "state": self._state})
        return float(p.reshape(-1)[0])

    # -- api -------------------------------------------------------------
    def push(self, frame_i16: np.ndarray) -> list[VadEvent]:
        """consume one 512-sample frame (int16, or float32 already in [-1, 1])."""
        if frame_i16.shape[0] != FRAME:
            raise ValueError(f"expected {FRAME} samples, got {frame_i16.shape[0]}")
        if frame_i16.dtype.kind == "f":
            frame = np.ascontiguousarray(frame_i16, dtype=np.float32)
        else:
            frame = frame_i16.astype(np.float32) / 32768.0
        p = self._prob(frame)
        self.last_prob = p
        self._n_samples += FRAME
        t_end = self._n_samples / SR
        events: list[VadEvent] = []
        if not self._active:
            self._ring.append(frame)
            self._run_speech = self._run_speech + 1 if p > self.threshold else 0
            if self._run_speech >= self.start_frames:
                n_take = min(len(self._ring), self._run_speech + self.preroll_frames)
                self._seg = list(self._ring)[-n_take:]
                self._ring.clear()
                self._active = True
                self._run_speech = 0
                self._run_silence = 0
                events.append(VadEvent("speech_start", max(0.0, t_end - n_take * FRAME_S), None, False))
            return events
        self._seg.append(frame)
        self._run_silence = self._run_silence + 1 if p < SILENCE_P else 0
        if self._run_silence >= self.silence_frames:
            events.append(self._finish(t_end - self._run_silence * FRAME_S, forced=False))
        elif len(self._seg) * FRAME >= self.max_samples:
            events.append(self._finish(t_end, forced=True))
        return events

    def speech_pcm(self) -> np.ndarray:
        """float32 of the active segment so far (empty when no segment is open)."""
        if not self._seg:
            return np.zeros(0, np.float32)
        return np.concatenate(self._seg)

    def _finish(self, t_speech_end: float, forced: bool) -> VadEvent:
        pcm = np.concatenate(self._seg)
        self._active = False
        self._seg = []
        self._run_silence = 0
        self._run_speech = 0
        self._ring.clear()
        self._reset_rnn()
        return VadEvent("speech_end", t_speech_end, pcm, forced)
