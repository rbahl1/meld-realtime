"""per-utterance face evidence (extension spec 2.1). one set of functions used
offline, from cache/video_{split}.npz (tools/extract_video_features.py), and
live, from the service's ring of face rows, so training and run time compute
the same features.

- FaceRows: the face rows of a window (one row per face per processed frame)
  plus, optionally, the times of every processed frame (frame_t), faces or not.
  presence is counted against frame_t; without it, against the frames that have
  at least one face.
- audio_envelope / frame_log_rms: log rms per 32 ms audio frame (the protocol's
  512-sample frame), t at the frame centre.
- select_speaker: largest, present, asd (mouth motion against the envelope) or
  mouth (mouth minus face motion). asd and mouth fall back to largest when no
  track is eligible. the asd correlation of the chosen track is computed under
  every rule (it is a feature).
- face_evidence -> FaceEvidence over t0 <= t < t1; vision_features_A (16) and
  _B (528); build_fusion3_features (29).
- face_discrepancy: jsd and direction of words against the raw face posterior.
- offline: load_video_cache, clip_rows, load_envelopes (cached per split).
- live helpers: rows_from_faces, FaceRing, EnvelopeRing.

no cv2 or onnx import here; meld_rt.experts.vision only for its label mapping.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meld_rt.discrepancy import direction as _direction
from meld_rt.discrepancy import jsd as _jsd
from meld_rt.experts.vision import EMBED_DIM, posterior_meld
from meld_rt.heads import EPS, FUSION3_DIM, VISION_A_DIM, VISION_EMBED_DIM, entropy_bits
from meld_rt.labels import EMOTIONS

ROOT = Path(__file__).resolve().parent.parent
N_EMO = len(EMOTIONS)
SR = 16000
HOP_S = 0.032                  # one protocol audio frame (512 samples)
LOG_RMS_FLOOR = 1e-5
RULES = ("largest", "present", "asd", "mouth")
ASD_MIN_SAMPLES = 4
MOUTH_MIN_SAMPLES = 3
TRACK_SCALE = 4.0              # n_tracks / 4 in the features
MOUTH_SCALE = 10.0             # mouth motion * 10 in the features
DIM_A = VISION_A_DIM           # 16
DIM_B = DIM_A + EMBED_DIM      # 528
assert VISION_EMBED_DIM == EMBED_DIM and FUSION3_DIM == 3 * N_EMO + 8
FEATURES_A = ([f"log_p_face_{e}" for e in EMOTIONS]
              + ["max_p", "entropy_bits", "presence", "log_area", "n_tracks_div4", "asd_corr", "score", "mouth_x10", "present"])
FEATURES_FUSION3 = ([f"log_p_text_{e}" for e in EMOTIONS] + [f"log_p_audio_{e}" for e in EMOTIONS]
                    + [f"log_p_face_{e}" for e in EMOTIONS]
                    + ["H_text", "H_audio", "H_face", "asr_conf", "p_laughter", "p_unknown", "face_present", "presence"])
FACE_DIRECTIONS = ("none", "face_more_negative", "face_more_positive", "face_surprised", "mixed")


# ---------------------------------------------------------------- rows

def _nan_emb(n: int) -> np.ndarray:
    return np.full((n, EMBED_DIM), np.nan, dtype=np.float32)


@dataclass
class FaceRows:
    """face rows of one window. emb rows are nan where no embedding was computed."""
    t: np.ndarray          # (n,) media seconds
    track: np.ndarray      # (n,) int
    area: np.ndarray       # (n,) box w * h divided by frame w * h
    score: np.ndarray      # (n,) yunet score
    logits: np.ndarray     # (n, 7) fer order
    mouth: np.ndarray      # (n,) mouth motion, nan on a track's first frame
    face: np.ndarray       # (n,) face motion, nan on a track's first frame
    emb: np.ndarray        # (n, 512) float32
    frame_t: np.ndarray | None = None  # every processed frame's time, faces or not

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=np.float64).reshape(-1)
        n = len(self.t)
        self.track = np.asarray(self.track, dtype=np.int64).reshape(n)
        self.area = np.asarray(self.area, dtype=np.float64).reshape(n)
        self.score = np.asarray(self.score, dtype=np.float64).reshape(n)
        self.logits = np.asarray(self.logits, dtype=np.float64).reshape(n, 7)
        self.mouth = np.asarray(self.mouth, dtype=np.float64).reshape(n)
        self.face = np.asarray(self.face, dtype=np.float64).reshape(n)
        self.emb = _nan_emb(n) if self.emb is None else np.asarray(self.emb, dtype=np.float32).reshape(n, EMBED_DIM)
        if self.frame_t is not None:
            self.frame_t = np.asarray(self.frame_t, dtype=np.float64).reshape(-1)

    def __len__(self) -> int:
        return len(self.t)

    @classmethod
    def empty(cls, frame_t=None) -> FaceRows:
        return cls(np.zeros(0), np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros((0, 7)), np.zeros(0),
                   np.zeros(0), _nan_emb(0), None if frame_t is None else np.asarray(frame_t, dtype=np.float64))

    def take(self, idx) -> FaceRows:
        """rows idx (mask or indices); frame_t is kept as it is."""
        return FaceRows(self.t[idx], self.track[idx], self.area[idx], self.score[idx], self.logits[idx],
                        self.mouth[idx], self.face[idx], self.emb[idx], self.frame_t)

    def window(self, t0: float = -np.inf, t1: float = np.inf) -> FaceRows:
        """rows and frame times with t0 <= t < t1."""
        out = self.take((self.t >= t0) & (self.t < t1))
        if self.frame_t is not None:
            out.frame_t = self.frame_t[(self.frame_t >= t0) & (self.frame_t < t1)]
        return out

    def n_window_frames(self) -> int:
        """frames in the window: every processed frame when frame_t is known, else the frames with a face."""
        return len(np.unique(self.frame_t)) if self.frame_t is not None else len(np.unique(self.t))

    @classmethod
    def concat(cls, parts: list[FaceRows]) -> FaceRows:
        if not parts:
            return cls.empty()
        has_ft = any(p.frame_t is not None for p in parts)
        ft = np.concatenate([p.frame_t if p.frame_t is not None else np.unique(p.t) for p in parts]) if has_ft else None
        return cls(np.concatenate([p.t for p in parts]), np.concatenate([p.track for p in parts]),
                   np.concatenate([p.area for p in parts]), np.concatenate([p.score for p in parts]),
                   np.concatenate([p.logits for p in parts]), np.concatenate([p.mouth for p in parts]),
                   np.concatenate([p.face for p in parts]), np.concatenate([p.emb for p in parts]), ft)


def rows_from_faces(t: float, faces: list, frame_w: int, frame_h: int) -> FaceRows:
    """the rows of one processed frame from meld_rt.experts.vision.Face objects (box in pixels).
    frame_t is [t] even when there is no face, so presence counts the frame."""
    n = len(faces)
    wh = float(frame_w) * float(frame_h)
    emb = _nan_emb(n)
    for i, f in enumerate(faces):
        if f.embed is not None:
            emb[i] = np.asarray(f.embed, dtype=np.float32).reshape(EMBED_DIM)
    return FaceRows(np.full(n, float(t)), np.array([f.track for f in faces], dtype=np.int64),
                    np.array([float(f.box[2]) * float(f.box[3]) / wh for f in faces]),
                    np.array([f.score for f in faces], dtype=np.float64),
                    np.array([np.asarray(f.logits, dtype=np.float64).reshape(7) for f in faces]).reshape(n, 7),
                    np.array([f.mouth_motion for f in faces], dtype=np.float64),
                    np.array([f.face_motion for f in faces], dtype=np.float64), emb, np.array([float(t)]))


class FaceRing:
    """the last max_s seconds of processed frames of one stream (live)."""

    def __init__(self, max_s: float = 60.0):
        self.max_s = float(max_s)
        self._frames: deque[FaceRows] = deque()

    def add(self, t: float, faces: list, frame_w: int, frame_h: int) -> None:
        self._frames.append(rows_from_faces(t, faces, frame_w, frame_h))
        while self._frames and self._frames[0].frame_t[0] < float(t) - self.max_s:
            self._frames.popleft()

    @property
    def last_t(self) -> float | None:
        return float(self._frames[-1].frame_t[0]) if self._frames else None

    def rows(self, t0: float = -np.inf, t1: float = np.inf) -> FaceRows:
        parts = [f for f in self._frames if t0 <= f.frame_t[0] < t1]
        return FaceRows.concat(parts) if parts else FaceRows.empty(np.zeros(0))


# ---------------------------------------------------------------- audio envelope

def frame_log_rms(frame) -> float:
    """log rms of one audio frame; int16 is scaled by 1/32768 as protocol.pcm16_to_f32 does."""
    x = np.asarray(frame)
    x = x.astype(np.float64) / 32768.0 if x.dtype == np.int16 else x.astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    return float(np.log(max(rms, LOG_RMS_FLOOR)))


def audio_envelope(pcm_f32, sr: int = SR, hop_s: float = HOP_S) -> tuple[np.ndarray, np.ndarray]:
    """(t, log_rms) per hop_s frame, t at the frame centres. the last partial frame is
    zero-padded, as protocol.pcm16_frames pads it before sending."""
    x = np.asarray(pcm_f32, dtype=np.float64).reshape(-1)
    hop = int(round(hop_s * sr))
    n = int(np.ceil(len(x) / hop)) if len(x) else 0
    if n == 0:
        return np.zeros(0), np.zeros(0)
    x = np.concatenate([x, np.zeros(n * hop - len(x))]).reshape(n, hop)
    rms = np.sqrt(np.mean(x * x, axis=1))
    return (np.arange(n) + 0.5) * hop / sr, np.log(np.maximum(rms, LOG_RMS_FLOOR))


class EnvelopeRing:
    """log rms per received audio frame with the media time of its centre, last max_s seconds (live)."""

    def __init__(self, max_s: float = 60.0, sr: int = SR):
        self.max_s, self.sr = float(max_s), int(sr)
        self._t: deque[float] = deque()
        self._v: deque[float] = deque()

    def add_frame(self, t_start: float, frame) -> None:
        """t_start: media time of the frame's first sample."""
        tc = float(t_start) + len(np.asarray(frame)) / (2.0 * self.sr)
        self._t.append(tc)
        self._v.append(frame_log_rms(frame))
        while self._t and self._t[0] < tc - self.max_s:
            self._t.popleft()
            self._v.popleft()

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self._t, dtype=np.float64), np.array(self._v, dtype=np.float64)


# ---------------------------------------------------------------- speaker selection

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else float("nan")


def _track_stats(rows: FaceRows, env_t, env_v) -> dict[int, dict]:
    """per track: rows, mean area, asd correlation (nan when < ASD_MIN_SAMPLES motion samples
    or no envelope), mean mouth minus face motion and its sample count."""
    env_t = np.asarray(env_t if env_t is not None else [], dtype=np.float64)
    env_v = np.asarray(env_v if env_v is not None else [], dtype=np.float64)
    out = {}
    for tr in np.unique(rows.track):
        m = rows.track == tr
        mouth, t = rows.mouth[m], rows.t[m]
        ok = np.isfinite(mouth)
        corr = float("nan")
        if ok.sum() >= ASD_MIN_SAMPLES and len(env_t):
            corr = _pearson(mouth[ok], np.interp(t[ok], env_t, env_v))
        diff = mouth - rows.face[m]
        okd = np.isfinite(diff)
        out[int(tr)] = {"n": int(m.sum()), "area": float(rows.area[m].mean()), "corr": corr,
                        "mouth_minus_face": float(diff[okd].mean()) if okd.any() else float("nan"), "n_diff": int(okd.sum())}
    return out


def _best(cands: list[int], key) -> int:
    """the candidate with the largest key; ties to the smallest track id (cands are sorted)."""
    best = cands[0]
    for c in cands[1:]:
        if key(c) > key(best):
            best = c
    return best


def select_speaker(rows: FaceRows, env_t, env_v, rule: str) -> tuple[int | None, dict]:
    """(track or None, info) over all rows given (window them first). info: rule,
    asd_corr (the chosen track's correlation, nan when it has fewer than 4 motion
    samples or there is no envelope), fallback (asd or mouth had no eligible track
    and used largest), n_tracks."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    stats = _track_stats(rows, env_t, env_v)
    info = {"rule": rule, "asd_corr": float("nan"), "fallback": False, "n_tracks": len(stats)}
    if not stats:
        return None, info
    tracks = sorted(stats)
    largest = _best(tracks, lambda c: stats[c]["area"])
    if rule == "largest":
        chosen = largest
    elif rule == "present":
        chosen = _best(tracks, lambda c: (stats[c]["n"], stats[c]["area"]))
    elif rule == "asd":
        ok = [c for c in tracks if np.isfinite(stats[c]["corr"])]
        chosen = _best(ok, lambda c: (stats[c]["corr"], stats[c]["area"])) if ok else largest
        info["fallback"] = not ok
    else:
        ok = [c for c in tracks if stats[c]["n_diff"] >= MOUTH_MIN_SAMPLES]
        chosen = _best(ok, lambda c: (stats[c]["mouth_minus_face"], stats[c]["area"])) if ok else largest
        info["fallback"] = not ok
    info["asd_corr"] = stats[chosen]["corr"]
    return int(chosen), info


# ---------------------------------------------------------------- evidence and features

@dataclass
class FaceEvidence:
    present: bool
    track: int | None
    p_face: np.ndarray            # (7,) MELD order, mean per-frame fer posterior of the chosen track (uniform when absent)
    n_frames: int                 # chosen-track frames
    presence: float               # chosen-track frames / frames in the window
    log_area: float
    n_tracks: int
    asd_corr: float               # 0 when nan
    score: float
    mouth: float                  # 0 when nan
    emb: np.ndarray = field(repr=False, default=None)  # (512,) float32, zeros when none
    rule: str = "largest"
    n_window_frames: int = 0
    fallback: bool = False

    @classmethod
    def absent(cls, rule: str = "largest", n_window_frames: int = 0) -> FaceEvidence:
        return cls(False, None, np.full(N_EMO, 1.0 / N_EMO), 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0,
                   np.zeros(EMBED_DIM, np.float32), rule, int(n_window_frames), False)

    def to_dict(self) -> dict:
        """the scalar fields for a state or a log (no embedding)."""
        return {"present": bool(self.present), "track": self.track, "n_frames": int(self.n_frames),
                "presence": round(float(self.presence), 4), "n_tracks": int(self.n_tracks),
                "asd_corr": round(float(self.asd_corr), 4), "rule": self.rule}


def face_evidence(rows: FaceRows, env_t, env_v, rule: str, t0: float = -np.inf, t1: float = np.inf) -> FaceEvidence:
    """evidence over the frames with t0 <= t < t1 for the speaker chosen by the rule."""
    w = rows.window(t0, t1)
    n_win = w.n_window_frames()
    track, info = select_speaker(w, env_t, env_v, rule)
    if track is None:
        return FaceEvidence.absent(rule, n_win)
    m = w.track == track
    n = int(m.sum())
    p = posterior_meld(w.logits[m]).mean(axis=0)
    mouth = w.mouth[m]
    mouth = mouth[np.isfinite(mouth)]
    emb = w.emb[m]
    emb = emb[np.isfinite(emb).all(axis=1)]
    corr = info["asd_corr"]
    return FaceEvidence(
        present=True, track=int(track), p_face=p / p.sum(), n_frames=n, presence=float(n / max(n_win, n)),
        log_area=float(np.log(max(float(w.area[m].mean()), EPS))), n_tracks=int(info["n_tracks"]),
        asd_corr=float(corr) if np.isfinite(corr) else 0.0, score=float(w.score[m].mean()),
        mouth=float(mouth.mean()) if mouth.size else 0.0,
        emb=emb.mean(axis=0).astype(np.float32) if len(emb) else np.zeros(EMBED_DIM, np.float32),
        rule=rule, n_window_frames=n_win, fallback=bool(info["fallback"]))


def vision_features_A(ev: FaceEvidence) -> np.ndarray:
    """(16,) [log p_face (7), max p, entropy bits, presence, log_area, n_tracks / 4, asd_corr, score, mouth * 10, present]."""
    p = np.clip(np.asarray(ev.p_face, dtype=np.float64), EPS, 1.0)
    lp = np.log(p)
    tail = [float(p.max()), entropy_bits(lp), float(ev.presence), float(ev.log_area), ev.n_tracks / TRACK_SCALE,
            float(ev.asd_corr), float(ev.score), float(ev.mouth) * MOUTH_SCALE, float(bool(ev.present))]
    return np.concatenate([lp, tail]).astype(np.float32)


def vision_features_B(ev: FaceEvidence) -> np.ndarray:
    """(528,) A plus the chosen track's mean embedding."""
    emb = np.zeros(EMBED_DIM, np.float32) if ev.emb is None else np.asarray(ev.emb, dtype=np.float32).reshape(EMBED_DIM)
    return np.concatenate([vision_features_A(ev), emb]).astype(np.float32)


def vision_features(ev: FaceEvidence, variant: str) -> np.ndarray:
    if variant not in ("A", "B"):
        raise ValueError(f"variant must be A or B, got {variant!r}")
    return vision_features_A(ev) if variant == "A" else vision_features_B(ev)


def calibrated_face(head, ev: FaceEvidence) -> np.ndarray:
    """(7,) calibrated face posterior (heads.VisionCalibrator); uniform when no face is present."""
    if not ev.present:
        return np.full(N_EMO, 1.0 / N_EMO)
    x = vision_features_B(ev) if head.input_dim == DIM_B else vision_features_A(ev)
    return np.exp(head.predict_log_proba(x))


def build_fusion3_features(logp_text, logp_audio, logp_face, asr_conf: float, p_laughter: float, p_unknown: float,
                           face_present: bool, presence: float) -> np.ndarray:
    """(29,) [log p_text (7), log p_audio (7), log p_face (7), H_text, H_audio, H_face (bits), asr_conf,
    p_laughter, p_unknown, face_present, presence]. with no face, log p_face is log(1/7) and both flags are 0."""
    lt = np.asarray(logp_text, dtype=np.float64).reshape(-1)
    la = np.asarray(logp_audio, dtype=np.float64).reshape(-1)
    lf = np.asarray(logp_face, dtype=np.float64).reshape(-1)
    if lt.size != N_EMO or la.size != N_EMO or lf.size != N_EMO:
        raise ValueError("fusion3 needs three 7-way log-posteriors")
    if not face_present:
        lf, presence = np.full(N_EMO, np.log(1.0 / N_EMO)), 0.0
    tail = [entropy_bits(lt), entropy_bits(la), entropy_bits(lf), float(asr_conf), float(p_laughter), float(p_unknown),
            float(bool(face_present)), float(presence)]
    return np.concatenate([lt, la, lf, tail]).astype(np.float32)


def face_discrepancy(p_text, p_face_raw, threshold: float, present: bool = True) -> tuple[float, bool, str]:
    """(jsd, flag, direction) of the words against the raw face posterior (FaceEvidence.p_face),
    with the words-tone direction rule of meld_rt.discrepancy and face_* names. (0, False, "none")
    when no face is present."""
    if not present:
        return 0.0, False, "none"
    d = _jsd(p_text, p_face_raw)
    flag = bool(d > threshold)
    return d, flag, _direction(p_text, p_face_raw, flag).replace("tone_", "face_")


# ---------------------------------------------------------------- offline caches

def load_video_cache(split: str, cache_dir: Path | None = None) -> dict:
    """cache/video_{split}.npz (layout in tools/extract_video_features.py) as a dict of arrays,
    plus 'index' (key -> clip), 'row_order', 'row_start', 'row_end' (face rows of each clip) and
    'emb_of_row' (face row -> e_vec row or -1)."""
    path = Path(cache_dir if cache_dir is not None else ROOT / "cache") / f"video_{split}.npz"
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    keys = [str(k) for k in d["keys"]]
    d["index"] = {k: i for i, k in enumerate(keys)}
    order = np.argsort(d["f_clip"], kind="stable")
    d["row_order"] = order
    clips = d["f_clip"][order]
    n = len(keys)
    d["row_start"] = np.searchsorted(clips, np.arange(n), side="left")
    d["row_end"] = np.searchsorted(clips, np.arange(n), side="right")
    emb_of_row = np.full(len(d["f_clip"]), -1, dtype=np.int64)
    emb_of_row[d["e_row"]] = np.arange(len(d["e_row"]))
    d["emb_of_row"] = emb_of_row
    d["fps_value"] = float(d["fps"])
    d["meta_json"] = json.loads(str(d["meta"])) if "meta" in d else {}
    d["path"] = str(path)
    return d


def clip_rows(cache: dict, key: str) -> FaceRows:
    """FaceRows of one clip with frame times f_frame / fps and frame_t over all n_frames decoded
    frames. a clip that is not in the cache, or failed to decode, has no frames."""
    i = cache["index"].get(str(key))
    if i is None:
        return FaceRows.empty(np.zeros(0))
    fps = cache["fps_value"]
    frame_t = np.arange(int(cache["n_frames"][i])) / fps
    r = cache["row_order"][cache["row_start"][i]:cache["row_end"][i]]
    r = r[np.lexsort((cache["f_track"][r], cache["f_frame"][r]))]
    box = cache["f_box"][r].astype(np.float64)
    emb = _nan_emb(len(r))
    e = cache["emb_of_row"][r]
    has = e >= 0
    if has.any():
        emb[has] = cache["e_vec"][e[has]].astype(np.float32)
    return FaceRows(cache["f_frame"][r] / fps, cache["f_track"][r], box[:, 2] * box[:, 3], cache["f_score"][r],
                    cache["f_logits"][r], cache["f_mouth"][r], cache["f_face"][r], emb, frame_t)


def wav_stats(wav_paths: list[Path]) -> np.ndarray:
    """(n, 2) int64 size in bytes and mtime in ns of each wav, -1 -1 when it is missing."""
    out = np.full((len(wav_paths), 2), -1, dtype=np.int64)
    for i, p in enumerate(wav_paths):
        try:
            st = Path(p).stat()
        except FileNotFoundError:
            continue
        out[i] = st.st_size, st.st_mtime_ns
    return out


def load_envelopes(keys: list[str], wav_paths: list[Path], cache_path: Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """key -> (t, log_rms) of each clip wav (empty arrays when the wav is missing). with cache_path,
    the envelopes are read from that npz when it holds exactly these keys and the same wav sizes and
    mtimes (a wav that appeared, went away or was rewritten is read again), else computed and written there."""
    keys = [str(k) for k in keys]
    if len(wav_paths) != len(keys):
        raise ValueError(f"{len(keys)} keys and {len(wav_paths)} wav paths")
    stats = wav_stats(wav_paths)
    if cache_path is not None and Path(cache_path).exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if ([str(k) for k in z["keys"]] == keys and float(z["hop_s"]) == HOP_S and "wav_stat" in z.files
                    and np.array_equal(z["wav_stat"], stats)):
                off, t, v = z["offsets"], z["t"], z["v"]
                return {k: (t[off[i]:off[i + 1]], v[off[i]:off[i + 1]]) for i, k in enumerate(keys)}
    import soundfile as sf

    out = {}
    for k, p in zip(keys, wav_paths):
        p = Path(p)
        if not p.exists():
            out[k] = (np.zeros(0), np.zeros(0))
            continue
        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        out[k] = audio_envelope(x, sr)
    if cache_path is not None:
        lens = [len(out[k][0]) for k in keys]
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        cat = lambda j: np.concatenate([out[k][j] for k in keys]) if keys else np.zeros(0)  # noqa: E731
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.array(keys), offsets=off, t=cat(0), v=cat(1), hop_s=np.float64(HOP_S), wav_stat=stats)
    return out
