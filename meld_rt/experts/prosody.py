"""four prosody scalars from praat (parselmouth): f0 median and p5-p95 range in
semitones re 100 hz over voiced frames, intensity std in db over frames within
30 db of the peak, and the speech rate in tokens/s passed in. zeros (except the
rate) when parselmouth is missing or the clip is too short for praat."""
from __future__ import annotations

import contextlib
import time

import numpy as np

try:
    import parselmouth  # type: ignore

    HAVE_PARSELMOUTH = True
except ImportError:  # pragma: no cover - optional dependency
    parselmouth = None
    HAVE_PARSELMOUTH = False

SR = 16000
MIN_S = 0.1  # praat needs a few pitch periods; shorter clips give zeros
PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0
REF_HZ = 100.0
INTENSITY_FLOOR_DB = 30.0

last_timing_ms: float = 0.0


def _semitones(f0_hz: np.ndarray) -> np.ndarray:
    return 12.0 * np.log2(f0_hz / REF_HZ)


def prosody_scalars(pcm_f32_16k: np.ndarray, tokens_per_s: float) -> np.ndarray:
    """(4,) float32: [f0_median_st, f0_range_st, intensity_std_db, tokens_per_s]."""
    global last_timing_ms
    t0 = time.perf_counter()
    out = np.zeros(4, np.float32)
    out[3] = float(tokens_per_s)
    if HAVE_PARSELMOUTH and len(pcm_f32_16k) >= int(MIN_S * SR):
        snd = parselmouth.Sound(np.asarray(pcm_f32_16k, dtype=np.float64), sampling_frequency=SR)
        # praat raises on sounds too short for its analysis windows; those stay at zero
        with contextlib.suppress(parselmouth.PraatError):
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
            f0 = pitch.selected_array["frequency"]
            voiced = f0[f0 > 0]
            if voiced.size >= 3:
                st = _semitones(voiced)
                out[0] = np.median(st)
                out[1] = np.percentile(st, 95) - np.percentile(st, 5)
        with contextlib.suppress(parselmouth.PraatError):
            inten = snd.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=0.01)
            v = inten.values[0]
            v = v[v > v.max() - INTENSITY_FLOOR_DB]
            if v.size >= 2:
                out[2] = v.std()
    last_timing_ms = (time.perf_counter() - t0) * 1000
    return out
