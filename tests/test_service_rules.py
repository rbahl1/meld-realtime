"""rules of meld_rt.service that need no model: the confidence the selector
buckets on (reported_confidence), on the posteriors of the committed en.wav demo."""

import numpy as np
import pytest

from meld_rt.heads import fuse_poe
from meld_rt.labels import EMOTIONS
from meld_rt.service import reported_confidence
from meld_rt.state import audio_emotion_vec9

# results/demo_log.jsonl before the fix: text joy 0.42, audio neutral 0.46 with unknown 0.53, product neutral 0.94
TEXT = {"neutral": 0.3467, "surprise": 0.0155, "fear": 0.0145, "sadness": 0.0184, "joy": 0.4229, "disgust": 0.0593, "anger": 0.1228}
AUDIO = {"neutral": 0.4634, "surprise": 0.0, "fear": 0.0, "sadness": 0.0064, "joy": 0.0009, "disgust": 0.0002, "anger": 0.0,
         "other": 0.0001, "unknown": 0.5289}


def test_uncalibrated_confidence_is_the_text_heads_max():
    p_text = np.array([TEXT[e] for e in EMOTIONS])
    p_text /= p_text.sum()
    fused = fuse_poe(p_text, np.array(audio_emotion_vec9(AUDIO)), 0.5)
    assert EMOTIONS[int(fused.argmax())] == "neutral" and fused.max() > 0.9
    assert reported_confidence("poe_uncalibrated", fused, p_text) == pytest.approx(p_text.max(), abs=1e-9)
    assert reported_confidence("poe_uncalibrated", fused, p_text) == pytest.approx(0.4229, abs=2e-4)
    for mode in ("stacked", "poe", "text_only"):
        assert reported_confidence(mode, fused, p_text) == pytest.approx(fused.max(), abs=1e-9)
    # no text head or text_failed: uniform text posterior -> 1/7, below any low_conf threshold
    assert reported_confidence("poe_uncalibrated", fused, np.full(7, 1 / 7)) == pytest.approx(1 / 7)
