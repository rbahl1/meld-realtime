from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from meld_rt import labels
from meld_rt.data import (
    MELD_DIR,
    Utt,
    class_weights,
    clip_path,
    context_rows,
    dialogues,
    emotion_counts,
    emotion_ids,
    key,
    load_split,
    parse_time,
    sentiment_ids,
)

COUNTS = {"train": 9989, "dev": 1109, "test": 2610}
DIALOGUES = {"train": 1038, "dev": 114, "test": 280}
PAIRS = {"train": 8951, "dev": 995, "test": 2330}
OVER_15_S = {"train": 25, "dev": 6, "test": 6}


@pytest.fixture(scope="module")
def splits() -> dict[str, list[Utt]]:
    return {s: load_split(s, MELD_DIR) for s in COUNTS}


def test_counts(splits):
    for s, n in COUNTS.items():
        assert len(splits[s]) == n
        assert len(dialogues(splits[s])) == DIALOGUES[s]
        assert len({u.dialogue_id for u in splits[s]}) == DIALOGUES[s]


def test_sorted_unique_and_split_tag(splits):
    for s, utts in splits.items():
        ids = [(u.dialogue_id, u.utterance_id) for u in utts]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)
        assert all(u.split == s for u in utts)
        assert all(u.emotion in labels.EMO_INDEX and u.sentiment in labels.SENT_INDEX for u in utts)


def test_parse_time():
    assert parse_time("00:16:16,059") == pytest.approx(16 * 60 + 16.059)
    assert parse_time("0:12:32,632") == pytest.approx(12 * 60 + 32.632)
    assert parse_time("01:00:00,000") == 3600.0


def test_text_kept_verbatim(splits):
    first = splits["train"][0]
    assert first.text == "also I was the point person on my company’s transition from the KL-5 to GR-6 system."
    assert first.speaker == "Chandler" and first.emotion == "neutral" and first.sentiment == "neutral"
    assert first.start_s == pytest.approx(976.059) and first.end_s == pytest.approx(981.731)
    assert sum("’" in u.text for u in splits["train"]) > 2000
    assert all(u.text.strip() for u in splits["train"])


def test_consecutive_pairs(splits):
    for s, n in PAIRS.items():
        rows = context_rows(splits[s], 1, False)
        assert sum(bool(c) for _, c in rows) == n
        assert n == COUNTS[s] - DIALOGUES[s]


def test_duration_anomalies(splits):
    nonpos = [(u.split, u.dialogue_id, u.utterance_id) for utts in splits.values() for u in utts if u.duration_s <= 0]
    assert nonpos == [("test", 155, 3)]
    longest = max(splits["test"], key=lambda u: u.duration_s)
    assert (longest.dialogue_id, longest.utterance_id) == (38, 4)
    assert longest.duration_s == pytest.approx(304.94, abs=0.01)
    assert max(u.duration_s for u in splits["train"]) < 60
    assert max(u.duration_s for u in splits["dev"]) < 60
    for s, n in OVER_15_S.items():
        assert sum(u.duration_s > 15 for u in splits[s]) == n


def test_key_and_clip_path(splits):
    u = splits["dev"][0]
    assert key(u) == "0_0"
    assert clip_path(u, Path("audio")) == Path("audio/dev/dia0_utt0.wav")
    u = Utt("test", 38, 4, "x", "y", "neutral", "neutral", 0.0, 1.0)
    assert key(u) == "38_4" and clip_path(u, Path("/a")) == Path("/a/test/dia38_utt4.wav")
    for s, utts in splits.items():
        assert len({key(u) for u in utts}) == COUNTS[s]


def test_context_rows_real(splits):
    utts = splits["train"]
    for k in (0, 1, 2):
        rows = context_rows(utts, k, False)
        assert [u for u, _ in rows] == utts
        pos = {(u.dialogue_id, u.utterance_id): i for i, u in enumerate(utts)}
        for u, ctx in rows:
            assert len(ctx) <= k
            assert all(c.dialogue_id == u.dialogue_id for c in ctx)
            assert all(c.utterance_id < u.utterance_id for c in ctx)
            assert [c.utterance_id for c in ctx] == sorted(c.utterance_id for c in ctx)
            # exactly the preceding rows of the file that belong to the same dialogue, at most k
            i = pos[(u.dialogue_id, u.utterance_id)]
            expected = [p for p in utts[max(i - k, 0):i] if p.dialogue_id == u.dialogue_id] if k else []
            assert ctx == expected
            assert all(pos[(c.dialogue_id, c.utterance_id)] < i for c in ctx)
    same = context_rows(utts, 2, True)
    assert all(c.speaker == u.speaker and c.dialogue_id == u.dialogue_id and c.utterance_id < u.utterance_id
               for u, ctx in same for c in ctx)
    assert sum(len(c) for _, c in same) < sum(len(c) for _, c in context_rows(utts, 2, False))


def test_context_rows_synthetic():
    def mk(d, i, spk):
        return Utt("dev", d, i, spk, f"t{d}{i}", "neutral", "neutral", 0.0, 1.0)

    a0, a1, a2, a3 = mk(0, 0, "A"), mk(0, 1, "B"), mk(0, 2, "A"), mk(0, 5, "B")
    b0, b1 = mk(1, 0, "A"), mk(1, 1, "A")
    utts = [a0, a1, a2, a3, b0, b1]
    rows = {key(u): c for u, c in context_rows(utts, 2, False)}
    assert rows["0_0"] == [] and rows["0_1"] == [a0] and rows["0_2"] == [a0, a1]
    assert rows["0_5"] == [a1, a2]  # gap in utterance ids still means previous rows
    assert rows["1_0"] == [] and rows["1_1"] == [b0]  # never crosses a dialogue boundary
    rows = {key(u): c for u, c in context_rows(utts, 2, True)}
    assert rows["0_1"] == [] and rows["0_2"] == [a0] and rows["0_5"] == [a1]
    assert rows["1_1"] == [b0]
    # unsorted input: previous still means earlier utterance_id, output keeps input order
    shuffled = [a3, b1, a1, b0, a2, a0]
    out = context_rows(shuffled, 1, False)
    assert [u for u, _ in out] == shuffled
    by_key = {key(u): c for u, c in out}
    assert by_key["0_5"] == [a2] and by_key["0_0"] == []
    # different splits with the same dialogue id never mix
    other = Utt("train", 0, 1, "A", "x", "neutral", "neutral", 0.0, 1.0)
    out = {key(u) + u.split: c for u, c in context_rows([a0, other], 1, False)}
    assert out["0_1train"] == []
    with pytest.raises(ValueError):
        context_rows(utts, -1, False)


def test_class_weights(splits):
    w = class_weights("meld_readme")
    assert w.shape == (7,) and w.dtype == np.float64
    assert w.tolist() == labels.MELD_README_WEIGHTS
    assert class_weights("none").tolist() == [1.0] * 7
    counts = emotion_counts(splits["train"])
    assert counts.tolist() == [4710, 1205, 268, 683, 1743, 271, 1109]
    b = class_weights("balanced", train=splits["train"])
    assert b.shape == (7,) and (b > 0).all()
    assert float((b * counts).sum()) == pytest.approx(9989.0)
    assert b.argmin() == labels.EMO_INDEX["neutral"] and b.argmax() == labels.EMO_INDEX["fear"]
    assert np.allclose(class_weights("balanced", meld_dir=MELD_DIR), b)
    with pytest.raises(ValueError):
        class_weights("uniform")


def test_label_ids(splits):
    utts = splits["dev"]
    e, s = emotion_ids(utts), sentiment_ids(utts)
    assert e.shape == s.shape == (1109,) and e.dtype == np.int64
    assert Counter(labels.EMOTIONS[i] for i in e) == Counter(u.emotion for u in utts)
    assert Counter(labels.SENTIMENTS[i] for i in s) == Counter(u.sentiment for u in utts)
    assert Counter(u.emotion for u in utts)["neutral"] == 470


def test_bad_split():
    with pytest.raises(ValueError):
        load_split("validation", MELD_DIR)
