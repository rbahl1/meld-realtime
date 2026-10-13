"""emotion lexicon of the reply generator (spec 4)."""

import re

import pytest

from meld_rt.labels import EMOTIONS
from meld_rt.response.lexicon import CLASS_OF, LEXICON, emotion_words_in, forbidden_classes, names_class, words_of


def test_every_class_and_neutral_has_no_words():
    assert list(LEXICON) == EMOTIONS
    assert LEXICON["neutral"] == ()
    for c in EMOTIONS[1:]:
        assert len(LEXICON[c]) >= 7, c


def test_words_are_lowercase_ascii_and_in_one_class_only():
    seen = {}
    for c, ws in LEXICON.items():
        assert len(set(ws)) == len(ws), c
        for w in ws:
            assert re.fullmatch(r"[a-z]+", w), w
            assert w not in seen, f"{w} in {seen.get(w)} and {c}"
            seen[w] = c
    assert seen == CLASS_OF


def test_spec_examples_and_inflections():
    # spec 4: "angry, anger, annoyed, mad, furious, irritated, frustrated for anger"
    assert {"angry", "anger", "annoyed", "mad", "furious", "irritated", "frustrated"} <= set(LEXICON["anger"])
    for c, w in [("joy", "happy"), ("joy", "happier"), ("sadness", "sad"), ("sadness", "sadness"), ("fear", "scared"),
                 ("fear", "afraid"), ("surprise", "surprised"), ("disgust", "disgusted")]:
        assert w in LEXICON[c]
    assert words_of(["joy", "neutral"]) == set(LEXICON["joy"])


def test_forbidden_classes():
    assert forbidden_classes([]) == [c for c in EMOTIONS if c != "neutral"]
    assert forbidden_classes(["neutral"]) == forbidden_classes([])
    assert "joy" not in forbidden_classes(["joy"]) and "anger" in forbidden_classes(["joy"])
    assert forbidden_classes(EMOTIONS) == []
    with pytest.raises(ValueError):
        forbidden_classes(["happiness"])


def test_emotion_words_in_matches_whole_words_any_case():
    found = emotion_words_in("You sound Sad, and ANGRY. A saddle is not sadness-free.")
    assert found == [("Sad", "sadness"), ("ANGRY", "anger"), ("sadness", "sadness")]
    assert emotion_words_in("The madness of the madder hatter") == [("madder", "anger")]
    assert emotion_words_in("") == []
    assert names_class("You look happy.", "joy") and not names_class("You look happy.", "anger")
    assert not names_class("anything", "neutral")
