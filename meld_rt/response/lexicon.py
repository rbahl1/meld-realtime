"""emotion words per MELD class, for the generator's logit mask and for
check_reply (spec 4). a word names a feeling: adjectives, nouns, verbs and
adverbs with their inflections, and the common words for showing one (crying,
pissed, freaked). neutral has none, so it can never be masked. a word sits in
one class only; the few that could go in two (upset, tense, hate, horror) are
put where the reply bank uses them or where the word is most often used.
"madness" is left out on purpose: it names insanity or chaos, not anger. words
are lowercase ascii letters only."""

from __future__ import annotations

import re

from meld_rt.labels import EMOTIONS

LEXICON: dict[str, tuple[str, ...]] = {
    "neutral": (),
    "surprise": (
        "surprise", "surprised", "surprises", "surprising", "surprisingly",
        "astonished", "astonishing", "astonishment", "amazed", "amazement",
        "shocked", "shock", "shocking", "stunned", "startled", "startling",
        "astounded", "dumbfounded", "flabbergasted",
    ),
    "fear": (
        "afraid", "fear", "fears", "feared", "fearful", "fearfully",
        "scared", "scary", "scare", "scares", "frightened", "frightening", "fright",
        "terrified", "terrifying", "terror", "anxious", "anxiously", "anxiety",
        "nervous", "nervously", "nervousness", "worried", "worry", "worries", "worrying",
        "panic", "panicked", "panicky", "panicking", "alarmed", "uneasy", "dread", "dreading",
        "tense", "apprehensive", "scaring", "terrify", "terrifies", "horror", "horrified",
        "horrify", "horrifying", "freaked",
    ),
    "sadness": (
        "sad", "sadder", "saddest", "sadly", "sadness", "unhappy", "unhappiness",
        "upset", "depressed", "depressing", "depression", "miserable", "misery",
        "heartbroken", "gloomy", "gloom", "sorrow", "sorrowful", "grief", "grieving",
        "disappointed", "disappointing", "disappointment", "lonely", "loneliness",
        "glum", "melancholy", "dejected", "tearful", "hurt", "hurting", "upsetting",
        "sadden", "saddened", "saddens", "saddening", "heartbreaking", "bummed",
        "cry", "cries", "cried", "crying",
    ),
    "joy": (
        "happy", "happier", "happiest", "happily", "happiness", "joy", "joyful",
        "joyous", "joyfully", "glad", "gladly", "pleased", "delighted", "delight",
        "delightful", "excited", "excitement", "thrilled", "cheerful", "cheery",
        "elated", "ecstatic", "overjoyed", "amused", "upbeat", "giddy", "proud",
        "exciting", "thrilling",
    ),
    "disgust": (
        "disgust", "disgusted", "disgusting", "disgustingly", "gross", "grossed",
        "revolted", "revolting", "revulsion", "repulsed", "repulsive", "repelled",
        "sickened", "nauseated", "appalled", "appalling", "distaste", "yuck", "eww",
        "disgusts",
    ),
    "anger": (
        "angry", "angrier", "angriest", "angrily", "anger", "angered", "annoyed",
        "annoying", "annoyance", "mad", "madder", "furious", "furiously", "fury",
        "irritated", "irritating", "irritation", "frustrated", "frustrating",
        "frustration", "rage", "raging", "enraged", "outraged", "outrage", "livid",
        "irate", "resentful", "resentment", "hostile", "hostility", "fuming",
        "angers", "annoy", "annoys", "frustrate", "frustrates", "irritate", "irritates",
        "infuriated", "infuriating", "pissed", "hate", "hates", "hated", "hating",
    ),
}

CLASS_OF: dict[str, str] = {w: c for c, ws in LEXICON.items() for w in ws}

_WORD = re.compile(r"[A-Za-z]+")


def words_of(classes) -> set[str]:
    """every lexicon word of the given classes."""
    return {w for c in classes for w in LEXICON[c]}


def forbidden_classes(allowed) -> list[str]:
    """the classes whose words a reply may not use, in EMOTIONS order."""
    allowed = set(allowed)
    unknown = allowed - set(EMOTIONS)
    if unknown:
        raise ValueError(f"unknown classes {sorted(unknown)}")
    return [c for c in EMOTIONS if c not in allowed and LEXICON[c]]


def emotion_words_in(text: str) -> list[tuple[str, str]]:
    """(word as written, class) for each lexicon word in the text, in order.
    a word is a maximal run of ascii letters, so "sadness," matches and
    "saddle" does not."""
    out = []
    for m in _WORD.finditer(text):
        c = CLASS_OF.get(m.group(0).lower())
        if c is not None:
            out.append((m.group(0), c))
    return out


def names_class(text: str, cls: str) -> bool:
    """the text uses a word of the class."""
    return any(c == cls for _, c in emotion_words_in(text))
