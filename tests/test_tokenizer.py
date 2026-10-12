"""the extracted roberta-base tokenizer files and the tokenizer built from them."""

import csv
import json
from pathlib import Path

import pytest

from meld_rt.experts.text import BOS_ID, EOS_ID, build_tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = ROOT / "assets" / "roberta_tokenizer"
MELD_DIR = ROOT / "meld"


@pytest.fixture(scope="module")
def vocab() -> dict:
    return json.loads((TOK_DIR / "vocab.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def merges() -> list[str]:
    return (TOK_DIR / "merges.txt").read_text(encoding="utf-8").splitlines()


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer(TOK_DIR)


def test_known_ids(vocab):
    assert vocab["<s>"] == 0
    assert vocab["<pad>"] == 1
    assert vocab["</s>"] == 2
    assert vocab["<unk>"] == 3
    assert vocab["Hello"] == 31414
    assert vocab["Ġworld"] == 232
    assert vocab["<mask>"] == 50264


def test_sizes(vocab, merges):
    assert len(vocab) == 50265
    assert sorted(vocab.values()) == list(range(50265))
    assert merges[0] == "#version: 0.2"
    assert len(merges) - 1 == 50000
    assert merges[1:4] == ["Ġ t", "Ġ a", "h e"]
    assert all(len(line.split(" ")) == 2 for line in merges[1:])


def test_encode_known_sentence(tok):
    enc = tok.encode("Hello world")
    assert enc.ids == [BOS_ID, 31414, 232, EOS_ID]
    assert enc.tokens == ["<s>", "Hello", "Ġworld", "</s>"]


def test_literal_special_tokens_are_text(tok):
    s = "I said <s> and <mask> here"
    ids = tok.encode(s).ids
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert BOS_ID not in ids[1:-1] and 50264 not in ids
    assert tok.decode(ids, skip_special_tokens=True) == s


def test_roundtrip_meld_sentences(tok):
    path = MELD_DIR / "dev_sent_emo.csv"
    if not path.exists():
        pytest.skip("meld csv not on disk")
    with path.open(encoding="utf-8", newline="") as f:
        sents = [row["Utterance"] for row in csv.DictReader(f)][:200]
    assert len(sents) == 200
    for s in sents:
        ids = tok.encode(s).ids
        assert ids[0] == BOS_ID and ids[-1] == EOS_ID
        assert tok.decode(ids, skip_special_tokens=True) == s
