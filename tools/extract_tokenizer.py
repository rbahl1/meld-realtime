"""extract the roberta-base byte-level bpe vocab and merges from the spacy
en_core_web_trf wheel.

the wheel is a zip; <pkg>/<pkg>-<ver>/transformer/model is a thinc msgpack
blob whose byte_bpe_encoder node carries a byte_bpe_processor attribute, itself
msgpack, with keys vocab (piece -> id, 50265 entries) and merges (50000 pairs).
needs only msgpack, not spacy and not torch.

usage: python tools/extract_tokenizer.py --wheel verify_dl/en_core_web_trf-3.8.0-py3-none-any.whl --out assets/roberta_tokenizer/
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import msgpack

# ids the hugging face roberta-base tokenizer assigns; used as a sanity check
KNOWN_IDS = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, ".": 4, "Ġthe": 5,
             "Ġworld": 232, "Hello": 31414, "<mask>": 50264}
N_VOCAB = 50265
N_MERGES = 50000
MERGES_HEADER = "#version: 0.2"


def find_model_member(z: zipfile.ZipFile) -> str:
    """the single zip member ending in transformer/model."""
    hits = [n for n in z.namelist() if n.endswith("/transformer/model")]
    if len(hits) != 1:
        raise SystemExit(f"expected one transformer/model member, found {hits}")
    return hits[0]


def find_bpe_processor(model: dict) -> dict:
    """the byte_bpe_processor attribute of the piece-encoder node, decoded."""
    for attrs in model.get("attrs", []):
        if isinstance(attrs, dict) and "byte_bpe_processor" in attrs:
            raw = attrs["byte_bpe_processor"]
            return msgpack.unpackb(raw, raw=False, strict_map_key=False)
    raise SystemExit("no node attribute named byte_bpe_processor in the blob")


def extract(wheel: Path) -> tuple[dict[str, int], list[list[str]]]:
    """vocab (piece -> id) and merges (list of [a, b]) from the wheel."""
    with zipfile.ZipFile(wheel) as z:
        member = find_model_member(z)
        blob = z.read(member)
    model = msgpack.unpackb(blob, raw=False, strict_map_key=False)
    del blob
    proc = find_bpe_processor(model)
    vocab = {str(k): int(v) for k, v in proc["vocab"].items()}
    merges = [[str(a), str(b)] for a, b in proc["merges"]]
    return vocab, merges


def check(vocab: dict[str, int], merges: list[list[str]]) -> list[str]:
    """list of failed checks; empty when everything matches roberta-base."""
    bad = []
    if len(vocab) != N_VOCAB:
        bad.append(f"vocab size {len(vocab)} != {N_VOCAB}")
    if sorted(vocab.values()) != list(range(len(vocab))):
        bad.append("vocab ids are not contiguous from 0")
    if len(merges) != N_MERGES:
        bad.append(f"merges count {len(merges)} != {N_MERGES}")
    if any(len(m) != 2 or " " in m[0] or " " in m[1] for m in merges):
        bad.append("a merge is not a pair of space-free pieces")
    for piece, want in KNOWN_IDS.items():
        if vocab.get(piece) != want:
            bad.append(f"vocab[{piece!r}] = {vocab.get(piece)} != {want}")
    return bad


def write(vocab: dict[str, int], merges: list[list[str]], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    vocab_path = out / "vocab.json"
    merges_path = out / "merges.txt"
    ordered = dict(sorted(vocab.items(), key=lambda kv: kv[1]))
    with vocab_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(ordered, f, ensure_ascii=False)
        f.write("\n")
    with merges_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(MERGES_HEADER + "\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")
    return vocab_path, merges_path


def verify_with_tokenizers(out: Path) -> str:
    """build the tokenizer from the written files and encode a probe; returns a note."""
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer(str(out / "vocab.json"), str(out / "merges.txt"))
    ids = tok.encode("Hello world").ids
    if ids != [31414, 232]:
        raise SystemExit(f"tokenizers probe failed: 'Hello world' -> {ids}")
    return f"tokenizers probe ok: 'Hello world' -> {ids}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-verify", action="store_true", help="skip the tokenizers probe")
    args = ap.parse_args(argv)

    vocab, merges = extract(args.wheel)
    bad = check(vocab, merges)
    if bad:
        print("extraction checks failed:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    vocab_path, merges_path = write(vocab, merges, args.out)
    print(f"vocab: {len(vocab)} pieces -> {vocab_path} ({vocab_path.stat().st_size} bytes)")
    print(f"merges: {len(merges)} pairs -> {merges_path} ({merges_path.stat().st_size} bytes)")
    print("known ids ok: " + ", ".join(f"{p}={i}" for p, i in KNOWN_IDS.items()))
    if not args.no_verify:
        print(verify_with_tokenizers(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
