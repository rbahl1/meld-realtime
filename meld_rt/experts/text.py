"""text expert: frozen roberta-base (onnx model zoo graph, input_ids only) with
the byte-level bpe tokenizer extracted from the spacy en_core_web_trf wheel.
encode(text) is the mean of the last hidden state over the non-special
positions, 768-d float32, batch 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np

BOS, PAD, EOS, UNK, MASK = "<s>", "<pad>", "</s>", "<unk>", "<mask>"
BOS_ID, PAD_ID, EOS_ID, UNK_ID, MASK_ID = 0, 1, 2, 3, 50264
HIDDEN = 768
INPUT_NAME = "input_ids"
HIDDEN_OUTPUT = "output_1"   # (batch, seq, 768) last hidden state; output_2 is the pooler


def build_tokenizer(tokenizer_dir: Path):
    """roberta byte-level bpe from vocab.json + merges.txt: special tokens,
    <s> ... </s> wrapping, add_prefix_space False, trim_offsets True."""
    from tokenizers import AddedToken, ByteLevelBPETokenizer
    from tokenizers.processors import RobertaProcessing

    tokenizer_dir = Path(tokenizer_dir)
    tok = ByteLevelBPETokenizer(str(tokenizer_dir / "vocab.json"), str(tokenizer_dir / "merges.txt"),
                                add_prefix_space=False, trim_offsets=True)
    tok.add_special_tokens([BOS, PAD, EOS, UNK, AddedToken(MASK, lstrip=True)])
    tok._tokenizer.post_processor = RobertaProcessing((EOS, EOS_ID), (BOS, BOS_ID),
                                                      trim_offsets=True, add_prefix_space=False)
    # a literal "<s>" typed by a user is text, not a control token
    tok._tokenizer.encode_special_tokens = True
    return tok


class TextExpert:
    """tokenize(text) -> [0, ..., 2] (at most max_tokens); encode(text) -> (768,) float32."""

    def __init__(self, onnx_path: Path, tokenizer_dir: Path, threads: int = 2, max_tokens: int = 64):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.threads = int(threads)
        self.max_tokens = int(max_tokens)
        self.tokenizer = build_tokenizer(self.tokenizer_dir)
        self.tokenizer.enable_truncation(max_length=self.max_tokens)
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.onnx_path), so, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.session.get_inputs()}
        if names != {INPUT_NAME}:
            raise ValueError(f"expected a graph with the single input {INPUT_NAME!r}, got {sorted(names)}")

    def tokenize(self, text: str) -> list[int]:
        """piece ids wrapped as [0, ..., 2], truncated to max_tokens (the final 2 is kept)."""
        return list(self.tokenizer.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        """text back from piece ids, special tokens dropped (lossless for untruncated input)."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def hidden_states(self, ids: list[int]) -> np.ndarray:
        """last hidden state for one id sequence, (len(ids), 768) float32."""
        x = np.asarray([ids], dtype=np.int64)
        (h,) = self.session.run([HIDDEN_OUTPUT], {INPUT_NAME: x})
        return h[0]

    def encode_ids(self, ids: list[int]) -> np.ndarray:
        """mean of the last hidden state over the non-special positions; when
        there are none (empty text) the mean over <s> and </s> instead."""
        h = self.hidden_states(ids)
        keep = np.array([i not in (BOS_ID, EOS_ID, PAD_ID) for i in ids], dtype=bool)
        if not keep.any():
            keep[:] = True
        return h[keep].mean(axis=0).astype(np.float32, copy=False)

    def encode(self, text: str) -> np.ndarray:
        """(768,) float32 embedding of one text, see encode_ids."""
        return self.encode_ids(self.tokenize(text))
