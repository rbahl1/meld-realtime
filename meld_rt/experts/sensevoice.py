"""sensevoice-small int8 through onnxruntime: transcript, asr confidence, language,
emotion and audio-event posteriors and (with the edited graph) pooled encoder
states, all from one forward pass.

front end mirrors sherpa-onnx: 80-dim fbank, 25/10 ms, hamming, dither 0,
snip_edges, samples scaled to int16 range (metadata normalize_samples=0), then
LFR(7, 6) with (7-1)//2 copies of the first frame as left padding and no tail
padding, then CMVN from the metadata neg_mean / inv_stddev. verified against
sherpa-onnx 1.13.8 on the five bundled test wavs (same text, lang, emotion, event).
logit rows: 0 language, 1 emotion, 2 event, 3 text norm, 4.. ctc over the text.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

from meld_rt import labels

EMO_TOKENS: list[str] = labels.SV_EMO_TOKENS
EVENT_TOKENS: list[str] = labels.SV_EVENT_TOKENS

SR = 16000
BLANK = 0
N_PREFIX_ROWS = 4
ENC_OUTPUT = "encoder_out"
MIN_SAMPLES = 1600  # shorter clips are zero-padded so the front end yields >= 2 lfr frames
TEXT_NORM_KEYS = {"woitn": "without_itn", "withitn": "with_itn"}


@dataclass
class SVResult:
    text: str
    asr_conf: float
    lang: str
    emo_probs: np.ndarray  # (9,) over EMO_TOKENS
    event_probs: np.ndarray  # (11,) over EVENT_TOKENS
    enc_pooled: np.ndarray | None  # (1024,) mean+max over speech frames, or None
    tokens_per_s: float
    n_frames: int  # lfr frames fed to the model (T)
    timing_ms: dict


def load_tokens(path: Path | str) -> list[str]:
    toks = []
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                toks.append(parts[0])
    return toks


def encoder_model_path(model_path: Path | str) -> Path:
    """models/sensevoice/model.int8.onnx -> models/sensevoice/model_enc.int8.onnx"""
    p = Path(model_path)
    stem = p.name.split(".", 1)
    return p.with_name(stem[0] + "_enc" + ("." + stem[1] if len(stem) > 1 else ""))


def _softmax(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64) - v.max()
    e = np.exp(v)
    return (e / e.sum()).astype(np.float32)


def apply_lfr(feats: np.ndarray, m: int, n: int) -> np.ndarray:
    """low frame rate stacking as sherpa-onnx does it: left pad (m-1)//2 copies of
    frame 0, windows of m frames every n frames, incomplete tail window dropped."""
    pad = (m - 1) // 2
    x = np.concatenate([np.repeat(feats[:1], pad, axis=0), feats], axis=0)
    t_out = (x.shape[0] - m) // n + 1
    if t_out <= 0:
        return np.zeros((0, m * feats.shape[1]), np.float32)
    idx = np.arange(m)[None, :] + n * np.arange(t_out)[:, None]
    return x[idx].reshape(t_out, m * feats.shape[1])


class SenseVoiceExpert:
    def __init__(self, model_path: Path | str, tokens_path: Path | str, threads: int = 4,
                 with_encoder_states: bool = False, text_norm: str = "woitn", language: str = "en"):
        path = Path(model_path)
        if with_encoder_states and "_enc" not in path.name and encoder_model_path(path).exists():
            path = encoder_model_path(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.model_path = path
        self.threads = threads
        outs = [o.name for o in self.sess.get_outputs()]
        self.with_encoder_states = with_encoder_states
        if with_encoder_states and ENC_OUTPUT not in outs:
            raise RuntimeError(f"{path} has no '{ENC_OUTPUT}' output; run tools/edit_sensevoice_graph.py first")
        self._out_names = ["logits"] + ([ENC_OUTPUT] if with_encoder_states else [])

        meta = self.sess.get_modelmeta().custom_metadata_map
        self.neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")], np.float32)
        self.inv_stddev = np.array([float(x) for x in meta["inv_stddev"].split(",")], np.float32)
        self.lfr_m = int(meta.get("lfr_window_size", 7))
        self.lfr_n = int(meta.get("lfr_window_shift", 6))
        self.normalize_samples = int(meta.get("normalize_samples", 0))
        self.lang_id = int(meta[f"lang_{language}"])
        self.text_norm_id = int(meta[TEXT_NORM_KEYS[text_norm]])
        self.language = language

        self.tokens = load_tokens(tokens_path)
        self.emo_ids = np.array([self.tokens.index(f"<|{t}|>") for t in EMO_TOKENS], np.int64)
        self.event_ids = np.array([self.tokens.index(f"<|{t}|>") for t in EVENT_TOKENS], np.int64)
        self.frame_shift_s = 0.01 * self.lfr_n

        self._fbank_opts = knf.FbankOptions()
        fo = self._fbank_opts.frame_opts
        fo.samp_freq = SR
        fo.dither = 0.0
        fo.window_type = "hamming"
        fo.snip_edges = True
        fo.frame_length_ms = 25.0
        fo.frame_shift_ms = 10.0
        mo = self._fbank_opts.mel_opts
        mo.num_bins = 80
        mo.low_freq = 20.0
        mo.high_freq = 0.0

    # -- front end -------------------------------------------------------
    def fbank(self, pcm_f32_16k: np.ndarray) -> np.ndarray:
        """(n_frames, 80) float32 log-mel fbank."""
        x = np.ascontiguousarray(pcm_f32_16k, dtype=np.float32)
        if x.shape[0] < MIN_SAMPLES:
            x = np.concatenate([x, np.zeros(MIN_SAMPLES - x.shape[0], np.float32)])
        if not self.normalize_samples:
            x = x * 32768.0
        fb = knf.OnlineFbank(self._fbank_opts)
        fb.accept_waveform(SR, x.tolist())
        fb.input_finished()
        n = fb.num_frames_ready
        if n == 0:
            return np.zeros((0, 80), np.float32)
        return np.stack([fb.get_frame(i) for i in range(n)]).astype(np.float32)

    def frontend(self, pcm_f32_16k: np.ndarray) -> np.ndarray:
        """(T, 560) float32: fbank -> lfr -> cmvn, the model input."""
        feats = apply_lfr(self.fbank(pcm_f32_16k), self.lfr_m, self.lfr_n)
        return (feats + self.neg_mean) * self.inv_stddev

    # -- model -----------------------------------------------------------
    def forward(self, feats: np.ndarray) -> list[np.ndarray]:
        """raw session outputs for (T, 560) features: [logits (1, T+4, vocab)[, encoder_out (1, T+4, 512)]]."""
        x = feats[None].astype(np.float32)
        return self.sess.run(self._out_names, {
            "x": x,
            "x_length": np.array([x.shape[1]], np.int32),
            "language": np.array([self.lang_id], np.int32),
            "text_norm": np.array([self.text_norm_id], np.int32),
        })

    def run(self, pcm_f32_16k: np.ndarray) -> SVResult:
        t0 = time.perf_counter()
        feats = self.frontend(pcm_f32_16k)
        t1 = time.perf_counter()
        outs = self.forward(feats)
        t2 = time.perf_counter()
        logits = outs[0][0]
        duration_s = max(len(pcm_f32_16k) / SR, 1e-3)

        lang_tok = self.tokens[int(logits[0].argmax())]
        lang = lang_tok[2:-2] if lang_tok.startswith("<|") and lang_tok.endswith("|>") else lang_tok
        emo_probs = _softmax(logits[1, self.emo_ids])
        event_probs = _softmax(logits[2, self.event_ids])

        text_rows = logits[N_PREFIX_ROWS:]
        ids = text_rows.argmax(-1)
        prev = np.concatenate([[BLANK], ids[:-1]])
        emit = np.flatnonzero((ids != BLANK) & (ids != prev))
        token_ids = ids[emit]
        text = "".join(self.tokens[i] for i in token_ids).replace("▁", " ").strip()
        if emit.size:
            rows = text_rows[emit].astype(np.float64)
            lse = np.log(np.exp(rows - rows.max(-1, keepdims=True)).sum(-1)) + rows.max(-1)
            asr_conf = float(np.mean(np.exp(rows.max(-1) - lse)))
        else:
            asr_conf = 0.0
        tokens_per_s = float(emit.size / duration_s)

        enc_pooled = None
        if self.with_encoder_states:
            enc = outs[1][0, N_PREFIX_ROWS:]
            if enc.shape[0] == 0:
                enc_pooled = np.zeros(2 * outs[1].shape[-1], np.float32)
            else:
                enc_pooled = np.concatenate([enc.mean(0), enc.max(0)]).astype(np.float32)
        t3 = time.perf_counter()
        timing = {"frontend": round((t1 - t0) * 1000, 2), "model": round((t2 - t1) * 1000, 2),
                  "decode": round((t3 - t2) * 1000, 2), "total": round((t3 - t0) * 1000, 2)}
        return SVResult(text=text, asr_conf=asr_conf, lang=lang, emo_probs=emo_probs, event_probs=event_probs,
                        enc_pooled=enc_pooled, tokens_per_s=tokens_per_s, n_frames=int(feats.shape[0]),
                        timing_ms=timing)
