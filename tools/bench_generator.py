#!/usr/bin/env python
"""bench of the streaming reply generator (spec 4) on MELD states.

states: n test clips with a cached audio clip, drawn by a seeded permutation;
the first half are gold-text states (the csv transcript, the text head on it,
fusion3 on gold text), the second half live-ASR states (the sensevoice
transcript with the service's asr gate, as live_asr_text in evaluate.py). a
state is the dict a vision-mode final would carry: transcript, text posterior,
raw sensevoice posterior and events, calibrated and raw face posteriors, fused
posterior, label and confidence, words-voice and words-face discrepancy,
degraded. it is built from cache/vision_post_{split}.npz (package A) and the
audio cache. until that file exists, --synthetic N builds n states from the
text and audio caches with the deployed two-way heads and no face (uniform
face, vision.present false, fused mode stacked).

for each model file and thread count: first-token and full-reply ms p50 / p95,
the first word's ms, the decode rate, words per reply, the check pass rate
(check_reply ok), the masked-step rate, rollbacks and their rework, the share
of replies that name the fused label's class, and 12 examples with their
evidence. writes results/generator_bench_<host>.json and prints the model the
choice rule of spec 4 picks.

choice rule (spec 4): the deployed model is the largest candidate whose
first-token p95 is at most 350 ms and whose decode rate is at least 15 tokens
per second at 4 threads on the idle container; among same-size files the int8
one unless q4 is faster by more than 20 % with a check pass rate no lower.
- the decode rate is decode_passes_per_s: forward passes after the first token
  (one per token, the pass that yields the end token and the passes of tokens a
  rollback undid included) over the time after the first token, so it measures
  the model and the loop, not how often the guards roll back. the rework is
  reported apart (rework_passes), and decode_tokens_per_s keeps the final-reply
  rate ((tokens - 1) over the same time) for comparison with earlier runs.
- "faster" is the cost of a reply of the same length at 4 threads: first-token
  p50 + n_ref passes at 1000 / decode_passes_per_s ms each, n_ref the mean
  tokens per reply of the two files pooled. q4 wins when cost(int8) / cost(q4)
  > 1.2, so a file that writes longer replies is not counted as slower.
- when no file meets both targets, the smallest model's file with the lower
  first-token p95 is named and meets_targets is false.
both definitions were set before the phase 2 bench (notes_C section 8).

usage:
  python tools/bench_generator.py                      # 200 states from cache/vision_post_test.npz
  python tools/bench_generator.py --synthetic 20       # before package A's file exists
  python tools/bench_generator.py --models Qwen2.5-0.5B-Instruct/model_int8.onnx --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from meld_rt import heads  # noqa: E402
from meld_rt.data import key, load_split  # noqa: E402
from meld_rt.discrepancy import discrepancy  # noqa: E402
from meld_rt.labels import EMOTIONS  # noqa: E402
from meld_rt.response import generator as G  # noqa: E402
from meld_rt.response.lexicon import LEXICON, names_class  # noqa: E402
from meld_rt.state import State, Vision, audio_emotion_dict, emotion7_dict, events_dict  # noqa: E402

FIRST_TOKEN_P95_MAX_MS = 350.0
DECODE_MIN_TPS = 15.0
Q4_SPEEDUP = 1.2
N_EXAMPLES = 12
PARROT_RUN = 5
UNIFORM = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS))


def pct(v, q) -> float | None:
    return None if len(v) == 0 else round(float(np.percentile(np.asarray(v, dtype=np.float64), q)), 2)


def _manifest(heads_dir: Path) -> dict:
    p = heads_dir / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def thresholds(manifest: dict, vision: bool) -> tuple[float, float]:
    """(low_conf, high_conf) the service uses: vision.thresholds with fusion3, else the core ones."""
    t = ((manifest.get("vision") or {}).get("thresholds") if vision else None) or manifest.get("thresholds") or {}
    return float(t.get("low_conf", 0.45)), float(t.get("high_conf", 0.70))


def make_state(i: int, said: str, asr_conf: float, degraded: list[str], p_text, emo9, ev11, fused, mode: str,
               jsd_flag: float, face: dict | None, face_flag_thr: float | None, rule: str = "") -> dict:
    """the dict of a vision-mode final state (service._run_final) from cached posteriors.
    face: None for no face, else p_face (calibrated), p_face_raw, track, n_frames, presence,
    n_tracks and asd_corr. rule is the speaker rule (empty without vision heads). a degraded asr gives the service's uniform fused posterior,
    the label neutral and a confidence of 0."""
    s = State.empty("bench", i, vision_mode=True)
    s.asr.text, s.asr.conf = said, round(float(asr_conf), 4)
    s.audio.emotion, s.audio.events = audio_emotion_dict(emo9), events_dict(ev11)
    ok = not degraded
    p_text = np.asarray(p_text, dtype=np.float64) if ok else UNIFORM
    s.text.emotion = emotion7_dict(p_text)
    f = np.asarray(fused, dtype=np.float64) if ok else UNIFORM
    j = int(np.argmax(f))
    s.fused.emotion, s.fused.mode, s.fused.calibrated = emotion7_dict(f), mode, True
    s.fused.label = EMOTIONS[j] if ok else "neutral"
    s.fused.confidence = round(float(f[j]), 4) if ok else 0.0
    s.degraded = list(degraded)
    if ok:
        d, flag, direction = discrepancy(p_text, emo9, jsd_flag)
        s.discrepancy.jsd, s.discrepancy.flag, s.discrepancy.direction = round(d, 4), flag, direction
    if face is None:
        s.vision = Vision(rule=rule)
        s.degraded.append("no_face")
    else:
        s.vision = Vision(emotion=emotion7_dict(face["p_face"]), raw=emotion7_dict(face["p_face_raw"]), present=True,
                          track=int(face["track"]), rule=rule, n_frames=int(face["n_frames"]),
                          presence=round(float(face["presence"]), 4), n_tracks=int(face["n_tracks"]),
                          asd_corr=round(float(face["asd_corr"]), 4))
        if ok and face_flag_thr is not None:
            from meld_rt.vision_features import face_discrepancy

            d, flag, direction = face_discrepancy(p_text, face["p_face_raw"], face_flag_thr, True)
            s.discrepancy.face_jsd, s.discrepancy.face_flag, s.discrepancy.face_direction = round(d, 4), flag, direction
    return s.to_dict()


def _gate(aud: dict) -> tuple[np.ndarray, np.ndarray]:
    from evaluate import asr_gate

    from meld_rt.service import ServiceConfig

    return asr_gate(aud["asr_text"], aud["asr_conf"], float(ServiceConfig().min_asr_conf))


def _draw(n_rows: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(row indices, is_live): a seeded permutation, first half gold, second half live."""
    order = np.random.default_rng(seed).permutation(n_rows)[:n]
    live = np.arange(len(order)) >= (len(order) + 1) // 2
    return order, live


def build_states(split: str, n: int, seed: int, synthetic: bool, cache: Path, heads_dir: Path, meld: Path,
                 live_ok: bool = True) -> tuple[list[dict], dict]:
    """(rows, info). each row: key, variant (gold | live), state dict."""
    from train_heads import audio_matrix, fusion_rows, load_text_cache, read_audio_cache, text_matrix

    utts = load_split(split, meld)
    manifest = _manifest(heads_dir)
    jsd_flag = float((manifest.get("thresholds") or {}).get("jsd_flag", 0.2))
    r = read_audio_cache(cache, split, utts)
    if r is None:
        raise SystemExit(f"cache/audio_{split}.npz is missing")
    aud, keep, _ = r
    failed, low = _gate(aud)
    order, live = _draw(len(keep), n, seed)
    if live_ok is False or split != "test":
        live[:] = False
    info = {"split": split, "seed": seed, "n": len(order), "n_gold": int((~live).sum()), "n_live": int(live.sum()),
            "jsd_flag": jsd_flag, "source": "synthetic" if synthetic else f"cache/vision_post_{split}.npz"}
    # how much the raw sensevoice posterior says on this split (the voice line reads the raw mass)
    named = aud["emo_probs"][:, :7]
    info["voice_raw_all_clips"] = {"n": len(named), "named_mass_p50": round(float(np.median(named.sum(1))), 4),
                                   "top_named_mass_p50": round(float(np.median(named.max(1))), 4),
                                   "share_top_named_mass_at_least_0.4": round(float((named.max(1) >= G.MODALITY_MIN_P).mean()), 4)}
    text_head = heads.TextContextHead.load(heads_dir / "text_ctx.npz")
    if synthetic:
        emb = load_text_cache(cache, split, utts)
        lp_text = text_head.predict_log_proba_batch(text_matrix(emb, utts, text_head.k))[keep]
        cal = heads.AudioCalibrator.load(heads_dir / "audio_cal.npz")
        fh = heads.Fusion.load(heads_dir / "fusion.npz")
        lp_audio = cal.predict_log_proba_batch(audio_matrix(aud, cal.uses_encoder_states))
        p_text, fused = np.exp(lp_text), np.exp(fh.predict_log_proba_batch(fusion_rows(lp_text, lp_audio, aud)))
        p_text_live = fused_live = None
        if live.any():
            lp_live = _live_text(cache, heads_dir, manifest, utts, aud, keep, text_head, lp_text)
            p_text_live = np.exp(lp_live)
            fused_live = np.exp(fh.predict_log_proba_batch(fusion_rows(lp_live, lp_audio, aud)))
        mode, face_thr, faces = "stacked", None, None
        info["thresholds"] = thresholds(manifest, False)
    else:
        path = cache / f"vision_post_{split}.npz"
        if not path.exists():
            raise SystemExit(f"{path} is missing: run tools/train_vision.py, or pass --synthetic N")
        with np.load(path, allow_pickle=False) as z:
            vp = {k: z[k] for k in z.files}
        meta = json.loads(str(vp["meta"]))
        pos = {str(k): i for i, k in enumerate(vp["keys"])}
        rows = np.array([pos[str(k)] for k in aud["keys"]], dtype=np.int64)
        p_text, fused = vp["p_text"][rows], vp["fused3"][rows]
        p_text_live = vp["p_text_live"][rows] if "p_text_live" in vp else None
        fused_live = vp["fused3_live"][rows] if "fused3_live" in vp else None
        if live.any() and (p_text_live is None or fused_live is None):
            raise SystemExit(f"{path.name} has no live-ASR fields")
        mode = "stacked_tav"
        face_thr = float((manifest.get("vision") or {}).get("jsd_face_flag", 1.0))
        faces = {f: vp[f][rows] for f in ("present", "p_face", "p_face_raw", "track", "n_frames", "presence", "n_tracks", "asd_corr")}
        faces["rule"] = meta.get("rule", "")
        info["thresholds"] = thresholds(manifest, True)
        info["vision_post"] = {"rule": meta.get("rule"), "variant": meta.get("variant"),
                               "face_share_in_sample": round(float(faces["present"][order].mean()), 4)}
        info["jsd_face_flag"] = face_thr

    by_key = {key(u): u for u in utts}
    out = []
    for i, (row, is_live) in enumerate(zip(order, live)):
        k = str(aud["keys"][row])
        if is_live:
            said, conf = str(aud["asr_text"][row]), float(aud["asr_conf"][row])
            degraded = ["asr_failed"] if failed[row] else (["low_asr_conf"] if low[row] else [])
            pt, pf = p_text_live[row], fused_live[row]
        else:
            said, conf, degraded = by_key[k].text, 1.0, []
            pt, pf = p_text[row], fused[row]
        face = None
        if faces is not None and bool(faces["present"][row]):
            face = {f: faces[f][row] for f in faces if f != "rule"}
        st = make_state(i + 1, said, conf, degraded, np.clip(pt, 0, None), aud["emo_probs"][row], aud["event_probs"][row],
                        np.clip(pf, 0, None), mode, jsd_flag, face, face_thr, faces["rule"] if faces else "")
        out.append({"key": k, "variant": "live" if is_live else "gold", "state": st})
    return out, info


def _live_text(cache, heads_dir, manifest, utts, aud, keep, text_head, lp_text):
    """the deployed text head on the sensevoice transcripts, through evaluate.Experiments (same
    embeddings file checks); gated rows are replaced by make_state."""
    from evaluate import Experiments

    zeros = np.zeros(len(keep), dtype=np.int64)     # labels and groups are not read here
    ex = Experiments(cache, heads_dir, manifest, utts, zeros, zeros, aud, keep, text_head, lp_text, None)
    lp = ex.lp_text_asr("")
    if lp is None:
        raise SystemExit("cache/text_asr_test.npy is missing (tools/embed_text.py --from-asr)")
    return lp


# ---------------------------------------------------------------- running

def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(t).lower())


def longest_shared_run(a: str, b: str) -> int:
    """the longest run of consecutive words two texts share."""
    x, y = _words(a), _words(b)
    best = 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def parrot(reply: str, said: str) -> dict:
    """a reply that repeats PARROT_RUN or more consecutive words of the transcript or of a
    few-shot example reply. not one of the checks; a quality count for prompt choices."""
    ex = max((longest_shared_run(reply, rep) for _, rep, _ in G.EXAMPLES), default=0)
    tr = longest_shared_run(reply, said)
    return {"transcript_run": tr, "example_run": ex, "transcript": tr >= PARROT_RUN, "example": ex >= PARROT_RUN}


def run_one(gen, rows: list[dict], low_conf: float, high_conf: float, examples_in: str,
            gap_style: str = G.GAP_STYLE) -> list[dict]:
    out = []
    for r in rows:
        msgs, grounding, allowed = G.build_messages(r["state"], low_conf, high_conf, examples_in=examples_in,
                                                    gap_style=gap_style)
        t0 = time.perf_counter()
        first_word = []

        def on_delta(_w, first_word=first_word, t0=t0):
            if not first_word:
                first_word.append((time.perf_counter() - t0) * 1000)

        res = gen.generate(msgs, allowed, on_delta)
        label = r["state"]["fused"]["label"]
        out.append({"key": r["key"], "variant": r["variant"], "text": res.text, "n_tokens": res.n_tokens,
                    "first_token_ms": round(res.first_token_ms, 2), "first_word_ms": round(first_word[0], 2) if first_word else None,
                    "total_ms": round(res.total_ms, 2), "stop_reason": res.stop_reason, "masked_steps": res.masked_steps,
                    "steps": res.steps, "backtracks": res.backtracks, "word_backtracks": res.word_backtracks,
                    "decode_passes": res.decode_passes, "rework_passes": res.rework_passes,
                    "register_masked_steps": res.register_masked_steps,
                    "prompt_tokens": res.prompt_tokens, "checks": G.check_reply(res.text, allowed), "allowed": allowed,
                    "grounding": grounding, "fused_label": label,
                    "names_fused": names_class(res.text, label) if LEXICON[label] else None,
                    "parrot": parrot(res.text, r["state"]["asr"]["text"]),
                    "evidence": msgs[-1]["content"]})
    return out


def passes_of(r: dict) -> int:
    """forward passes after the first token of one reply row. rows of earlier runs have no
    decode_passes: every step but a rolled-back one and the step that chose the end token
    ran one pass then, so it is derived from steps, backtracks and the stop reason."""
    if r.get("decode_passes") is not None:
        return int(r["decode_passes"])
    return max(int(r["steps"]) - int(r["backtracks"]) - int(r["stop_reason"] == "eos"), 0)


def summarize(rs: list[dict]) -> dict:
    ft = [r["first_token_ms"] for r in rs]
    tot = [r["total_ms"] for r in rs]
    fw = [r["first_word_ms"] for r in rs if r["first_word_ms"] is not None]
    dec_tok = sum(max(r["n_tokens"] - 1, 0) for r in rs)
    dec_steps = sum(max(r["steps"] - 1, 0) for r in rs)
    passes = sum(passes_of(r) for r in rs)
    rework = [r.get("rework_passes") for r in rs]
    dec_s = sum(max(r["total_ms"] - r["first_token_ms"], 0.0) for r in rs) / 1000
    per = [(r["n_tokens"] - 1) / ((r["total_ms"] - r["first_token_ms"]) / 1000) for r in rs
           if r["n_tokens"] > 1 and r["total_ms"] > r["first_token_ms"]]
    per_pass = [passes_of(r) / ((r["total_ms"] - r["first_token_ms"]) / 1000) for r in rs
                if passes_of(r) > 0 and r["total_ms"] > r["first_token_ms"]]
    words = [r["checks"]["n_words"] for r in rs]
    named = [r["names_fused"] for r in rs if r["names_fused"] is not None and r["allowed"]]
    fails: dict[str, int] = {}
    for r in rs:
        c = r["checks"]
        for k in ("over_length", "exclamation", "empty", "non_ascii", "non_printable"):
            fails[k] = fails.get(k, 0) + int(bool(c.get(k)))
        for k in ("forbidden_words", "meta_words", "prompt_echo"):
            fails[k] = fails.get(k, 0) + int(bool(c[k]))
    stops: dict[str, int] = {}
    for r in rs:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    steps = sum(r["steps"] for r in rs)
    return {"n": len(rs),
            "first_token_ms": {"p50": pct(ft, 50), "p95": pct(ft, 95)},
            "first_word_ms": {"p50": pct(fw, 50), "p95": pct(fw, 95), "n": len(fw)},
            "full_reply_ms": {"p50": pct(tot, 50), "p95": pct(tot, 95)},
            "decode_passes_per_s": round(passes / dec_s, 2) if dec_s > 0 else None,
            "decode_passes_per_s_p50": pct(per_pass, 50),
            "decode_tokens_per_s": round(dec_tok / dec_s, 2) if dec_s > 0 else None,
            "decode_tokens_per_s_p50": pct(per, 50),
            "decode_steps_per_s": round(dec_steps / dec_s, 2) if dec_s > 0 else None,
            "rework_passes": ({"n": int(sum(rework)), "share": round(sum(rework) / passes, 4) if passes else None}
                              if all(x is not None for x in rework) else None),
            "tokens_per_reply": {"mean": round(float(np.mean([r["n_tokens"] for r in rs])), 2), "max": max(r["n_tokens"] for r in rs)},
            "prompt_tokens": {"p50": pct([r["prompt_tokens"] for r in rs], 50), "max": max(r["prompt_tokens"] for r in rs)},
            "words_per_reply": {"mean": round(float(np.mean(words)), 2), "p50": pct(words, 50), "max": int(max(words))},
            "check_pass_rate": round(float(np.mean([r["checks"]["ok"] for r in rs])), 4),
            "parrot_rate": {"transcript": round(float(np.mean([r["parrot"]["transcript"] for r in rs])), 4),
                            "example": round(float(np.mean([r["parrot"]["example"] for r in rs])), 4),
                            "run_words": PARROT_RUN},
            "usable_rate": round(float(np.mean([r["checks"]["ok"] and not (r["parrot"]["transcript"] or r["parrot"]["example"])
                                                for r in rs])), 4),
            "check_failures": fails,
            "masked_step_rate": round(sum(r["masked_steps"] for r in rs) / steps, 4) if steps else None,
            "replies_with_masked_step": round(float(np.mean([r["masked_steps"] > 0 for r in rs])), 4),
            "backtracks": int(sum(r["backtracks"] for r in rs)),
            "word_backtracks": int(sum(r.get("word_backtracks") or 0 for r in rs)),
            "replies_with_backtrack": round(float(np.mean([r["backtracks"] > 0 for r in rs])), 4),
            "register_masked_step_rate": round(sum(r["register_masked_steps"] for r in rs) / steps, 4) if steps else None,
            "names_fused_class": {"share": round(float(np.mean(named)), 4) if named else None, "n": len(named),
                                  "of": "replies whose fused label is not neutral and whose allowed list is not empty"},
            "low_confidence_replies": int(sum(not r["allowed"] for r in rs)),
            "stop_reasons": stops}


def choose(results: dict) -> dict:
    """the choice rule of the module docstring, on the 4-thread results."""
    rows = []
    for name, by_t in results.items():
        r = by_t.get("4")
        if r is None:
            continue
        s = r["summary"]
        model, onnx = name.split("/")
        size = os.path.getsize(G.LLM_ROOT / model / "onnx" / onnx) if (G.LLM_ROOT / model / "onnx" / onnx).exists() else 0
        ok = (s["first_token_ms"]["p95"] is not None and s["first_token_ms"]["p95"] <= FIRST_TOKEN_P95_MAX_MS
              and (s["decode_passes_per_s"] or 0) >= DECODE_MIN_TPS)
        rows.append({"file": name, "model": model, "onnx": onnx, "params_order": _model_rank(model), "bytes": size,
                     "meets_targets": ok, "first_token_p95": s["first_token_ms"]["p95"], "first_token_p50": s["first_token_ms"]["p50"],
                     "decode_passes_per_s": s["decode_passes_per_s"], "tokens_per_reply": s["tokens_per_reply"]["mean"],
                     "full_reply_p50": s["full_reply_ms"]["p50"], "check_pass_rate": s["check_pass_rate"]})
    if not rows:
        return {"deployed": None, "reason": "no 4-thread results"}
    ok = [r for r in rows if r["meets_targets"]]
    if ok:
        top = max(r["params_order"] for r in ok)
        same = {r["onnx"]: r for r in ok if r["params_order"] == top}
        pick, why = _pick_file(same)
        meets = True
    else:
        top = min(r["params_order"] for r in rows)
        same = [r for r in rows if r["params_order"] == top]
        pick = min(same, key=lambda r: (r["first_token_p95"] if r["first_token_p95"] is not None else 1e9))
        why, meets = "no file meets both targets: the smallest model's file with the lower first-token p95", False
    return {"deployed": pick["file"], "meets_targets": meets, "reason": why, "candidates": rows,
            "edit": f'DEFAULT_MODEL = "{pick["model"]}", DEFAULT_ONNX = "{pick["onnx"]}" in meld_rt/response/generator.py',
            "rule": {"first_token_p95_max_ms": FIRST_TOKEN_P95_MAX_MS, "decode_min_passes_per_s": DECODE_MIN_TPS,
                     "q4_same_length_speedup_over_int8": Q4_SPEEDUP, "threads": 4}}


def _model_rank(model: str) -> int:
    return list(G.CANDIDATES).index(model) if model in G.CANDIDATES else -1


def reply_cost_ms(row: dict, n_ref: float) -> float:
    """ms of a reply of n_ref tokens: the first token at its p50, then n_ref forward passes
    at the file's decode rate (the last one yields the end token)."""
    return float(row["first_token_p50"] or 0.0) + n_ref * 1000.0 / max(float(row["decode_passes_per_s"] or 0.0), 1e-9)


def _pick_file(same: dict) -> tuple[dict, str]:
    i8, q4 = same.get("model_int8.onnx"), same.get("model_q4.onnx")
    if i8 and q4:
        n_ref = (float(i8["tokens_per_reply"]) + float(q4["tokens_per_reply"])) / 2
        faster = reply_cost_ms(i8, n_ref) / max(reply_cost_ms(q4, n_ref), 1e-9)
        if faster > Q4_SPEEDUP and q4["check_pass_rate"] >= i8["check_pass_rate"]:
            return q4, (f"q4 is {faster:.2f}x faster than int8 on a reply of {n_ref:.2f} tokens with a check pass rate"
                        " no lower")
        return i8, (f"int8 (q4 speedup {faster:.2f}x on a reply of {n_ref:.2f} tokens, q4 pass rate {q4['check_pass_rate']}"
                    f" vs {i8['check_pass_rate']})")
    only = i8 or q4
    return only, "the only file of the largest model that meets both targets"


def candidate_files(spec: list[str] | None) -> list[str]:
    if spec:
        return spec
    return [f"{m}/{f}" for m in G.CANDIDATES for f in G.CANDIDATE_FILES if (G.LLM_ROOT / m / "onnx" / f).exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="states (half gold text, half live asr)")
    ap.add_argument("--synthetic", type=int, default=None, metavar="N",
                    help="n states from the text and audio caches with no face (before cache/vision_post_test.npz exists)")
    ap.add_argument("--split", default="test", help="test (default); dev for prompt choices, gold text only")
    ap.add_argument("--models", nargs="*", help="<model>/<onnx file> under models/llm/ (default: every candidate on disk)")
    ap.add_argument("--threads", nargs="*", type=int, default=[2, 4])
    ap.add_argument("--examples-in", default=G.EXAMPLES_IN, choices=G.EXAMPLE_PLACES)
    ap.add_argument("--gap-style", default=G.GAP_STYLE, choices=G.GAP_STYLES,
                    help="voice and face lines when a direction against the words is flagged (notes_C section 8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads")
    ap.add_argument("--meld", type=Path, default=ROOT / "meld")
    ap.add_argument("--out", type=Path, default=ROOT / f"results/generator_bench_{socket.gethostname()}.json")
    a = ap.parse_args(argv)
    from bench_components import loadavg, machine

    synthetic = a.synthetic is not None
    n = a.synthetic if synthetic else a.n
    t_all = time.perf_counter()
    rows, info = build_states(a.split, n, a.seed, synthetic, a.cache, a.heads, a.meld)
    low_conf, high_conf = info["thresholds"]
    out = {"host": socket.gethostname(), "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python tools/bench_generator.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "machine": machine(), "loadavg_start": loadavg(), "states": info, "low_conf": low_conf, "high_conf": high_conf,
           "examples_in": a.examples_in, "gap_style": a.gap_style, "max_new_tokens": G.MAX_NEW_TOKENS, "results": {}}
    texts: dict[str, dict[str, list[str]]] = {}
    for name in candidate_files(a.models):
        model, onnx = name.split("/")
        out["results"][name] = {}
        for t in a.threads:
            gen = G.ReplyGenerator(G.LLM_ROOT / model, threads=t, onnx_file=onnx)
            t0 = time.perf_counter()
            msgs, _, allowed = G.build_messages(rows[0]["state"], low_conf, high_conf, examples_in=a.examples_in,
                                                gap_style=a.gap_style)
            gen.generate(msgs, allowed)   # warm-up: the first onnxruntime calls and the prefix cache
            warm_ms = (time.perf_counter() - t0) * 1000
            la0 = loadavg()
            rs = run_one(gen, rows, low_conf, high_conf, a.examples_in, a.gap_style)
            s = summarize(rs)
            texts.setdefault(name, {})[str(t)] = [r["text"] for r in rs]
            out["results"][name][str(t)] = {"summary": s, "load_ms": round(gen.load_ms, 1), "warmup_ms": round(warm_ms, 1),
                                            "loadavg_start": la0, "loadavg_end": loadavg(), "replies": rs}
            print(f"{name} t{t}: first token p50/p95 {s['first_token_ms']['p50']}/{s['first_token_ms']['p95']} ms, "
                  f"reply p50/p95 {s['full_reply_ms']['p50']}/{s['full_reply_ms']['p95']} ms, "
                  f"{s['decode_passes_per_s']} passes/s ({s['decode_tokens_per_s']} reply tok/s), pass {s['check_pass_rate']}, "
                  f"masked {s['masked_step_rate']}, "
                  f"names fused {s['names_fused_class']['share']}, load {la0}", flush=True)
            del gen
        by_t = texts.get(name, {})
        if len(by_t) > 1:
            ts = list(by_t)
            out["results"][name]["same_text_across_threads"] = round(
                float(np.mean([len({by_t[t][i] for t in ts}) == 1 for i in range(len(rows))])), 4)
    out["choice"] = choose(out["results"])
    ref = next((by_t for by_t in out["results"].values() if by_t), None)
    if ref:
        t_key = "4" if "4" in ref else next(k for k in ref if k.isdigit())
        out["examples"] = [{k: r[k] for k in ("key", "variant", "evidence", "grounding", "allowed", "text", "checks", "first_token_ms", "total_ms")}
                           | {"model": next(iter(out["results"])), "threads": int(t_key)} for r in ref[t_key]["replies"][:N_EXAMPLES]]
    out["loadavg_end"] = loadavg()
    out["wall_s"] = round(time.perf_counter() - t_all, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    c = out["choice"]
    print(f"choice: {c.get('deployed')} (meets targets: {c.get('meets_targets')}; {c.get('reason')})")
    if c.get("edit"):
        print(f"set {c['edit']}")
    print(f"wrote {a.out} in {out['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
