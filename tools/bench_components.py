#!/usr/bin/env python
"""per-module cpu latency at 1, 2 and 4 intra-op threads (design 1.5, 1.12).

builds the bench clips once from models/sensevoice/test_wavs/en.wav (7.15 s):
slices for 2, 3.3, 5 and 7.15 s, tilings for 10 and 15 s, written as 16 khz
pcm16 wavs under --clips. then, per thread count: session load times, and
p50/p95 over --runs timed calls (after one warm-up) of the fbank front end,
sensevoice (model and total), prosody, the text expert at 16/32/64 tokens,
silero vad per 512-sample frame (always 1 thread), each trained head, the
fixed fusion and discrepancy rules, and the response selector. model file
sizes and the machine go in the same json. every number is measured here.

usage: python tools/bench_components.py --threads 1,2,4 --clips bench_clips/ --out results/bench_components_$(hostname).json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meld_rt import protocol as P  # noqa: E402
from meld_rt.labels import EMOTIONS  # noqa: E402

EN_WAV = ROOT / "models/sensevoice/test_wavs/en.wav"
DURATIONS = (2.0, 3.3, 5.0, 7.15, 10.0, 15.0)
TOKEN_LENGTHS = (16, 32, 64)
MODEL_FILES = {"silero_vad.onnx": "silero_vad.onnx", "sensevoice/model.int8.onnx": "sensevoice/model.int8.onnx",
               "sensevoice/model_enc.int8.onnx": "sensevoice/model_enc.int8.onnx", "roberta-base-11.onnx": "roberta-base-11.onnx",
               "piper/en_US-lessac-medium.onnx": "piper/en_US-lessac-medium/en_US-lessac-medium.onnx"}
FLAGS_OF_INTEREST = ("avx2", "fma", "avx512f", "avx512bw", "avx512_vnni", "amx_tile", "amx_int8", "amx_bf16")
HEAD_FILES = ("text_ctx", "audio_cal", "fusion", "sentiment")
# a sentence long enough to yield 64 pieces; truncated to each token budget
LONG_TEXT = ("well that went great, the GR-6 system failed again and nobody in the control room noticed until the third "
             "alarm, so we spent the whole afternoon rewriting the checklist that everyone said was already finished "
             "and then the printer jammed twice before the meeting started and the coffee machine gave up too")


# -- machine ---------------------------------------------------------------

def cpu_info(cpuinfo: Path = Path("/proc/cpuinfo")) -> dict:
    """model name and the instruction flags that matter for int8 inference, from /proc/cpuinfo.
    without that file (windows, macos) the name and flags come from cpu_name_fallback and
    cpu_flags_fallback, and model_source and flags_source say so."""
    model, flags = "", set()
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = set(line.split(":", 1)[1].split())
    except OSError:
        model, source = cpu_name_fallback()
        return {"model_name": model, "model_source": source, "nproc": os.cpu_count(),
                "flags": cpu_flags_fallback(FLAGS_OF_INTEREST), "flags_source": "numpy"}
    return {"model_name": model, "nproc": os.cpu_count(), "flags": {f: (f in flags) for f in FLAGS_OF_INTEREST}}


def cpu_name_fallback(platform_name: str | None = None) -> tuple[str, str]:
    """(cpu name, where it came from) where /proc/cpuinfo and lscpu are missing: on windows the
    Name of Win32_Processor (cpython's _wmi module, which platform.processor() also queries),
    else and on failure platform.processor()."""
    if (platform_name or sys.platform) == "win32":
        try:
            import _wmi

            rows = _wmi.exec_query("SELECT Name FROM Win32_Processor").split("\0")
        except Exception:  # noqa: BLE001 - a label only, platform.processor() follows
            rows = []
        for item in rows:
            k, _, v = item.partition("=")
            if k == "Name" and v.strip():
                return v.strip(), "Win32_Processor"
    return platform.processor(), "platform.processor()"


# flags of interest -> the names in numpy's runtime cpu dispatch table (cpuid). the amx flags are not in it
NUMPY_FLAGS = {"avx2": "AVX2", "fma": "FMA3", "avx512f": "AVX512F", "avx512bw": "AVX512BW", "avx512_vnni": "AVX512VNNI"}


def cpu_flags_fallback(flags) -> dict:
    """flags where /proc/cpuinfo and lscpu are missing, from numpy's runtime cpu detection,
    None (unknown) for a flag numpy does not report."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
    except ImportError:
        feats = {}
    return {f: (bool(feats[NUMPY_FLAGS[f]]) if NUMPY_FLAGS.get(f) in feats else None) for f in flags}


def mem_fallback() -> dict:
    """MemTotal and MemAvailable in MB (2**20 bytes, as /proc/meminfo // 1024) from psutil."""
    import psutil

    vm = psutil.virtual_memory()
    return {"MemTotal": int(vm.total) // 2**20, "MemAvailable": int(vm.available) // 2**20}


def machine(meminfo: Path = Path("/proc/meminfo"), cpuinfo: Path = Path("/proc/cpuinfo")) -> dict:
    import onnxruntime as ort
    mem = {}
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0]) // 1024
    except OSError:
        mem = mem_fallback()
    except ValueError:
        pass
    return {"hostname": socket.gethostname(), "cpu": cpu_info(cpuinfo), "mem_mb": mem, "platform": platform.platform(),
            "python": platform.python_version(), "onnxruntime": ort.__version__, "numpy": np.__version__}


def loadavg() -> list[float]:
    """the 1, 5 and 15 minute load averages, [] where the os has none (windows)."""
    if not hasattr(os, "getloadavg"):
        return []
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


# -- clips -----------------------------------------------------------------

def trim_silence(x: np.ndarray, vad_model: Path, endpoint_silence_ms: int = 500) -> np.ndarray:
    """cut the source to what silero marks as speech (first speech_start to last
    speech_end, pre-roll included), so tilings of the result stay one vad segment:
    the source has 1.1 s of leading silence and room tone after the last word."""
    from meld_rt.experts.vad import VadSegmenter

    vad = VadSegmenter(vad_model, endpoint_silence_ms=endpoint_silence_ms, threads=1)
    frames = P.pcm16_frames(np.concatenate([P.to_pcm16(x), np.zeros(P.SR, np.int16)]))
    starts, ends = [], []
    for f in frames:
        for e in vad.push(f):
            (starts if e.kind == "speech_start" else ends).append(e.t_media)
    if not starts or not ends:
        return x
    return x[int(starts[0] * P.SR):int(ends[-1] * P.SR)]


def build_clips(src: Path, out_dir: Path, durations=DURATIONS, trim: bool = False,
                vad_model: Path = ROOT / "models/silero_vad.onnx") -> list[dict]:
    """slices (duration <= source) and tilings (longer) of the source wav, pcm16 16 khz;
    with trim the source is first cut to the span silero marks as speech."""
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != P.SR:
        raise ValueError(f"{src} is {sr} hz, need {P.SR}")
    if trim:
        x = trim_silence(x, vad_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in durations:
        n = round(d * P.SR)
        if n <= len(x):
            y, how = x[:n], "slice"
        else:
            reps = int(np.ceil(n / len(x)))
            y, how = np.tile(x, reps)[:n], "tile"
        name = f"{'en_dense' if trim else 'en'}_{d:g}s.wav"
        sf.write(str(out_dir / name), P.to_pcm16(y), P.SR, subtype="PCM_16")
        rows.append({"name": name, "file": str(out_dir / name), "duration_s": round(len(y) / P.SR, 3),
                     "source": how + (" of the trimmed source" if trim else ""), "label": f"{d:g}"})
    return rows


def load_pcm(path: str) -> np.ndarray:
    x, _ = sf.read(path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(x.mean(axis=1), dtype=np.float32)


# -- timing ----------------------------------------------------------------

def stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "min": round(float(a.min()), 3), "mean": round(float(a.mean()), 3), "n": int(a.size)}


def timed(fn, runs: int, warmup: int = 1) -> list[float]:
    """wall ms of each of `runs` calls after `warmup` untimed ones."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def load_timed(fn):
    t0 = time.perf_counter()
    obj = fn()
    return obj, round(time.perf_counter() - t0, 3)


def token_ids(text_expert, n: int) -> list[int]:
    """exactly n ids: <s>, n-2 pieces, </s>."""
    body = [i for i in text_expert.tokenize(LONG_TEXT) if i not in (0, 2)]
    # tokenize truncates at max_tokens; repeat the real pieces to reach longer budgets
    while len(body) < n - 2:
        body = body + body
    return [0] + body[:n - 2] + [2]


def bench_threads(threads: int, clips: list[dict], pcms: dict[str, np.ndarray], runs: int, models: Path, heads: Path,
                  log) -> dict:
    from meld_rt.discrepancy import discrepancy
    from meld_rt.experts import prosody as pros
    from meld_rt.experts.sensevoice import SenseVoiceExpert
    from meld_rt.experts.text import TextExpert
    from meld_rt.experts.vad import VadSegmenter
    from meld_rt.heads import (
        AudioCalibrator,
        Fusion,
        SentimentHead,
        TextContextHead,
        build_fusion_features,
        fuse_poe,
    )
    from meld_rt.response.selector import ResponseSelector
    from meld_rt.state import State, audio_emotion_dict, emotion7_dict

    res: dict = {"threads": threads, "load_s": {}, "loadavg_start": loadavg()}
    sv, t_sv = load_timed(lambda: SenseVoiceExpert(models / "sensevoice/model.int8.onnx", models / "sensevoice/tokens.txt",
                                                   threads=threads))
    te, t_te = load_timed(lambda: TextExpert(models / "roberta-base-11.onnx", ROOT / "assets/roberta_tokenizer", threads=threads))
    vad, t_vad = load_timed(lambda: VadSegmenter(models / "silero_vad.onnx", threads=1))
    sel, t_sel = load_timed(ResponseSelector)
    res["load_s"].update({"sensevoice": t_sv, "text": t_te, "vad": t_vad, "selector": t_sel})
    klass = {"text_ctx": TextContextHead, "audio_cal": AudioCalibrator, "fusion": Fusion, "sentiment": SentimentHead}
    t0 = time.perf_counter()
    hd = {name: klass[name].load(heads / f"{name}.npz") for name in HEAD_FILES if (heads / f"{name}.npz").exists()}
    res["load_s"]["heads"] = round(time.perf_counter() - t0, 3)
    res["heads_present"] = sorted(hd)
    # the first call of a session is slower; the warm-up pass mirrors the service
    sv.run(np.zeros(P.SR, np.float32))
    te.encode("hello")

    # front end, sensevoice model and total, prosody, per clip
    fe, model, total, decode, prosody, frames = {}, {}, {}, {}, {}, {}
    for c in clips:
        pcm = pcms[c["name"]]
        tm = []

        def one(pcm=pcm, tm=tm):
            tm.append(sv.run(pcm).timing_ms)

        timed(one, runs, warmup=1)
        tm = tm[1:]
        fe[c["label"]] = stats([t["frontend"] for t in tm])
        model[c["label"]] = stats([t["model"] for t in tm])
        decode[c["label"]] = stats([t["decode"] for t in tm])
        total[c["label"]] = stats([t["total"] for t in tm])
        r = sv.run(pcm)
        frames[c["label"]] = {"lfr_frames": r.n_frames, "tokens": len(r.text.split()), "tokens_per_s": round(r.tokens_per_s, 2)}
        prosody[c["label"]] = stats(timed(lambda pcm=pcm, r=r: pros.prosody_scalars(pcm, r.tokens_per_s), runs))
        log(f"  t={threads} {c['label']:>5} s: frontend {fe[c['label']]['p50']:.1f} sensevoice {model[c['label']]['p50']:.1f} "
            f"total {total[c['label']]['p50']:.1f} prosody {prosody[c['label']]['p50']:.1f} ms (p50)")
    res.update({"frontend_ms": fe, "sensevoice_ms": model, "sensevoice_decode_ms": decode, "sensevoice_total_ms": total,
                "prosody_ms": prosody, "clip_frames": frames, "prosody_backend": "parselmouth" if pros.HAVE_PARSELMOUTH else "none"})

    # text expert by token count
    text = {}
    for n in TOKEN_LENGTHS:
        ids = token_ids(te, n)
        text[str(n)] = stats(timed(lambda ids=ids: te.encode_ids(ids), runs))
    res["text_ms"] = text
    log(f"  t={threads} text 16/32/64: {text['16']['p50']:.1f} / {text['32']['p50']:.1f} / {text['64']['p50']:.1f} ms (p50)")

    # vad per frame, 1 thread, over the 2 s clip (62 frames, one warm-up frame)
    pcm16 = P.to_pcm16(pcms[clips[0]["name"]])
    fr = P.pcm16_frames(pcm16)
    vad.push(fr[0])
    vad_ms = []
    for f in fr[1:]:
        t0 = time.perf_counter()
        vad.push(f)
        vad_ms.append((time.perf_counter() - t0) * 1000)
    res["vad_frame_ms"] = stats(vad_ms)
    vad.reset(keep_clock=False)

    # heads and rules on a fixed synthetic input (numpy; thread count is irrelevant)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal(768).astype(np.float32)
    rows = max(runs, 200)
    hm = {}
    if "text_ctx" in hd:
        h = hd["text_ctx"]
        x = TextContextHead.features(emb, [], [], 3, h.k)
        hm["text_ctx"] = stats(timed(lambda: h.predict_log_proba(x), rows))
    if "sentiment" in hd:
        h = hd["sentiment"]
        if h.variant == "text_only":
            xs = TextContextHead.features(emb, [], [], 3, h.k)
        else:
            xs = rng.standard_normal(h.input_dim).astype(np.float32)
        hm["sentiment"] = stats(timed(lambda: h.predict_log_proba(xs), rows))
    p9 = rng.dirichlet(np.ones(9))
    p11 = rng.dirichlet(np.ones(11))
    if "audio_cal" in hd:
        h = hd["audio_cal"]
        xa = AudioCalibrator.features(p9, p11, np.zeros(4, np.float32), rng.standard_normal(1024).astype(np.float32) if h.uses_encoder_states else None)
        hm["audio_cal"] = stats(timed(lambda: h.predict_log_proba(xa), rows))
    pt = rng.dirichlet(np.ones(7))
    pa7 = np.asarray(fuse_poe(np.full(7, 1 / 7), p9))
    xf = build_fusion_features(np.log(pt), np.log(pa7), 0.9, 0.01, 0.1)
    if "fusion" in hd:
        hm["fusion"] = stats(timed(lambda: hd["fusion"].predict_log_proba(xf), rows))
    hm["fuse_poe"] = stats(timed(lambda: fuse_poe(pt, p9, 0.5), rows))
    hm["build_fusion_features"] = stats(timed(lambda: build_fusion_features(np.log(pt), np.log(pa7), 0.9, 0.01, 0.1), rows))
    hm["discrepancy"] = stats(timed(lambda: discrepancy(pt, p9, 0.2), rows))
    res["heads_ms"] = hm

    # selector on a fixed high-confidence state with a keyphrase
    st = State.empty("bench", 7)
    st.asr.text = "well that went great, the GR-6 system failed again"
    st.text.keyphrase = "the GR-6 system"
    st.audio.emotion = audio_emotion_dict(p9)
    st.text.emotion = emotion7_dict(pt)
    f = np.asarray(fuse_poe(pt, p9))
    st.fused.emotion, st.fused.label, st.fused.confidence = emotion7_dict(f), EMOTIONS[int(f.argmax())], 0.8
    st.fused.mode = "poe_uncalibrated"
    st.discrepancy.jsd, st.discrepancy.flag, st.discrepancy.direction = 0.3, True, "tone_more_negative"
    recent: list[str] = []
    res["selector_ms"] = stats(timed(lambda: sel.select(st, recent), rows))
    log(f"  t={threads} vad/frame {res['vad_frame_ms']['p50']:.3f} ms, heads {', '.join(f'{k} {v['p50']:.3f}' for k, v in hm.items())} ms, "
        f"selector {res['selector_ms']['p50']:.3f} ms (p50)")
    res["loadavg_end"] = loadavg()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="component latency bench at several thread counts")
    ap.add_argument("--threads", default="1,2,4", help="comma-separated intra-op thread counts")
    ap.add_argument("--clips", type=Path, default=ROOT / "bench_clips")
    ap.add_argument("--out", type=Path, default=ROOT / f"results/bench_components_{socket.gethostname()}.json")
    ap.add_argument("--runs", type=int, default=7, help="timed runs per measurement after one warm-up (>= 5)")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads")
    ap.add_argument("--source", type=Path, default=EN_WAV)
    ap.add_argument("--durations", default=",".join(f"{d:g}" for d in DURATIONS), help="clip lengths in seconds")
    ap.add_argument("--trim-silence", action="store_true", help="cut the source to the span silero marks as speech first")
    ap.add_argument("--build-only", action="store_true", help="write the clips and exit (for the replay bench)")
    a = ap.parse_args(argv)
    durations = tuple(float(d) for d in a.durations.split(",") if d.strip())
    if a.runs < 5:
        ap.error("--runs must be at least 5")
    threads = [int(t) for t in a.threads.split(",") if t.strip()]

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    t_start = time.perf_counter()
    clips = build_clips(a.source, a.clips, durations, trim=a.trim_silence, vad_model=a.models / "silero_vad.onnx")
    log(f"bench clips: {', '.join(f'{c['label']} s ({c['source']})' for c in clips)} from {a.source.name}")
    if a.build_only:
        print("\n".join(f"{c['file']} {c['duration_s']} s" for c in clips))
        return 0
    pcms = {c["name"]: load_pcm(c["file"]) for c in clips}
    out = {"host": socket.gethostname(), "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python tools/bench_components.py " + " ".join(sys.argv[1:] if argv is None else argv),
           "machine": machine(), "loadavg_start": loadavg(), "runs": a.runs, "warmup": 1,
           "source_wav": a.source.relative_to(ROOT).as_posix() if a.source.is_relative_to(ROOT) else a.source.as_posix(),
           "clips": [{k: v for k, v in c.items() if k != "file"} for c in clips],
           "model_files": {k: (a.models / v).stat().st_size if (a.models / v).exists() else None for k, v in MODEL_FILES.items()},
           "head_files": {p.name: p.stat().st_size for p in sorted(a.heads.glob("*.npz"))} if a.heads.exists() else {},
           "notes": ["sensevoice_ms is the onnxruntime call alone; sensevoice_total_ms adds the front end and the ctc decode",
                     "vad is always 1 thread (the ingest thread); heads, rules and selector are numpy and do not scale with threads",
                     "text_ms rows are by exact input length in pieces including <s> and </s>"],
           "by_threads": {}}
    for t in threads:
        log(f"threads={t}")
        out["by_threads"][str(t)] = bench_threads(t, clips, pcms, a.runs, a.models, a.heads, log)
        gc.collect()  # the sessions of this thread count are gone before the next ones load
    out["loadavg_end"] = loadavg()
    out["wall_s"] = round(time.perf_counter() - t_start, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    log(f"wrote {a.out} in {out['wall_s']} s")
    # the summary table, stdout
    print(f"sensevoice p50 ms by clip length (threads x seconds), {out['host']}:")
    print("threads | " + " | ".join(f"{c['label']} s" for c in clips))
    for t in threads:
        r = out["by_threads"][str(t)]
        print(f"{t} | " + " | ".join(f"{r['sensevoice_ms'][c['label']]['p50']:.0f}" for c in clips))
    print("frontend p50 ms: " + ", ".join(f"{c['label']} s {out['by_threads'][str(threads[-1])]['frontend_ms'][c['label']]['p50']:.1f}" for c in clips)
          + f" (threads={threads[-1]})")
    for t in threads:
        r = out["by_threads"][str(t)]
        print(f"threads={t}: text 16/32/64 tokens {r['text_ms']['16']['p50']:.1f}/{r['text_ms']['32']['p50']:.1f}/{r['text_ms']['64']['p50']:.1f} ms, "
              f"load sensevoice {r['load_s']['sensevoice']} s text {r['load_s']['text']} s vad {r['load_s']['vad']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
