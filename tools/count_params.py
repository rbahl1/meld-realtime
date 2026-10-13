#!/usr/bin/env python
"""parameter count of everything on the inference path, recomputed from the
files on disk (design 1.5, 1.12): onnx initializer element counts per model
file (an int8 initializer counts its dequantized element count, one parameter
per element) plus the weight and bias arrays of the trained .npz heads. piper
is counted separately (offline tts for the proxy clips). writes results/params.md
and results/params.json and prints the table; nothing in them is typed.

the extension (spec 4: --vision --reply llm) is a second table on top of the
core total: YuNet and FER (all initializers, as above), the vision heads, the
commit policy and the deployed LLM file (meld_rt.response.generator
DEFAULT_MODEL / DEFAULT_ONNX). the LLM is counted by weight tensors
(count_onnx_weights): every quantized weight once at its dequantized element
count (a 4-bit MatMulNBits weight counts K x N, not its packed bytes), float
weights as they are, and not the quantization scales and zero points, the
scalar constants, the integer shape constants or the exporter's constant
tables (names starting with "/", ex. the rotary cos / sin caches).

usage: python tools/count_params.py --models models/ --heads heads/ --out results/params.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (component, role, relative path under --models, trained) in the order of the 1.5 table
ONNX_ROWS = [
    ("Silero VAD v4", "end-pointing", "silero_vad.onnx", "frozen"),
    ("SenseVoice-small int8", "ASR + emotion + event posterior", "sensevoice/model.int8.onnx", "frozen"),
    ("RoBERTa-base encoder", "text embedding", "roberta-base-11.onnx", "frozen"),
]
PIPER_ROW = ("Piper en_US-lessac-medium", "TTS for the proxy clips (offline)", "piper/en_US-lessac-medium/en_US-lessac-medium.onnx", "frozen")
# (contract file name, component, role) for the heads of design 6.2 WP4
HEAD_ROWS = [
    ("text_ctx.npz", "TextContextHead", "text posterior with context"),
    ("audio_cal.npz", "AudioCalibrator", "audio posterior"),
    ("fusion.npz", "Fusion", "fused posterior"),
    ("sentiment.npz", "SentimentHead", "sentiment"),
]
# (component, role, relative path under --models, trained) of the extension (spec 4)
EXT_ONNX_ROWS = [
    ("YuNet 2023mar", "face detection (vision)", "vision/face_detection_yunet_2023mar.onnx", "frozen"),
    ("FER MobileFaceNet 2022july", "facial expression (vision)", "vision/facial_expression_recognition_mobilefacenet_2022july.onnx", "frozen"),
]
EXT_HEAD_ROWS = [
    ("vision_cal.npz", "VisionCalibrator", "face posterior"),
    ("fusion3.npz", "Fusion3", "fused posterior, three inputs"),
    ("sentiment3.npz", "Sentiment3", "sentiment, three inputs"),
    ("commit_policy.npz", "CommitPolicy", "when to commit the label"),
]
COMMIT_KEYS = ("p_w", "p_b", "qw_w", "qw_b", "qw_W1", "qw_b1", "qw_W2", "qw_b2")
# input positions that hold a scale or a zero point, per quantization op
QUANT_AUX_INPUTS = {"DequantizeLinear": (1, 2), "QuantizeLinear": (1, 2), "MatMulInteger": (2, 3),
                    "MatMulNBits": (2, 3), "DynamicQuantizeMatMul": (2, 3), "MatMulIntegerToFloat": (2, 3, 4, 5),
                    "GatherBlockQuantized": (2, 3)}
INT_TYPES = {onnx.TensorProto.INT64, onnx.TensorProto.INT32, onnx.TensorProto.INT16, onnx.TensorProto.BOOL}
WEIGHT_KEYS = ("W", "b", "W1", "b1", "W2", "b2")
STANDARDIZER_KEYS = ("mean", "scale")
INT8_TYPES = {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
PENDING = "pending (needs MELD audio)"


def _numel(dims) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def count_onnx(path: Path) -> dict:
    """element counts of the graph initializers of one onnx file, by dtype, plus
    the elements held by Constant nodes (reported, not added to the total)."""
    m = onnx.load(str(path), load_external_data=False)
    by_dtype: Counter = Counter()
    n_init = 0
    for init in m.graph.initializer:
        by_dtype[onnx.TensorProto.DataType.Name(init.data_type)] += _numel(init.dims)
        n_init += 1
    const = 0
    for node in m.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    const += _numel(a.t.dims)
    int8 = sum(v for k, v in by_dtype.items() if k in ("INT8", "UINT8"))
    total = int(sum(by_dtype.values()))
    return {"params": total, "n_initializers": n_init, "by_dtype": dict(by_dtype), "int8_elements": int(int8),
            "constant_node_elements": int(const), "bytes": path.stat().st_size,
            "opset": [(o.domain or "ai.onnx", int(o.version)) for o in m.opset_import]}


def count_onnx_weights(path: Path) -> dict:
    """weight tensors of a quantized LLM, each counted once (module docstring)."""
    m = onnx.load(str(path), load_external_data=False)
    aux: set[str] = set()
    nbits: dict[str, int] = {}
    for node in m.graph.node:
        for i in QUANT_AUX_INPUTS.get(node.op_type, ()):
            if i < len(node.input):
                aux.add(node.input[i])
        if node.op_type == "MatMulNBits" and len(node.input) > 1:
            at = {a.name: a.i for a in node.attribute}
            nbits[node.input[1]] = int(at["K"]) * int(at["N"])
    groups: Counter = Counter()
    by_dtype: Counter = Counter()
    for init in m.graph.initializer:
        n = _numel(init.dims)
        if init.name in aux or init.name.endswith(("_scale", "_scales", "_zero_point", "_zero_points")):
            groups["scales_zero_points"] += n
        elif init.data_type in INT_TYPES:
            groups["integer_constants"] += n
        elif len(init.dims) == 0:
            groups["scalar_constants"] += n
        elif init.name.startswith("/"):
            groups["exporter_constants"] += n
        else:
            k = nbits.get(init.name, n)
            groups["weights"] += k
            by_dtype[onnx.TensorProto.DataType.Name(init.data_type) + ("_4bit" if init.name in nbits else "")] += k
    return {"params": int(groups["weights"]), "by_dtype": dict(by_dtype), "excluded": {k: int(v) for k, v in groups.items() if k != "weights"},
            "n_initializers": len(m.graph.initializer), "bytes": path.stat().st_size}


def count_commit(path: Path) -> dict:
    """the learned arrays of heads/commit_policy.npz (meld_rt/commit.py): the P(correct)
    logistic weights and the Q_wait model. the standardizer is not counted."""
    with np.load(path, allow_pickle=False) as z:
        weights = {k: int(z[k].size) for k in COMMIT_KEYS if k in z}
        std = sum(int(z[k].size) for k in STANDARDIZER_KEYS if k in z)
        kind = str(z["qw_kind"]) if "qw_kind" in z else "?"
    return {"params": int(sum(weights.values())), "weights": weights, "standardizer_elements": std, "kind": kind,
            "bytes": path.stat().st_size}


def count_npz(path: Path) -> dict:
    """weights and biases of one head file; the standardizer (mean, scale) is
    reported next to them and not counted as parameters."""
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        weights = {k: int(z[k].size) for k in WEIGHT_KEYS if k in z}
        standardizer = sum(int(z[k].size) for k in STANDARDIZER_KEYS if k in z)
        kind = str(z["kind"]) if "kind" in z else "lr"
        k = int(z["k"]) if "k" in z and z["k"].shape == () else None
        variant = str(z["variant"]) if "variant" in z else "default"
        shapes = [tuple(int(d) for d in z[kk].shape) for kk in ("W", "W1", "W2") if kk in z]
    arch = "-".join(str(s[0]) for s in shapes) + (f"-{shapes[-1][1]}" if shapes else "")
    return {"params": int(sum(weights.values())), "weights": weights, "standardizer_elements": standardizer,
            "kind": kind, "k": k, "variant": variant, "arch": arch, "keys": keys, "bytes": path.stat().st_size}


def fmt_m(n: int | None) -> str:
    return "-" if n is None else f"{n / 1e6:.3f} M"


def build(models: Path, heads: Path) -> dict:
    rows = []
    total = 0
    for comp, role, rel, trained in ONNX_ROWS:
        p = models / rel
        if not p.exists():
            rows.append({"component": comp, "role": role, "file": rel, "trained": trained, "params": None, "note": "file missing"})
            continue
        c = count_onnx(p)
        total += c["params"]
        note = f"{c['n_initializers']} initializers"
        if c["int8_elements"]:
            note += f", {c['int8_elements'] / 1e6:.3f} M int8 elements counted dequantized"
        rows.append({"component": comp, "role": role, "file": rel, "trained": trained, "params": c["params"],
                     "bytes": c["bytes"], "by_dtype": c["by_dtype"], "constant_node_elements": c["constant_node_elements"],
                     "note": note})

    head_files = sorted(heads.glob("*.npz")) if heads.exists() else []
    on_path = set()
    for fname, comp, role in HEAD_ROWS:
        p = heads / fname
        if not p.exists():
            rows.append({"component": comp, "role": role, "file": f"{heads.name}/{fname}", "trained": "trained",
                         "params": None, "note": PENDING if head_files else "pending"})
            continue
        on_path.add(p.name)
        c = count_npz(p)
        total += c["params"]
        note = f"{c['kind']} {c['arch']}" + (f", k={c['k']}" if c["k"] is not None else "") + \
            (f", variant {c['variant']}" if c["variant"] != "default" else "") + \
            f"; standardizer {c['standardizer_elements']} elements not counted"
        rows.append({"component": comp, "role": role, "file": f"{heads.name}/{fname}", "trained": "trained",
                     "params": c["params"], "bytes": c["bytes"], "weights": c["weights"], "note": note})
    extra = []
    for p in head_files:
        if p.name not in on_path:
            c = count_npz(p)
            extra.append({"file": f"{heads.name}/{p.name}", "params": c["params"], "note": "not on the inference path"})
    rows.append({"component": "Prosody, discrepancy, selector", "role": "rules", "file": "-", "trained": "none", "params": 0, "note": "no learned parameters"})

    piper = None
    pp = models / PIPER_ROW[2]
    if pp.exists():
        c = count_onnx(pp)
        piper = {"component": PIPER_ROW[0], "role": PIPER_ROW[1], "file": PIPER_ROW[2], "trained": PIPER_ROW[3],
                 "params": c["params"], "bytes": c["bytes"], "by_dtype": c["by_dtype"],
                 "note": f"{c['n_initializers']} initializers, used by tools/make_proxy_clips.py only, not on the inference path"}
    ext = build_extension(models, heads)
    return {"rows": rows, "total_inference_path": total, "piper": piper,
            "total_with_piper": total + piper["params"] if piper else None,
            "heads_pending": [r["component"] for r in rows if r.get("params") is None and r["trained"] == "trained"],
            "other_head_files": [e for e in extra if Path(e["file"]).name not in {f for f, _, _ in EXT_HEAD_ROWS}],
            "extension_rows": ext["rows"], "llm_candidates": ext["llm_candidates"],
            "total_extension_only": ext["total"], "total_with_extension": total + ext["total"], "cap": 6_000_000_000}


def build_extension(models: Path, heads: Path) -> dict:
    """the rows of the extension table; missing files are listed with params None."""
    from meld_rt.response import generator as G

    rows, total = [], 0
    for comp, role, rel, trained in EXT_ONNX_ROWS:
        p = models / rel
        if not p.exists():
            rows.append({"component": comp, "role": role, "file": rel, "trained": trained, "params": None, "note": "file missing"})
            continue
        c = count_onnx(p)
        total += c["params"]
        rows.append({"component": comp, "role": role, "file": rel, "trained": trained, "params": c["params"], "bytes": c["bytes"],
                     "by_dtype": c["by_dtype"], "note": f"{c['n_initializers']} initializers"})
    for fname, comp, role in EXT_HEAD_ROWS:
        p = heads / fname
        if not p.exists():
            rows.append({"component": comp, "role": role, "file": f"{heads.name}/{fname}", "trained": "trained", "params": None,
                         "note": "pending"})
            continue
        if fname == "commit_policy.npz":
            c = count_commit(p)
            note = f"P(correct) logistic regression and Q_wait {c['kind']}; standardizer {c['standardizer_elements']} elements not counted"
        else:
            c = count_npz(p)
            note = f"{c['kind']} {c['arch']}" + (f", variant {c['variant']}" if c["variant"] != "default" else "") + \
                f"; standardizer {c['standardizer_elements']} elements not counted"
        total += c["params"]
        rows.append({"component": comp, "role": role, "file": f"{heads.name}/{fname}", "trained": "trained", "params": c["params"],
                     "bytes": c["bytes"], "weights": c["weights"], "note": note})
    rel = f"llm/{G.DEFAULT_MODEL}/onnx/{G.DEFAULT_ONNX}"
    p = models / rel
    comp, role = f"{G.DEFAULT_MODEL} ({G.DEFAULT_ONNX})", "streaming reply"
    if p.exists():
        c = count_onnx_weights(p)
        total += c["params"]
        ex = ", ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in sorted(c["excluded"].items()))
        rows.append({"component": comp, "role": role, "file": rel, "trained": "frozen", "params": c["params"], "bytes": c["bytes"],
                     "by_dtype": c["by_dtype"], "excluded": c["excluded"],
                     "note": f"weight tensors once, dequantized element count; not counted: {ex}"})
    else:
        rows.append({"component": comp, "role": role, "file": rel, "trained": "frozen", "params": None, "note": "file missing"})
    cands = []
    for model in G.CANDIDATES:
        for f in G.CANDIDATE_FILES:
            q = models / "llm" / model / "onnx" / f
            if q.exists() and q != p:
                c = count_onnx_weights(q)
                cands.append({"file": f"llm/{model}/onnx/{f}", "params": c["params"], "bytes": c["bytes"], "excluded": c["excluded"],
                              "note": "candidate, not deployed"})
    return {"rows": rows, "total": total, "llm_candidates": cands}


def render(res: dict, models: Path, heads: Path) -> str:
    out = ["# Parameter count", "",
           (f"Written by `python tools/count_params.py --models {models} --heads {heads}` on {res['generated_at']}, "
            "from the files on disk. An ONNX row is the sum of the element counts of the graph initializers. "
            "An int8 initializer counts one parameter per element, which is its dequantized count. "
            "A head row is the sum of the weight and bias arrays in its `.npz` file. "
            "The standardizer mean and scale are not parameters and are listed in the note. "
            "Tensors held by Constant nodes are not initializers and are not counted."),
           "", "| Component | Role | File | Params | Trained | Note |", "|---|---|---|---|---|---|"]
    for r in res["rows"]:
        f = "-" if r["file"] == "-" else f"`{r['file']}`"
        out.append(f"| {r['component']} | {r['role']} | {f} | {fmt_m(r['params'])} | {r['trained']} | {r['note']} |")
    out.append(f"| **Total on the inference path** | | | **{fmt_m(res['total_inference_path'])}** | | "
               f"{res['total_inference_path'] / res['cap'] * 100:.2f} % of the 6 B cap"
               + (f"; pending: {', '.join(res['heads_pending'])}" if res["heads_pending"] else "") + " |")
    p = res["piper"]
    if p:
        out.append(f"| {p['component']} | {p['role']} | `{p['file']}` | {fmt_m(p['params'])} | {p['trained']} | {p['note']} |")
        out.append(f"| Total including Piper | | | {fmt_m(res['total_with_piper'])} | | |")
    else:
        out.append("| Piper en_US-lessac-medium | TTS for the proxy clips (offline) | - | not on disk | | |")
    out += ["", "Extension (`--vision --reply llm`, spec 4), on top of the core total above:", "",
            "| Component | Role | File | Params | Trained | Note |", "|---|---|---|---|---|---|"]
    for r in res.get("extension_rows", []):
        out.append(f"| {r['component']} | {r['role']} | `{r['file']}` | {fmt_m(r['params'])} | {r['trained']} | {r['note']} |")
    if res.get("extension_rows"):
        pend = [r["component"] for r in res["extension_rows"] if r["params"] is None]
        out.append(f"| **Total with the extension** | | | **{fmt_m(res['total_with_extension'])}** | | "
                   f"{res['total_with_extension'] / res['cap'] * 100:.2f} % of the 6 B cap; extension alone "
                   f"{fmt_m(res['total_extension_only'])}" + (f"; pending: {', '.join(pend)}" if pend else "") + " |")
    if res.get("llm_candidates"):
        out += ["", "Other LLM candidates on disk (not deployed): " +
                ", ".join(f"`{c['file']}` ({fmt_m(c['params'])})" for c in res["llm_candidates"]) + "."]
    if res["other_head_files"]:
        out += ["", "Other files in `heads/` (not on the inference path): " +
                ", ".join(f"`{e['file']}` ({fmt_m(e['params'])})" for e in res["other_head_files"]) + "."]
    out += ["", "Exact counts:", ""]
    for r in res["rows"] + ([p] if p else []) + res.get("extension_rows", []):
        if r.get("params") is not None and r.get("file") != "-":
            dt = ", ".join(f"{k} {v:,}" for k, v in sorted(r.get("by_dtype", {}).items())) if r.get("by_dtype") else \
                ", ".join(f"{k} {v:,}" for k, v in r.get("weights", {}).items())
            out.append(f"- `{r['file']}`: {r['params']:,} ({dt}), {r.get('bytes', 0):,} bytes on disk")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="recompute the parameter table of design 1.5 from the files on disk")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--heads", type=Path, default=ROOT / "heads")
    ap.add_argument("--out", type=Path, default=ROOT / "results/params.md")
    ap.add_argument("--json", type=Path, default=None, help="default: --out with a .json suffix")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = build(a.models, a.heads)
    res["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    res["wall_s"] = round(time.perf_counter() - t0, 2)
    res["command"] = f"python tools/count_params.py --models {a.models} --heads {a.heads} --out {a.out}"
    md = render(res, a.models, a.heads)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    (a.json or a.out.with_suffix(".json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(md, end="")
    print(f"wrote {a.out} and {a.json or a.out.with_suffix('.json')} in {res['wall_s']} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
