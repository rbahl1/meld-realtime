#!/usr/bin/env python
"""parameter count of everything on the inference path, recomputed from the
files on disk (design 1.5, 1.12): onnx initializer element counts per model
file (an int8 initializer counts its dequantized element count, one parameter
per element) plus the weight and bias arrays of the trained .npz heads. piper
is counted separately (offline tts for the proxy clips). writes results/params.md
and results/params.json and prints the table; nothing in them is typed.

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
    return {"rows": rows, "total_inference_path": total, "piper": piper,
            "total_with_piper": total + piper["params"] if piper else None,
            "heads_pending": [r["component"] for r in rows if r.get("params") is None and r["trained"] == "trained"],
            "other_head_files": extra, "cap": 6_000_000_000}


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
    if res["other_head_files"]:
        out += ["", "Other files in `heads/` (not on the inference path): " +
                ", ".join(f"`{e['file']}` ({fmt_m(e['params'])})" for e in res["other_head_files"]) + "."]
    out += ["", "Exact counts:", ""]
    for r in res["rows"] + ([p] if p else []):
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
