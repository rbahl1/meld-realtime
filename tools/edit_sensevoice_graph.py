#!/usr/bin/env python
"""append the float input of the ctc projection as a second graph output named
encoder_out (N, T+4, 512), so the expert can read encoder states from the same
pass. walks the graph from the logits output back to the projection matmul and
takes its activation (the float tensor before the dynamic quantization)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TENSOR = "/encoder/tp_norm/Add_1_output_0"
ENC_OUTPUT = "encoder_out"
MATMUL_OPS = {"MatMul", "MatMulInteger", "Gemm", "QLinearMatMul", "MatMulIntegerToFloat", "DynamicQuantizeMatMul"}
QUANT_OPS = {"DynamicQuantizeLinear", "QuantizeLinear"}


def find_ctc_input(graph: onnx.GraphProto, output_name: str = "logits") -> tuple[str, int]:
    """name of the float activation feeding the final projection, and its hidden size."""
    prod = {o: n for n in graph.node for o in n.output}
    inits = {i.name: i for i in graph.initializer}
    node = prod[output_name]
    hops = 0
    while node.op_type not in MATMUL_OPS:
        nxt = [i for i in node.input if i in prod]
        if not nxt or hops > 20:
            raise RuntimeError(f"no projection matmul found above {output_name}")
        node = prod[nxt[0]]
        hops += 1
    act = node.input[0]
    weight = node.input[1]
    hidden = int(inits[weight].dims[0]) if weight in inits else -1
    if act in prod and prod[act].op_type in QUANT_OPS:
        act = prod[act].input[0]
    return act, hidden


def edit(src: Path, dst: Path, expected: str | None = EXPECTED_TENSOR) -> tuple[str, int]:
    model = onnx.load(str(src))
    g = model.graph
    if any(o.name == ENC_OUTPUT for o in g.output):
        raise RuntimeError(f"{src} already has an output named {ENC_OUTPUT}")
    act, hidden = find_ctc_input(g)
    if expected and act != expected:
        print(f"warning: ctc input is {act}, design named {expected}", file=sys.stderr)
    g.node.append(onnx.helper.make_node("Identity", [act], [ENC_OUTPUT], name="/encoder_out_identity"))
    g.output.append(onnx.helper.make_tensor_value_info(ENC_OUTPUT, onnx.TensorProto.FLOAT, ["N", "T", hidden]))
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return act, hidden


def verify(src: Path, dst: Path, wav: Path, tokens: Path, threads: int = 2) -> dict:
    """run both graphs on one wav; returns max |logits diff| and the encoder_out shape."""
    sys.path.insert(0, str(ROOT))
    import soundfile as sf

    from meld_rt.experts.sensevoice import SenseVoiceExpert

    pcm, sr = sf.read(str(wav), dtype="float32")
    assert sr == 16000, sr
    a = SenseVoiceExpert(src, tokens, threads=threads)
    b = SenseVoiceExpert(dst, tokens, threads=threads, with_encoder_states=True)
    feats = a.frontend(pcm)
    la = a.forward(feats)[0]
    lb, enc = b.forward(feats)
    return {"max_abs_logit_diff": float(np.abs(la - lb).max()), "logits_shape": list(la.shape),
            "encoder_out_shape": list(enc.shape), "n_lfr_frames": int(feats.shape[0])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=Path, default=ROOT / "models/sensevoice/model.int8.onnx")
    ap.add_argument("--out", dest="dst", type=Path, default=ROOT / "models/sensevoice/model_enc.int8.onnx")
    ap.add_argument("--tokens", type=Path, default=ROOT / "models/sensevoice/tokens.txt")
    ap.add_argument("--verify-wav", type=Path, default=ROOT / "models/sensevoice/test_wavs/en.wav")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    act, hidden = edit(args.src, args.dst)
    print(f"appended {ENC_OUTPUT} <- {act} (hidden {hidden}) -> {args.dst} in {time.perf_counter() - t0:.1f} s")
    if args.no_verify or not args.verify_wav.exists():
        return 0
    v = verify(args.src, args.dst, args.verify_wav, args.tokens)
    print(f"verify on {args.verify_wav.name}: max |logits diff| {v['max_abs_logit_diff']:.2e}, "
          f"logits {v['logits_shape']}, {ENC_OUTPUT} {v['encoder_out_shape']}, lfr frames {v['n_lfr_frames']}")
    ok = v["max_abs_logit_diff"] <= args.tol and v["encoder_out_shape"] == [1, v["n_lfr_frames"] + 4, hidden]
    print("ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
