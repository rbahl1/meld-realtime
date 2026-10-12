# Parameter count (recomputed from the files on disk)

Written by `python tools/count_params.py --models models --heads heads` on 2026-10-12T04:58:09Z. ONNX rows sum the element counts of the graph initializers (an int8 initializer counts one parameter per element, its dequantized count); head rows sum the weight and bias arrays of the `.npz` files (the standardizer mean and scale are not parameters and are listed in the note). Constant-node tensors are not initializers and are not counted.

| Component | Role | File | Params | Trained | Note |
|---|---|---|---|---|---|
| Silero VAD v4 | end-pointing | `silero_vad.onnx` | 0.156 M | frozen | 40 initializers |
| SenseVoice-small int8 | ASR + emotion + event posterior | `sensevoice/model.int8.onnx` | 234.000 M | frozen | 1479 initializers, 233.103 M int8 elements counted dequantized |
| RoBERTa-base encoder | text embedding | `roberta-base-11.onnx` | 124.646 M | frozen | 283 initializers |
| TextContextHead | text posterior with context | `heads/text_ctx.npz` | 0.200 M | trained | mlp 772-256-7, k=0, variant full_dialogue; standardizer 1544 elements not counted |
| AudioCalibrator | audio posterior | `heads/audio_cal.npz` | - | trained | pending (needs MELD audio) |
| Fusion | fused posterior | `heads/fusion.npz` | - | trained | pending (needs MELD audio) |
| SentimentHead | sentiment | `heads/sentiment.npz` | 0.002 M | trained | lr 772-3, k=0, variant text_only; standardizer 1544 elements not counted |
| Prosody, discrepancy, selector | rules | - | 0.000 M | none | no learned parameters |
| **Total on the inference path** | | | **359.003 M** | | 5.98 % of the 6 B cap; pending: AudioCalibrator, Fusion |
| Piper en_US-lessac-medium | TTS (optional spoken reply) | `piper/en_US-lessac-medium/en_US-lessac-medium.onnx` | 15.651 M | frozen | 396 initializers; optional, not on the measured path |
| Total with Piper enabled | | | 374.654 M | | |

Other files in `heads/` (not on the inference path): `heads/text_ctx_k0.npz` (0.200 M).

Exact counts:

- `silero_vad.onnx`: 155,908 (FLOAT 155,908); 643,854 bytes on disk
- `sensevoice/model.int8.onnx`: 233,999,729 (FLOAT 896,600, UINT8 233,103,129); 239,233,841 bytes on disk
- `roberta-base-11.onnx`: 124,645,716 (FLOAT 124,645,632, INT64 84); 498,649,858 bytes on disk
- `heads/text_ctx.npz`: 199,687 (W1 197,632, b1 256, W2 1,792, b2 7); 808,726 bytes on disk
- `heads/sentiment.npz`: 2,319 (W 2,316, b 3); 18,328 bytes on disk
- `piper/en_US-lessac-medium/en_US-lessac-medium.onnx`: 15,650,546 (FLOAT 15,650,472, INT32 1, INT64 73); 63,149,198 bytes on disk
