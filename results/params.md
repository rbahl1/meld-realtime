# Parameter count

Written by `python tools/count_params.py --models models --heads heads` on 2026-10-12T20:58:13Z, from the files on disk. An ONNX row is the sum of the element counts of the graph initializers. An int8 initializer counts one parameter per element, which is its dequantized count. A head row is the sum of the weight and bias arrays in its `.npz` file. The standardizer mean and scale are not parameters and are listed in the note. Tensors held by Constant nodes are not initializers and are not counted.

| Component | Role | File | Params | Trained | Note |
|---|---|---|---|---|---|
| Silero VAD v4 | end-pointing | `silero_vad.onnx` | 0.156 M | frozen | 40 initializers |
| SenseVoice-small int8 | ASR + emotion + event posterior | `sensevoice/model.int8.onnx` | 234.000 M | frozen | 1479 initializers, 233.103 M int8 elements counted dequantized |
| RoBERTa-base encoder | text embedding | `roberta-base-11.onnx` | 124.646 M | frozen | 283 initializers |
| TextContextHead | text posterior with context | `heads/text_ctx.npz` | 0.200 M | trained | mlp 772-256-7, k=0, variant full_dialogue; standardizer 1544 elements not counted |
| AudioCalibrator | audio posterior | `heads/audio_cal.npz` | 0.000 M | trained | lr 24-7, variant A; standardizer 48 elements not counted |
| Fusion | fused posterior | `heads/fusion.npz` | 0.000 M | trained | lr 19-7; standardizer 38 elements not counted |
| SentimentHead | sentiment | `heads/sentiment.npz` | 0.000 M | trained | lr 19-3, variant fused; standardizer 38 elements not counted |
| Prosody, discrepancy, selector | rules | - | 0.000 M | none | no learned parameters |
| **Total on the inference path** | | | **359.001 M** | | 5.98 % of the 6 B cap |
| Piper en_US-lessac-medium | TTS for the proxy clips (offline) | `piper/en_US-lessac-medium/en_US-lessac-medium.onnx` | 15.651 M | frozen | 396 initializers, used by tools/make_proxy_clips.py only, not on the inference path |
| Total including Piper | | | 374.652 M | | |

Other files in `heads/` (not on the inference path): `heads/audio_cal_A.npz` (0.000 M), `heads/audio_cal_B.npz` (0.007 M), `heads/text_ctx_k0.npz` (0.200 M).

Exact counts:

- `silero_vad.onnx`: 155,908 (FLOAT 155,908), 643,854 bytes on disk
- `sensevoice/model.int8.onnx`: 233,999,729 (FLOAT 896,600, UINT8 233,103,129), 239,233,841 bytes on disk
- `roberta-base-11.onnx`: 124,645,716 (FLOAT 124,645,632, INT64 84), 498,649,858 bytes on disk
- `heads/text_ctx.npz`: 199,687 (W1 197,632, b1 256, W2 1,792, b2 7), 808,726 bytes on disk
- `heads/audio_cal.npz`: 175 (W 168, b 7), 3,622 bytes on disk
- `heads/fusion.npz`: 140 (W 133, b 7), 3,192 bytes on disk
- `heads/sentiment.npz`: 60 (W 57, b 3), 3,010 bytes on disk
- `piper/en_US-lessac-medium/en_US-lessac-medium.onnx`: 15,650,546 (FLOAT 15,650,472, INT32 1, INT64 73), 63,149,198 bytes on disk
