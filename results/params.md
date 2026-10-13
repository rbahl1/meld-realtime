# Parameter count

Written by `python tools/count_params.py --models models --heads heads` on 2026-10-13T17:04:43Z, from the files on disk. An ONNX row is the sum of the element counts of the graph initializers. An int8 initializer counts one parameter per element, which is its dequantized count. A head row is the sum of the weight and bias arrays in its `.npz` file. The standardizer mean and scale are not parameters and are listed in the note. Tensors held by Constant nodes are not initializers and are not counted.

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

Extension (`--vision --reply llm`), on top of the core total above:

| Component | Role | File | Params | Trained | Note |
|---|---|---|---|---|---|
| YuNet 2023mar | face detection (vision) | `vision/face_detection_yunet_2023mar.onnx` | 0.053 M | frozen | 112 initializers |
| FER MobileFaceNet 2022july | facial expression (vision) | `vision/facial_expression_recognition_mobilefacenet_2022july.onnx` | 1.194 M | frozen | 135 initializers |
| VisionCalibrator | face posterior | `heads/vision_cal.npz` | 0.004 M | trained | lr 528-7, variant B; standardizer 1056 elements not counted |
| Fusion3 | fused posterior, three inputs | `heads/fusion3.npz` | 0.000 M | trained | lr 29-7, variant tav; standardizer 58 elements not counted |
| Sentiment3 | sentiment, three inputs | `heads/sentiment3.npz` | 0.000 M | trained | lr 29-3, variant tav; standardizer 58 elements not counted |
| CommitPolicy | when to commit the label | `heads/commit_policy.npz` | 0.000 M | trained | P(correct) logistic regression and Q_wait mlp; standardizer 28 elements not counted |
| Qwen2.5-0.5B-Instruct (model_int8.onnx) | streaming reply | `llm/Qwen2.5-0.5B-Instruct/onnx/model_int8.onnx` | 494.033 M | frozen | weight tensors once, dequantized element count; not counted: exporter constants 4,194,352, integer constants 2,005, scalar constants 149, scales zero points 338 |
| **Total with the extension** | | | **854.286 M** | | 14.24 % of the 6 B cap; extension alone 495.284 M |

Other LLM candidates on disk (not deployed): `llm/Qwen2.5-0.5B-Instruct/onnx/model_q4.onnx` (494.033 M), `llm/Qwen2.5-1.5B-Instruct/onnx/model_int8.onnx` (1543.714 M), `llm/Qwen2.5-1.5B-Instruct/onnx/model_q4.onnx` (1543.714 M).

Other files in `heads/` (not on the inference path): `heads/ablation_no_events_audio_cal.npz` (0.000 M), `heads/ablation_no_events_fusion.npz` (0.000 M), `heads/ablation_no_events_no_prosody_audio_cal.npz` (0.000 M), `heads/ablation_no_events_no_prosody_fusion.npz` (0.000 M), `heads/ablation_no_prosody_audio_cal.npz` (0.000 M), `heads/ablation_no_prosody_fusion.npz` (0.000 M), `heads/audio_cal_A.npz` (0.000 M), `heads/audio_cal_B.npz` (0.007 M), `heads/sentiment_text_only.npz` (0.002 M), `heads/text_ctx_k0.npz` (0.200 M).

Exact counts:

- `silero_vad.onnx`: 155,908 (FLOAT 155,908), 643,854 bytes on disk
- `sensevoice/model.int8.onnx`: 233,999,729 (FLOAT 896,600, UINT8 233,103,129), 239,233,841 bytes on disk
- `roberta-base-11.onnx`: 124,645,716 (FLOAT 124,645,632, INT64 84), 498,649,858 bytes on disk
- `heads/text_ctx.npz`: 199,687 (W1 197,632, b1 256, W2 1,792, b2 7), 808,726 bytes on disk
- `heads/audio_cal.npz`: 175 (W 168, b 7), 3,622 bytes on disk
- `heads/fusion.npz`: 140 (W 133, b 7), 3,192 bytes on disk
- `heads/sentiment.npz`: 60 (W 57, b 3), 3,010 bytes on disk
- `piper/en_US-lessac-medium/en_US-lessac-medium.onnx`: 15,650,546 (FLOAT 15,650,472, INT32 1, INT64 73), 63,149,198 bytes on disk
- `vision/face_detection_yunet_2023mar.onnx`: 53,121 (FLOAT 53,112, INT64 9), 232,589 bytes on disk
- `vision/facial_expression_recognition_mobilefacenet_2022july.onnx`: 1,193,927 (FLOAT 1,193,927), 4,791,892 bytes on disk
- `heads/vision_cal.npz`: 3,703 (W 3,696, b 7), 22,018 bytes on disk
- `heads/fusion3.npz`: 210 (W 203, b 7), 3,810 bytes on disk
- `heads/sentiment3.npz`: 90 (W 87, b 3), 3,202 bytes on disk
- `heads/commit_policy.npz`: 272 (p_w 14, p_b 1, qw_W1 224, qw_b1 16, qw_W2 16, qw_b2 1), 13,150 bytes on disk
- `llm/Qwen2.5-0.5B-Instruct/onnx/model_int8.onnx`: 494,032,768 (FLOAT 71,552, INT8 357,826,560, UINT8 136,134,656), 512,096,557 bytes on disk
