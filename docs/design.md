# Design

The system as I built it. Why each choice was made, and what would change it, is in docs/decisions.md. Measured numbers are produced by scripts into results/ and cited from there. External files, checksums and licences are in PROVENANCE.md.

## Framing: words versus tone

I took the challenge's premise, that tone changes how words should be read, literally. The system turns 16 kHz audio into one state per utterance: a text posterior over the seven MELD emotions, an audio posterior, how far the two disagree and in which direction, a fused label with a calibrated confidence, and a short reply chosen from that state. The disagreement is a first-class field, not a by-product.

I chose the Text + Audio track; the MELD authors themselves left visual features out of their baselines (MELD paper, Section 4.1). All inference is local, on CPU, through onnxruntime.

## Three processes

The package is `meld_rt`. Three processes talk over ZeroMQ on localhost. The client (`meld_rt/clients/replay.py`, `mic.py`) reads a WAV at real-time pace or captures from a microphone and pushes 512-sample (32 ms) PCM16 frames with capture timestamps. The service (`meld_rt/service.py`) is one process with three threads: ingest pulls frames into a 30 s ring buffer, runs Silero VAD on every frame and hands segments and provisional requests to the worker through a bounded queue; the worker runs the experts and heads in order, serialized; the publisher owns the PUB socket. The monitor (`meld_rt/clients/monitor.py`) subscribes, prints every message with its timings and writes JSONL. A robot body would subscribe the same way.

## Protocol

Input is PUSH to PULL at `tcp://127.0.0.1:5555`; output is PUB to SUB at `tcp://127.0.0.1:5556`; both are defined in `meld_rt/protocol.py`. Input topics: `audio` (PCM16 payload; header `stream_id`, `seq`, `t_capture_ns`), `text` and `ctl` (`start`, `flush`, `reset`, `stop`). Output topics: `state.provisional`, `state.final`, `response`, `health` at 1 Hz and `error`.

Clocks. The media clock is cumulative samples divided by 16000 since `start` and is the only clock inside state fields (`t_media_start`, `t_media_end`), so replays are deterministic. Two wall clocks exist and are never subtracted from each other: the client's monotonic `t_capture_ns`, echoed as `t_speech_end_capture_ns` for the frame in which the endpoint was detected, and the service's monotonic `t_service_recv_ns` (arrival of that frame) and `t_emitted_ns`. Service-side latency is `t_emitted_ns - t_service_recv_ns`. Client-side latency is the monitor's own now minus `t_speech_end_capture_ns`, on one host.

Provisional versus final. While speech is active, a provisional state is computed at most once per second and only if the worker is idle. For the first 4 s the whole segment is decoded and `text.partial` is filled; after that, provisionals drop to 0.5 Hz on the trailing 4 s window and carry the audio posterior only (`window` `"tail"`). No provisional costs more than one SenseVoice pass over 4 s.

Backpressure. The queue holds at most 2 s of pending work; when non-empty, provisional requests are skipped and counted in `health.dropped_provisionals`. Audio frames are never dropped; if the worker falls more than 5 s behind, `error{code: "overloaded"}` is published. Failures set `degraded` instead of stopping: a sequence gap opens a new `utt_id`, an ASR failure gives a neutral state with confidence 0, a text failure falls back to the audio posterior.

## One utterance, start to finish

VAD marks `speech_start` after three consecutive frames with p > 0.5 and `speech_end` after 400 ms below 0.35, or forces an endpoint at 15 s. At `speech_end` the worker computes fbank, LFR and CMVN and runs SenseVoice-small int8 once. From that one pass come the greedy CTC transcript, an ASR confidence, the softmax over the nine emotion tokens at logit row 1, the softmax over the eleven event tokens at row 2 and, with a graph edit, pooled 512-d encoder states. RoBERTa-base encodes the transcript (at most 64 pieces), mean-pooled to 768-d. Discrepancy is the Jensen-Shannon divergence between the text and audio posteriors, a flag against a dev-chosen threshold, and a direction from class valence: `tone_more_negative`, `tone_more_positive`, `tone_surprised`, `mixed` or `none`. The selector picks a line; the publisher sends `state.final`, then `response`. A `text` message skips the audio steps, sets `fused.mode` to `"text_only"` and `degraded` to `["no_audio"]`, and otherwise follows the same path.

## Components

| Component | Input dims | Source | Parameters | Trained |
|---|---|---|---|---|
| Silero VAD v4 | 512-sample frame | `silero_vad.onnx`, sherpa-onnx release | 0.16 M | frozen |
| SenseVoice-small int8 | segment fbank | `model.int8.onnx`, sherpa-onnx release 2024-07-17 | 234.00 M | frozen |
| RoBERTa-base | 64 pieces | `roberta-base-11.onnx`, ONNX model zoo; tokenizer from the spaCy `en_core_web_trf` wheel | 124.65 M | frozen |
| TextContextHead, MLP 2308-256-7 | 3 x 768 embeddings + 4 flags | own | 0.59 M (0.20 M at k = 0) | trained |
| AudioCalibrator, LR 24-7 or 1048-7 | 9 log probs + 11 events + 4 prosody (+ 1024 pooled) | own | 0.01 M | trained |
| Fusion, LR 19-7 | 7 + 7 log probs, 2 entropies, ASR confidence, laughter, unknown | own | 140 | trained |
| SentimentHead, LR 19-3 | the fusion features | own | 60 | trained |
| Prosody, discrepancy, selector | rules | own | 0 | none |

Total on the inference path: 359.4 M, 6.0 % of the 6 B cap, recomputed by `tools/count_params.py` into results/params.md. Piper (15.65 M) is optional spoken output, counted separately. The fusion head trains on cross-fitted expert predictions and is temperature-scaled on dev. Until MELD audio is cached, the live path uses a fixed product of experts and says so: `fused.mode` is `"poe_uncalibrated"`, `calibrated` is false. Context is a causal window, k chosen on dev; a runtime-regime row restricts it to the same speaker's previous utterances, as at the lamp.

## State schema

```json
{"kind": "final", "stream_id": "s1", "utt_id": 7, "seq_range": [1200, 1356],
 "t_media_start": 10.02, "t_media_end": 12.40,
 "t_speech_end_capture_ns": 171000000000, "t_service_recv_ns": 171000000400, "t_emitted_ns": 171133000400,
 "window": "full", "continued_from": null,
 "asr": {"text": "well that went great", "conf": 0.91, "lang": "en", "tokens_per_s": 2.9},
 "audio": {"emotion": {"joy": 0.05, "sadness": 0.06, "anger": 0.52, "neutral": 0.20, "fear": 0.02, "disgust": 0.03, "surprise": 0.02, "other": 0.01, "unknown": 0.09},
           "events": {"speech": 0.97, "laughter": 0.01, "bgm": 0.01, "applause": 0.0, "cry": 0.0, "sneeze": 0.0, "breath": 0.0, "cough": 0.0, "sing": 0.0, "speech_noise": 0.01, "event_unk": 0.0},
           "prosody": {"f0_median_st": 44.1, "f0_range_st": 9.3, "intensity_std_db": 6.2, "rate_tps": 2.9},
           "encoder_states_used": true},
 "text": {"partial": null, "emotion": {"neutral": 0.3, "surprise": 0.05, "fear": 0.02, "sadness": 0.05, "joy": 0.5, "disgust": 0.03, "anger": 0.05}, "context_k": 2, "keyphrase": "went great"},
 "fused": {"emotion": {"neutral": 0.22, "surprise": 0.04, "fear": 0.02, "sadness": 0.06, "joy": 0.08, "disgust": 0.03, "anger": 0.55},
           "label": "anger", "confidence": 0.55, "sentiment": "negative", "mode": "stacked", "calibrated": true},
 "discrepancy": {"jsd": 0.41, "flag": true, "direction": "tone_more_negative"},
 "dialogue": {"turn_index": 7, "prev_label": "neutral", "emotion_changed": true, "trajectory": ["neutral", "neutral", "joy", "neutral", "anger"]},
 "timing_ms": {"vad_tail": 400, "frontend": 6, "sensevoice": 95, "prosody": 14, "text_encoder": 17, "heads": 1, "service_total": 133, "provisional_in_flight": 0},
 "degraded": []}
```

Provisional states carry `kind` `"provisional"`, `text.partial` (null in tail mode), the window's audio posterior, a `stability` value and no response. Posterior vectors follow the MELD README order in `meld_rt/labels.py`; the dataclass is in `meld_rt/state.py`.

## Replies and grounding

Replies come from `meld_rt/response/bank.yaml`: about 100 lines of at most 12 words that I wrote, in four styles, `acknowledge`, `ask`, `name_the_gap` and `hold_back`. The index is (fused label, discrepancy direction), 28 cells with at least 3 variants each; `none` and `mixed` share a cell. A confidence bucket overrides the cell: low takes a `hold_back` line that makes no emotion claim, mid prefers `ask`, high prefers `acknowledge` or `name_the_gap`. Selection is deterministic, with one keyphrase slot filled from the transcript. The `grounding` field lists the state fields that licensed the line.

## What real time means here

The lamp must react inside the natural pause after a person stops talking, and show it is listening while they talk. My targets at 4 threads, measured from the arrival of the endpoint frame and including any provisional in flight:

- utterances up to 5 s: final state p50 at most 250 ms, p95 at most 400 ms
- utterances up to 8 s: p50 at most 350 ms, p95 at most 550 ms
- utterances up to the 15 s cap: p50 at most 700 ms, p95 at most 1000 ms

Also reply text within 5 ms of the final, provisionals at 1 Hz for the first 4 s of speech and 0.5 Hz after, and sustained compute below 0.3 s per second of audio. The share of MELD utterances in each bucket is in results/data_stats.md. The 400 ms VAD tail is a choice, not compute. Assumed: one speaker near the microphone, 4 CPU cores, no GPU. Whether the targets are met is in results/measurements.md.

## Evaluation in brief

Published splits; selection, thresholds, calibration and k on dev only; test once per condition. Weighted F1 is primary, as in every published MELD row, with macro F1, per-class F1, bootstrap CIs over dialogues and ECE beside it. Conditions: majority class; text only with and without context; audio only, raw and calibrated; fused, stacked and product of experts; ablations of events and prosody; the live condition on ASR transcripts with WER; the disagreement subset; early commit on 2 s and 4 s prefixes; sentiment. Comparison rows are MELD paper Tables 11 and 13, EmoBERTa Table 2 and the SenseVoice report Table 8, whose SER training corpora are unlisted, so the audio number is never called zero-shot.

## Left out, and why

Vision and the three-modality extension: no camera, no reachable video, and speaker-face selection in multi-party frames is its own project. An LLM reply generator: nondeterministic, slow on CPU, grounding not unit-testable. Fine-tuning SenseVoice, wav2vec2 or HuBERT: GPU hours for a few points, not reproducible by a reviewer in minutes. Bidirectional context models: they use future utterances. Arousal and valence fields: no MELD labels to score them. Reinforcement learning in the core: MELD has no reward for a reply; the one defensible target, when to commit a label, is a gated extension.
