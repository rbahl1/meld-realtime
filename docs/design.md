# Design

This document describes the system as built. Part 1 (1.1 to 1.13) is the core on text and audio. Part 2 is the optional extension that adds the face, a commit policy fitted by reinforcement learning and a streamed reply. [docs/decisions.md](decisions.md) gives the reason for each choice, the alternatives and what would change it. Every number below comes from a file that a script wrote into `results/` or `heads/`, from a config file, from the code, or from a paper, and the source is named. Code comments cite the sections by number (ex. "design 1.10").

## 1.1 Track and framing

The challenge says that tone changes how words should be understood. The system makes that relation part of its output. For each utterance in a 16 kHz audio stream it publishes a state with a text posterior over the seven MELD emotions, an audio posterior from the voice, how far the two disagree and in which direction, a fused label with a confidence, and a short reply chosen from that state.

The primary track is text and audio. The MELD authors left visual features out of their own baselines, "as video-based speaker identification and localization is an open problem" (MELD paper, Section 4.1). The extension of part 2 adds the face from the video as a third input, with a rule that picks the speaker's face. It is off unless the service runs with `--vision` (the face and the commit policy) or `--reply llm` (the streamed reply).

All inference runs locally on the CPU, through onnxruntime and, for the two face models of the extension, OpenCV DNN. Nothing is sent to a remote service.

## 1.2 Processes and threads

The package is `meld_rt`. Three processes talk over ZeroMQ on localhost.

1. A client pushes audio. `meld_rt/clients/replay.py` reads a WAV file at real-time pace (`--realtime`) or as fast as possible, or sends one typed utterance. `meld_rt/clients/mic.py` captures from a microphone through sounddevice. Both send 512-sample (32 ms) PCM16 frames stamped with the client's capture time.
2. The service (`meld_rt/service.py`) is one process with three threads. The ingest thread pulls frames, runs Silero VAD on every frame and applies the provisional policy (1.3). It hands jobs to the worker through a FIFO queue. The worker thread runs the experts, the heads, the discrepancy rule and the reply selector one job at a time, so SenseVoice can use every core. The publisher thread owns the PUB socket, stamps the emission time and sends a health message once per second.
3. The monitor (`meld_rt/clients/monitor.py`) subscribes, prints one line per message with its timings and appends every message to a JSONL file. A robot body would subscribe the same way.

The only audio store is the VAD segmenter of each stream. It holds the frames of the open segment, and a ring of 16 frames (pre-roll plus the start run) while no segment is open.

SenseVoice runs with 4 intra-op threads (`--threads`), RoBERTa with 2 (`--text-threads`) and the VAD with 1 on the ingest thread. The extension adds a vision thread (2.6) and a reply thread (2.9).

The runtime dependencies are in `pyproject.toml`: numpy, onnxruntime, kaldi-native-fbank (the front end), tokenizers, pyzmq, pyyaml (the reply bank and the head config), soundfile (WAV input and output), scikit-learn (head training, and it brings scipy, which the clients use to resample), psutil (health RSS and the benches), jiwer (WER) and onnx (the graph edit and the parameter count). The dev group adds pytest, msgpack (tokenizer extraction), sherpa-onnx (the transcript parity test and the Piper proxy clips) and praat-parselmouth (prosody). Without parselmouth the prosody scalars are zero except the speech rate. The `mic` extra adds sounddevice. The `vision` extra adds opencv-python-headless, which runs the face models of the extension. The `finetune` extra (torch, transformers) is declared, but no script uses it.

## 1.3 One utterance, start to finish

1. End-pointing. A segment opens after 8 consecutive frames (256 ms) with a speech probability above 0.5. The count is the larger of 3 and `min_speech_ms` (250) divided by 32 ms and rounded up. The segment audio starts 8 frames (256 ms, `preroll_ms` 250 rounded up to whole frames) before the first of those frames, so `t_media_start` lies 256 ms before the speech onset. The segment closes after a run of frames below 0.35 that lasts `--endpoint-silence-ms`, rounded up to whole frames: 16 frames (512 ms) at the default 500 ms and 13 frames (416 ms) at 400 ms. `t_media_end` is the time the speech stopped, and the segment audio keeps the silence tail. A segment that reaches 15 s (`--max-segment-s`) is closed by force, and the next one carries `continued_from`.
2. Provisionals. While a segment is open, a provisional state is computed at most once per second (`--provisional-hz`) and only when the worker is idle and nothing is queued. While the segment is at most 4 s long (`--provisional-window-s`), the whole segment is decoded and `text.partial` holds its transcript. After that, provisionals come at half the rate on the trailing 4 s window, with `window` `"tail"` and no partial text. Provisionals run SenseVoice only. `--provisional-budget-ms` (0, off by default) skips the next provisional when the last one took longer than the budget. Queued provisionals are dropped at the endpoint.
3. Audio expert. At the endpoint the worker computes the fbank, LFR and CMVN front end and runs SenseVoice-small int8 once over the segment. That one pass gives the greedy CTC transcript, an ASR confidence (the mean of the maximum softmax over the emitted tokens), the language, the softmax over the nine emotion tokens at logit row 1, the softmax over the eleven event tokens at row 2, and the tokens per second. With `--encoder-states` and the edited graph from `tools/edit_sensevoice_graph.py`, the same pass also gives the encoder states, pooled (mean and maximum) to 1024 values. Prosody adds four scalars from parselmouth: the F0 median and the F0 range in semitones, the standard deviation of the intensity in dB, and the speech rate.
4. Text expert. RoBERTa-base encodes the transcript (at most 64 pieces), and the mean over the non-special positions gives 768 values. It does not run when the ASR failed or its confidence is below 0.3.
5. Heads. The TextContextHead gives the text posterior. The AudioCalibrator gives the audio posterior when `heads/audio_cal.npz` exists. Without that file the audio posterior is the SenseVoice posterior renormalized over its seven named classes. How the fused posterior is made depends on `fused.mode` (1.6). The SentimentHead gives the sentiment.
6. Discrepancy. `discrepancy.jsd` is the Jensen-Shannon divergence in bits between the 7-way text posterior and the raw SenseVoice posterior renormalized over its seven named classes (1.6). The flag is set when it exceeds a threshold. The threshold is `jsd_flag` from `heads/manifest.json`, 0.3831 from dev. A manifest without that value leaves the service at `--jsd-flag`, whose default is 0.2. The direction is `none` when the flag is off. Otherwise it is `tone_surprised` when the audio gives surprise more than 0.4 and the text less than 0.2. Failing that, it is `tone_more_negative` or `tone_more_positive` when the expected valence of the audio posterior differs from that of the text posterior by more than 0.1, and `mixed` when it does not. Valence is 1 for joy, 0 for neutral and surprise, and -1 for sadness, anger, disgust and fear (`meld_rt/labels.py`).
7. Dialogue. Each stream keeps its last 8 finals. The state reports the turn index, the previous label, whether the label changed and the last five labels.
8. Reply. The selector picks one line (1.9).
9. Publish. The publisher sends `state.final` and then `response`.

A `text` message skips steps 1 to 3 and the discrepancy, sets `fused.mode` to `"text_only"` and `degraded` to `["no_audio"]`, and otherwise follows the same path.

## 1.4 Protocol, clocks and failures

Input is PUSH to PULL at `tcp://127.0.0.1:5555`. Output is PUB to SUB at `tcp://127.0.0.1:5556`. `meld_rt/protocol.py` defines both. Input messages are `[topic, header_json, payload]` and output messages are `[topic, json]`.

| topic | direction | content |
|---|---|---|
| `audio` | in | header `stream_id`, `seq`, `t_capture_ns`, `sr` (16000), `n_samples` (512), `fmt` (`pcm16`), and a payload of 1024 bytes of little-endian int16 |
| `text` | in | header `stream_id`, `seq`, `t_capture_ns`, `text`, `speaker`, and an empty payload |
| `ctl` | in | header `stream_id`, `cmd`: `start` (media clock to 0), `flush` (force an endpoint), `reset` (clear the dialogue), `stop` (close the stream) |
| `state.provisional`, `state.final` | out | the state of 1.8 |
| `response` | out | `stream_id`, `utt_id`, `text`, `template_id`, `style`, `grounding`, `t_emitted_ns` |
| `health` | out, 1 Hz | `stream_id`, `dropped_provisionals`, `queue_s`, `queue_audio_s`, `fusion_mode`, `model_versions`, `rss_mb`, `models_loaded`, `uptime_s`, `t_emitted_ns` |
| `error` | out | `stream_id`, `code` (`audio_gap`, `overloaded`, `asr_failed`, `text_failed`, `bad_message`), `msg` |

The extension adds the input topic `video`, the output topics `state.commit` and `response.delta`, and the error code `vision_failed`, sent when the face expert cannot load (2.5, 2.6).

The media clock is the number of samples since `start` divided by 16000. It is the only clock inside state fields (`t_media_start`, `t_media_end`), so a replay gives the same times on every host. Two wall clocks exist, and they are never subtracted from each other. The client's monotonic clock gives `t_capture_ns`, which comes back as `t_speech_end_capture_ns` for the frame in which the endpoint was detected. The service's monotonic clock gives `t_service_recv_ns`, the arrival of that frame, and `t_emitted_ns`. Service-side latency is `t_emitted_ns` minus `t_service_recv_ns` (`timing_ms.service_total`). The monitor computes client-side latency as its own clock minus `t_speech_end_capture_ns`, which is valid only when client and monitor share a host.

Audio frames are never dropped, and a final is never skipped. A provisional is skipped, and counted in `health.dropped_provisionals`, when the worker is busy, when a job is queued, or when the last provisional took longer than `--provisional-budget-ms`. `health.queue_s` is the worker's lag in seconds: the age of the oldest unfinished job, queued or running. `health.queue_audio_s` beside it is the audio waiting in the queue. When `queue_s` passes 5 s, the service publishes the error `overloaded`. It does not repeat the error until the lag has dropped below 2.5 s.

Failures set `degraded`, and the service keeps running.

- A sequence gap longer than 500 ms publishes `audio_gap`, closes the open segment as a forced final marked `audio_gap` when the segment holds at least 250 ms, and resets the VAD.
- A SenseVoice exception, an empty transcript or an ASR confidence below 0.3 gives `degraded` `asr_failed` or `low_asr_conf`, a uniform fused posterior, `fused.label` `neutral`, a confidence of 0 and a "did not catch that" reply.
- A text-expert exception gives `text_failed`. The fused posterior is then the audio posterior, with `calibrated` false.
- A missing `heads/text_ctx.npz` gives `no_text_head` and a uniform text posterior.
- A malformed message gives `bad_message` and is ignored.

Models load on the first message, or at start with `--preload`. A stream closed by `stop` is kept for 60 s, so a client that comes back with the same `stream_id` keeps its dialogue.

## 1.5 Components and parameters

| component | role | file | inputs | parameters | trained |
|---|---|---|---|---|---|
| Silero VAD v4 | end-pointing | `silero_vad.onnx` | one 512-sample frame | 0.156 M | frozen |
| SenseVoice-small int8 | transcript, emotion and event posteriors | `sensevoice/model.int8.onnx` | the segment's fbank | 234.000 M | frozen |
| RoBERTa-base | text embedding | `roberta-base-11.onnx` | up to 64 pieces | 124.646 M | frozen |
| TextContextHead, MLP 772-256-7 | text posterior | `heads/text_ctx.npz` | 768 × (k + 1) + 4 values, k = 0 | 0.200 M | trained |
| AudioCalibrator A, LR 24-7 | audio posterior | `heads/audio_cal.npz` | 9 log probabilities, 11 events, 4 prosody scalars (variant B adds 1024 encoder values and was not chosen) | 175 | trained |
| Fusion, LR 19-7 | fused posterior | `heads/fusion.npz` | two 7-way log posteriors, their entropies, ASR confidence, laughter, unknown mass | 140 | trained |
| SentimentHead, LR 19-3 | sentiment | `heads/sentiment.npz` | the 19 fusion inputs (`fused` variant, deployed), or the text head's inputs (772-3, `text_only` variant) without audio heads | 60 | trained |
| prosody, discrepancy, selector | rules | | | 0 | none |

The total on the inference path is 359.001 M, 5.98 % of the 6 B cap ([results/params.md](../results/params.md), written by `tools/count_params.py` from the files on disk). The three logistic-regression heads are listed there as 0.000 M each, and their exact counts are the ones above. Piper en_US-lessac-medium (15.651 M) synthesizes the proxy clips offline and is not on the inference path. The latency of each component is in [results/measurements.md](../results/measurements.md).

With `--vision --reply llm` the extension adds these components (the extension table of [results/params.md](../results/params.md)):

| component | role | file | inputs | parameters | trained |
|---|---|---|---|---|---|
| YuNet 2023mar | face detection | `vision/face_detection_yunet_2023mar.onnx` | one frame, scaled down to a long side of 480 px | 0.053 M | frozen |
| FER MobileFaceNet 2022july | facial expression and a 512-value embedding | `vision/facial_expression_recognition_mobilefacenet_2022july.onnx` | one aligned 112 × 112 face | 1.194 M | frozen |
| VisionCalibrator B, LR 528-7 | face posterior | `heads/vision_cal.npz` | 16 face values and the 512-value mean embedding of the chosen track | 3,703 | trained |
| Fusion3, LR 29-7 | fused posterior | `heads/fusion3.npz` | three 7-way log posteriors, their entropies, ASR confidence, laughter, unknown mass, a face flag, the face presence | 210 | trained |
| Sentiment3, LR 29-3 | sentiment | `heads/sentiment3.npz` | the 29 fusion3 inputs | 90 | trained |
| CommitPolicy, LR 14-1 and MLP 14-16-1 | when to commit the label | `heads/commit_policy.npz` | 14 tick features | 272 | trained |
| Qwen2.5-0.5B-Instruct int8 | streamed reply | `llm/Qwen2.5-0.5B-Instruct/onnx/model_int8.onnx` | the prompt of 2.9 | 494.033 M | frozen |

With the extension the total is 854.286 M, 14.24 % of the 6 B cap, and the extension alone is 495.284 M. The reply model counts each quantized weight once at its dequantized element count, without the quantization scales and zero points or the exporter's constant tables (the rule in `tools/count_params.py`). The other LLM files on disk were benched and are not on the inference path.

## 1.6 Trained heads and fusion modes

`tools/train_heads.py` trains the heads from cached features with the settings in `configs/heads.yaml`. Every choice is made on dev by weighted F1, and test is never read.

The text stage is done. The TextContextHead is an MLP with 256 hidden units, trained over a grid of k (0, 1, 2), class weights (MELD README, balanced, none) and learning rate (0.001, 0.0003, 0.0001), with early stopping on dev (patience 6, at most 60 epochs). Dev picked k = 0, no class weights and a learning rate of 0.0001, and a temperature of 1.2228 fitted on dev calibrates the head ([heads/manifest.json](../heads/manifest.json)). The selector thresholds come from the calibrated dev predictions. An isotonic fit of correctness on the confidence gives the dev accuracy as a rising function of the confidence. `low_conf` is the smallest confidence, on a grid of 0.01, at which that accuracy reaches one half (0.39), so below it a label claim is more often wrong than right. `high_conf` is where it reaches 70 % (0.73). The audio stage applies the same rule to the fusion head's calibrated dev predictions, which gives 0.47 and 0.73, and the service uses those (`thresholds` in [heads/manifest.json](../heads/manifest.json)). A logistic regression over the same text features gives a text-only sentiment head (C 0.1, no class weights, dev weighted F1 65.25, temperature 1.4197). `--text-only` writes it as `heads/sentiment.npz` when there is no audio stage ([results/01_text_only.md](../results/01_text_only.md)). `--sentiment-text-only` fits the same head with the deployed config, seed and k, and writes it next to the deployed heads as `heads/sentiment_text_only.npz`, so the fused `heads/sentiment.npz` stays as it is (`sentiment_text_only` in [heads/manifest.json](../heads/manifest.json)). The text grid took 316.7 s and the whole run, with the audio stage, 435.8 s ([heads/manifest.json](../heads/manifest.json)). [results/01_text_only.md](../results/01_text_only.md) has the grid.

The audio stage ran on MELD audio. It trains two AudioCalibrators, A on the 24 slot, event and prosody inputs and B with the 1024 encoder values added, and keeps the better one on dev. Both expert heads are refit in 5 folds split by dialogue, so the fusion head trains on out-of-fold posteriors only. The fusion head is then temperature-scaled on dev. The same stage picks the product-of-experts weight from 0.25, 0.5, 0.75 and 1.0, fits a fused sentiment head on the 19 fusion inputs, and sets `jsd_flag` to the 0.8 quantile of the dev JSD between the text posterior and the raw SenseVoice posterior. CSV rows without a cached clip are left out of this stage and counted in the manifest (one in train and one in dev). The service computes the run-time JSD against the same raw SenseVoice posterior, so the threshold and the run-time JSD come from the same audio posterior. On dev, calibrator A scores 45.96 weighted F1 against 44.43 for B, so A is kept. The fusion head scores 57.95 on dev, the product of experts 55.56 at its chosen weight of 0.75, and the fused sentiment head 67.79. `jsd_flag` comes out at 0.3831 ([RESULTS.md](../RESULTS.md), "Choices made on dev"). The stacked fusion scores higher on dev than the product of experts, so it is deployed.

`fused.mode` says how the fused posterior was made, and `fused.calibrated` says whether its confidence is calibrated.

| mode | when | fused posterior | `fused.confidence` | `calibrated` |
|---|---|---|---|---|
| `stacked` | `heads/fusion.npz` exists | the fusion head | its maximum | true |
| `poe` | no fusion head, both expert heads present | product of the text and calibrated audio posteriors | its maximum | false |
| `poe_uncalibrated` | an expert head is missing, in practice the audio calibrator | product of the text posterior and the raw SenseVoice posterior | the text head's probability of the fused label | false |
| `text_only` | a text message, or `--fusion-mode text_only` | the text posterior | its maximum | true when the text head exists |
| `stacked_tav` | `--vision` and `heads/fusion3.npz` exists (2.4) | fusion3 over text, audio and face | its maximum | true |

The product raises the audio posterior to a power w. Without an audio stage in the manifest, w is 0.5. With it, the service reads the dev-chosen w (0.75) from `audio_fusion.poe.weight` in the manifest (`poe_weight_from` in `meld_rt/service.py`).

In `poe_uncalibrated` mode the maximum of the product is not a probability of being right, because two soft posteriors multiply into a sharp one. So the reported confidence is the calibrated text head's probability of the fused label. `fused.emotion` and `fused.label` stay the product and its argmax. `tests/test_service_rules.py` checks this rule. The core demo of 1.8 runs in `stacked` mode, the second demo in `stacked_tav`, and neither uses it. The selector also caps every state with `calibrated` false at the mid bucket (1.9), so such a state can ask but never asserts.

## 1.7 Conversational context

The TextContextHead sees the current utterance and up to k previous ones, with same-speaker flags, the turn index divided by 20 and a first-turn flag. In MELD the previous utterances are those of the same dialogue, whoever spoke them. Dev picked k = 0, so the deployed head sees the current utterance and the two turn flags only. With a larger k, the service would pass the stream's own previous finals with the same-speaker flags set to 1, because one person talks to the robot. The robot's own lines are never fed back, because MELD has no such lines to train on.

The evaluation has a runtime-regime row for that difference: the same head with the window restricted to the same speaker's previous utterances in the dialogue. No future utterance is ever used. `ctl` `reset` clears the dialogue of a stream.

## 1.8 State schema

This is the final state of the demo in [results/demo_log.jsonl](../results/demo_log.jsonl), without the three fields the monitor adds when it logs (`topic`, `t_recv_ns`, `latency_client_ms`). `tools/render_demo_docs.py` copies it from the log:

```json
{
 "kind": "final",
 "stream_id": "demo",
 "utt_id": 1,
 "seq_range": [
  26,
  217
 ],
 "t_media_start": 0.832,
 "t_media_end": 6.464,
 "t_speech_end_capture_ns": 8222443854187,
 "t_service_recv_ns": 8222452139315,
 "t_emitted_ns": 8222875437256,
 "window": "full",
 "continued_from": null,
 "asr": {
  "text": "the tribal chieftain called for the boy and presented him with 50 pieces of gold",
  "conf": 0.9243,
  "lang": "en",
  "tokens_per_s": 3.092
 },
 "audio": {
  "emotion": {
   "neutral": 0.4558860957622528,
   "surprise": 1.2566384612000547e-05,
   "fear": 9.928767212841194e-06,
   "sadness": 0.00714103365316987,
   "joy": 0.0009436953696422279,
   "disgust": 0.00020711463002953678,
   "anger": 2.095875061058905e-05,
   "other": 7.766694761812687e-05,
   "unknown": 0.5357009172439575
  },
  "events": {
   "speech": 0.9994488954544067,
   "bgm": 0.0005384060787037015,
   "laughter": 7.825335615052609e-07,
   "applause": 2.3162508000496018e-07,
   "cry": 1.4367492440214846e-07,
   "sneeze": 8.824137864849035e-08,
   "breath": 1.3686611737284693e-06,
   "cough": 4.460976924747229e-06,
   "sing": 1.0301091606379487e-06,
   "speech_noise": 2.690720066311769e-06,
   "event_unk": 1.8842168856281205e-06
  },
  "prosody": {
   "f0_median_st": 4.853423118591309,
   "f0_range_st": 24.192941665649414,
   "intensity_std_db": 7.585581302642822,
   "rate_tps": 3.0924479961395264
  },
  "encoder_states_used": false
 },
 "text": {
  "partial": null,
  "emotion": {
   "neutral": 0.34665821157399634,
   "surprise": 0.015504439377634457,
   "fear": 0.014496428715798678,
   "sadness": 0.018408880451890812,
   "joy": 0.4228928595278545,
   "disgust": 0.059279918798797404,
   "anger": 0.1227592615540278
  },
  "context_k": 0,
  "keyphrase": "the tribal chieftain called"
 },
 "fused": {
  "emotion": {
   "neutral": 0.5519663536873755,
   "surprise": 0.007264129753390723,
   "fear": 0.008743568897943612,
   "sadness": 0.08765835523628694,
   "joy": 0.29297454569028947,
   "disgust": 0.021315230124747026,
   "anger": 0.030077816609966734
  },
  "label": "neutral",
  "confidence": 0.552,
  "sentiment": "neutral",
  "mode": "stacked",
  "calibrated": true
 },
 "discrepancy": {
  "jsd": 0.4213,
  "flag": true,
  "direction": "tone_more_negative"
 },
 "dialogue": {
  "turn_index": 0,
  "prev_label": null,
  "emotion_changed": false,
  "trajectory": [
   "neutral"
  ]
 },
 "timing_ms": {
  "vad_tail": 512.0,
  "frontend": 9.37,
  "sensevoice": 210.81,
  "prosody": 30.34,
  "text_encoder": 31.31,
  "heads": 0.74,
  "service_total": 423.3,
  "provisional_in_flight": 139.2
 },
 "degraded": []
}
```

- `audio.emotion` has nine keys: the seven SenseVoice emotion classes under their MELD names, in the MELD README order, then `other` and `unknown`. `text.emotion` and `fused.emotion` have the seven MELD keys in the same order. `audio.events` has the eleven SenseVoice events in `EVENT_KEYS` order (`meld_rt/labels.py`).
- `fused.mode`, `fused.confidence` and `fused.calibrated` follow 1.6. `fused.sentiment` is one of MELD's three sentiment classes.
- `timing_ms` is in milliseconds. `vad_tail` is the silence tail inside the segment (512 at the default 500 ms, as in the demo). `provisional_in_flight` is how long the final waited for a provisional that was running when the endpoint came. The publisher sets `service_total`.
- `degraded` lists the failures of 1.4.
- A provisional state has `kind` `"provisional"`, `window` `"full"` or `"tail"`, `text.partial` (null on the tail window), a uniform `text.emotion` because no text expert runs, a `fused` block that is the audio posterior over the seven named classes, zero prosody except the rate, and `stability`, which is 1 minus the JSD to the previous provisional. It gets no response. In vision mode the full-window provisionals also run the text expert and fusion3 (2.7).

The dataclasses are in `meld_rt/state.py`.

The second demo runs the extension of part 2 on MELD test clip 4_7 (`dia4_utt7.mp4`, kept outside the repository). The clip was picked by a rule written before the choice: a test clip whose chosen face is present in at least 80 % of the frames, whose SenseVoice transcript passes the ASR gate, on which the text head is wrong on that transcript and fusion3 is right on the same transcript, the lowest dialogue id and then the lowest utterance id. `tools/pick_demo_clip.py` applies the rule and writes [results/demo_clip.json](../results/demo_clip.json): 122 of the 2610 test clips meet it, and 4_7 comes first.

<!-- demo_tav: start. tools/render_demo_docs.py writes this block from results/demo_tav_log.jsonl -->
The committed log [results/demo_tav_log.jsonl](../results/demo_tav_log.jsonl) came from `./demo_tav.sh` on one MELD test clip, with the vision heads, the commit policy and the deployed reply generator. `tools/render_demo_docs.py` writes this block from that log. `demo_tav.sh` ends with this summary:

```
3 provisional, 1 final, 1 commit, 4 delta, 1 response, 0 error
utt 1 media 0.352-2.976 s
  transcript  "i fell asleep before i could take a shower and i wouldn don't have time" (conf 0.9643)
  labels      text neutral  voice unknown  face neutral@0.36 (raw sadness, presence 1.0, track 0 of 1, rule asd)
  fused       anger@0.3869 (stacked_tav, calibrated=True, sentiment negative)
  discrepancy words-voice jsd 0.615 flag True tone_more_negative; words-face jsd 0.2181 flag False none  degraded []
  commit      neutral@0.3742 at 2.016 s after the segment start (policy, 1.12 s before the final, matches final False), q_emit 0.2626 q_wait 0.2259
  timing      service_total 233.97 ms (sensevoice 101.89, text 34.34, vision wait 0.01, vision aggregate 0.61, heads 0.65, provisional in flight 77.1); client-side 234.599 ms; vad tail 512.0 ms
  reply       'What happened to you?' [llm, llm, fallback None]
  streamed    4 deltas, first 399.08 ms after the endpoint frame arrived, generation 370.82 ms, 5 tokens, checks {'n_words': 4, 'over_length': False, 'exclamation': False, 'forbidden_words': [], 'empty': False, 'meta_words': [], 'prompt_echo': [], 'non_ascii': False, 'non_printable': False, 'ok': True}
  grounding   ['fused.confidence=0.39<0.56', "asr.text='i fell asleep before i could take a shower and i wouldn don't have time'", 'text.emotion=neutral@0.42', 'audio.emotion=anger@0.10', 'vision.emotion=neutral@0.36', 'discrepancy.direction=tone_more_negative']
```

These are the main fields of the final state, with its vision, commit and discrepancy blocks, and the response that followed it.

```json
{"kind": "final", "stream_id": "demo", "utt_id": 1, "t_media_start": 0.352, "t_media_end": 2.976,
 "asr": {"text": "i fell asleep before i could take a shower and i wouldn don't have time", "conf": 0.9643, "lang": "en", "tokens_per_s": 5.421},
 "fused": {"label": "anger", "confidence": 0.3869, "sentiment": "negative", "mode": "stacked_tav", "calibrated": true},
 "discrepancy": {"jsd": 0.615, "flag": true, "direction": "tone_more_negative", "face_jsd": 0.2181, "face_flag": false, "face_direction": "none"},
 "vision": {"emotion": {"neutral": 0.36132347587255836, "surprise": 0.18555391538945346, "fear": 0.032915650291476105, "sadness": 0.07035760286387191, "joy": 0.11808420945603687, "disgust": 0.051020824537271754, "anger": 0.1807443215893316},
            "raw": {"neutral": 0.24426903004813266, "surprise": 0.2294099877049439, "fear": 0.003641177866865003, "sadness": 0.3293730738436831, "joy": 0.04404611505046414, "disgust": 0.1361755441147386, "anger": 0.01308507137117281},
            "present": true, "track": 0, "rule": "asd", "n_frames": 24, "presence": 1.0, "n_tracks": 1, "asd_corr": 0.3941},
 "commit": {"label": "neutral", "confidence": 0.3742, "t_commit_s": 2.016, "matches_final": false, "source": "policy"},
 "timing_ms": {"vad_tail": 512.0, "frontend": 4.07, "sensevoice": 101.89, "prosody": 13.94, "text_encoder": 34.34, "heads": 0.65, "service_total": 233.97, "provisional_in_flight": 77.1, "vision_wait": 0.01, "vision_aggregate": 0.61},
 "degraded": []}
```

```json
{"stream_id": "demo", "utt_id": 1, "text": "What happened to you?", "template_id": "llm", "style": "generated", "source": "llm", "model": "Qwen2.5-0.5B-Instruct", "fallback": null,
 "grounding": ["fused.confidence=0.39<0.56", "asr.text='i fell asleep before i could take a shower and i wouldn don't have time'", "text.emotion=neutral@0.42", "audio.emotion=anger@0.10", "vision.emotion=neutral@0.36", "discrepancy.direction=tone_more_negative"],
 "checks": {"n_words": 4, "over_length": false, "exclamation": false, "forbidden_words": [], "empty": false, "meta_words": [], "prompt_echo": [], "non_ascii": false, "non_printable": false, "ok": true},
 "n_tokens": 5, "stop_reason": "eos", "first_delta_ms": 399.08, "gen_ms": 370.82, "t_emitted_ns": 13979890257417}
```

The face was found in 24 frames of the utterance (presence 1.0, 1 track, rule `asd`). Its calibrated top class is neutral, and the top class of the raw FER posterior is sadness. The text head's top class for the words is neutral. SenseVoice puts most of its mass on `unknown`, and its top named class is anger. The words-voice JSD, 0.615, is above the dev threshold of 0.3831, so the direction `tone_more_negative` is flagged. The words-face JSD, 0.2181, is below the threshold of 0.5729 (`vision.jsd_face_flag` in [heads/manifest.json](../heads/manifest.json)), so the face raises no flag. Fusion3 (`stacked_tav`) gives anger with a calibrated confidence of 0.3869. That is below `low_conf` (0.56) of the vision mode, so the reply names no feeling and asks a question. The commit policy emitted neutral at 2.016 s after the segment start, before the endpoint, and that label does not match the final's. The final state left the service 233.97 ms after the endpoint frame arrived. The reply came from `Qwen2.5-0.5B-Instruct` in 4 deltas. The first left the service 399.08 ms after the endpoint frame arrived, and the generation took 370.82 ms. The reply passes every check of `check_reply`.
<!-- demo_tav: end -->

In this clip the face alone reads neutral, and so do the words. Of the three inputs, only the voice has anger as its top named class. SenseVoice puts most of its named mass on anger, and fusion3 gives anger, which is the MELD label of the clip. The commit policy committed neutral at 2.016 s, so the early label was wrong. The next provisional already read anger (0.4844 in the log), but the policy is not consulted again after its first emit. One clip shows the path through the system. It is not a measurement, and the test numbers are in 2.12.

## 1.9 Replies and grounding

The replies come from `meld_rt/response/bank.yaml`. The bank has 124 lines: 112 in 28 cells of four, 6 `hold_back` lines that make no emotion claim, 2 lines for a failed transcript, and 4 event prefixes. A cell is indexed by the fused label and the discrepancy direction (7 labels × 4 directions), and `mixed` shares the `none` cell. Each cell has `acknowledge` and `ask` lines, and each cell whose direction is not `none` also has `name_the_gap` lines. The loader rejects a line longer than 12 words, a line with a keyphrase slot longer than 8 words, and any exclamation mark. The lines address the speaker directly and use no contractions. A `name_the_gap` line says what the words and the tone disagree about (ex. "The words are reasonable. The tone is angry."). `tests/test_selector.py::test_bank_shape` checks the shape: 28 cells, an `acknowledge` and an `ask` line in every cell, `name_the_gap` lines where the direction is not `none`, at least two lines per cell without a slot, unique ids, and ASCII text.

The selection is a pure function of the state and the stream's recent lines.

1. A failed or low-confidence transcript takes one of the two "did not catch that" lines.
2. A confidence below `low_conf` (0.47, from the fusion head's dev predictions) takes a `hold_back` line.
3. Otherwise the cell gives the candidates. In the mid bucket the order of preference is `ask`, then `name_the_gap` when the direction is not `none`, then `acknowledge`. In the high bucket (a confidence of at least `high_conf`, 0.73, and `fused.calibrated` true) it is `acknowledge` first, or `name_the_gap` first when the direction is not `none`. An uncalibrated state never reaches the high bucket, and its grounding says `rule=uncalibrated_cap` when the cap changed the bucket.
4. A line used in the last 5 turns of the stream is avoided. When a cell has no fresh line left, a fresh `hold_back` line is used and the grounding says `rule=no_repeat`.
5. A line with a `{keyphrase}` slot takes the longest run of up to four content words from the transcript, as an exact substring (`meld_rt/discrepancy.py`). The run can be a clause fragment (ex. `the GR-6 system failed`), so every slot line puts it in single quotes.
6. When laughter or cry is above 0.5 in `audio.events`, one of two short clauses goes in front.

Ties break by a CRC32 of the utterance id and the line id, so a replay gives the same lines. The `grounding` list names the state fields that licensed the line, with their values (ex. `fused.label=neutral@0.55` and `discrepancy.direction=tone_more_negative` for the demo's line, `text.keyphrase='...'`, or `fused.confidence=` with the confidence, `<` and `low_conf` for a hold-back line).

MELD clips come from a sitcom with a laugh track, so a laughter event there can be the audience rather than the speaker. In front of the robot it would be the user's own laughter. The event features can therefore learn a signal that only MELD has, and the event ablation and the disagreement audit are the checks (1.11).

With `--reply llm` the reply is generated and streamed instead, and the bank line is its fallback (2.9). The bank stays the default.

## 1.10 Real time

The robot should react inside the pause after a person stops talking and show that it is listening while they talk. The targets below are measured at 4 threads from the arrival of the endpoint frame, and they include any provisional still running. They are the `TARGETS` of `tools/replay_bench.py`.

| what | target |
|---|---|
| final state, utterances up to 5 s | p50 at most 250 ms, p95 at most 400 ms |
| final state, 5 to 8 s | p50 at most 350 ms, p95 at most 550 ms |
| final state, 8 s to the 15 s cap | p50 at most 700 ms, p95 at most 1000 ms |
| reply after the final state | p95 at most 5 ms |
| provisionals in the first 4 s of speech | median gap of 1 s, within 0.1 s |
| sustained compute | at most 0.3 s per second of audio |

Over all three splits, 83.3 % of MELD utterances last at most 5 s, 95.2 % at most 8 s and 99.7 % at most 15 s (subtitle durations, [results/data_stats.md](../results/data_stats.md)). The VAD tail is not compute. It adds 512 ms at the default setting to what the user waits, and it is reported apart from the targets. The assumptions are one speaker near the microphone, 16 kHz mono audio, utterances up to 15 s, 4 CPU cores and no GPU.

[results/measurements.md](../results/measurements.md) has the container numbers, and every target passes there. All benches ran at 500 ms with the stacked fusion deployed. On the proxy clips, finals up to 5 s took 146.5 ms at p50 and 258.0 ms at p95 (n 19). On the long clips, finals from 8 to 15 s took 403.5 ms at p50 and 530.1 ms at p95 (n 9). On 200 MELD test clips, finals up to 5 s took 152.3 ms at p50 and 241.7 ms at p95 (n 168), and finals from 8 to 15 s took 371.2 ms and 491.3 ms (n 14). The MELD clips are cut at the end of the subtitle, and the VAD segments do not always line up with them. A final is attributed to the clip in which its speech stopped, or to the clip before when it stopped in the silence between two clips. By that rule, 30 of the 200 MELD clips have no final and 24 are split into several finals (the clip-attribution row of the same file), so the MELD buckets group finals by segment length. The laptop numbers are not measured yet.

## 1.11 Evaluation

The splits are the published ones. Model selection, temperatures, thresholds and k are chosen on dev only, and each test condition is run once. Weighted F1 is the primary metric because every published MELD row uses it. Macro F1, accuracy, per-class F1, the confusion matrix, a 95 % bootstrap interval over test dialogues (1000 resamples, seed 0) and the ECE (10 bins) sit beside it. `tools/evaluate.py` scores the conditions, and [RESULTS.md](../RESULTS.md) summarizes them.

| conditions | status |
|---|---|
| majority floor, text only at k = 0, at the best k and with the runtime-regime window | done |
| audio only from the raw SenseVoice tag, audio calibrators A and B, stacked fusion, product of experts with the dev weight and with the fixed weight, fused sentiment | done |
| ablations without events, without prosody and without both, the live condition on SenseVoice transcripts with WER and the service's ASR gate, the disagreement subset, early commit on 2 s and 4 s prefixes | done |
| text-only sentiment, from `heads/sentiment_text_only.npz` beside the fused head | done |
| the face alone, fusion3, sentiment3 and the commit policy of the extension | done (2.12) |

The audio and fused conditions would leave out test rows without a cached clip and report the text head on the same rows (`text_best_k_same_rows_weighted_f1` in [results/metrics.json](../results/metrics.json)). Every test row has a clip, so no row is left out.

The stacked fusion scores 60.77 against 59.62 for the text head on gold text, a small gain in weighted F1 with overlapping intervals. Its macro F1 is lower (39.24 against 41.22), and it scores 0.00 on fear against 20.29 for the text head. On the disagreement subset it scores 49.62 against 51.44 for the text head, so the evaluation gives no support for the claim that the audio branch helps where words and tone disagree ([docs/decisions.md](decisions.md), Fusion). On SenseVoice transcripts the text head scores 42.61 and the fusion 47.05. Dropping the event and prosody inputs leaves the fused score inside the deployed fusion's interval. [RESULTS.md](../RESULTS.md) has every row.

The last check is a listening audit: a person listens to 50 flagged test clips and marks which branch was right and where any laughter came from (`tools/audit_disagreement.py`, [docs/runbook_laptop.md](runbook_laptop.md)). The tool draws the sample from the cached test audio. The listening has not been done.

The published rows are the MELD paper's Tables 11 and 13, EmoBERTa's Table 2 and Table 8 of the FunAudioLLM report. [PROVENANCE.md](../PROVENANCE.md) quotes what that report says about the data behind SenseVoice's emotion labels. Because MELD may be in that data, the audio-only number is never called zero-shot.

## 1.12 Measurements

- `tools/count_params.py` counts the ONNX initializers and the `.npz` head arrays and writes [results/params.md](../results/params.md) and [results/params.json](../results/params.json).
- `tools/bench_components.py` times each module at 1, 2 and 4 threads on slices and tilings of `en.wav` and writes `results/bench_components_<host>.json`.
- `tools/pause_stats.py` runs the service's VAD over WAV files and writes `results/pause_stats_<label>.json`: every pause inside speech, and the clips that each endpoint setting splits.
- `tools/replay_bench.py` streams clips at real-time pace through the real service, with a monitor in the same process, and writes `results/replay_bench_<host>.json`: latency by length bucket, the reply delay, the provisional cadence, the compute per second of audio, RSS and CPU.
- `tools/bench_generator.py` times the reply generator on 200 test states and writes `results/generator_bench_<host>.json` (2.9). `tools/replay_bench.py --video` streams MP4 clips with their frames (2.11).
- `tools/render_measurements.py` renders [results/measurements.md](../results/measurements.md) from those files, groups them by the host that ran them, and checks each target.

The container recorded an "Intel(R) Xeon(R) Processor @ 2.10GHz" with nproc 4, 16093 MB of memory, AVX-512 VNNI and AMX, and no GPU ([results/bench_components_vm.json](../results/bench_components_vm.json)). [docs/runbook_laptop.md](runbook_laptop.md) gives the laptop steps. Nothing in the laptop section is typed by hand.

## 1.13 Left out

- A live camera session. `meld_rt/clients/mic.py --camera` exists, but the build machine has no camera, so the face path has only run on MELD clips.
- Merging the fragments of one face. A track ends when it misses one frame, so one face can give several tracks. In the replay bench with video, the window of a final with a face held 3.0 tracks at p50 and 73.0 at most, counting the fragments of every person in the frame (`n_tracks_when_present` in [results/replay_bench_vm_tav.json](../results/replay_bench_vm_tav.json)). The speaker rules choose among these fragments.
- Fine-tuning RoBERTa, SenseVoice, wav2vec2 or HuBERT. It needs GPU hours or long CPU runs, and a reviewer could not rerun it in minutes.
- A text head trained on SenseVoice transcripts of train and dev. The deployed head is trained on gold text, and the live ASR rows score it on transcripts ([RESULTS.md](../RESULTS.md)).
- Context models that read future utterances, because the live system only has the past.
- Arousal and valence fields. MELD has no labels to score them.
- Reinforcement learning on the reply. MELD has no reward for a reply. The RL component is the commit policy (2.8).
- A spoken reply. Piper is present but only makes the proxy clips.
- Speaker identification. One user in front of the robot is assumed. The speaker rule of 2.2 picks a face in the frame and does not identify anyone.

## 2 The three-input extension

`--vision` turns on the video input, the face, fusion3 and the commit policy. `--reply llm` turns on the streamed reply. Without either flag the service publishes the topics and keys of part 1 and nothing else (`tests/test_service_vision.py::test_core_service_publishes_the_core_topics_and_keys_only`). Every model choice that reads labels was made on dev, and the claim rules for the test runs were fixed before them (2.12).

## 2.1 Face expert

`meld_rt/experts/vision.py` runs two OpenCV zoo models in OpenCV DNN on each frame. YuNet finds the faces on a copy of the frame scaled down to a long side of 480 px and keeps at most 6 with a score of at least 0.6, the largest first. Each face is warped to 112 × 112 by the similarity that maps its five landmarks onto the template of the expression model. That model is MobileFaceNet, taken to be trained on RAF-DB ([PROVENANCE.md](../PROVENANCE.md)). It gives 7 logits, one per MELD class, and the 512-value input of its last layer, used as an embedding. Tracks link faces across frames by greedy IoU matching at 0.3 or more, and a track that misses one frame ends. For each face the expert also gives a mouth motion and a face motion: the mean absolute difference to the same track's previous aligned gray face, over the mouth box and over the whole face.

Training and run time use the same frame rate and width. `tools/extract_video_features.py` ran the expert on every MELD clip at 10.0 frames per second, scaled to 640 px wide, and wrote `cache/video_{split}.npz` (not committed). The replay client sends frames at the same rate and width. In the replay bench with video, the vision thread spent 30.49 ms per processed frame on average over 7044 frames, the JPEG decode and the row building included ([results/measurements.md](../results/measurements.md)).

## 2.2 Speaker rule

A MELD frame often shows several people, so the speaker's face has to be chosen. Four rules were tried over the frames of an utterance.

- `largest`: the track with the largest mean area.
- `present`: the track seen in the most frames.
- `asd`: the track whose mouth motion has the highest Pearson correlation with the log RMS envelope of the audio, among tracks with at least 4 motion samples, else `largest`.
- `mouth`: the track with the highest mean of mouth motion minus face motion, among tracks with at least 3 samples, else `largest`.

`tools/train_vision.py` fitted calibrator A (2.3) on train under each rule and kept the rule with the best dev weighted F1 over all dev rows. It kept `asd`. The four rules differ by less than one point on dev. They do pick different tracks: 81.50 % of the 1103 dev clips with a face have more than one track, and on 67.09 % the four rules do not all choose the same one ([RESULTS.md](../RESULTS.md), "Choices made on dev"). A camera at the robot that sees one person gives the same answer under every rule.

| rule | dev weighted F1, all rows | fallback rate on dev clips with a face (%) |
|---|---|---|
| `largest` | 25.19 | 0.00 |
| `present` | 25.57 | 0.00 |
| `asd` (deployed) | 25.82 | 4.08 |
| `mouth` | 25.53 | 3.26 |

## 2.3 Face evidence and calibrator

`meld_rt/vision_features.py` holds the functions that the offline tools and the service both call. `face_evidence` takes the face rows of a window, the audio envelope and the rule. It returns the chosen track with the mean of its per-frame expression posteriors in MELD order (`vision.raw` in the state), its frame count, its presence (its frames over the processed frames of the window), the log of its mean area, the number of tracks, the `asd` correlation, the mean detector score, the mean mouth motion and the mean embedding. Without a face in the window the posterior is uniform. Offline the window is the whole clip, and in the service it is the VAD segment. `tests/test_service_vision.py::test_live_rows_and_the_video_cache_give_the_same_features` checks that the service's rows and the video cache give the same features under all four rules.

The calibrator is a logistic regression with a temperature fitted on dev. Variant A reads 16 values: the 7 log posteriors, their maximum and entropy, the presence, the log area, the number of tracks divided by 4, the `asd` correlation, the score, the mouth motion times 10 and a face flag. Variant B adds the 512 values of the mean embedding. Both are fitted on the train clips with a face, over the C and class-weight grid of `configs/heads.yaml`. On dev, B scores 30.20 weighted F1 against 25.82 for A, so B is deployed, with C 10, no class weights and a temperature of 1.3566 (`vision.calibrator` in [heads/manifest.json](../heads/manifest.json)). Its output is `vision.emotion` in the state.

The words-face discrepancy follows the words-tone rule of 1.3. It compares the text posterior with the raw expression posterior (`vision.raw`), as the words-tone JSD uses the raw SenseVoice posterior, and its directions are `face_more_negative`, `face_more_positive`, `face_surprised` and `mixed`. The flag threshold is 0.5729, the 0.8 quantile of the dev JSD on the dev clips with a face (`vision.jsd_face_flag`).

## 2.4 Fusion3

Fusion3 (`heads/fusion3.npz`) is the stacked fusion of 1.6 with the face added. It is a logistic regression over 29 values: the log posteriors of text, audio and face (7 each), their three entropies, the ASR confidence, laughter, the unknown mass, a face flag and the presence. Without a face, fusion3 still runs, with the uniform face posterior and the last two values at 0. The three expert heads are refit in the 5 folds by dialogue of `tools/train_heads.py`, and fusion3 trains on their out-of-fold train posteriors. Refit on these folds, the two-way fusion comes out equal to the deployed one array for array (`reproduces_deployed_fusion` in `vision.cross_fit`). Dev chose C 10 without class weights, and a dev temperature of 0.8428 calibrates it. Fusion3 scores 59.53 weighted F1 on dev against 57.95 for the two-way fusion. Sentiment3 (`heads/sentiment3.npz`) is the same regression over the three sentiment classes and scores 67.77 on dev ([results/RESULTS_vision.md](../results/RESULTS_vision.md)).

In vision mode `fused.mode` is `stacked_tav`. The rule of 1.6, applied to fusion3's calibrated dev predictions, gives `low_conf` 0.56 and `high_conf` 0.73 (`vision.thresholds`), and the selector and the reply prompt use these.

## 2.5 Video input and media clock

A `video` message carries one JPEG frame (quality 80) and the header `stream_id`, `seq`, `t_capture_ns`, `t_media`, `w`, `h` and `fmt` (`jpeg`). `t_media` is the frame's time in seconds on the media clock of 1.4. The replay client (`meld_rt/clients/replay.py --video clip.mp4`) decodes the audio and the frames of one MP4 file with ffmpeg and sends both at real-time pace on one schedule: audio frame i at i × 32 ms and video frame k at k / fps (10.0 by default), its presentation time. The audio and the frames of a clip are both taken to start at media time 0, and the offset between them was not measured. The live client (`meld_rt/clients/mic.py --camera N`) stamps a frame with its capture time minus the capture time of the first audio sample sent after `start`, both on the client's monotonic clock. The two streams then share the media clock up to the difference between the camera's and the microphone's capture latency. In vision mode a gap in the audio sequence moves the audio media clock forward by the missing frames, so audio and video stay on one clock after the gap.

`state.commit` carries an early label (2.8), and `response.delta` carries one word of a streamed reply (2.9). A `video` message to a service without `--vision` gets `bad_message`, once per stream, and opens no stream.

## 2.6 Vision thread

A fourth thread owns one face expert per stream, so the ingest thread never decodes a frame and the worker never runs a face model. Ingest puts each video message on a queue of at most 4 frames. When the queue is full, the oldest frame is dropped and counted (`dropped_video_frames` in `health`). The vision thread skips a frame whose media time is less than 0.75 / `--video-fps-max` (10.0) after the last processed frame of its stream. That keeps 10 fps frames with capture jitter and skips frames from a faster camera. The thread keeps the face rows of the last 60 s of media time, and the ingest thread keeps the log RMS of every audio frame for the `asd` rule.

At a final the worker waits until the vision thread has processed a frame at `t_media_end` minus 0.1 s, or until the stream has no frame queued or in progress, for at most `--vision-lag-ms` (150.0). A wait that reaches the limit adds `vision_lag` to `degraded`. The second condition exists because the replay sends no video during the silence after a clip and MELD clips end at the end of the subtitle, so for many finals the frame at `t_media_end` minus 0.1 s never comes. The worker then computes the face evidence from `t_media_start` to `t_media_end`, the calibrated face posterior, fusion3 and sentiment3. The new `degraded` codes are `no_face`, `vision_lag`, `vision_unavailable`, `vision_failed` and `no_vision_head`.

## 2.7 Provisionals in vision mode

In vision mode the full-window provisionals of 1.3 run on the segment grid, at the first audio frame that ends at or after 1, 2, 3 and 4 s from the segment start (1.024, 2.016, 3.008 and 4.0 s in `tests/test_service_vision.py::test_a_long_utterance_gets_commit_ticks_1_2_3_4`). The fourth is still a full window. Each runs SenseVoice as in the core, the text expert on the partial transcript when its ASR confidence is at least 0.3 and it is not empty (uniform otherwise), the audio calibrator with the prosody at zero except the rate, and the face evidence from the segment start to the provisional's end. A provisional does not wait for the vision thread. Fusion3 fuses the three, and `stability` compares it with the previous fusion3 provisional. Tail-window provisionals keep the audio-only fused posterior of the core and get the `vision` block over their window. The full-window provisionals are the decision points of the commit policy.

In the replay bench with video, all 538 full-window provisionals ran fusion3, their median gap was 0.99 s, none was dropped, and the text expert on the partial transcript took 41.56 ms at p50 (`results.provisionals` and `results.vision` in [results/replay_bench_vm_tav.json](../results/replay_bench_vm_tav.json)).

## 2.8 Commit policy

The commit policy decides at each full-window provisional whether to commit the fused label now or to wait for the next one. When it never commits, the final is the commit. The reward is 1 for a correct label minus c times the commit time in seconds from the segment start. This is an optimal-stopping problem, and `tools/train_commit_policy.py` solves it offline by fitted Q-iteration.

- Q_emit is P(correct) minus c times the tick time. P(correct) is a logistic regression on the standardized tick features, with C 0.01 chosen by dev log loss. It does not depend on c.
- Q_wait is regressed on the value of the next tick, the larger of its Q_emit and Q_wait, or on the endpoint reward after the last tick. The regression is repeated until the commit decisions on the train ticks stop changing, at most 10 times. One model covers all ticks. The candidates are ridge regression and a one-hidden-layer MLP with 16 units, chosen for each c on dev reward.
- The policy commits when Q_emit is at least Q_wait.

The 14 features (`FEATURES` in `meld_rt/commit.py`) are the maximum, the margin between the top two and the entropy of the fused posterior, the tick time and index, the stability against the previous tick, the run of ticks with the same label, the maxima of the text, audio and face posteriors, the face flag, how many modalities agree with the fused label (a uniform posterior does not count), the word count divided by 10 and the ASR confidence. The trainer and the service build them with the same code.

The traces (`tools/build_commit_traces.py`, `cache/commit_traces_{split}.npz`, not committed) run every MELD clip through the service's experts at t = 1, 2, 3 and 4 s while the segment is open (t below the clip length plus the 0.512 s endpoint silence): SenseVoice on the audio prefix, the text head on its transcript, the face over [0, t) and fusion3. Train ticks use the fold heads of 2.4, so their posteriors are out of fold, and dev and test use the deployed heads. The endpoint comes at the clip length plus 0.512 s plus 0.152 s, the p50 final latency of utterances up to 5 s on the MELD replay bench (1.10). Train has 25384 ticks of 9988 clips ([results/commit_policy.md](../results/commit_policy.md)). Three baselines are tuned on dev for every c: commit at the endpoint, at a fixed time T of 1 to 4 s, or at the first tick whose fused maximum reaches a threshold tau from 0.3 to 0.95. The deployed policy (`heads/commit_policy.npz`, `fqi_mlp_c0.05`) uses c = 0.05, so one more second of waiting raises the reward only if it raises the accuracy by more than five points. Dev chose the MLP for Q_wait at that cost, and its iteration stopped at the cap of 10 before the decisions stopped changing.

In the service the policy runs at every full-window provisional where fusion3 ran. On its first commit the service publishes `state.commit` with the label, the fused confidence, `t_commit_s`, `q_emit`, `q_wait` and the policy name, and it does not ask the policy again for that utterance. The final carries the `commit` block (2.10). Live, the tick time (`t_s`) counts from the VAD segment start, 256 ms before the detected speech onset (1.3), while the traces count from the clip start. The VAD can also split or merge clips, while offline one clip is one segment.

## 2.9 Streamed reply

With `--reply llm` the worker hands each final to a reply thread, so the next utterance is not blocked. The thread runs Qwen2.5-0.5B-Instruct (the onnx-community int8 export) in onnxruntime with a kv cache, greedy, with at most 40 new tokens and 2 threads by default (`--llm-threads`). Each word goes out as a `response.delta` once it has passed the guards below, and a `response` ends the reply with its text, grounding, checks, model and timings. The prompt, the masks and the checks are in `meld_rt/response/generator.py`.

The system prompt sets the role (a small desk robot that has heard the person and may have seen their face), the length (one or two short sentences, at most 20 words), the register (speak to them as "you", no exclamation marks or emoji, no mention of the notes, models, numbers or probabilities) and the rule to use only the notes. Three fixed examples follow it. The user turn gives the evidence in plain sentences: the transcript in quotes, what the words, the voice and the face suggest (or that no face was seen), any words-voice or words-face disagreement, the fused label with a confidence word (low, medium or high, from the thresholds of 2.4), and last what the reply may do. A modality suggests a class when its top class has a probability of at least 0.4. The voice line reads the raw SenseVoice posterior with OTHER and EMO_UNKNOWN in the denominator, so it seldom names a feeling: its top named class reaches 0.4 on a share of 0.0333 of the test clips (`states.voice_raw_all_clips` in [results/generator_bench_vm.json](../results/generator_bench_vm.json)). The system prompt is the same for every reply, so its kv cache is computed once and reused.

The reply may name only the classes in `allowed`: the fused label, and the top class of each modality whose top class has a probability of at least 0.4. Below `low_conf`, or when the transcript was not caught, `allowed` is empty, the prompt gives no label and asks for one short question, and the grounding starts with the confidence and `low_conf`, as for a hold-back line of the bank. `grounding` lists every state field that went into the prompt, in the format of 1.9.

`meld_rt/response/lexicon.py` lists the emotion words of each MELD class with their inflections, and neutral has none. At every step the decoder masks the tokens that are whole emotion words of a class outside `allowed`, with and without a leading space, in lower, capitalized and upper case. Many emotion words split into several tokens, so a word guard also checks each finished word: a forbidden word rolls the kv cache back to its first token, bans that token there and decodes again. Phrases copied from the prompt's notes and runs of 4 transcript words are rolled back the same way, on a budget of their own. Each budget allows 6 rollbacks, and a reply whose budget is spent ends at the last word that passed. A register mask removes tokens with "!", control bytes, non-ASCII bytes and special tokens other than the two stop tokens. A word is sent only after it passes, so nothing sent is taken back.

`check_reply` runs on the final text, and its result is `response.checks`: the word count and whether it is over 20, an exclamation mark, emotion words of classes outside `allowed`, an empty reply, meta words such as "model" or "probability", phrases copied from the prompt, and non-ASCII or non-printable characters. The checks test the form of a reply. They do not test whether the reply follows from what the person said.

The bank line of 1.9 is sent as one delta when the generator is missing or fails, when its first word does not come within `--llm-first-token-timeout-s` (2.0 s), when the reply comes back empty, or when a newer final of the same stream is already waiting. A reply still streaming after `--llm-reply-timeout-s` (30.0 s) ends with the words already sent. With `--reply bank` and the extension on, the bank line also goes out as one `response.delta`, so a client that reads deltas works in both modes.

The model was chosen by a rule fixed before the bench on 200 states: the largest candidate with a file whose first-token p95 is at most 350.0 ms and whose decode rate is at least 15.0 passes per s at 4 threads, and within one model the int8 file unless q4 is faster by more than 1.2x at the same reply length with a check pass rate no lower. These definitions of the decode rate and of a faster q4 were set after preliminary benches on 20 states, and the original wording picks the same file ([docs/decisions.md](decisions.md), Streamed reply). Every file replied to the same 200 test states ([results/generator_bench_vm.json](../results/generator_bench_vm.json), [results/generator_bench_vm_1.5b.json](../results/generator_bench_vm_1.5b.json)).

| file, 4 threads | first token p50 / p95 ms | decode passes per s | check pass rate | meets both targets |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct int8 | 70.94 / 100.27 | 28.47 | 0.97 | yes |
| Qwen2.5-0.5B-Instruct q4 | 155.02 / 196.14 | 26.87 | 0.995 | yes |
| Qwen2.5-1.5B-Instruct int8 | 149.35 / 194.54 | 13.22 | 0.97 | no |
| Qwen2.5-1.5B-Instruct q4 | 473.33 / 580.74 | 11.64 | 0.995 | no |

Neither 1.5B file reaches the decode rate. Within the 0.5B model, q4 is slower than int8 at the same reply length ("q4 speedup 0.80x on a reply of 9.82 tokens", the reason recorded in the file), so the deployed file is the 0.5B int8 one. At the service's 2 threads the same bench gives it a first-token p95 of 136.69 ms and 20.09 passes per s. The bank stays the default reply. Its lines and their grounding are unit-tested, and it answers within the 5 ms target of 1.10.

## 2.10 State additions

In vision mode every state carries these fields. Without the extension none of them exists, and the core JSON of 1.8 is unchanged.

| field | on | content |
|---|---|---|
| `vision` | provisionals and finals | `emotion` (the calibrated face posterior, 7 MELD keys), `raw` (the mean expression posterior of the chosen track), `present`, `track`, `rule`, `n_frames`, `presence`, `n_tracks`, `asd_corr`. Null for a typed `text` message |
| `commit` | finals | `label`, `confidence`, `t_commit_s`, `matches_final`, `source` (`policy`, or `endpoint` when the final is the commit and `t_commit_s` is the segment length plus the VAD tail). Null without a commit policy |
| `discrepancy.face_jsd`, `face_flag`, `face_direction` | provisionals and finals | the words-face JSD, flag and direction of 2.3. The JSD is null without a face or a text posterior |
| `timing_ms.vision_wait`, `timing_ms.vision_aggregate` | provisionals and finals | the wait for the vision thread and the face evidence, in ms |
| `fused.mode` | provisionals and finals | `stacked_tav` when fusion3 ran |

With `--reply llm` the `response` also carries `source` (`llm` or `bank`), `fallback`, `model`, `checks`, `n_tokens`, `first_delta_ms` (from the arrival of the endpoint frame), `gen_ms`, `stop_reason` and `masked_steps`. With the extension on, `health` adds the vision, reply, generator and commit policy status and, per stream, the video frames received, dropped, skipped and bad and the mean vision time per frame. The demo of 1.8 shows a vision final and its response.

## 2.11 Real time with video

Two replay benches streamed the first 200 MP4 files of the MELD.Raw test split in sorted order, with their frames at 10.0 fps, at real-time pace, with `--vision` and the commit policy on. 198 of them are clips of the audio bench of 1.10, and 2 (`dia108_utt1`, `dia108_utt2`) are files with no row in the test CSV. One ran with `--reply llm` and the other with `--reply bank`, which separates the generator's cost ([results/measurements.md](../results/measurements.md), from [results/replay_bench_vm_tav.json](../results/replay_bench_vm_tav.json) and [results/replay_bench_vm_tav_bank.json](../results/replay_bench_vm_tav_bank.json)). Every final-state target of 1.10 passes in both.

| final state after the endpoint frame | target p50 / p95 ms | `--reply llm`, p50 / p95 ms (n) | `--reply bank`, p50 / p95 ms (n) | audio only (1.10), p50 / p95 ms (n) |
|---|---|---|---|---|
| up to 5 s | 250 / 400 | 178.4 / 347.7 (175) | 172.1 / 392.8 (175) | 152.3 / 241.7 (168) |
| 5 to 8 s | 350 / 550 | 336.2 / 536.6 (18) | 340.6 / 509.4 (18) | 282.9 / 376.5 (19) |
| 8 s to the 15 s cap | 700 / 1000 | 651.9 / 809.5 (13) | 576.0 / 875.6 (13) | 371.2 / 491.3 (14) |

The finals are slower than in the audio bench. Two p95 cells are near their targets: 392.8 against 400 ms (up to 5 s, `--reply bank`) and 536.6 against 550 ms (5 to 8 s, `--reply llm`). The vision thread, and with `--reply llm` the reply thread, run beside the worker on the same 4 cores. In the longest bucket the SenseVoice stage takes 525.7 ms at p50 with `--reply llm`, against 280.5 ms in the audio bench.

| item | `--reply llm` | `--reply bank` |
|---|---|---|
| reply after the final state, p50 / p95 ms | 549.92 / 1333.29, the whole generation | 0.24 / 0.68 |
| first streamed word after the endpoint frame, p50 / p95 ms | 388.85 / 855.99 | |
| replies from the generator | 206 of 206, no fallback | |
| labels committed before the endpoint | 168 of 206 finals | 168 of 206 finals |
| commit time of those, p50 / p95 s from the segment start | 1.02 / 2.66 | 1.02 / 2.66 |
| committed label equal to the final's | share 0.767 | share 0.767 |
| vision thread per processed frame, mean | 30.49 ms over 7044 frames | 29.72 ms over 7044 frames |
| video frames dropped / skipped | 0 / 0 | 0 / 0 |
| face evidence at the final, p50 / p95 ms | 1.35 / 3.04 | 1.21 / 2.44 |
| worker compute per second of audio (target at most 0.3 s) | 0.1549 s | 0.1507 s |
| whole process, cpu s per second of audio | 1.0986 | 0.8081 |
| peak RSS of service, client and monitor | 3422.7 MB | 1644.9 MB |

No target for the streamed reply was set before the run, so its first word and its whole reply are reported without a pass or a fail. The bank keeps the 5 ms target of 1.10 and passes it. Worker compute leaves out the vision and reply threads, as in the audio benches. The process cpu includes them and the client's MP4 decoding. In the audio bench it was 0.458 cpu s per second of audio.

## 2.12 Evaluation

The extension follows the protocol of 1.11: choices on dev, each test condition run once, and 95 % intervals over test dialogues (1000 resamples, seed 0). `tools/evaluate_vision.py` writes [results/RESULTS_vision.md](../results/RESULTS_vision.md), and `tools/train_commit_policy.py` writes [results/commit_policy.md](../results/commit_policy.md). Both claim rules were fixed before the runs. The face adds to text and audio only if the paired 95 % interval of fusion3 minus the two-way fusion on gold-text test lies above zero. The policy adds measured value over a baseline at a cost c only where the paired interval of the test reward difference lies above zero.

| condition, MELD test | n | weighted F1 [95 % CI] | macro F1 | accuracy |
|---|---|---|---|---|
| majority class ([RESULTS.md](../RESULTS.md)) | 2610 | 31.27 [28.62, 34.31] | 9.28 | 48.12 |
| face alone, raw expression label | 2610 | 32.22 [29.77, 34.89] | 15.92 | 34.44 |
| face alone, calibrator B | 2610 | 34.37 [31.83, 37.32] | 12.70 | 47.05 |
| two-way fusion, gold text | 2610 | 60.77 [58.65, 63.00] | 39.24 | 63.98 |
| fusion3, gold text | 2610 | 61.56 [59.45, 63.79] | 40.50 | 64.60 |
| two-way fusion, SenseVoice transcripts | 2610 | 47.05 [44.53, 49.57] | 27.50 | 52.53 |
| fusion3, SenseVoice transcripts | 2610 | 47.96 [45.60, 50.40] | 28.82 | 53.18 |

Calibrated, the face alone scores 34.37 weighted F1 against 31.27 for the majority class, with an accuracy below the majority class's (47.05 against 48.12) and an F1 of 0.00 on fear, sadness and disgust. A face was found in 99.46 % of the test clips. Added to text and audio, the face helps by a small margin. Fusion3 scores 0.80 points above the two-way fusion on gold text, with a paired interval of [0.20, 1.48], so by the rule the face adds measured value. On SenseVoice transcripts the difference is 0.91 [0.31, 1.52]. Fusion3 changes 80 of the 2610 gold-text labels of the two-way fusion, 36 from wrong to right and 20 from right to wrong. On the face-disagreement subset (557 test clips whose words-face JSD is above 0.5729), fusion3 scores 64.04 against 62.87 for the two-way fusion and 64.88 for the text head. Both paired intervals include zero ([-0.29, 2.60] and [-3.71, 2.20]), so no gain is shown where the face and the words disagree. Sentiment3 scores 69.03 against 69.17 for the two-way sentiment head.

The commit policy against each baseline, as the test reward difference with its paired 95 % interval:

| c | policy minus endpoint | policy minus fixed time | policy minus threshold |
|---|---|---|---|
| 0.02 | 0.0346 [0.0234, 0.0471] | 0.0124 [0.0017, 0.0238] | 0.0051 [-0.0035, 0.0132] |
| 0.05, deployed | 0.1189 [0.1004, 0.1402] | -0.0014 [-0.0113, 0.0088] | 0.0021 [-0.0058, 0.0093] |
| 0.1 | 0.2717 [0.2418, 0.3074] | -0.0010 [-0.0064, 0.0049] | 0.0013 [-0.0037, 0.0065] |
| 0.2 | 0.5775 [0.5245, 0.6464] | 0.0000 [0.0000, 0.0000] | 0.0060 [0.0016, 0.0104] |

By the rule, the policy adds measured value over the endpoint at every cost, over the tuned fixed time at c = 0.02 only and over the tuned threshold at c = 0.2 only. At the deployed c = 0.05 the gain over the fixed time and the threshold is not shown, and committing at 1 s has a slightly higher reward (0.4498 against 0.4484). At c = 0.05 the policy commits 86.74 % of the test labels before the endpoint, at a mean of 1.362 s against 4.046 s. The earlier labels are right less often: an accuracy of 51.65 against 53.18 at the endpoint, and a weighted F1 of 41.81 against 47.96. Charging every tick commit 0.152 s more, as the endpoint is charged for the final's computation, changes none of these outcomes.

## Repository layout

```
meld_rt/                   the package
  service.py               ingest, worker and publisher threads, vision and reply threads (2.6, 2.9), CLI
  protocol.py              wire format, topics, clocks
  state.py                 the state dataclasses (1.8, 2.10)
  labels.py                class orders, SenseVoice tokens, valence
  heads.py                 numpy forward passes of the trained heads
  discrepancy.py           JSD, flag, direction, keyphrase
  data.py                  MELD CSV loading and context rows
  metrics.py               F1, ECE, bootstrap intervals
  vision_features.py       face rows, speaker rules, face evidence, fusion3 inputs (2.2 to 2.4)
  commit.py                tick features and the numpy commit policy (2.8)
  experts/                 sensevoice.py, text.py, vad.py, prosody.py, vision.py
  response/                bank.yaml (124 reply lines), selector.py, generator.py, lexicon.py
  clients/                 replay.py, mic.py, monitor.py
tools/
  fetch_models.py          download and verify the models (--groups core,vision,llm for the extension), write results/provenance_hashes.json
  fetch_meld_csv.py        download the MELD CSVs into meld/
  meld_raw_source.py       hub revision, size, sha256 and licence of MELD.Raw, results/meld_raw_source.json
  stream_meld_raw.py       stream MELD.Raw.tar.gz through extract_audio.py without unpacking it
  extract_tokenizer.py     vocab.json and merges.txt from the spaCy wheel
  edit_sensevoice_graph.py add the encoder_out output (model_enc.int8.onnx)
  make_proxy_clips.py      Piper TTS of MELD dev text into proxy/
  embed_text.py            RoBERTa embedding cache, cache/text_{split}.npy
  extract_audio.py         MELD clips to 16 kHz WAV under audio/
  cache_audio_features.py  SenseVoice and prosody cache, cache/audio_{split}.npz
  train_heads.py           heads/*.npz and heads/manifest.json
  evaluate.py              test conditions, results/RESULTS.md, results/metrics.json
  report_text_only.py      results/01_text_only.md
  render_results.py        RESULTS.md
  render_demo_docs.py      the demo blocks of README.md and 1.8 from results/demo_log.jsonl and results/demo_tav_log.jsonl
  pick_demo_clip.py        the demo clip by the rule written before the choice, results/demo_clip.json
  audit_disagreement.py    sample flagged test clips for a listening audit, count the marks
  meld_stats.py            results/data_stats.md
  count_params.py          results/params.md, results/params.json
  bench_components.py      results/bench_components_<host>.json
  pause_stats.py           results/pause_stats_<label>.json
  replay_bench.py          results/replay_bench_<host>.json, with --video for the extension
  render_measurements.py   results/measurements.md
  extract_video_features.py  per-frame face features of the MELD clips, cache/video_{split}.npz
  train_vision.py          heads/vision_cal.npz, fusion3.npz, sentiment3.npz, the manifest's vision section
  evaluate_vision.py       results/RESULTS_vision.md, results/metrics_vision.json
  build_commit_traces.py   prefix traces, cache/commit_traces_{split}.npz
  train_commit_policy.py   heads/commit_policy.npz, results/commit_policy.md, results/commit_policy.json
  bench_generator.py       results/generator_bench_<host>.json and the model choice
configs/heads.yaml         training grid and threshold rules
heads/                     trained heads (.npz) and manifest.json
assets/roberta_tokenizer/  vocab.json, merges.txt
results/                   every measured number, written by the tools above
tests/                     pytest suite
docs/                      design.md, decisions.md, runbook_laptop.md
demo.sh                    one utterance through the system
demo_tav.sh                one MELD clip with video through the extension
```

`models/`, `meld/`, `cache/`, `audio/`, `proxy/`, `bench_clips/`, `verify_dl/` and `papers/` are created by the tools or downloaded, and they are not committed. The MELD MP4 files that `demo_tav.sh` and the video bench stream are kept outside the repository.
