# Design

This document describes the system as built. [docs/decisions.md](decisions.md) gives the reason for each choice, the alternatives and what would change it. Every number below comes from a file that a script wrote into `results/` or `heads/`, from a config file, from the code, or from a paper, and the source is named. Code comments cite the sections by number (ex. "design 1.10").

## 1.1 Track and framing

The challenge says that tone changes how words should be understood. The system makes that relation part of its output. For each utterance in a 16 kHz audio stream it publishes a state with a text posterior over the seven MELD emotions, an audio posterior from the voice, how far the two disagree and in which direction, a fused label with a confidence, and a short reply chosen from that state.

The track is text and audio. Vision is not used. The MELD authors left visual features out of their own baselines, "as video-based speaker identification and localization is an open problem" (MELD paper, Section 4.1).

All inference runs locally on the CPU through onnxruntime. Nothing is sent to a remote service.

## 1.2 Processes and threads

The package is `meld_rt`. Three processes talk over ZeroMQ on localhost.

1. A client pushes audio. `meld_rt/clients/replay.py` reads a WAV file at real-time pace (`--realtime`) or as fast as possible, or sends one typed utterance. `meld_rt/clients/mic.py` captures from a microphone through sounddevice. Both send 512-sample (32 ms) PCM16 frames stamped with the client's capture time.
2. The service (`meld_rt/service.py`) is one process with three threads. The ingest thread pulls frames, runs Silero VAD on every frame and applies the provisional policy (1.3). It hands jobs to the worker through a FIFO queue. The worker thread runs the experts, the heads, the discrepancy rule and the reply selector one job at a time, so SenseVoice can use every core. The publisher thread owns the PUB socket, stamps the emission time and sends a health message once per second.
3. The monitor (`meld_rt/clients/monitor.py`) subscribes, prints one line per message with its timings and appends every message to a JSONL file. A robot body would subscribe the same way.

The only audio store is the VAD segmenter of each stream. It holds the frames of the open segment, and a ring of 16 frames (pre-roll plus the start run) while no segment is open.

SenseVoice runs with 4 intra-op threads (`--threads`), RoBERTa with 2 (`--text-threads`) and the VAD with 1 on the ingest thread.

The runtime dependencies are in `pyproject.toml`: numpy, onnxruntime, kaldi-native-fbank (the front end), tokenizers, pyzmq, pyyaml (the reply bank and the head config), soundfile (WAV input and output), scikit-learn (head training, and it brings scipy, which the clients use to resample), psutil (health RSS and the benches), jiwer (WER) and onnx (the graph edit and the parameter count). The dev group adds pytest, msgpack (tokenizer extraction), sherpa-onnx (the transcript parity test and the Piper proxy clips) and praat-parselmouth (prosody). Without parselmouth the prosody scalars are zero except the speech rate. The `mic` extra adds sounddevice. The `finetune` extra (torch, transformers) is declared, but no script uses it.

## 1.3 One utterance, start to finish

1. End-pointing. A segment opens after 8 consecutive frames (256 ms) with a speech probability above 0.5. The count is the larger of 3 and `min_speech_ms` (250) divided by 32 ms and rounded up. The segment audio starts 8 frames (256 ms, `preroll_ms` 250 rounded up to whole frames) before the first of those frames, so `t_media_start` lies 256 ms before the speech onset. The segment closes after a run of frames below 0.35 that lasts `--endpoint-silence-ms`, rounded up to whole frames: 13 frames (416 ms) at the default 400 ms and 16 frames (512 ms) at 500 ms. `t_media_end` is the time the speech stopped, and the segment audio keeps the silence tail. A segment that reaches 15 s (`--max-segment-s`) is closed by force, and the next one carries `continued_from`.
2. Provisionals. While a segment is open, a provisional state is computed at most once per second (`--provisional-hz`) and only when the worker is idle and nothing is queued. While the segment is at most 4 s long (`--provisional-window-s`), the whole segment is decoded and `text.partial` holds its transcript. After that, provisionals come at half the rate on the trailing 4 s window, with `window` `"tail"` and no partial text. Provisionals run SenseVoice only. `--provisional-budget-ms` (0, off by default) skips the next provisional when the last one took longer than the budget. Queued provisionals are dropped at the endpoint.
3. Audio expert. At the endpoint the worker computes the fbank, LFR and CMVN front end and runs SenseVoice-small int8 once over the segment. That one pass gives the greedy CTC transcript, an ASR confidence (the mean of the maximum softmax over the emitted tokens), the language, the softmax over the nine emotion tokens at logit row 1, the softmax over the eleven event tokens at row 2, and the tokens per second. With `--encoder-states` and the edited graph from `tools/edit_sensevoice_graph.py`, the same pass also gives the encoder states, pooled (mean and maximum) to 1024 values. Prosody adds four scalars from parselmouth: the F0 median and the F0 range in semitones, the standard deviation of the intensity in dB, and the speech rate.
4. Text expert. RoBERTa-base encodes the transcript (at most 64 pieces), and the mean over the non-special positions gives 768 values. It does not run when the ASR failed or its confidence is below 0.3.
5. Heads. The TextContextHead gives the text posterior. The AudioCalibrator gives the audio posterior when `heads/audio_cal.npz` exists. Until then the audio posterior is the SenseVoice posterior renormalized over its seven named classes. How the fused posterior is made depends on `fused.mode` (1.6). The SentimentHead gives the sentiment.
6. Discrepancy. `discrepancy.jsd` is the Jensen-Shannon divergence in bits between the 7-way text and audio posteriors. The flag is set when it exceeds a threshold. The threshold is `jsd_flag` from `heads/manifest.json` when that is a number. It is null until MELD audio exists, so the service uses `--jsd-flag`, whose default is 0.2. The direction is `none` when the flag is off. Otherwise it is `tone_surprised` when the audio gives surprise more than 0.4 and the text less than 0.2. Failing that, it is `tone_more_negative` or `tone_more_positive` when the expected valence of the audio posterior differs from that of the text posterior by more than 0.1, and `mixed` when it does not. Valence is 1 for joy, 0 for neutral and surprise, and -1 for sadness, anger, disgust and fear (`meld_rt/labels.py`).
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
| AudioCalibrator, LR 24-7 or 1048-7 | audio posterior | `heads/audio_cal.npz` | 9 log probabilities, 11 events, 4 prosody scalars, optionally 1024 encoder values | pending | trained |
| Fusion, LR 19-7 | fused posterior | `heads/fusion.npz` | two 7-way log posteriors, their entropies, ASR confidence, laughter, unknown mass | pending | trained |
| SentimentHead, LR 772-3 | sentiment | `heads/sentiment.npz` | the text head's inputs (`text_only` variant), or the 19 fusion inputs (19-3) once audio exists | 0.002 M | trained |
| prosody, discrepancy, selector | rules | | | 0 | none |

The total on the inference path is 359.003 M, 5.98 % of the 6 B cap ([results/params.md](../results/params.md), written by `tools/count_params.py` from the files on disk). It leaves out the two pending heads, which do not exist yet. Piper en_US-lessac-medium (15.651 M) synthesizes the proxy clips offline and is not on the inference path. The latency of each component is in [results/measurements.md](../results/measurements.md).

## 1.6 Trained heads and fusion modes

`tools/train_heads.py` trains the heads from cached features with the settings in `configs/heads.yaml`. Every choice is made on dev by weighted F1, and test is never read.

The text stage is done. The TextContextHead is an MLP with 256 hidden units, trained over a grid of k (0, 1, 2), class weights (MELD README, balanced, none) and learning rate (0.001, 0.0003, 0.0001), with early stopping on dev (patience 6, at most 60 epochs). Dev picked k = 0, no class weights and a learning rate of 0.0001, and a temperature of 1.2228 fitted on dev calibrates the head ([heads/manifest.json](../heads/manifest.json)). The selector thresholds come from the calibrated dev predictions. An isotonic fit of correctness on the confidence gives the dev accuracy as a rising function of the confidence. `low_conf` is the smallest confidence, on a grid of 0.01, at which that accuracy reaches one half (0.39), so below it a label claim is more often wrong than right. `high_conf` is where it reaches 70 % (0.73). With `--text-only`, a logistic regression over the same text features gives the sentiment head (C 0.1, no class weights, temperature 1.4197). The grid took 326.6 s and the whole run 335.9 s ([heads/manifest.json](../heads/manifest.json)). [results/01_text_only.md](../results/01_text_only.md) has the grid.

The audio stage is pending until MELD audio exists. It trains two AudioCalibrators, A on the 24 slot, event and prosody inputs and B with the 1024 encoder values added, and keeps the better one on dev. Both expert heads are refit in 5 folds split by dialogue, so the fusion head trains on out-of-fold posteriors only. The fusion head is then temperature-scaled on dev. The same stage picks the product-of-experts weight from 0.25, 0.5, 0.75 and 1.0, fits a fused sentiment head on the 19 fusion inputs, and sets `jsd_flag` to the 0.8 quantile of the dev JSD between the text posterior and the raw SenseVoice posterior. CSV rows without a cached clip are left out of this stage and counted in the manifest. The service computes the run-time JSD against the same raw SenseVoice posterior, so the threshold and the run-time JSD come from the same audio posterior.

`fused.mode` says how the fused posterior was made, and `fused.calibrated` says whether its confidence is calibrated.

| mode | when | fused posterior | `fused.confidence` | `calibrated` |
|---|---|---|---|---|
| `stacked` | `heads/fusion.npz` exists | the fusion head | its maximum | true |
| `poe` | no fusion head, both expert heads present | product of the text and calibrated audio posteriors | its maximum | false |
| `poe_uncalibrated` | an expert head is missing, which today means the audio calibrator | product of the text posterior and the raw SenseVoice posterior | the text head's probability of the fused label | false |
| `text_only` | a text message, or `--fusion-mode text_only` | the text posterior | its maximum | true when the text head exists |

The product raises the audio posterior to a power w. Before the audio stage has run, w is 0.5. After it, the service reads the dev-chosen w from `audio_fusion.poe.weight` in the manifest (`poe_weight_from` in `meld_rt/service.py`).

In `poe_uncalibrated` mode the maximum of the product is not a probability of being right, because two soft posteriors multiply into a sharp one. So the reported confidence is the calibrated text head's probability of the fused label. `fused.emotion` and `fused.label` stay the product and its argmax. In the demo (1.8) the text head's top class is joy, the product's label is neutral, and the text head gives neutral 0.3467, which is below `low_conf`, so the selector holds back. The selector also caps every state with `calibrated` false at the mid bucket (1.9), so such a state can ask but never asserts.

## 1.7 Conversational context

The TextContextHead sees the current utterance and up to k previous ones, with same-speaker flags, the turn index divided by 20 and a first-turn flag. In MELD the previous utterances are those of the same dialogue, whoever spoke them. Dev picked k = 0, so the deployed head sees the current utterance and the two turn flags only. With a larger k, the service would pass the stream's own previous finals with the same-speaker flags set to 1, because one person talks to the robot. The robot's own lines are never fed back, because MELD has no such lines to train on.

The evaluation has a runtime-regime row for that difference: the same head with the window restricted to the same speaker's previous utterances in the dialogue. No future utterance is ever used. `ctl` `reset` clears the dialogue of a stream.

## 1.8 State schema

This is the final state of the demo in [results/demo_log.jsonl](../results/demo_log.jsonl), without the three fields the monitor adds when it logs (`topic`, `t_recv_ns`, `latency_client_ms`):

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
 "t_speech_end_capture_ns": 5697676059315,
 "t_service_recv_ns": 5697680112696,
 "t_emitted_ns": 5698136805066,
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
   "neutral": 0.935781761916961,
   "surprise": 0.00021973849837607988,
   "fear": 0.0001826222123707147,
   "sadness": 0.006219462140109823,
   "joy": 0.05193871198636549,
   "disgust": 0.0034108129244094796,
   "anger": 0.0022468903214074463
  },
  "label": "neutral",
  "confidence": 0.3467,
  "sentiment": "neutral",
  "mode": "poe_uncalibrated",
  "calibrated": false
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
  "frontend": 14.48,
  "sensevoice": 175.17,
  "prosody": 30.64,
  "text_encoder": 36.65,
  "heads": 0.73,
  "service_total": 456.69,
  "provisional_in_flight": 197.6
 },
 "degraded": []
}
```

- `audio.emotion` has nine keys: the seven SenseVoice emotion classes under their MELD names, in the MELD README order, then `other` and `unknown`. `text.emotion` and `fused.emotion` have the seven MELD keys in the same order. `audio.events` has the eleven SenseVoice events in `EVENT_KEYS` order (`meld_rt/labels.py`).
- `fused.mode`, `fused.confidence` and `fused.calibrated` follow 1.6. `fused.sentiment` is one of MELD's three sentiment classes.
- `timing_ms` is in milliseconds. `vad_tail` is the silence tail inside the segment (512 at the 500 ms setting of the demo). `provisional_in_flight` is how long the final waited for a provisional that was running when the endpoint came. The publisher sets `service_total`.
- `degraded` lists the failures of 1.4.
- A provisional state has `kind` `"provisional"`, `window` `"full"` or `"tail"`, `text.partial` (null on the tail window), a uniform `text.emotion` because no text expert runs, a `fused` block that is the audio posterior over the seven named classes, zero prosody except the rate, and `stability`, which is 1 minus the JSD to the previous provisional. It gets no response.

The dataclasses are in `meld_rt/state.py`.

## 1.9 Replies and grounding

The replies come from `meld_rt/response/bank.yaml`. The bank has 124 lines: 112 in 28 cells of four, 6 `hold_back` lines that make no emotion claim, 2 lines for a failed transcript, and 4 event prefixes. A cell is indexed by the fused label and the discrepancy direction (7 labels × 4 directions), and `mixed` shares the `none` cell. Each cell has `acknowledge` and `ask` lines, and each cell whose direction is not `none` also has `name_the_gap` lines. The loader rejects a line longer than 12 words, a line with a keyphrase slot longer than 8 words, and any exclamation mark. The lines are a draft for the candidate to edit. The first line of the file and its `draft` key say so, and `tests/test_selector.py::test_bank_shape` asserts both, so the marker and those two assertions are removed together.

The selection is a pure function of the state and the stream's recent lines.

1. A failed or low-confidence transcript takes one of the two "did not catch that" lines.
2. A confidence below `low_conf` (0.39 from dev) takes a `hold_back` line.
3. Otherwise the cell gives the candidates. In the mid bucket the order of preference is `ask`, then `name_the_gap` when the direction is not `none`, then `acknowledge`. In the high bucket (a confidence of at least `high_conf`, 0.73, and `fused.calibrated` true) it is `acknowledge` first, or `name_the_gap` first when the direction is not `none`. An uncalibrated state never reaches the high bucket, and its grounding says `rule=uncalibrated_cap` when the cap changed the bucket.
4. A line used in the last 5 turns of the stream is avoided. When a cell has no fresh line left, a fresh `hold_back` line is used and the grounding says `rule=no_repeat`.
5. A line with a `{keyphrase}` slot takes the longest run of up to four content words from the transcript, as an exact substring (`meld_rt/discrepancy.py`).
6. When laughter or cry is above 0.5 in `audio.events`, one of two short clauses goes in front.

Ties break by a CRC32 of the utterance id and the line id, so a replay gives the same lines. The `grounding` list names the state fields that licensed the line, with their values (ex. `fused.label=<label>@<confidence>`, `discrepancy.direction=tone_more_negative`, `text.keyphrase='...'`, or `fused.confidence=0.35<0.39` for the demo's hold-back line).

MELD clips come from a sitcom with a laugh track, so a laughter event there can be the audience rather than the speaker. In front of the robot it would be the user's own laughter. The event features can therefore learn a signal that only MELD has, and the event ablation and the disagreement audit are the checks (1.11).

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

Over all three splits, 83.3 % of MELD utterances last at most 5 s, 95.2 % at most 8 s and 99.7 % at most 15 s (subtitle durations, [results/data_stats.md](../results/data_stats.md)). The VAD tail is not compute. It adds 416 ms at the default setting to what the user waits, and it is reported apart from the targets. The assumptions are one speaker near the microphone, 16 kHz mono audio, utterances up to 15 s, 4 CPU cores and no GPU.

[results/measurements.md](../results/measurements.md) has the container numbers, and every target passes there. On the proxy clips at 400 ms, finals up to 5 s took 173.9 ms at p50 and 261.8 ms at p95 (n 19). On the long clips at 500 ms, finals from 8 to 15 s took 490.8 ms at p50 and 654.0 ms at p95 (n 9). The laptop numbers are pending.

## 1.11 Evaluation

The splits are the published ones. Model selection, temperatures, thresholds and k are chosen on dev only, and each test condition is run once. Weighted F1 is the primary metric because every published MELD row uses it. Macro F1, accuracy, per-class F1, the confusion matrix, a 95 % bootstrap interval over test dialogues (1000 resamples, seed 0) and the ECE (10 bins) sit beside it. `tools/evaluate.py` scores the conditions, and [RESULTS.md](../RESULTS.md) summarizes them.

| conditions | status |
|---|---|
| majority floor, text only at k = 0, at the best k and with the runtime-regime window, text-only sentiment | done |
| audio only from the raw SenseVoice tag, audio calibrators A and B, stacked fusion, product of experts with the dev weight and with the fixed weight, fused sentiment | code ready, needs MELD audio |
| ablations without events and without prosody, the live condition on SenseVoice transcripts with WER, the disagreement subset, early commit on 2 s and 4 s prefixes | needs MELD audio and code in `tools/evaluate.py` that does not exist yet |

The audio and fused conditions leave out test rows that have no cached clip. Each such row reports how many rows were left out and the text head's weighted F1 on the same rows (`text_best_k_same_rows_weighted_f1` in [results/metrics.json](../results/metrics.json)), so the comparison uses the same utterances.

A person also listens to 50 flagged test clips and marks which branch was right and where any laughter came from (`tools/audit_disagreement.py`, [docs/runbook_laptop.md](runbook_laptop.md)). The audit needs MELD audio and has not been done.

The published rows are the MELD paper's Tables 11 and 13, EmoBERTa's Table 2 and Table 8 of the FunAudioLLM report. [PROVENANCE.md](../PROVENANCE.md) quotes what that report says about the data behind SenseVoice's emotion labels. Because MELD may be in that data, the audio-only number is never called zero-shot.

## 1.12 Measurements

- `tools/count_params.py` counts the ONNX initializers and the `.npz` head arrays and writes [results/params.md](../results/params.md) and [results/params.json](../results/params.json).
- `tools/bench_components.py` times each module at 1, 2 and 4 threads on slices and tilings of `en.wav` and writes `results/bench_components_<host>.json`.
- `tools/replay_bench.py` streams clips at real-time pace through the real service, with a monitor in the same process, and writes `results/replay_bench_<host>.json`: latency by length bucket, the reply delay, the provisional cadence, the compute per second of audio, RSS and CPU.
- `tools/render_measurements.py` renders [results/measurements.md](../results/measurements.md) from those files, groups them by the host that ran them, and checks each target.

The container recorded an "Intel(R) Xeon(R) Processor @ 2.10GHz" with nproc 4, 16093 MB of memory, AVX-512 VNNI and AMX, and no GPU ([results/bench_components_vm.json](../results/bench_components_vm.json)). [docs/runbook_laptop.md](runbook_laptop.md) gives the laptop steps. Nothing in the laptop section is typed by hand.

## 1.13 Left out

- Vision and the three-modality extension. There is no camera here, the video was not reachable, and picking the speaker's face in a multi-party sitcom frame is its own project.
- A reply generated by an LLM. It would be nondeterministic and slow on a CPU, and its grounding could not be unit-tested.
- Fine-tuning RoBERTa, SenseVoice, wav2vec2 or HuBERT. It needs GPU hours or long CPU runs, and a reviewer could not rerun it in minutes.
- Context models that read future utterances, because the live system only has the past.
- Arousal and valence fields. MELD has no labels to score them.
- Reinforcement learning. MELD has no reward for a reply, and the one candidate (when to commit the label) was not built.
- A spoken reply. Piper is present but only makes the proxy clips.
- Speaker identification. One user in front of the robot is assumed.

## Repository layout

```
meld_rt/                   the package
  service.py               ingest, worker and publisher threads, CLI
  protocol.py              wire format, topics, clocks
  state.py                 the state dataclasses (1.8)
  labels.py                class orders, SenseVoice tokens, valence
  heads.py                 numpy forward passes of the trained heads
  discrepancy.py           JSD, flag, direction, keyphrase
  data.py                  MELD CSV loading and context rows
  metrics.py               F1, ECE, bootstrap intervals
  experts/                 sensevoice.py, text.py, vad.py, prosody.py
  response/                bank.yaml (124 draft lines), selector.py
  clients/                 replay.py, mic.py, monitor.py
tools/
  fetch_models.py          download and verify the models, write results/provenance_hashes.json
  fetch_meld_csv.py        download the MELD CSVs into meld/
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
  audit_disagreement.py    sample flagged test clips for a listening audit, count the marks
  meld_stats.py            results/data_stats.md
  count_params.py          results/params.md, results/params.json
  bench_components.py      results/bench_components_<host>.json
  replay_bench.py          results/replay_bench_<host>.json
  render_measurements.py   results/measurements.md
configs/heads.yaml         training grid and threshold rules
heads/                     trained heads (.npz) and manifest.json
assets/roberta_tokenizer/  vocab.json, merges.txt
results/                   every measured number, written by the tools above
tests/                     pytest suite
docs/                      design.md, decisions.md, runbook_laptop.md
demo.sh                    one utterance through the system
```

`models/`, `meld/`, `cache/`, `audio/`, `proxy/`, `bench_clips/`, `verify_dl/` and `papers/` are created by the tools or downloaded, and they are not committed.
