# Decisions

Each entry gives the decision, the alternatives that were considered, the reasons, and what would change it. Numbers measured here are cited from the file a script wrote. Published numbers are cited with their source. [docs/design.md](design.md) describes the system itself.

## Track

The primary track is text and audio. Vision is not used.

Alternatives considered: text and vision, or all three modalities as the optional extension.

The MELD authors left visual features out of their own baselines, "as video-based speaker identification and localization is an open problem" (MELD paper, Section 4.1). The raw video was not reachable when the track was chosen. It later came inside the MELD.Raw archive with the audio, and only the audio track was used. The build machine has no camera. A robot's camera would see one person up close, which multi-party sitcom frames do not resemble. Audio carries tone, laughter and timing, which is the part of the challenge's premise that MELD can test. Tested on MELD with gold text, the audio branch adds a small gain in weighted F1 on the whole test set, inside overlapping intervals. It lowers the macro F1, and it adds no gain on the utterances where the words and the tone disagree most (Fusion, below).

What would change it: a single-face camera at the robot. A face-expression expert would then be a third posterior next to text and audio, through the same fusion head.

## Audio expert

SenseVoice-small int8 runs through onnxruntime behind the front end in `meld_rt/experts/sensevoice.py`, and its logits are read directly: the transcript, the 9-way emotion posterior, the 11-way event posterior and, with one added graph output, the encoder states of the same pass.

Alternatives considered: fine-tuning wav2vec2 or HuBERT on MELD audio, Whisper for the transcript plus a separate emotion model, or SenseVoice through the sherpa-onnx wrapper at run time.

One frozen model of 234.000 M parameters ([results/params.md](../results/params.md)) gives everything in one pass, and seven of its emotion classes map one to one onto MELD's seven. The FunAudioLLM report gives it a MELD weighted F1 of 57.7 without fine-tuning on the target domain (Table 8). That is above the best text-only row of the MELD paper, DialogueRNN at 57.03 (Table 11). On published numbers, the audio expert alone is at the level of the text baselines. Measured here, it is not: the raw SenseVoice tag scores 48.52 weighted F1 on MELD test, and on 97.51 % of the test clips more than half of its emotion mass is on OTHER or EMO_UNKNOWN ([RESULTS.md](../RESULTS.md)). The report does not describe its MELD protocol in enough detail to reproduce its number. Fine-tuning an audio encoder on four CPU cores would take most of the timebox, and a reviewer could not rerun it. The sherpa-onnx wrapper does not expose the logit rows, so sherpa-onnx stays a dev dependency for the transcript parity test (`tests/test_sensevoice.py`) and the Piper proxy clips. The report does not list the corpora behind SenseVoice's emotion labels ([PROVENANCE.md](../PROVENANCE.md) quotes it), so the audio-only number is never called zero-shot. The encoder-state readout was tested as calibrator B against calibrator A. Dev chose A (45.96 against 44.43 weighted F1), and B also scores lower on test (46.32 against 48.02), so the deployed calibrator does not read the encoder states ([RESULTS.md](../RESULTS.md)).

What would change it: evidence that MELD audio is in SenseVoice's emotion training data. The audio-only rows would then be marked as possibly seen, and the defensible audio branch would be another model. A GPU budget that makes a wav2vec2 fine-tune reproducible would also reopen the choice.

## Text expert

The text expert is `roberta-base-11.onnx` from the ONNX model zoo, frozen and mean-pooled to 768 values, with the tokenizer taken from the spaCy `en_core_web_trf` wheel.

Alternatives considered: fine-tuning RoBERTa-base or RoBERTa-large on MELD text, the transformer weights inside the spaCy wheel, or static word vectors.

Frozen features train in minutes, keep the run time onnxruntime-only and are reproducible. Their cost is measured, not assumed: the frozen head scores 59.62 weighted F1 on test with gold text and 42.61 on SenseVoice transcripts ([RESULTS.md](../RESULTS.md)), and EmoBERTa's fine-tuned RoBERTa-large without context reports 63.46 (Table 2). The spaCy wheel's transformer was trained further inside its pipeline (its `meta.json` lists OntoNotes 5 as a training source and roberta-base as the starting point), so those weights are not the published encoder and are not used. Static vectors lose word order.

What would change it: a fine-tune that beats the frozen head on dev by more than the bootstrap interval. It was planned as a timed experiment and was not run, and the `finetune` extra in `pyproject.toml` is unused. A fine-tuned encoder exported to ONNX would go through the same `TextExpert` interface with the same parameter count.

## Tokenizer

`vocab.json` and `merges.txt` come from the `byte_bpe_processor` attribute inside the `en_core_web_trf-3.8.0` wheel, extracted with msgpack by `tools/extract_tokenizer.py` and committed under `assets/roberta_tokenizer/`.

Alternatives considered: waiting for the Hugging Face Hub, running the spaCy pipeline at run time, or rebuilding the merges from GPT-2.

The Hub was not reachable from the build environment when the tokenizer was set up. The attribute holds the byte-level BPE vocabulary (50265 pieces) and its 50000 merges, and the ids match the published roberta-base tokenizer: `<s>` 0, `</s>` 2, `Hello` 31414, `Ġworld` 232 (`tools/extract_tokenizer.py`). Extraction needs only msgpack, not spaCy and not torch. `tests/test_tokenizer.py` asserts the known ids and runs a decode and encode round trip on MELD sentences.

What would change it: a byte-for-byte comparison with the published tokenizer files, and a mismatch would replace them. The Hub was reachable later in the build, and the comparison has not been done.

## Fusion

The fusion head is a multinomial logistic regression over both 7-way log posteriors, their entropies, the ASR confidence, the laughter probability and the unknown mass (19 inputs), trained on cross-fitted expert posteriors and temperature-scaled on dev. It is trained on MELD audio and deployed (`fused.mode` `stacked`). A product of experts with one dev-chosen weight is its comparison. When the audio calibrator is missing, the live path falls back to a fixed product of the text posterior and the raw SenseVoice posterior, and the state says so (`fused.mode` `poe_uncalibrated`, `calibrated` false).

Alternatives considered: a concatenation MLP over the 768-value and 1024-value embeddings, cross-attention between audio frames and text tokens, or a single product-of-experts weight.

MELD's train split has 9989 utterances ([results/data_stats.md](../results/data_stats.md)), which does not support a large fusion model trained on a CPU. The stacked model has per-modality weights that can be read, and a calibrated confidence, which the selector needs. The fixed product keeps both modalities on the live path when the audio calibrator is missing.

The fixed product needed its own confidence rule. The maximum of a product of two soft posteriors is sharp, and it is not a probability of being right. Reported as the confidence, it let the selector speak as if it were sure. So in `poe_uncalibrated` mode the reported confidence is the text head's temperature-scaled probability of the fused label, the only calibrated number on that path, and the selector never puts an uncalibrated state above the mid bucket (`reported_confidence` in `meld_rt/service.py`, `tests/test_service_rules.py`, `tests/test_selector.py`).

What would change it: if the product of experts matches the stacked model on dev within the interval, the simpler rule is deployed. If the disagreement subset shows no gain over text alone, the fusion claim comes out of the write-up.

Outcome: at its dev-chosen weight of 0.75 the product of experts scores 55.56 weighted F1 on dev, against 57.95 for the stacked fusion ([RESULTS.md](../RESULTS.md), "Choices made on dev"), so the stacked fusion stays deployed. On the 489 test utterances with a JSD above the dev threshold, the stacked fusion scores 49.62 against 51.44 for the text head alone, so the second rule applies. The write-up makes no claim that the audio branch helps where the words and the tone disagree. On the whole test set with gold text, the fusion scores 60.77 against 59.62 for the text head. The gain is small, and the two 95 % intervals overlap. Its macro F1 is lower (39.24 against 41.22), and it scores 0.00 on fear, against 20.29 for the text head. On SenseVoice transcripts, which is what the running system sees, the fusion scores 47.05 and the text head 42.61 ([RESULTS.md](../RESULTS.md), "Gold text and live ASR transcripts"). The fusion stays deployed because it scores higher than the text head on dev (57.95 against 57.56).

## Context

The text head sees the current utterance and up to two previous utterance embeddings with same-speaker flags, and k (0, 1 or 2) is chosen on dev. Dev chose k = 0.

Alternatives considered: a GRU over the dialogue, a dialogue transformer, or no context at all.

EmoBERTa puts the gain from past context at about one point (64.55 with past utterances against 63.46 without, Table 2). A window head can capture that, needs no torch at run time and is causal. The robot hears one speaker while MELD is multi-party, so the same head is also scored with the window restricted to the same speaker's earlier utterances. On dev, the best k = 1 and k = 2 points reached 56.14 and 55.75 weighted F1 against 57.56 at k = 0 ([results/01_text_only.md](../results/01_text_only.md)). The deployed head is therefore the k = 0 head of 0.200 M parameters, and context leaves the run-time path except for the turn-index flags.

What would change it: a fine-tuned encoder whose dev score rises with context, or a robot that talks to several users, which would also need speaker continuity.

## Replies

The reply is one of the 124 lines in `meld_rt/response/bank.yaml`, chosen by fused label and discrepancy direction, with a confidence rule that forces a line without an emotion claim when the posterior is weak, a keyphrase slot filled from the transcript, and a `grounding` list that names the state fields that licensed the line. The high bucket, which asserts a feeling or names the gap, needs a calibrated confidence. The loader enforces the word limits and rejects exclamation marks.

Alternatives considered: a small instruct LLM, GPT-2 small from the ONNX zoo fine-tuned on MELD next lines, or retrieval of MELD next utterances.

An LLM would add latency on a CPU, would be nondeterministic, could not have its grounding unit-tested, and had its weights on a host the build environment could not reach at the time. Retrieval would return copyrighted sitcom lines that are not grounded in the audio. The selector takes 0.009 ms at p50 at 4 threads ([results/bench_components_vm.json](../results/bench_components_vm.json)), and the reply followed the final state within 0.420 ms at p95 on the proxy clips and 0.350 ms on the MELD test clips ([results/measurements.md](../results/measurements.md)). Its behavior is tested: every cell has at least three variants, changing only the audio posterior changes the line on a fixed set of cases, a low confidence always gives `hold_back`, no line repeats within five turns, and an uncalibrated state never reaches the high bucket (`tests/test_selector.py`).

What would change it: a need for open-ended dialogue rather than a reaction, or a GPU. An LLM would then take the state as its prompt, and the grounding list would become a check on its output.

## Streaming

Audio goes in over PUSH/PULL and the state comes out over PUB/SUB, as raw PCM16 frames with a small JSON header. A media clock sits in every state field, and the two wall clocks are never compared. Provisionals are bounded to a 4 s window and run only when the worker is idle. Audio frames are never dropped, while provisionals are.

Alternatives considered: FastAPI with WebSocket JSON, gRPC, or a single process with an in-memory queue.

An HTTP layer would put base64 or text frames and transport time inside the latency that is reported. gRPC means code generation for no gain at this size. A single process could not show one input travelling through the system to two outputs over a real socket. PUB/SUB lets a monitor, a logger and a robot body subscribe at once without the service knowing about them. The bounded provisional policy limits how long a final can wait behind a provisional to one SenseVoice pass over at most 4 s of audio.

What would change it: a remote client. ZeroMQ would stay inside, and a WebSocket bridge at the edge would be measured as a separate cost.

## Real time

Real time is defined per turn: three latency targets by utterance length (design 1.10), the reply within 5 ms, provisionals at 1 Hz during the first 4 s of speech, and sustained compute below 0.3 s per second of audio.

Alternatives considered: one latency number for all lengths, frame-level latency, or a real-time factor alone.

SenseVoice's cost grows with the clip. At 4 threads its call takes 83.3 ms at p50 on 2 s of audio and 420.9 ms on 15 s ([results/measurements.md](../results/measurements.md)), so a single target would be false for the longest utterances or loose for the short ones. The buckets follow the MELD duration distribution ([results/data_stats.md](../results/data_stats.md)). The VAD tail is a setting, not compute, so it is reported apart.

What would change it: a laptop bench that misses a bucket at 2 threads, which would lower the 15 s cap or shrink the provisional window.

## End-pointing

Silero VAD v4 opens a segment after 8 frames (256 ms) above 0.5 and keeps 256 ms of pre-roll. It closes the segment after a run of frames below 0.35 set by `--endpoint-silence-ms`, 500 ms by default, which is 16 frames or 512 ms. It forces an endpoint at 15 s.

Alternatives considered: the endpoint of a streaming ASR model, energy-based end-pointing, or fixed-length chunks.

A streaming ASR model would be a second model doing a job the VAD already does. Energy fails with background noise. Fixed chunks cut words. The start rule takes the larger of three frames and `min_speech_ms` (250), so a click or a short noise does not open a segment. The right tail sets how long the robot waits after the speaker stops before it answers, so it is a flag. 600 ms (a 608 ms tail) splits fewer MELD dev clips than 500 ms, and every turn then waits the longer tail. Every bench ran at 500 ms. 600 ms was not benched.

The default is 500 ms rather than 400 ms. `tools/pause_stats.py` runs the service's VAD over a set of clips and records every pause inside speech (a run of frames below 0.35 that more speech ends), then counts the clips each setting splits. On the 1019 MELD dev clips in which the VAD found speech, 400 ms splits 135 clips by a pause and 500 ms splits 123 ([results/pause_stats_meld_dev.json](../results/pause_stats_meld_dev.json)). Before MELD audio was available, the choice rested on `en.wav`, the only clip measured at the time that 400 ms split ([results/pause_stats_available.json](../results/pause_stats_available.json)). A split makes the robot answer the first half of a sentence and then the second half as a new turn, and the text head reads each half alone. The cost of 500 ms is the longer tail on every turn. For utterances up to 5 s, the time from the end of speech to the final state comes to 658.47 ms at p50 and 770.03 ms at p95 at 500 ms, against 562.47 ms and 674.03 ms at 400 ms ([results/pause_stats_meld_dev.json](../results/pause_stats_meld_dev.json)). Each figure is the tail plus the service-side p50 / p95 of [results/replay_bench_vm.json](../results/replay_bench_vm.json), which ran at 500 ms, so the 400 ms figures assume the same service-side time. A reply to half a sentence is a worse failure than a slower reply, so the default was the shortest candidate that kept every clip measured at the time whole. The committed demo and the end-to-end test ran at 500 ms, so neither needs an override.

What would change it: the same script on MELD dev audio, `uv run python tools/pause_stats.py --wavs "audio/dev/*.wav" --label meld_dev --bench results/replay_bench_vm.json`. If 400 ms splits the same number of dev clips by a pause as 500 ms (`clips_split_by_pause` in `results/pause_stats_meld_dev.json`), the default goes back to 400 ms, because the longer tail then keeps no extra clip whole. The same holds if every extra split at 400 ms falls at a change of speaker or at the laugh track. If 500 ms splits dev clips that 600 ms keeps whole, the next step is to listen to those clips. The default moves to 600 ms only if the second segment is the same speaker after a pause. A MELD clip is cut from a sitcom, so its second segment can also be another speaker or the laugh track.

Outcome: 400 ms splits more dev clips than 500 ms (above), so the default stays at 500 ms. 600 ms splits 92 of the same clips (`clips_split_by_pause` in [results/pause_stats_meld_dev.json](../results/pause_stats_meld_dev.json)). Nobody has listened to the extra splits at 400 ms, so the speaker-change condition was not tested. 500 ms splits dev clips that 600 ms keeps whole, so the next step is the listening check above. It has not been done, and the default stays at 500 ms until it is.

## Discrepancy threshold

The flag threshold is 0.3831, the 0.8 quantile of the dev JSD between the text posterior and the raw SenseVoice posterior. `tools/train_heads.py` sets it, and the service reads it from `heads/manifest.json`. The service computes the run-time JSD against the same raw SenseVoice posterior ([docs/design.md](design.md), section 1.6). A manifest without a dev value leaves the service at its default of 0.2 (`--jsd-flag`).

Alternatives considered: a threshold tuned on the proxy clips, or no flag until audio exists.

The proxy clips are affectless TTS, so their JSD says nothing about real speech. The flag picks the reply cell, so the live path needs some threshold. A quantile of the dev JSD ties it to real MELD speech. On test, 18.74 % of the utterances are above it when the text posterior comes from gold text ([RESULTS.md](../RESULTS.md)).

What would change it: the disagreement audit ([docs/runbook_laptop.md](runbook_laptop.md)), which needs a person to listen and has not been done.

## Evaluation

Weighted F1 is primary, with macro F1, per-class F1, bootstrap intervals over dialogues and ECE beside it. Gold-text and ASR-text conditions are separate rows, with an early-commit row, a disagreement subset and ablations of the features most likely to be dataset artifacts. Selection is on dev only, and test runs once per condition. Class weights are chosen on dev by weighted F1, and the macro-F1 cost is recorded.

Alternatives considered: accuracy alone, macro F1 as primary, or selection on test.

Weighted F1 is what every published MELD row uses (MELD paper, EmoBERTa), so it is the only number that can be compared. The selector consumes the confidence, so calibration has to be measured. The laugh track and the subtitle cuts in MELD can leak into the event and prosody features, so those ablations are the artifact check. On dev, choosing by weighted F1 picked no class weights and cost 0.90 macro-F1 points against the best macro-F1 point ([results/01_text_only.md](../results/01_text_only.md)).

What would change it: nothing in the protocol. Every condition is scored except the text-only sentiment row, which needs a text-only sentiment head ([RESULTS.md](../RESULTS.md), "Pending rows").

Outcome of the artifact check: without the event and prosody inputs the stacked fusion scores 60.82 against 60.77 for the deployed one, and the two single ablations also stay inside the deployed fusion's interval ([RESULTS.md](../RESULTS.md), "Ablations"). The event and prosody inputs add nothing measurable on MELD test, so the fused score does not rest on a laugh-track signal in them. Without them the calibrator reads 9 inputs instead of 24 and the fusion 18 instead of 19, at no measured cost. The deployed heads were not changed and still read those inputs.

## Runtime

Every model runs through onnxruntime, and the heads are numpy forward passes from `.npz` files.

Alternatives considered: torch at run time, or sherpa-onnx as the run-time wrapper.

One dependency behaves the same in the container and on the laptop, int8 kernels are available, and inference needs no large install. sherpa-onnx stays a dev dependency for the transcript parity test and the Piper proxy clips.

What would change it: a failure of the parity test that the front end could not fix. The transcript would then come from sherpa-onnx, and the encoder-state readout would be lost.

## Reinforcement learning

There is no RL. The one candidate, when to commit the label while the person is still talking, was not built.

Alternatives considered: a reply-style bandit, or RL in the core from the start.

The only reward MELD offers is label correctness, and the only sequential decision in this system is when to commit. That is an optimal-stopping problem: EMIT or WAIT at each provisional tick, a reward of one for a correct label minus a cost per elapsed second, trained offline by fitted Q-iteration on logged prefixes, and compared with three baselines (emit at the endpoint, at a fixed time, or when the maximum probability passes a threshold). The 2 s and 4 s prefix features of MELD audio are now cached, and the early-commit rows score them ([RESULTS.md](../RESULTS.md), "Early commit"), but the stopping policy was not built. A reply-style bandit has no reward at all.

What would change it: time to train the stopping policy on the cached prefixes and compare it with the three baselines.

## Compared with the usual stack

The usual design is a fine-tuned RoBERTa and a fine-tuned wav2vec2 or HuBERT with late fusion, a small LLM for the reply, and FastAPI. Published fine-tuned text rows reach 63.46 to 65.61 weighted F1 (EmoBERTa, Table 2). This system reads emotion, events and a transcript from one frozen ASR model in one pass, stacks two small expert heads, keeps the relation between words and tone in the state as a field, selects a reply that can be audited, and streams over ZeroMQ with a media clock. Each part is cheaper to run, deterministic, or easier to test than its usual counterpart. The frozen text head scores 59.62 ([RESULTS.md](../RESULTS.md)), so the price of not fine-tuning is measured against those rows. On the audio side, the stacked fusion scores 60.77, and DialogueRNN on text and audio (60.25, MELD paper, Table 11) is inside its interval. On SenseVoice transcripts instead of gold text, the fusion scores 47.05 ([RESULTS.md](../RESULTS.md)).
