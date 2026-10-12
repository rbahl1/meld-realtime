# Decisions

Each entry gives the decision, the alternatives that were considered, the reasons, and what would change it. Numbers measured here are cited from the file a script wrote. Published numbers are cited with their source. [docs/design.md](design.md) describes the system itself.

## Track

The primary track is text and audio. Vision is not used.

Alternatives considered: text and vision, or all three modalities as the optional extension.

The MELD authors left visual features out of their own baselines, "as video-based speaker identification and localization is an open problem" (MELD paper, Section 4.1). The raw video was not reachable from the build environment, and the build machine has no camera. A robot's camera would see one person up close, which multi-party sitcom frames do not resemble. Audio carries tone, laughter and timing, which is the part of the challenge's premise that MELD can test.

What would change it: a reachable video source and a single-face camera at the robot. A face-expression expert would then be a third posterior next to text and audio, through the same fusion head.

## Audio expert

SenseVoice-small int8 runs through onnxruntime behind the front end in `meld_rt/experts/sensevoice.py`, and its logits are read directly: the transcript, the 9-way emotion posterior, the 11-way event posterior and, with one added graph output, the encoder states of the same pass.

Alternatives considered: fine-tuning wav2vec2 or HuBERT on MELD audio, Whisper for the transcript plus a separate emotion model, or SenseVoice through the sherpa-onnx wrapper at run time.

One frozen model of 234.000 M parameters ([results/params.md](../results/params.md)) gives everything in one pass, and seven of its emotion classes map one to one onto MELD's seven. The FunAudioLLM report gives it a MELD weighted F1 of 57.7 without fine-tuning on the target domain (Table 8). That is above the best text-only row of the MELD paper, DialogueRNN at 57.03 (Table 11). On published numbers, the audio expert alone is at the level of the text baselines. Fine-tuning an audio encoder on four CPU cores would take most of the timebox, and a reviewer could not rerun it. The sherpa-onnx wrapper does not expose the logit rows, so sherpa-onnx stays a dev dependency for the transcript parity test (`tests/test_sensevoice.py`) and the Piper proxy clips. The report does not list the corpora behind SenseVoice's emotion labels ([PROVENANCE.md](../PROVENANCE.md) quotes it), so the audio-only number is never called zero-shot. The encoder-state readout is an ablation: calibrator B against calibrator A on dev, pending MELD audio.

What would change it: evidence that MELD audio is in SenseVoice's emotion training data. The audio-only rows would then be marked as possibly seen, and the defensible audio branch would be another model. A GPU budget that makes a wav2vec2 fine-tune reproducible would also reopen the choice.

## Text expert

The text expert is `roberta-base-11.onnx` from the ONNX model zoo, frozen and mean-pooled to 768 values, with the tokenizer taken from the spaCy `en_core_web_trf` wheel.

Alternatives considered: fine-tuning RoBERTa-base or RoBERTa-large on MELD text, the transformer weights inside the spaCy wheel, or static word vectors.

Frozen features train in minutes, keep the run time onnxruntime-only and are reproducible. Their cost is measured, not assumed: the frozen head scores 59.62 weighted F1 on test ([RESULTS.md](../RESULTS.md)), and EmoBERTa's fine-tuned RoBERTa-large without context reports 63.46 (Table 2). The spaCy wheel's transformer was trained further inside its pipeline (its `meta.json` lists OntoNotes 5 as a training source and roberta-base as the starting point), so those weights are not the published encoder and are not used. Static vectors lose word order.

What would change it: a fine-tune that beats the frozen head on dev by more than the bootstrap interval. It was planned as a timed experiment and was not run, and the `finetune` extra in `pyproject.toml` is unused. A fine-tuned encoder exported to ONNX would go through the same `TextExpert` interface with the same parameter count.

## Tokenizer

`vocab.json` and `merges.txt` come from the `byte_bpe_processor` attribute inside the `en_core_web_trf-3.8.0` wheel, extracted with msgpack by `tools/extract_tokenizer.py` and committed under `assets/roberta_tokenizer/`.

Alternatives considered: waiting for the Hugging Face Hub, running the spaCy pipeline at run time, or rebuilding the merges from GPT-2.

The Hub was not reachable from the build environment. The attribute holds the byte-level BPE vocabulary (50265 pieces) and its 50000 merges, and the ids match the published roberta-base tokenizer: `<s>` 0, `</s>` 2, `Hello` 31414, `Ġworld` 232 (`tools/extract_tokenizer.py`). Extraction needs only msgpack, not spaCy and not torch. `tests/test_tokenizer.py` asserts the known ids and runs a decode and encode round trip on MELD sentences.

What would change it: access to the Hub. The files would then be compared byte for byte with the published tokenizer, and a mismatch would replace them.

## Fusion

The fusion head is a multinomial logistic regression over both 7-way log posteriors, their entropies, the ASR confidence, the laughter probability and the unknown mass (19 inputs), trained on cross-fitted expert posteriors and temperature-scaled on dev. It waits for MELD audio. A product of experts with one dev-chosen weight is its comparison. Until MELD audio exists, the live path uses a fixed product of the text posterior and the raw SenseVoice posterior, and the state says so (`fused.mode` `poe_uncalibrated`, `calibrated` false).

Alternatives considered: a concatenation MLP over the 768-value and 1024-value embeddings, cross-attention between audio frames and text tokens, or a single product-of-experts weight.

MELD's train split has 9989 utterances ([results/data_stats.md](../results/data_stats.md)), which does not support a large fusion model trained on a CPU. The stacked model has per-modality weights that can be read, and a calibrated confidence, which the selector needs. The fixed product keeps both modalities on the live path before MELD audio exists.

The fixed product needed its own confidence rule. The maximum of a product of two soft posteriors is sharp, and it is not a probability of being right. Reported as the confidence, it let the selector speak as if it were sure. So in `poe_uncalibrated` mode the reported confidence is the text head's temperature-scaled probability of the fused label, the only calibrated number on that path, and the selector never puts an uncalibrated state above the mid bucket (`reported_confidence` in `meld_rt/service.py`, `tests/test_service_rules.py`, `tests/test_selector.py`). In the demo the product's label is neutral, the text head gives neutral 0.3467, below `low_conf`, and the selector holds back ([results/demo_log.jsonl](../results/demo_log.jsonl)).

What would change it: if the product of experts matches the stacked model on dev within the interval, the simpler rule is deployed. If the disagreement subset shows no gain over text alone, the fusion claim comes out of the write-up.

## Context

The text head sees the current utterance and up to two previous utterance embeddings with same-speaker flags, and k (0, 1 or 2) is chosen on dev. Dev chose k = 0.

Alternatives considered: a GRU over the dialogue, a dialogue transformer, or no context at all.

EmoBERTa puts the gain from past context at about one point (64.55 with past utterances against 63.46 without, Table 2). A window head can capture that, needs no torch at run time and is causal. The robot hears one speaker while MELD is multi-party, so the same head is also scored with the window restricted to the same speaker's earlier utterances. On dev, the best k = 1 and k = 2 points reached 56.14 and 55.75 weighted F1 against 57.56 at k = 0 ([results/01_text_only.md](../results/01_text_only.md)). The deployed head is therefore the k = 0 head of 0.200 M parameters, and context leaves the run-time path except for the turn-index flags.

What would change it: a fine-tuned encoder whose dev score rises with context, or a robot that talks to several users, which would also need speaker continuity.

## Replies

The reply is one of the 124 lines in `meld_rt/response/bank.yaml`, chosen by fused label and discrepancy direction, with a confidence rule that forces a line without an emotion claim when the posterior is weak, a keyphrase slot filled from the transcript, and a `grounding` list that names the state fields that licensed the line. The high bucket, which asserts a feeling or names the gap, needs a calibrated confidence. The lines are a draft for the candidate to edit.

Alternatives considered: a small instruct LLM, GPT-2 small from the ONNX zoo fine-tuned on MELD next lines, or retrieval of MELD next utterances.

An LLM would add latency on a CPU, would be nondeterministic, could not have its grounding unit-tested, and had its weights on a host the build environment could not reach. Retrieval would return copyrighted sitcom lines that are not grounded in the audio. The selector takes 0.009 ms at p50 at 4 threads ([results/bench_components_vm.json](../results/bench_components_vm.json)), and the reply followed the final state within 0.330 ms at p95 on the proxy clips ([results/replay_bench_vm.json](../results/replay_bench_vm.json)). Its behavior is tested: every cell has at least three variants, changing only the audio posterior changes the line on a fixed set of cases, a low confidence always gives `hold_back`, no line repeats within five turns, and an uncalibrated state never reaches the high bucket (`tests/test_selector.py`).

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

Silero VAD v4 opens a segment after 8 frames (256 ms) above 0.5 and keeps 256 ms of pre-roll. It closes the segment after a run of frames below 0.35 set by `--endpoint-silence-ms`, 400 ms by default, which is 13 frames or 416 ms. It forces an endpoint at 15 s.

Alternatives considered: the endpoint of a streaming ASR model, energy-based end-pointing, or fixed-length chunks.

A streaming ASR model would be a second model doing a job the VAD already does. Energy fails with background noise. Fixed chunks cut words. The start rule takes the larger of three frames and `min_speech_ms` (250), so a click or a short noise does not open a segment. The right tail depends on how patient the robot should feel, so it is a flag. `en.wav` from the SenseVoice release has a 0.45 s pause ([tests/test_service_e2e.py](../tests/test_service_e2e.py)), which the 416 ms tail splits into two utterances, so the demo and the end-to-end test use 500 ms (512 ms of tail). 600 ms is the patient setting. The benches ran at 400 ms on the proxy clips and at 500 ms on the long clips. 600 ms was not benched.

What would change it: the user's preference after the live demo, or a bench showing that 400 ms splits too many MELD utterances.

## Discrepancy threshold

Until MELD audio exists, the flag threshold is the service default of 0.2 (`--jsd-flag`). Once audio is cached, `tools/train_heads.py` sets it to the 0.8 quantile of the dev JSD, and the service reads it from `heads/manifest.json`. That quantile is taken against the raw SenseVoice posterior, while the service compares against the calibrated one once it exists ([docs/design.md](design.md), section 1.6), so one of the two has to change before the dev value is used.

Alternatives considered: a threshold tuned on the proxy clips, or no flag until audio exists.

The proxy clips are affectless TTS, so their JSD says nothing about real speech. The flag picks the reply cell, so the live path needs some threshold. A fixed default that is named as a default ([results/01_text_only.md](../results/01_text_only.md), [RESULTS.md](../RESULTS.md)) is the plain choice.

What would change it: the dev quantile, as soon as MELD audio is cached, and the disagreement audit ([docs/runbook_laptop.md](runbook_laptop.md)).

## Evaluation

Weighted F1 is primary, with macro F1, per-class F1, bootstrap intervals over dialogues and ECE beside it. Gold-text and ASR-text conditions are separate rows, with an early-commit row, a disagreement subset and ablations of the features most likely to be dataset artifacts. Selection is on dev only, and test runs once per condition. Class weights are chosen on dev by weighted F1, and the macro-F1 cost is recorded.

Alternatives considered: accuracy alone, macro F1 as primary, or selection on test.

Weighted F1 is what every published MELD row uses (MELD paper, EmoBERTa), so it is the only number that can be compared. The selector consumes the confidence, so calibration has to be measured. The laugh track and the subtitle cuts in MELD can leak into the event and prosody features, so those ablations are the artifact check. On dev, choosing by weighted F1 picked no class weights and cost 0.90 macro-F1 points against the best macro-F1 point ([results/01_text_only.md](../results/01_text_only.md)).

What would change it: nothing in the protocol. Four of its conditions still need code in `tools/evaluate.py` ([RESULTS.md](../RESULTS.md), "Pending rows").

## Runtime

Every model runs through onnxruntime, and the heads are numpy forward passes from `.npz` files.

Alternatives considered: torch at run time, or sherpa-onnx as the run-time wrapper.

One dependency behaves the same in the container and on the laptop, int8 kernels are available, and inference needs no large install. sherpa-onnx stays a dev dependency for the transcript parity test and the Piper proxy clips.

What would change it: a failure of the parity test that the front end could not fix. The transcript would then come from sherpa-onnx, and the encoder-state readout would be lost.

## Reinforcement learning

There is no RL. The one candidate, when to commit the label while the person is still talking, was not built.

Alternatives considered: a reply-style bandit, or RL in the core from the start.

The only reward MELD offers is label correctness, and the only sequential decision in this system is when to commit. That is an optimal-stopping problem: EMIT or WAIT at each provisional tick, a reward of one for a correct label minus a cost per elapsed second, trained offline by fitted Q-iteration on logged prefixes, and compared with three baselines (emit at the endpoint, at a fixed time, or when the maximum probability passes a threshold). It needs the 2 s and 4 s prefix features of MELD audio, which are not cached yet. A reply-style bandit has no reward at all.

What would change it: cached MELD audio with its prefixes, and time left after the pending audio rows.

## Compared with the usual stack

The usual design is a fine-tuned RoBERTa and a fine-tuned wav2vec2 or HuBERT with late fusion, a small LLM for the reply, and FastAPI. Published fine-tuned text rows reach 63.46 to 65.61 weighted F1 (EmoBERTa, Table 2). This system reads emotion, events and a transcript from one frozen ASR model in one pass, stacks two small expert heads, keeps the relation between words and tone in the state as a field, selects a reply that can be audited, and streams over ZeroMQ with a media clock. Each part is cheaper to run, deterministic, or easier to test than its usual counterpart. The frozen text head scores 59.62 ([RESULTS.md](../RESULTS.md)), so the price of not fine-tuning is measured against those rows. The audio side of the comparison waits for MELD audio.
