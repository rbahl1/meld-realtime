# Decisions

One entry per decision: what I decided, what else I considered, why, and what would make me change it. Published comparison numbers are cited with their source. Anything measured in this repository is cited from results/ rather than typed here. The system itself is described in docs/design.md.

## Track: text and audio, not text and vision

**Decision.** The primary track is Text + Audio. Vision is not used.

**Alternatives.** Text + Vision; all three modalities as the optional extension.

**Why.** The MELD authors left visual features out of their own baselines because picking the speaker's face in a multi-party frame is an open problem (MELD paper, Section 4.1). The raw video was not reachable from my build environment and the machine has no camera. The lamp's camera would see one person up close, which sitcom frames do not resemble. Audio carries tone, laughter and timing, which is the part of the challenge's premise I can test on MELD.

**What would change it.** A reachable video source plus a single-face camera feed at the lamp. A face-expression expert would then slot in as a third posterior next to text and audio, through the same fusion head.

## Audio features: read SenseVoice directly

**Decision.** I run SenseVoice-small int8 through onnxruntime with my own front end and read its logits directly: the transcript, the nine-way emotion posterior, the eleven-way audio-event posterior and, with a one-tensor edit of the ONNX graph, a 512-d encoder sequence from the same pass.

**Alternatives.** Fine-tune wav2vec2 or HuBERT (published sizes roughly 95 M to 316 M) on MELD audio. Run Whisper for the transcript plus a separate emotion model. Call SenseVoice through the sherpa-onnx wrapper at runtime.

**Why.** One frozen 234 M model (recomputed by `tools/count_params.py`) gives everything in one pass, and its emotion classes map one-to-one onto MELD's seven. Its published MELD weighted F1 without target-domain fine-tuning is 57.7 (FunAudioLLM report, arXiv 2407.04051, Table 8), on par with the MELD paper's text baselines (text-CNN 55.02, bcLSTM text 56.44; MELD paper, Table 11), so the audio branch is a real expert rather than a weak side channel. Fine-tuning an audio encoder on four CPU cores for the two to three points that the MELD paper reports from adding audio (Table 11) would take most of the timebox and could not be rerun by a reviewer. The sherpa-onnx wrapper hides the logit rows I need, so I keep it as a dev dependency for the transcript parity test only. The report does not list the SER training corpora, so overlap with MELD cannot be excluded; I state that in PROVENANCE.md and never call the audio number zero-shot. The encoder-state readout is a measured ablation: if it does not beat the slot posterior on dev, the deployed calibrator keeps its 24 inputs.

**What would change it.** Evidence that MELD was in SenseVoice's SER training data. The audio-only rows would then be marked contaminated and the only defensible audio branch would be a different model. A GPU and time budget that make a wav2vec2 fine-tune reproducible would also reopen it.

## Text encoder: frozen RoBERTa-base from the ONNX zoo

**Decision.** The text expert is `roberta-base-11.onnx` from the ONNX model zoo, frozen, mean-pooled to 768-d, with the tokenizer extracted from the spaCy `en_core_web_trf` wheel.

**Alternatives.** Fine-tune RoBERTa-base or RoBERTa-large on MELD text. Use the transformer weights inside the spaCy wheel. Static word vectors.

**Why.** Frozen features train in minutes, keep the runtime onnxruntime-only and are reproducible by anyone. The cost is a few points against fine-tuned numbers, which I state and bound with the EmoBERTa rows (RoBERTa-large, no context: 63.46 weighted F1; arXiv 2108.12009, Table 2). The spaCy wheel's transformer weights drifted during OntoNotes training (its `meta.json` lists the tagger, parser and NER pipeline), so they are not the published encoder and I do not use them. Static vectors lose too much.

**What would change it.** The fine-tune is a planned experiment with a fixed budget: one timed epoch first, then the full run only if it fits. It exports back to ONNX and is served through the same `TextExpert` interface, so nothing in the runtime changes and the parameter count stays the same. If it beats the frozen head on dev by more than the bootstrap CI, it becomes the deployed text expert. If the Hugging Face Hub stays blocked, the zoo graph's HF-style initializer names let me build the torch model locally.

## Fusion: a stacked logistic regression over calibrated experts

**Decision.** A 140-parameter multinomial logistic regression (recomputed by `tools/count_params.py`) over both log-posteriors, their entropies, ASR confidence, laughter probability and the unknown mass, trained on cross-fitted expert predictions and temperature-scaled on dev. Product of experts with one dev-chosen weight is the ablation.

**Alternatives.** A concatenation MLP over the raw 768-d and 1024-d embeddings. Cross-attention over audio frames and text tokens. A single product-of-experts weight.

**Why.** MELD's training set (size in results/data_stats.md) does not support a large fusion model trained on CPU. The stacked model gives inspectable per-modality weights and a calibrated confidence, which the reply selector needs for its hold-back rule. Until MELD audio exists the live path uses a fixed product of experts and says so in the state (`fused.mode` `"poe_uncalibrated"`), so the demo never claims a calibration it does not have.

**What would change it.** If product of experts matches the stacked model on dev within the CI, the simpler rule is deployed. If the disagreement-subset row shows no gain over text alone, the fusion claim comes out of the write-up.

## Context: a causal window, not a recurrent model

**Decision.** The text head sees the current utterance and up to two previous utterance embeddings with same-speaker flags; k in {0, 1, 2} is chosen on dev.

**Alternatives.** A GRU over the dialogue. A dialogue transformer. No context at all.

**Why.** EmoBERTa bounds the gain from past context at about one point (64.55 with past utterances versus 63.46 without; Table 2). A window head captures most of that, needs no torch at runtime and is strictly causal. The lamp sees one speaker and MELD is multi-party, so I also evaluate the same head with the window restricted to the same speaker's previous utterances and report both rows. The robot's own lines are never fed back, because MELD has no such lines to train on.

**What would change it.** If k = 0 wins on dev, the head shrinks to 0.20 M parameters and context leaves the runtime path. A lamp with several users would need speaker continuity, which I left out.

## Replies: a bank indexed by the state, not a generator

**Decision.** The reply is one of about 100 lines I wrote, indexed by fused label and discrepancy direction, with a confidence rule that forces a no-claim line when the posterior is weak, a keyphrase slot filled from the transcript, and a `grounding` list naming the state fields that licensed the line.

**Alternatives.** A small instruct LLM of 0.5 to 1.5 B parameters. GPT-2 small from the ONNX zoo fine-tuned on MELD next lines. Retrieval of MELD next utterances.

**Why.** An LLM would add a large fraction of a second per turn on CPU (my estimate, not measured), is nondeterministic, cannot have its grounding unit-tested, has its weights on a host I could not reach, and is the default answer. Retrieval returns copyrighted sitcom dialogue that is not grounded in the audio. The bank answers in under a millisecond, deterministically, with tests that can fail: every cell has at least three variants, changing only the audio posterior changes the line on a fixed set of cases, low confidence always yields `hold_back`, and no line repeats within five turns. The `grounding` list makes every reply auditable in the demo log.

**What would change it.** A requirement for open-ended dialogue rather than a reaction, or a GPU. An LLM would then take the state as its prompt and the grounding list would become a post-check on its output.

## Streaming: ZeroMQ with a media clock

**Decision.** PUSH/PULL for audio in, PUB/SUB for state out, raw PCM16 frames with a small JSON header, a media clock in every state field, and two wall-clock domains that are never compared. Provisionals are bounded to a 4 s window and run only when the worker is idle; audio frames are never dropped, provisionals are.

**Alternatives.** FastAPI with WebSocket JSON. gRPC. A single process with an in-memory queue.

**Why.** An HTTP layer buys nothing here and would put base64 or text frames and transport time inside the latency I report. gRPC means codegen for no gain at this size. A single process could not show one input travelling through the system to two outputs over a real socket. PUB/SUB lets a monitor, a logger and a robot body subscribe at once without the service knowing. The bounded provisional policy means a final never waits behind a long provisional.

**What would change it.** A remote client. I would keep ZeroMQ inside and add a WebSocket bridge at the edge, reported as a separate cost. A need for request-reply semantics would argue for a REQ/REP control channel beside the stream.

## Real time: turn-level targets by utterance length

**Decision.** Real time is defined at the turn level with three length buckets: final state within 250 ms median and 400 ms p95 after the endpoint for utterances up to 5 s, 350 and 550 ms up to 8 s, 700 and 1000 ms up to the 15 s cap, plus the reply within 5 ms, provisionals at 1 Hz for the first 4 s and sustained compute below 0.3 s per second of audio. Whether the targets are met is in results/measurements.md.

**Alternatives.** One number for all lengths. Frame-level latency. A real-time factor alone.

**Why.** SenseVoice's cost grows with the clip (see results/measurements.md), so a single target would be false for the longest utterances. The buckets follow the MELD duration distribution (results/data_stats.md). The 400 ms VAD tail is a choice, not compute, so it is reported separately. The laptop column stays empty until the user runs the bench there; nothing is typed from memory.

**What would change it.** A bench showing a bucket is missed at 2 threads would lower the 15 s cap or shrink the tail window. A product decision on how patient the lamp should feel would change the tail flag.

## Evaluation: weighted F1, plus what the selector needs

**Decision.** Weighted F1 is primary; macro F1, per-class F1, bootstrap CIs over dialogues and ECE are reported beside it. Gold-text and ASR-text conditions are separate rows, with an early-commit row, a disagreement subset and ablations that remove the features most likely to be dataset artifacts. Selection is on dev only; test runs once per condition.

**Alternatives.** Accuracy alone. Macro F1 as primary. Selecting on test.

**Why.** Weighted F1 is what every published MELD row uses (MELD paper, EmoBERTa), so it is the only number that is comparable. The selector consumes confidence, so calibration has to be measured. The laugh track and subtitle cuts in MELD can leak into event and prosody features, so those ablations are the artifact check. Class weighting is chosen on dev by weighted F1 and the macro-F1 cost of that choice is recorded.

**What would change it.** Nothing about the protocol. The deployed class-weight scheme is whatever dev picks.

## Runtime: onnxruntime only

**Decision.** Every model runs through onnxruntime; the heads are numpy forward passes from `.npz` files; torch exists only in the optional fine-tune extra.

**Alternatives.** Torch at runtime. sherpa-onnx as the runtime wrapper.

**Why.** One dependency behaves the same in the container and on the laptop, int8 paths are available, and inference needs no large install. sherpa-onnx stays a dev dependency for the transcript parity test and optional Piper TTS.

**What would change it.** A failure of the parity test that I could not fix in my own front end. The transcript would then come from sherpa-onnx and the encoder-state readout would be lost.

## Tokenizer: extracted from the spaCy wheel

**Decision.** I extract `vocab.json` and `merges.txt` from the `byte_bpe_processor` attribute inside the `en_core_web_trf-3.8.0` wheel with `msgpack`, commit them to `assets/roberta_tokenizer/`, and test known ids.

**Alternatives.** Wait for the Hugging Face Hub. Run the spaCy pipeline at runtime. Rebuild the merges from GPT-2.

**Why.** The attribute holds the original byte-level BPE vocab and merges, and the ids match the published roberta-base tokenizer (`<s>` 0, `</s>` 2, `Hello` 31414, `Ġworld` 232). Extraction needs only `msgpack`, not spaCy and not torch. The ids are asserted in tests and a decode-encode round trip is run on MELD sentences, so this is a tested fact, not a spot check.

**What would change it.** If the Hub opens, the files are compared byte for byte with the published tokenizer and the result recorded in PROVENANCE.md. A mismatch would replace the files.

## End-pointing: Silero VAD with a silence tail

**Decision.** Silero VAD v4 with a 400 ms silence tail, 250 ms pre-roll and a 15 s cap; the tail is a CLI flag with 600 ms shown as the patient setting.

**Alternatives.** The endpoint of a streaming ASR model. Energy-based end-pointing. Fixed-length chunks.

**Why.** A streaming ASR model would be a second model doing a job VAD does well. Energy fails with background noise. Fixed chunks cut words. The right tail depends on how the lamp should feel, so it is a flag and both values are benched.

**What would change it.** The user's preference after the demo, or a bench showing the 400 ms tail splits too many MELD utterances.

## Reinforcement learning: out of the core

**Decision.** No RL in the core. The one candidate, when to commit the label while the person is still talking, is a gated extension built only if the core is finished and measured by the end of day 2.

**Alternatives.** A reply-style bandit. RL inside the core from the start.

**Why.** The only reward MELD offers is label correctness, and the only sequential decision in this system is when to commit. That is an optimal-stopping problem: EMIT or WAIT at each provisional tick, reward one if the label is correct minus a cost per elapsed second, trainable offline by fitted Q-iteration on logged prefixes and comparable with three baselines (emit at the endpoint, at a fixed time, or when the maximum probability passes a threshold). A reply-style bandit has no reward. The prefix features are a long CPU run (estimate) and the challenge prefers a finished small system. If the policy is built and loses to the threshold rule, that is reported and the rule stays.

**What would change it.** The core done and measured with time left on day 2.

## Compared with the default stack

The design most reviewers expect is fine-tuned RoBERTa plus fine-tuned wav2vec2 or HuBERT with late fusion, a small LLM for the reply, and FastAPI. It would score about the same on MELD (published fine-tuned text-only rows sit at 63 to 65 weighted F1, and audio adds one to three points) and every piece is expensive where I am constrained: two fine-tuning jobs on four CPU cores, weights on a blocked host, a generator that adds nondeterminism to every turn, and a transport that puts serialization inside the numbers I have to report. Mine reads emotion, events and an embedding from one frozen ASR model in one pass, stacks two calibrated experts with a few hundred parameters, puts the words-versus-tone relation in the state as a field, selects an auditable reply, and streams over ZeroMQ with a media clock. Each piece is cheaper, deterministic or more testable than the default, and each is measured.
