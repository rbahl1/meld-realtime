# Decisions

Each entry gives the decision, the alternatives that were considered, the reasons, and what would change it. Numbers measured here are cited from the file a script wrote. Published numbers are cited with their source. [docs/design.md](design.md) describes the system itself.

## Track

The primary track is text and audio. The optional extension adds the face from the video as a third input (`--vision`), and the core runs without it.

Alternatives considered: text and vision as the primary track, or all three modalities in the core.

The MELD authors left visual features out of their own baselines, "as video-based speaker identification and localization is an open problem" (MELD paper, Section 4.1). The raw video was not reachable when the track was chosen. It later came inside the MELD.Raw archive with the audio. The core uses only the audio track, and the extension reads the frames too. The build machine has no camera. A robot's camera would see one person up close, which multi-party sitcom frames do not resemble. Audio carries tone, laughter and timing, which is the part of the challenge's premise that MELD can test. Tested on MELD with gold text, the audio branch adds a small gain in weighted F1 on the whole test set, inside overlapping intervals. It lowers the macro F1, and it adds no gain on the utterances where the words and the tone disagree most (Fusion, below).

What would change it: a single-face camera at the robot, and a measured gain from the face in sessions with it. The face would then be on by default instead of behind a flag.

Outcome: the extension adds the face, and it is measured on MELD frames only. No live camera session has run. Alone the face scores 34.37 weighted F1 on test, against 31.27 for the majority class (Face expert, below). Added to text and audio it gives a small gain whose paired interval lies above zero (Three-way fusion, below). In the demo clip ([README.md](../README.md), "Text, voice and face") the face and the words read neutral. SenseVoice's top named class is anger, and fusion3 gives anger, the MELD label.

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

## Face expert

The face expert runs two models of the OpenCV zoo in OpenCV DNN (`meld_rt/experts/vision.py`). YuNet finds the faces, and the facial expression model (MobileFaceNet, taken to be trained on RAF-DB, [PROVENANCE.md](../PROVENANCE.md)) gives 7 logits and a 512-value embedding per face.

Alternatives considered: the same two models in onnxruntime, or a face model fine-tuned on MELD frames.

Both models are frozen and small, 0.053 M and 1.194 M parameters ([results/params.md](../results/params.md)), so they can run on every frame on the CPU. The seven expression classes map one to one onto MELD's seven, as SenseVoice's do. The embedding is the input of the model's last layer, and calibrator B reads it ([docs/design.md](design.md), section 2.3). The face models are the one exception to onnxruntime (Runtime, below), and the reason is speed. In onnxruntime every PReLU of the expression graph is wrapped in layout reorders, and on one thread the graph ran slower per face than in OpenCV DNN. No committed script repeats that timing, so it is not in a results file. The measured cost is that of the whole vision thread: 30.49 ms per processed frame on average over 7044 frames in the replay bench with video (10.0 frames per second), the JPEG decode and the row building included ([results/measurements.md](../results/measurements.md)). Fine-tuning a face model on MELD frames has the cost that kept the audio and text encoders frozen (Audio expert, above).

What would change it: an onnxruntime build that runs the expression graph as fast as OpenCV DNN, which would remove the second runtime, or a face expert that beats the calibrated one on dev by more than its interval. A dataset catalog lists RAF-DB as non-commercial, and its own terms were not read ([PROVENANCE.md](../PROVENANCE.md)), so any other use needs another expression model or the RAF-DB terms checked.

Outcome: the face alone is weak on MELD. A face was found in 99.46 % of the test clips. The raw expression label scores 32.22 weighted F1 on test and the calibrated face 34.37, against 31.27 for the majority class, and the calibrated face scores 0.00 F1 on fear, sadness and disgust ([results/RESULTS_vision.md](../results/RESULTS_vision.md), [RESULTS.md](../RESULTS.md)). The vision thread dropped no frame in either replay bench with video.

## Speaker face

The speaker's face is the track whose mouth motion follows the log RMS envelope of the audio best, by Pearson correlation (rule `asd`), chosen on dev among four rules.

Alternatives considered: the largest face (`largest`), the face seen in the most frames (`present`), or the face with the most mouth motion beyond its whole-face motion (`mouth`).

A MELD frame often shows several people, so the speaker's face has to be chosen. Of the 1103 dev clips with a face, 81.50 % have more than one track, and on 67.09 % the four rules do not all choose the same one ([RESULTS.md](../RESULTS.md), "Choices made on dev"). `asd` uses the audio, which a robot has as well, and it falls back to `largest` when no track has 4 motion samples. The four rules are cheap to compare, so the choice was left to dev: calibrator A was fitted on train under each rule and scored on all dev rows.

What would change it: a camera at the robot that sees one person, where every rule gives the same answer. Merging the fragments of one face ([docs/design.md](design.md), section 1.13) would change the tracks the rules choose among, and the dev comparison would then run again.

Outcome: dev chose `asd` at 25.82 weighted F1, against 25.57 for `present`, 25.53 for `mouth` and 25.19 for `largest`, and `asd` fell back to `largest` on 4.08 % of the dev clips with a face ([results/RESULTS_vision.md](../results/RESULTS_vision.md)). The four rules differ by less than one point on dev, although they often choose different faces.

## Fusion

The fusion head is a multinomial logistic regression over both 7-way log posteriors, their entropies, the ASR confidence, the laughter probability and the unknown mass (19 inputs), trained on cross-fitted expert posteriors and temperature-scaled on dev. It is trained on MELD audio and deployed (`fused.mode` `stacked`). A product of experts with one dev-chosen weight is its comparison. When the audio calibrator is missing, the live path falls back to a fixed product of the text posterior and the raw SenseVoice posterior, and the state says so (`fused.mode` `poe_uncalibrated`, `calibrated` false).

Alternatives considered: a concatenation MLP over the 768-value and 1024-value embeddings, cross-attention between audio frames and text tokens, or a single product-of-experts weight.

MELD's train split has 9989 utterances ([results/data_stats.md](../results/data_stats.md)), which does not support a large fusion model trained on a CPU. The stacked model has per-modality weights that can be read, and a calibrated confidence, which the selector needs. The fixed product keeps both modalities on the live path when the audio calibrator is missing.

The fixed product needed its own confidence rule. The maximum of a product of two soft posteriors is sharp, and it is not a probability of being right. Reported as the confidence, it let the selector speak as if it were sure. So in `poe_uncalibrated` mode the reported confidence is the text head's temperature-scaled probability of the fused label, the only calibrated number on that path, and the selector never puts an uncalibrated state above the mid bucket (`reported_confidence` in `meld_rt/service.py`, `tests/test_service_rules.py`, `tests/test_selector.py`).

What would change it: if the product of experts matches the stacked model on dev within the interval, the simpler rule is deployed. If the disagreement subset shows no gain over text alone, the fusion claim comes out of the write-up.

Outcome: at its dev-chosen weight of 0.75 the product of experts scores 55.56 weighted F1 on dev, against 57.95 for the stacked fusion ([RESULTS.md](../RESULTS.md), "Choices made on dev"), so the stacked fusion stays deployed. On the 489 test utterances with a JSD above the dev threshold, the stacked fusion scores 49.62 against 51.44 for the text head alone, so the second rule applies. The write-up makes no claim that the audio branch helps where the words and the tone disagree. On the whole test set with gold text, the fusion scores 60.77 against 59.62 for the text head. The gain is small, and the two 95 % intervals overlap. Its macro F1 is lower (39.24 against 41.22), and it scores 0.00 on fear, against 20.29 for the text head. On SenseVoice transcripts, which is what the running system sees, the fusion scores 47.05 and the text head 42.61 ([RESULTS.md](../RESULTS.md), "Gold text and live ASR transcripts"). The fusion stays deployed because it scores higher than the text head on dev (57.95 against 57.56).

## Three-way fusion

In vision mode the fused posterior comes from fusion3 (`heads/fusion3.npz`, `fused.mode` `stacked_tav`). It is a multinomial logistic regression over 29 inputs: the log posteriors of text, audio and face (7 each), their three entropies, the ASR confidence, the laughter probability, the unknown mass, a face flag and the presence. It is trained on out-of-fold train posteriors of the three experts and temperature-scaled on dev. Without a face it runs with the uniform face posterior and the last two inputs at 0.

Alternatives considered: the alternatives of the two-way fusion (Fusion, above) with the face added, or the two-way fusion with the face shown in the state only.

The reasons of the two-way fusion hold: 9989 train utterances, weights that can be read per modality, and a calibrated confidence for the selector and the reply prompt. The three expert heads are refit in the 5 folds by dialogue of `tools/train_heads.py`. Refit on these folds, the two-way fusion comes out equal to the deployed one array for array (`reproduces_deployed_fusion` in `vision.cross_fit` of [heads/manifest.json](../heads/manifest.json)), so fusion3 learns from the same out-of-fold rows as the deployed two-way fusion, with the face added. The claim rule was fixed before the test run: the write-up says the face adds to text and audio only if the paired 95 % interval of fusion3 minus the two-way fusion on gold-text test lies above zero.

What would change it: if fusion3 did not score above the two-way fusion on dev, vision mode would keep the two-way fusion and show the face in the state only. A new face expert or speaker rule would be a new condition, scored once on test under the same rule.

Outcome: fusion3 scores 59.53 weighted F1 on dev against 57.95 for the two-way fusion, so it is deployed in vision mode. On gold-text test it scores 61.56 against 60.77, a paired difference of 0.80 points [0.20, 1.48] ([results/RESULTS_vision.md](../results/RESULTS_vision.md)). The interval lies above zero, so by the rule the face adds measured value. The gain is small: fusion3 changes 80 of the 2610 gold-text labels of the two-way fusion, 36 from wrong to right and 20 from right to wrong. On SenseVoice transcripts the difference is 0.91 [0.31, 1.52]. On the 557 test clips where the face disagrees most with the words (JSD above 0.5729), fusion3 scores 64.04 against 62.87 for the two-way fusion and 64.88 for the text head, and both paired intervals include zero ([-0.29, 2.60] and [-3.71, 2.20]). No gain is shown there, so the write-up makes no claim that the face helps where it disagrees with the words. Sentiment3 scores 69.03 against 69.17 for the two-way sentiment head.

## Context

The text head sees the current utterance and up to two previous utterance embeddings with same-speaker flags, and k (0, 1 or 2) is chosen on dev. Dev chose k = 0.

Alternatives considered: a GRU over the dialogue, a dialogue transformer, or no context at all.

EmoBERTa puts the gain from past context at about one point (64.55 with past utterances against 63.46 without, Table 2). A window head can capture that, needs no torch at run time and is causal. The robot hears one speaker while MELD is multi-party, so the same head is also scored with the window restricted to the same speaker's earlier utterances. On dev, the best k = 1 and k = 2 points reached 56.14 and 55.75 weighted F1 against 57.56 at k = 0 ([results/01_text_only.md](../results/01_text_only.md)). The deployed head is therefore the k = 0 head of 0.200 M parameters, and context leaves the run-time path except for the turn-index flags.

What would change it: a fine-tuned encoder whose dev score rises with context, or a robot that talks to several users, which would also need speaker continuity.

## Replies

The default reply is one of the 124 lines in `meld_rt/response/bank.yaml`, chosen by fused label and discrepancy direction, with a confidence rule that forces a line without an emotion claim when the posterior is weak, a keyphrase slot filled from the transcript, and a `grounding` list that names the state fields that licensed the line. The high bucket, which asserts a feeling or names the gap, needs a calibrated confidence. The loader enforces the word limits and rejects exclamation marks.

Alternatives considered: a small instruct LLM, GPT-2 small from the ONNX zoo fine-tuned on MELD next lines, or retrieval of MELD next utterances.

An LLM would add latency on a CPU, would be nondeterministic, could not have its grounding unit-tested, and had its weights on a host the build environment could not reach at the time. Retrieval would return copyrighted sitcom lines that are not grounded in the audio. The selector takes 0.009 ms at p50 at 4 threads ([results/bench_components_vm.json](../results/bench_components_vm.json)), and the reply followed the final state within 0.420 ms at p95 on the proxy clips and 0.350 ms on the MELD test clips ([results/measurements.md](../results/measurements.md)). Its behavior is tested: every cell has at least three variants, changing only the audio posterior changes the line on a fixed set of cases, a low confidence always gives `hold_back`, no line repeats within five turns, and an uncalibrated state never reaches the high bucket (`tests/test_selector.py`).

What would change it: a need for open-ended dialogue rather than a reaction, or a GPU. An LLM would then take the state as its prompt, and the grounding list would become a check on its output.

Outcome: the extension builds the LLM path as an option on the CPU (`--reply llm`, Streamed reply, below). The LLM takes the state as its prompt, and the bank stays the default and the fallback. Of the four reasons above, the latency held: in the two replay benches with video, the reply follows the final state by 549.92 ms at p50 with the LLM and by 0.24 ms with the bank ([results/measurements.md](../results/measurements.md)). The other three did not hold as stated. Greedy decoding gave the same text at 2 and 4 threads on all 200 bench states (`same_text_across_threads` 1.0 in [results/generator_bench_vm.json](../results/generator_bench_vm.json)). The grounding list and the masks are unit-tested (`tests/test_generator.py`), although whether a reply follows from what the person said is not tested, and no person rated the replies. The weights were reachable later in the build.

## Streamed reply

With `--reply llm` the reply comes from Qwen2.5-0.5B-Instruct (the onnx-community int8 export) in onnxruntime, greedy, with at most 40 new tokens, and goes out word by word as `response.delta`. A lexicon mask keeps it from naming an emotion that the state does not license, and the bank line is its fallback. The bank stays the default reply.

Alternatives considered: Qwen2.5-1.5B-Instruct, the q4 export of either model, or the bank alone (Replies, above).

The model runs locally on the CPU, like the rest of the system, and the model card of its base model gives the licence apache-2.0 ([PROVENANCE.md](../PROVENANCE.md)). The choice rule was fixed before the bench on 200 states: the largest model with a file whose first-token p95 is at most 350.0 ms and whose decode rate is at least 15.0 passes per s at 4 threads, and within one model the int8 file unless q4 is faster by more than 1.2x at the same reply length with a check pass rate no lower. The decode rate and the q4 test were defined this way after preliminary benches of all four files on 20 states (the docstring of `tools/bench_generator.py`). The original wording asked for 15 tokens per s and a q4 file faster by more than 20 %, and it picks the same file: at 4 threads the 0.5B int8 file writes 23.74 final-reply tokens per s and the 1.5B files 8.79 and 9.49, and the 0.5B q4 file has the longer full-reply p50 (496.41 against 378.52 ms, [results/measurements.md](../results/measurements.md)). Every file replied to the same 200 test states ([results/generator_bench_vm.json](../results/generator_bench_vm.json), [results/generator_bench_vm_1.5b.json](../results/generator_bench_vm_1.5b.json)). Neither 1.5B file reaches the decode rate (13.22 and 11.64 passes per s). The 0.5B q4 file is slower than int8 at the same reply length ("q4 speedup 0.80x on a reply of 9.82 tokens", the reason recorded in the file). The 0.5B int8 file meets both targets, with a first-token p95 of 100.27 ms and 28.47 passes per s.

The prompt asks the model to name only the feelings that the evidence supports, and the decoder enforces it. It masks the tokens of the emotion words of every class outside `allowed` (`meld_rt/response/lexicon.py`), and a word guard rolls back an emotion word that spans several tokens. Below `low_conf`, `allowed` is empty, and the reply asks a question and names no feeling. In the bench the mask changed the greedy token at least once in a share of 0.11 of the deployed file's replies, and no final reply named a class outside `allowed` (`forbidden_words` 0 in `check_failures`). The bank line goes out as one delta when the generator is missing or fails, when its first word does not come within 2.0 s, when the reply comes back empty, or when a newer final of the stream is waiting. The bank lines are unit-tested and answer within the 5 ms target of design 1.10, which is why the bank stays the default.

What would change it: a machine on which a 1.5B file meets both targets, since the rule would then deploy it. Ratings of the replies by a person would also change it, since the checks test the form of a reply and not whether it follows from what was said. Ratings that favored the generated replies would make `--reply llm` the default, and ratings that favored the bank lines would drop the option.

Outcome: in the replay bench with video and `--reply llm`, all 206 replies came from the generator, with no fallback. The first streamed word left the service 388.85 / 855.99 ms (p50 / p95) after the endpoint frame arrived, and the whole reply 549.92 / 1333.29 ms after the final state ([results/measurements.md](../results/measurements.md)). No target for these was set before the run, so they have no pass or fail. In the generator bench the deployed file passes `check_reply` on a share of 0.97 of the replies, so 6 of the 200 replies fail: 4 are over 20 words, 2 copy a phrase of the prompt (one of them is also over length) and 1 has a meta word (`check_failures` and `replies` in [results/generator_bench_vm.json](../results/generator_bench_vm.json)). In the demo the fused confidence, 0.3869, was below `low_conf` (0.56), so the reply held back. It asked "What happened to you?" and named no feeling ([results/demo_tav_log.jsonl](../results/demo_tav_log.jsonl)).

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

What would change it: nothing in the protocol. Every condition is scored. The text-only sentiment row reads its own head file (`heads/sentiment_text_only.npz`), so the fused sentiment head that the service uses did not change ([RESULTS.md](../RESULTS.md), "Sentiment on MELD test").

Outcome of the artifact check: without the event and prosody inputs the stacked fusion scores 60.82 against 60.77 for the deployed one, and the two single ablations also stay inside the deployed fusion's interval ([RESULTS.md](../RESULTS.md), "Ablations"). The event and prosody inputs add nothing measurable on MELD test, so the fused score does not rest on a laugh-track signal in them. Without them the calibrator reads 9 inputs instead of 24 and the fusion 18 instead of 19, at no measured cost. The deployed heads were not changed and still read those inputs.

## Runtime

Every model runs through onnxruntime except the two face models of the extension, which run in OpenCV DNN (Face expert, above). The heads, the commit policy included, are numpy forward passes from `.npz` files.

Alternatives considered: torch at run time, or sherpa-onnx as the run-time wrapper.

One dependency behaves the same in the container and on the laptop, int8 kernels are available, and inference needs no large install. sherpa-onnx stays a dev dependency for the transcript parity test and the Piper proxy clips.

What would change it: a failure of the parity test that the front end could not fix. The transcript would then come from sherpa-onnx, and the encoder-state readout would be lost.

## Reinforcement learning

The RL component is a commit policy. At each full-window provisional of an utterance it commits the fused label now or waits for the next one, and without a commit the final is the commit. It is trained offline by fitted Q-iteration on logged prefixes and runs in the service with `--vision` (`meld_rt/commit.py`, `heads/commit_policy.npz`).

Alternatives considered: a reply-style bandit, or RL in the core from the start.

The only reward MELD offers is label correctness, and the only sequential decision in this system is when to commit. A reply-style bandit has no reward at all. When to commit is an optimal-stopping problem: EMIT or WAIT at each provisional tick, a reward of one for a correct label minus a cost per elapsed second, trained offline by fitted Q-iteration on logged prefixes, and compared with three baselines (emit at the endpoint, at a fixed time, or when the maximum probability passes a threshold). The traces run every MELD clip through the service's experts at 1, 2, 3 and 4 s while the segment is open, with out-of-fold posteriors on train ([docs/design.md](design.md), section 2.8). The claim rule was fixed before the test run: RL adds measured value over a baseline at a cost c only where the paired 95 % interval of the test reward difference lies above zero.

What would change it: a reward for replies (ex. a person's reaction in a live session), which would make the reply a target for RL as well. Traces from live sessions at the robot, where one face is in view, would replace the MELD traces, and the policy and the baselines would be fitted and compared again.

Outcome: by the rule the policy adds measured value over the endpoint at every cost of the grid, over the tuned fixed time at c = 0.02 only (0.0124 [0.0017, 0.0238]) and over the tuned threshold at c = 0.2 only (0.0060 [0.0016, 0.0104]) ([results/commit_policy.md](../results/commit_policy.md)). At the deployed c = 0.05 no gain over the fixed time or the threshold is shown. At that cost the policy commits 86.74 % of the test labels before the endpoint, at a mean of 1.362 s against 4.046 s, and the early labels are right less often: an accuracy of 51.65 against 53.18 at the endpoint, and a weighted F1 of 41.81 against 47.96. Its Q_wait model, an MLP chosen on dev, stopped at the cap of 10 iterations before its decisions stopped changing. In the replay bench with video it committed 168 of 206 finals before the endpoint, and the committed label equals the final's for a share of 0.767 ([results/measurements.md](../results/measurements.md)). In the demo it committed neutral at 2.016 s after the segment start, while the final and the MELD label are anger, so the early label was wrong ([results/demo_tav_log.jsonl](../results/demo_tav_log.jsonl)).

## Commit cost

The deployed commit policy uses a cost of c = 0.05 per second (`heads/commit_policy.npz`). The policy and the three baselines are also fitted and tested at 0.02, 0.1 and 0.2, and the whole grid is reported.

Alternatives considered: choosing c on dev, or reporting the deployed cost alone.

The reward is one for a correct label minus c times the commit time in seconds. At c = 0.05, one more second of waiting raises the reward only if it raises the accuracy by more than five points. The cost c says how much a second of waiting is worth against a correct label. That is a preference about the robot, and MELD cannot measure it. It cannot be chosen on dev either, because each c defines its own reward and rewards under different costs are not comparable. So 0.05 was fixed before the run, and the grid around it is reported so that a reader with another preference can read the outcome at that cost ([results/commit_policy.md](../results/commit_policy.md)).

What would change it: a measure of what an early label is worth (ex. sessions in which the robot acts on the committed label and a person judges the timing). A cost from such a measure would replace 0.05.

Outcome: the cost decides which claims hold (Reinforcement learning, above). At c = 0.05 a commit at 1 s has a slightly higher test reward than the policy (0.4498 against 0.4484).

## Compared with the usual stack

The usual design is a fine-tuned RoBERTa and a fine-tuned wav2vec2 or HuBERT with late fusion, a small LLM for the reply, and FastAPI. Published fine-tuned text rows reach 63.46 to 65.61 weighted F1 (EmoBERTa, Table 2). This system reads emotion, events and a transcript from one frozen ASR model in one pass, stacks two small expert heads, keeps the relation between words and tone in the state as a field, selects a reply that can be audited, and streams over ZeroMQ with a media clock. Each part is cheaper to run, deterministic, or easier to test than its usual counterpart. The frozen text head scores 59.62 ([RESULTS.md](../RESULTS.md)), so the price of not fine-tuning is measured against those rows. On the audio side, the stacked fusion scores 60.77, and DialogueRNN on text and audio (60.25, MELD paper, Table 11) is inside its interval. On SenseVoice transcripts instead of gold text, the fusion scores 47.05 ([RESULTS.md](../RESULTS.md)). The extension adds a face expert and, as an option next to the bank, a small LLM for the reply (Streamed reply, above).
