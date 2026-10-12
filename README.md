# meld-realtime

This is a prototype for the real-time multimodal challenge on MELD, text and audio track. It listens to a stream of 16 kHz speech, or takes typed text, cuts it into utterances, and publishes a state for each one: what the words say (a text posterior over the seven MELD emotions), what the voice says (an audio posterior and audio events), how far the two disagree and in which direction, and a fused MELD emotion with a confidence. A short reply is then chosen from that state, together with the list of state fields that licensed it.

Everything runs locally on the CPU through onnxruntime: Silero VAD for end-pointing, SenseVoice-small for the transcript, the emotion posterior and the audio events in one pass, a frozen RoBERTa-base for the text, and small numpy heads trained on MELD. The inference path has 359.001 M parameters ([results/params.md](results/params.md)). Messages travel over ZeroMQ with a media clock, so a replayed file and a live microphone take the same path. The audio heads were trained on MELD audio from the Hugging Face mirror of MELD.Raw, and no audio is in the repository ([PROVENANCE.md](PROVENANCE.md)).

## The demo

```
./demo.sh
```

The script starts the service, a monitor and the replay client, streams one clip at real-time pace, prints every provisional state, the final state and the reply, and writes every message to [results/demo_log.jsonl](results/demo_log.jsonl). The default clip is `proxy/dev_0001.wav` (made by `tools/make_proxy_clips.py`), else `en.wav` from the SenseVoice release. `THREADS` (4), `ENDPOINT_MS` (500) and `DEMO_LOG` change the settings.

The committed log came from `./demo.sh models/sensevoice/test_wavs/en.wav` at the default endpoint silence of 500 ms, with the heads trained on MELD audio in `heads/`. `tools/render_demo_docs.py` writes the rest of this section from that log. `demo.sh` ends with this summary:

```
5 provisional, 1 final, 1 response, 0 error
utt 1 media 0.832-6.464 s
  transcript  'the tribal chieftain called for the boy and presented him with 50 pieces of gold' (conf 0.9243)
  audio       unknown  text joy  fused neutral@0.552 (stacked, calibrated=True, sentiment neutral)
  discrepancy jsd 0.4213 flag True direction tone_more_negative  degraded []
  timing      service_total 423.3 ms (sensevoice 210.81, text 31.31, prosody 30.34, heads 0.74, provisional in flight 139.2); client-side 432.097 ms; vad tail 512.0 ms
  reply       'Why the lower voice, if the news itself is neutral?' [neutral_neg_04, ask] after_final 0.198 ms
  grounding   ['fused.label=neutral@0.55', 'discrepancy.direction=tone_more_negative']
```

These are some of the fields of that final state, and the response that followed it. [docs/design.md](docs/design.md) (section 1.8) shows the whole state.

```json
{"kind": "final", "stream_id": "demo", "utt_id": 1, "t_media_start": 0.832, "t_media_end": 6.464,
 "asr": {"text": "the tribal chieftain called for the boy and presented him with 50 pieces of gold", "conf": 0.9243, "lang": "en", "tokens_per_s": 3.092},
 "text": {"partial": null, "context_k": 0, "keyphrase": "the tribal chieftain called"},
 "fused": {"label": "neutral", "confidence": 0.552, "sentiment": "neutral", "mode": "stacked", "calibrated": true},
 "discrepancy": {"jsd": 0.4213, "flag": true, "direction": "tone_more_negative"},
 "timing_ms": {"vad_tail": 512.0, "frontend": 9.37, "sensevoice": 210.81, "prosody": 30.34, "text_encoder": 31.31, "heads": 0.74, "service_total": 423.3, "provisional_in_flight": 139.2},
 "degraded": []}
```

```json
{"stream_id": "demo", "utt_id": 1, "text": "Why the lower voice, if the news itself is neutral?", "template_id": "neutral_neg_04", "style": "ask", "grounding": ["fused.label=neutral@0.55", "discrepancy.direction=tone_more_negative"], "t_emitted_ns": 8222875635354}
```

The text head's top class for the words is joy. SenseVoice puts most of its mass on `unknown` and most of the rest on neutral. The JSD between the text posterior and the raw SenseVoice posterior, 0.4213, is above the dev threshold of 0.3831 (`jsd_flag` in [heads/manifest.json](heads/manifest.json)), and the voice is less positive than the words, so the direction is `tone_more_negative`. The service runs in `stacked` mode, so the fusion head gives the label, neutral, with a calibrated confidence of 0.552. That lies between `low_conf` (0.47) and `high_conf` (0.73), so the selector is in the mid bucket, prefers an `ask` line and takes one from the neutral and `tone_more_negative` cell. `en.wav` is a read sentence, so the run shows the mechanism and is not an emotion result. The final state left the service 423.3 ms after the endpoint frame arrived, 139.2 ms of which it spent waiting for a provisional that was still running. The 512 ms of silence that ends the utterance comes before that.

## Setup on Ubuntu 24.04

```
sudo apt install curl ffmpeg libportaudio2
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python tools/fetch_models.py --dest models/
uv run python tools/fetch_meld_csv.py
uv run python tools/make_proxy_clips.py
uv run pytest -q
```

`uv sync` installs Python 3.12 (`.python-version`) and the locked dependencies, with the dev group: pytest, sherpa-onnx (for the proxy clips), praat-parselmouth (prosody) and msgpack. `uv sync --extra mic` adds sounddevice for the microphone client. ffmpeg is only needed to extract MELD audio, and libportaudio2 only for the microphone.

`tools/fetch_models.py` downloads each file, checks its sha256, extracts the archive members it needs and writes what it checked to [results/provenance_hashes.json](results/provenance_hashes.json). These are the files it keeps (sizes from [results/provenance_hashes.json](results/provenance_hashes.json)):

| file | bytes | use |
|---|---|---|
| `models/sensevoice/model.int8.onnx` | 239233841 | transcript, emotion and event posteriors |
| `models/sensevoice/tokens.txt` | 315894 | SenseVoice vocabulary |
| `models/sensevoice/test_wavs/` | 5 files | `en.wav` for the demo and the bench clips |
| `models/roberta-base-11.onnx` | 498649858 | text encoder |
| `models/silero_vad.onnx` | 643854 | end-pointing |
| `models/piper/en_US-lessac-medium/en_US-lessac-medium.onnx` | 63149198 | TTS for the proxy clips only |
| `verify_dl/en_core_web_trf-3.8.0-py3-none-any.whl` | 457421864 | only to rerun `tools/extract_tokenizer.py` |

The SenseVoice and Piper archives are downloaded to a temporary folder and deleted after extraction, so their own sizes are not recorded. [PROVENANCE.md](PROVENANCE.md) lists the URL, hash and licence of every external file.

`tools/fetch_meld_csv.py` puts the three MELD CSV files in `meld/` (their sizes and hashes are in [results/data_stats.md](results/data_stats.md)). `tools/make_proxy_clips.py` synthesizes 100 MELD dev utterances with Piper into `proxy/` for the demo and the replay bench. Their latencies are real, but their emotion labels are not a result, because TTS is affectless. `uv run python tools/edit_sensevoice_graph.py` writes `models/sensevoice/model_enc.int8.onnx`, which only `--encoder-states` needs.

## Running the service and the clients

Each command goes in its own terminal:

```
uv run python -m meld_rt.service --preload
uv run python -m meld_rt.clients.monitor --jsonl session.jsonl
uv run python -m meld_rt.clients.replay --wav models/sensevoice/test_wavs/en.wav --realtime --wait-ready 60
```

`replay --text "well that went great"` sends a typed utterance instead. For the microphone, `uv run python -m meld_rt.clients.mic --list-devices` lists the inputs and `--device N` picks one. Any number of subscribers can listen on the output port at the same time.

| service flag | default | what it sets |
|---|---|---|
| `--threads` | 4 | SenseVoice intra-op threads |
| `--text-threads` | 2 | RoBERTa intra-op threads |
| `--endpoint-silence-ms` | 500 | silence that ends an utterance |
| `--max-segment-s` | 15 | forced endpoint |
| `--provisional-hz` | 1 | provisional rate in the first window |
| `--provisional-window-s` | 4 | length of the provisional window |
| `--provisional-budget-ms` | 0 (off) | skip the next provisional after a slow one |
| `--fusion-mode` | `auto` | `stacked`, `poe` or `text_only` instead |
| `--jsd-flag` | 0.2 | discrepancy threshold while the manifest has no dev value |
| `--no-prosody`, `--encoder-states`, `--preload` | off | drop prosody, read encoder states, load models at start |
| `--in-addr`, `--out-addr` | `tcp://127.0.0.1:5555`, `:5556` | sockets |

The endpoint silence sets how long the robot waits after the speaker stops before it answers. The default of 500 ms becomes 16 frames of 32 ms, a 512 ms tail (`vad_tail_ms` in [results/replay_bench_vm.json](results/replay_bench_vm.json)). `tools/pause_stats.py` runs the service's VAD over MELD dev audio and counts the clips that each setting splits by a pause ([results/pause_stats_meld_dev.json](results/pause_stats_meld_dev.json)). Of the 1019 dev clips in which the VAD found speech, 400 ms (a 416 ms tail) splits 135, 500 ms splits 123 and 600 ms (a 608 ms tail) splits 92. 400 ms splits more clips than 500 ms, so the default stays at 500 ms under the rule in [docs/decisions.md](docs/decisions.md). Moving to 600 ms would need a person to listen to the clips that 500 ms splits and 600 ms keeps whole, and that check has not been done. Every bench ran at 500 ms.

## Real time

The robot should answer inside the pause after a person stops talking, and show that it is listening while they talk. Real time is therefore defined per turn and measured from the arrival of the endpoint frame, at 4 threads, with any provisional still running counted in. The silence tail comes on top and is reported apart. [docs/design.md](docs/design.md) (section 1.10) gives the reasons and assumptions. These are the container numbers from [results/measurements.md](results/measurements.md), all at the default endpoint silence of 500 ms with the stacked fusion deployed. The proxy bench streams 20 Piper clips, the long bench streams four tilings of `en.wav` (6 to 15 s) three times each, and the MELD bench streams 200 MELD test clips. The n in each cell counts the finals in that bucket.

| target | limit | proxy clips | long clips | MELD test clips |
|---|---|---|---|---|
| final state, utterances up to 5 s, p50 / p95 | 250 / 400 ms | 146.5 / 258.0 ms (n 19) | none | 152.3 / 241.7 ms (n 168) |
| final state, 5 to 8 s, p50 / p95 | 350 / 550 ms | 227.7 / 227.7 ms (n 1) | 264.2 / 283.9 ms (n 3) | 282.9 / 376.5 ms (n 19) |
| final state, 8 s to the 15 s cap, p50 / p95 | 700 / 1000 ms | none | 403.5 / 530.1 ms (n 9) | 371.2 / 491.3 ms (n 14) |
| reply after the final state, p95 | 5 ms | 0.420 ms | 0.280 ms | 0.350 ms |
| gap between provisionals in the first 4 s, p50 | 1 s (within 0.1 s) | 1.020 s | 1.020 s | 1.020 s |
| worker compute per second of audio | 0.3 s | 0.108 s | 0.102 s | 0.099 s |
| peak RSS of service, client and monitor in one process | no limit | 1187.7 MB | 1204.4 MB | 1262.1 MB |

Every target passes in the container, in every bucket that has finals. The MELD clips are cut at the end of the subtitle, and the VAD segments do not always line up with them. A final is attributed to the clip in which its speech stopped, or to the clip before when it stopped in the silence between two clips. By that rule, 30 of the 200 MELD clips have no final and 24 are split into several finals (the clip-attribution row of [results/measurements.md](results/measurements.md)). The MELD buckets therefore group finals by segment length, not by clip. The container recorded an "Intel(R) Xeon(R) Processor @ 2.10GHz" with nproc 4, 16093 MB of memory, AVX-512 VNNI and AMX, no GPU, Python 3.12.13 and onnxruntime 1.31.0 ([results/bench_components_vm.json](results/bench_components_vm.json)). The model files on disk take 0.6 MB (VAD), 228.2 MB (SenseVoice), 475.5 MB (RoBERTa), and 228.2 MB more for the optional edited graph. The benches ran one at a time, and every section of [results/measurements.md](results/measurements.md) records the load average at its start and end. The demo above is a single run, which waited behind a provisional. The laptop column is not measured yet ([docs/runbook_laptop.md](docs/runbook_laptop.md)).

## Results

All numbers are on MELD test, in percent. [RESULTS.md](RESULTS.md) has every row with its observations, rendered by `tools/render_results.py`, and [results/RESULTS.md](results/RESULTS.md) has the confusion matrices. The gold-text column feeds the text head the CSV transcript. The live column feeds it the SenseVoice transcript of the same clip, as the service does, with the service's ASR gate. The live column is the realistic number for the running system on MELD clips.

| model | gold text: weighted F1 [95 % CI] | SenseVoice transcripts: weighted F1 [95 % CI] | macro F1, gold / transcripts |
|---|---|---|---|
| text only, deployed head (k = 0, frozen RoBERTa-base) | 59.62 [57.38, 62.03] | 42.61 [40.28, 45.37] | 41.22 / 23.86 |
| text and audio, stacked fusion (deployed) | 60.77 [58.65, 63.00] | 47.05 [44.53, 49.57] | 39.24 / 27.50 |

| condition | weighted F1 [95 % CI] | macro F1 |
|---|---|---|
| majority class | 31.27 [28.62, 34.31] | 9.28 |
| audio only, raw SenseVoice tag | 48.52 [46.33, 50.96] | 28.18 |
| audio only, calibrator A (deployed) | 48.02 [45.66, 50.64] | 26.92 |
| product of experts, dev weight | 58.26 [55.91, 60.69] | 35.86 |
| stacked fusion without events and prosody | 60.82 [58.73, 63.08] | 39.40 |
| sentiment, fused head (3 classes) | 69.17 [67.37, 70.96] | 66.45 |

| published row | weighted F1 | source |
|---|---|---|
| text-CNN, text | 55.02 | MELD paper, Table 11 |
| DialogueRNN, text | 57.03 | MELD paper, Table 11 |
| DialogueRNN, text and audio | 60.25 | MELD paper, Table 11 |
| EmoBERTa, RoBERTa-large, no context | 63.46 | EmoBERTa, Table 2 |
| SenseVoice-S, audio, no fine-tuning on MELD | 57.7 | FunAudioLLM report, Table 8 |
| DialogueRNN sentiment, text and audio | 67.56 | MELD paper, Table 13 |

The stacked fusion gains 1.15 points of weighted F1 over the text head on gold text ([RESULTS.md](RESULTS.md)). The gain is small, and the two intervals overlap. Its macro F1 is lower (39.24 against 41.22), and it scores 0.00 on fear, against 20.29 for the text head ([RESULTS.md](RESULTS.md), "Per-class F1"). On the 489 test utterances where the text posterior and the raw SenseVoice posterior disagree most (JSD above the dev threshold), the fusion scores 49.62 against 51.44 for the text head alone, so the audio branch shows no gain where words and tone disagree. Dropping the event and prosody inputs leaves the fused score inside the deployed fusion's interval, so those inputs add nothing measurable on MELD. The SenseVoice transcripts have a corpus WER of 0.348488 on test (`cache/wer_test.json`), and on them the text head trained on gold text drops from 59.62 to 42.61. The next step would be to train the text head on SenseVoice transcripts of train and dev. It was not done.

The raw SenseVoice tag scores 48.52, below the 57.7 that the FunAudioLLM report gives for MELD (Table 8). On 97.51 % of the test clips, more than half of its emotion mass is on OTHER or EMO_UNKNOWN. The report does not describe its MELD protocol in enough detail to reproduce it, and it does not list the corpora behind SenseVoice's emotion labels, so MELD may be among them and the SenseVoice row is not a zero-shot number ([PROVENANCE.md](PROVENANCE.md)). The text-only sentiment head is not scored in this run, because the committed sentiment head is the fused one ([results/01_text_only.md](results/01_text_only.md) has the earlier text-only run).

## Parameters

| component | parameters |
|---|---|
| SenseVoice-small int8 | 234.000 M |
| RoBERTa-base | 124.646 M |
| Silero VAD v4 | 0.156 M |
| TextContextHead (MLP 772-256-7) | 0.200 M |
| AudioCalibrator A (LR 24-7) | 175 |
| Fusion (LR 19-7) | 140 |
| SentimentHead, fused (LR 19-3) | 60 |
| total on the inference path | 359.001 M, 5.98 % of the 6 B cap |

`tools/count_params.py` counts these from the files on disk ([results/params.md](results/params.md)). Piper (15.651 M) only makes the proxy clips and is not on the inference path.

## Completed and left out

| item | state |
|---|---|
| streaming service with VAD end-pointing, provisionals, the final state and a grounded reply | done, tested |
| text heads, trained on dev choices and scored once on test | done ([RESULTS.md](RESULTS.md)) |
| MELD audio: WAV extraction, feature caches, audio calibrators, fusion head, fused sentiment head, discrepancy threshold from dev | done ([heads/manifest.json](heads/manifest.json)) |
| audio, fused, ablation, live ASR, disagreement-subset and early-commit rows | done ([RESULTS.md](RESULTS.md)) |
| parameter count and container measurements, with a replay bench on 200 MELD test clips | done ([results/params.md](results/params.md), [results/measurements.md](results/measurements.md)) |
| one input through to both outputs | done ([results/demo_log.jsonl](results/demo_log.jsonl)) |
| reply bank of 124 lines | done, tested (`meld_rt/response/bank.yaml`) |
| a text head trained on SenseVoice transcripts of train and dev | not done |
| fine-tuning RoBERTa on MELD text | not done |
| the RL extension (when to commit the label) | not done |
| laptop measurements and a live microphone session | not done, to be run on the laptop ([docs/runbook_laptop.md](docs/runbook_laptop.md)) |
| listening audit of 50 flagged clips | not done: `tools/audit_disagreement.py` draws the sample, and a person has to listen |
| vision and the three-modality extension | left out |
| an LLM reply generator | left out |
| fine-tuning SenseVoice, wav2vec2 or HuBERT | left out |
| a spoken reply | left out: Piper only makes the proxy clips |
| macOS | not measured |

[docs/design.md](docs/design.md) (section 1.13) and [docs/decisions.md](docs/decisions.md) give the reasons.

## How the audio results were produced

The audio is `MELD.Raw.tar.gz` from the Hugging Face mirror `declare-lab/MELD` at revision `9abc51ee7903424ffb971297608aa6d3d0de3bfa`. `tools/meld_raw_source.py` writes its size, LFS sha256 and licence field to [results/meld_raw_source.json](results/meld_raw_source.json), and [PROVENANCE.md](PROVENANCE.md) lists them. The build container fetched the archive from `resolve/main`, which serves that revision, and checked its size only. It then read the archive as a stream with `python3 -I tools/stream_meld_raw.py MELD.Raw.tar.gz <staging folder> . --chunk 400 --jobs 4`: the MP4 files of each split went to the staging folder in batches of 400, and `tools/extract_audio.py --raw <staging folder> --out audio/ --split <split> --jobs 4 --delete-mp4` converted each batch. The archive was then deleted. With room to unpack the archive, these commands do the same conversion in one pass and then everything that followed it in the container:

```
curl -L -o MELD.Raw.tar.gz https://huggingface.co/datasets/declare-lab/MELD/resolve/9abc51ee7903424ffb971297608aa6d3d0de3bfa/MELD.Raw.tar.gz
sha256sum MELD.Raw.tar.gz
tar -xzf MELD.Raw.tar.gz
for s in train dev test; do tar -xzf MELD.Raw/$s.tar.gz -C MELD.Raw; done
uv run python tools/extract_audio.py --raw MELD.Raw --out audio/ --split train,dev,test --jobs 4
uv run python tools/edit_sensevoice_graph.py
uv run python tools/cache_audio_features.py --split dev --audio audio/ --out cache/audio_dev.npz --prefixes 2,4 --encoder-states --threads 4
uv run python tools/cache_audio_features.py --split test --audio audio/ --out cache/audio_test.npz --prefixes 2,4 --encoder-states --threads 4
uv run python tools/cache_audio_features.py --split train --audio audio/ --out cache/audio_train.npz --prefixes 2,4 --encoder-states --threads 4
uv run python tools/embed_text.py --split train --out cache/text_train.npy
uv run python tools/embed_text.py --split dev --out cache/text_dev.npy
uv run python tools/embed_text.py --split test --out cache/text_test.npy
uv run python tools/embed_text.py --split test --from-asr
uv run python tools/embed_text.py --split test --from-asr asr_text_p2
uv run python tools/embed_text.py --split test --from-asr asr_text_p4
uv run python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml
uv run python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml --ablation
uv run python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json
uv run python tools/render_results.py
uv run python tools/replay_bench.py --clips "audio/test/*.wav" --n 200 --realtime --out results/replay_bench_vm_meld.json
uv run python tools/pause_stats.py --wavs "audio/dev/*.wav" --label meld_dev --bench results/replay_bench_vm.json
uv run python tools/count_params.py --models models/ --heads heads/ --out results/params.md
uv run python tools/render_measurements.py --out results/measurements.md
```

The sha256 printed by the second command should be the LFS hash (`x-linked-etag` in [results/meld_raw_source.json](results/meld_raw_source.json)). `extract_audio.py` writes 16 kHz mono WAV files under `audio/` and lists missing or failed clips in `audio/missing.json`: train clip `125_3` is an MP4 that ffmpeg could not open and dev clip `110_7` was not found (`audio_rows_dropped` in [heads/manifest.json](heads/manifest.json)). `cache_audio_features.py` runs the service's own SenseVoice and prosody code on every clip, with the 2 s and 4 s prefixes for the early-commit rows. `embed_text.py --from-asr` embeds the SenseVoice transcripts for the live ASR rows. `train_heads.py` without `--text-only` adds the audio calibrators, the fusion head, the fused sentiment head and the dev JSD threshold, and the service then runs in `stacked` mode. `--ablation` trains the ablation heads and leaves the deployed heads as they are. The text and audio caches are not committed. [docs/runbook_laptop.md](docs/runbook_laptop.md) describes the listening audit that follows.

## Laptop runbook

[docs/runbook_laptop.md](docs/runbook_laptop.md) lists the commands for the laptop in order: setup, the measurements, the live microphone session, how to re-render [results/measurements.md](results/measurements.md), and the listening audit.

## macOS

On macOS, Homebrew takes the place of apt: `brew install portaudio` for the microphone client and `brew install ffmpeg` for MELD audio. Nothing was run or measured on macOS. The host block of [results/measurements.md](results/measurements.md) reads `lscpu`, `free` and `/proc`, which macOS does not have, so there the CPU model and the memory would read "unknown" and every CPU flag "no".

## Repository layout

```
meld_rt/          the package: service, protocol, state, experts, heads, selector, clients
tools/            downloads, caches, training, evaluation, benches and report renderers
configs/          heads.yaml, the training grid and threshold rules
heads/            trained heads and manifest.json
assets/           the RoBERTa tokenizer files
results/          every measured number, written by the tools
tests/            pytest suite
docs/             design.md, decisions.md, runbook_laptop.md
demo.sh           one utterance through the system
RESULTS.md        evaluation summary
PROVENANCE.md     external files, hashes and licences
AI_USE.md         how an AI coding agent was used
```

[docs/design.md](docs/design.md) has the full layout, file by file.

## AI use

[AI_USE.md](AI_USE.md) says how an AI coding agent was used to build this.
