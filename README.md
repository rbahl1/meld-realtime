# meld-realtime

This is a prototype for the real-time multimodal challenge on MELD, text and audio track. It listens to a stream of 16 kHz speech, or takes typed text, cuts it into utterances, and publishes a state for each one: what the words say (a text posterior over the seven MELD emotions), what the voice says (an audio posterior and audio events), how far the two disagree and in which direction, and a fused MELD emotion with a confidence. A short reply is then chosen from that state, together with the list of state fields that licensed it.

Everything runs locally on the CPU through onnxruntime: Silero VAD for end-pointing, SenseVoice-small for the transcript, the emotion posterior and the audio events in one pass, a frozen RoBERTa-base for the text, and small numpy heads trained on MELD. The inference path has 359.003 M parameters ([results/params.md](results/params.md)). Messages travel over ZeroMQ with a media clock, so a replayed file and a live microphone take the same path. The audio and fused rows of the evaluation wait for MELD audio, which was not available in the build environment.

## The demo

```
./demo.sh
```

The script starts the service, a monitor and the replay client, streams one clip at real-time pace, prints every provisional state, the final state and the reply, and writes every message to [results/demo_log.jsonl](results/demo_log.jsonl). The default clip is `proxy/dev_0001.wav` (made by `tools/make_proxy_clips.py`), else `en.wav` from the SenseVoice release. `THREADS` (4), `ENDPOINT_MS` (500) and `DEMO_LOG` change the settings.

The committed log came from `./demo.sh models/sensevoice/test_wavs/en.wav` at the default endpoint silence of 500 ms. The script ends with this summary:

```
5 provisional, 1 final, 1 response, 0 error
utt 1 media 0.832-6.464 s
  transcript  'the tribal chieftain called for the boy and presented him with 50 pieces of gold' (conf 0.9243)
  audio       unknown  text joy  fused neutral@0.3467 (poe_uncalibrated, calibrated=False, sentiment neutral)
  discrepancy jsd 0.4213 flag True direction tone_more_negative  degraded []
  timing      service_total 456.69 ms (sensevoice 175.17, text 36.65, prosody 30.64, heads 0.73, provisional in flight 197.6); client-side 461.279 ms; vad tail 512.0 ms
  reply       'Go on, I am listening.' [hold_back_01, hold_back] after_final 0.217 ms
  grounding   ['fused.confidence=0.35<0.39']
```

These are some of the fields of that final state, and the response that followed it. [docs/design.md](docs/design.md) (section 1.8) shows the whole state.

```json
{"kind": "final", "stream_id": "demo", "utt_id": 1, "t_media_start": 0.832, "t_media_end": 6.464,
 "asr": {"text": "the tribal chieftain called for the boy and presented him with 50 pieces of gold", "conf": 0.9243, "lang": "en", "tokens_per_s": 3.092},
 "text": {"partial": null, "context_k": 0, "keyphrase": "the tribal chieftain called"},
 "fused": {"label": "neutral", "confidence": 0.3467, "sentiment": "neutral", "mode": "poe_uncalibrated", "calibrated": false},
 "discrepancy": {"jsd": 0.4213, "flag": true, "direction": "tone_more_negative"},
 "timing_ms": {"vad_tail": 512.0, "frontend": 14.48, "sensevoice": 175.17, "prosody": 30.64, "text_encoder": 36.65, "heads": 0.73, "service_total": 456.69, "provisional_in_flight": 197.6},
 "degraded": []}
```

```json
{"stream_id": "demo", "utt_id": 1, "text": "Go on, I am listening.", "template_id": "hold_back_01", "style": "hold_back", "grounding": ["fused.confidence=0.35<0.39"], "t_emitted_ns": 5698137021589}
```

The text head's top class for the words is joy. SenseVoice puts most of its mass on `unknown` and most of the rest on neutral. The JSD between the text posterior and the raw SenseVoice posterior, 0.4213, is above the default flag threshold of 0.2, and the voice is less positive than the words, so the direction is `tone_more_negative`. There is no audio calibrator yet, so the mode is `poe_uncalibrated`. The product picks neutral, and the reported confidence is the text head's probability of neutral, 0.3467. That is below `low_conf` (0.39), so the selector holds back and makes no emotion claim. `en.wav` is a read sentence, so the run shows the mechanism and is not an emotion result. The final state left the service 456.69 ms after the endpoint frame arrived, 197.6 ms of which it spent waiting for a provisional that was still running. The 512 ms of silence that ends the utterance comes before that.

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

The endpoint silence sets how long the robot waits after the speaker stops before it answers. The default of 500 ms becomes 16 frames of 32 ms, a 512 ms tail (`vad_tail_ms` in [results/replay_bench_vm_long.json](results/replay_bench_vm_long.json)). `tools/pause_stats.py` measured the pauses inside speech on the available audio ([results/pause_stats_available.json](results/pause_stats_available.json)). The longest is the 448 ms pause in `en.wav`, which 400 ms (a 416 ms tail) splits into two utterances and 500 ms keeps as one. The longest pause in a proxy clip is 384 ms. [docs/decisions.md](docs/decisions.md) weighs the splits against the longer wait. Like 500 ms, 600 ms (a 608 ms tail) splits no measured clip, and every turn then waits the longer tail. The benches ran at 400 ms and 500 ms only.

## Real time

The robot should answer inside the pause after a person stops talking, and show that it is listening while they talk. Real time is therefore defined per turn and measured from the arrival of the endpoint frame, at 4 threads, with any provisional still running counted in. The silence tail comes on top and is reported apart. [docs/design.md](docs/design.md) (section 1.10) gives the reasons and assumptions. These are the container numbers from [results/measurements.md](results/measurements.md):

| target | limit | proxy clips, 400 ms | long clips, 500 ms |
|---|---|---|---|
| final state, utterances up to 5 s, p50 / p95 | 250 / 400 ms | 173.9 / 261.8 ms (n 19) | no clips |
| final state, 5 to 8 s, p50 / p95 | 350 / 550 ms | 315.0 / 315.0 ms (n 1) | 319.5 / 353.8 ms (n 3) |
| final state, 8 s to the 15 s cap, p50 / p95 | 700 / 1000 ms | no clips | 490.8 / 654.0 ms (n 9) |
| reply after the final state, p95 | 5 ms | 0.330 ms | 0.260 ms |
| gap between provisionals in the first 4 s, p50 | 1 s (within 0.1 s) | 1.020 s | 1.020 s |
| worker compute per second of audio | 0.3 s | 0.120 s | 0.123 s |
| peak RSS of service, client and monitor in one process | no limit | 1187.8 MB | 1212.1 MB |

Every target passes in the container. The container recorded an "Intel(R) Xeon(R) Processor @ 2.10GHz" with nproc 4, 16093 MB of memory, AVX-512 VNNI and AMX, no GPU, Python 3.12.13 and onnxruntime 1.31.0 ([results/bench_components_vm.json](results/bench_components_vm.json)). The model files on disk take 0.6 MB (VAD), 228.2 MB (SenseVoice), 475.5 MB (RoBERTa), and 228.2 MB more for the optional edited graph. The benches ran one at a time, and every section of [results/measurements.md](results/measurements.md) records the load average at its start and end. The demo above is a single run, which waited behind a provisional. The laptop column is pending ([docs/runbook_laptop.md](docs/runbook_laptop.md)).

## Results

All numbers are on MELD test, in percent. [RESULTS.md](RESULTS.md) has the full summary, rendered by `tools/render_results.py`, and [results/RESULTS.md](results/RESULTS.md) has the confusion matrices.

| condition | weighted F1 [95 % CI] | macro F1 |
|---|---|---|
| majority class | 31.27 [28.62, 34.31] | 9.28 |
| text only, deployed head (k = 0, frozen RoBERTa-base) | 59.62 [57.38, 62.03] | 41.22 |
| text only, same-speaker window | 59.61 [57.36, 62.08] | 40.82 |
| sentiment, text only (3 classes) | 66.24 [64.41, 68.20] | 63.33 |

| published row | weighted F1 | source |
|---|---|---|
| text-CNN, text | 55.02 | MELD paper, Table 11 |
| DialogueRNN, text | 57.03 | MELD paper, Table 11 |
| DialogueRNN, text and audio | 60.25 | MELD paper, Table 11 |
| EmoBERTa, RoBERTa-large, no context | 63.46 | EmoBERTa, Table 2 |
| SenseVoice-S, audio, no fine-tuning on MELD | 57.7 | FunAudioLLM report, Table 8 |
| DialogueRNN sentiment, text and audio | 67.56 | MELD paper, Table 13 |

The FunAudioLLM report does not list the corpora behind SenseVoice's emotion labels, so MELD may be among them and the SenseVoice row is not a zero-shot number ([PROVENANCE.md](PROVENANCE.md)). Pending: the audio-only, fused and fused-sentiment rows, whose code is ready and which need MELD audio, and the ablations, the live ASR condition, the disagreement subset and early commit, which also need code in `tools/evaluate.py`.

## Parameters

| component | parameters |
|---|---|
| SenseVoice-small int8 | 234.000 M |
| RoBERTa-base | 124.646 M |
| Silero VAD v4 | 0.156 M |
| TextContextHead (MLP 772-256-7) | 0.200 M |
| SentimentHead (LR 772-3) | 0.002 M |
| total on the inference path | 359.003 M, 5.98 % of the 6 B cap |

`tools/count_params.py` counts these from the files on disk ([results/params.md](results/params.md)). The audio calibrator and the fusion head will add their weights once MELD audio exists. Piper (15.651 M) only makes the proxy clips and is not on the inference path.

## Completed and left out

| item | state |
|---|---|
| streaming service with VAD end-pointing, provisionals, the final state and a grounded reply | done, tested |
| text-only heads, trained on dev choices and scored once on test | done ([RESULTS.md](RESULTS.md)) |
| parameter count and container measurements | done ([results/params.md](results/params.md), [results/measurements.md](results/measurements.md)) |
| one input through to both outputs | done ([results/demo_log.jsonl](results/demo_log.jsonl)) |
| reply bank of 124 lines | done, tested (`meld_rt/response/bank.yaml`) |
| audio calibrator, fusion head, discrepancy threshold from dev, and the audio and fused rows | pending: MELD audio was not available in the build environment, and the tools exist |
| ablations, live ASR condition, disagreement subset, early commit | pending: need MELD audio and code in `tools/evaluate.py` |
| laptop measurements and a live microphone session | pending, to be run on the laptop ([docs/runbook_laptop.md](docs/runbook_laptop.md)) |
| listening audit of 50 flagged clips | pending: needs MELD audio and a listener |
| vision and the three-modality extension | left out |
| an LLM reply generator | left out |
| fine-tuning RoBERTa, SenseVoice, wav2vec2 or HuBERT | left out |
| reinforcement learning | left out |
| a spoken reply | left out: Piper only makes the proxy clips |
| macOS | not measured |

[docs/design.md](docs/design.md) (section 1.13) and [docs/decisions.md](docs/decisions.md) give the reasons.

## Once MELD audio exists

Download `MELD.Raw.tar.gz` (the link is in the MELD repository README), unpack it and the `train`, `dev` and `test` tarballs inside it, then run:

```
uv run python tools/extract_audio.py --raw /data/MELD.Raw --out audio/ --split train,dev,test --jobs 2
uv run python tools/pause_stats.py --wavs "audio/dev/*.wav" --label meld_dev --bench results/replay_bench_vm.json
uv run python tools/edit_sensevoice_graph.py
uv run python tools/cache_audio_features.py --split train --audio audio/ --out cache/audio_train.npz --prefixes 2,4 --encoder-states
uv run python tools/cache_audio_features.py --split dev --audio audio/ --out cache/audio_dev.npz --prefixes 2,4 --encoder-states
uv run python tools/cache_audio_features.py --split test --audio audio/ --out cache/audio_test.npz --prefixes 2,4 --encoder-states
uv run python tools/embed_text.py --split train --out cache/text_train.npy
uv run python tools/embed_text.py --split dev --out cache/text_dev.npy
uv run python tools/embed_text.py --split test --out cache/text_test.npy
uv run python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml
uv run python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json
uv run python tools/report_text_only.py --manifest heads/manifest.json --metrics results/metrics.json --out results/01_text_only.md
uv run python tools/render_results.py
uv run python tools/count_params.py --models models/ --heads heads/ --out results/params.md
uv run python tools/render_measurements.py --out results/measurements.md
```

`extract_audio.py` writes 16 kHz mono WAV files under `audio/` and lists missing or failed clips in `audio/missing.json`. `pause_stats.py` measures the pauses inside MELD dev speech, which is the check on the endpoint default ([docs/decisions.md](docs/decisions.md)). `cache_audio_features.py` runs the service's own SenseVoice and prosody code on every clip and can resume from a checkpoint. `train_heads.py` without `--text-only` adds the audio calibrators, the fusion head, the fused sentiment head and the dev JSD threshold, and the service then runs in `stacked` mode. The text caches are not committed, so a fresh checkout builds them with `embed_text.py`. [docs/runbook_laptop.md](docs/runbook_laptop.md) describes the listening audit that follows.

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
