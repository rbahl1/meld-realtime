# meld-realtime

This is a prototype for the real-time multimodal challenge on MELD, text and audio track. It listens to a stream of 16 kHz speech, or takes typed text, cuts it into utterances, and publishes a state for each one: what the words say (a text posterior over the seven MELD emotions), what the voice says (an audio posterior and audio events), how far the two disagree and in which direction, and a fused MELD emotion with a confidence. A short reply is then chosen from that state, together with the list of state fields that licensed it. An optional mode (`--vision`) adds the face as a third input from a camera or a video file and can commit the label while the person is still talking. `--reply llm` streams a reply from a small local LLM ("Text, voice and face" below).

Everything runs locally on the CPU through onnxruntime: Silero VAD for end-pointing, SenseVoice-small for the transcript, the emotion posterior and the audio events in one pass, a frozen RoBERTa-base for the text, and small numpy heads trained on MELD. The inference path has 359.001 M parameters, and 854.286 M with the extension ([results/params.md](results/params.md)). Messages travel over ZeroMQ with a media clock, so a replayed file and a live microphone take the same path. The audio heads were trained on MELD audio from the Hugging Face mirror of MELD.Raw, and the face heads on its video. No audio or video is in the repository ([PROVENANCE.md](PROVENANCE.md)).

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

`uv sync` installs Python 3.12 (`.python-version`) and the locked dependencies, with the dev group: pytest, sherpa-onnx (for the proxy clips), praat-parselmouth (prosody) and msgpack. `uv sync --extra mic` adds sounddevice for the microphone client. ffmpeg is only needed to extract MELD audio and to read MP4 files, and libportaudio2 only for the microphone.

The three-input mode needs two more steps:

```
uv sync --extra vision
uv run python tools/fetch_models.py --dest models/ --groups core,vision,llm
```

`--extra vision` adds opencv-python-headless, which runs the two face models. `uv sync` keeps only the extras it is given, so the camera session needs `uv sync --extra mic --extra vision`. The default group of `fetch_models.py` is `core`. `vision` adds the face models, `llm` the deployed reply generator, and `--groups all` also the three other generator files that `tools/bench_generator.py` compared.

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
| `models/vision/face_detection_yunet_2023mar.onnx` | 232589 | face detection (group `vision`) |
| `models/vision/facial_expression_recognition_mobilefacenet_2022july.onnx` | 4791892 | facial expression (group `vision`) |
| `models/llm/Qwen2.5-0.5B-Instruct/onnx/model_int8.onnx` | 512096557 | streamed reply (group `llm`) |
| `models/llm/Qwen2.5-0.5B-Instruct/tokenizer.json` | 7031673 | its tokenizer, with `config.json` and `generation_config.json` (group `llm`) |

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

The three-input mode uses the same clients:

```
uv run python -m meld_rt.service --vision --reply llm --preload
uv run python -m meld_rt.clients.monitor --jsonl session.jsonl
uv run python -m meld_rt.clients.replay --video clip.mp4 --realtime --wait-ready 300
```

`replay --video` sends the audio and the frames of one MP4 file on one media clock. `mic --camera 0` adds the frames of a camera to the microphone stream. The monitor then also prints the face label, the commit and the reply as its words arrive.

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
| `--vision` | off | take video, read the face, fuse three inputs, run the commit policy |
| `--video-fps-max`, `--vision-lag-ms` | 10, 150 | frames processed per second, longest wait of a final for the vision thread |
| `--commit-policy`, `--no-commit` | `heads/commit_policy.npz` | when to commit the label (vision mode only) |
| `--reply` | `bank` | `llm` streams a generated reply as `response.delta` messages |
| `--llm-dir`, `--llm-threads` | `models/llm/Qwen2.5-0.5B-Instruct`, 2 | generator model and threads |
| `--llm-first-token-timeout-s` | 2.0 | fall back to the bank line when no word comes in time |

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

## Text, voice and face

This is the optional extension of the challenge. With `--vision` the service takes video frames next to the audio, on the same media clock. A vision thread finds the faces in each frame with YuNet and reads their expression with the FER model of the OpenCV zoo, both in OpenCV DNN. At a final, the speaker's face is the track whose mouth motion follows the audio envelope best (rule `asd`, chosen on dev). A calibrator turns that face's expressions into a face posterior, and fusion3, a logistic regression over 29 features of the text, audio and face posteriors, gives the fused label. The state gains a `vision` block, a words-against-face discrepancy and a `commit` block. At each full-window provisional (1, 2, 3 and 4 s into the segment), a commit policy trained by fitted Q-iteration decides whether to commit the fused label now (`state.commit`) or wait. With `--reply llm`, Qwen2.5-0.5B-Instruct (int8, in onnxruntime) writes the reply from the state and streams it word by word. The decoder masks the emotion words of every class the state does not license, and below `low_conf` the reply names no feeling and asks a question. Without `--vision` the core runs as before.

### Demo

```
DEMO_MP4=/path/to/dia4_utt7.mp4 ./demo_tav.sh
```

The script starts the service with `--vision --reply llm --preload`, a monitor and the replay client, streams the audio and the frames of one MP4 file at real-time pace, prints a summary per utterance and writes every message to [results/demo_tav_log.jsonl](results/demo_tav_log.jsonl). The MP4 file is not in the repository. The committed run streamed MELD test clip 4_7 (`dia4_utt7.mp4` in the test split of MELD.Raw, which [docs/runbook_laptop.md](docs/runbook_laptop.md) shows how to fetch). The clip was picked by a rule written before the choice: a test clip whose chosen face is present in at least 80 % of the frames, whose SenseVoice transcript passes the ASR gate, on which the text head is wrong on that transcript and fusion3 is right on the same transcript, the lowest dialogue id and then the lowest utterance id. `tools/pick_demo_clip.py` applies the rule and writes [results/demo_clip.json](results/demo_clip.json): 122 of the 2610 test clips meet it, and 4_7 comes first.

<!-- demo_tav: start. tools/render_demo_docs.py writes this block from results/demo_tav_log.jsonl -->
The committed log [results/demo_tav_log.jsonl](results/demo_tav_log.jsonl) came from `./demo_tav.sh` on one MELD test clip, with the vision heads, the commit policy and the deployed reply generator. `tools/render_demo_docs.py` writes this block from that log. `demo_tav.sh` ends with this summary:

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

The face was found in 24 frames of the utterance (presence 1.0, 1 track, rule `asd`). Its calibrated top class is neutral, and the top class of the raw FER posterior is sadness. The text head's top class for the words is neutral. SenseVoice puts most of its mass on `unknown`, and its top named class is anger. The words-voice JSD, 0.615, is above the dev threshold of 0.3831, so the direction `tone_more_negative` is flagged. The words-face JSD, 0.2181, is below the threshold of 0.5729 (`vision.jsd_face_flag` in [heads/manifest.json](heads/manifest.json)), so the face raises no flag. Fusion3 (`stacked_tav`) gives anger with a calibrated confidence of 0.3869. That is below `low_conf` (0.56) of the vision mode, so the reply names no feeling and asks a question. The commit policy emitted neutral at 2.016 s after the segment start, before the endpoint, and that label does not match the final's. The final state left the service 233.97 ms after the endpoint frame arrived. The reply came from `Qwen2.5-0.5B-Instruct` in 4 deltas. The first left the service 399.08 ms after the endpoint frame arrived, and the generation took 370.82 ms. The reply passes every check of `check_reply`.
<!-- demo_tav: end -->

The MELD label of this clip is anger (`meld/test_sent_emo.csv`). The words and the face read neutral. Of the three inputs, only the voice has anger as its top named class, and fusion3 gives anger. The rule asked for a clip on which the three inputs together get the label right and the words alone do not. It did not ask that the face be the input that makes the difference, and in this clip it is not. The early label (neutral) was wrong, and the provisional after the commit already read anger (0.4844 in the log).

### Accuracy on MELD test

All numbers are in percent, from [results/RESULTS_vision.md](results/RESULTS_vision.md). [RESULTS.md](RESULTS.md) has every row with its observations. The speaker rule, the calibrator, fusion3 and its thresholds were chosen on dev by `tools/train_vision.py` (`heads/manifest.json`, section `vision`).

| condition | weighted F1 [95 % CI] | macro F1 |
|---|---|---|
| face only, raw FER label | 32.22 [29.77, 34.89] | 15.92 |
| face only, calibrator B (deployed) | 34.37 [31.83, 37.32] | 12.70 |
| text and audio, stacked fusion, gold text | 60.77 [58.65, 63.00] | 39.24 |
| text, audio and face, fusion3, gold text | 61.56 [59.45, 63.79] | 40.50 |
| text and audio, stacked fusion, SenseVoice transcripts | 47.05 [44.53, 49.57] | 27.50 |
| text, audio and face, fusion3, SenseVoice transcripts | 47.96 [45.60, 50.40] | 28.82 |
| sentiment over the fusion3 features (3 classes), gold text | 69.03 [67.05, 70.88] | 66.44 |

| fusion3 minus | text input | rows | n | weighted F1 difference [95 % CI] |
|---|---|---|---|---|
| stacked fusion | gold text | all | 2610 | 0.80 [0.20, 1.48] |
| stacked fusion | SenseVoice transcripts | all | 2610 | 0.91 [0.31, 1.52] |
| stacked fusion | gold text | face-disagreement subset | 557 | 1.17 [-0.29, 2.60] |
| text head | gold text | face-disagreement subset | 557 | -0.84 [-3.71, 2.20] |

A face was found in 99.46 % of the test clips. The face alone is weak. The calibrated face scores 34.37, against 31.27 for the majority class ([RESULTS.md](RESULTS.md)), and it scores 0.00 F1 on fear, sadness and disgust. The claim rule was fixed before the run: the face adds to text and audio only if the paired 95 % interval of fusion3 minus the stacked fusion on gold text lies above zero. It does, so the face adds measured value, 0.80 points of weighted F1. Fusion3 changes 80 of the 2610 gold-text labels of the stacked fusion, 36 from wrong to right and 20 from right to wrong. On the 557 clips where the face disagrees most with the words (JSD above the dev threshold of 0.5729), no gain is shown: both intervals include zero. The four speaker rules differ by less than one point on dev (25.19 to 25.82 weighted F1 with calibrator A), although on 67.09 % of the dev clips with a face they do not all pick the same track. The sentiment head over the fusion3 features scores 69.03, against 69.17 for the two-way sentiment head.

### Real time with video

Both benches stream the same 200 MP4 files (912.832 s of audio and 7044 frames) at real-time pace through the service with `--vision`, at 4 threads ([results/measurements.md](results/measurements.md)). They are the first 200 names of the MELD.Raw test split in sorted order: 198 are clips of the MELD bench above, and 2 (`dia108_utt1`, `dia108_utt2`) are files with no row in the test CSV. One run replies with the LLM and one with the bank, which separates the cost of the generator. The n in each cell counts the finals in that bucket.

| target | limit | `--reply llm` ([json](results/replay_bench_vm_tav.json)) | `--reply bank` ([json](results/replay_bench_vm_tav_bank.json)) |
|---|---|---|---|
| final state, utterances up to 5 s, p50 / p95 | 250 / 400 ms | 178.4 / 347.7 ms (n 175) | 172.1 / 392.8 ms (n 175) |
| final state, 5 to 8 s, p50 / p95 | 350 / 550 ms | 336.2 / 536.6 ms (n 18) | 340.6 / 509.4 ms (n 18) |
| final state, 8 s to the 15 s cap, p50 / p95 | 700 / 1000 ms | 651.9 / 809.5 ms (n 13) | 576.0 / 875.6 ms (n 13) |
| first streamed word after the endpoint frame, p50 / p95 | none set | 388.85 / 855.99 ms | |
| whole reply after the final state, p50 / p95 | 5 ms p95 for the bank | 549.92 / 1333.29 ms | 0.24 / 0.68 ms |
| vision thread per processed frame, mean | none set | 30.49 ms | 29.72 ms |
| video frames dropped | none set | 0 | 0 |
| worker compute per second of audio | 0.3 s | 0.1549 s | 0.1507 s |
| peak RSS of service, client and monitor in one process | no limit | 3422.7 MB | 1644.9 MB |

Every final-state target passes in both runs. No target for the generated reply was set before the run, so its rows have no pass or fail. All 206 replies came from the generator, with no fallback to the bank. In the same run the policy committed 168 of the 206 finals before the endpoint, at 1.02 / 2.66 s (p50 / p95) after the segment start, and the committed label equals the final's for a share of 0.767 of the finals (an endpoint commit counts as equal). The laptop runs are not done ([docs/runbook_laptop.md](docs/runbook_laptop.md)).

### Streamed reply

`tools/bench_generator.py` ran each candidate file on the same 200 test states, 100 with gold text and 100 with SenseVoice transcripts ([results/generator_bench_vm.json](results/generator_bench_vm.json), [results/generator_bench_vm_1.5b.json](results/generator_bench_vm_1.5b.json)). These are the rows at 4 threads:

| file | first token p95 ms | decode passes per s | full reply p50 / p95 ms | words per reply | check pass rate |
|---|---|---|---|---|---|
| `Qwen2.5-0.5B-Instruct/model_int8.onnx` (deployed) | 100.27 | 28.47 | 378.52 / 896.2 | 7.58 | 0.97 |
| `Qwen2.5-0.5B-Instruct/model_q4.onnx` | 196.14 | 26.87 | 496.41 / 830.08 | 7.81 | 0.995 |
| `Qwen2.5-1.5B-Instruct/model_int8.onnx` | 194.54 | 13.22 | 1228.94 / 2985.06 | 9.15 | 0.97 |
| `Qwen2.5-1.5B-Instruct/model_q4.onnx` | 580.74 | 11.64 | 1330.17 / 2541.36 | 8.51 | 0.995 |

The choice rule was fixed before the bench on 200 states. It deploys the largest model with a file whose first-token p95 is at most 350.0 ms and whose decode rate is at least 15.0 passes per s at 4 threads. Among the files of one model it keeps int8 unless q4 is faster by more than 1.2x at the same reply length with a check pass rate no lower. These definitions of the decode rate and of a faster q4 were set after preliminary benches on 20 states, and the original wording picks the same file ([docs/decisions.md](docs/decisions.md), "Streamed reply"). No file of Qwen2.5-1.5B-Instruct reaches the decode rate, so the deployed file is `Qwen2.5-0.5B-Instruct/model_int8.onnx`. The 0.5B q4 file is not faster (0.80x on a reply of 9.82 tokens), although its check pass rate is higher. The checks cover length, register, emotion words and prompt echo. They do not judge whether a reply fits what was said, and no person rated the replies.

### Commit policy

At each full-window provisional (1, 2, 3 and 4 s while the segment is open) the policy commits the fused label or waits. Without a commit, the label of the endpoint is committed. The reward is one for a correct label minus c times the commit time in seconds. The policy was trained by fitted Q-iteration on the train traces (25384 ticks of 9988 clips). Its Q_wait model and the parameters of the three baselines were chosen on dev for each c, and test was read once per method and c. The deployed c is 0.05. These are the test rows at c = 0.05 ([results/commit_policy.md](results/commit_policy.md)):

| method | mean reward [95 % CI] | accuracy | weighted F1 | mean commit s | committed before the endpoint | policy minus method [95 % CI] |
|---|---|---|---|---|---|---|
| policy | 0.4484 [0.4264, 0.4732] | 51.65 | 41.81 | 1.362 | 86.74 % | |
| endpoint | 0.3295 [0.3040, 0.3562] | 53.18 | 47.96 | 4.046 | 0.00 % | 0.1189 [0.1004, 0.1402] |
| fixed time (1 s) | 0.4498 [0.4281, 0.4723] | 49.96 | 40.30 | 0.997 | 97.01 % | -0.0014 [-0.0113, 0.0088] |
| threshold (0.5) | 0.4463 [0.4236, 0.4715] | 51.19 | 41.71 | 1.312 | 88.05 % | 0.0021 [-0.0058, 0.0093] |

The claim rule was fixed before the run: RL adds measured value over a baseline at a cost c only where the paired 95 % interval of the test reward difference lies above zero. By that rule the policy beats the endpoint at every cost of the grid (0.02, 0.05, 0.1 and 0.2). It beats the tuned fixed time only at c = 0.02 (0.0124 [0.0017, 0.0238]) and the tuned threshold only at c = 0.2 (0.0060 [0.0016, 0.0104]). At the deployed c = 0.05 no gain over the fixed time or the threshold is shown. The reward charges for time, so the policy commits earlier and is right less often: at c = 0.05 it commits at a mean of 1.362 s against 4.046 s at the endpoint, with a weighted F1 of 41.81 against 47.96. Its Q_wait model, an MLP, stopped at the cap of 10 iterations before its decisions stopped changing. Charging every tick commit 0.152 s more, as the endpoint commit is charged for the computation of the final, changes none of these outcomes.

### How these results were produced

The runs, in order: `tools/extract_video_features.py` (the faces of every MELD clip, read from MELD.Raw), `tools/train_vision.py`, `tools/evaluate_vision.py`, `tools/build_commit_traces.py`, `tools/train_commit_policy.py`, `tools/bench_generator.py`, `demo_tav.sh` and `tools/replay_bench.py --video`. Each result file of these runs records the command that wrote it, except the demo log, which holds only the service's messages. The face features, the traces and the per-clip posteriors stay in `cache/`, which is not committed.

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

With `--vision --reply llm` the extension adds these:

| component | parameters |
|---|---|
| YuNet 2023mar | 0.053 M |
| FER MobileFaceNet 2022july | 1.194 M |
| VisionCalibrator B (LR 528-7) | 3,703 |
| Fusion3 (LR 29-7) | 210 |
| Sentiment3 (LR 29-3) | 90 |
| CommitPolicy (P(correct) LR and Q_wait MLP) | 272 |
| Qwen2.5-0.5B-Instruct, `model_int8.onnx` | 494.033 M |
| total with the extension | 854.286 M, 14.24 % of the 6 B cap |

`tools/count_params.py` counts these from the files on disk ([results/params.md](results/params.md)). For the quantized LLM it counts every weight tensor once at its dequantized element count and leaves out the quantization scales and zero points and the exporter's constant tables. Piper (15.651 M) only makes the proxy clips and is not on the inference path.

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
| the face as a third input: MELD video features, speaker rule, face calibrator, fusion3 and its sentiment head, chosen on dev and scored once on test | done ([results/RESULTS_vision.md](results/RESULTS_vision.md)) |
| video in the service: vision thread, three-input state, replay of MP4 files, camera client | done, tested |
| a streamed reply from a local LLM, with the bank line as fallback | done ([results/generator_bench_vm.json](results/generator_bench_vm.json)) |
| the RL extension: a commit policy by fitted Q-iteration against three baselines | done ([results/commit_policy.md](results/commit_policy.md)) |
| three inputs through to all outputs, and a replay bench with video on 200 MP4 files of the MELD.Raw test split | done ([results/demo_tav_log.jsonl](results/demo_tav_log.jsonl), [results/measurements.md](results/measurements.md)) |
| a text head trained on SenseVoice transcripts of train and dev | not done |
| fine-tuning RoBERTa on MELD text | not done |
| laptop measurements (the two benches with video included), a live microphone session and a camera session | not done, to be run on the laptop ([docs/runbook_laptop.md](docs/runbook_laptop.md)) |
| listening audit of 50 flagged clips | not done: `tools/audit_disagreement.py` draws the sample, and a person has to listen |
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

[docs/runbook_laptop.md](docs/runbook_laptop.md) lists the commands for the laptop in order: setup, the measurements, the live microphone session, the camera session and the benches with video, how to re-render [results/measurements.md](results/measurements.md), and the listening audit.

## macOS

On macOS, Homebrew takes the place of apt: `brew install portaudio` for the microphone client and `brew install ffmpeg` for MELD audio and MP4 files. Nothing was run or measured on macOS. The host block of [results/measurements.md](results/measurements.md) reads `lscpu`, `free` and `/proc`, which macOS does not have, so there the CPU model and the memory would read "unknown" and every CPU flag "no".

## Repository layout

```
meld_rt/          the package: service, protocol, state, experts, heads, selector, clients,
                  face features, commit policy, reply generator
tools/            downloads, caches, training, evaluation, benches and report renderers
configs/          heads.yaml, the training grid and threshold rules
heads/            trained heads, the commit policy and manifest.json
assets/           the RoBERTa tokenizer files
results/          every measured number, written by the tools
tests/            pytest suite
docs/             design.md, decisions.md, runbook_laptop.md
demo.sh           one utterance through the system
demo_tav.sh       one MELD clip with face, voice and words, and a streamed reply
RESULTS.md        evaluation summary
PROVENANCE.md     external files, hashes and licences
AI_USE.md         how an AI coding agent was used
```

[docs/design.md](docs/design.md) has the full layout, file by file.

## AI use

[AI_USE.md](AI_USE.md) says how an AI coding agent was used to build this.
