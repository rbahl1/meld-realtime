# Laptop runbook

These are the steps to run on the laptop, in order. Each step says how long it took in the build container, where that was recorded. The laptop will differ, and a step that was never timed says so. Run everything from the repository root.

## 1. Setup

```
sudo apt install curl ffmpeg libportaudio2
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra mic
uv run python tools/fetch_models.py --dest models/
uv run python tools/fetch_meld_csv.py
uv run python tools/make_proxy_clips.py
uv run pytest -q
```

None of these steps was timed. The downloads depend on the network, and [README.md](../README.md) lists the file sizes. `fetch_models.py` rewrites [results/provenance_hashes.json](../results/provenance_hashes.json) with the hashes it computed. On a correct download the file comes out the same, apart from its time stamp. `pytest` should pass. Its tests skip themselves when models or data are missing.

## 2. Measurements

Close other heavy programs and keep the laptop on mains power. Then run, one at a time:

```
uv run python tools/count_params.py --models models/ --heads heads/ --out results/params.md
uv run python tools/bench_components.py --threads 1,2,4 --clips bench_clips/ --out results/bench_components_$(hostname).json
uv run python tools/replay_bench.py --clips "proxy/*.wav" --n 20 --realtime --out results/replay_bench_$(hostname).json
uv run python tools/bench_components.py --build-only --trim-silence --durations 6,9,12,15 --clips bench_clips/long
uv run python tools/replay_bench.py --clips "bench_clips/long/*.wav" --n 0 --repeat 3 --realtime --endpoint-silence-ms 500 --out results/replay_bench_$(hostname)_long.json
```

| step | container time | where it was recorded |
|---|---|---|
| `count_params.py` | 1.15 s | `wall_s` in [results/params.json](../results/params.json) |
| `bench_components.py` | 78.8 s | `wall_s` in [results/bench_components_vm.json](../results/bench_components_vm.json) |
| `replay_bench.py` on the proxy clips | 2.67 s to load the models, then 73.504 s to stream 73.536 s of audio | `load_s` and `sent` in [results/replay_bench_vm.json](../results/replay_bench_vm.json) |
| `bench_components.py --build-only` | not timed | |
| `replay_bench.py` on the long clips | 2.56 s to load the models, then 138.4 s to stream 138.432 s of audio | `load_s` and `sent` in [results/replay_bench_vm_long.json](../results/replay_bench_vm_long.json) |

The replay benches stream in real time, so they cannot finish faster than their audio. `count_params.py` should reproduce the same counts, because it reads the same files. A different total means a different model file, and the hashes in [results/provenance_hashes.json](../results/provenance_hashes.json) will show which one.

Nothing needs to be pasted. Each bench writes a JSON file named after the laptop's host name, and its own record of the machine (CPU model, flags, memory, Python and onnxruntime versions) goes in the same file.

## 3. Live microphone session

Three terminals, about three minutes of talking:

```
uv run python -m meld_rt.service --preload
uv run python -m meld_rt.clients.monitor --jsonl ~/mic_session.jsonl
uv run python -m meld_rt.clients.mic --list-devices
uv run python -m meld_rt.clients.mic --device N --seconds 180
```

The JSONL file stays outside the repository because it holds what was said. The service uses the default endpoint silence of 500 ms (a 512 ms tail), so a pause shorter than 512 ms stays inside one utterance. When the session ends, the microphone client prints how many frames it sent and how many input overflows it saw. Write that line and what you observed (ex. the device used, utterances that were split or merged, replies that did not fit, how long the answers felt) in `results/laptop_notes.md`. The renderer copies that file verbatim under the laptop heading.

## 4. Re-render the measurements file

```
uv run python tools/render_measurements.py --out results/measurements.md
```

The renderer reads every `results/bench_components_*.json`, `results/replay_bench_*.json` and `results/generator_bench_*.json`. Files whose recorded host is the build container (`vm`, set by `--container-host`) go under the container heading. Files from any other host replace the "not yet measured" table under the laptop heading, followed by `results/laptop_notes.md`. The first block of the file describes the machine that rendered it. Commit the new JSON files, `results/laptop_notes.md`, `results/params.md`, `results/params.json` and `results/measurements.md` together.

## 5. Disagreement audit

The MELD audio steps are done in the build container: the WAV extraction, the feature caches, the audio heads and every audio row ([README.md](../README.md), "How the audio results were produced"). The audit is the step that needs a person. It reads `audio/test/`, `cache/audio_test.npz` and `cache/text_test.npy`, and none of them is committed, so the laptop rebuilds them for the test split first:

```
curl -L -o MELD.Raw.tar.gz https://huggingface.co/datasets/declare-lab/MELD/resolve/9abc51ee7903424ffb971297608aa6d3d0de3bfa/MELD.Raw.tar.gz
tar -xzf MELD.Raw.tar.gz MELD.Raw/test.tar.gz
tar -xzf MELD.Raw/test.tar.gz -C MELD.Raw
uv run python tools/extract_audio.py --raw MELD.Raw --out audio/ --split test --jobs 4
uv run python tools/edit_sensevoice_graph.py
uv run python tools/cache_audio_features.py --split test --audio audio/ --out cache/audio_test.npz --prefixes 2,4 --encoder-states --threads 4
uv run python tools/embed_text.py --split test --out cache/text_test.npy
uv run python tools/audit_disagreement.py sample --n 50
```

None of these steps was timed on the laptop, and their container times are not in a committed file. The archive is 10878146150 bytes (`x-linked-size` in [results/meld_raw_source.json](../results/meld_raw_source.json)).

`audit_disagreement.py sample` scores every test clip, draws 50 of the clips that the discrepancy rule flags (seed 0) at the dev threshold from `heads/manifest.json`, and writes `cache/audit_disagreement.csv`. The file holds MELD text, so it stays in `cache/`, which is not committed. Each row gives the key, the WAV path, the speaker, the utterance, the gold label, the text and audio labels, the JSD, the threshold, the direction and the laughter probability, followed by three empty columns.

For each row, play the clip (ex. `ffplay -nodisp -autoexit <wav>`) and fill in:

- `right_branch`: `text` when the text label fits what you hear and the audio label does not, `audio` for the reverse, `both` or `neither`.
- `laughter_source`: `speaker` when the person on the clip laughs, `audience` when the laugh track does, `none` when nobody laughs.
- `note`: anything else, optional.

Then count the marks:

```
uv run python tools/audit_disagreement.py tally
```

This writes results/disagreement_audit.md with counts only, by direction, and the threshold that was used. `--threshold` sets the threshold by hand.

## 6. Camera session and benches with video

These steps are for the optional three-input mode. None of them has run on the laptop. The setup adds OpenCV and the face and reply models to the core. `uv sync` keeps only the extras it is given, so both extras go in one command:

```
uv sync --extra mic --extra vision
uv run python tools/fetch_models.py --dest models/ --groups core,vision,llm
uv run pytest -q
```

[README.md](../README.md) lists the sizes of the new files. These steps were not timed.

### Camera session

Three terminals, about three minutes of talking in front of the camera:

```
uv run python -m meld_rt.service --vision --reply llm --preload
uv run python -m meld_rt.clients.monitor --jsonl ~/camera_session.jsonl
uv run python -m meld_rt.clients.mic --device N --camera 0 --fps 10 --seconds 180
```

`--camera 0` is the first camera OpenCV finds, and another index picks another one. The monitor prints the face label and its presence, the commit and the reply as its words arrive. The log holds the states and the replies, not the frames. It stays outside the repository because it holds what was said. The camera client stamps each frame with its capture time against the first audio sample, so a frame's media time is late by the camera's own delay, which was not measured. Write what you observed in `results/laptop_notes.md`, as in step 3 (ex. the camera used, whether the face label followed your face, whether a commit came before you stopped talking, how long the reply took to start).

### Benches with video

The benches stream MELD test clips as MP4 files, which are not in the repository. Step 5 downloads `MELD.Raw.tar.gz` and unpacks its test split into `MELD.Raw/`. If step 5 has not run, run its first three commands. Then, one at a time and with no other heavy process running:

```
TEST_MP4=$(find MELD.Raw -type d -name output_repeated_splits_test | head -1)
uv run python tools/replay_bench.py --video --clips "$TEST_MP4/dia*.mp4" --n 200 --realtime --reply llm --llm-dir models/llm/Qwen2.5-0.5B-Instruct --out results/replay_bench_$(hostname)_tav.json
uv run python tools/replay_bench.py --video --clips "$TEST_MP4/dia*.mp4" --n 200 --realtime --reply bank --out results/replay_bench_$(hostname)_tav_bank.json
```

| step | container time | where it was recorded |
|---|---|---|
| `replay_bench.py --video --reply llm` | 9.58 s to load the models, then 912.8 s to stream 912.832 s of audio and 7044 frames | `load_s` and `sent` in [results/replay_bench_vm_tav.json](../results/replay_bench_vm_tav.json) |
| `replay_bench.py --video --reply bank` | 2.75 s to load the models, then 912.8 s to stream 912.832 s of audio and 7044 frames | `load_s` and `sent` in [results/replay_bench_vm_tav_bank.json](../results/replay_bench_vm_tav_bank.json) |

`--n 200` takes the first 200 names in sorted order, as the container run did. Then re-render with the command of step 4. The renderer puts the video benches of any host but the container under the laptop heading.

The three-input demo of [README.md](../README.md) streams test clip 4_7 from the same folder. `DEMO_LOG` writes the log to another path, so the committed log stays as it is:

```
DEMO_MP4=$TEST_MP4/dia4_utt7.mp4 DEMO_LOG=$HOME/demo_tav_laptop.jsonl ./demo_tav.sh
```
