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
| `count_params.py` | 1.2 s | `wall_s` in [results/params.json](../results/params.json) |
| `bench_components.py` | 78.8 s | `wall_s` in [results/bench_components_vm.json](../results/bench_components_vm.json) |
| `replay_bench.py` on the proxy clips | 2.98 s to load the models, then 73.504 s to stream 73.536 s of audio | `load_s` and `sent` in [results/replay_bench_vm.json](../results/replay_bench_vm.json) |
| `bench_components.py --build-only` | not timed | |
| `replay_bench.py` on the long clips | 2.93 s to load the models, then 138.4 s to stream 138.432 s of audio | `load_s` and `sent` in [results/replay_bench_vm_long.json](../results/replay_bench_vm_long.json) |

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

The JSONL file stays outside the repository because it holds what was said. When the session ends, the microphone client prints how many frames it sent and how many input overflows it saw. Write that line and what you observed (ex. the device used, utterances that were split or merged, replies that did not fit, how long the answers felt) in `results/laptop_notes.md`. The renderer copies that file verbatim under the laptop heading.

## 4. Re-render the measurements file

```
uv run python tools/render_measurements.py --out results/measurements.md
```

The renderer reads every `results/bench_components_*.json` and `results/replay_bench_*.json`. Files whose recorded host is the build container (`vm`, set by `--container-host`) go under the container heading. Files from any other host replace the "not yet measured" table under the laptop heading, followed by `results/laptop_notes.md`. The first block of the file describes the machine that rendered it. Commit the new JSON files, `results/laptop_notes.md`, `results/params.md`, `results/params.json` and `results/measurements.md` together.

## 5. Disagreement audit, once MELD audio exists

First run the steps in [README.md](../README.md) under "Once MELD audio exists". None of them has been timed. Then:

```
uv run python tools/audit_disagreement.py sample --n 50
```

This scores every test clip as the service would, draws 50 of the clips that the discrepancy rule flags (seed 0), and writes `cache/audit_disagreement.csv`. The file holds MELD text, so it stays in `cache/`, which is not committed. Each row gives the key, the WAV path, the speaker, the utterance, the gold label, the text and audio labels, the JSD, the threshold, the direction and the laughter probability, followed by three empty columns.

For each row, play the clip (ex. `ffplay -nodisp -autoexit <wav>`) and fill in:

- `right_branch`: `text` when the text label fits what you hear and the audio label does not, `audio` for the reverse, `both` or `neither`.
- `laughter_source`: `speaker` when the person on the clip laughs, `audience` when the laugh track does, `none` when nobody laughs.
- `note`: anything else, optional.

Then count the marks:

```
uv run python tools/audit_disagreement.py tally
```

This writes results/disagreement_audit.md with counts only, by direction, and the threshold that was used. Read [docs/design.md](design.md) (section 1.6) on the threshold before drawing the sample: the dev threshold and the service do not use the same audio posterior yet, and `--threshold` sets one by hand.
