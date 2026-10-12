# Measurements

Rendered by `python tools/render_measurements.py --out results/measurements.md` from the json files under `results/`; the header comes from `lscpu`, `nproc`, `free -m`, the running interpreter and `.git`. Targets are the real-time definition of design 1.10; every measured number was written by the script named in its section.

## Container (this host)

- host: `vm`, rendered 2026-10-12T15:15:32Z, commit `e75af6c6210f (main)`
- lscpu: Intel(R) Xeon(R) Processor @ 2.10GHz; nproc 4; flags of interest: avx2 yes, fma yes, avx512f yes, avx512bw yes, avx512_vnni yes, amx_tile yes, amx_int8 yes, amx_bf16 yes
- free -m:

```
total        used        free      shared  buff/cache   available
Mem:           16093         766       13328          15        2255       15327
Swap:              0           0           0
```

- python 3.12.13 (Linux-6.18.44-fc-v114-x86_64-with-glibc2.39), onnxruntime 1.31.0

### Parameters (`results/params.json`, written by `tools/count_params.py`)

Generated 2026-10-12T15:15:32Z.

| Component | File | Params | Trained | Note |
|---|---|---|---|---|
| Silero VAD v4 | `silero_vad.onnx` | 0.156 M | frozen | 40 initializers |
| SenseVoice-small int8 | `sensevoice/model.int8.onnx` | 234.000 M | frozen | 1479 initializers, 233.103 M int8 elements counted dequantized |
| RoBERTa-base encoder | `roberta-base-11.onnx` | 124.646 M | frozen | 283 initializers |
| TextContextHead | `heads/text_ctx.npz` | 0.200 M | trained | mlp 772-256-7, k=0, variant full_dialogue; standardizer 1544 elements not counted |
| AudioCalibrator | `heads/audio_cal.npz` | - | trained | pending (needs MELD audio) |
| Fusion | `heads/fusion.npz` | - | trained | pending (needs MELD audio) |
| SentimentHead | `heads/sentiment.npz` | 0.002 M | trained | lr 772-3, k=0, variant text_only; standardizer 1544 elements not counted |
| Prosody, discrepancy, selector | - | 0.000 M | none | no learned parameters |
| **Total on the inference path** | | **359.003 M** | | 5.98 % of the 6 B cap; pending: AudioCalibrator, Fusion |
| Piper en_US-lessac-medium | `piper/en_US-lessac-medium/en_US-lessac-medium.onnx` | 15.651 M | frozen | 396 initializers; optional, not on the measured path |
| Total with Piper enabled | | 374.654 M | | |

### Component bench (`bench_components_vm.json`, written by `tools/bench_components.py`)

Generated 2026-10-12T15:10:03Z on `vm`; 7 timed runs after 1 warm-up, p50 / p95 in ms; load average [0.52, 0.53, 0.51] at start, [1.81, 0.87, 0.63] at the end; wall 78.8 s. Clips: 2 s (slice), 3.3 s (slice), 5 s (slice), 7.15 s (slice), 10 s (tile), 15 s (tile) from `models/sensevoice/test_wavs/en.wav`.

**SenseVoice-small int8, onnxruntime call** (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 179.7 | 264.5 | 430.5 | 552.5 | 778.8 | 1150.5 |
| 2 | 118.5 | 172.5 | 248.2 | 343.2 | 482.3 | 648.6 |
| 4 | 83.3 | 117.3 | 184.7 | 232.3 | 278.4 | 420.9 |

**SenseVoice-small int8, onnxruntime call** (p95, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 183.3 | 286.6 | 466.3 | 582.5 | 814.1 | 1183.9 |
| 2 | 120.8 | 180.9 | 272.7 | 388.8 | 511.7 | 706.1 |
| 4 | 158.5 | 145.7 | 216.4 | 257.6 | 312.1 | 501.0 |

**SenseVoice total (front end + model + CTC decode)** (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 184.3 | 272.4 | 444.0 | 567.9 | 806.6 | 1182.9 |
| 2 | 123.1 | 180.6 | 264.4 | 358.5 | 506.6 | 688.1 |
| 4 | 87.9 | 124.8 | 197.1 | 251.5 | 302.8 | 451.1 |

**Front end (fbank + LFR + CMVN)** (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 3.6 | 5.5 | 8.6 | 11.5 | 17.1 | 24.8 |
| 2 | 3.8 | 5.8 | 9.0 | 11.6 | 17.3 | 25.0 |
| 4 | 3.6 | 5.8 | 8.5 | 11.8 | 15.9 | 23.7 |

**Prosody (parselmouth)** (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 7.7 | 10.5 | 18.1 | 21.6 | 35.8 | 34.5 |
| 2 | 6.3 | 10.9 | 14.4 | 20.0 | 27.0 | 34.7 |
| 4 | 10.8 | 11.3 | 14.7 | 19.1 | 28.2 | 39.4 |

**RoBERTa-base text expert by input length** (p50, ms)

| threads | 16 tok | 32 tok | 64 tok |
|---|---|---|---|
| 1 | 51.1 | 68.3 | 111.5 |
| 2 | 27.4 | 45.7 | 64.2 |
| 4 | 16.9 | 25.3 | 41.4 |

**RoBERTa-base text expert by input length** (p95, ms)

| threads | 16 tok | 32 tok | 64 tok |
|---|---|---|---|
| 1 | 53.9 | 69.1 | 136.9 |
| 2 | 28.3 | 63.9 | 151.5 |
| 4 | 23.0 | 29.0 | 43.4 |

**VAD, heads, rules, selector, session loads**

| threads | VAD per 512-sample frame (1 thread) | heads (p50 ms) | selector (p50 ms) | load: sensevoice / text / vad / heads / selector (s) |
|---|---|---|---|---|
| 1 | 0.176 / p95 0.423 | text_ctx 0.056, sentiment 0.014, fuse_poe 0.025, build_fusion_features 0.012, discrepancy 0.053 | 0.005 | 1.425 / 1.771 / 0.011 / 0.006 / 0.027 |
| 2 | 0.157 / p95 0.212 | text_ctx 0.057, sentiment 0.014, fuse_poe 0.025, build_fusion_features 0.012, discrepancy 0.055 | 0.005 | 1.246 / 0.79 / 0.009 / 0.003 / 0.022 |
| 4 | 0.316 / p95 0.378 | text_ctx 0.088, sentiment 0.024, fuse_poe 0.042, build_fusion_features 0.020, discrepancy 0.095 | 0.009 | 1.234 / 0.791 / 0.009 / 0.003 / 0.023 |

Model files on disk: `silero_vad.onnx` 0.6 MB, `sensevoice/model.int8.onnx` 228.2 MB, `sensevoice/model_enc.int8.onnx` 228.2 MB, `roberta-base-11.onnx` 475.5 MB, `piper/en_US-lessac-medium.onnx` 60.2 MB; heads: `sentiment.npz` 18 KB, `text_ctx.npz` 790 KB, `text_ctx_k0.npz` 790 KB.

Machine recorded by the bench: Intel(R) Xeon(R) Processor @ 2.10GHz, nproc 4, python 3.12.13, onnxruntime 1.31.0, numpy 2.5.4.

### Replay bench (`replay_bench_vm.json`, written by `tools/replay_bench.py`)

Generated 2026-10-12T15:12:46Z on `vm`: 20 clips from `proxy` (durations p50 2.58 s, max 5.00 s), real-time pace, 1.0 s of silence after each clip, one stream, 1 warm-up clip(s) not measured; service with 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 400 ms, provisionals 1.0 Hz over a 4.0 s window, fusion mode `poe`, heads ['sentiment.npz', 'text_ctx.npz', 'text_ctx_k0.npz']. Models loaded in 2.98 s. Load average [1.6, 0.85, 0.63] at start, [1.03, 0.85, 0.65] at the end. 20 finals, 20 responses, 51 provisionals, errors: none.

**Final state after the endpoint, by utterance length (design 1.10 targets at 4 threads)**

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | client-side p50 / p95 ms | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | prov. in flight p50 | pass |
|---|---|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 19 | 173.9 / 261.8 | 250 / 400 | 174.4 / 262.7 | 174.2 / 262.0 | 5.6 / 112.1 / 8.0 / 44.2 / 0.7 | 0.0 | pass |
| 5 to 8 s (95 % cumulative) | 1 | 315.0 / 315.0 | 350 / 550 | 316.2 / 316.2 | 315.1 / 315.1 | 10.0 / 219.5 / 16.7 / 66.2 / 0.9 | 0.0 | pass |
| 8 s to the 15 s cap (99.7 % cumulative) | 0 | no utterances in this bucket | 700 / 1000 | - | - | - | - | n/a |
| all | 20 | 182.1 / 269.9 | - | 184.2 / 270.8 | 182.3 / 270.2 | 5.6 / 115.4 / 8.0 / 44.7 / 0.7 | - | - |

**Other targets of design 1.10**

| target | measured | pass |
|---|---|---|
| reply text within 5.0 ms of the final state (p95) | p50 0.190, p95 0.330, max 0.420 ms over 20 replies | pass |
| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 1.020 s (n 24), tail-window gap p50 - s (n 0), full-to-tail switch gap p50 1.020 s, first provisional 1.020 s after speech start, 44 full + 7 tail provisionals, 14/14 utterances over 1.5 s got one, dropped 0, provisional SenseVoice p50 84.1 ms | pass |
| sustained compute below 0.3 s per second of audio | worker 0.120 s per audio s (0.167 per speech s); whole process 0.565 cpu s per audio s, cpu % sampled mean 55.4 max 250.3 | pass |
| memory | rss 1117.6 MB after load, peak 1187.8 MB during the run (service + client + monitor in one process) | - |
| worker lag | max health queue_s 0.169 s; degraded finals 0, empty transcripts 0 | - |
| clip attribution (t_media_end within [media_start, media_end + 0.25 s]) | clips without a final: none; clips split into several finals: none; finals outside every clip span: none | - |

Notes: service, replay client and monitor run in one process on this host; rss and cpu include all three. service_total_ms is the service-side latency: endpoint frame arrival (t_service_recv_ns) to the final's emission. client_ms is the monitor's arrival time minus the capture time of the endpoint frame, valid because both run here. response_ms is from the endpoint frame arrival to the response's emission. proxy clips are piper tts: latencies are real, emotion labels are not a result. a final is attributed to the clip whose media span holds its t_media_end (the time the speech stopped), 0.25 s past the clip end allowed; the segment start carries the vad pre-roll and is not used.

### Replay bench (`replay_bench_vm_long.json`, written by `tools/replay_bench.py`)

Generated 2026-10-12T15:15:26Z on `vm`: 12 clips from `bench_clips/long` (durations p50 10.50 s, max 15.00 s), real-time pace, 1.0 s of silence after each clip, one stream, 1 warm-up clip(s) not measured; service with 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 500 ms, provisionals 1.0 Hz over a 4.0 s window, fusion mode `poe`, heads ['sentiment.npz', 'text_ctx.npz', 'text_ctx_k0.npz']. Models loaded in 2.93 s. Load average [0.8, 0.81, 0.64] at start, [0.73, 0.72, 0.62] at the end. 12 finals, 12 responses, 84 provisionals, errors: none.

**Final state after the endpoint, by utterance length (design 1.10 targets at 4 threads)**

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | client-side p50 / p95 ms | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | prov. in flight p50 | pass |
|---|---|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 0 | no utterances in this bucket | 250 / 400 | - | - | - | - | n/a |
| 5 to 8 s (95 % cumulative) | 3 | 319.5 / 353.8 | 350 / 550 | 320.7 / 354.8 | 319.7 / 354.0 | 10.9 / 245.3 / 24.5 / 40.6 / 0.8 | 0.0 | pass |
| 8 s to the 15 s cap (99.7 % cumulative) | 9 | 490.8 / 654.0 | 700 / 1000 | 491.8 / 655.5 | 491.0 / 654.2 | 21.2 / 373.5 / 45.2 / 47.0 / 0.8 | 0.0 | pass |
| all | 12 | 459.8 / 646.2 | - | 460.8 / 647.9 | 460.0 / 646.4 | 20.1 / 352.4 / 42.4 / 45.6 / 0.8 | - | - |

**Other targets of design 1.10**

| target | measured | pass |
|---|---|---|
| reply text within 5.0 ms of the final state (p95) | p50 0.200, p95 0.260, max 0.280 ms over 12 replies | pass |
| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 1.020 s (n 24), tail-window gap p50 2.020 s (n 36), full-to-tail switch gap p50 1.020 s, first provisional 1.020 s after speech start, 36 full + 48 tail provisionals, 12/12 utterances over 1.5 s got one, dropped 0, provisional SenseVoice p50 136.0 ms | pass |
| sustained compute below 0.3 s per second of audio | worker 0.123 s per audio s (0.135 per speech s); whole process 0.563 cpu s per audio s, cpu % sampled mean 55.3 max 359.8 | pass |
| memory | rss 1108.3 MB after load, peak 1212.1 MB during the run (service + client + monitor in one process) | - |
| worker lag | max health queue_s 0.385 s; degraded finals 0, empty transcripts 0 | - |
| clip attribution (t_media_end within [media_start, media_end + 0.25 s]) | clips without a final: none; clips split into several finals: none; finals outside every clip span: none | - |

Notes: service, replay client and monitor run in one process on this host; rss and cpu include all three. service_total_ms is the service-side latency: endpoint frame arrival (t_service_recv_ns) to the final's emission. client_ms is the monitor's arrival time minus the capture time of the endpoint frame, valid because both run here. response_ms is from the endpoint frame arrival to the response's emission. proxy clips are piper tts: latencies are real, emotion labels are not a result. a final is attributed to the clip whose media span holds its t_media_end (the time the speech stopped), 0.25 s past the clip end allowed; the segment start carries the vad pre-roll and is not used.

## Laptop (user-run)

Every cell below stays `[laptop: not yet measured]` until the user runs the same scripts on the laptop and pastes the block they print (design 1.12). Nothing here is typed from memory.

Commands:

```
python tools/count_params.py --models models/ --heads heads/ --out results/params.md
python tools/bench_components.py --threads 1,2,4 --clips bench_clips/ --out results/bench_components_$(hostname).json
python tools/replay_bench.py --clips "proxy/*.wav" --n 20 --realtime --out results/replay_bench_$(hostname).json
python tools/render_measurements.py --out results/measurements.md
```

| item | laptop |
|---|---|
| lscpu model name, flags (avx512_vnni, amx) | [laptop: not yet measured] |
| nproc, free -m, python, onnxruntime | [laptop: not yet measured] |
| parameters on the inference path | [laptop: not yet measured] |
| SenseVoice p50 ms at 1 thread(s), 2 / 3.3 / 5 / 7.15 / 10 / 15 s | [laptop: not yet measured] |
| SenseVoice p50 ms at 2 thread(s), 2 / 3.3 / 5 / 7.15 / 10 / 15 s | [laptop: not yet measured] |
| SenseVoice p50 ms at 4 thread(s), 2 / 3.3 / 5 / 7.15 / 10 / 15 s | [laptop: not yet measured] |
| front end p50 ms, 2 / 3.3 / 5 / 7.15 / 10 / 15 s | [laptop: not yet measured] |
| RoBERTa p50 ms at 16 / 32 / 64 tokens, 1 / 2 / 4 threads | [laptop: not yet measured] |
| VAD per frame, heads, selector, session loads | [laptop: not yet measured] |
| final after endpoint, up to 5 s: p50 / p95 ms and pass | [laptop: not yet measured] |
| final after endpoint, 5 to 8 s: p50 / p95 ms and pass | [laptop: not yet measured] |
| final after endpoint, 8 s to the 15 s cap: p50 / p95 ms and pass | [laptop: not yet measured] |
| reply after final p95 ms | [laptop: not yet measured] |
| provisional cadence, dropped provisionals | [laptop: not yet measured] |
| compute s per audio s, cpu %, peak rss | [laptop: not yet measured] |
| 3-minute mic.py session (notes) | [laptop: not yet measured] |
