# Measurements

Rendered by `python tools/render_measurements.py --out results/measurements.md` from the json files under `results/`; the header comes from `lscpu`, `nproc`, `free -m`, the running interpreter and `.git`. Targets are the real-time definition of design 1.10; every measured number was written by the script named in its section.

## Container (this host)

- host: `vm`, rendered 2026-10-12T05:12:41Z, commit `60ef2c9edb28 (main)`
- lscpu: Intel(R) Xeon(R) Processor @ 2.10GHz; nproc 4; flags of interest: avx2 yes, fma yes, avx512f yes, avx512bw yes, avx512_vnni yes, amx_tile yes, amx_int8 yes, amx_bf16 yes
- free -m:

```
total        used        free      shared  buff/cache   available
Mem:           16093         784       10081          15        5559       15308
Swap:              0           0           0
```

- python 3.12.13 (Linux-6.18.44-fc-v114-x86_64-with-glibc2.39), onnxruntime 1.31.0

### Parameters (`results/params.json`, written by `tools/count_params.py`)

Generated 2026-10-12T04:58:09Z.

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

Generated 2026-10-12T04:47:39Z on `vm`; 7 timed runs after 1 warm-up, p50 / p95 in ms; load average [0.09, 0.32, 0.84] at start, [1.67, 0.7, 0.94] at the end; wall 70.4 s. Clips: 2 s (slice), 3.3 s (slice), 5 s (slice), 7.15 s (slice), 10 s (tile), 15 s (tile) from `models/sensevoice/test_wavs/en.wav`.

**SenseVoice-small int8, onnxruntime call** (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 180.5 | 258.3 | 352.2 | 491.3 | 668.8 | 981.0 |
| 2 | 100.9 | 147.7 | 218.8 | 287.0 | 413.1 | 581.1 |
| 4 | 72.2 | 102.5 | 136.0 | 187.6 | 317.9 | 479.5 |

**SenseVoice-small int8, onnxruntime call** (p95, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 211.4 | 265.4 | 409.5 | 527.2 | 681.6 | 1034.9 |
| 2 | 116.1 | 166.4 | 250.8 | 315.4 | 468.2 | 644.5 |
| 4 | 78.4 | 107.4 | 148.5 | 206.0 | 342.7 | 748.0 |

**SenseVoice total (front end + model + CTC decode)** (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 186.7 | 266.0 | 362.4 | 508.0 | 687.6 | 1028.9 |
| 2 | 104.9 | 154.5 | 229.7 | 301.3 | 432.5 | 607.9 |
| 4 | 76.5 | 110.3 | 148.8 | 206.6 | 338.6 | 516.4 |

**Front end (fbank + LFR + CMVN)** (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 4.8 | 5.3 | 7.4 | 11.4 | 14.6 | 22.3 |
| 2 | 3.3 | 5.0 | 8.0 | 10.7 | 14.4 | 21.0 |
| 4 | 3.2 | 5.7 | 7.6 | 10.9 | 16.5 | 26.5 |

**Prosody (parselmouth)** (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 5.6 | 10.6 | 13.1 | 31.0 | 23.4 | 43.5 |
| 2 | 6.5 | 11.2 | 15.1 | 18.0 | 22.9 | 34.0 |
| 4 | 7.5 | 9.4 | 12.1 | 15.7 | 25.9 | 32.3 |

**RoBERTa-base text expert by input length** (p50, ms)

| threads | 16 tok | 32 tok | 64 tok |
|---|---|---|---|
| 1 | 48.3 | 72.2 | 116.0 |
| 2 | 24.7 | 34.1 | 53.5 |
| 4 | 15.1 | 22.7 | 34.5 |

**RoBERTa-base text expert by input length** (p95, ms)

| threads | 16 tok | 32 tok | 64 tok |
|---|---|---|---|
| 1 | 55.2 | 82.3 | 134.9 |
| 2 | 29.8 | 38.3 | 54.7 |
| 4 | 16.2 | 23.0 | 46.5 |

**VAD, heads, rules, selector, session loads**

| threads | VAD per 512-sample frame (1 thread) | heads (p50 ms) | selector (p50 ms) | load: sensevoice / text / vad / heads / selector (s) |
|---|---|---|---|---|
| 1 | 0.142 / p95 0.198 | text_ctx 0.052, sentiment 0.014, fuse_poe 0.024, build_fusion_features 0.011, discrepancy 0.048 | 0.005 | 1.475 / 1.141 / 0.012 / 0.007 / 0.029 |
| 2 | 0.128 / p95 0.221 | text_ctx 0.047, sentiment 0.012, fuse_poe 0.020, build_fusion_features 0.010, discrepancy 0.041 | 0.005 | 0.989 / 0.706 / 0.008 / 0.003 / 0.02 |
| 4 | 0.138 / p95 0.203 | text_ctx 0.050, sentiment 0.013, fuse_poe 0.020, build_fusion_features 0.011, discrepancy 0.046 | 0.005 | 1.027 / 0.644 / 0.008 / 0.003 / 0.02 |

Model files on disk: `silero_vad.onnx` 0.6 MB, `sensevoice/model.int8.onnx` 228.2 MB, `sensevoice/model_enc.int8.onnx` 228.2 MB, `roberta-base-11.onnx` 475.5 MB, `piper/en_US-lessac-medium.onnx` 60.2 MB; heads: `sentiment.npz` 18 KB, `text_ctx.npz` 790 KB, `text_ctx_k0.npz` 790 KB.

Machine recorded by the bench: Intel(R) Xeon(R) Processor @ 2.10GHz, nproc 4, python 3.12.13, onnxruntime 1.31.0, numpy 2.5.4.

### Replay bench (`replay_bench_vm.json`, written by `tools/replay_bench.py`)

Generated 2026-10-12T05:10:02Z on `vm`: 20 clips from `proxy` (durations p50 2.58 s, max 5.00 s), real-time pace, 1.0 s of silence after each clip, one stream, 1 warm-up clip(s) not measured; service with 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 400 ms, provisionals 1.0 Hz over a 4.0 s window, fusion mode `poe`, heads ['sentiment.npz', 'text_ctx.npz', 'text_ctx_k0.npz']. Models loaded in 2.57 s. Load average [0.27, 0.42, 0.62] at start, [0.62, 0.47, 0.62] at the end. 20 finals, 20 responses, 51 provisionals, errors: none.

**Final state after the endpoint, by utterance length (design 1.10 targets at 4 threads)**

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | client-side p50 / p95 ms | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | prov. in flight p50 | pass |
|---|---|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 19 | 199.1 / 261.2 | 250 / 400 | 200.0 / 262.4 | 199.3 / 261.4 | 5.3 / 131.3 / 8.7 / 42.0 / 0.6 | 0.0 | pass |
| 5 to 8 s (95 % cumulative) | 1 | 367.5 / 367.5 | 350 / 550 | 368.4 / 368.4 | 367.7 / 367.7 | 11.8 / 266.5 / 17.3 / 53.5 / 0.6 | 0.0 | **FAIL** |
| 8 s to the 15 s cap (99.7 % cumulative) | 0 | no utterances in this bucket | 700 / 1000 | - | - | - | - | n/a |
| all | 20 | 201.8 / 339.4 | - | 202.6 / 343.7 | 202.0 / 339.6 | 5.3 / 133.1 / 8.8 / 42.5 / 0.6 | - | - |

**Other targets of design 1.10**

| target | measured | pass |
|---|---|---|
| reply text within 5.0 ms of the final state (p95) | p50 0.210, p95 0.430, max 0.930 ms over 20 replies | pass |
| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 1.020 s (n 24), tail-window gap p50 - s (n 0), full-to-tail switch gap p50 1.020 s, first provisional 1.020 s after speech start, 44 full + 7 tail provisionals, 14/14 utterances over 1.5 s got one, dropped 0, provisional SenseVoice p50 93.8 ms | pass |
| sustained compute below 0.3 s per second of audio | worker 0.128 s per audio s (0.179 per speech s); whole process 0.551 cpu s per audio s, cpu % sampled mean 54.0 max 271.9 | pass |
| memory | rss 1117.2 MB after load, peak 1186.8 MB during the run (service + client + monitor in one process) | - |
| worker lag | max health queue_s 0.183 s; clips without a final: none; clips split into several finals: none; degraded finals 0, empty transcripts 0 | - |

Notes: service, replay client and monitor run in one process on this host; rss and cpu include all three. service_total_ms is the service-side latency: endpoint frame arrival (t_service_recv_ns) to the final's emission. client_ms is the monitor's arrival time minus the capture time of the endpoint frame, valid because both run here. response_ms is from the endpoint frame arrival to the response's emission. proxy clips are piper tts: latencies are real, emotion labels are not a result.

### Replay bench (`replay_bench_vm_long.json`, written by `tools/replay_bench.py`)

Generated 2026-10-12T05:12:41Z on `vm`: 12 clips from `bench_clips/long` (durations p50 10.50 s, max 15.00 s), real-time pace, 1.0 s of silence after each clip, one stream, 1 warm-up clip(s) not measured; service with 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 500 ms, provisionals 1.0 Hz over a 4.0 s window, fusion mode `poe`, heads ['sentiment.npz', 'text_ctx.npz', 'text_ctx_k0.npz']. Models loaded in 2.61 s. Load average [0.93, 0.55, 0.64] at start, [0.1, 0.35, 0.55] at the end. 12 finals, 12 responses, 84 provisionals, errors: none.

**Final state after the endpoint, by utterance length (design 1.10 targets at 4 threads)**

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | client-side p50 / p95 ms | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | prov. in flight p50 | pass |
|---|---|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 0 | no utterances in this bucket | 250 / 400 | - | - | - | - | n/a |
| 5 to 8 s (95 % cumulative) | 3 | 293.4 / 298.6 | 350 / 550 | 294.2 / 299.4 | 293.5 / 298.8 | 8.9 / 209.9 / 26.8 / 30.1 / 0.7 | 0.0 | pass |
| 8 s to the 15 s cap (99.7 % cumulative) | 9 | 455.7 / 587.5 | 700 / 1000 | 456.6 / 588.8 | 455.9 / 588.0 | 18.5 / 336.4 / 44.1 / 42.2 / 0.7 | 0.0 | pass |
| all | 12 | 421.9 / 568.2 | - | 422.7 / 569.4 | 422.2 / 568.6 | 16.0 / 318.6 / 36.8 / 41.8 / 0.7 | - | - |

**Other targets of design 1.10**

| target | measured | pass |
|---|---|---|
| reply text within 5.0 ms of the final state (p95) | p50 0.190, p95 0.600, max 0.720 ms over 12 replies | pass |
| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 1.020 s (n 24), tail-window gap p50 2.020 s (n 36), full-to-tail switch gap p50 1.020 s, first provisional 1.020 s after speech start, 36 full + 48 tail provisionals, 12/12 utterances over 1.5 s got one, dropped 0, provisional SenseVoice p50 127.0 ms | pass |
| sustained compute below 0.3 s per second of audio | worker 0.113 s per audio s (0.124 per speech s); whole process 0.504 cpu s per audio s, cpu % sampled mean 49.6 max 347.1 | pass |
| memory | rss 1133.0 MB after load, peak 1235.9 MB during the run (service + client + monitor in one process) | - |
| worker lag | max health queue_s 0.444 s; clips without a final: none; clips split into several finals: none; degraded finals 0, empty transcripts 0 | - |

Notes: service, replay client and monitor run in one process on this host; rss and cpu include all three. service_total_ms is the service-side latency: endpoint frame arrival (t_service_recv_ns) to the final's emission. client_ms is the monitor's arrival time minus the capture time of the endpoint frame, valid because both run here. response_ms is from the endpoint frame arrival to the response's emission. proxy clips are piper tts: latencies are real, emotion labels are not a result.

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
