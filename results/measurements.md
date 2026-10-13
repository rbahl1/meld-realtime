# Measurements

Rendered by `python tools/render_measurements.py --out results/measurements.md` from the JSON files under `results/`. The first block describes the machine that rendered this file, from `lscpu`, `nproc`, `free -m`, the running interpreter and `.git`. Each bench section names the host that ran it and the machine it recorded. The targets are the real-time definition of design 1.10 (docs/design.md). Every measured number was written by the script named in its section.

## Host that rendered this file

- host `vm`, rendered 2026-10-13T04:21:09Z, commit `710402915647 (extension)`
- CPU (lscpu): Intel(R) Xeon(R) Processor @ 2.10GHz
- nproc: 4
- flags of interest: avx2 yes, fma yes, avx512f yes, avx512bw yes, avx512_vnni yes, amx_tile yes, amx_int8 yes, amx_bf16 yes
- python 3.12.13 (Linux-6.18.44-fc-v114-x86_64-with-glibc2.39), onnxruntime 1.31.0
- free -m:

```
total        used        free      shared  buff/cache   available
Mem:           16093        1268        6553          14        8613       14825
Swap:              0           0           0
```

## Container (host `vm`)

### Parameters (`results/params.json`, written by `tools/count_params.py`)

Generated 2026-10-13T04:21:04Z.

| Component | File | Params | Trained | Note |
|---|---|---|---|---|
| Silero VAD v4 | `silero_vad.onnx` | 0.156 M | frozen | 40 initializers |
| SenseVoice-small int8 | `sensevoice/model.int8.onnx` | 234.000 M | frozen | 1479 initializers, 233.103 M int8 elements counted dequantized |
| RoBERTa-base encoder | `roberta-base-11.onnx` | 124.646 M | frozen | 283 initializers |
| TextContextHead | `heads/text_ctx.npz` | 0.200 M | trained | mlp 772-256-7, k=0, variant full_dialogue; standardizer 1544 elements not counted |
| AudioCalibrator | `heads/audio_cal.npz` | 0.000 M | trained | lr 24-7, variant A; standardizer 48 elements not counted |
| Fusion | `heads/fusion.npz` | 0.000 M | trained | lr 19-7; standardizer 38 elements not counted |
| SentimentHead | `heads/sentiment.npz` | 0.000 M | trained | lr 19-3, variant fused; standardizer 38 elements not counted |
| Prosody, discrepancy, selector | - | 0.000 M | none | no learned parameters |
| **Total on the inference path** | | **359.001 M** | | 5.98 % of the 6 B cap |
| Piper en_US-lessac-medium | `piper/en_US-lessac-medium/en_US-lessac-medium.onnx` | 15.651 M | frozen | 396 initializers, used by tools/make_proxy_clips.py only, not on the inference path |
| Total including Piper | | 374.652 M | | |

### Component bench (`bench_components_vm.json`, written by `tools/bench_components.py`)

Generated 2026-10-12T15:10:03Z on `vm` by `python tools/bench_components.py --threads 1,2,4 --clips bench_clips/ --out results/bench_components_vm.json`. Each cell is the p50 or p95 in ms of 7 timed runs after 1 warm-up run. The load average was [0.52, 0.53, 0.51] at the start and [1.81, 0.87, 0.63] at the end. The bench took 78.8 s. Clips: 2 s (slice), 3.3 s (slice), 5 s (slice), 7.15 s (slice), 10 s (tile), 15 s (tile) from `models/sensevoice/test_wavs/en.wav`.

#### SenseVoice-small int8, onnxruntime call (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 179.7 | 264.5 | 430.5 | 552.5 | 778.8 | 1150.5 |
| 2 | 118.5 | 172.5 | 248.2 | 343.2 | 482.3 | 648.6 |
| 4 | 83.3 | 117.3 | 184.7 | 232.3 | 278.4 | 420.9 |

#### SenseVoice-small int8, onnxruntime call (p95, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 183.3 | 286.6 | 466.3 | 582.5 | 814.1 | 1183.9 |
| 2 | 120.8 | 180.9 | 272.7 | 388.8 | 511.7 | 706.1 |
| 4 | 158.5 | 145.7 | 216.4 | 257.6 | 312.1 | 501.0 |

#### SenseVoice total (front end + model + CTC decode) (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 184.3 | 272.4 | 444.0 | 567.9 | 806.6 | 1182.9 |
| 2 | 123.1 | 180.6 | 264.4 | 358.5 | 506.6 | 688.1 |
| 4 | 87.9 | 124.8 | 197.1 | 251.5 | 302.8 | 451.1 |

#### Front end (fbank + LFR + CMVN) (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 3.6 | 5.5 | 8.6 | 11.5 | 17.1 | 24.8 |
| 2 | 3.8 | 5.8 | 9.0 | 11.6 | 17.3 | 25.0 |
| 4 | 3.6 | 5.8 | 8.5 | 11.8 | 15.9 | 23.7 |

#### Prosody (parselmouth) (p50, ms)

| threads | 2 s | 3.3 s | 5 s | 7.15 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| 1 | 7.7 | 10.5 | 18.1 | 21.6 | 35.8 | 34.5 |
| 2 | 6.3 | 10.9 | 14.4 | 20.0 | 27.0 | 34.7 |
| 4 | 10.8 | 11.3 | 14.7 | 19.1 | 28.2 | 39.4 |

#### RoBERTa-base text expert by input length (p50, ms)

| threads | 16 tok | 32 tok | 64 tok |
|---|---|---|---|
| 1 | 51.1 | 68.3 | 111.5 |
| 2 | 27.4 | 45.7 | 64.2 |
| 4 | 16.9 | 25.3 | 41.4 |

#### RoBERTa-base text expert by input length (p95, ms)

| threads | 16 tok | 32 tok | 64 tok |
|---|---|---|---|
| 1 | 53.9 | 69.1 | 136.9 |
| 2 | 28.3 | 63.9 | 151.5 |
| 4 | 23.0 | 29.0 | 43.4 |

#### VAD, heads, rules, selector, session loads

| threads | VAD per 512-sample frame (1 thread) | heads (p50 ms) | selector (p50 ms) | load: sensevoice / text / vad / heads / selector (s) |
|---|---|---|---|---|
| 1 | 0.176 / p95 0.423 | text_ctx 0.056, sentiment 0.014, fuse_poe 0.025, build_fusion_features 0.012, discrepancy 0.053 | 0.005 | 1.425 / 1.771 / 0.011 / 0.006 / 0.027 |
| 2 | 0.157 / p95 0.212 | text_ctx 0.057, sentiment 0.014, fuse_poe 0.025, build_fusion_features 0.012, discrepancy 0.055 | 0.005 | 1.246 / 0.79 / 0.009 / 0.003 / 0.022 |
| 4 | 0.316 / p95 0.378 | text_ctx 0.088, sentiment 0.024, fuse_poe 0.042, build_fusion_features 0.020, discrepancy 0.095 | 0.009 | 1.234 / 0.791 / 0.009 / 0.003 / 0.023 |

Model files on disk: `silero_vad.onnx` 0.6 MB, `sensevoice/model.int8.onnx` 228.2 MB, `sensevoice/model_enc.int8.onnx` 228.2 MB, `roberta-base-11.onnx` 475.5 MB, `piper/en_US-lessac-medium.onnx` 60.2 MB. Heads: `sentiment.npz` 18 KB, `text_ctx.npz` 790 KB, `text_ctx_k0.npz` 790 KB.

Machine recorded by the bench: Intel(R) Xeon(R) Processor @ 2.10GHz, nproc 4, MemTotal 16093 MB, python 3.12.13, onnxruntime 1.31.0, numpy 2.5.4. Flags of interest: avx2 yes, fma yes, avx512f yes, avx512bw yes, avx512_vnni yes, amx_tile yes, amx_int8 yes, amx_bf16 yes.

### Replay bench (`replay_bench_vm.json`, written by `tools/replay_bench.py`)

Generated 2026-10-12T20:38:11Z on `vm` by `python tools/replay_bench.py --clips proxy/*.wav --n 20 --realtime --out results/replay_bench_vm.json`. The clip attribution was recomputed from the stored utterances and clip spans at 2026-10-12T22:08:04Z by `python tools/replay_bench.py --reattribute results/replay_bench_vm.json results/replay_bench_vm_long.json results/replay_bench_vm_meld.json`. The latencies were not touched.

- Clips: 20 from `proxy` (duration p50 2.58 s, max 5.00 s), streamed at real-time pace on one stream with 1.0 s of silence after each clip. Before the measured run, the first clip was streamed once on a separate stream as a warm-up. Streaming the 73.536 s of audio took 73.504 s.
- Service: 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 500 ms, provisionals at 1.0 Hz over a 4.0 s window, fusion mode `auto` resolved to `stacked`, head files in `heads/`: audio_cal.npz, audio_cal_A.npz, audio_cal_B.npz, fusion.npz, sentiment.npz, text_ctx.npz, text_ctx_k0.npz. The finals report `fused.mode` `stacked` (20).
- Models loaded in 2.67 s. The load average was [0.63, 1.69, 2.84] at the start and [1.31, 1.68, 2.74] at the end.
- Messages: 20 finals, 20 responses, 51 provisionals, errors: none.

#### Final state after the endpoint, by utterance length (targets of design 1.10, at 4 threads)

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | client-side p50 / p95 ms | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | prov. in flight p50 | pass |
|---|---|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 19 | 146.5 / 258.0 | 250 / 400 | 147.2 / 259.1 | 146.7 / 258.2 | 4.5 / 95.2 / 7.9 / 40.2 / 0.6 | 0.0 | pass |
| 5 to 8 s (95 % cumulative) | 1 | 227.7 / 227.7 | 350 / 550 | 228.9 / 228.9 | 228.0 / 228.0 | 7.4 / 152.7 / 13.5 / 52.1 / 0.7 | 0.0 | pass |
| 8 s to the 15 s cap (99.7 % cumulative) | 0 | no utterances in this bucket | 700 / 1000 | - | - | - | - | n/a |
| all | 20 | 155.6 / 252.6 | - | 156.4 / 253.7 | 155.8 / 252.7 | 4.5 / 103.4 / 8.0 / 40.3 / 0.6 | - | - |

#### Other targets of design 1.10

| target | measured | pass |
|---|---|---|
| reply text within 5.0 ms of the final state (p95) | p50 0.180, p95 0.420, max 0.800 ms over 20 replies | pass |
| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 1.020 s (n 24), tail-window gap p50 - s (n 0), full-to-tail switch gap p50 1.020 s, first provisional 1.020 s after speech start, 44 full + 7 tail provisionals, 14/14 utterances over 1.5 s got one, dropped 0, provisional SenseVoice p50 72.4 ms | pass |
| sustained compute below 0.3 s per second of audio | worker 0.108 s per audio s (0.151 per speech s), whole process 0.504 cpu s per audio s, cpu % sampled mean 49.5 max 296.9 | pass |
| memory | rss 1117.5 MB after load, peak 1187.7 MB during the run (service + client + monitor in one process) | - |
| worker lag | max health queue_s 0.334 s, degraded finals 0, empty transcripts 0 | - |
| clip attribution (t_media_end within [media_start, next clip's media_start)) | Clips without a final: none. Clips split into several finals: none. Finals outside every clip span: none. No final ends after the end of its clip. | - |

Definitions:

- Service-side latency runs from the arrival of the endpoint frame (`t_service_recv_ns`) to the emission of the final state.
- Client-side latency is the monitor's arrival time minus the capture time of the endpoint frame. It is valid because client and monitor run on one host.
- Endpoint to response runs from the arrival of the endpoint frame to the emission of the response.
- The service, the replay client and the monitor run in one process, so RSS and CPU include all three.
- A final belongs to the clip whose span holds its `t_media_end`, the time the speech stopped (rule: `t_media_end within [media_start, next clip's media_start)`). A span runs from the start of a clip to the start of the next one, so it includes the silence streamed after the clip. The segment start is not used because it carries the VAD pre-roll.
- The proxy clips are Piper TTS of MELD dev text. Their latencies are real. Their emotion labels are not a result.

### Replay bench (`replay_bench_vm_long.json`, written by `tools/replay_bench.py`)

Generated 2026-10-12T20:40:50Z on `vm` by `python tools/replay_bench.py --clips bench_clips/long/*.wav --n 0 --repeat 3 --realtime --out results/replay_bench_vm_long.json`. The clip attribution was recomputed from the stored utterances and clip spans at 2026-10-12T22:08:04Z by `python tools/replay_bench.py --reattribute results/replay_bench_vm.json results/replay_bench_vm_long.json results/replay_bench_vm_meld.json`. The latencies were not touched.

- Clips: 12 from `bench_clips/long` (duration p50 10.50 s, max 15.00 s), streamed at real-time pace on one stream with 1.0 s of silence after each clip. Before the measured run, the first clip was streamed once on a separate stream as a warm-up. Streaming the 138.432 s of audio took 138.4 s.
- Service: 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 500 ms, provisionals at 1.0 Hz over a 4.0 s window, fusion mode `auto` resolved to `stacked`, head files in `heads/`: audio_cal.npz, audio_cal_A.npz, audio_cal_B.npz, fusion.npz, sentiment.npz, text_ctx.npz, text_ctx_k0.npz. The finals report `fused.mode` `stacked` (12).
- Models loaded in 2.56 s. The load average was [1.0, 1.58, 2.69] at the start and [0.57, 1.15, 2.36] at the end.
- Messages: 12 finals, 12 responses, 84 provisionals, errors: none.

#### Final state after the endpoint, by utterance length (targets of design 1.10, at 4 threads)

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | client-side p50 / p95 ms | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | prov. in flight p50 | pass |
|---|---|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 0 | no utterances in this bucket | 250 / 400 | - | - | - | - | n/a |
| 5 to 8 s (95 % cumulative) | 3 | 264.2 / 283.9 | 350 / 550 | 265.1 / 284.7 | 264.4 / 284.1 | 9.3 / 186.1 / 23.1 / 43.5 / 0.7 | 0.0 | pass |
| 8 s to the 15 s cap (99.7 % cumulative) | 9 | 403.5 / 530.1 | 700 / 1000 | 404.4 / 531.0 | 403.8 / 530.3 | 17.7 / 305.5 / 41.5 / 43.1 / 0.7 | 0.0 | pass |
| all | 12 | 379.9 / 523.8 | - | 380.8 / 524.7 | 380.2 / 524.0 | 16.9 / 284.0 / 36.2 / 43.3 / 0.7 | - | - |

#### Other targets of design 1.10

| target | measured | pass |
|---|---|---|
| reply text within 5.0 ms of the final state (p95) | p50 0.200, p95 0.280, max 0.290 ms over 12 replies | pass |
| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 1.020 s (n 24), tail-window gap p50 2.020 s (n 36), full-to-tail switch gap p50 1.020 s, first provisional 1.020 s after speech start, 36 full + 48 tail provisionals, 12/12 utterances over 1.5 s got one, dropped 0, provisional SenseVoice p50 116.8 ms | pass |
| sustained compute below 0.3 s per second of audio | worker 0.102 s per audio s (0.113 per speech s), whole process 0.468 cpu s per audio s, cpu % sampled mean 46.1 max 325.7 | pass |
| memory | rss 1117.2 MB after load, peak 1204.4 MB during the run (service + client + monitor in one process) | - |
| worker lag | max health queue_s 0.425 s, degraded finals 0, empty transcripts 0 | - |
| clip attribution (t_media_end within [media_start, next clip's media_start)) | Clips without a final: none. Clips split into several finals: none. Finals outside every clip span: none. 9 of the 12 finals end after the end of their clip, by up to 0.032 s. | - |

Definitions:

- Service-side latency runs from the arrival of the endpoint frame (`t_service_recv_ns`) to the emission of the final state.
- Client-side latency is the monitor's arrival time minus the capture time of the endpoint frame. It is valid because client and monitor run on one host.
- Endpoint to response runs from the arrival of the endpoint frame to the emission of the response.
- The service, the replay client and the monitor run in one process, so RSS and CPU include all three.
- A final belongs to the clip whose span holds its `t_media_end`, the time the speech stopped (rule: `t_media_end within [media_start, next clip's media_start)`). A span runs from the start of a clip to the start of the next one, so it includes the silence streamed after the clip. The segment start is not used because it carries the VAD pre-roll.
- The clips are cut and tiled from `models/sensevoice/test_wavs/en.wav` by `tools/bench_components.py`.

### Replay bench (`replay_bench_vm_meld.json`, written by `tools/replay_bench.py`)

Generated 2026-10-12T20:56:04Z on `vm` by `python tools/replay_bench.py --clips audio/test/*.wav --n 200 --realtime --out results/replay_bench_vm_meld.json`. The clip attribution was recomputed from the stored utterances and clip spans at 2026-10-12T22:08:04Z by `python tools/replay_bench.py --reattribute results/replay_bench_vm.json results/replay_bench_vm_long.json results/replay_bench_vm_meld.json`. The latencies were not touched.

- Clips: 200 from `audio/test` (duration p50 2.75 s, max 16.06 s), streamed at real-time pace on one stream with 1.0 s of silence after each clip. Before the measured run, the first clip was streamed once on a separate stream as a warm-up. Streaming the 903.424 s of audio took 903.392 s.
- Service: 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 500 ms, provisionals at 1.0 Hz over a 4.0 s window, fusion mode `auto` resolved to `stacked`, head files in `heads/`: audio_cal.npz, audio_cal_A.npz, audio_cal_B.npz, fusion.npz, sentiment.npz, text_ctx.npz, text_ctx_k0.npz. The finals report `fused.mode` `stacked` (201).
- Models loaded in 2.79 s. The load average was [0.56, 1.12, 2.34] at the start and [0.79, 0.71, 1.3] at the end.
- Messages: 201 finals, 201 responses, 577 provisionals, errors: none.

#### Final state after the endpoint, by utterance length (targets of design 1.10, at 4 threads)

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | client-side p50 / p95 ms | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | prov. in flight p50 | pass |
|---|---|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 168 | 152.3 / 241.7 | 250 / 400 | 153.2 / 243.0 | 152.4 / 241.9 | 4.9 / 92.7 / 7.7 / 37.6 / 0.6 | 0.0 | pass |
| 5 to 8 s (95 % cumulative) | 19 | 282.9 / 376.5 | 350 / 550 | 283.7 / 378.1 | 283.1 / 376.7 | 10.4 / 209.7 / 21.4 / 38.8 / 0.7 | 0.0 | pass |
| 8 s to the 15 s cap (99.7 % cumulative) | 14 | 371.2 / 491.3 | 700 / 1000 | 372.1 / 492.3 | 371.4 / 491.5 | 16.0 / 280.5 / 30.2 / 39.9 / 0.7 | 0.0 | pass |
| all | 201 | 163.8 / 353.2 | - | 165.6 / 353.8 | 163.9 / 353.3 | 5.3 / 103.8 / 8.8 / 37.9 / 0.6 | - | - |

#### Other targets of design 1.10

| target | measured | pass |
|---|---|---|
| reply text within 5.0 ms of the final state (p95) | p50 0.170, p95 0.350, max 0.590 ms over 201 replies | pass |
| provisionals at 1 Hz during the first 4 s of speech | full-window gap p50 1.020 s (n 259), tail-window gap p50 2.020 s (n 59), full-to-tail switch gap p50 1.020 s, first provisional 1.020 s after speech start, 458 full + 119 tail provisionals, 154/154 utterances over 1.5 s got one, dropped 0, provisional SenseVoice p50 79.9 ms | pass |
| sustained compute below 0.3 s per second of audio | worker 0.099 s per audio s (0.128 per speech s), whole process 0.458 cpu s per audio s, cpu % sampled mean 45.6 max 331.5 | pass |
| memory | rss 1137.4 MB after load, peak 1262.1 MB during the run (service + client + monitor in one process) | - |
| worker lag | max health queue_s 0.378 s, degraded finals 2, empty transcripts 0 | - |
| clip attribution (t_media_end within [media_start, next clip's media_start)) | 30 clips without a final (dia100_utt18.wav#13, dia100_utt19.wav#14, dia100_utt8.wav#21, dia101_utt1.wav#24, dia102_utt3.wav#30, dia102_utt4.wav#31, dia102_utt7.wav#34, dia102_utt8.wav#35, dia102_utt9.wav#36, dia105_utt0.wav#46, dia105_utt1.wav#47, dia105_utt2.wav#48, dia107_utt1.wav#54, dia107_utt2.wav#58, dia107_utt5.wav#61, dia107_utt6.wav#62, dia108_utt3.wav#67, dia108_utt7.wav#71, dia108_utt8.wav#72, dia109_utt4.wav#78, dia110_utt3.wav#89, dia111_utt8.wav#108, dia113_utt14.wav#138, dia114_utt2.wav#151, dia116_utt1.wav#162, dia116_utt10.wav#163, dia116_utt11.wav#164, dia116_utt3.wav#166, dia118_utt0.wav#182, dia118_utt1.wav#183). 24 clips split into several finals (dia103_utt0.wav#37, dia103_utt3.wav#40, dia103_utt4.wav#41, dia107_utt0.wav#53, dia107_utt10.wav#55, dia109_utt1.wav#75, dia109_utt2.wav#76, dia111_utt3.wav#103, dia112_utt12.wav#114, dia112_utt16.wav#118, dia112_utt17.wav#119, dia113_utt0.wav#132, dia114_utt3.wav#152, dia114_utt4.wav#153, dia115_utt3.wav#158, dia115_utt4.wav#159, dia115_utt5.wav#160, dia116_utt2.wav#165, dia116_utt5.wav#168, dia118_utt10.wav#184, dia118_utt8.wav#192, dia119_utt1.wav#195, dia119_utt2.wav#196, dia119_utt3.wav#197). Finals outside every clip span: none. 112 of the 201 finals end after the end of their clip, by up to 0.576 s. | - |

Definitions:

- Service-side latency runs from the arrival of the endpoint frame (`t_service_recv_ns`) to the emission of the final state.
- Client-side latency is the monitor's arrival time minus the capture time of the endpoint frame. It is valid because client and monitor run on one host.
- Endpoint to response runs from the arrival of the endpoint frame to the emission of the response.
- The service, the replay client and the monitor run in one process, so RSS and CPU include all three.
- A final belongs to the clip whose span holds its `t_media_end`, the time the speech stopped (rule: `t_media_end within [media_start, next clip's media_start)`). A span runs from the start of a clip to the start of the next one, so it includes the silence streamed after the clip. The segment start is not used because it carries the VAD pre-roll.
- The clips are MELD utterances, converted to 16 kHz WAV by `tools/extract_audio.py`. They are cut at the end of the subtitle, so the detected end of speech can fall after the end of the clip. The buckets group finals by segment length, and the clip-attribution row shows where the segments and the clips differ.

### Replay bench with video, `--reply llm` (`replay_bench_vm_tav.json`, written by `tools/replay_bench.py --video`)

Generated 2026-10-13T02:57:56Z on `vm` by `python tools/replay_bench.py --video --clips /home/user/meld-data/mp4/test/*.mp4 --n 200 --realtime --reply llm --llm-dir models/llm/Qwen2.5-0.5B-Instruct --out results/replay_bench_vm_tav.json`.

- Clips: 200 MP4 files from `/home/user/meld-data/mp4/test` (duration p50 2.77 s, max 16.06 s). The replay client sends the audio and the frames (10.0 fps) of each clip on one media clock, at real-time pace, on one stream, with 1.0 s of silence after each clip. Before the measured run, the first clip was streamed once on a separate stream as a warm-up. Streaming 912.832 s of audio and 7044 frames took 912.8 s.
- Service: `--vision`, 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 500 ms, frame-rate cap 10.0 fps, vision lag 150.0 ms, speaker rule `asd`, vision heads vision_cal, fusion3, sentiment3, commit policy `fqi_mlp_c0.05`, replies `llm` from `Qwen2.5-0.5B-Instruct`. The finals report `fused.mode` `stacked_tav` (206).
- Models loaded in 9.58 s. The load average was [2.07, 3.2, 3.56] at the start and [1.42, 1.58, 2.23] at the end.
- Messages: 206 finals, 206 responses, 619 provisionals, errors: none.

#### Final state after the endpoint, by utterance length (targets of design 1.10, at 4 threads)

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | pass |
|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 175 | 178.4 / 347.7 | 250 / 400 | 735.1 / 1480.0 | 5.4 / 113.5 / 9.3 / 37.0 / 0.7 | pass |
| 5 to 8 s (95 % cumulative) | 18 | 336.2 / 536.6 | 350 / 550 | 950.2 / 1412.6 | 12.4 / 249.1 / 22.4 / 40.8 / 0.8 | pass |
| 8 s to the 15 s cap (99.7 % cumulative) | 13 | 651.9 / 809.5 | 700 / 1000 | 1234.7 / 2243.1 | 16.8 / 525.7 / 33.4 / 40.9 / 0.7 | pass |
| all | 206 | 193.9 / 546.8 | - | 772.6 / 1593.6 | 5.8 / 121.4 / 11.1 / 37.9 / 0.7 | - |

#### Reply, commit, vision, compute and memory

| item | measured | target | pass |
|---|---|---|---|
| reply after the final state (p50 / p95 / max ms, n) | 549.92 / 1333.29 / 2440.38, n 206 | none here: the 5.0 ms p95 target of design 1.10 is the bank's | n/a |
| first streamed delta after the endpoint frame (p50 / p95 ms, n) | 388.85 / 855.99, n 206 | none set before the run | n/a |
| generation (p50 / p95 ms) | 546.83 / 1329.88 | none set before the run | n/a |
| reply source | generator 206, bank 0 of 206 responses, 1338 deltas, fallbacks: none | - | - |
| label committed before the endpoint | share 0.8155 (168 of 206 finals with a commit block) | - | - |
| commit time of the 168 commits before the endpoint (p50 / p95 s after the segment start) | 1.02 / 2.66 | - | - |
| committed label equal to the final's label | share 0.767 of the 206 finals with a commit block (an endpoint commit is the final) | - | - |
| vision thread per processed frame (mean ms) | 30.49 over 7044 frames | - | - |
| video frames dropped / skipped / bad | 0 / 0 / 0, finals that waited the whole vision lag: 0 | - | - |
| vision wait at the final (p50 / p95 ms) | 0.01 / 0.03 | - | - |
| face evidence at the final (p50 / p95 ms) | 1.35 / 3.04 | - | - |
| worker compute per second of audio | 0.1549 s (0.2003 per speech s) | below 0.3 s | pass |
| whole process cpu per second of audio | 1.0986 cpu s, cpu % sampled mean 109.48 max 390.7 | - | - |
| memory | rss 2942.8 MB after load, peak 3422.7 MB during the run | - | - |

Definitions:

- Service-side latency runs from the arrival of the endpoint frame (`t_service_recv_ns`) to the emission of the final state.
- Endpoint to response runs from the arrival of the endpoint frame to the emission of the response.
- Reply after the final state runs from the emission of the final state to the emission of the response. With `--reply llm` it includes the whole generation. The first streamed delta runs from the arrival of the endpoint frame to the emission of the first `response.delta`, so it includes the computation of the final state.
- The commit time counts from the start of the VAD segment. A final without an earlier commit is its own commit, at the endpoint.
- The vision thread time per frame covers the JPEG decode, the face expert and the row building.
- Worker compute keeps the rule of the audio benches (the worker's stage timings of every final and provisional). It leaves out the vision thread and the reply thread. The process cpu includes every thread.
- The service, the replay client and the monitor run in one process, so RSS and CPU include all three, and also the client's MP4 decoding and JPEG encoding.

### Replay bench with video, `--reply bank` (`replay_bench_vm_tav_bank.json`, written by `tools/replay_bench.py --video`)

Generated 2026-10-13T03:13:38Z on `vm` by `python tools/replay_bench.py --video --clips /home/user/meld-data/mp4/test/*.mp4 --n 200 --realtime --reply bank --out results/replay_bench_vm_tav_bank.json`.

- Clips: 200 MP4 files from `/home/user/meld-data/mp4/test` (duration p50 2.77 s, max 16.06 s). The replay client sends the audio and the frames (10.0 fps) of each clip on one media clock, at real-time pace, on one stream, with 1.0 s of silence after each clip. Before the measured run, the first clip was streamed once on a separate stream as a warm-up. Streaming 912.832 s of audio and 7044 frames took 912.8 s.
- Service: `--vision`, 4 SenseVoice threads, 2 RoBERTa threads, endpoint silence 500 ms, frame-rate cap 10.0 fps, vision lag 150.0 ms, speaker rule `asd`, vision heads vision_cal, fusion3, sentiment3, commit policy `fqi_mlp_c0.05`, replies `bank`. The finals report `fused.mode` `stacked_tav` (206).
- Models loaded in 2.75 s. The load average was [1.11, 1.5, 2.18] at the start and [0.65, 1.2, 1.58] at the end.
- Messages: 206 finals, 206 responses, 619 provisionals, errors: none.

#### Final state after the endpoint, by utterance length (targets of design 1.10, at 4 threads)

| bucket | n | service-side p50 / p95 ms | target p50 / p95 | endpoint to response p50 / p95 ms | stages p50 (fe / sv / pros / txt / heads) | pass |
|---|---|---|---|---|---|---|
| up to 5 s (83 % of MELD) | 175 | 172.1 / 392.8 | 250 / 400 | 172.3 / 393.0 | 5.1 / 108.8 / 8.6 / 37.3 / 0.7 | pass |
| 5 to 8 s (95 % cumulative) | 18 | 340.6 / 509.4 | 350 / 550 | 340.9 / 509.6 | 10.9 / 255.3 / 24.3 / 36.2 / 0.8 | pass |
| 8 s to the 15 s cap (99.7 % cumulative) | 13 | 576.0 / 875.6 | 700 / 1000 | 576.4 / 875.9 | 17.4 / 492.8 / 35.7 / 42.2 / 0.7 | pass |
| all | 206 | 196.5 / 522.5 | - | 197.0 / 522.7 | 5.7 / 120.8 / 10.2 / 37.7 / 0.7 | - |

#### Reply, commit, vision, compute and memory

| item | measured | target | pass |
|---|---|---|---|
| reply after the final state (p50 / p95 / max ms, n) | 0.24 / 0.68 / 4.88, n 206 | 5.0 ms p95 | pass |
| label committed before the endpoint | share 0.8155 (168 of 206 finals with a commit block) | - | - |
| commit time of the 168 commits before the endpoint (p50 / p95 s after the segment start) | 1.02 / 2.66 | - | - |
| committed label equal to the final's label | share 0.767 of the 206 finals with a commit block (an endpoint commit is the final) | - | - |
| vision thread per processed frame (mean ms) | 29.72 over 7044 frames | - | - |
| video frames dropped / skipped / bad | 0 / 0 / 0, finals that waited the whole vision lag: 0 | - | - |
| vision wait at the final (p50 / p95 ms) | 0.01 / 0.02 | - | - |
| face evidence at the final (p50 / p95 ms) | 1.21 / 2.44 | - | - |
| worker compute per second of audio | 0.1507 s (0.1948 per speech s) | below 0.3 s | pass |
| whole process cpu per second of audio | 0.8081 cpu s, cpu % sampled mean 80.55 max 395.0 | - | - |
| memory | rss 1181.7 MB after load, peak 1644.9 MB during the run | - | - |

Definitions:

- Service-side latency runs from the arrival of the endpoint frame (`t_service_recv_ns`) to the emission of the final state.
- Endpoint to response runs from the arrival of the endpoint frame to the emission of the response.
- Reply after the final state runs from the emission of the final state to the emission of the response. With `--reply bank` the response is one bank line.
- The commit time counts from the start of the VAD segment. A final without an earlier commit is its own commit, at the endpoint.
- The vision thread time per frame covers the JPEG decode, the face expert and the row building.
- Worker compute keeps the rule of the audio benches (the worker's stage timings of every final and provisional). It leaves out the vision thread and the reply thread. The process cpu includes every thread.
- The service, the replay client and the monitor run in one process, so RSS and CPU include all three, and also the client's MP4 decoding and JPEG encoding.

### Reply generator bench (`generator_bench_vm.json`, written by `tools/bench_generator.py`)

Generated 2026-10-13T02:22:13Z on `vm` by `python tools/bench_generator.py --n 200 --threads 2 4 --models Qwen2.5-0.5B-Instruct/model_int8.onnx Qwen2.5-0.5B-Instruct/model_q4.onnx`. The states are 200 test clips from `cache/vision_post_test.npz` (seed 0): 100 on gold text and 100 on SenseVoice transcripts. Decoding is greedy, with at most 40 new tokens. The load average was [1.45, 3.37, 4.29] at the start and [3.67, 3.0, 3.64] at the end. The bench took 528.5 s.

| file | threads | first token p50 / p95 ms | first word p50 / p95 ms | full reply p50 / p95 ms | decode passes per s | final-reply tokens per s | check pass rate | meets the targets at 4 threads |
|---|---|---|---|---|---|---|---|---|
| `Qwen2.5-0.5B-Instruct/model_int8.onnx` | 2 | 103.78 / 136.69 | 159.48 / 323.26 | 531.98 / 1304.45 | 20.09 | 16.76 | 0.97 |  |
| `Qwen2.5-0.5B-Instruct/model_int8.onnx` | 4 | 70.94 / 100.27 | 108.75 / 233.55 | 378.52 / 896.2 | 28.47 | 23.74 | 0.97 | yes |
| `Qwen2.5-0.5B-Instruct/model_q4.onnx` | 2 | 274.86 / 354.14 | 349.75 / 458.58 | 792.29 / 1291.37 | 17.33 | 14.84 | 0.995 |  |
| `Qwen2.5-0.5B-Instruct/model_q4.onnx` | 4 | 155.02 / 196.14 | 202.44 / 292.25 | 496.41 / 830.08 | 26.87 | 23.01 | 0.995 | yes |

Targets at 4 threads (`choice.rule` in the file, defined in the docstring of `tools/bench_generator.py`): first-token p95 at most 350.0 ms and a decode rate of at least 15.0 passes per s. Among the files of one model, q4 replaces int8 only when it is faster by more than 1.2x at the same reply length with a check pass rate no lower. The choice of this run covers its own files only. It is `Qwen2.5-0.5B-Instruct/model_int8.onnx`, which meets both targets. The reason recorded in the file: "int8 (q4 speedup 0.80x on a reply of 9.82 tokens, q4 pass rate 0.995 vs 0.97)".

Definitions:

- First token runs from the call to the first chosen token. First word runs to the first word passed to the stream.
- Decode passes per s counts every forward pass after the first token (rolled-back tokens and the end token included) over the time after the first token. This is the rate the targets use. Final-reply tokens per s counts only the tokens of the final reply.
- The check pass rate is the share of replies that pass `check_reply` (length, register, emotion words, prompt echo).

### Reply generator bench (`generator_bench_vm_1.5b.json`, written by `tools/bench_generator.py`)

Generated 2026-10-13T02:31:02Z on `vm` by `python tools/bench_generator.py --n 200 --threads 4 --models Qwen2.5-1.5B-Instruct/model_int8.onnx Qwen2.5-1.5B-Instruct/model_q4.onnx --out results/generator_bench_vm_1.5b.json`. The states are 200 test clips from `cache/vision_post_test.npz` (seed 0): 100 on gold text and 100 on SenseVoice transcripts. Decoding is greedy, with at most 40 new tokens. The load average was [3.67, 3.0, 3.64] at the start and [4.01, 3.84, 3.78] at the end. The bench took 621.4 s.

| file | threads | first token p50 / p95 ms | first word p50 / p95 ms | full reply p50 / p95 ms | decode passes per s | final-reply tokens per s | check pass rate | meets the targets at 4 threads |
|---|---|---|---|---|---|---|---|---|
| `Qwen2.5-1.5B-Instruct/model_int8.onnx` | 4 | 149.35 / 194.54 | 352.49 / 783.58 | 1228.94 / 2985.06 | 13.22 | 8.79 | 0.97 | no |
| `Qwen2.5-1.5B-Instruct/model_q4.onnx` | 4 | 473.33 / 580.74 | 575.33 / 773.84 | 1330.17 / 2541.36 | 11.64 | 9.49 | 0.995 | no |

Targets at 4 threads (`choice.rule` in the file, defined in the docstring of `tools/bench_generator.py`): first-token p95 at most 350.0 ms and a decode rate of at least 15.0 passes per s. Among the files of one model, q4 replaces int8 only when it is faster by more than 1.2x at the same reply length with a check pass rate no lower. The choice of this run covers its own files only. It is `Qwen2.5-1.5B-Instruct/model_int8.onnx`, which does not meet both targets. The reason recorded in the file: "no file meets both targets: the smallest model's file with the lower first-token p95".

Definitions:

- First token runs from the call to the first chosen token. First word runs to the first word passed to the stream.
- Decode passes per s counts every forward pass after the first token (rolled-back tokens and the end token included) over the time after the first token. This is the rate the targets use. Final-reply tokens per s counts only the tokens of the final reply.
- The check pass rate is the share of replies that pass `check_reply` (length, register, emotion words, prompt echo).

The deployed reply generator is `Qwen2.5-0.5B-Instruct/model_int8.onnx` (`DEFAULT_MODEL` and `DEFAULT_ONNX` in `meld_rt/response/generator.py`).

## Laptop (user-run)

Every cell below reads `[laptop: not yet measured]` until the benches run on the laptop. When their JSON files are in `results/`, this section shows them in place of the table. docs/runbook_laptop.md gives the steps. Nothing here is typed from memory.

Commands:

```
uv run python tools/count_params.py --models models/ --heads heads/ --out results/params.md
uv run python tools/bench_components.py --threads 1,2,4 --clips bench_clips/ --out results/bench_components_$(hostname).json
uv run python tools/replay_bench.py --clips "proxy/*.wav" --n 20 --realtime --out results/replay_bench_$(hostname).json
uv run python tools/bench_components.py --build-only --trim-silence --durations 6,9,12,15 --clips bench_clips/long
uv run python tools/replay_bench.py --clips "bench_clips/long/*.wav" --n 0 --repeat 3 --realtime --endpoint-silence-ms 500 --out results/replay_bench_$(hostname)_long.json
uv run python tools/render_measurements.py --out results/measurements.md
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
