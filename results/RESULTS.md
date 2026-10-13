# Results on MELD test

Written by `python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json`. Every number below was computed by that command from the inputs listed. Model selection, temperature and thresholds were chosen on dev by `tools/train_heads.py`, and the test split was read once per condition. Weighted F1 is the primary metric, as in the published rows. The CIs are 95 % percentile bootstrap intervals over test dialogues (1000 resamples, seed 0). ECE uses 10 equal-width confidence bins. All metrics are in percent.

## Inputs

| file | bytes | sha256 |
| --- | --- | --- |
| `cache/text_test.npy` | 8018048 | `bd454c8e0f496e649b8e9f61a1acd098ffe478c51184a30e28185c71135e3ce7` |
| `meld/test_sent_emo.csv` | 294526 | `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888` |
| `cache/audio_test.npz` | 30360152 | `a9e22761afa2096ec65841c9f8adc9af541a05e2517e8c0b9e7a1337c58007e2` |
| `heads/ablation_no_events_audio_cal.npz` | 3914 | `78d7d422f66c6b05063c7f432eda1a798e5467583676c61833703c58f7b88e43` |
| `heads/ablation_no_events_fusion.npz` | 4134 | `34aff92fe1310121b84ce7e0b1508ce0a426f66a09566a6dbf46238a8bd2fd4c` |
| `heads/ablation_no_events_no_prosody_audio_cal.npz` | 3814 | `62fc27d16bcd2ed08df600748942915fe08015dbe8c573b1bad4ccb2a5da769b` |
| `heads/ablation_no_events_no_prosody_fusion.npz` | 4210 | `2d5f52fa8083caec2761124f31ec13902ab1d7084057317b2be61713dc3fbf4d` |
| `heads/ablation_no_prosody_audio_cal.npz` | 4230 | `5076cfbf31deb33a0b3e3f74e937d08991a56e094c20e89effa87fcd4423ac35` |
| `heads/ablation_no_prosody_fusion.npz` | 4186 | `dd45f19b505091ba6a2f61368e9e6484b2f76d4d580bf5ffd9f9b086fbb42e73` |
| `heads/audio_cal.npz` | 3622 | `49be433bf1a2b0b7cb635dda743c4416af0ed8958fc28235f84f5f3a31509cfc` |
| `heads/audio_cal_A.npz` | 3622 | `49be433bf1a2b0b7cb635dda743c4416af0ed8958fc28235f84f5f3a31509cfc` |
| `heads/audio_cal_B.npz` | 40486 | `4cb548fa197a50e7d83241e03af7cc64237bcfa406e69dae18850a12f3c9c2d4` |
| `heads/fusion.npz` | 3192 | `19fb219aa787d75611ec39d65067bc8afdd982bc807ea9b4601729e8c95c154d` |
| `heads/sentiment.npz` | 3010 | `4cd0467fe75f6b97f956994d8b099c63a256f52f68a36fd105766d7301d0abbf` |
| `heads/sentiment_text_only.npz` | 18328 | `65c6614faa7a553bf62897048bb807eb52173e4ec873a036d50903b95225b78d` |
| `heads/text_ctx.npz` | 808726 | `916806bf4f0a0055e989c6e73a289441378c719b9a4fb68c0628ad652d97dd92` |
| `heads/text_ctx_k0.npz` | 808726 | `916806bf4f0a0055e989c6e73a289441378c719b9a4fb68c0628ad652d97dd92` |
| `cache/text_asr_test.npy` | 8018048 | `241d14d7567d99dc4d27f94a43e7278581ab0221460b113e8d9453cd74070f8b` |
| `cache/text_asr_test_p2.npy` | 8018048 | `cdf06ab64553082240ed46debf44f142e67716ad2aac1d8ae4448357857a71f0` |
| `cache/text_asr_test_p4.npy` | 8018048 | `7bfd045932117bfeda275bc99c0654d8166000c2360a54404caabc5867fe5179` |

Deployed text head, from `heads/manifest.json`: k = 0, class weights `none`, temperature 1.2228, thresholds `{"low_conf": 0.47, "high_conf": 0.73, "jsd_flag": 0.3831}`, text_only = False, scikit-learn 1.9.1.

## Emotion (7 classes)

| condition | status | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | done | 31.27 | [28.62, 34.31] | 9.28 | 48.12 | 0.97 | train prior as posterior; label = neutral |
| text_k0 | done | 59.62 | [57.38, 62.03] | 41.22 | 62.76 | 3.83 | class weights none, T=1.223 |
| text_best_k | done | 59.62 | [57.38, 62.03] | 41.22 | 62.76 | 3.83 | k=0, class weights none, T=1.223 |
| text_runtime_regime | done | 59.61 | [57.36, 62.08] | 40.82 | 62.87 | 3.54 | same head as text_best_k, same-speaker window (k=0: differs from text_best_k only through the turn-index flags) |
| audio_raw_tag | done | 48.52 | [46.33, 50.96] | 28.18 | 51.34 | 19.77 | argmax over the seven named tokens |
| audio_cal_A | done | 48.02 | [45.66, 50.64] | 26.92 | 53.72 | 2.84 | T=1.015 |
| audio_cal_B | done | 46.32 | [44.14, 48.43] | 28.51 | 47.78 | 4.73 | T=2.061 |
| fused_stacked | done | 60.77 | [58.65, 63.00] | 39.24 | 63.98 | 1.52 | T=0.846 |
| fused_poe | done | 58.26 | [55.91, 60.69] | 35.86 | 63.14 | 11.12 | w=0.75; calibrated audio posterior |
| fused_poe_uncalibrated | done | 59.00 | [56.73, 61.36] | 36.65 | 63.49 | 13.14 | w=0.5; raw sensevoice posterior, uncalibrated |
| ablation_no_events | done | 60.91 | [58.82, 63.33] | 39.53 | 64.10 | 2.90 | calibrator A without events (13 inputs, T=0.990), fusion with 18 inputs (T=0.852) |
| ablation_no_prosody | done | 60.91 | [58.80, 63.09] | 39.38 | 64.14 | 1.65 | calibrator A without prosody (20 inputs, T=1.000), fusion with 19 inputs (T=0.849) |
| ablation_no_events_no_prosody | done | 60.82 | [58.73, 63.08] | 39.40 | 63.98 | 2.05 | calibrator A without events and prosody (9 inputs, T=0.990), fusion with 18 inputs (T=0.852) |
| live_asr_text_only | done | 42.61 | [40.28, 45.37] | 23.86 | 48.08 | 11.51 | sensevoice transcripts, corpus WER 0.348488 over 2610 utterances (cache/wer_test.json), service asr gate: 35 empty transcripts (asr_failed) and 38 with asr_conf below 0.3 (low_asr_conf) get a uniform posterior and the label neutral, heads trained on gold text |
| live_asr_text | done | 47.05 | [44.53, 49.57] | 27.50 | 52.53 | 9.71 | T=0.846, sensevoice transcripts, corpus WER 0.348488 over 2610 utterances (cache/wer_test.json), service asr gate: 35 empty transcripts (asr_failed) and 38 with asr_conf below 0.3 (low_asr_conf) get a uniform posterior and the label neutral, heads trained on gold text |
| disagreement_subset | done | 49.62 | [44.54, 54.32] | 37.73 | 52.35 | 5.87 | stacked fusion on the 489 of 2610 rows with jsd above 0.3831 (thresholds.jsd_flag, chosen on dev), jsd of the gold-text posterior against the raw sensevoice posterior |
| early_commit_audio_2s | done | 45.86 | [43.48, 48.55] | 24.20 | 53.18 | 2.24 | offline prefix experiment: first 2 s, 925 of 2610 clips are not longer than 2 s and use their full-clip values |
| early_commit_fused_2s | done | 44.35 | [41.86, 46.95] | 24.53 | 51.57 | 14.68 | offline prefix experiment: first 2 s, 925 of 2610 clips are not longer than 2 s and use their full-clip values |
| early_commit_audio_4s | done | 47.34 | [44.88, 49.88] | 25.79 | 53.64 | 2.46 | offline prefix experiment: first 4 s, 1920 of 2610 clips are not longer than 4 s and use their full-clip values |
| early_commit_fused_4s | done | 46.06 | [43.79, 48.41] | 26.16 | 52.18 | 11.36 | offline prefix experiment: first 4 s, 1920 of 2610 clips are not longer than 4 s and use their full-clip values |

### Published rows (emotion)

| row | weighted F1 | source |
| --- | --- | --- |
| MELD paper Table 11, text-CNN, text | 55.02 | Poria et al., ACL 2019, arXiv 1810.02508, Table 11 |
| MELD paper Table 11, bcLSTM, text | 56.44 | same |
| MELD paper Table 11, bcLSTM, audio | 39.08 | same |
| MELD paper Table 11, bcLSTM, text+audio | 59.25 | same |
| MELD paper Table 11, DialogueRNN, text | 57.03 | same |
| MELD paper Table 11, DialogueRNN, audio | 41.79 | same |
| MELD paper Table 11, DialogueRNN, text+audio | 60.25 | same |
| EmoBERTa Table 2, RoBERTa-large, no past or future utterances | 63.46 | Kim and Vossen, arXiv 2108.12009, Table 2 |
| EmoBERTa Table 2, only past | 64.55 | same |
| EmoBERTa Table 2, past and future | 65.61 | same |
| EmoBERTa Table 2, without speaker names | 65.07 | same |
| EmoBERTa Table 2, COSMIC | 65.21 | same |
| EmoBERTa Table 2, DialogXL | 62.41 | same |
| FunAudioLLM Table 8, SenseVoice-S on MELD, no target-domain fine-tune (WF1; UA 50.8, WA 57.8, F1 44.6) | 57.70 | arXiv 2407.04051, Table 8; the report does not list its SER training corpora, so overlap with MELD cannot be excluded |
| FunAudioLLM Table 8, EmoBox reference row (UA 31.5, WA 51.9, F1 32.9) |  | arXiv 2407.04051, Table 8; no WF1 given |

## Sentiment (3 classes)

| condition | status | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment_text_only | done | 66.24 | [64.41, 68.20] | 63.33 | 66.70 | 2.04 | heads/sentiment_text_only.npz, k=0, C=0.1, T=1.420 |
| sentiment_fused | done | 69.17 | [67.37, 70.96] | 66.45 | 69.46 | 1.88 | T=0.805 |

### Published rows (sentiment)

| row | weighted F1 | source |
| --- | --- | --- |
| MELD paper Table 13, text-CNN, text | 64.25 | Poria et al., ACL 2019, Table 13 |
| MELD paper Table 13, DialogueRNN, text | 66.10 | same |
| MELD paper Table 13, DialogueRNN, audio | 49.61 | same |
| MELD paper Table 13, DialogueRNN, text+audio | 67.56 | same |

## Per-class F1 (emotion)

| condition | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 64.98 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| text_k0 | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| text_best_k | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| text_runtime_regime | 78.01 | 53.82 | 20.29 | 24.55 | 54.92 | 15.91 | 38.25 |
| audio_raw_tag | 67.22 | 31.26 | 0.00 | 16.92 | 38.19 | 1.82 | 41.85 |
| audio_cal_A | 69.38 | 29.73 | 0.00 | 20.63 | 32.28 | 0.00 | 36.40 |
| audio_cal_B | 65.34 | 35.54 | 6.84 | 24.63 | 33.29 | 6.20 | 27.74 |
| fused_stacked | 77.87 | 54.91 | 0.00 | 28.16 | 56.07 | 10.53 | 47.12 |
| fused_poe | 76.79 | 52.78 | 0.00 | 18.93 | 53.52 | 5.63 | 43.35 |
| fused_poe_uncalibrated | 77.27 | 52.79 | 0.00 | 17.65 | 55.09 | 8.22 | 45.55 |
| ablation_no_events | 77.84 | 55.54 | 0.00 | 29.29 | 56.44 | 10.96 | 46.65 |
| ablation_no_prosody | 77.90 | 55.47 | 0.00 | 27.96 | 55.96 | 10.53 | 47.83 |
| ablation_no_events_no_prosody | 77.78 | 55.17 | 0.00 | 28.37 | 56.04 | 10.96 | 47.47 |
| live_asr_text_only | 65.06 | 19.72 | 9.38 | 16.22 | 31.51 | 4.44 | 20.69 |
| live_asr_text | 67.53 | 19.94 | 7.55 | 26.62 | 35.62 | 0.00 | 35.22 |
| disagreement_subset | 64.43 | 56.80 | 0.00 | 23.33 | 49.40 | 28.57 | 41.56 |
| early_commit_audio_2s | 69.42 | 28.75 | 0.00 | 12.17 | 27.11 | 0.00 | 31.92 |
| early_commit_fused_2s | 66.83 | 23.66 | 7.27 | 16.73 | 27.93 | 0.00 | 29.26 |
| early_commit_audio_4s | 69.32 | 28.89 | 0.00 | 13.33 | 31.87 | 0.00 | 37.08 |
| early_commit_fused_4s | 67.41 | 22.28 | 7.27 | 20.08 | 34.21 | 0.00 | 31.87 |
| DialogueRNN text+audio (MELD paper Table 11) | 77.44 | 52.51 | 11.68 | 34.59 | 54.40 | 7.89 | 43.65 |

### Confusion matrix of majority (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1256 | 0 | 0 | 0 | 0 | 0 | 0 |
| surprise | 281 | 0 | 0 | 0 | 0 | 0 | 0 |
| fear | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| sadness | 208 | 0 | 0 | 0 | 0 | 0 | 0 |
| joy | 402 | 0 | 0 | 0 | 0 | 0 | 0 |
| disgust | 68 | 0 | 0 | 0 | 0 | 0 | 0 |
| anger | 345 | 0 | 0 | 0 | 0 | 0 | 0 |

### Confusion matrix of text_k0 (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1102 | 50 | 2 | 16 | 61 | 5 | 20 |
| surprise | 56 | 155 | 2 | 2 | 36 | 4 | 26 |
| fear | 18 | 7 | 7 | 7 | 6 | 0 | 5 |
| sadness | 116 | 12 | 4 | 33 | 23 | 0 | 20 |
| joy | 125 | 25 | 1 | 1 | 225 | 3 | 22 |
| disgust | 34 | 7 | 0 | 2 | 4 | 9 | 12 |
| anger | 121 | 41 | 3 | 6 | 63 | 4 | 107 |

### Confusion matrix of text_best_k (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1102 | 50 | 2 | 16 | 61 | 5 | 20 |
| surprise | 56 | 155 | 2 | 2 | 36 | 4 | 26 |
| fear | 18 | 7 | 7 | 7 | 6 | 0 | 5 |
| sadness | 116 | 12 | 4 | 33 | 23 | 0 | 20 |
| joy | 125 | 25 | 1 | 1 | 225 | 3 | 22 |
| disgust | 34 | 7 | 0 | 2 | 4 | 9 | 12 |
| anger | 121 | 41 | 3 | 6 | 63 | 4 | 107 |

### Confusion matrix of text_runtime_regime (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1107 | 48 | 2 | 15 | 61 | 5 | 18 |
| surprise | 56 | 155 | 2 | 2 | 38 | 3 | 25 |
| fear | 18 | 7 | 7 | 7 | 6 | 0 | 5 |
| sadness | 118 | 11 | 4 | 34 | 22 | 0 | 19 |
| joy | 126 | 25 | 1 | 1 | 226 | 2 | 21 |
| disgust | 35 | 6 | 0 | 4 | 5 | 7 | 11 |
| anger | 122 | 43 | 3 | 6 | 63 | 3 | 105 |

### Confusion matrix of audio_raw_tag (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 938 | 61 | 0 | 20 | 130 | 18 | 89 |
| surprise | 119 | 78 | 0 | 3 | 48 | 6 | 27 |
| fear | 28 | 1 | 0 | 0 | 9 | 1 | 11 |
| sadness | 121 | 17 | 0 | 22 | 19 | 5 | 24 |
| joy | 173 | 30 | 0 | 5 | 156 | 4 | 34 |
| disgust | 34 | 2 | 0 | 1 | 12 | 1 | 18 |
| anger | 122 | 29 | 0 | 1 | 41 | 7 | 145 |

Abstain rate (OTHER + EMO_UNKNOWN above half the mass): 97.51 %.

### Confusion matrix of audio_cal_A (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1098 | 38 | 0 | 10 | 60 | 1 | 49 |
| surprise | 158 | 66 | 0 | 0 | 33 | 0 | 24 |
| fear | 35 | 1 | 0 | 1 | 5 | 0 | 8 |
| sadness | 139 | 11 | 0 | 26 | 14 | 0 | 18 |
| joy | 253 | 24 | 0 | 2 | 107 | 0 | 16 |
| disgust | 43 | 1 | 0 | 2 | 10 | 0 | 12 |
| anger | 183 | 22 | 0 | 3 | 32 | 0 | 105 |

### Confusion matrix of audio_cal_B (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 888 | 78 | 33 | 49 | 115 | 24 | 69 |
| surprise | 104 | 102 | 3 | 10 | 33 | 9 | 20 |
| fear | 23 | 5 | 4 | 5 | 7 | 0 | 6 |
| sadness | 93 | 16 | 8 | 42 | 20 | 7 | 22 |
| joy | 190 | 43 | 4 | 12 | 126 | 5 | 22 |
| disgust | 24 | 8 | 3 | 3 | 7 | 4 | 19 |
| anger | 140 | 41 | 12 | 12 | 47 | 12 | 81 |

### Confusion matrix of fused_stacked (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1105 | 41 | 0 | 14 | 59 | 3 | 34 |
| surprise | 61 | 151 | 1 | 0 | 41 | 1 | 26 |
| fear | 25 | 3 | 0 | 7 | 4 | 0 | 11 |
| sadness | 112 | 10 | 4 | 39 | 15 | 0 | 28 |
| joy | 134 | 24 | 0 | 3 | 224 | 0 | 17 |
| disgust | 35 | 4 | 0 | 4 | 5 | 4 | 16 |
| anger | 110 | 36 | 1 | 2 | 49 | 0 | 147 |

### Confusion matrix of fused_poe (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1166 | 23 | 0 | 6 | 48 | 1 | 12 |
| surprise | 90 | 133 | 0 | 0 | 39 | 0 | 19 |
| fear | 30 | 4 | 0 | 4 | 4 | 0 | 8 |
| sadness | 144 | 8 | 0 | 23 | 15 | 0 | 18 |
| joy | 164 | 19 | 0 | 1 | 205 | 0 | 13 |
| disgust | 40 | 4 | 0 | 1 | 6 | 2 | 15 |
| anger | 147 | 32 | 0 | 0 | 47 | 0 | 119 |

### Confusion matrix of fused_poe_uncalibrated (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1144 | 25 | 0 | 4 | 57 | 1 | 25 |
| surprise | 83 | 137 | 0 | 0 | 40 | 0 | 21 |
| fear | 30 | 5 | 0 | 3 | 4 | 0 | 8 |
| sadness | 135 | 10 | 0 | 21 | 17 | 0 | 25 |
| joy | 148 | 20 | 0 | 1 | 219 | 0 | 14 |
| disgust | 39 | 5 | 0 | 1 | 7 | 3 | 13 |
| anger | 126 | 36 | 0 | 0 | 49 | 1 | 133 |

### Confusion matrix of ablation_no_events (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1101 | 42 | 0 | 13 | 62 | 1 | 37 |
| surprise | 60 | 153 | 1 | 0 | 41 | 0 | 26 |
| fear | 24 | 2 | 0 | 8 | 4 | 0 | 12 |
| sadness | 112 | 10 | 2 | 41 | 16 | 0 | 27 |
| joy | 131 | 24 | 0 | 3 | 228 | 0 | 16 |
| disgust | 35 | 4 | 0 | 3 | 5 | 4 | 17 |
| anger | 110 | 35 | 0 | 4 | 50 | 0 | 146 |

### Confusion matrix of ablation_no_prosody (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1107 | 39 | 0 | 15 | 58 | 3 | 34 |
| surprise | 59 | 152 | 1 | 1 | 42 | 1 | 25 |
| fear | 25 | 2 | 0 | 7 | 4 | 0 | 12 |
| sadness | 114 | 10 | 3 | 39 | 15 | 0 | 27 |
| joy | 135 | 25 | 0 | 4 | 223 | 0 | 15 |
| disgust | 36 | 4 | 0 | 3 | 5 | 4 | 16 |
| anger | 110 | 35 | 1 | 2 | 48 | 0 | 149 |

### Confusion matrix of ablation_no_events_no_prosody (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1099 | 42 | 0 | 13 | 62 | 1 | 39 |
| surprise | 59 | 152 | 1 | 2 | 41 | 0 | 26 |
| fear | 24 | 2 | 0 | 8 | 3 | 0 | 13 |
| sadness | 115 | 10 | 1 | 40 | 16 | 0 | 26 |
| joy | 132 | 24 | 0 | 5 | 225 | 0 | 16 |
| disgust | 34 | 5 | 0 | 3 | 5 | 4 | 17 |
| anger | 107 | 35 | 1 | 3 | 49 | 0 | 150 |

### Confusion matrix of live_asr_text_only (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1013 | 13 | 1 | 18 | 136 | 8 | 67 |
| surprise | 178 | 35 | 0 | 2 | 40 | 3 | 23 |
| fear | 32 | 2 | 3 | 0 | 9 | 0 | 4 |
| sadness | 121 | 6 | 0 | 21 | 28 | 1 | 31 |
| joy | 249 | 7 | 4 | 2 | 124 | 3 | 13 |
| disgust | 45 | 2 | 0 | 0 | 8 | 2 | 11 |
| anger | 220 | 9 | 6 | 8 | 40 | 5 | 57 |

### Confusion matrix of live_asr_text (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1064 | 16 | 0 | 20 | 99 | 0 | 57 |
| surprise | 180 | 35 | 0 | 2 | 31 | 1 | 32 |
| fear | 34 | 0 | 2 | 3 | 5 | 0 | 6 |
| sadness | 118 | 3 | 0 | 37 | 20 | 0 | 30 |
| joy | 258 | 5 | 0 | 2 | 127 | 2 | 8 |
| disgust | 40 | 1 | 0 | 2 | 7 | 0 | 18 |
| anger | 201 | 10 | 1 | 4 | 22 | 1 | 106 |

### Confusion matrix of disagreement_subset (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 125 | 18 | 0 | 3 | 10 | 3 | 11 |
| surprise | 12 | 48 | 0 | 0 | 8 | 1 | 6 |
| fear | 5 | 3 | 0 | 2 | 1 | 0 | 6 |
| sadness | 21 | 3 | 3 | 7 | 5 | 0 | 7 |
| joy | 25 | 9 | 0 | 0 | 41 | 0 | 6 |
| disgust | 7 | 1 | 0 | 1 | 2 | 3 | 0 |
| anger | 23 | 12 | 0 | 1 | 18 | 0 | 32 |

### Confusion matrix of early_commit_audio_2s (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1145 | 37 | 0 | 4 | 48 | 0 | 22 |
| surprise | 168 | 68 | 0 | 0 | 28 | 0 | 17 |
| fear | 37 | 2 | 0 | 1 | 6 | 0 | 4 |
| sadness | 159 | 19 | 0 | 14 | 8 | 0 | 8 |
| joy | 273 | 33 | 0 | 2 | 82 | 0 | 12 |
| disgust | 52 | 3 | 0 | 0 | 5 | 0 | 8 |
| anger | 209 | 30 | 0 | 1 | 26 | 0 | 79 |

### Confusion matrix of early_commit_fused_2s (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1119 | 21 | 1 | 13 | 73 | 0 | 29 |
| surprise | 191 | 44 | 1 | 1 | 22 | 1 | 21 |
| fear | 38 | 1 | 2 | 2 | 5 | 0 | 2 |
| sadness | 148 | 6 | 1 | 21 | 15 | 0 | 17 |
| joy | 303 | 8 | 0 | 1 | 87 | 0 | 3 |
| disgust | 54 | 1 | 0 | 2 | 2 | 0 | 9 |
| anger | 240 | 10 | 0 | 3 | 17 | 2 | 73 |

### Confusion matrix of early_commit_audio_4s (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1113 | 41 | 0 | 10 | 48 | 0 | 44 |
| surprise | 161 | 65 | 0 | 0 | 34 | 0 | 21 |
| fear | 33 | 2 | 0 | 1 | 6 | 0 | 8 |
| sadness | 151 | 14 | 0 | 16 | 11 | 0 | 16 |
| joy | 262 | 25 | 0 | 2 | 102 | 0 | 11 |
| disgust | 47 | 1 | 0 | 1 | 7 | 0 | 12 |
| anger | 188 | 21 | 0 | 2 | 30 | 0 | 104 |

### Confusion matrix of early_commit_fused_4s (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1086 | 17 | 0 | 15 | 89 | 0 | 49 |
| surprise | 181 | 40 | 1 | 2 | 30 | 0 | 27 |
| fear | 34 | 1 | 2 | 2 | 4 | 0 | 7 |
| sadness | 132 | 4 | 0 | 26 | 21 | 0 | 25 |
| joy | 269 | 5 | 0 | 1 | 117 | 2 | 8 |
| disgust | 44 | 1 | 0 | 1 | 3 | 0 | 19 |
| anger | 220 | 10 | 2 | 4 | 18 | 0 | 91 |

### Confusion matrix of sentiment_text_only (rows true, columns predicted)

| true \ pred | neutral | positive | negative |
| --- | --- | --- | --- |
| neutral | 1004 | 87 | 165 |
| positive | 134 | 287 | 100 |
| negative | 253 | 130 | 450 |

### Confusion matrix of sentiment_fused (rows true, columns predicted)

| true \ pred | neutral | positive | negative |
| --- | --- | --- | --- |
| neutral | 997 | 66 | 193 |
| positive | 127 | 275 | 119 |
| negative | 208 | 84 | 541 |

## Ablations

Each ablation refits calibrator A without the named inputs, and a fusion head over its posterior, with the same folds, grids and dev selection as the deployed heads (`tools/train_heads.py --ablation`). Without events the fusion head also loses p_laughter. Audio only is the ablated calibrator by itself.

| condition | calibrator inputs | fusion inputs | fused weighted F1 | fused accuracy | audio-only weighted F1 | audio-only accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| deployed (fused_stacked, audio_cal_A) | 24 | 19 | 60.77 | 63.98 | 48.02 | 53.72 |
| ablation_no_events | 13 | 18 | 60.91 | 64.10 | 48.23 | 54.02 |
| ablation_no_prosody | 20 | 19 | 60.91 | 64.14 | 47.62 | 53.56 |
| ablation_no_events_no_prosody | 9 | 18 | 60.82 | 63.98 | 47.55 | 53.49 |

## Live ASR text

The SenseVoice transcripts of test have a corpus WER of 0.348488 over 2610 utterances (`cache/wer_test.json`). The rows apply the service's gate on a final state. 35 transcripts are empty (asr_failed) and 38 have an ASR confidence below 0.3 (low_asr_conf). These 73 rows get a uniform posterior and the label neutral, as `Service._run_final` publishes them. The without-gate columns score every row through the heads, with the text expert's empty-text embedding for the empty transcripts. The heads are the deployed ones, trained on gold text. The gold columns score the same heads on the gold text of the same rows.

| condition | weighted F1 | accuracy | without gate weighted F1 | without gate accuracy | gold text weighted F1 | gold text accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| live_asr_text_only | 42.61 | 48.08 | 42.68 | 48.12 | 59.62 | 62.76 |
| live_asr_text | 47.05 | 52.53 | 47.12 | 52.57 | 60.77 | 63.98 |

## Disagreement subset

489 of 2610 test utterances (18.74 %) have a JSD above 0.3831 between the text posterior and the raw SenseVoice posterior. The JSD uses the service's discrepancy function (`meld_rt.discrepancy`) and the dev threshold (`thresholds.jsd_flag` in `heads/manifest.json`). The text posterior is the gold-text one with the full-dialogue window. The service computes it on the ASR transcript with its own stream context, and only for a transcript that passes its ASR gate.

| rows | n | text weighted F1 | text accuracy | audio weighted F1 | audio accuracy | fused weighted F1 | fused accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subset | 489 | 51.44 | 53.37 | 23.44 | 26.79 | 49.62 | 52.35 |
| complement | 2121 | 61.34 | 64.92 | 53.94 | 59.92 | 63.28 | 66.67 |

## Early commit

These rows are offline prefix experiments, not the service's provisional states. Each row scores the first 2 s or 4 s of every clip with the full calibrator inputs of the prefix, including its cached prosody. A clip that is not longer than the prefix keeps its full-clip values, as the cache stores them. The fused rows take the text posterior from the prefix transcript. The service runs no text expert on a provisional state. The full-clip reference is the same pipeline on the whole clip: the deployed calibrator for audio, and the stacked fusion with the full transcript for fused, without the service's ASR gate (the without-gate columns of live_asr_text). Agreement is the share of clips whose prefix label equals the full-clip label. The cut columns cover only the clips longer than the prefix.

| condition | clips with full-clip values | weighted F1 | full clip weighted F1 | accuracy | full clip accuracy | agreement | cut clips | cut weighted F1 | cut full clip weighted F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_commit_audio_2s | 925 | 45.86 | 48.02 | 53.18 | 53.72 | 89.46 | 1685 | 43.15 | 46.34 |
| early_commit_fused_2s | 925 | 44.35 | 47.12 | 51.57 | 52.57 | 85.82 | 1685 | 43.09 | 46.98 |
| early_commit_audio_4s | 1920 | 47.34 | 48.02 | 53.64 | 53.72 | 96.63 | 690 | 42.65 | 45.06 |
| early_commit_fused_4s | 1920 | 46.06 | 47.12 | 52.18 | 52.57 | 94.48 | 690 | 42.62 | 45.99 |

A provisional state runs the deployed calibrator with the prosody set to zero except the speech rate (`Service._run_provisional`). The next table scores the audio rows again with those inputs on the same prefixes.

| condition | weighted F1 | accuracy | cut weighted F1 | cut accuracy |
| --- | --- | --- | --- | --- |
| early_commit_audio_2s, provisional inputs | 46.49 | 52.49 | 43.22 | 49.97 |
| early_commit_audio_4s, provisional inputs | 48.03 | 53.41 | 42.97 | 49.13 |

## Definitions

- Runtime-regime window: the previous utterances are restricted to the same speaker within the dialogue, the same-speaker flags are 1, and the turn index counts that speaker's earlier utterances (what a single user in front of the robot has said so far). The head is the one trained on the full-dialogue window.
- Majority floor: every test utterance gets the train majority class. Its posterior is the train class prior, which is what the ECE column scores.
- Rows without a clip: the audio and fused conditions leave out test rows that have no cached clip. The note says how many, and `results/metrics.json` gives the text head's weighted F1 on the same rows (`text_best_k_same_rows_weighted_f1`).
