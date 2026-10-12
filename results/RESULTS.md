# Results on MELD test

Written by `python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json`. Every number below was computed by that command from the inputs listed. Model selection, temperature and thresholds were chosen on dev by `tools/train_heads.py`, and the test split was read once per condition. Weighted F1 is the primary metric, as in the published rows. The CIs are 95 % percentile bootstrap intervals over test dialogues (1000 resamples, seed 0). ECE uses 10 equal-width confidence bins. All metrics are in percent.

## Inputs

| file | bytes | sha256 |
| --- | --- | --- |
| `cache/text_test.npy` | 8018048 | `bd454c8e0f496e649b8e9f61a1acd098ffe478c51184a30e28185c71135e3ce7` |
| `meld/test_sent_emo.csv` | 294526 | `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888` |
| `cache/audio_test.npz` | 30360152 | `a9e22761afa2096ec65841c9f8adc9af541a05e2517e8c0b9e7a1337c58007e2` |
| `heads/audio_cal.npz` | 3622 | `49be433bf1a2b0b7cb635dda743c4416af0ed8958fc28235f84f5f3a31509cfc` |
| `heads/audio_cal_A.npz` | 3622 | `49be433bf1a2b0b7cb635dda743c4416af0ed8958fc28235f84f5f3a31509cfc` |
| `heads/audio_cal_B.npz` | 40486 | `4cb548fa197a50e7d83241e03af7cc64237bcfa406e69dae18850a12f3c9c2d4` |
| `heads/fusion.npz` | 3192 | `19fb219aa787d75611ec39d65067bc8afdd982bc807ea9b4601729e8c95c154d` |
| `heads/sentiment.npz` | 3010 | `4cd0467fe75f6b97f956994d8b099c63a256f52f68a36fd105766d7301d0abbf` |
| `heads/text_ctx.npz` | 808726 | `916806bf4f0a0055e989c6e73a289441378c719b9a4fb68c0628ad652d97dd92` |
| `heads/text_ctx_k0.npz` | 808726 | `916806bf4f0a0055e989c6e73a289441378c719b9a4fb68c0628ad652d97dd92` |

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
| ablation_no_events | pending (needs MELD audio); also needs the ablation heads of experiment 02 | | | | | | 6. fused without event features |
| ablation_no_prosody | pending (needs MELD audio); also needs the ablation heads of experiment 02 | | | | | | 6. fused without prosody |
| ablation_no_events_no_prosody | pending (needs MELD audio); also needs the ablation heads of experiment 02 | | | | | | 6. fused without events and prosody |
| live_asr_text | pending (needs MELD audio); also needs cache/text_asr_test.npy (embed_text.py on the sensevoice transcripts) | | | | | | 7. fused with sensevoice transcripts instead of gold text, plus WER |
| disagreement_subset | pending (needs MELD audio); needs the jsd threshold chosen on dev with audio | | | | | | 8. utterances with jsd above the dev threshold: size, per-branch accuracy, fused gain |
| early_commit | pending (needs MELD audio); needs the _p2/_p4 prefix features of cache/audio_test.npz | | | | | | 9. audio-only and fused label on 2 s and 4 s prefixes versus the full clip |

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
| sentiment_text_only | pending (run tools/train_heads.py --text-only first) | | | | | | 10. sentiment, text-only head (logistic regression over the text context features) |
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

### Confusion matrix of sentiment_fused (rows true, columns predicted)

| true \ pred | neutral | positive | negative |
| --- | --- | --- | --- |
| neutral | 997 | 66 | 193 |
| positive | 127 | 275 | 119 |
| negative | 208 | 84 | 541 |

## Definitions

- Runtime-regime window: the previous utterances are restricted to the same speaker within the dialogue, the same-speaker flags are 1, and the turn index counts that speaker's earlier utterances (what a single user in front of the robot has said so far). The head is the one trained on the full-dialogue window.
- Majority floor: every test utterance gets the train majority class. Its posterior is the train class prior, which is what the ECE column scores.
- Rows without a clip: the audio and fused conditions leave out test rows that have no cached clip. The note says how many, and `results/metrics.json` gives the text head's weighted F1 on the same rows (`text_best_k_same_rows_weighted_f1`).
- Pending rows: the audio and fused rows wait for MELD audio (`tools/extract_audio.py`, `tools/cache_audio_features.py`, then `tools/train_heads.py` without `--text-only`). The ablation, live ASR, disagreement and early-commit rows also need code that this script does not have yet.
