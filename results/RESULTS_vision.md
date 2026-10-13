# Three inputs on MELD test

`python tools/evaluate_vision.py --cache cache/ --heads heads/ --audio audio/ --meld meld/ --out results/RESULTS_vision.md results/metrics_vision.json`

## Inputs

| file | bytes | sha256 |
| --- | --- | --- |
| `cache/video_test.npz` | 104034302 | `87c64ec30e51e95259765bef1aee06858df6b3d0da06e400b4d625855311f561` |
| `cache/envelope_test.npz` | 4511090 | `9214e621fa573c37c63df6c793d56196782666286fd5646365df24c9a76cfb18` |
| `cache/text_test.npy` | 8018048 | `bd454c8e0f496e649b8e9f61a1acd098ffe478c51184a30e28185c71135e3ce7` |
| `cache/audio_test.npz` | 30360152 | `a9e22761afa2096ec65841c9f8adc9af541a05e2517e8c0b9e7a1337c58007e2` |
| `cache/text_asr_test.npy` | 8018048 | `241d14d7567d99dc4d27f94a43e7278581ab0221460b113e8d9453cd74070f8b` |
| `meld/test_sent_emo.csv` | 294526 | `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888` |
| `heads/text_ctx.npz` | 808726 | `916806bf4f0a0055e989c6e73a289441378c719b9a4fb68c0628ad652d97dd92` |
| `heads/audio_cal.npz` | 3622 | `49be433bf1a2b0b7cb635dda743c4416af0ed8958fc28235f84f5f3a31509cfc` |
| `heads/fusion.npz` | 3192 | `19fb219aa787d75611ec39d65067bc8afdd982bc807ea9b4601729e8c95c154d` |
| `heads/vision_cal.npz` | 22018 | `c06849f97c476bbb94eb7b10825ef874227591ef52e0febfd887f3916f022d7a` |
| `heads/fusion3.npz` | 3810 | `abd9babc406418a87c6a3d57432d313656756a2d95992059cea362e83537d15f` |
| `heads/sentiment3.npz` | 3202 | `62e099844deb6a13620058d8a21876b49b142671010b6980312efe8268d432a0` |

| setting | value |
| --- | --- |
| speaker rule | asd |
| calibrator | B |
| jsd_face_flag | 0.5729 |
| vision-mode thresholds | low_conf 0.56, high_conf 0.73 (dev) |
| face coverage, test rows | 99.46 |

## Emotion (7 classes)

| condition | n | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | face coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vision_raw_fer | 2610 | 32.22 | [29.77, 34.89] | 15.92 | 34.44 | 36.63 | 99.46 |
| vision_cal | 2610 | 34.37 | [31.83, 37.32] | 12.70 | 47.05 | 8.32 | 99.46 |
| text_best_k | 2610 | 59.62 | [57.38, 62.03] | 41.22 | 62.76 | 3.83 |  |
| fused_stacked | 2610 | 60.77 | [58.65, 63.00] | 39.24 | 63.98 | 1.52 |  |
| fused3_stacked | 2610 | 61.56 | [59.45, 63.79] | 40.50 | 64.60 | 2.03 | 99.46 |
| live_asr_text | 2610 | 47.05 | [44.53, 49.57] | 27.50 | 52.53 | 9.71 |  |
| live_asr_fused3 | 2610 | 47.96 | [45.60, 50.40] | 28.82 | 53.18 | 9.13 | 99.46 |
| text_best_k_face_rows | 2596 | 59.58 | [57.28, 61.95] | 41.28 | 62.71 | 3.86 |  |
| fused_stacked_face_rows | 2596 | 60.74 | [58.61, 62.93] | 39.26 | 63.94 | 1.50 |  |
| fused3_stacked_face_rows | 2596 | 61.54 | [59.46, 63.67] | 40.54 | 64.56 | 2.07 |  |
| live_asr_text_face_rows | 2596 | 47.09 | [44.61, 49.61] | 27.58 | 52.54 | 9.69 |  |
| live_asr_fused3_face_rows | 2596 | 47.98 | [45.71, 50.40] | 28.88 | 53.16 | 9.11 |  |
| face_disagreement_subset | 557 | 64.04 | [59.64, 68.28] | 47.10 | 66.79 | 3.03 |  |

## Sentiment (3 classes)

| condition | n | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | face coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment3 | 2610 | 69.03 | [67.05, 70.88] | 66.44 | 69.31 | 2.85 |  |

## Paired differences (weighted F1, a minus b)

Paired bootstrap over test dialogues, 1000 resamples, seed 0.

| comparison | a | b | rows | n | difference | 95 % CI | excludes zero | share at or below zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gold_all | fused3_stacked | fused_stacked | all | 2610 | 0.80 | [0.20, 1.48] | yes | 0.60 |
| live_all | live_asr_fused3 | live_asr_text | all | 2610 | 0.91 | [0.31, 1.52] | yes | 0.10 |
| gold_face_rows | fused3_stacked | fused_stacked | face | 2596 | 0.80 | [0.20, 1.49] | yes | 0.60 |
| live_face_rows | live_asr_fused3 | live_asr_text | face | 2596 | 0.89 | [0.28, 1.50] | yes | 0.10 |
| gold_face_disagreement | fused3_stacked | fused_stacked | disagree | 557 | 1.17 | [-0.29, 2.60] | no | 5.80 |
| gold_face_disagreement_vs_text | fused3_stacked | text_best_k | disagree | 557 | -0.84 | [-3.71, 2.20] | no | 71.40 |

| claim rule | interval | face adds |
| --- | --- | --- |
| the face adds to text and audio only if the paired 95 % interval of fused3_stacked minus fused_stacked (weighted f1, gold-text test, all clips with audio) lies above zero | [0.20, 1.48] | yes |

## Face-disagreement subset

jsd_face_flag 0.5729, 557 of 2596 clips with a face.

| rows | n | text weighted F1 | text accuracy | two-way weighted F1 | two-way accuracy | fused3 weighted F1 | fused3 accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subset | 557 | 64.88 | 66.97 | 62.87 | 65.89 | 64.04 | 66.79 |
| complement | 2039 | 57.94 | 61.55 | 59.98 | 63.41 | 60.63 | 63.95 |

## Per-class F1

| condition | n | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vision_raw_fer | 2610 | 52.53 | 6.85 | 0.00 | 17.25 | 28.08 | 3.56 | 3.17 |
| vision_cal | 2610 | 64.38 | 3.88 | 0.00 | 0.00 | 11.22 | 0.00 | 9.38 |
| text_best_k | 2610 | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| fused_stacked | 2610 | 77.87 | 54.91 | 0.00 | 28.16 | 56.07 | 10.53 | 47.12 |
| fused3_stacked | 2610 | 78.12 | 54.44 | 3.64 | 30.18 | 57.36 | 10.39 | 49.39 |
| live_asr_text | 2610 | 67.53 | 19.94 | 7.55 | 26.62 | 35.62 | 0.00 | 35.22 |
| live_asr_fused3 | 2610 | 67.73 | 20.29 | 7.41 | 29.17 | 37.32 | 2.78 | 37.06 |
| text_best_k_face_rows | 2596 | 77.86 | 53.82 | 20.59 | 24.09 | 54.72 | 19.35 | 38.49 |
| fused_stacked_face_rows | 2596 | 77.80 | 55.11 | 0.00 | 28.26 | 55.92 | 10.53 | 47.19 |
| fused3_stacked_face_rows | 2596 | 78.05 | 54.65 | 3.70 | 30.28 | 57.22 | 10.39 | 49.46 |
| live_asr_text_face_rows | 2596 | 67.54 | 20.06 | 7.69 | 26.81 | 35.72 | 0.00 | 35.22 |
| live_asr_fused3_face_rows | 2596 | 67.70 | 20.41 | 7.55 | 29.27 | 37.43 | 2.78 | 37.06 |
| face_disagreement_subset | 557 | 79.27 | 68.31 | 13.33 | 15.00 | 62.65 | 32.00 | 59.14 |

## Speaker rules on dev

| rule | dev weighted F1, all rows | dev weighted F1, face rows | fallback rate, dev face rows | chosen |
| --- | --- | --- | --- | --- |
| largest | 25.19 | 25.24 | 0.00 |  |
| present | 25.57 | 25.62 | 0.00 |  |
| asd | 25.82 | 25.87 | 4.08 | yes |
| mouth | 25.53 | 25.58 | 3.26 |  |

| dev | weighted F1 | macro F1 | accuracy |
| --- | --- | --- | --- |
| calibrator A | 25.82 | 9.04 | 42.65 |
| calibrator B | 30.20 | 14.02 | 42.20 |
| two-way fusion | 57.95 | 38.86 | 61.37 |
| fusion3 | 59.53 | 40.53 | 62.64 |
| sentiment3 | 67.77 | 65.83 | 68.23 |

### Confusion matrix of vision_raw_fer (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 676 | 59 | 0 | 84 | 354 | 70 | 13 |
| surprise | 129 | 14 | 0 | 22 | 82 | 29 | 5 |
| fear | 28 | 1 | 0 | 6 | 11 | 2 | 2 |
| sadness | 105 | 9 | 0 | 37 | 46 | 9 | 2 |
| joy | 181 | 12 | 0 | 27 | 162 | 16 | 4 |
| disgust | 27 | 4 | 0 | 11 | 20 | 4 | 2 |
| anger | 172 | 29 | 0 | 34 | 77 | 27 | 6 |

### Confusion matrix of vision_cal (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1174 | 11 | 1 | 5 | 46 | 1 | 18 |
| surprise | 241 | 6 | 0 | 1 | 21 | 1 | 11 |
| fear | 46 | 0 | 0 | 1 | 3 | 0 | 0 |
| sadness | 195 | 3 | 1 | 0 | 4 | 0 | 5 |
| joy | 363 | 3 | 0 | 2 | 29 | 0 | 5 |
| disgust | 64 | 0 | 0 | 0 | 2 | 0 | 2 |
| anger | 308 | 5 | 1 | 2 | 10 | 0 | 19 |

### Confusion matrix of fused3_stacked (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1107 | 38 | 0 | 17 | 53 | 3 | 38 |
| surprise | 62 | 144 | 0 | 0 | 41 | 1 | 33 |
| fear | 25 | 1 | 1 | 7 | 4 | 0 | 12 |
| sadness | 107 | 9 | 3 | 43 | 15 | 0 | 31 |
| joy | 136 | 22 | 0 | 2 | 226 | 0 | 16 |
| disgust | 35 | 4 | 0 | 4 | 5 | 4 | 16 |
| anger | 106 | 30 | 1 | 4 | 42 | 1 | 161 |

### Confusion matrix of live_asr_fused3 (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1061 | 15 | 1 | 21 | 92 | 0 | 66 |
| surprise | 178 | 35 | 0 | 3 | 30 | 2 | 33 |
| fear | 34 | 0 | 2 | 3 | 5 | 0 | 6 |
| sadness | 114 | 2 | 0 | 42 | 19 | 0 | 31 |
| joy | 255 | 4 | 0 | 3 | 131 | 0 | 9 |
| disgust | 39 | 1 | 0 | 2 | 5 | 1 | 20 |
| anger | 196 | 7 | 1 | 6 | 18 | 1 | 116 |

### Confusion matrix of fused3_stacked_face_rows (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1099 | 38 | 0 | 17 | 53 | 3 | 38 |
| surprise | 62 | 144 | 0 | 0 | 40 | 1 | 32 |
| fear | 24 | 1 | 1 | 7 | 4 | 0 | 12 |
| sadness | 106 | 9 | 3 | 43 | 15 | 0 | 31 |
| joy | 136 | 22 | 0 | 2 | 224 | 0 | 16 |
| disgust | 35 | 4 | 0 | 4 | 5 | 4 | 16 |
| anger | 106 | 30 | 1 | 4 | 42 | 1 | 161 |

### Confusion matrix of live_asr_fused3_face_rows (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 1053 | 15 | 1 | 21 | 92 | 0 | 66 |
| surprise | 176 | 35 | 0 | 3 | 30 | 2 | 33 |
| fear | 33 | 0 | 2 | 3 | 5 | 0 | 6 |
| sadness | 113 | 2 | 0 | 42 | 19 | 0 | 31 |
| joy | 253 | 4 | 0 | 3 | 131 | 0 | 9 |
| disgust | 39 | 1 | 0 | 2 | 5 | 1 | 20 |
| anger | 196 | 7 | 1 | 6 | 18 | 1 | 116 |

### Confusion matrix of face_disagreement_subset (rows true, columns predicted)

| true \ pred | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 174 | 9 | 0 | 1 | 5 | 0 | 6 |
| surprise | 18 | 83 | 0 | 0 | 9 | 0 | 7 |
| fear | 2 | 1 | 1 | 2 | 0 | 0 | 7 |
| sadness | 12 | 4 | 1 | 3 | 2 | 0 | 10 |
| joy | 23 | 11 | 0 | 0 | 52 | 0 | 2 |
| disgust | 4 | 2 | 0 | 1 | 2 | 4 | 8 |
| anger | 11 | 16 | 0 | 1 | 8 | 0 | 55 |

### Confusion matrix of sentiment3 (rows true, columns predicted)

| true \ pred | neutral | positive | negative |
| --- | --- | --- | --- |
| neutral | 980 | 62 | 214 |
| positive | 130 | 267 | 124 |
| negative | 203 | 68 | 562 |
