# Results on MELD test

Written by `python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json`. Every number below was computed by that command from the inputs listed. Model selection, temperature and thresholds were chosen on dev by `tools/train_heads.py`, and the test split was read once per condition. Weighted F1 is the primary metric, as in the published rows. The CIs are 95 % percentile bootstrap intervals over test dialogues (1000 resamples, seed 0). ECE uses 10 equal-width confidence bins. All metrics are in percent.

## Inputs

| file | bytes | sha256 |
| --- | --- | --- |
| `cache/text_test.npy` | 8018048 | `bd454c8e0f496e649b8e9f61a1acd098ffe478c51184a30e28185c71135e3ce7` |
| `meld/test_sent_emo.csv` | 294526 | `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888` |
| `heads/sentiment.npz` | 18328 | `65c6614faa7a553bf62897048bb807eb52173e4ec873a036d50903b95225b78d` |
| `heads/text_ctx.npz` | 808726 | `916806bf4f0a0055e989c6e73a289441378c719b9a4fb68c0628ad652d97dd92` |
| `heads/text_ctx_k0.npz` | 808726 | `916806bf4f0a0055e989c6e73a289441378c719b9a4fb68c0628ad652d97dd92` |

Deployed text head, from `heads/manifest.json`: k = 0, class weights `none`, temperature 1.2228, thresholds `{"low_conf": 0.39, "high_conf": 0.73, "jsd_flag": null}`, text_only = True, scikit-learn 1.9.1.

## Emotion (7 classes)

| condition | status | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | done | 31.27 | [28.62, 34.31] | 9.28 | 48.12 | 0.97 | train prior as posterior; label = neutral |
| text_k0 | done | 59.62 | [57.38, 62.03] | 41.22 | 62.76 | 3.83 | class weights none, T=1.223 |
| text_best_k | done | 59.62 | [57.38, 62.03] | 41.22 | 62.76 | 3.83 | k=0, class weights none, T=1.223 |
| text_runtime_regime | done | 59.61 | [57.36, 62.08] | 40.82 | 62.87 | 3.54 | same head as text_best_k, same-speaker window (k=0: differs from text_best_k only through the turn-index flags) |
| audio_raw_tag | pending (needs MELD audio) | | | | | | 3. audio only, raw sensevoice tag: argmax over the seven named tokens |
| audio_cal_A | pending (needs MELD audio) | | | | | | 4. audio only, AudioCalibrator A (slots + events + prosody) |
| audio_cal_B | pending (needs MELD audio) | | | | | | 4. audio only, AudioCalibrator B (+ encoder states) |
| fused_stacked | pending (needs MELD audio) | | | | | | 5. fused, stacked logistic regression (deployed) |
| fused_poe | pending (needs MELD audio) | | | | | | 5. fused, product of experts with the dev-chosen weight |
| fused_poe_uncalibrated | pending (needs MELD audio) | | | | | | 5. fused, fixed poe_uncalibrated fallback (w = 0.5) |
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
| sentiment_text_only | done | 66.24 | [64.41, 68.20] | 63.33 | 66.70 | 2.04 | k=0, C=0.1, T=1.420 |
| sentiment_fused | pending (needs MELD audio) | | | | | | 10. sentiment, fused head over the 19 fusion features |

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

### Confusion matrix of sentiment_text_only (rows true, columns predicted)

| true \ pred | neutral | positive | negative |
| --- | --- | --- | --- |
| neutral | 1004 | 87 | 165 |
| positive | 134 | 287 | 100 |
| negative | 253 | 130 | 450 |

## Definitions

- Runtime-regime window: the previous utterances are restricted to the same speaker within the dialogue, the same-speaker flags are 1, and the turn index counts that speaker's earlier utterances (what a single user in front of the robot has said so far). The head is the one trained on the full-dialogue window.
- Majority floor: every test utterance gets the train majority class. Its posterior is the train class prior, which is what the ECE column scores.
- Rows without a clip: the audio and fused conditions leave out test rows that have no cached clip. The note says how many, and `results/metrics.json` gives the text head's weighted F1 on the same rows (`text_best_k_same_rows_weighted_f1`).
- Pending rows: the audio and fused rows wait for MELD audio (`tools/extract_audio.py`, `tools/cache_audio_features.py`, then `tools/train_heads.py` without `--text-only`). The ablation, live ASR, disagreement and early-commit rows also need code that this script does not have yet.
