# Results

Rendered by `python tools/render_results.py` from `results/metrics.json` (written by `python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json`) and `heads/manifest.json` (written by `python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml` and `python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml --ablation`). The full report with the confusion matrices is [results/RESULTS.md](results/RESULTS.md), and the dev grid of the text head is in [results/01_text_only.md](results/01_text_only.md). Model selection, temperature and thresholds were chosen on dev. The test split was read once per condition. Weighted F1 is the primary metric, as in the published rows. The intervals are 95 % percentile bootstrap intervals over test dialogues (1000 resamples, seed 0). ECE uses 10 equal-width confidence bins. All metrics are in percent.

## Observations

The deployed text head scores 59.62 weighted F1 on test (95 % CI [57.38, 62.03]), against 31.27 for the majority floor. The deployed stacked fusion scores 60.77 [58.65, 63.00], 1.15 points above the text head, and the two intervals overlap, with the text head's score inside the fusion's interval. On the 489 test utterances whose JSD is above the dev threshold (18.74 %), the fusion scores 49.62 against 51.44 for the text head alone, so it does not help where the words and the voice disagree. On SenseVoice transcripts (corpus WER 0.348488), the text head scores 42.61 instead of 59.62 on gold text, and the fusion 47.05 instead of 60.77. Without the event and prosody inputs the fusion scores 60.82, 0.05 points above the deployed one, inside its interval, so those inputs add nothing measurable here. The raw SenseVoice tag scores 48.52 against the 57.70 of the FunAudioLLM report (Table 8), and on 97.51 % of the test clips most of its emotion mass is on OTHER or EMO_UNKNOWN.

## Emotion on MELD test (7 classes)

| condition | what it is | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE |
|---|---|---|---|---|---|---|
| `majority` | majority class of train (neutral) for every test utterance. Posterior = train class prior | 2610 | 31.27 [28.62, 34.31] | 9.28 | 48.12 | 0.97 |
| `text_k0` | text only, k = 0 (heads/text_ctx_k0.npz, gold transcripts) | 2610 | 59.62 [57.38, 62.03] | 41.22 | 62.76 | 3.83 |
| `text_best_k` | text only, best k on dev, full-dialogue window (heads/text_ctx.npz) | 2610 | 59.62 [57.38, 62.03] | 41.22 | 62.76 | 3.83 |
| `text_runtime_regime` | text only, best k, runtime-regime window: same-speaker previous utterances only, flags 1 | 2610 | 59.61 [57.36, 62.08] | 40.82 | 62.87 | 3.54 |
| `audio_raw_tag` | audio only, raw sensevoice tag: argmax over the seven named tokens | 2610 | 48.52 [46.33, 50.96] | 28.18 | 51.34 | 19.77 |
| `audio_cal_A` | audio only, AudioCalibrator A (slots + events + prosody) | 2610 | 48.02 [45.66, 50.64] | 26.92 | 53.72 | 2.84 |
| `audio_cal_B` | audio only, AudioCalibrator B (+ encoder states) | 2610 | 46.32 [44.14, 48.43] | 28.51 | 47.78 | 4.73 |
| `fused_stacked` | fused, stacked logistic regression (deployed) | 2610 | 60.77 [58.65, 63.00] | 39.24 | 63.98 | 1.52 |
| `fused_poe` | fused, product of experts with the dev-chosen weight | 2610 | 58.26 [55.91, 60.69] | 35.86 | 63.14 | 11.12 |
| `fused_poe_uncalibrated` | fused, fixed poe_uncalibrated fallback (w = 0.5) | 2610 | 59.00 [56.73, 61.36] | 36.65 | 63.49 | 13.14 |

Calibrator A lowers the ECE of the audio posterior from 19.77 to 2.84. Its weighted F1 is 48.02, 0.50 points below the raw SenseVoice tag. Calibrator B, which adds the encoder states, scores 46.32 against 48.02 for calibrator A. The product of experts scores 58.26 at the dev weight and 59.00 at the fixed weight, against 60.77 for the stacked fusion. Its ECE is 11.12 and 13.14, against 1.52.

## Gold text and live ASR transcripts

| condition | text input | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE |
|---|---|---|---|---|---|---|
| `text_best_k` | gold CSV text | 2610 | 59.62 [57.38, 62.03] | 41.22 | 62.76 | 3.83 |
| `live_asr_text_only` | SenseVoice transcripts | 2610 | 42.61 [40.28, 45.37] | 23.86 | 48.08 | 11.51 |
| `fused_stacked` | gold CSV text | 2610 | 60.77 [58.65, 63.00] | 39.24 | 63.98 | 1.52 |
| `live_asr_text` | SenseVoice transcripts | 2610 | 47.05 [44.53, 49.57] | 27.50 | 52.53 | 9.71 |

The transcripts have a corpus WER of 0.348488 over 2610 test utterances (`cache/wer_test.json`). The rows apply the service's ASR gate: 35 empty transcripts and 38 transcripts with an ASR confidence below 0.3 get a uniform posterior and the label neutral. Without the gate the rows score 42.68 (`live_asr_text_only`) and 47.12 (`live_asr_text`). The heads were trained on gold text. A text head trained on SenseVoice transcripts of train and dev was not built.

On the transcripts, the text head scores 42.61 against 59.62 on gold text (17.01 points below), and the stacked fusion scores 47.05 against 60.77 on gold text (13.72 points below).

## Ablations

| condition | dropped | calibrator inputs | fusion inputs | weighted F1 [95 % CI] | macro F1 | accuracy | ECE | audio-only weighted F1 | dev weighted F1 of the fusion |
|---|---|---|---|---|---|---|---|---|---|
| `fused_stacked` (deployed) | nothing | 24 | 19 | 60.77 [58.65, 63.00] | 39.24 | 63.98 | 1.52 | 48.02 | 57.95 |
| `ablation_no_events` | events | 13 | 18 | 60.91 [58.82, 63.33] | 39.53 | 64.10 | 2.90 | 48.23 | 58.96 |
| `ablation_no_prosody` | prosody | 20 | 19 | 60.91 [58.80, 63.09] | 39.38 | 64.14 | 1.65 | 47.62 | 58.32 |
| `ablation_no_events_no_prosody` | events and prosody | 9 | 18 | 60.82 [58.73, 63.08] | 39.40 | 63.98 | 2.05 | 47.55 | 58.72 |

The ablation heads were trained by `python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml --ablation` with the same folds, grids and dev selection as the deployed heads. Without events the fusion also loses the laughter probability. The deployed heads were not changed, and the reference heads that the same run retrains match the deployed files (`reproduces_deployed` in `heads/manifest.json`).

Without events, the fusion scores 60.91, 0.14 points above the deployed fusion and inside its interval. Without prosody, the fusion scores 60.91, 0.14 points above the deployed fusion and inside its interval. Without events and prosody, the fusion scores 60.82, 0.05 points above the deployed fusion and inside its interval. No ablation leaves the deployed fusion's interval, so the event and prosody inputs add nothing that this test set can measure. On dev, every ablated fusion also scores at least as high as the deployed one.

## Disagreement subset

| rows | n | text weighted F1 | audio weighted F1 | fused weighted F1 | text accuracy | audio accuracy | fused accuracy |
|---|---|---|---|---|---|---|---|
| subset | 489 | 51.44 | 23.44 | 49.62 | 53.37 | 26.79 | 52.35 |
| complement | 2121 | 61.34 | 53.94 | 63.28 | 64.92 | 59.92 | 66.67 |

`disagreement_subset` holds the 489 of 2610 test utterances (18.74 %) whose JSD between the gold-text posterior and the raw SenseVoice posterior is above 0.3831 (`thresholds.jsd_flag`, the dev threshold). The stacked fusion scores 49.62 [44.54, 54.32] weighted F1 on it. The audio column is the deployed calibrator.

On the subset, the fusion scores 49.62 against 51.44 for the text head alone, 1.82 points below it. On the complement it scores 63.28 against 61.34, 1.94 points above. So the fusion shows no gain over the text head where the words and the voice disagree, and the decision rule in [docs/decisions.md](docs/decisions.md) (Fusion) takes that claim out of the write-up.

## Early commit

| condition | branch | prefix | weighted F1 [95 % CI] | full-clip weighted F1 | label agreement with the full clip | clips longer than the prefix | their prefix weighted F1 | their full-clip weighted F1 | their label agreement |
|---|---|---|---|---|---|---|---|---|---|
| `early_commit_audio_2s` | audio | 2 s | 45.86 [43.48, 48.55] | 48.02 | 89.46 | 1685 | 43.15 | 46.34 | 83.68 |
| `early_commit_fused_2s` | fused | 2 s | 44.35 [41.86, 46.95] | 47.12 | 85.82 | 1685 | 43.09 | 46.98 | 78.04 |
| `early_commit_audio_4s` | audio | 4 s | 47.34 [44.88, 49.88] | 48.02 | 96.63 | 690 | 42.65 | 45.06 | 87.25 |
| `early_commit_fused_4s` | fused | 4 s | 46.06 [43.79, 48.41] | 47.12 | 94.48 | 690 | 42.62 | 45.99 | 79.13 |

These are offline prefix experiments, not the service's provisional states. A clip that is not longer than the prefix keeps its full-clip values. The fused rows take the text posterior from the prefix transcript, and their full-clip reference is the stacked fusion on the full transcript without the ASR gate. Agreement is in percent of clips. [results/RESULTS.md](results/RESULTS.md) also scores the audio rows with the inputs a provisional state uses.

On the 1685 clips longer than 2 s, the audio label from the prefix scores 43.15 against 46.34 on the full clip (3.19 points below) and matches the full-clip label on 83.68 % of them. On the 1685 clips longer than 2 s, the fused label from the prefix scores 43.09 against 46.98 on the full clip (3.89 points below) and matches the full-clip label on 78.04 % of them. On the 690 clips longer than 4 s, the audio label from the prefix scores 42.65 against 45.06 on the full clip (2.41 points below) and matches the full-clip label on 87.25 % of them. On the 690 clips longer than 4 s, the fused label from the prefix scores 42.62 against 45.99 on the full clip (3.37 points below) and matches the full-clip label on 79.13 % of them.

## Sentiment on MELD test (3 classes)

| condition | what it is | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE |
|---|---|---|---|---|---|---|
| `sentiment_fused` | sentiment, fused head over the 19 fusion features | 2610 | 69.17 [67.37, 70.96] | 66.45 | 69.46 | 1.88 |

The fused sentiment head scores 69.17, next to 66.10 for DialogueRNN on text and 67.56 on text and audio (MELD paper, Table 13). `sentiment_text_only` is not scored in this run (see the pending rows).

## Per-class F1 (emotion, test)

| condition | neutral | surprise | fear | sadness | joy | disgust | anger |
|---|---|---|---|---|---|---|---|
| `majority` | 64.98 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `text_k0` | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| `text_best_k` | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| `text_runtime_regime` | 78.01 | 53.82 | 20.29 | 24.55 | 54.92 | 15.91 | 38.25 |
| `audio_raw_tag` | 67.22 | 31.26 | 0.00 | 16.92 | 38.19 | 1.82 | 41.85 |
| `audio_cal_A` | 69.38 | 29.73 | 0.00 | 20.63 | 32.28 | 0.00 | 36.40 |
| `audio_cal_B` | 65.34 | 35.54 | 6.84 | 24.63 | 33.29 | 6.20 | 27.74 |
| `fused_stacked` | 77.87 | 54.91 | 0.00 | 28.16 | 56.07 | 10.53 | 47.12 |
| `fused_poe` | 76.79 | 52.78 | 0.00 | 18.93 | 53.52 | 5.63 | 43.35 |
| `fused_poe_uncalibrated` | 77.27 | 52.79 | 0.00 | 17.65 | 55.09 | 8.22 | 45.55 |
| `ablation_no_events` | 77.84 | 55.54 | 0.00 | 29.29 | 56.44 | 10.96 | 46.65 |
| `ablation_no_prosody` | 77.90 | 55.47 | 0.00 | 27.96 | 55.96 | 10.53 | 47.83 |
| `ablation_no_events_no_prosody` | 77.78 | 55.17 | 0.00 | 28.37 | 56.04 | 10.96 | 47.47 |
| `live_asr_text_only` | 65.06 | 19.72 | 9.38 | 16.22 | 31.51 | 4.44 | 20.69 |
| `live_asr_text` | 67.53 | 19.94 | 7.55 | 26.62 | 35.62 | 0.00 | 35.22 |
| `early_commit_audio_2s` | 69.42 | 28.75 | 0.00 | 12.17 | 27.11 | 0.00 | 31.92 |
| `early_commit_fused_2s` | 66.83 | 23.66 | 7.27 | 16.73 | 27.93 | 0.00 | 29.26 |
| `early_commit_audio_4s` | 69.32 | 28.89 | 0.00 | 13.33 | 31.87 | 0.00 | 37.08 |
| `early_commit_fused_4s` | 67.41 | 22.28 | 7.27 | 20.08 | 34.21 | 0.00 | 31.87 |
| DialogueRNN text+audio (MELD paper, Table 11) | 77.44 | 52.51 | 11.68 | 34.59 | 54.40 | 7.89 | 43.65 |
| test utterances in the class | 1256 | 281 | 50 | 208 | 402 | 68 | 345 |

The disagreement subset covers only part of test, so its per-class F1 is in [results/RESULTS.md](results/RESULTS.md).

The text head's macro F1 is 41.22, and its two lowest per-class scores are disgust (19.35) and fear (20.29), the two classes with the fewest test utterances (68 and 50). The stacked fusion's macro F1 is 39.24, 1.98 points below the text head, and it scores lower on neutral (77.87 against 77.93), fear (0.00 against 20.29) and disgust (10.53 against 19.35).

## Published rows

| row | task | weighted F1 | source |
|---|---|---|---|
| MELD paper Table 11, text-CNN, text | emotion | 55.02 | Poria et al., ACL 2019, arXiv 1810.02508, Table 11 |
| MELD paper Table 11, bcLSTM, text | emotion | 56.44 | Poria et al., ACL 2019, arXiv 1810.02508, Table 11 |
| MELD paper Table 11, bcLSTM, audio | emotion | 39.08 | Poria et al., ACL 2019, arXiv 1810.02508, Table 11 |
| MELD paper Table 11, bcLSTM, text+audio | emotion | 59.25 | Poria et al., ACL 2019, arXiv 1810.02508, Table 11 |
| MELD paper Table 11, DialogueRNN, text | emotion | 57.03 | Poria et al., ACL 2019, arXiv 1810.02508, Table 11 |
| MELD paper Table 11, DialogueRNN, audio | emotion | 41.79 | Poria et al., ACL 2019, arXiv 1810.02508, Table 11 |
| MELD paper Table 11, DialogueRNN, text+audio | emotion | 60.25 | Poria et al., ACL 2019, arXiv 1810.02508, Table 11 |
| EmoBERTa Table 2, RoBERTa-large, no past or future utterances | emotion | 63.46 | Kim and Vossen, arXiv 2108.12009, Table 2 |
| EmoBERTa Table 2, only past | emotion | 64.55 | Kim and Vossen, arXiv 2108.12009, Table 2 |
| EmoBERTa Table 2, past and future | emotion | 65.61 | Kim and Vossen, arXiv 2108.12009, Table 2 |
| EmoBERTa Table 2, without speaker names | emotion | 65.07 | Kim and Vossen, arXiv 2108.12009, Table 2 |
| EmoBERTa Table 2, COSMIC | emotion | 65.21 | Kim and Vossen, arXiv 2108.12009, Table 2 |
| EmoBERTa Table 2, DialogXL | emotion | 62.41 | Kim and Vossen, arXiv 2108.12009, Table 2 |
| FunAudioLLM Table 8, SenseVoice-S on MELD, no target-domain fine-tune (WF1, UA 50.8, WA 57.8, F1 44.6) | emotion | 57.70 | arXiv 2407.04051, Table 8. The report does not list its SER training corpora, so overlap with MELD cannot be excluded |
| FunAudioLLM Table 8, EmoBox reference row (UA 31.5, WA 51.9, F1 32.9) | emotion |  | arXiv 2407.04051, Table 8. No WF1 given |
| MELD paper Table 13, text-CNN, text | sentiment | 64.25 | Poria et al., ACL 2019, Table 13 |
| MELD paper Table 13, DialogueRNN, text | sentiment | 66.10 | Poria et al., ACL 2019, Table 13 |
| MELD paper Table 13, DialogueRNN, audio | sentiment | 49.61 | Poria et al., ACL 2019, Table 13 |
| MELD paper Table 13, DialogueRNN, text+audio | sentiment | 67.56 | Poria et al., ACL 2019, Table 13 |

The best text-only row of the MELD paper, 57.03 (Table 11), is below the lower end of the text head's interval. Fine-tuned RoBERTa-large without context reaches 63.46 in EmoBERTa (Table 2), above the upper end of that interval. DialogueRNN on text and audio, 60.25 (Table 11), is inside the stacked fusion's interval. The FunAudioLLM report gives SenseVoice-S 57.70 weighted F1 on MELD (Table 8), above the upper end of the interval of the raw SenseVoice tag measured here (48.52 [46.33, 50.96]). On 97.51 % of the test clips, more than half of the tag's emotion mass is on OTHER or EMO_UNKNOWN. The report does not state its MELD protocol (the split, the class mapping, how OTHER and unknown are scored) in enough detail to reproduce it.

## Pending rows

| condition | what it measures | what it needs |
|---|---|---|
| `sentiment_text_only` | sentiment, text-only head (logistic regression over the text context features) | a text-only `heads/sentiment.npz` (`tools/train_heads.py --text-only`). The committed sentiment head is the fused one, which the service uses, so this row is not scored in this run. The earlier text-only run is in [results/01_text_only.md](results/01_text_only.md) |

## Choices made on dev (`heads/manifest.json`)

- Text head: grid point `k=0,cw=none,lr=0.0001` (k = 0, class weights `none`), dev weighted F1 57.56, temperature 1.2228.
- Audio calibrator: variant A (24 inputs, C = 1.0, class weights `none`), dev weighted F1 45.96 against 44.43 for variant B (1048 inputs), temperature 1.0155.
- Fusion head: C = 10.0, class weights `none`, trained on out-of-fold expert posteriors (5 folds by dialogue), dev weighted F1 57.95, temperature 0.8459.
- Product of experts: weight 0.75 from the grid 0.25, 0.5, 0.75, 1.0, dev weighted F1 55.56. The stacked fusion scores higher on dev, so it stays deployed (decision rule in [docs/decisions.md](docs/decisions.md), Fusion).
- Selector thresholds in use: low_conf 0.47, high_conf 0.73, from the fusion head's calibrated dev predictions (the text head alone gives 0.39 and 0.73).
- Discrepancy flag threshold: 0.3831, the 0.8 quantile of the dev JSD between the text posterior and the raw SenseVoice posterior.
- Sentiment head (fused): C = 1.0, class weights `none`, dev weighted F1 67.79, temperature 0.8054.
- CSV rows without an audio clip, left out of the audio stage: 1 in train (`125_3`) and 1 in dev (`110_7`).
- Ablation heads: dev weighted F1 of the fusion 57.95 with every input, 58.96 without events, 58.32 without prosody, 58.72 without events and prosody.
- Audio and fusion stage: trained.
