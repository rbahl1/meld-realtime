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

## Text, voice and face on MELD test

Rendered from `results/metrics_vision.json` (written by `python tools/evaluate_vision.py --cache cache/ --heads heads/ --audio audio/ --meld meld/ --out results/RESULTS_vision.md results/metrics_vision.json`). The speaker rule (`asd`), the face calibrator (variant B), fusion3 (a logistic regression over 29 features built from the text, audio and face posteriors) and its thresholds were chosen on dev by `tools/train_vision.py` (`heads/manifest.json`, section `vision`). A face was found in 99.46 % of the 2610 test clips. In `vision_raw_fer` and `vision_cal`, a clip without a face gets the uniform posterior and the label neutral. The test protocol and the intervals are those of the main tables. The paired differences resample test dialogues (1000 resamples, seed 0) and score both conditions on each resample.

The face alone scores 34.37 weighted F1 with the calibrator and 32.22 as the raw FER label, against 59.62 for the text head. Fusion3 scores 61.56 [59.45, 63.79] on gold text against 60.77 for the two-way fusion, a paired difference of 0.80 points [0.20, 1.48]. On SenseVoice transcripts it scores 47.96 against 47.05, a paired difference of 0.91 points [0.31, 1.52]. The claim rule was fixed before the run: the face adds to text and audio only if the paired 95 % interval of fusion3 minus the two-way fusion on gold text lies above zero. It does, so by this rule the face adds measured value here. Fusion3 changes 80 of the 2610 gold-text labels of the two-way fusion: 36 from wrong to right, 20 from right to wrong and 24 from one wrong label to another.

| condition | what it is | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE | face coverage |
|---|---|---|---|---|---|---|---|
| `vision_raw_fer` | argmax of the averaged FER posterior of the chosen track. No face: uniform, label neutral. All test rows | 2610 | 32.22 [29.77, 34.89] | 15.92 | 34.44 | 36.63 | 99.46 |
| `vision_cal` | deployed vision calibrator (heads/vision_cal.npz). No face: uniform, label neutral. All test rows | 2610 | 34.37 [31.83, 37.32] | 12.70 | 47.05 | 8.32 | 99.46 |
| `text_best_k` | reference: deployed text head, gold text, rows with audio | 2610 | 59.62 [57.38, 62.03] | 41.22 | 62.76 | 3.83 |  |
| `fused_stacked` | reference: deployed two-way fusion (heads/fusion.npz), gold text | 2610 | 60.77 [58.65, 63.00] | 39.24 | 63.98 | 1.52 |  |
| `fused3_stacked` | fusion3 (heads/fusion3.npz) over gold-text, audio and face posteriors | 2610 | 61.56 [59.45, 63.79] | 40.50 | 64.60 | 2.03 | 99.46 |
| `live_asr_text` | reference: two-way fusion on the SenseVoice transcripts with the service's ASR gate | 2610 | 47.05 [44.53, 49.57] | 27.50 | 52.53 | 9.71 |  |
| `live_asr_fused3` | fusion3 on the SenseVoice transcripts with the service's ASR gate | 2610 | 47.96 [45.60, 50.40] | 28.82 | 53.18 | 9.13 | 99.46 |
| `text_best_k_face_rows` | reference: text head, gold text, clips with a face | 2596 | 59.58 [57.28, 61.95] | 41.28 | 62.71 | 3.86 |  |
| `fused_stacked_face_rows` | reference: two-way fusion, gold text, clips with a face | 2596 | 60.74 [58.61, 62.93] | 39.26 | 63.94 | 1.50 |  |
| `fused3_stacked_face_rows` | fusion3, gold text, clips with a face | 2596 | 61.54 [59.46, 63.67] | 40.54 | 64.56 | 2.07 |  |
| `live_asr_text_face_rows` | reference: two-way fusion, transcripts with the ASR gate, clips with a face | 2596 | 47.09 [44.61, 49.61] | 27.58 | 52.54 | 9.69 |  |
| `live_asr_fused3_face_rows` | fusion3, transcripts with the ASR gate, clips with a face | 2596 | 47.98 [45.71, 50.40] | 28.88 | 53.16 | 9.11 |  |
| `face_disagreement_subset` | fusion3 on the clips with a face whose words-face JSD is above vision.jsd_face_flag | 557 | 64.04 [59.64, 68.28] | 47.10 | 66.79 | 3.03 |  |
| `sentiment3` | sentiment over the 29 fusion3 features (heads/sentiment3.npz), gold text | 2610 | 69.03 [67.05, 70.88] | 66.44 | 69.31 | 2.85 |  |

`sentiment3` is a 3-class task. It scores 69.03 against 69.17 for the deployed two-way sentiment head (`sentiment_fused`).

### Paired differences (weighted F1, a minus b, in points)

| comparison | a | b | rows | n | difference | 95 % CI | interval above zero |
|---|---|---|---|---|---|---|---|
| `gold_all` | `fused3_stacked` | `fused_stacked` | all | 2610 | 0.80 | [0.20, 1.48] | yes |
| `live_all` | `live_asr_fused3` | `live_asr_text` | all | 2610 | 0.91 | [0.31, 1.52] | yes |
| `gold_face_rows` | `fused3_stacked` | `fused_stacked` | face | 2596 | 0.80 | [0.20, 1.49] | yes |
| `live_face_rows` | `live_asr_fused3` | `live_asr_text` | face | 2596 | 0.89 | [0.28, 1.50] | yes |
| `gold_face_disagreement` | `fused3_stacked` | `fused_stacked` | disagree | 557 | 1.17 | [-0.29, 2.60] | no |
| `gold_face_disagreement_vs_text` | `fused3_stacked` | `text_best_k` | disagree | 557 | -0.84 | [-3.71, 2.20] | no |

### Label flips (fusion3 against the two-way fusion)

| text input | n | labels changed | wrong to right | right to wrong | wrong to another wrong label |
|---|---|---|---|---|---|
| gold CSV text | 2610 | 80 | 36 | 20 | 24 |
| SenseVoice transcripts, ASR gate | 2610 | 86 | 37 | 20 | 29 |

### Face-disagreement subset

| rows | n | text weighted F1 | two-way weighted F1 | fusion3 weighted F1 | text accuracy | two-way accuracy | fusion3 accuracy |
|---|---|---|---|---|---|---|---|
| subset | 557 | 64.88 | 62.87 | 64.04 | 66.97 | 65.89 | 66.79 |
| complement | 2039 | 57.94 | 59.98 | 60.63 | 61.55 | 63.41 | 63.95 |

The subset holds the 557 of 2596 test clips with a face (21.46 %) whose JSD between the gold-text posterior and the raw face posterior is above 0.5729 (`vision.jsd_face_flag`, chosen on dev). There, fusion3 scores 64.04 against 62.87 for the two-way fusion and 64.88 for the text head. The paired differences to the two-way fusion (1.17 points [-0.29, 2.60]) and to the text head (-0.84 points [-3.71, 2.20]) both have intervals that include zero.

### Choices made on dev (`heads/manifest.json`, section `vision`)

| speaker rule | dev weighted F1, all rows | dev weighted F1, rows with a face | fallback rate on dev (%) | chosen |
|---|---|---|---|---|
| `largest` | 25.19 | 25.24 | 0.00 |  |
| `present` | 25.57 | 25.62 | 0.00 |  |
| `asd` | 25.82 | 25.87 | 4.08 | yes |
| `mouth` | 25.53 | 25.58 | 3.26 |  |

- Of the 1103 dev clips with a face, 81.50 % have more than one track, and on 67.09 % the four rules do not all choose the same track.
- Face calibrator: variant B, dev weighted F1 30.20 against 25.82 for variant A.
- Fusion3: dev weighted F1 59.53 against 57.95 for the two-way fusion.
- Sentiment3: dev weighted F1 67.77.
- Selector thresholds in the vision mode: low_conf 0.56, high_conf 0.73. Words-face flag: JSD above 0.5729.

## Commit policy (fitted Q-iteration)

Rendered from `results/commit_policy.json` (written by `python tools/train_commit_policy.py`). At each full-window provisional of an utterance (t = 1, 2, 3 and 4 s while the segment is open), the policy commits the fused label or waits. Without a commit, the label of the endpoint is committed. The reward is one for a correct label minus c times the commit time in seconds. The policy is trained by fitted Q-iteration on the train traces (25384 ticks of 9988 clips). The Q_wait model and the parameters of the three baselines were chosen on dev for each c, and test was read once per method and c. The deployed policy uses c = 0.05. The intervals are 95 % bootstrap intervals over test dialogues (1000 resamples, seed 0).

At c = 0.05 the policy commits 86.74 % of the test labels before the endpoint, at a mean of 1.362 s after the segment start against 4.046 s at the endpoint. Its accuracy is 51.65 against 53.18 at the endpoint, and its weighted F1 41.81 against 47.96. Its Q_wait model (MLP) stopped at the cap of 10 iterations before the emit decisions on the train ticks stopped changing. The claim rule was fixed before the run: RL adds measured value over a baseline at a cost c only where the paired 95 % interval of the test reward difference lies above zero. By that rule the policy adds measured value over the endpoint at every cost, over the fixed time at c = 0.02 only and over the threshold at c = 0.2 only. At the deployed c = 0.05 the gain over the fixed time and over the threshold is not shown. Charging every tick commit 0.152 s more, as the endpoint commit is charged for the computation of the final, changes none of these outcomes.

### Choices made on dev

| c | Q_wait model | iterations | dev reward of the policy | fixed T | dev reward fixed | tau | dev reward threshold | dev reward endpoint |
|---|---|---|---|---|---|---|---|---|
| 0.02 | ridge, alpha 1 | 9 | 0.4514 | 2 s | 0.4383 | 0.7 | 0.4445 | 0.4161 |
| 0.05 | MLP, alpha 0.0001 | 10 | 0.4052 | 1 s | 0.4005 | 0.5 | 0.4007 | 0.3025 |
| 0.1 | ridge, alpha 10 | 3 | 0.3462 | 1 s | 0.3506 | 0.3 | 0.3477 | 0.1132 |
| 0.2 | ridge, alpha 100 | 2 | 0.2509 | 1 s | 0.2509 | 0.3 | 0.2433 | -0.2655 |

### Test, every method and cost

| c | method | mean reward [95 % CI] | accuracy | weighted F1 | mean commit s | committed before the endpoint |
|---|---|---|---|---|---|---|
| 0.02 | policy | 0.4855 [0.4635, 0.5104] | 52.26 | 44.20 | 1.855 | 69.77 % |
| 0.02 | endpoint | 0.4509 [0.4280, 0.4746] | 53.18 | 47.96 | 4.046 | 0.00 % |
| 0.02 | fixed time (2 s) | 0.4731 [0.4521, 0.4957] | 51.15 | 44.17 | 1.920 | 78.16 % |
| 0.02 | threshold (0.7) | 0.4804 [0.4581, 0.5043] | 51.92 | 43.80 | 1.938 | 66.48 % |
| 0.05 | policy | 0.4484 [0.4264, 0.4732] | 51.65 | 41.81 | 1.362 | 86.74 % |
| 0.05 | endpoint | 0.3295 [0.3040, 0.3562] | 53.18 | 47.96 | 4.046 | 0.00 % |
| 0.05 | fixed time (1 s) | 0.4498 [0.4281, 0.4723] | 49.96 | 40.30 | 0.997 | 97.01 % |
| 0.05 | threshold (0.5) | 0.4463 [0.4236, 0.4715] | 51.19 | 41.71 | 1.312 | 88.05 % |
| 0.1 | policy | 0.3989 [0.3764, 0.4234] | 50.65 | 40.19 | 1.076 | 95.36 % |
| 0.1 | endpoint | 0.1272 [0.0871, 0.1612] | 53.18 | 47.96 | 4.046 | 0.00 % |
| 0.1 | fixed time (1 s) | 0.3999 [0.3782, 0.4224] | 49.96 | 40.30 | 0.997 | 97.01 % |
| 0.1 | threshold (0.3) | 0.3977 [0.3753, 0.4212] | 50.11 | 40.15 | 1.035 | 95.86 % |
| 0.2 | policy | 0.3002 [0.2785, 0.3226] | 49.96 | 40.30 | 0.997 | 97.01 % |
| 0.2 | endpoint | -0.2773 [-0.3472, -0.2203] | 53.18 | 47.96 | 4.046 | 0.00 % |
| 0.2 | fixed time (1 s) | 0.3002 [0.2785, 0.3226] | 49.96 | 40.30 | 0.997 | 97.01 % |
| 0.2 | threshold (0.3) | 0.2942 [0.2718, 0.3175] | 50.11 | 40.15 | 1.035 | 95.86 % |

### Test, policy minus each baseline (paired bootstrap over dialogues)

| c | baseline | reward difference | 95 % CI | RL adds measured value (claim rule) |
|---|---|---|---|---|
| 0.02 | endpoint | 0.0346 | [0.0234, 0.0471] | yes |
| 0.02 | fixed | 0.0124 | [0.0017, 0.0238] | yes |
| 0.02 | threshold | 0.0051 | [-0.0035, 0.0132] | no |
| 0.05 | endpoint | 0.1189 | [0.1004, 0.1402] | yes |
| 0.05 | fixed | -0.0014 | [-0.0113, 0.0088] | no |
| 0.05 | threshold | 0.0021 | [-0.0058, 0.0093] | no |
| 0.1 | endpoint | 0.2717 | [0.2418, 0.3074] | yes |
| 0.1 | fixed | -0.0010 | [-0.0064, 0.0049] | no |
| 0.1 | threshold | 0.0013 | [-0.0037, 0.0065] | no |
| 0.2 | endpoint | 0.5775 | [0.5245, 0.6464] | yes |
| 0.2 | fixed | 0.0000 | [0.0000, 0.0000] | no |
| 0.2 | threshold | 0.0060 | [0.0016, 0.0104] | yes |

The full tables, the P(correct) grid and the sensitivity rows are in [results/commit_policy.md](results/commit_policy.md).

## Streamed reply

Rendered from `results/generator_bench_vm.json` and `results/generator_bench_vm_1.5b.json` (written by `tools/bench_generator.py`, commands in the files). Every run replies to the same 200 test states, 100 with gold text and 100 with SenseVoice transcripts, drawn from `cache/vision_post_test.npz` with seed 0. Decoding is greedy, with at most 40 new tokens. The decode rate counts every forward pass after the first token, and the check pass rate is the share of replies that pass `check_reply` (length, register, emotion words, prompt echo).

| file | threads | first token p50 / p95 ms | first word p50 / p95 ms | full reply p50 / p95 ms | decode passes per s | words per reply | check pass rate |
|---|---|---|---|---|---|---|---|
| `Qwen2.5-0.5B-Instruct/model_int8.onnx` | 2 | 103.78 / 136.69 | 159.48 / 323.26 | 531.98 / 1304.45 | 20.09 | 7.58 | 0.97 |
| `Qwen2.5-0.5B-Instruct/model_int8.onnx` | 4 | 70.94 / 100.27 | 108.75 / 233.55 | 378.52 / 896.2 | 28.47 | 7.58 | 0.97 |
| `Qwen2.5-0.5B-Instruct/model_q4.onnx` | 2 | 274.86 / 354.14 | 349.75 / 458.58 | 792.29 / 1291.37 | 17.33 | 7.81 | 0.995 |
| `Qwen2.5-0.5B-Instruct/model_q4.onnx` | 4 | 155.02 / 196.14 | 202.44 / 292.25 | 496.41 / 830.08 | 26.87 | 7.81 | 0.995 |
| `Qwen2.5-1.5B-Instruct/model_int8.onnx` | 4 | 149.35 / 194.54 | 352.49 / 783.58 | 1228.94 / 2985.06 | 13.22 | 9.15 | 0.97 |
| `Qwen2.5-1.5B-Instruct/model_q4.onnx` | 4 | 473.33 / 580.74 | 575.33 / 773.84 | 1330.17 / 2541.36 | 11.64 | 8.51 | 0.995 |

The choice rule was fixed before the bench on 200 states. It deploys the largest model with a file whose first-token p95 is at most 350.0 ms and whose decode rate is at least 15.0 passes per s at 4 threads. Among the files of one model it keeps int8 unless q4 is faster by more than 1.2x at the same reply length with a check pass rate no lower. These definitions of the decode rate and of a faster q4 were set after preliminary benches on 20 states (the docstring of `tools/bench_generator.py`). No file of Qwen2.5-1.5B-Instruct meets both targets (decode rates 13.22 and 11.64 passes per s). The deployed file is `Qwen2.5-0.5B-Instruct/model_int8.onnx`. The reason recorded by its run: "int8 (q4 speedup 0.80x on a reply of 9.82 tokens, q4 pass rate 0.995 vs 0.97)". The original wording asked for 15 tokens per s and a q4 file faster by more than 20 %, and it picks the same file: at 4 threads the int8 and q4 files of Qwen2.5-0.5B-Instruct write 23.74 and 23.01 final-reply tokens per s, those of Qwen2.5-1.5B-Instruct 8.79 and 9.49, and the q4 file of Qwen2.5-0.5B-Instruct has the longer full-reply p50 (496.41 against 378.52 ms). In the replay bench with video and `--reply llm` (`results/replay_bench_vm_tav.json`), the first streamed word is emitted 388.85 / 855.99 ms (p50 / p95) after the endpoint frame arrives, and the whole reply 549.92 / 1333.29 ms after the final state. [results/measurements.md](results/measurements.md) has the rest of that bench.
