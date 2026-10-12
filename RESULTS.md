# Results

Rendered by `python tools/render_results.py` from `results/metrics.json` (written by `python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json`) and `heads/manifest.json` (written by `python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml --text-only`). The full report with the confusion matrices is [results/RESULTS.md](results/RESULTS.md), and the dev grid of the text head is in [results/01_text_only.md](results/01_text_only.md). Model selection, temperature and thresholds were chosen on dev. The test split was read once per condition. Weighted F1 is the primary metric, as in the published rows. The intervals are 95 % percentile bootstrap intervals over test dialogues (1000 resamples, seed 0). ECE uses 10 equal-width confidence bins. All metrics are in percent.

## Emotion on MELD test (7 classes)

| condition | what it is | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE |
|---|---|---|---|---|---|---|
| `majority` | majority class of train (neutral) for every test utterance; posterior = train class prior | 2610 | 31.27 [28.62, 34.31] | 9.28 | 48.12 | 0.97 |
| `text_k0` | text only, k = 0 (heads/text_ctx_k0.npz, gold transcripts) | 2610 | 59.62 [57.38, 62.03] | 41.22 | 62.76 | 3.83 |
| `text_best_k` | text only, best k on dev, full-dialogue window (heads/text_ctx.npz) | 2610 | 59.62 [57.38, 62.03] | 41.22 | 62.76 | 3.83 |
| `text_runtime_regime` | text only, best k, runtime-regime window: same-speaker previous utterances only, flags 1 | 2610 | 59.61 [57.36, 62.08] | 40.82 | 62.87 | 3.54 |

## Sentiment on MELD test (3 classes)

| condition | what it is | n | weighted F1 [95 % CI] | macro F1 | accuracy | ECE |
|---|---|---|---|---|---|---|
| `sentiment_text_only` | sentiment, text-only head (logistic regression over the text context features) | 2610 | 66.24 [64.41, 68.20] | 63.33 | 66.70 | 2.04 |

## Per-class F1 (emotion, test)

| condition | neutral | surprise | fear | sadness | joy | disgust | anger |
|---|---|---|---|---|---|---|---|
| `majority` | 64.98 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `text_k0` | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| `text_best_k` | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| `text_runtime_regime` | 78.01 | 53.82 | 20.29 | 24.55 | 54.92 | 15.91 | 38.25 |
| DialogueRNN text+audio (MELD paper, Table 11) | 77.44 | 52.51 | 11.68 | 34.59 | 54.40 | 7.89 | 43.65 |
| test utterances in the class | 1256 | 281 | 50 | 208 | 402 | 68 | 345 |

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
| FunAudioLLM Table 8, SenseVoice-S on MELD, no target-domain fine-tune (WF1; UA 50.8, WA 57.8, F1 44.6) | emotion | 57.70 | arXiv 2407.04051, Table 8; the report does not list its SER training corpora, so overlap with MELD cannot be excluded |
| FunAudioLLM Table 8, EmoBox reference row (UA 31.5, WA 51.9, F1 32.9) | emotion |  | arXiv 2407.04051, Table 8; no WF1 given |
| MELD paper Table 13, text-CNN, text | sentiment | 64.25 | Poria et al., ACL 2019, Table 13 |
| MELD paper Table 13, DialogueRNN, text | sentiment | 66.10 | Poria et al., ACL 2019, Table 13 |
| MELD paper Table 13, DialogueRNN, audio | sentiment | 49.61 | Poria et al., ACL 2019, Table 13 |
| MELD paper Table 13, DialogueRNN, text+audio | sentiment | 67.56 | Poria et al., ACL 2019, Table 13 |

## Pending rows

| condition | what it measures | what it needs |
|---|---|---|
| `audio_raw_tag` | audio only, raw sensevoice tag: argmax over the seven named tokens | `cache/audio_test.npz` (MELD audio through `tools/extract_audio.py` and `tools/cache_audio_features.py`) |
| `audio_cal_A` | audio only, AudioCalibrator A (slots + events + prosody) | `cache/audio_test.npz` and `heads/audio_cal_A.npz` (`tools/train_heads.py` without `--text-only`, on the train and dev audio caches) |
| `audio_cal_B` | audio only, AudioCalibrator B (+ encoder states) | as audio_cal_A, with caches written with `--encoder-states` so that `heads/audio_cal_B.npz` exists |
| `fused_stacked` | fused, stacked logistic regression (deployed) | `cache/audio_test.npz`, `heads/audio_cal.npz` and `heads/fusion.npz` (`tools/train_heads.py` without `--text-only`) |
| `fused_poe` | fused, product of experts with the dev-chosen weight | `cache/audio_test.npz` and `heads/audio_cal.npz`, with the weight from `heads/manifest.json` |
| `fused_poe_uncalibrated` | fused, fixed poe_uncalibrated fallback (w = 0.5) | `cache/audio_test.npz` only (the raw SenseVoice posterior at w = 0.5) |
| `ablation_no_events` | fused without event features | MELD audio, heads trained without these features, and no code yet: tools/evaluate.py has no branch that scores this row |
| `ablation_no_prosody` | fused without prosody | MELD audio, heads trained without these features, and no code yet: tools/evaluate.py has no branch that scores this row |
| `ablation_no_events_no_prosody` | fused without events and prosody | MELD audio, heads trained without these features, and no code yet: tools/evaluate.py has no branch that scores this row |
| `live_asr_text` | fused with sensevoice transcripts instead of gold text, plus WER | MELD audio, text embeddings of the SenseVoice test transcripts (`tools/embed_text.py` embeds the CSV text only), and no code yet: tools/evaluate.py has no branch that scores this row |
| `disagreement_subset` | utterances with jsd above the dev threshold: size, per-branch accuracy, fused gain | MELD audio, the jsd threshold that `tools/train_heads.py` picks on dev once audio exists, and no code yet: tools/evaluate.py has no branch that scores this row |
| `early_commit` | audio-only and fused label on 2 s and 4 s prefixes versus the full clip | MELD audio (the 2 s and 4 s prefix features are in `cache/audio_test.npz` once cached), and no code yet: tools/evaluate.py has no branch that scores this row |
| `sentiment_fused` | sentiment, fused head over the 19 fusion features | `cache/audio_test.npz`, `heads/audio_cal.npz` and a fused `heads/sentiment.npz` (`tools/train_heads.py` without `--text-only`) |

## Choices made on dev (`heads/manifest.json`)

- Text head: grid point `k=0,cw=none,lr=0.0001` (k = 0, class weights `none`), dev weighted F1 57.56, temperature 1.2228.
- Selector thresholds from dev: low_conf 0.39, high_conf 0.73.
- Discrepancy flag threshold: not chosen yet (needs MELD audio). The service uses its `--jsd-flag` default until then.
- Sentiment head (text_only): C = 0.1, class weights `none`, dev weighted F1 65.25, temperature 1.4197.
- Audio and fusion stage: pending (needs MELD audio).

## Observations

The deployed text head scores 59.62 weighted F1 on test (95 % CI [57.38, 62.03]), against 31.27 for the majority floor. The best text-only row of the MELD paper, 57.03 (Table 11), is below the lower end of that interval. Fine-tuned RoBERTa-large without context reaches 63.46 in EmoBERTa (Table 2), above its upper end. Macro F1 is 41.22, and the two lowest per-class scores are disgust (19.35) and fear (20.29), the two classes with the fewest test utterances (68 and 50). The text-only sentiment head scores 66.24, next to 66.10 for DialogueRNN on text and 67.56 on text and audio (MELD paper, Table 13). Of the 18 conditions, 13 are pending, and the table above says what each one needs.
