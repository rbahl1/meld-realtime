# Experiment 01, text-only heads

Rendered by `python tools/report_text_only.py --manifest heads/manifest.json --metrics results/metrics.json --out results/01_text_only.md` from `heads/manifest.json` (written by `python tools/train_heads.py --cache cache/ --out heads/ --config configs/heads.yaml --text-only`) and `results/metrics.json` (written by `python tools/evaluate.py --cache cache/ --heads heads/ --out results/RESULTS.md results/metrics.json`). The inputs are frozen RoBERTa-base embeddings, mean-pooled, in `cache/text_{split}.npy` (written by `tools/embed_text.py`). The TextContextHead is an MLP with 256 hidden units and early stopping on dev weighted F1 (scikit-learn 1.9.1, 2 threads, seed 0). Metrics are in percent.

## Dev grid (weighted F1 / macro F1, best epoch / epochs run)

| k \ class weights | meld_readme (wF1 / mF1) | balanced (wF1 / mF1) | none (wF1 / mF1) |
| --- | --- | --- | --- |
| k = 0 | 54.01 / 39.22 (lr 0.0001, ep 33/39) | 55.06 / 42.71 (lr 0.0003, ep 17/23) | 57.56 / 41.81 (lr 0.0001, ep 12/18) |
| k = 1 | 55.32 / 40.07 (lr 0.0001, ep 39/45) | 55.35 / 41.18 (lr 0.0001, ep 30/36) | 56.14 / 39.26 (lr 0.0001, ep 23/29) |
| k = 2 | 54.68 / 38.15 (lr 0.0001, ep 41/47) | 55.75 / 40.10 (lr 0.0001, ep 25/31) | 55.48 / 38.33 (lr 0.0003, ep 4/10) |

All grid points:

| grid point | weighted F1 | macro F1 | accuracy | best epoch / run | wall s |
| --- | --- | --- | --- | --- | --- |
| k=0,cw=meld_readme,lr=0.001 | 53.88 | 39.41 | 56.54 | 10 / 16 | 5.5 |
| k=0,cw=meld_readme,lr=0.0003 | 53.96 | 38.75 | 56.99 | 17 / 23 | 7.9 |
| k=0,cw=meld_readme,lr=0.0001 | 54.01 | 39.22 | 57.44 | 33 / 39 | 12.4 |
| k=0,cw=balanced,lr=0.001 | 53.21 | 41.46 | 52.30 | 5 / 11 | 3.6 |
| k=0,cw=balanced,lr=0.0003 | 55.06 | 42.71 | 54.82 | 17 / 23 | 8.0 |
| k=0,cw=balanced,lr=0.0001 | 54.68 | 42.19 | 53.83 | 25 / 31 | 9.9 |
| k=0,cw=none,lr=0.001 | 55.56 | 36.69 | 59.06 | 1 / 7 | 2.2 |
| k=0,cw=none,lr=0.0003 | 57.22 | 40.99 | 60.50 | 7 / 13 | 4.0 |
| k=0,cw=none,lr=0.0001 | 57.56 | 41.81 | 60.87 | 12 / 18 | 5.8 |
| k=1,cw=meld_readme,lr=0.001 | 54.60 | 39.32 | 57.35 | 5 / 11 | 6.2 |
| k=1,cw=meld_readme,lr=0.0003 | 54.95 | 39.73 | 57.44 | 19 / 25 | 14.4 |
| k=1,cw=meld_readme,lr=0.0001 | 55.32 | 40.07 | 57.71 | 39 / 45 | 26.2 |
| k=1,cw=balanced,lr=0.001 | 54.15 | 39.83 | 55.00 | 8 / 14 | 8.9 |
| k=1,cw=balanced,lr=0.0003 | 55.04 | 40.69 | 55.46 | 12 / 18 | 10.3 |
| k=1,cw=balanced,lr=0.0001 | 55.35 | 41.18 | 55.73 | 30 / 36 | 20.3 |
| k=1,cw=none,lr=0.001 | 54.82 | 39.60 | 57.26 | 7 / 13 | 7.7 |
| k=1,cw=none,lr=0.0003 | 56.08 | 40.08 | 58.52 | 9 / 15 | 8.5 |
| k=1,cw=none,lr=0.0001 | 56.14 | 39.26 | 58.79 | 23 / 29 | 17.6 |
| k=2,cw=meld_readme,lr=0.001 | 52.59 | 37.14 | 55.18 | 6 / 12 | 9.8 |
| k=2,cw=meld_readme,lr=0.0003 | 53.58 | 37.16 | 56.81 | 9 / 15 | 12.1 |
| k=2,cw=meld_readme,lr=0.0001 | 54.68 | 38.15 | 57.53 | 41 / 47 | 39.0 |
| k=2,cw=balanced,lr=0.001 | 54.46 | 40.63 | 55.91 | 14 / 20 | 16.8 |
| k=2,cw=balanced,lr=0.0003 | 55.27 | 39.72 | 56.09 | 10 / 16 | 13.0 |
| k=2,cw=balanced,lr=0.0001 | 55.75 | 40.10 | 56.45 | 25 / 31 | 25.8 |
| k=2,cw=none,lr=0.001 | 54.13 | 38.02 | 56.27 | 2 / 8 | 7.2 |
| k=2,cw=none,lr=0.0003 | 55.48 | 38.33 | 58.16 | 4 / 10 | 8.0 |
| k=2,cw=none,lr=0.0001 | 54.91 | 36.76 | 58.25 | 9 / 15 | 12.3 |

Training the grid took 326.6 s. The whole run took 335.9 s.

## Chosen configuration (dev)

- Grid point `k=0,cw=none,lr=0.0001`: k = 0, class weights `none`, dev weighted F1 57.56, macro F1 41.81, accuracy 60.87, 199687 parameters in `heads/text_ctx.npz`.
- Temperature 1.2228: dev NLL from 1.1998 to 1.1798, dev ECE from 6.74 to 3.59.
- Selector thresholds: low_conf = 0.39 and high_conf = 0.73 (dev). jsd_flag = None (needs MELD audio), so the service uses its `--jsd-flag` default.
- The k = 0 row uses `k=0,cw=none,lr=0.0001` in `heads/text_ctx_k0.npz`.
- Sentiment head (text_only): logistic regression over TextContextHead features, k=0 (768*(k+1)+4), C = 0.1, class weights `none`, dev weighted F1 65.25, temperature 1.4197.

## Test (one run per condition)

| condition | weighted F1 | 95 % CI | macro F1 | accuracy | ECE | note |
| --- | --- | --- | --- | --- | --- | --- |
| majority | 31.27 | [28.62, 34.31] | 9.28 | 48.12 | 0.97 | train prior as posterior; label = neutral |
| text_k0 | 59.62 | [57.38, 62.03] | 41.22 | 62.76 | 3.83 | class weights none, T=1.223 |
| text_best_k | 59.62 | [57.38, 62.03] | 41.22 | 62.76 | 3.83 | k=0, class weights none, T=1.223 |
| text_runtime_regime | 59.61 | [57.36, 62.08] | 40.82 | 62.87 | 3.54 | same head as text_best_k, same-speaker window (k=0: differs from text_best_k only through the turn-index flags) |
| sentiment_text_only | 66.24 | [64.41, 68.20] | 63.33 | 66.70 | 2.04 | k=0, C=0.1, T=1.420 |

### Per-class F1 (test)

| condition | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 64.98 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| text_k0 | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| text_best_k | 77.93 | 53.63 | 20.29 | 24.00 | 54.88 | 19.35 | 38.42 |
| text_runtime_regime | 78.01 | 53.82 | 20.29 | 24.55 | 54.92 | 15.91 | 38.25 |

### Published text-only rows (test weighted F1, sources in results/RESULTS.md)

- MELD paper Table 11, text-CNN, text: 55.02
- MELD paper Table 11, bcLSTM, text: 56.44
- MELD paper Table 11, DialogueRNN, text: 57.03

## Observations

On dev, the grid point `k=0,cw=none,lr=0.0001` (k = 0, class weights `none`) has the best weighted F1, 57.56, with a macro F1 of 41.81.
The best macro F1 on dev is 42.71, at `k=0,cw=balanced,lr=0.0003`, whose weighted F1 is 55.06. Choosing by weighted F1 costs 0.90 macro-F1 points and gains 2.50 weighted-F1 points.
The MELD README class weights reach at best 55.32 weighted F1 and 40.07 macro F1, 2.24 and 1.75 points below the chosen point.
Context did not help on dev: the best points reach 56.14 at k = 1 and 55.75 at k = 2, against 57.56 at k = 0. The deployed head sees the current utterance only.
On test, the deployed head scores 59.62 weighted F1 (95 % CI [57.38, 62.03]). The k = 0 head scores 59.62, and the deployed head with the same-speaker runtime window scores 59.61.
The published text-only rows of the MELD paper (Table 11) are text-CNN 55.02, bcLSTM 56.44 and DialogueRNN 57.03.
EmoBERTa, a fine-tuned RoBERTa-large without context, reports 63.46 (Table 2).
