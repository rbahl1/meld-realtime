# MELD csv statistics

Computed from the csvs below by `python tools/meld_stats.py --meld meld/ --out results/data_stats.md`. No audio was used.

| file | bytes | sha256 |
| --- | --- | --- |
| `train_sent_emo.csv` | 1119290 | `d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db` |
| `dev_sent_emo.csv` | 121640 | `2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c` |
| `test_sent_emo.csv` | 294526 | `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888` |

Definitions: duration = EndTime - StartTime of the subtitle timestamps; percentiles are numpy linear interpolation; words = whitespace-separated tokens of the Utterance column; consecutive pairs = adjacent rows of the same dialogue in Utterance_ID order (utterance ids have gaps in some dialogues, so a pair need not have consecutive ids); speaker and emotion change rates are over those pairs. `all` pools the three splits.

## Counts

| split | utterances | dialogues | speakers | consecutive pairs | speaker change % | emotion change % |
| --- | --- | --- | --- | --- | --- | --- |
| train | 9989 | 1038 | 260 | 8951 | 77.6 | 60.0 |
| dev | 1109 | 114 | 47 | 995 | 78.7 | 61.0 |
| test | 2610 | 280 | 100 | 2330 | 75.8 | 58.1 |
| all | 13708 | 1432 | 304 | 12276 | 77.3 | 59.7 |

## Emotion classes (count, share)

| split | neutral | surprise | fear | sadness | joy | disgust | anger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4710 (47.2 %) | 1205 (12.1 %) | 268 (2.7 %) | 683 (6.8 %) | 1743 (17.4 %) | 271 (2.7 %) | 1109 (11.1 %) |
| dev | 470 (42.4 %) | 150 (13.5 %) | 40 (3.6 %) | 111 (10.0 %) | 163 (14.7 %) | 22 (2.0 %) | 153 (13.8 %) |
| test | 1256 (48.1 %) | 281 (10.8 %) | 50 (1.9 %) | 208 (8.0 %) | 402 (15.4 %) | 68 (2.6 %) | 345 (13.2 %) |
| all | 6436 (47.0 %) | 1636 (11.9 %) | 358 (2.6 %) | 1002 (7.3 %) | 2308 (16.8 %) | 361 (2.6 %) | 1607 (11.7 %) |

## Sentiment classes (count, share)

| split | neutral | positive | negative |
| --- | --- | --- | --- |
| train | 4710 (47.2 %) | 2334 (23.4 %) | 2945 (29.5 %) |
| dev | 470 (42.4 %) | 233 (21.0 %) | 406 (36.6 %) |
| test | 1256 (48.1 %) | 521 (20.0 %) | 833 (31.9 %) |
| all | 6436 (47.0 %) | 3088 (22.5 %) | 4184 (30.5 %) |

## Subtitle durations (s)

| split | median | mean | p90 | p95 | max | <= 5 s % | <= 8 s % | <= 15 s % | > 15 s clips | non-positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2.46 | 3.14 | 6.34 | 7.92 | 41.04 | 83.4 | 95.1 | 99.7 | 25 | 0 |
| dev | 2.45 | 3.12 | 5.97 | 7.60 | 28.53 | 83.9 | 95.7 | 99.5 | 6 | 0 |
| test | 2.46 | 3.31 | 6.26 | 7.76 | 304.94 | 82.8 | 95.5 | 99.8 | 6 | 1 |
| all | 2.46 | 3.17 | 6.30 | 7.88 | 304.94 | 83.3 | 95.2 | 99.7 | 37 | 1 |

## Words per utterance

| split | mean | median | p95 | max |
| --- | --- | --- | --- | --- |
| train | 7.95 | 6 | 20 | 69 |
| dev | 7.91 | 6 | 20 | 37 |
| test | 8.21 | 7 | 21 | 45 |
| all | 8.00 | 6 | 20 | 69 |

## Timestamp anomalies (duration <= 0 or > 60 s)

| split | dialogue_id | utterance_id | start s | end s | duration s | words |
| --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 4 | 811.060 | 1116.000 | 304.940 | 7 |
| test | 155 | 3 | 752.632 | 752.632 | 0.000 | 3 |
| test | 220 | 0 | 104.270 | 339.338 | 235.068 | 3 |

MELD README emotion class weights, index order ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']: `[4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0]`.
