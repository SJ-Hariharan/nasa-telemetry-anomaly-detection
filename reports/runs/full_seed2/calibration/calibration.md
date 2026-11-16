# Calibration studies: run `full_seed2`

## Threshold-estimation uncertainty (100 circular block-bootstrap resamples, block length 50, of each channel's validation scores; one-sided near the top, read as lower bounds)

| Method | Metric | Point estimate | 90% interval |
|---|---|---|---|
| baseline | precision | 0.301 | [0.268, 0.303] |
| baseline | recall | 0.195 | [0.197, 0.212] |
| baseline | f1 | 0.237 | [0.229, 0.247] |
| baseline | event_f1 | 0.056 | [0.043, 0.060] |
| isolation_forest | precision | 0.302 | [0.264, 0.308] |
| isolation_forest | recall | 0.350 | [0.342, 0.384] |
| isolation_forest | f1 | 0.324 | [0.300, 0.340] |
| isolation_forest | event_f1 | 0.060 | [0.043, 0.064] |
| lstm | precision | 0.280 | [0.265, 0.278] |
| lstm | recall | 0.704 | [0.704, 0.728] |
| lstm | f1 | 0.400 | [0.386, 0.402] |
| lstm | event_f1 | 0.079 | [0.056, 0.077] |

## Leave-channels-out calibration (held-out folds, rule chosen on the other folds by event F1)

| Method | Channels | Rule | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | all (81) | default q=0.995, m=1 | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 2881 |
| baseline | all (81) | LCO-selected | 0.325 | 0.177 | 0.229 | 0.053 | 0.037 | 0.596 | 0.070 | 1613 |
| baseline | calibratable (59) | default q=0.995, m=1 | 0.365 | 0.159 | 0.221 | 0.042 | 0.033 | 0.769 | 0.063 | 1778 |
| baseline | calibratable (59) | LCO-selected | 0.379 | 0.137 | 0.201 | 0.034 | 0.071 | 0.462 | 0.124 | 468 |
| isolation_forest | all (81) | default q=0.995, m=1 | 0.302 | 0.350 | 0.324 | 0.116 | 0.031 | 0.683 | 0.060 | 2195 |
| isolation_forest | all (81) | LCO-selected | 0.235 | 0.118 | 0.157 | 0.055 | 0.075 | 0.221 | 0.112 | 283 |
| isolation_forest | calibratable (59) | default q=0.995, m=1 | 0.364 | 0.423 | 0.391 | 0.112 | 0.038 | 0.769 | 0.072 | 1536 |
| isolation_forest | calibratable (59) | LCO-selected | 0.314 | 0.139 | 0.193 | 0.046 | 0.078 | 0.205 | 0.113 | 188 |
| lstm | all (81) | default q=0.995, m=1 | 0.280 | 0.704 | 0.400 | 0.260 | 0.041 | 0.942 | 0.079 | 2281 |
| lstm | all (81) | LCO-selected | 0.358 | 0.497 | 0.416 | 0.128 | 0.118 | 0.558 | 0.195 | 434 |
| lstm | calibratable (59) | default q=0.995, m=1 | 0.318 | 0.651 | 0.427 | 0.212 | 0.036 | 0.923 | 0.070 | 1911 |
| lstm | calibratable (59) | LCO-selected | 0.518 | 0.398 | 0.450 | 0.056 | 0.086 | 0.410 | 0.142 | 342 |

Rules chosen per fold (q, m):

* baseline: fold 0: (0.99, 1.5), fold 1: (0.995, 5.0), fold 2: (0.995, 5.0), fold 3: (0.995, 5.0), fold 4: (0.99, 5.0)
* isolation_forest: fold 0: (0.99, 2.0), fold 1: (1.0, 1.5), fold 2: (0.995, 1.5), fold 3: (0.99, 2.0), fold 4: (0.99, 2.0)
* lstm: fold 0: (1.0, 5.0), fold 1: (1.0, 5.0), fold 2: (1.0, 5.0), fold 3: (1.0, 5.0), fold 4: (1.0, 5.0)
