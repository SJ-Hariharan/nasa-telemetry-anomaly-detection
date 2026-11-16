# Calibration studies: run `full_seed1`

## Threshold-estimation uncertainty (100 circular block-bootstrap resamples, block length 50, of each channel's validation scores; one-sided near the top, read as lower bounds)

| Method | Metric | Point estimate | 90% interval |
|---|---|---|---|
| baseline | precision | 0.301 | [0.268, 0.303] |
| baseline | recall | 0.195 | [0.197, 0.212] |
| baseline | f1 | 0.237 | [0.229, 0.247] |
| baseline | event_f1 | 0.056 | [0.043, 0.060] |
| isolation_forest | precision | 0.274 | [0.245, 0.292] |
| isolation_forest | recall | 0.325 | [0.318, 0.380] |
| isolation_forest | f1 | 0.298 | [0.280, 0.329] |
| isolation_forest | event_f1 | 0.062 | [0.045, 0.066] |
| lstm | precision | 0.262 | [0.249, 0.262] |
| lstm | recall | 0.704 | [0.706, 0.727] |
| lstm | f1 | 0.382 | [0.369, 0.385] |
| lstm | event_f1 | 0.063 | [0.057, 0.083] |

## Leave-channels-out calibration (held-out folds, rule chosen on the other folds by event F1)

| Method | Channels | Rule | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | all (81) | default q=0.995, m=1 | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 2881 |
| baseline | all (81) | LCO-selected | 0.325 | 0.177 | 0.229 | 0.053 | 0.037 | 0.596 | 0.070 | 1613 |
| baseline | calibratable (59) | default q=0.995, m=1 | 0.365 | 0.159 | 0.221 | 0.042 | 0.033 | 0.769 | 0.063 | 1778 |
| baseline | calibratable (59) | LCO-selected | 0.379 | 0.137 | 0.201 | 0.034 | 0.071 | 0.462 | 0.124 | 468 |
| isolation_forest | all (81) | default q=0.995, m=1 | 0.274 | 0.325 | 0.298 | 0.123 | 0.032 | 0.702 | 0.062 | 2180 |
| isolation_forest | all (81) | LCO-selected | 0.236 | 0.118 | 0.158 | 0.055 | 0.104 | 0.221 | 0.142 | 198 |
| isolation_forest | calibratable (59) | default q=0.995, m=1 | 0.333 | 0.391 | 0.360 | 0.119 | 0.035 | 0.795 | 0.067 | 1702 |
| isolation_forest | calibratable (59) | LCO-selected | 0.309 | 0.134 | 0.187 | 0.046 | 0.123 | 0.205 | 0.154 | 114 |
| lstm | all (81) | default q=0.995, m=1 | 0.262 | 0.704 | 0.382 | 0.283 | 0.032 | 0.933 | 0.063 | 2897 |
| lstm | all (81) | LCO-selected | 0.350 | 0.495 | 0.410 | 0.132 | 0.126 | 0.519 | 0.203 | 375 |
| lstm | calibratable (59) | default q=0.995, m=1 | 0.311 | 0.650 | 0.420 | 0.219 | 0.035 | 0.910 | 0.068 | 1932 |
| lstm | calibratable (59) | LCO-selected | 0.555 | 0.394 | 0.461 | 0.048 | 0.073 | 0.359 | 0.122 | 354 |

Rules chosen per fold (q, m):

* baseline: fold 0: (0.99, 1.5), fold 1: (0.995, 5.0), fold 2: (0.995, 5.0), fold 3: (0.995, 5.0), fold 4: (0.99, 5.0)
* isolation_forest: fold 0: (0.99, 2.0), fold 1: (1.0, 1.5), fold 2: (0.99, 2.0), fold 3: (0.99, 2.0), fold 4: (0.99, 2.0)
* lstm: fold 0: (1.0, 5.0), fold 1: (1.0, 5.0), fold 2: (1.0, 5.0), fold 3: (1.0, 5.0), fold 4: (1.0, 5.0)
