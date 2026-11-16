# Calibration studies: run `full`

## Threshold-estimation uncertainty (100 circular block-bootstrap resamples, block length 50, of each channel's validation scores; one-sided near the top, read as lower bounds)

| Method | Metric | Point estimate | 90% interval |
|---|---|---|---|
| baseline | precision | 0.301 | [0.268, 0.303] |
| baseline | recall | 0.195 | [0.197, 0.212] |
| baseline | f1 | 0.237 | [0.229, 0.247] |
| baseline | event_f1 | 0.056 | [0.043, 0.060] |
| isolation_forest | precision | 0.268 | [0.236, 0.287] |
| isolation_forest | recall | 0.299 | [0.297, 0.363] |
| isolation_forest | f1 | 0.283 | [0.266, 0.318] |
| isolation_forest | event_f1 | 0.065 | [0.046, 0.068] |
| lstm | precision | 0.269 | [0.257, 0.268] |
| lstm | recall | 0.688 | [0.690, 0.727] |
| lstm | f1 | 0.387 | [0.375, 0.389] |
| lstm | event_f1 | 0.099 | [0.080, 0.096] |

## Leave-channels-out calibration (held-out folds, rule chosen on the other folds by event F1)

| Method | Channels | Rule | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | all (81) | default q=0.995, m=1 | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 2881 |
| baseline | all (81) | LCO-selected | 0.325 | 0.177 | 0.229 | 0.053 | 0.037 | 0.596 | 0.070 | 1613 |
| baseline | calibratable (59) | default q=0.995, m=1 | 0.365 | 0.159 | 0.221 | 0.042 | 0.033 | 0.769 | 0.063 | 1778 |
| baseline | calibratable (59) | LCO-selected | 0.379 | 0.137 | 0.201 | 0.034 | 0.071 | 0.462 | 0.124 | 468 |
| isolation_forest | all (81) | default q=0.995, m=1 | 0.268 | 0.299 | 0.283 | 0.117 | 0.034 | 0.692 | 0.065 | 2025 |
| isolation_forest | all (81) | LCO-selected | 0.271 | 0.183 | 0.219 | 0.070 | 0.072 | 0.385 | 0.121 | 518 |
| isolation_forest | calibratable (59) | default q=0.995, m=1 | 0.322 | 0.358 | 0.339 | 0.114 | 0.038 | 0.782 | 0.073 | 1534 |
| isolation_forest | calibratable (59) | LCO-selected | 0.335 | 0.214 | 0.261 | 0.064 | 0.074 | 0.410 | 0.125 | 400 |
| lstm | all (81) | default q=0.995, m=1 | 0.269 | 0.688 | 0.387 | 0.267 | 0.052 | 0.933 | 0.099 | 1764 |
| lstm | all (81) | LCO-selected | 0.343 | 0.500 | 0.407 | 0.137 | 0.147 | 0.548 | 0.231 | 332 |
| lstm | calibratable (59) | default q=0.995, m=1 | 0.321 | 0.629 | 0.425 | 0.202 | 0.040 | 0.910 | 0.077 | 1701 |
| lstm | calibratable (59) | LCO-selected | 0.514 | 0.402 | 0.451 | 0.058 | 0.093 | 0.397 | 0.150 | 303 |

Rules chosen per fold (q, m):

* baseline: fold 0: (0.99, 1.5), fold 1: (0.995, 5.0), fold 2: (0.995, 5.0), fold 3: (0.995, 5.0), fold 4: (0.99, 5.0)
* isolation_forest: fold 0: (0.99, 1.5), fold 1: (0.99, 1.5), fold 2: (0.99, 1.5), fold 3: (0.99, 1.5), fold 4: (1.0, 1.5)
* lstm: fold 0: (1.0, 5.0), fold 1: (1.0, 5.0), fold 2: (1.0, 5.0), fold 3: (1.0, 5.0), fold 4: (0.999, 5.0)
