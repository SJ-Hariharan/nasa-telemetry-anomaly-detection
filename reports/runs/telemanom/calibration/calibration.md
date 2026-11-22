# Calibration studies: run `telemanom`

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

## Leave-channels-out calibration (held-out folds, rule chosen on the other folds by event F1)

| Method | Channels | Rule | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | all (81) | default q=0.995, m=1 | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 2881 |
| baseline | all (81) | LCO-selected | 0.325 | 0.177 | 0.229 | 0.053 | 0.037 | 0.596 | 0.070 | 1613 |
| baseline | calibratable (55) | default q=0.995, m=1 | 0.137 | 0.020 | 0.035 | 0.017 | 0.033 | 0.795 | 0.064 | 1677 |
| baseline | calibratable (55) | LCO-selected | 0.046 | 0.004 | 0.007 | 0.010 | 0.083 | 0.466 | 0.140 | 378 |
| isolation_forest | all (81) | default q=0.995, m=1 | 0.268 | 0.299 | 0.283 | 0.117 | 0.034 | 0.692 | 0.065 | 2025 |
| isolation_forest | all (81) | LCO-selected | 0.271 | 0.183 | 0.219 | 0.070 | 0.072 | 0.385 | 0.121 | 518 |
| isolation_forest | calibratable (55) | default q=0.995, m=1 | 0.307 | 0.301 | 0.304 | 0.090 | 0.037 | 0.767 | 0.070 | 1478 |
| isolation_forest | calibratable (55) | LCO-selected | 0.327 | 0.154 | 0.209 | 0.042 | 0.109 | 0.384 | 0.169 | 230 |

Rules chosen per fold (q, m):

* baseline: fold 0: (0.99, 1.5), fold 1: (0.995, 5.0), fold 2: (0.995, 5.0), fold 3: (0.995, 5.0), fold 4: (0.99, 5.0)
* isolation_forest: fold 0: (0.99, 1.5), fold 1: (0.99, 1.5), fold 2: (0.99, 1.5), fold 3: (0.99, 1.5), fold 4: (1.0, 1.5)

## telemanom: dynamic-threshold pruning parameter p (leave-channels-out, event F1)

| Channels | Rule | Precision | Recall | F1 | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|
| all (81) | default p=0.13 | 0.593 | 0.241 | 0.343 | 0.787 | 0.673 | 0.725 | 19 |
| all (81) | LCO-selected p | 0.588 | 0.233 | 0.333 | 0.782 | 0.654 | 0.712 | 19 |
| calibratable (55) | default p=0.13 | 0.611 | 0.273 | 0.377 | 0.712 | 0.644 | 0.676 | 19 |
| calibratable (55) | LCO-selected p | 0.606 | 0.260 | 0.363 | 0.703 | 0.616 | 0.657 | 19 |

p chosen per fold: fold 0: 0.13, fold 1: 0.13, fold 2: 0.13, fold 3: 0.2, fold 4: 0.13

Post-hoc trade-off over all channels (diagnostic only, not used to choose p):

| p | Event recall | Event precision | Event F1 | Event FP | Point F1 |
|---|---|---|---|---|---|
| 0.05 | 0.750 | 0.557 | 0.639 | 62 | 0.365 |
| 0.13 | 0.673 | 0.787 | 0.725 | 19 | 0.343 |
| 0.2 | 0.625 | 0.844 | 0.718 | 12 | 0.323 |
| 0.3 | 0.510 | 0.981 | 0.671 | 1 | 0.283 |
| 0.4 | 0.462 | 1.000 | 0.632 | 0 | 0.257 |
| 0.5 | 0.413 | 1.000 | 0.585 | 0 | 0.230 |
