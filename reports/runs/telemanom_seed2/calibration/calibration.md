# Calibration studies: run `telemanom_seed2`


## telemanom: dynamic-threshold pruning parameter p (leave-channels-out, event F1)

| Channels | Rule | Precision | Recall | F1 | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|
| all (81) | default p=0.13 | 0.623 | 0.242 | 0.349 | 0.837 | 0.692 | 0.758 | 14 |
| all (81) | LCO-selected p | 0.644 | 0.215 | 0.323 | 0.915 | 0.625 | 0.743 | 6 |
| calibratable (61) | default p=0.13 | 0.633 | 0.264 | 0.373 | 0.806 | 0.725 | 0.763 | 14 |
| calibratable (61) | LCO-selected p | 0.663 | 0.228 | 0.339 | 0.895 | 0.637 | 0.745 | 6 |

p chosen per fold: fold 0: 0.13, fold 1: 0.13, fold 2: 0.13, fold 3: 0.2, fold 4: 0.2

Post-hoc trade-off over all channels (diagnostic only, not used to choose p):

| p | Event recall | Event precision | Event F1 | Event FP | Point F1 |
|---|---|---|---|---|---|
| 0.05 | 0.740 | 0.626 | 0.678 | 46 | 0.369 |
| 0.13 | 0.692 | 0.837 | 0.758 | 14 | 0.349 |
| 0.2 | 0.625 | 0.956 | 0.756 | 3 | 0.313 |
| 0.3 | 0.558 | 1.000 | 0.716 | 0 | 0.287 |
| 0.4 | 0.442 | 1.000 | 0.613 | 0 | 0.229 |
| 0.5 | 0.413 | 1.000 | 0.585 | 0 | 0.207 |
