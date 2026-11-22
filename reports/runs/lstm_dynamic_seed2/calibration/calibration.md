# Calibration studies: run `lstm_dynamic_seed2`


## lstm_dynamic: dynamic-threshold pruning parameter p (leave-channels-out, event F1)

| Channels | Rule | Precision | Recall | F1 | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|
| all (81) | default p=0.13 | 0.578 | 0.218 | 0.316 | 0.818 | 0.606 | 0.696 | 14 |
| all (81) | LCO-selected p | 0.589 | 0.207 | 0.307 | 0.797 | 0.567 | 0.663 | 15 |
| calibratable (60) | default p=0.13 | 0.565 | 0.201 | 0.297 | 0.770 | 0.603 | 0.676 | 14 |
| calibratable (60) | LCO-selected p | 0.579 | 0.188 | 0.284 | 0.741 | 0.551 | 0.632 | 15 |

p chosen per fold: fold 0: 0.13, fold 1: 0.2, fold 2: 0.13, fold 3: 0.2, fold 4: 0.13

Post-hoc trade-off over all channels (diagnostic only, not used to choose p):

| p | Event recall | Event precision | Event F1 | Event FP | Point F1 |
|---|---|---|---|---|---|
| 0.05 | 0.721 | 0.630 | 0.673 | 44 | 0.356 |
| 0.13 | 0.606 | 0.818 | 0.696 | 14 | 0.316 |
| 0.2 | 0.567 | 0.894 | 0.694 | 7 | 0.309 |
| 0.3 | 0.510 | 0.914 | 0.654 | 5 | 0.284 |
| 0.4 | 0.433 | 0.978 | 0.600 | 1 | 0.239 |
| 0.5 | 0.375 | 0.975 | 0.542 | 1 | 0.209 |
