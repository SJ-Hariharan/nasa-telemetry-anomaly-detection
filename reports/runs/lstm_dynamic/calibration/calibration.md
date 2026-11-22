# Calibration studies: run `lstm_dynamic`


## lstm_dynamic: dynamic-threshold pruning parameter p (leave-channels-out, event F1)

| Channels | Rule | Precision | Recall | F1 | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|
| all (81) | default p=0.13 | 0.599 | 0.216 | 0.317 | 0.840 | 0.606 | 0.704 | 12 |
| all (81) | LCO-selected p | 0.617 | 0.212 | 0.316 | 0.845 | 0.577 | 0.686 | 11 |
| calibratable (61) | default p=0.13 | 0.591 | 0.201 | 0.300 | 0.800 | 0.608 | 0.691 | 12 |
| calibratable (61) | LCO-selected p | 0.615 | 0.196 | 0.298 | 0.804 | 0.570 | 0.667 | 11 |

p chosen per fold: fold 0: 0.13, fold 1: 0.2, fold 2: 0.13, fold 3: 0.13, fold 4: 0.13

Post-hoc trade-off over all channels (diagnostic only, not used to choose p):

| p | Event recall | Event precision | Event F1 | Event FP | Point F1 |
|---|---|---|---|---|---|
| 0.05 | 0.683 | 0.563 | 0.617 | 55 | 0.350 |
| 0.13 | 0.606 | 0.840 | 0.704 | 12 | 0.317 |
| 0.2 | 0.558 | 0.906 | 0.690 | 6 | 0.302 |
| 0.3 | 0.519 | 0.964 | 0.675 | 2 | 0.289 |
| 0.4 | 0.433 | 0.978 | 0.600 | 1 | 0.238 |
| 0.5 | 0.375 | 1.000 | 0.545 | 0 | 0.207 |
