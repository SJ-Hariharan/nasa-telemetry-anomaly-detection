# Calibration studies: run `telemanom_seed1`


## telemanom: dynamic-threshold pruning parameter p (leave-channels-out, event F1)

| Channels | Rule | Precision | Recall | F1 | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|
| all (81) | default p=0.13 | 0.564 | 0.217 | 0.314 | 0.791 | 0.654 | 0.716 | 18 |
| all (81) | LCO-selected p | 0.572 | 0.216 | 0.313 | 0.786 | 0.635 | 0.702 | 18 |
| calibratable (58) | default p=0.13 | 0.560 | 0.239 | 0.335 | 0.731 | 0.645 | 0.685 | 18 |
| calibratable (58) | LCO-selected p | 0.571 | 0.237 | 0.335 | 0.723 | 0.618 | 0.667 | 18 |

p chosen per fold: fold 0: 0.13, fold 1: 0.2, fold 2: 0.13, fold 3: 0.13, fold 4: 0.13

Post-hoc trade-off over all channels (diagnostic only, not used to choose p):

| p | Event recall | Event precision | Event F1 | Event FP | Point F1 |
|---|---|---|---|---|---|
| 0.05 | 0.731 | 0.571 | 0.641 | 57 | 0.343 |
| 0.13 | 0.654 | 0.791 | 0.716 | 18 | 0.314 |
| 0.2 | 0.615 | 0.842 | 0.711 | 12 | 0.307 |
| 0.3 | 0.538 | 0.966 | 0.691 | 2 | 0.276 |
| 0.4 | 0.433 | 0.957 | 0.596 | 2 | 0.218 |
| 0.5 | 0.394 | 0.976 | 0.562 | 1 | 0.202 |
