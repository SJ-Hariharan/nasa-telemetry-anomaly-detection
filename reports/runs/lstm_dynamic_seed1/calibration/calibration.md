# Calibration studies: run `lstm_dynamic_seed1`


## lstm_dynamic: dynamic-threshold pruning parameter p (leave-channels-out, event F1)

| Channels | Rule | Precision | Recall | F1 | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|
| all (81) | default p=0.13 | 0.608 | 0.228 | 0.332 | 0.893 | 0.644 | 0.749 | 8 |
| all (81) | LCO-selected p | 0.608 | 0.228 | 0.332 | 0.893 | 0.644 | 0.749 | 8 |
| calibratable (57) | default p=0.13 | 0.620 | 0.241 | 0.347 | 0.852 | 0.613 | 0.713 | 8 |
| calibratable (57) | LCO-selected p | 0.620 | 0.241 | 0.347 | 0.852 | 0.613 | 0.713 | 8 |

p chosen per fold: fold 0: 0.13, fold 1: 0.13, fold 2: 0.13, fold 3: 0.13, fold 4: 0.13

Post-hoc trade-off over all channels (diagnostic only, not used to choose p):

| p | Event recall | Event precision | Event F1 | Event FP | Point F1 |
|---|---|---|---|---|---|
| 0.05 | 0.721 | 0.641 | 0.679 | 42 | 0.349 |
| 0.13 | 0.644 | 0.893 | 0.749 | 8 | 0.332 |
| 0.2 | 0.587 | 0.938 | 0.722 | 4 | 0.311 |
| 0.3 | 0.538 | 0.966 | 0.691 | 2 | 0.289 |
| 0.4 | 0.442 | 0.979 | 0.609 | 1 | 0.240 |
| 0.5 | 0.385 | 0.976 | 0.552 | 1 | 0.204 |
