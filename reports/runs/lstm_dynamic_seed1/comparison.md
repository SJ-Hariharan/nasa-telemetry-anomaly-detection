All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_dynamic | 0.608 | 0.228 | 0.332 | 0.021 | 0.893 | 0.644 | 0.749 | 67 | 37 | 8 | 0.157 | 124.000 | 0.691 | 0.467 |

57 calibratable channels (excluding 24 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_dynamic | 0.620 | 0.241 | 0.347 | 0.020 | 0.852 | 0.613 | 0.713 | 46 | 29 | 8 | 0.217 | 99.000 | 0.680 | 0.464 |