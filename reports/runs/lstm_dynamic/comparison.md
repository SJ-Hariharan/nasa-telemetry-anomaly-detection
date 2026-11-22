All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_dynamic | 0.599 | 0.216 | 0.317 | 0.021 | 0.840 | 0.606 | 0.704 | 63 | 41 | 12 | 0.235 | 119.000 | 0.678 | 0.461 |

61 calibratable channels (excluding 20 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_dynamic | 0.591 | 0.201 | 0.300 | 0.021 | 0.800 | 0.608 | 0.691 | 48 | 31 | 12 | 0.305 | 100.000 | 0.658 | 0.455 |