All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_dynamic | 0.578 | 0.218 | 0.316 | 0.023 | 0.818 | 0.606 | 0.696 | 63 | 41 | 14 | 0.275 | 119.000 | 0.679 | 0.465 |

60 calibratable channels (excluding 21 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_dynamic | 0.565 | 0.201 | 0.297 | 0.023 | 0.770 | 0.603 | 0.676 | 47 | 31 | 14 | 0.358 | 99.000 | 0.660 | 0.459 |