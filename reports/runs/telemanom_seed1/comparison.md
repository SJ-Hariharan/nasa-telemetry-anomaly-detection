All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| telemanom | 0.564 | 0.217 | 0.314 | 0.024 | 0.791 | 0.654 | 0.716 | 68 | 36 | 18 | 0.353 | 102.500 | 0.695 | 0.469 |

58 calibratable channels (excluding 23 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| telemanom | 0.560 | 0.239 | 0.335 | 0.025 | 0.731 | 0.645 | 0.685 | 49 | 27 | 18 | 0.486 | 84.000 | 0.710 | 0.483 |