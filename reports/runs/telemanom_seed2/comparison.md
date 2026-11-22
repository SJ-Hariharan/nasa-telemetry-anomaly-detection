All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| telemanom | 0.623 | 0.242 | 0.349 | 0.021 | 0.837 | 0.692 | 0.758 | 72 | 32 | 14 | 0.275 | 94.000 | 0.704 | 0.477 |

61 calibratable channels (excluding 20 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| telemanom | 0.633 | 0.264 | 0.373 | 0.021 | 0.806 | 0.725 | 0.763 | 58 | 22 | 14 | 0.363 | 79.000 | 0.706 | 0.482 |