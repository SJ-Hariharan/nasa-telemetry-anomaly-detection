All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 86 | 18 | 2881 | 56.540 | 37.000 | 0.590 | 0.260 |
| isolation_forest | 0.268 | 0.299 | 0.283 | 0.117 | 0.034 | 0.692 | 0.065 | 72 | 32 | 2025 | 39.741 | 27.000 | 0.635 | 0.292 |
| telemanom | 0.593 | 0.241 | 0.343 | 0.024 | 0.787 | 0.673 | 0.725 | 70 | 34 | 19 | 0.373 | 84.000 | 0.700 | 0.484 |

55 calibratable channels (excluding 26 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | false_alarms_per_10k | median_decision_latency | roc_auc_macro | pr_auc_macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.137 | 0.020 | 0.035 | 0.017 | 0.033 | 0.795 | 0.064 | 58 | 15 | 1677 | 46.431 | 48.000 | 0.534 | 0.207 |
| isolation_forest | 0.307 | 0.301 | 0.304 | 0.090 | 0.037 | 0.767 | 0.070 | 56 | 17 | 1478 | 40.921 | 31.500 | 0.668 | 0.333 |
| telemanom | 0.611 | 0.273 | 0.377 | 0.023 | 0.712 | 0.644 | 0.676 | 47 | 26 | 19 | 0.526 | 69.000 | 0.701 | 0.493 |