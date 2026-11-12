All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | roc_auc_macro | pr_auc_macro | roc_auc_pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 86 | 18 | 2881 | 0.590 | 0.260 | 0.597 |
| isolation_forest | 0.268 | 0.299 | 0.283 | 0.117 | 0.034 | 0.692 | 0.065 | 72 | 32 | 2025 | 0.635 | 0.292 | 0.586 |
| lstm | 0.269 | 0.688 | 0.387 | 0.267 | 0.052 | 0.933 | 0.099 | 97 | 7 | 1764 | 0.677 | 0.456 | 0.749 |

59 calibratable channels (excluding 22 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | roc_auc_macro | pr_auc_macro | roc_auc_pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.365 | 0.159 | 0.221 | 0.042 | 0.033 | 0.769 | 0.063 | 60 | 18 | 1778 | 0.530 | 0.212 | 0.580 |
| isolation_forest | 0.322 | 0.358 | 0.339 | 0.114 | 0.038 | 0.782 | 0.073 | 61 | 17 | 1534 | 0.655 | 0.329 | 0.656 |
| lstm | 0.321 | 0.629 | 0.425 | 0.202 | 0.040 | 0.910 | 0.077 | 71 | 7 | 1701 | 0.648 | 0.423 | 0.750 |