All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | roc_auc_macro | pr_auc_macro | roc_auc_pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 86 | 18 | 2881 | 0.590 | 0.260 | 0.597 |
| isolation_forest | 0.302 | 0.350 | 0.324 | 0.116 | 0.031 | 0.683 | 0.060 | 71 | 33 | 2195 | 0.626 | 0.286 | 0.593 |
| lstm | 0.280 | 0.704 | 0.400 | 0.260 | 0.041 | 0.942 | 0.079 | 98 | 6 | 2281 | 0.679 | 0.456 | 0.755 |

59 calibratable channels (excluding 22 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | roc_auc_macro | pr_auc_macro | roc_auc_pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.365 | 0.159 | 0.221 | 0.042 | 0.033 | 0.769 | 0.063 | 60 | 18 | 1778 | 0.530 | 0.212 | 0.580 |
| isolation_forest | 0.364 | 0.423 | 0.391 | 0.112 | 0.038 | 0.769 | 0.072 | 60 | 18 | 1536 | 0.644 | 0.320 | 0.659 |
| lstm | 0.318 | 0.651 | 0.427 | 0.212 | 0.036 | 0.923 | 0.070 | 72 | 6 | 1911 | 0.658 | 0.423 | 0.754 |