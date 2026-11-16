All 81 channels:

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | roc_auc_macro | pr_auc_macro | roc_auc_pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 86 | 18 | 2881 | 0.590 | 0.260 | 0.597 |
| isolation_forest | 0.274 | 0.325 | 0.298 | 0.123 | 0.032 | 0.702 | 0.062 | 73 | 31 | 2180 | 0.629 | 0.290 | 0.590 |
| lstm | 0.262 | 0.704 | 0.382 | 0.283 | 0.032 | 0.933 | 0.063 | 97 | 7 | 2897 | 0.690 | 0.456 | 0.743 |

59 calibratable channels (excluding 22 with a constant training signal / degenerate validation scores):

| method | precision | recall | f1 | fpr | event_precision | event_recall | event_f1 | event_tp | event_fn | event_fp | roc_auc_macro | pr_auc_macro | roc_auc_pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.365 | 0.159 | 0.221 | 0.042 | 0.033 | 0.769 | 0.063 | 60 | 18 | 1778 | 0.530 | 0.212 | 0.580 |
| isolation_forest | 0.333 | 0.391 | 0.360 | 0.119 | 0.035 | 0.795 | 0.067 | 62 | 16 | 1702 | 0.651 | 0.329 | 0.659 |
| lstm | 0.311 | 0.650 | 0.420 | 0.219 | 0.035 | 0.910 | 0.068 | 71 | 7 | 1932 | 0.660 | 0.423 | 0.752 |