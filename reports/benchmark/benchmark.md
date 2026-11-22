# Benchmark comparison (NASA SMAP/MSL, 81 channels, test labels used for scoring only)

Protocol: same 81 channels and metrics for every method (micro point metrics; an event is found if any
predicted point overlaps it; a false-alarm event is a predicted run overlapping no label; no point-adjust).
Headline settings are predeclared: validation-quantile threshold (q = 0.995) for static methods, published Telemanom
parameters (p = 0.13) for dynamic ones. Ranking criterion, fixed in advance: mean event F1 over seeds.
Decision latency includes the dynamic threshold's batch delay (up to 69 points).

## Headline (predeclared settings, mean ± std over seeds)

| Method | Threshold | Seeds | Event F1 | Event recall | Events found | Event precision | False-alarm events | False alarms / 10k pts | Point F1 | Point precision | Point recall | Median decision latency (pts) | PR-AUC (macro) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| telemanom | dynamic | 3 | 0.733 ± 0.022 | 0.673 ± 0.019 | 70 ± 2/104 | 0.805 ± 0.028 | 17 ± 3 | 0.3 ± 0.1 | 0.335 ± 0.019 | 0.593 ± 0.029 | 0.234 ± 0.014 | 94 ± 9 | 0.477 ± 0.007 |
| lstm_dynamic | dynamic | 3 | 0.716 ± 0.028 | 0.619 ± 0.022 | 64 ± 2/104 | 0.851 ± 0.039 | 11 ± 3 | 0.2 ± 0.1 | 0.322 ± 0.009 | 0.595 ± 0.015 | 0.220 ± 0.007 | 121 ± 3 | 0.464 ± 0.003 |
| lstm | static | 3 | 0.080 ± 0.018 | 0.936 ± 0.006 | 97 ± 1/104 | 0.042 ± 0.010 | 2314 ± 567 | 45.4 ± 11.1 | 0.390 ± 0.009 | 0.270 ± 0.009 | 0.699 ± 0.009 | 19 ± 3 | 0.456 ± 0.000 |
| isolation_forest | static | 3 | 0.062 ± 0.003 | 0.692 ± 0.010 | 72 ± 1/104 | 0.033 ± 0.002 | 2133 ± 94 | 41.9 ± 1.8 | 0.302 ± 0.021 | 0.282 ± 0.018 | 0.325 ± 0.025 | 27 ± 0 | 0.289 ± 0.003 |
| baseline | static | 3 | 0.056 ± 0.000 | 0.827 ± 0.000 | 86 ± 0/104 | 0.029 ± 0.000 | 2881 ± 0 | 56.5 ± 0.0 | 0.237 ± 0.000 | 0.301 ± 0.000 | 0.195 ± 0.000 | 37 ± 0 | 0.260 ± 0.000 |

Methods: **telemanom** - Telemanom-aligned LSTM (l_s 250, 2x80, 10-step) + Telemanom dynamic threshold; **lstm_dynamic** - Same saved reference LSTM models + Telemanom dynamic threshold (ablation, no retraining); **lstm** - Reference LSTM forecaster (window 100, 2x64), validation-quantile threshold; **isolation_forest** - Isolation Forest on trailing-window features, validation-quantile threshold; **baseline** - Rolling z-score, validation-quantile threshold

## Operating points (seed-42 run of each method; LCO = rule chosen on other channel folds)

| Method | Operating point | Event recall | Event precision | Event F1 | False-alarm events | Point F1 |
|---|---|---|---|---|---|---|
| baseline | predeclared | 0.827 | 0.029 | 0.056 | 2881 | 0.237 |
| baseline | leave-channels-out | 0.596 | 0.037 | 0.070 | 1613 | 0.229 |
| isolation_forest | predeclared | 0.692 | 0.034 | 0.065 | 2025 | 0.283 |
| isolation_forest | leave-channels-out | 0.385 | 0.072 | 0.121 | 518 | 0.219 |
| lstm | predeclared | 0.933 | 0.052 | 0.099 | 1764 | 0.387 |
| lstm | leave-channels-out | 0.548 | 0.147 | 0.231 | 332 | 0.407 |
| lstm_dynamic | predeclared | 0.606 | 0.840 | 0.704 | 12 | 0.317 |
| lstm_dynamic | leave-channels-out | 0.577 | 0.845 | 0.686 | 11 | 0.316 |
| telemanom | predeclared | 0.673 | 0.787 | 0.725 | 19 | 0.343 |
| telemanom | leave-channels-out | 0.654 | 0.782 | 0.712 | 19 | 0.333 |

## Reproducibility checks

* baseline seed 42: full vs telemanom: identical
* isolation_forest seed 42: full vs telemanom: identical
