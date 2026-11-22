# Experiments

Every number below was produced by this repository's code on the real benchmark (81 labelled channels, 104 labelled
anomaly sequences after merging P-2's overlapping rows). Nothing is copied from papers. Raw outputs:
`reports/runs/<run>/`; cross-run table: `reports/benchmark/benchmark.md` (`python -m src.cli compare`).

## Protocol (fixed before the new runs)

* Per channel: chronological 80/20 split of the train file into fit / validation; scalers and models fitted on the fit
  part; test labels are joined only after every prediction is made (`test_thresholds_do_not_depend_on_test_labels`).
* Static methods (`baseline`, `isolation_forest`, `lstm`): threshold = 99.5% quantile of validation scores.
* Dynamic methods (`lstm_dynamic`, `telemanom`): Telemanom's label-free threshold with the published constants
  (p = 0.13, 30 x 70 windows, buffer 100). Nothing was tuned on test labels.
* Metrics: micro point precision/recall/F1; event metrics where a labelled sequence is found if any flagged sample
  overlaps it and a false-alarm event is a flagged run overlapping no label (the paper's convention; no point-adjust).
  Decision latency includes the dynamic threshold's batch delay.
* Ranking criterion, chosen in advance: mean event F1 over seeds. Point metrics, false alarms and latency are always
  reported next to it.
* Every method is scored on all 81 channels. D-12 has only 312 training samples, too few for one l_s = 250
  training window: the Telemanom-aligned method raises no alarm there and its event counts as missed (not excluded).

## Runs

| Run | Config | What | Status |
|---|---|---|---|
| `full`, `full_seed1`, `full_seed2` | `default.yaml`, `seed1/2.yaml` | reference: z-score, Isolation Forest, LSTM (static thresholds) | published reference, unchanged |
| `lstm_dynamic`, `_seed1`, `_seed2` | `lstm_dynamic*.yaml` | the reference runs' **saved LSTM models** + dynamic threshold (`derive`, no training) | new |
| `telemanom`, `_seed1`, `_seed2` | `telemanom*.yaml` | Telemanom-aligned LSTM (l_s 250, 2x80, 10-step) + dynamic threshold; `telemanom` also re-runs z-score and Isolation Forest | new |

`lstm_dynamic` recomputes forecasts from the saved models and checks them against the reference run's stored forecasts:
maximum difference 0.0 on all 81 channels in all three seeds, so it changes only the decision rule. The z-score and
Isolation Forest re-runs inside `telemanom` reproduce `full` exactly (same F1, event F1 and false-alarm count).

## Headline (all 81 channels, mean ± std over 3 seeds)

| Method | Event F1 | Events found (of 104) | False-alarm events | Event precision | Point F1 | Point recall | Median decision latency (samples) |
|---|---|---|---|---|---|---|---|
| Telemanom-aligned LSTM + dynamic threshold (`telemanom`) | 0.733 ± 0.022 | 70 ± 2 | 17 ± 3 | 0.805 ± 0.028 | 0.335 ± 0.019 | 0.234 ± 0.014 | 94 |
| Reference LSTM models + dynamic threshold (ablation) (`lstm_dynamic`) | 0.716 ± 0.028 | 64 ± 2 | 11 ± 3 | 0.851 ± 0.039 | 0.322 ± 0.009 | 0.220 ± 0.007 | 121 |
| Reference LSTM, static threshold (`lstm`) | 0.080 ± 0.018 | 97 ± 1 | 2314 ± 567 | 0.042 ± 0.010 | 0.390 ± 0.009 | 0.699 ± 0.009 | 19 |
| Isolation Forest, static threshold (`isolation_forest`) | 0.062 ± 0.003 | 72 ± 1 | 2133 ± 94 | 0.033 ± 0.002 | 0.302 ± 0.021 | 0.325 ± 0.025 | 27 |
| Rolling z-score, static threshold (`baseline`) | 0.056 ± 0.000 | 86 ± 0 | 2881 ± 0 | 0.029 ± 0.000 | 0.237 ± 0.000 | 0.195 ± 0.000 | 37 |

On the 59 channels whose static threshold is calibratable in the reference run (same set for every method; see
`reports/benchmark/runs.csv`, subset `calibratable`) the ordering is the same.

## What changed the result: the threshold, not the network

* **Same models, new rule.** `lstm` → `lstm_dynamic` keeps every weight and changes only the decision rule:
  false-alarm events fall from 2314 to 11 (mean over seeds), event F1 rises from
  0.080 to 0.716.
* **New network, same rule.** `lstm_dynamic` → `telemanom` changes only the forecaster. Event F1 moves by less than
  the seed spread; the Telemanom-aligned model finds a few more events and raises a few more false alarms.
  Under this protocol it does not clearly beat the cheaper ablation.
* **The cost.** Dynamic methods find fewer labelled events (64-70 of 104 vs 97 for the reference LSTM), have
  lower point recall (~0.22 vs 0.70) and lower point F1 (~0.33 vs 0.39), and raise alerts later (median decision
  latency ~90-120 samples vs ~19), partly because decisions are released per 70-sample batch.

## Why the reference LSTM raised 1,764 false alarms (seed 42)

* Thresholds flag a median 0.54% of validation points, but the median **test** false-positive rate is 6.3%: test
  errors are wider than the short validation segment (20% of a train file, often a few hundred samples) suggested.
* 59% of false-alarm runs peak below 1.5x the threshold (borderline crossings) and 38% last 5 samples or fewer
  (flicker); median length 9 samples.
* They are concentrated: 10 channels produce 63%. Uncalibratable (constant-training) channels account for only 63.
* Validation-vs-test score shift is a partial explanation only: KS distance correlates with the LSTM's point FPR
  (Spearman 0.51) but not with its false-alarm event count (-0.31).

The dynamic threshold addresses exactly these: epsilon follows recent errors, pruning keeps only sequences clearly
separated from the largest normal error, flags are padded and merged, and single points are dropped.

## Operating points

Leave-channels-out (LCO): for each of 5 channel folds, the rule maximising event F1 on the other 4 folds' labels is
applied to the held-out fold.

| Method | Operating point | Events found | False-alarm events | Event F1 | Point F1 |
|---|---|---|---|---|---|
| `baseline` | predeclared | 86 | 2881 | 0.056 | 0.237 |
| `baseline` | leave-channels-out | 62 | 1613 | 0.070 | 0.229 |
| `isolation_forest` | predeclared | 72 | 2025 | 0.065 | 0.283 |
| `isolation_forest` | leave-channels-out | 40 | 518 | 0.121 | 0.219 |
| `lstm` | predeclared | 97 | 1764 | 0.099 | 0.387 |
| `lstm` | leave-channels-out | 57 | 332 | 0.231 | 0.407 |
| `lstm_dynamic` | predeclared | 63 | 12 | 0.704 | 0.317 |
| `lstm_dynamic` | leave-channels-out | 60 | 11 | 0.686 | 0.316 |
| `telemanom` | predeclared | 70 | 19 | 0.725 | 0.343 |
| `telemanom` | leave-channels-out | 68 | 19 | 0.712 | 0.333 |

LCO keeps p = 0.13 in most folds for both dynamic methods. For the static methods LCO moves to much stricter
thresholds, which cuts false alarms by 1.8-5x but leaves hundreds. Post-hoc p curves (`reports/runs/<run>/calibration/
dynamic_p_curve.csv`, labelled diagnostic, never used to choose) show the trade-off: for `lstm_dynamic` (seed 42)
p = 0.05 gives 71/104 events with 55 false alarms; p = 0.3 gives 54/104 with 2.

## Runtime (Apple 8-core CPU, no GPU)

| Step | Cost |
|---|---|
| Reference run (`full`) | ~1.0 CPU-hour (sum of per-channel times) |
| Telemanom-aligned run, per seed | ~7 single-thread process-hours; ~2 h wall-clock with 4 parallel shards |
| `derive` (lstm_dynamic) | ~17 min per seed (inference only) |
| Calibration study, per run | 1-2 min |

GPU training was not available here and is not verified; no speed-up is claimed.

## Not done / not verified

* Exact reproduction of the paper: not claimed (differences in `docs/methodology.md` §8).
* Additional modern models (e.g. TranAD): not added. Published SMAP/MSL numbers for such models commonly use
  point-adjusted F1 and a different input setup, so they are not comparable with these numbers without re-running
  them under this protocol, which we did not do.
* Colab/Kaggle and CUDA execution: not run (the notebook was executed locally in smoke mode only).
