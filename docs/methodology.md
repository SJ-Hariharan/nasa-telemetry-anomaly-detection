# Methodology

## 1. Problem framing

For each telemetry channel we learn what *nominal* behaviour looks like from the benchmark's train file. We
then assign each test timestep an **anomaly score**, and turn scores into yes/no predictions with a
threshold calibrated without labels. Labels are used only to evaluate. This is unsupervised detection
with a held-out labelled test set, which matches the benchmark's intended use.

## 2. Splits and leakage controls

| Data | Used for | Never used for |
|---|---|---|
| train[0 : split) (first 80%) | fitting scaler and models | — |
| train[split : n) (last 20%) | LSTM early stopping; threshold calibration | gradient updates, scaler stats |
| test | final scoring and evaluation | any fitting, scaling stats or threshold choice |
| test labels | metrics only, attached after prediction | anything upstream |

* **Chronological validation.** Random splits would let the model learn from moments *after* the ones
  it is validated on, and adjacent windows overlap almost completely. That makes validation
  error optimistic.
* **Causal features.** A score at time t uses only x[0..t]. Tests verify this by changing the future
  and checking that past scores stay identical (`test_baseline_is_causal_and_detects_spike`,
  `test_isolation_features_are_causal`, `test_forecast_windows_alignment`).
* **Test series are scored on their own.** The test file is not guaranteed to follow on from the train
  file, so we do not borrow train history. The first W test points get no score and are predicted normal.
* **Benchmark-level leakage we cannot remove.** The authors scaled values with test min/max. See limitations.

## 3. Detectors

| | Baseline | Isolation Forest | LSTM |
|---|---|---|---|
| Input | value only | 17 trailing-window features | value + command flags, W = 100 |
| Learns | nothing (rolling window) | tree ensemble over fit-part features | next-value forecaster |
| Score at t | $\lvert x_t-\mu_{t-W..t-1}\rvert / \max(\sigma_{t-W..t-1}, 0.01)$ | $-$`score_samples` (higher = easier to isolate) | EWMA$_{30}$ of $\lvert x_t-\hat x_t\rvert$ |
| Warm-up | 100 | 49 | 100 |

**Isolation Forest features** (`isolation_features`): lags 0–9, rolling mean/std/min/max over 50 steps,
x_t − rolling mean, first difference, and command activity in the window.

**LSTM**: 2 layers, hidden 64, dropout 0.3, Adam (lr 2e-3), batch 128, MSE loss, gradient clipping 1.0,
≤ 60 epochs with early stopping (patience 6) on chronological validation MSE, ≤ 4,000 training windows per
channel (even stride). Departures from Telemanom (W 250, layers [80, 80], 10-step predictions, nonparametric
dynamic thresholding) were made for CPU budget and simplicity. We do not claim to replicate the paper's numbers.

## 4. Threshold calibration

Per channel and method: `thr = Quantile_0.995(validation scores)`. In words: on held-out, presumed-nominal
data, about 0.5% of points would exceed it. The quantile is fixed a priori in `configs/default.yaml`.
**No value was tuned on test labels.** `sensitivity.csv` shows what would happen at other thresholds
(multipliers 0.5–5×), but it is a post-hoc diagnostic, explicitly not a selection step.

Known weaknesses: the validation part is assumed nominal (it is unlabelled). The LSTM's validation errors
are slightly optimistic because the same split selects its best epoch. And constant channels give a
threshold of ~0, so any change gets flagged.

## 5. Evaluation

* **Point-wise**: confusion matrix over all test timesteps of the 81 channels, micro-aggregated.
  Precision, recall, F1, FPR, FNR, plus accuracy, reported only to show it is misleading.
* **Event-wise** (Hundman et al. style): a labelled sequence is detected if ≥ 1 predicted point falls inside
  it. Each predicted run that overlaps no labelled sequence is one false-positive event.
  Overlapping/adjacent anomalous timesteps therefore count as one incident, not many.
  No "point-adjust".
* **Threshold-free**: ROC-AUC and PR-AUC on `score / channel threshold`, which puts scores from
  different channels on a comparable scale before pooling.
* **Error analysis**: per-channel metrics, per-event detection and latency, false-alarm runs, event
  recall by anomaly class (point / contextual), per-spacecraft breakdown.
* All methods are compared on the **same** channel set. Channels where any method failed would be
  excluded and listed (none failed in the reported run).

## 6. Reproducibility

Seed 42 is reset before every (channel, method) fit. Configs are saved with each run, the dataset revision
and hashes are pinned, and package lower bounds are in `requirements.txt`. PyTorch CPU results should
reproduce closely on the same machine. Bit-exactness across hardware is not guaranteed.
