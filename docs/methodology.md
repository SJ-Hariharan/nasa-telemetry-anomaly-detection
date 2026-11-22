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
are slightly optimistic because the same split selects its best epoch. Constant channels give a
threshold of ~0, so any change gets flagged. Such thresholds are marked `calibratable: false` in the run summary
and in each saved model, and inference warns when it uses one. Every validation set is small (62–862 points), so
a 99.5% quantile rests on fewer than 5 expected exceedances in **every** channel. §7 measures what that costs.

## 5. Evaluation

* **Point-wise**: confusion matrix over all test timesteps of the 81 channels, micro-aggregated.
  Precision, recall, F1, FPR, FNR, plus accuracy, reported only to show it is misleading.
* **Event-wise** (Hundman et al. style): a labelled sequence is detected if ≥ 1 predicted point falls inside
  it. Each predicted run that overlaps no labelled sequence is one false-positive event.
  Overlapping/adjacent anomalous timesteps therefore count as one incident, not many.
  No "point-adjust".
* **Threshold-free**: ROC-AUC and PR-AUC computed **per channel** on raw scores (warm-up positions excluded)
  and averaged over channels (macro), with a bootstrap-over-channels 95% interval. A pooled variant
  (`score / channel threshold`, concatenated) is also written, for reference only. Dividing by a per-channel
  threshold does not make ranks comparable across channels, and on this run it reversed the baseline/IF order.
* **Calibratable vs all channels**: a channel whose training signal is constant, or whose validation scores have
  no spread, gets a threshold of ~0, which is not a calibration. Such channels are flagged
  (`calibratable: false` in `training_summary.json`, `uncalibratable_channels` in `metrics.json`), and every
  comparison is reported twice: on all channels, and on calibratable channels only.
* **Validation-to-test shift**: `per_channel.csv` has a `ks_val_test` column, the KS distance between the validation
  scores the threshold was calibrated on and the test scores, so claims about "shift" can be checked per channel.
* **Error analysis**: per-channel metrics, per-event detection and latency, false-alarm runs, event
  recall by anomaly class (point / contextual), per-spacecraft breakdown.
* All methods are compared on the **same** channel set. Channels where any method failed would be
  excluded and listed (none failed in the reported run).

## 6. Reproducibility

Seed 42 is reset before every (channel, method) fit. The LSTM also seeds itself before building the network and
again before training, so its own `seed` parameter fully determines its weights wherever it is used
(`test_lstm_seed_alone_controls_weights`). This was checked to reproduce the reported run bit-identically. Configs are saved with each run, the dataset revision
and hashes are pinned, and package lower bounds are in `requirements.txt`. PyTorch CPU results should
reproduce closely on the same machine. Bit-exactness across hardware is not guaranteed.

## 7. Calibration studies (on saved scores, no retraining)

`python -m src.cli calibration --run full` writes `reports/runs/full/calibration/`.

**Threshold-estimation uncertainty.** For each channel and method, the validation scores are resampled 100 times
with a circular block bootstrap (blocks of 50 contiguous points, because the scores are strongly autocorrelated)
and the 99.5% quantile is recomputed. Each replicate applies one resampled threshold per channel to the unchanged
test scores and recomputes the micro metrics; the random stream is keyed by method name. No label influences any
threshold. Limitation: a resampled quantile cannot exceed the sample maximum, so the ranges are biased towards lower
thresholds and indicate the size of the noise, not a calibrated confidence interval.

**Leave-channels-out (LCO) calibration.** This tests whether a *rule* chosen on some channels generalises to unseen
channels. The protocol was fixed before running:
* Rule family, computed from each channel's own validation scores: `thr = median + m · (Q_q − median)`,
  q ∈ {0.99, 0.995, 0.999, 1.0}, m ∈ {1, 1.5, 2, 3, 5}. (q = 0.995, m = 1) is the default threshold.
  The form is location-invariant, so it is also meaningful for Isolation Forest scores.
* 5 folds of channels, stratified by spacecraft, fold seed 0.
* For each fold, the rule with the best micro **event F1** on the other four folds is selected (ties → smaller m,
  then smaller q). It is then applied to the held-out fold.
* The held-out predictions of all folds are pooled and reported next to the default rule.

The grid's margins mirror the multipliers of the earlier post-hoc sensitivity analysis (EXPERIMENTS.md discloses
this). The held-out fold's labels never influence its rule (`test_lco_rule_ignores_heldout_labels`). Labels of the
other folds' channels *are* used, as in any cross-validation. The LCO numbers therefore estimate how a rule
selected this way performs on new channels. They are a separate, labelled experiment, not a replacement for the
headline results, which stay at the a-priori default. The grid was not changed after seeing results.

**Seed variance.** `configs/seed1.yaml` and `seed2.yaml` repeat the full experiment with seeds 1 and 2 (run
directories `full_seed1`, `full_seed2`). `python -m src.cli seeds --runs full full_seed1 full_seed2` summarises
mean, standard deviation and range per metric. Only Isolation Forest and the LSTM are seed-dependent; the
baseline is deterministic.

## 8. Telemanom-aligned method and the dynamic threshold

Two methods use the nonparametric dynamic threshold of Hundman et al. (2018), implemented in
`src/models/dynamic_threshold.py` from the reference code (`telemanom/errors.py`, `config.yaml`):

| Method | Forecaster | Decision rule |
|---|---|---|
| `telemanom` | new LSTM with the paper's settings: l_s = 250, 2 x 80 units, dropout 0.3, 10-step horizon (1-step forecast scored), batch 64, <= 35 epochs, patience 10, min_delta 3e-4, last epoch kept, raw [-1, 1] units, all training windows | dynamic threshold |
| `lstm_dynamic` | the **saved reference LSTM models** of `full` / `full_seed1` / `full_seed2`, unchanged | dynamic threshold |

`lstm_dynamic` is an ablation: comparing it with `lstm` changes only the decision rule; comparing it with
`telemanom` changes only the forecaster. Both use EWMA span 105 and the start-up fill of the first 250 smoothed errors.

**Why it is protocol-valid.** The dynamic threshold reads no labels; it is computed from the smoothed errors and
values of the series being judged, with the published parameters (p = 0.13, 30 x 70 windows, buffer 100, z in
2.5-11.5), fixed before any run here. Those constants were chosen by the paper's authors on this benchmark, which
may favour the method slightly; we did not tune them.

**Causality.** Each window decides only its newest 70-sample batch (the first window decides its whole 2,100 samples
at once), so a decision uses at most 69 samples after the flagged point. The evaluation reports both the first
flagged point (`median_latency`) and when the alert could be raised (`median_decision_latency`). The historical
replay in the app uses only windows completed by the replay time.

**Known differences from the reference implementation** (so this is "aligned", not an exact reproduction):
* PyTorch instead of Keras 2 / TensorFlow 1; different initialisation and numerics.
* Our chronological 80/20 train split feeds early stopping (Keras `validation_split=0.2` also takes the last 20%).
* The final, partial window decides every remaining position (the reference leaves up to 69 positions undecided).
* Channels shorter than one window are a single window (the reference shrinks the window count but not its length).
* Evaluation merges P-2's overlapping label rows and scores 81 channels (T-10 has no labels); the paper reports
  105 sequences across 82 channels. Event matching is "any overlap", as in the paper; no point-adjust.

**Operating point.** The pruning parameter p is the knob. `python -m src.cli calibration --run telemanom` selects p
by leave-channels-out (p chosen on 4 folds' labels, applied to the 5th) and writes a post-hoc p curve that is
labelled diagnostic and not used for any choice.
