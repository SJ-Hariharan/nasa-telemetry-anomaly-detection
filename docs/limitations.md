# Limitations

## Scope

* **Historical, offline analysis.** Nothing here monitors a spacecraft or runs inside NASA systems. The
  dashboard's "simulated signal" is synthetic and labelled as such.
* **An anomaly is a statistically unusual observation**, not a confirmed fault. Labels mark what experts
  considered anomalous in incident reports. A flagged point may be a benign mode change, and a real
  problem may be statistically ordinary.

## Data (benchmark-level, cannot be fixed downstream)

1. **Test-informed scaling.** The authors min-max scaled each channel using the *test* set's min/max. Any
   method sees values whose range was set by test data. This favours every method equally but means the
   numbers are not a clean simulation of deployment.
2. **Label quality.** The labels come from incident reports: start/end boundaries are approximate,
   and some anomalous behaviour may be unlabelled. P-2's duplicate rows disagree on the boundaries.
   T-10 has no labels at all.
3. **"Nominal" training data is assumed, not guaranteed.** If train/validation contain anomalies, models
   learn them as normal and thresholds rise.
4. **Short histories, long tests.** Median train length is ~2.4k points, test often 3–8k. Regimes absent from
   training will look anomalous (19 channels are completely constant in training).
5. **Anonymised channels.** No physical meaning or cross-channel relationships can be used.
6. The benchmark is known in the literature to contain "trivial" anomalies that simple methods catch and
   ambiguous ones nobody catches. Benchmark scores do not transfer directly to other systems.

## Method

* **Per-channel univariate scoring.** Cross-channel correlations (a valuable signal in real operations)
  are ignored. Commands are used only by the LSTM (inputs) and Isolation Forest (activity count).
* **Thresholds:** one fixed quantile (99.5%) for every channel and method. With very constant validation
  scores the threshold collapses towards zero, so any change is flagged. The LSTM validation split
  serves both early stopping and calibration (mild optimism).
* **LSTM simplifications vs Telemanom:** window 100 (not 250), hidden 64 (not 80), 1-step (not 10-step)
  forecasts, at most 4,000 training windows per channel, validation-quantile thresholds instead of
  nonparametric dynamic thresholding with pruning. Our numbers are **not** a replication of the paper.
* **Score smoothing differs:** only the LSTM's errors are EWMA-smoothed (as in Telemanom). The baseline and
  Isolation Forest scores are raw, which partly explains their more fragmented predictions.
* **Event metric leniency:** a single predicted point inside a long labelled sequence counts as detecting it.
  Latency and point-wise metrics are reported alongside to expose this.
* **No hyper-parameter search.** Defaults were chosen from the literature and one convergence check on a
  single channel (P-1). A proper study would tune on validation data or with nested cross-validation over channels.
* **Single seed.** Variance across seeds was not measured for the full run.

## Engineering

* CPU-only training, sequential over channels (LSTM 3–160 s per channel, 45 s on average; 61.6 min for all 81).
* Joblib files are pickles: load only model directories produced by this project.
* Inference assumes input in the benchmark's scaled units with the same command columns as the
  channel's training data. Missing command columns are filled with 0, with a warning.

## What the full run showed (see `EXPERIMENTS.md`)

* **22 channels cannot be calibrated.** 19 have a constant training signal, and 3 more have a constant validation segment
  for the baseline. For them the 99.5% validation quantile is ~0, so any test deviation is flagged. They produce 52% of the
  baseline's and 44% of the LSTM's point false positives (LSTM FPR 0.46 on them vs 0.20 elsewhere). The pipeline now flags them instead of
  reporting a "calibrated" threshold, and results are given with and without them. A principled fallback (pooled or
  hierarchical calibration across channels of the same spacecraft) is not implemented.
* **Small validation sets.** 13 channels have fewer than 300 validation points (minimum 62). The 99.5% quantile of
  62 points is essentially the validation maximum, so thresholds on these channels are noisy.
* **The 99.5% quantile is too low for this benchmark.** Post-hoc, a 3× higher LSTM threshold would roughly double
  event F1. That was observed with test labels and is *not* applied. A legitimate fix would choose the calibration
  rule on a held-out set of *channels* (leave-channels-out), or use dynamic/pruned thresholds as in Telemanom.
* **Shift is a partial explanation.** The per-channel KS distance between validation and test scores correlates
  moderately with FPR for the LSTM (Spearman 0.51) and IF (0.38), and not at all for the baseline. Raw-signal drift
  statistics correlate weakly (|ρ| ≤ 0.27). Earlier wording attributed the false alarms wholesale to
  "distribution shift". That was not supported and has been replaced.
* **AUC caveats.** Only the macro (per-channel mean) AUC is a fair ranking figure; the pooled normalised-score AUC is
  kept for reference. The LSTM's macro ROC-AUC lead over Isolation Forest is within the bootstrap intervals, and on
  calibratable channels IF is marginally ahead. The LSTM's lead on PR-AUC and recall is clear.
* **Event FP counts are not comparable across methods.** Only the LSTM's scores are smoothed, and `merge_gap = 0`.
* **Threshold-multiplier sensitivity is meaningless for Isolation Forest** (bounded score scale), so it is excluded from that analysis.
* LSTM misses are mostly long contextual anomalies whose values stay in a familiar range (6 of its 7 misses).
