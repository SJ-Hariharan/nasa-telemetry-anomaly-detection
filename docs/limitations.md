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
* **Three seeds only.** Seeds 42, 1 and 2 were run. Point metrics are stable (LSTM F1 ± 0.009), but event false alarms vary
  by ± 567 and event F1 by ± 0.018. The headline seed (42) is the most favourable on event F1, event precision and false alarms (not on event recall). Three seeds
  are enough to show the spread, but not to estimate it precisely.

## Engineering

* CPU-only training, sequential over channels (LSTM 3–160 s per channel, 45 s on average; 61.6 min for all 81).
* Joblib files are pickles: load only model directories produced by this project. Channel ids and run names are
  validated (`^[A-Z]-\d{1,3}$`, no `..`) before they become paths, so a CLI argument cannot point the loader elsewhere.
  A tampered file *inside* `models/` would still execute on load.
* Per-channel score tables and saved models are generated, not version-controlled (~18 MB + ~186 MB per full run).
  A fresh clone needs `download` + a ~1 h training run before the per-channel pages and inference work **for the NASA
  runs**. The offline demo (`python -m src.cli demo`) makes every page usable in seconds, but only on synthetic data
  with easy, hand-made anomalies. It demonstrates the workflow and changes none of the scientific limitations above.
* Inference warns when a saved model's threshold is not a calibration (constant training channel). It does not
  refuse to score.
* Inference assumes input in the benchmark's scaled units with the same command columns as the
  channel's training data. Missing command columns are filled with 0, with a warning.

## What the full run showed (see `EXPERIMENTS.md`)

* **22 channels cannot be calibrated.** 19 have a constant training signal, and 3 more have a constant validation segment
  for the baseline. For them the 99.5% validation quantile is ~0, so any test deviation is flagged. They produce 52% of the
  baseline's and 44% of the LSTM's point false positives (LSTM FPR 0.46 on them vs 0.20 elsewhere). The pipeline now flags them instead of
  reporting a "calibrated" threshold, and results are given with and without them. A principled fallback (pooled or
  hierarchical calibration across channels of the same spacecraft) is not implemented.
* **Small validation sets.** Every channel has fewer than 1,000 validation points (62–862), so every 99.5% quantile
  rests on fewer than 5 expected exceedances. Per-channel thresholds are therefore noisy: the block-bootstrap 90% range is
  a median 21% of the LSTM threshold (up to 97%), and up to 95× for some baseline channels. The effect on the
  aggregate results is about 0.01–0.05 in F1 (LSTM F1 range [0.375, 0.389], event F1 [0.080, 0.096];
  `reports/runs/full/calibration/`). These ranges are biased low (a resampled quantile cannot exceed the sample
  maximum) and are lower bounds on the noise. Per-channel results on short channels should not be over-read.
* **The 99.5% quantile is too low for this benchmark.** Post-hoc, a 3× higher LSTM threshold would roughly double
  event F1. That was observed with test labels and is *not* applied to the headline results.
* **Leave-channels-out calibration (done, with caveats).** Selecting the rule on other channels and applying it to
  held-out channels cuts LSTM false-alarm events from 1,764 to 332 and raises event F1 from 0.099 to 0.231. The
  price is event recall, which falls from 0.933 to 0.548. Caveats:
  1. The selected LSTM rule is at the **edge of the pre-specified grid** (m = 5 in 15/15 folds over three seeds,
     q = 1.0 in 14/15), and so is the baseline's (m = 5 in 4/5 folds per seed). A wider grid might select differently.
     It was deliberately not widened after seeing results. The grid's margins mirror the earlier post-hoc
     sensitivity multipliers, so its design was informed by that test-label analysis.
  5. The LSTM's LCO lead holds on all 81 channels but not on the 59 calibratable ones (seed 1: LSTM lowest).
  2. The selection objective (event F1) was fixed a priori. A different operational objective (for example a recall
     floor) would select a different rule.
  3. The folds share a benchmark and labelling process, so this estimates generalisation to new channels of the
     *same* missions, not to other spacecraft.
  4. It is one fold assignment (seed 0). Fold-to-fold variation is visible in `calibration.json`.
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
