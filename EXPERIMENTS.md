# Experiments

Every number here was produced by the code in this repository on the real benchmark data. Nothing is
estimated or copied from papers. Raw outputs: `reports/runs/<run>/`.

> **Revision note.** After an independent review (`FABLE_REVIEW_BRIEF.md` §12), the evaluation was corrected
> without retraining: (D1) channels whose threshold cannot be calibrated are now flagged and the comparison is
> reported with and without them; (D2) the primary threshold-free metric is the **macro** (per-channel mean)
> AUC with a bootstrap interval, not the pooled normalised-score AUC; (D3) the "distribution shift" claim was
> replaced by measured evidence (a per-channel KS statistic). Warm-up positions are excluded from AUCs.
> Scores, thresholds and predictions are unchanged; the first version of this file reported pooled ROC-AUC
> 0.753 / 0.603 / 0.590, which overstated the LSTM's ranking advantage.

## Run log

| Run | Config | Channels | Purpose | Status |
|---|---|---|---|---|
| `smoke` | `configs/smoke.yaml` | 3 (P-1, T-1, M-6) | End-to-end check, tiny LSTM (2 epochs) | Done, ~30 s, run twice: bit-identical |
| P-1 convergence check | 30 epochs / batch 64 vs ≤ 60 epochs / batch 128 / lr 2e-3 | 1 | Was the LSTM undertrained? | First setting still improving at its last epoch (val MSE 0.155). Second converged (0.142) and became the default |
| `full` (attempt 1) | default | 81 | First full run | **Discarded.** Stopped after the internal review found the LSTM validation-scoring mismatch (LEARNING_LOG §11) |
| `full` (attempt 2) | default | 56/81 | Corrected code | Interrupted when the agent session ended (no error). Outputs deleted. Resume support added |
| **`full`** (final) | `configs/default.yaml`, seed 42 | **81** | Reported results | **Done.** Training 3,698 s (61.6 min, CPU), 0 failures. Re-evaluated after the review (evaluation only, ~20 s) |
| T-9 re-run (review) | default | 1 | Determinism check | Thresholds identical to the last digit, LSTM weights bit-identical to the saved model |
| T-9, C-2 re-run (release prep) | default | 2 | Check that the LSTM seeding fix (D5) does not change results | Thresholds and LSTM weights bit-identical to the saved `full` run |
| `full` calibration studies | `python -m src.cli calibration --run full` | 81 | Threshold uncertainty (D4) + leave-channels-out calibration | Done, ~15 s, saved scores only |
| **`full_seed1`, `full_seed2`** | `configs/seed1.yaml`, `seed2.yaml` | 81 each | Seed variance | **Done (both).** 81/81 channels each, 0 failures. Recorded compute: seed 1 73.9 min; seed 2 106.4 min (machine shared with other work). Seed 1's automatic evaluation step crashed with an `ImportError` (code edited while the process ran, LEARNING_LOG §14) and was re-run alone. Seed 2 was killed by a machine reboot after 72/81 channels and resumed with `--resume`. Both events are logged in each `run.log` |

The headline results come from one configuration with seed 42. Two further seeds measure seed variance (see below).
Hyper-parameters were not searched. Thresholds were
fixed from validation data before any test metric was computed.

## Setup (final run)

* 81 channels: 54 SMAP + 27 MSL. T-10 is excluded (no labels), P-2's two label rows are merged. 104 labelled events (61 "point", 43 "contextual"). 12.5% of test points are labelled anomalous.
* Per channel: chronological 80/20 split of the train file, a scaler fitted on the first 80%, one model of each type.
  The threshold is the 99.5% quantile of that channel's validation scores (validation length 62–862 points, median 538).
* Rolling z-score: W = 100 past points. Isolation Forest: 200 trees on 17 trailing features (10 lags, rolling stats over 50).
  LSTM: W = 100, 2×64, dropout 0.3, Adam lr 2e-3, batch 128, ≤ 60 epochs, patience 6, ≤ 4,000 strided training
  windows, value + command inputs, score = EWMA(span 30) of |forecast error|. Median best epoch 14; 2 of 81 channels were still improving at the 60-epoch cap.
* **Uncalibratable channels (22).** 19 channels have a constant training signal (`constant_train`), so no nominal
  variability can be learned; three more (G-3, D-16, M-7) have a constant validation segment for the baseline. Their
  "99.5% quantile" threshold is ~0 (the float32 epsilon for the baseline, 1e-4–2e-3 for the LSTM) and any test deviation
  is flagged. They are kept in the "all channels" table and excluded from the "calibratable" table.

## Results

### All 81 channels (micro-aggregated point and event metrics; macro per-channel AUCs)

| Method | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Events TP / FN / FP | ROC-AUC macro [95% CI] | PR-AUC macro | Median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 86 / 18 / 2,881 | 0.590 [0.55, 0.63] | 0.260 | 37 steps |
| Isolation Forest | 0.268 | 0.299 | 0.283 | 0.117 | 0.034 | 0.692 | 0.065 | 72 / 32 / 2,025 | 0.635 [0.59, 0.68] | 0.292 | 27 steps |
| **LSTM forecaster** | 0.269 | **0.688** | **0.387** | 0.267 | **0.052** | **0.933** | **0.099** | 97 / 7 / 1,764 | **0.677** [0.62, 0.73] | **0.456** | **16 steps** |

### The 59 calibratable channels only

| Method | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Events TP / FN / FP | ROC-AUC macro [95% CI] | PR-AUC macro |
|---|---|---|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.365 | 0.159 | 0.221 | 0.042 | 0.033 | 0.769 | 0.063 | 60 / 18 / 1,778 | 0.530 [0.49, 0.57] | 0.212 |
| Isolation Forest | 0.322 | 0.358 | 0.339 | 0.114 | 0.038 | 0.782 | 0.073 | 61 / 17 / 1,534 | **0.655** [0.60, 0.70] | 0.329 |
| **LSTM forecaster** | 0.321 | **0.629** | **0.425** | 0.202 | **0.040** | **0.910** | **0.077** | 71 / 7 / 1,701 | 0.648 [0.58, 0.71] | **0.423** |

Pooled normalised-score AUCs (score ÷ channel threshold, then concatenated) are still written to
`comparison.csv` as `roc_auc_pooled` / `pr_auc_pooled` for reference (LSTM 0.749 / 0.317 on all channels). They are
**not** a valid cross-method ranking metric: a channel with a ~0 threshold outranks every other channel's points, so
pooling reverses the baseline/IF order (0.597 vs 0.586 pooled, 0.590 vs 0.635 macro) and triples the LSTM's margin.

### By spacecraft (all channels)

| Method | SMAP F1 | SMAP ROC macro | SMAP event FP | MSL F1 | MSL ROC macro | MSL event FP |
|---|---|---|---|---|---|---|
| Rolling z-score | 0.253 | 0.585 | 1,883 | 0.156 | 0.599 | 998 |
| Isolation Forest | 0.312 | 0.626 | 1,659 | 0.166 | 0.653 | 366 |
| LSTM | 0.407 | 0.671 | 1,605 | 0.242 | 0.689 | 159 |

### Event recall by anomaly class (all channels)

| Class | Rolling z-score | Isolation Forest | LSTM |
|---|---|---|---|
| point (61) | 0.918 | 0.705 | 0.984 |
| contextual (43) | 0.698 | 0.674 | 0.860 |

## Interpretation

1. **The LSTM has the best recall, PR-AUC, event detection and latency.** It catches 97/104 events (71/78 on
   calibratable channels), earliest (median 16 steps), and its macro PR-AUC (0.456 / 0.423) is well above the others.
   Its macro ROC-AUC lead is modest: 0.677 vs 0.635 (IF) with overlapping 95% intervals on all channels, and on the
   calibratable channels IF is marginally ahead (0.655 vs 0.648). **The ranking advantage is real on PR-AUC and
   recall, and not established on ROC-AUC.** It misses 7 events, 6 of them contextual (e.g. M-1's 1,141-step event).
2. **Precision is the main weakness, for every method, and a quarter of it is a calibration failure.**
   The 22 uncalibratable channels (27% of channels) produce 52% of the baseline's, 27% of Isolation Forest's and 44% of
   the LSTM's point false positives; the LSTM's FPR is 0.457 on them vs 0.202 elsewhere (A-1, B-1, R-1, D-13 at 0.99 with a test signal that takes
   only 2–5 distinct values). Excluding them raises LSTM point F1 from 0.387 to 0.425 and precision from 0.27 to 0.32.
   That is still far too many alarms for operational use.
3. **Validation-to-test score shift is measurable and explains part of the rest.** `per_channel.csv` carries
   `ks_val_test`, the Kolmogorov–Smirnov distance between each channel's validation scores (what the threshold was
   calibrated on) and its test scores. Median KS: 0.16 (baseline), 0.32 (IF), 0.46 (LSTM); 37 of 81 channels have
   LSTM KS > 0.5. Spearman correlation between KS and per-channel FPR: 0.51 (LSTM), 0.38 (IF), −0.04 (baseline).
   So for the learned models, channels whose test scores are distributed differently from their validation scores
   do raise more false alarms, but the relation is moderate, and simple drift statistics of the raw signal
   (test/train std ratio, mean shift) correlate weakly with FPR (|ρ| ≤ 0.27). "Distribution shift" is therefore a
   partial explanation, not the whole one; thresholds estimated from 62–300 validation points (13 channels) are
   another, and the remaining false alarms are concentrated: the 10 worst channels hold 60–78% of each method's FP events.
4. **Event recall is lenient; event precision is not.** A labelled event counts as detected if any predicted point
   overlaps it, so a detector that fires often gets high event recall cheaply (baseline: 0.827 recall, 0.029 precision).
   Event FP counts are also not comparable across methods: only the LSTM's scores are EWMA-smoothed, so 55% / 48% of
   baseline / IF false-alarm runs are single timesteps vs 15% for the LSTM. Read event recall together with event FP and latency.
5. **Isolation Forest vs the baseline:** IF has higher point F1 and macro AUC on every subset; the baseline only wins
   on "any overlap" event recall, which rewards its fragmented alarms.
6. **Point F1 and event F1 rank the methods the same way (LSTM > IF > baseline) but differ by 4×**, which is why both are reported.

### Post-hoc threshold sensitivity (diagnostic only, NOT used for any choice)

Each calibrated threshold was multiplied by k ∈ {0.5 … 5} and test metrics recomputed (`sensitivity.csv`).
For the LSTM, raising every threshold 3× would raise event F1 from 0.099 to 0.225 and point F1 from 0.387
to 0.430, at the cost of event recall (0.933 → 0.606). This shows that **the 99.5% quantile is too low for
this benchmark's test distribution.** Choosing k from this table would mean tuning on test labels, so the
reported results stay at k = 1. A legitimate fix would choose the calibration rule on held-out *channels* (see limitations).
Isolation Forest is excluded from this analysis: its scores lie in a narrow band around 0.5, so ×0.5 flags
every point (one giant "event" that overlaps every label: a meaningless event F1 of 0.90) and ×1.5 flags none.

### Error analysis

* **False alarms:** `false_alarms.csv` lists every predicted run with no overlap. LSTM false-alarm runs cluster on
  channels whose test regimes are absent from training (F-1, E-9, T-1, E-8, E-2) and on the uncalibratable channels.
* **Misses:** `events.csv` gives every labelled event with detected / latency / peak score per method. The LSTM's 7 misses are
  G-1, D-11, F-3, M-1, M-4 and T-13 (×2): mostly long contextual events where the signal stays in a familiar range.
* **Timelines:** `reports/runs/full/figures/timeline_*.png` and the dashboard's Channel explorer and Events & errors pages show
  scores, thresholds, predictions and labels together for any channel.

## Calibration studies (saved scores, no retraining)

Protocols: `docs/methodology.md` §7. Outputs: `reports/runs/full/calibration/` (`calibration.md`, `calibration.json`,
`threshold_bootstrap.csv`). Reproduce with `python -m src.cli calibration --run full` (~15 s).

### Threshold-estimation uncertainty (review finding D4)

Every channel has fewer than 5 expected validation exceedances at the 99.5% quantile (62–862 validation points).
Validation scores are strongly autocorrelated (median lag-1 autocorrelation 0.80 baseline, 0.94 Isolation Forest,
0.98 LSTM), so the bootstrap resamples **contiguous blocks of 50 points** (circular block bootstrap). An earlier
version resampled points independently, which understated the spread (LSTM F1 [0.385, 0.390]); the final audit
caught this. Per-channel thresholds are noisy: the 90% range spans a median 21% of the LSTM threshold (maximum 97%),
5% for Isolation Forest and 25% for the baseline (up to 95× on channels whose validation scores are almost constant).
Propagated to the aggregate metrics (100 replicates, point estimate [5th, 95th percentile]):

| Method | Precision | Recall | F1 | Event F1 |
|---|---|---|---|---|
| Rolling z-score | 0.301 [0.268, 0.303] | 0.195 [0.197, 0.212] | 0.237 [0.229, 0.247] | 0.056 [0.043, 0.060] |
| Isolation Forest | 0.268 [0.236, 0.287] | 0.299 [0.297, 0.363] | 0.283 [0.266, 0.318] | 0.065 [0.046, 0.068] |
| LSTM | 0.269 [0.257, 0.268] | 0.688 [0.690, 0.727] | 0.387 [0.375, 0.389] | 0.099 [0.080, 0.096] |

**How to read the ranges.** A resampled 99.5% quantile can never exceed the sample's maximum, so bootstrap thresholds
are biased *low*. Precision ranges sit at or below the point estimate and recall ranges at or above it, and several
ranges exclude the point estimate. They show the **size** of threshold-estimation noise; they are not calibrated
confidence intervals, and should be read as lower bounds on that noise.

**Finding:** threshold-estimation noise moves the aggregate metrics by about 0.01–0.05 in F1 and 0.01–0.02 in event F1.
That is far less than the gap between precision and recall, and far less than the effect of the calibration rule
(next section). It does not explain the low precision.

### Leave-channels-out calibration

Rule chosen on four folds of channels by micro event F1 and applied to the fifth. Pooled held-out results:

| Method | Channels | Rule | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Event FP |
|---|---|---|---|---|---|---|---|---|---|---|
| Rolling z-score | all (81) | default (q 0.995, m 1) | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 2,881 |
| Rolling z-score | all (81) | leave-channels-out | 0.325 | 0.177 | 0.229 | 0.053 | 0.037 | 0.596 | 0.070 | 1,613 |
| Rolling z-score | calibratable (59) | default (q 0.995, m 1) | 0.365 | 0.159 | 0.221 | 0.042 | 0.033 | 0.769 | 0.063 | 1,778 |
| Rolling z-score | calibratable (59) | leave-channels-out | 0.379 | 0.137 | 0.201 | 0.034 | 0.071 | 0.462 | 0.124 | 468 |
| Isolation Forest | all (81) | default (q 0.995, m 1) | 0.268 | 0.299 | 0.283 | 0.117 | 0.034 | 0.692 | 0.065 | 2,025 |
| Isolation Forest | all (81) | leave-channels-out | 0.271 | 0.183 | 0.219 | 0.070 | 0.072 | 0.385 | 0.121 | 518 |
| Isolation Forest | calibratable (59) | default (q 0.995, m 1) | 0.322 | 0.358 | 0.339 | 0.114 | 0.038 | 0.782 | 0.073 | 1,534 |
| Isolation Forest | calibratable (59) | leave-channels-out | 0.335 | 0.214 | 0.261 | 0.064 | 0.074 | 0.410 | 0.125 | 400 |
| LSTM | all (81) | default (q 0.995, m 1) | 0.269 | 0.688 | 0.387 | 0.267 | 0.052 | 0.933 | 0.099 | 1,764 |
| LSTM | all (81) | leave-channels-out | 0.343 | 0.500 | 0.407 | 0.137 | 0.147 | 0.548 | 0.231 | 332 |
| LSTM | calibratable (59) | default (q 0.995, m 1) | 0.321 | 0.629 | 0.425 | 0.202 | 0.040 | 0.910 | 0.077 | 1,701 |
| LSTM | calibratable (59) | leave-channels-out | 0.514 | 0.402 | 0.451 | 0.058 | 0.093 | 0.397 | 0.150 | 303 |

Rules (q, m) chosen per fold: baseline: (0.99, 1.5), (0.995, 5.0), (0.995, 5.0), (0.995, 5.0), (0.99, 5.0); IF: (0.99, 1.5), (0.99, 1.5), (0.99, 1.5), (0.99, 1.5), (1.0, 1.5); LSTM: (1.0, 5.0), (1.0, 5.0), (1.0, 5.0), (1.0, 5.0), (0.999, 5.0).

**Findings**
1. A rule chosen on *other* channels generalises. For the LSTM, held-out false-alarm events fall from 1,764 to
   332, event precision rises from 0.052 to 0.147 and event F1 from 0.099 to 0.231. The price is event recall
   (0.933 → 0.548). Point F1 changes little (0.387 → 0.407) because recall drops as precision rises.
2. On all 81 channels the LSTM keeps the highest event F1 under LCO (0.231 vs 0.121 for IF and 0.070 for the baseline).
   **This does not hold on the 59 calibratable channels:** there the LCO event F1 is 0.150 / 0.125 / 0.124 (LSTM / IF /
   baseline) for seed 42, but 0.122 / 0.154 / 0.124 for seed 1, where the LSTM is lowest. On calibratable channels
   the LSTM's LCO gain is also smaller (event F1 0.077 → 0.150, event recall 0.397).
3. Every fold chose a much more conservative rule than the default, consistent with the post-hoc sensitivity
   analysis. Unlike that analysis, this estimate never uses the evaluated channel's labels.
4. **Caveat: choices at the grid edge.** For the LSTM, m = 5 (the largest margin) is chosen in all 15 folds over the three
   seeds, and q = 1.0 (the validation maximum, the largest quantile) in 14 of 15. The baseline chooses m = 5 in 4 of 5
   folds on every seed. The optimum may lie beyond the grid. It was deliberately not widened after seeing results.
   **Disclosure:** the grid was fixed before running this experiment, but its margins (1–5) mirror the multipliers of
   the earlier post-hoc sensitivity analysis, which had been run with test labels on the same channels. Selection
   *within* the grid is leak-free; the *design* of the grid was informed by that earlier look.
5. Under LCO the "calibratable" subset still uses rules chosen on folds that include uncalibratable channels.

These numbers are a cross-validated *estimate* for a different operating policy. The headline results above remain
at the a-priori default threshold.

## Seed variance

Three complete runs with identical configuration except the seed: `full` (42), `full_seed1` (1), `full_seed2` (2).
Summary: `reports/seed_summary.json` (`python -m src.cli seeds --runs full full_seed1 full_seed2`). Values are
mean ± standard deviation [min, max] over the three seeds, all 81 channels:

| Method | Point F1 | Recall | Precision | Event recall | Event F1 | False-alarm events | ROC-AUC macro | PR-AUC macro |
|---|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.237 (identical) | 0.195 | 0.301 | 0.827 | 0.056 | 2,881 | 0.590 | 0.260 |
| Isolation Forest | 0.302 ± 0.021 [0.283, 0.324] | 0.325 ± 0.025 | 0.282 ± 0.018 | 0.692 ± 0.010 | 0.062 ± 0.003 | 2,133 ± 94 | 0.630 ± 0.005 | 0.289 ± 0.003 |
| LSTM | 0.390 ± 0.009 [0.382, 0.400] | 0.699 ± 0.009 | 0.270 ± 0.009 | 0.936 ± 0.006 (97, 97, 98 of 104) | 0.080 ± 0.018 [0.063, 0.099] | 2,314 ± 567 [1,764, 2,897] | 0.682 ± 0.007 | 0.456 ± 0.000 |

On the 59 calibratable channels: LSTM F1 0.424 ± 0.004, IF 0.363 ± 0.026, baseline 0.221; macro ROC-AUC LSTM
0.655 ± 0.007, IF 0.650 ± 0.006, baseline 0.530.

**Findings**
1. **The baseline is identical across seeds**, as it must be (it has no randomness). This is a sanity check on the seeding.
2. **Point-level LSTM results are stable.** F1 varies by ±0.009, recall by ±0.009, and macro ROC-AUC by ±0.007. Its point-F1 and
   PR-AUC lead over Isolation Forest holds in every seed. On all channels its macro ROC-AUC is also higher in every seed
   (0.677–0.690 vs 0.626–0.635). On calibratable channels the two overlap (0.648–0.660 vs 0.644–0.655).
3. **Event-level results are not stable.** LSTM false-alarm events range from 1,764 to 2,897 and event F1 from 0.063 to
   0.099. **The seed-42 run used for the headline tables is the most favourable of the three on event F1, event
   precision and false-alarm count** (not on event recall: seed 2 detects 98/104).
   Quote the event F1 as 0.080 ± 0.018 rather than 0.099 when seed variance matters.
4. Macro PR-AUC is 0.4560 / 0.4556 / 0.4556. Per-channel PR-AUC changes in 79 of 81 channels between seeds
   (e.g. P-1: 0.076 vs 0.166), but the changes average out. This was checked because it looked suspicious.
5. **The leave-channels-out result holds on every seed.** LSTM held-out false-alarm events fall 1,764 → 332, 2,897 → 375 and
   2,281 → 434. Event F1 rises 0.099 → 0.231, 0.063 → 0.203 and 0.079 → 0.195. Event recall falls to 0.548 / 0.519 / 0.558.
   The selected LSTM rule is m = 5 (the grid edge) in all 15 folds. For Isolation Forest the event-F1-selected rule
   costs a lot of point F1 (0.283 → 0.219, 0.298 → 0.158, 0.324 → 0.157). Calibration outputs:
   `reports/runs/<run>/calibration/`.
6. Seed variance in event false alarms (±567) is larger than the threshold-estimation uncertainty measured by the
   bootstrap. Both are smaller than the effect of the calibration rule itself.

## Comparison with the literature (qualitative only)

Hundman et al. (2018) report much higher event-level precision on this benchmark. Their pipeline differs:
nonparametric dynamic thresholds, anomaly pruning, 250-step windows, 10-step forecasts, larger models, and a
different event-scoring protocol. **These results are not a replication and should not be compared number-for-number.**
The ranking (LSTM forecaster > simple baselines on recall) is consistent with the paper's message.

## Reproduce

```bash
python -m src.cli all --config configs/default.yaml           # ~62 min CPU; add --resume after an interruption
python -m src.cli evaluate --config configs/default.yaml      # re-evaluate from saved scores (~20 s)
python -m src.cli calibration --run full                      # calibration studies (~15 s)
python -m src.cli all --config configs/seed1.yaml --resume    # seed-variance runs (~1-2 h each)
python -m src.cli all --config configs/seed2.yaml --resume
python -m src.cli seeds --runs full full_seed1 full_seed2
```
