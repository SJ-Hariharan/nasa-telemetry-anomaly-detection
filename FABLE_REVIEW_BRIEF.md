# Review brief: NASA telemetry anomaly detection

A self-contained brief for an independent technical reviewer. Please try to **break** the claims below,
not confirm them. File references are relative to the repository root.

## 1. Objective

Compare three unsupervised detectors (rolling z-score, Isolation Forest, LSTM one-step forecaster) on the public
NASA SMAP/MSL "Telemanom" benchmark under one leakage-controlled protocol. Report honest point- and
event-level results, and provide a dashboard and an inference path. This is historical benchmark analysis only:
it does not monitor spacecraft, and a flagged anomaly is not a confirmed fault.

## 2. Architecture

```
src/data/download.py      pinned HF mirror + GitHub labels, SHA-256 verified
src/data/loader.py        npy (col 0 = value, rest = one-hot commands), inclusive label intervals
src/data/validation.py    shape / range / label / duplicate checks
src/preprocessing/scaling.py   chronological 80/20 split of TRAIN; scaler fitted on the first 80%
src/features/windows.py   causal forecast windows, Isolation Forest features
src/models/{detectors,lstm,thresholds,persistence}.py
src/pipeline.py           per channel: fit → validation scores → threshold → score test → join labels LAST
src/evaluation/{metrics,evaluate}.py
src/evaluation/calibration.py  threshold block bootstrap, leave-channels-out rule selection, seed summary
src/inference/predict.py  validated CSV → saved bundle → scores / flags
app/dashboard.py          Streamlit, reads artifacts only
```

## 3. Data and provenance

* Hugging Face mirror `lxr11111/telemanom`, pinned commit `d93d47d7…` (`HF_REVISION`). Every file is checked against
  the mirror's LFS SHA-256. The original S3 zip returns 403.
* Labels: `labeled_anomalies.csv` from `khundman/telemanom`, pinned by SHA-256 (`LABELS_SHA256`). Byte-identical to the mirror's copy.
* 82 channel files; T-10 has no labels and is excluded; P-2's two label rows are merged by union → **81 channels, 104 events**.
* Known benchmark issue: the authors min-max scaled with **test** statistics. This cannot be undone and is documented.

## 4. Preprocessing and leakage controls

* Split: first 80% of the *train* file = fit, last 20% = validation. Test is never used for fitting or thresholds.
* The value column is z-scaled with fit-part statistics. Constant channels fall back to std = 1 (19 channels). Commands are left as 0/1.
* Windows are causal: input rows i … i+W−1 predict row i+W (`target_idx` returned explicitly).
* Test is scored as its own sequence from t = 0. The first W points are warm-up (NaN score → predicted normal).
* Labels are attached to the score table only after all predictions exist.
  `test_thresholds_do_not_depend_on_test_labels` changes the labels and asserts thresholds are unchanged.

## 5. Models and training configuration (`configs/default.yaml`)

| Model | Settings | Score |
|---|---|---|
| Rolling z-score | W = 100 past points (x_t excluded), min rolling std 0.01 | \|x_t − mean\| / std |
| Isolation Forest | 200 trees, 17 trailing features (10 lags, rolling stats over 50, diff, command activity), seed 42 | −score_samples |
| LSTM | W 100, 2×64, dropout 0.3, Adam 2e-3, MSE, batch 128, ≤ 60 epochs, patience 6 (val MSE), grad clip 1, ≤ 4k strided windows | EWMA(span 30) of \|y − ŷ\| |

Threshold: per channel, 99.5% quantile of **validation** scores. Validation is scored from `split − warmup` so
that warm-up and EWMA state match the test scoring (`validation_scores`). Seed 42 is reset per (channel, method).

## 6. Evaluation protocol

* Point metrics: confusion counts summed over channels (micro). Accuracy is reported but uninformative (12.5% positives).
* Event metrics: a labelled event is a TP if any predicted point overlaps it. Each predicted run with no overlap is one FP event.
  **No point-adjust.** Latency = first detected index − event start.
* ROC-AUC / PR-AUC per channel (warm-up excluded), macro-averaged with a bootstrap CI. The pooled normalised AUC is kept for reference only.
* Channels with a constant training signal or degenerate validation scores are flagged `calibratable: false`. Every comparison is reported on all channels and on calibratable channels.
* `ks_val_test` per channel: KS distance between validation and test scores (shift diagnostic).
* Methods are compared on the identical channel set. Post-hoc threshold multipliers are reported as a labelled diagnostic only.

## 7. Actual results (final run, 81 channels, 61.6 min CPU, 0 failures)

| Method | P | R | F1 | FPR | Event R | Event P | Event F1 | Event FP | ROC-AUC macro | PR-AUC macro |
|---|---|---|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.301 | 0.195 | 0.237 | 0.065 | 0.827 | 0.029 | 0.056 | 2,881 | 0.590 | 0.260 |
| Isolation Forest | 0.268 | 0.299 | 0.283 | 0.117 | 0.692 | 0.034 | 0.065 | 2,025 | 0.635 | 0.292 |
| LSTM | 0.269 | 0.688 | 0.387 | 0.267 | 0.933 | 0.052 | 0.099 | 1,764 | 0.677 | 0.456 |

On the 59 calibratable channels: F1 0.221 / 0.339 / 0.425, ROC-AUC macro 0.530 / 0.655 / 0.648, PR-AUC macro 0.212 / 0.329 / 0.423.
Validation flag rate is ~0.5% by construction. Median test FPR per channel is 0.7% / 1.7% / 6.3%, and the mean is 7.8% / 13.6% / 25.2%.
Full analysis: `EXPERIMENTS.md`.

## 8. Tests

`pytest -q` → **59 passed** (~10–60 s). They cover:
* hand-computed metrics;
* inclusive label alignment;
* scaler fitted on the fit part only;
* baseline causality (a future change does not alter a past score);
* forecast-window alignment;
* LSTM learning;
* calibration under test-like conditions;
* bundle save/load giving identical scores;
* inference input validation;
* an end-to-end synthetic run with reload, which reproduces the pipeline's scores;
* threshold label-independence;
* resume equivalence;
* every dashboard page rendering headless, including a simulated fresh clone without generated artifacts;
* LSTM seed self-sufficiency (D5), path-like channel/run names and oversized CSVs (D7);
* calibration status saved in bundles and surfaced at inference;
* calibration studies: default rule reproduces the run's thresholds, folds partition and stratify,
  held-out labels cannot influence a fold's rule.

## 9. Known limitations (abridged; full list in `docs/limitations.md`)

* No hyper-parameter search, simplified Telemanom (1-step forecasts, no dynamic thresholds or pruning). Three seeds (42, 1, 2).
* The 99.5% quantile is evidently too low (low precision for every method). Leave-channels-out calibration shows a
  more conservative rule generalises to unseen channels, but its LSTM choice sits at the edge of the pre-specified grid.
* The LSTM validation split is used both for early stopping and for calibration.
* Pooled AUC mixes per-channel score scales (now secondary; macro AUC is primary).

## 10. High-risk decisions: please scrutinise

1. **Label alignment.** Labels are treated as inclusive `[start, end]` indices into the *test* array (`label_vector`). Is that the dataset's convention?
2. **Scoring the test file from t = 0** with no train context. The first 100 test points can never be flagged. Would prepending the train tail be more correct, or would it be leakage-neutral but change comparability?
3. **Validation scoring offset** (`validation_scores`: score `train[split−warmup:]`, drop the first `warmup`). Does it truly make validation scores distributed like test scores, including for the Isolation Forest feature warm-up (`stat_window − 1` = 49)?
4. **LSTM:** strided cap on training windows; early stopping on the same validation part that sets the threshold; EWMA started fresh per sequence. Any target leakage in `forecast_windows` (target at `i+W`, inputs `i…i+W−1`)?
5. **Thresholds:** an eps floor when validation scores are constant (19 constant channels → any test deviation flagged). Is a different fallback more defensible?
6. **Evaluation validity:** micro aggregation, "any overlap" event TP, one FP per predicted run, no point-adjust, AUC on pooled heterogeneous scores. Are any of these misleading as reported?
7. **Inference:** `frame_to_array` fills missing command columns with 0 (with a warning) and rejects non-0/1 commands and non-finite values. Is that safe? Is the 500k-row cap adequate?
8. **Security:** `np.load(allow_pickle=False)`, `torch.load(weights_only=True)`, but **joblib (pickle) for Isolation Forest models** (scaler statistics are plain JSON in `meta.json`). Loading an untrusted `models/` directory would execute code. Is the documentation of this sufficient for a portfolio project?
9. **Reproducibility:** pinned data, lock file, per-(channel, method) seeding. Are any sources of nondeterminism still unhandled (e.g. `n_jobs=-1` in Isolation Forest, torch thread count)?

## 11. Key files for review (in priority order)

`src/pipeline.py` → `src/features/windows.py` → `src/models/lstm.py` → `src/models/thresholds.py` →
`src/evaluation/metrics.py` → `src/evaluation/evaluate.py` → `src/data/loader.py` → `src/inference/predict.py` → `tests/`.

## 12. Review outcome (first independent review, 2026-11-12)

Findings D1–D3 (and D6) were accepted and fixed in the evaluation without retraining. The others are documented in
`docs/limitations.md` and remain open.

| Finding | Status |
|---|---|
| D1 degenerate thresholds on constant-training channels (43–47% of false positives; 44–52% counting all 22 uncalibratable) | Fixed: flagged, dual reporting |
| D2 pooled normalised AUC tripled the LSTM's ranking lead and reversed baseline/IF | Fixed: macro AUC + CI is primary |
| D3 "distribution shift" asserted without evidence | Fixed: per-channel KS statistic, text rewritten |
| D4 99.5% quantile on 62–300-point validation sets | Measured (release prep): all 81 channels < 5 expected exceedances; bootstrap shows aggregate effect is small |
| D5 `LSTMDetector(seed=)` does not control weight init outside the pipeline | Fixed (release prep); `full` run still reproduces bit-identically |
| D6 warm-up rows counted as perfect negatives in AUC | Fixed |
| D7 CSV read before size check; `chan_id` path interpolation | Fixed (release prep): name validation, bounded read, exit code 2 |
| Event FP not comparable across methods (smoothing) | Open (documented) |

## 13. Release preparation (2026-11-15): new findings and changes

| ID | Finding | Status |
|---|---|---|
| N1 | Committed `reports/runs/full/run.log` contains absolute home-directory paths | Code now logs project-relative paths. The committed log (and Git history) still contains them: owner decision before the first push |
| N2 | A fresh clone crashed the Telemetry explorer, Anomaly analysis and Inference pages (scores/models are gitignored) | Fixed: pages explain how to regenerate; regression test simulates a fresh clone |
| N3 | Inference on an uncalibratable channel (e.g. M-6: 1,039/2,049 points flagged) gave no warning | Fixed: bundles record `calibratable`; CLI and dashboard warn |
| N4 | `scipy` imported directly but only installed transitively | Fixed: listed in `requirements.txt` |
| N5 | Broken link `docs/inference_and_deployment.md` in architecture doc | Fixed |
| R1 | Single seed | Done: seeds 1 and 2 complete (81/81 each). Point metrics stable, event false alarms vary ±567; seed 42 is the most favourable on event metrics (EXPERIMENTS.md, Seed variance) |
| N6 | `seeds` CLI wrote `reports/runs/seed_summary.json`, not `reports/seed_summary.json` as documented | Fixed + CLI regression test |
| N7 | Hooks | `.githooks/pre-commit` (whitespace, secrets/data/weights/size, Python syntax) and `pre-push` (full test suite); tested incl. failure cases |
| R2 | Leave-channels-out calibration | Implemented on saved scores (`src/evaluation/calibration.py`) |

**Questions for the final review**
1. Is the leave-channels-out protocol (`src/evaluation/calibration.py::leave_channels_out`) free of leakage into the
   held-out fold? Is "pooled held-out predictions" the right way to report it?
2. Is the bootstrap in `calibration_uncertainty` a fair estimate of threshold-estimation uncertainty? It resamples
   validation points independently, ignoring autocorrelation.
3. Does the double seeding in `LSTMDetector.fit` fully determine the result? What other nondeterminism remains
   (`n_jobs=-1` in Isolation Forest, torch thread count)?
4. Are the name validation and bounded CSV read sufficient for the CLI and dashboard threat model (local use)?
5. Do README, EXPERIMENTS, methodology, limitations and this brief agree with each other and with the artifacts?

## 14. Final release-candidate audit (2026-11-16): outcome

No Critical or High findings. Each finding was reproduced before acting.

| ID | Finding (severity) | Resolution |
|---|---|---|
| A1 | Default LCO rule not bit-identical to stored thresholds (Low) | Fixed: `Q + (m-1)(Q - median)`; exact-equality test; LCO results unchanged on all three runs |
| A2 | Bootstrap test name claimed containment it did not test (Low) | Fixed: renamed, plus an order-independence regression test |
| A3 | One RNG shared across methods (Low) | Fixed: stream keyed by method name; test |
| A4 | Hook: word splitting, quoted non-ASCII names, `%` in names, working-tree (not staged) checks (Low) | Fixed: NUL-separated names, staged blob sizes, staged content compiled; 8 hook cases tested |
| A5 | Dashboard inference channel list could include stray files (Low) | Fixed: filtered with `CHANNEL_ID` |
| B1/C1 | iid bootstrap on autocorrelated scores; one-sided ranges (Medium) | Fixed: circular block bootstrap (block 50); ranges re-reported as lower bounds on noise in every doc |
| B2 | Grid-edge caveat incomplete (Low-Medium) | Fixed: q = 1.0 edge and baseline m = 5 disclosed |
| B3 | "LSTM best under LCO" holds only on all channels (Low-Medium) | Fixed: calibratable-subset results stated |
| B4 | "Most favourable seed" overstated (Low) | Fixed: "event F1, precision and false alarms, not event recall" |
| B5 | Small doc inconsistencies (Low) | Fixed (joblib scope, configs list, architecture list) |
| B6/N1 | Committed `full/run.log` has home-directory paths | Owner decision: not rewritten (history preserved); new seed logs sanitised before first commit |
| C2 | LCO grid design mirrors earlier test-label sensitivity analysis (Low-Medium) | Disclosed in EXPERIMENTS, methodology and limitations |
| C3 | README installs `>=` requirements (Low) | README now points to `requirements-lock.txt` for exact versions |
| C4 | CSV bounded by rows only (Low) | Accepted for local use (documented) |
