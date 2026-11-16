# Learning log

One entry per milestone, in the order the project was built. Each lists what was introduced, where it lives,
which decisions were made (and why), what went wrong, and what you should be able to explain without notes.
Deeper explanations: `docs/learning/`.

---

## 1. Environment and repository

**Concepts:** virtual environments, dependency pinning (lower bounds), reproducible project layout.
**Files:** `requirements.txt`, `.gitignore`, `configs/`, `src/utils/config.py`, `src/utils/reproducibility.py`.
**Decisions**
* One `.venv` with Python 3.11. torch, scikit-learn and streamlit all ship stable wheels for it.
* YAML configs with `inherit:` so `smoke.yaml` overrides only a few keys of `default.yaml`.
* Paths in configs are relative to the project root (`resolve_path`): no machine-specific paths.
**What happened:** the sandboxed shell could not write `.git` hooks, and pip's SSL check failed through
the network proxy. Both were environment issues, not code issues, and were run outside the sandbox.
**Explain independently:** why seeding Python, NumPy *and* PyTorch is necessary, and why it still doesn't
guarantee bit-identical results across machines.
**Exercise:** add a new key to `configs/smoke.yaml` and print `load_config("configs/smoke.yaml")` to see the merge.

## 2. Data acquisition and provenance

**Concepts:** data provenance, checksums (SHA-256), idempotent downloads, pinned revisions, mirrors.
**Files:** `src/data/download.py`, `data/raw/telemanom/MANIFEST.json`, `docs/dataset.md`.
**Functions:** `build_manifest`, `download_dataset`, `_fetch` (retries), `sha256_of`.
**Decisions**
* The original S3 zip returns 403, and Kaggle needs credentials. We used a Hugging Face mirror **pinned to
  a commit hash**, so the files cannot change under us, and verified every file with the mirror's LFS SHA-256.
* Labels come from the canonical GitHub repo and are cross-checked byte-for-byte against the mirror's copy (identical).
* `np.load(..., allow_pickle=False)`: downloaded files must never be able to execute code.
**What went wrong:** downloads truncated at ~1.5 MB through the sandbox proxy (`IncompleteRead`). Retries
alone did not help, but the checksum check *refused to save the corrupt files*. That is exactly why verification exists.
**Explain independently:** what a hash proves (integrity vs a reference) and what it doesn't (that the reference itself is the original).

## 3. Validation

**Concepts:** schema/shape checks, label-range checks, duplicates, distribution sanity checks.
**Files:** `src/data/validation.py`, `src/data/loader.py` (`read_raw_labels`, `load_labels`, `merge_intervals`, `label_vector`).
**Findings (real, from the data):** P-2 has two label rows, T-10 has no labels, 19 channels are constant in
training, and 11 MSL channels exceed [-1, 1] (M-6 reaches 258).
**Decisions:** merge P-2 by interval union (documented), exclude T-10, keep out-of-range values.
We don't "fix" data we don't understand; we document it.
**Common mistake:** treating `[start, end]` as half-open. Here both ends are inclusive (`label_vector` test).
**Explain independently:** why validation errors stop the pipeline but warnings don't.

## 4. EDA

**File:** `notebooks/exploratory_analysis.ipynb` (executed; figures in `reports/figures/`).
**Concepts:** class imbalance (12.5% anomalous points), heterogeneous channels, autocorrelation, distribution shift.
**Decisions it justified:** per-channel models/thresholds; event metrics; window W = 100; std fallback for constant channels.
**Exercise:** pick a channel where test std ≫ train std and predict which detector will raise most false alarms there. Check in the dashboard.

## 5. Preprocessing and windows (leakage control)

**Files:** `src/preprocessing/scaling.py`, `src/features/windows.py`.
**Functions:** `chronological_split_index`, `ValueScaler.fit/transform`, `forecast_windows`, `isolation_features`.
**Decisions**
* Validation = the *last* 20% of train (chronological), never shuffled timesteps.
* The scaler is fitted on the fit part only. Command columns are left as 0/1.
* `forecast_windows` returns `target_idx`, so each error is written back to the exact timestep it predicts.
**Common mistakes:** fitting the scaler on train+test; shuffling timesteps before splitting; off-by-one target alignment.
**Explain independently:** why shuffling *windows* during training is fine but shuffling before the split is not.

## 6. Detectors

**Files:** `src/models/detectors.py`, `src/models/lstm.py`.
**Concepts:** rolling z-score; isolation principle; sequence forecasting; LSTM gates and cell state; early stopping.
**Decisions**
* A common `fit(train_scaled, split_idx)` / `score(array)` interface, so the rest of the code is model-agnostic.
* The baseline excludes x_t from its own reference window (otherwise a spike inflates its own std).
* The LSTM forecasts one step ahead and the score is the EWMA of |error|.
* LSTM training: the first try (30 epochs, batch 64) was still improving at the last epoch, i.e. undertrained.
  We switched to batch 128, lr 2e-3, ≤ 60 epochs + early stopping, which gave a lower val MSE on P-1 (0.142 vs 0.155).
**Explain independently:** what the cell state does that the hidden state alone cannot.

## 7. Thresholds and post-processing

**File:** `src/models/thresholds.py` (`calibrate_threshold`, `apply_threshold`, `runs`).
**Decision:** threshold = 99.5% quantile of validation scores, per channel, fixed before seeing test results.
**Common mistake:** choosing the threshold that maximises test F1. That reports a number you could never achieve on new data.
**Exercise:** change `quantile` to 0.99 in `configs/smoke.yaml`, rerun, and explain the precision/recall change.

## 8. Evaluation and error analysis

**Files:** `src/evaluation/metrics.py`, `src/evaluation/evaluate.py`.
**Concepts:** confusion matrix, precision/recall/F1/FPR/FNR, event-level metrics, micro-averaging, ROC/PR-AUC, latency.
**Decisions:** report both point and event metrics; no point-adjust; compare methods on identical channel sets;
the sensitivity analysis is labelled post-hoc.
**Explain independently:** why point F1 and event F1 can rank methods differently.

## 9. Persistence, inference, dashboard

**Files:** `src/models/persistence.py`, `src/inference/predict.py`, `app/dashboard.py`, `src/cli.py`.
**Decisions:** a bundle = scaler + threshold + params + weights. `torch.load(weights_only=True)`.
Joblib only for files this project wrote. Inference validates input and never refits. The dashboard only reads artifacts.
**Verified:** a round-trip test shows loaded models reproduce training-time scores exactly. Headless `AppTest` renders every page.

## 10. Testing

**Files:** `tests/`. Unit tests (alignment, causality, leakage, metrics by hand), an integration test (synthetic
data in the benchmark layout, end-to-end + reload), a label-independence test, and dashboard render tests.
**Explain independently:** why a synthetic dataset is right for *tests* but must never appear in *results*.

## 11. Internal code review (before the full run)

Two reviewers (the agents that wrote `docs/learning/`) read the code independently. Verified findings and fixes:

| Finding | Severity | Fix |
|---|---|---|
| LSTM validation scores were EWMA-smoothed continuously from the fit part, unlike test scores | Calibration bias (thresholds slightly low) | `validation_scores()` scores from `split - warmup`, the same way the test file is scored; new test |
| Label file came from GitHub `master` and was not re-verified on re-runs | Reproducibility | Pinned by SHA-256 (`LABELS_SHA256`), re-hashed every run |
| LSTM shuffle generator hard-coded to seed 0 | Config seed partly ignored | Uses `cfg["seed"]` |
| No guard for non-finite validation loss | Crash with unclear error | Explicit error |
| Overlap check ran on already-merged labels (could never fire) | Dead check | Runs on raw label rows |
| Only lower-bound versions | Reproducibility | `requirements-lock.txt` (`pip freeze`) |
| Dashboard cache could show stale runs | UX | "Reload run data" button |

The partially finished full run was **discarded and restarted** after the fixes, so every reported number
comes from the corrected code. **Lesson:** calibration data must be scored by exactly the same procedure as the
data the threshold is applied to. Leakage-style bugs often hide in *how* something is computed, not *what*.

## 12. The full run: interruption, resume, and reading the results

**Files:** `src/pipeline.py` (`run_training(..., resume=True)`), `src/cli.py` (`--resume`), `EXPERIMENTS.md`.
**What happened:** the corrected full run was killed at channel 56/81 when the agent session ended. There were no errors,
but it lost ~45 minutes of compute. Fix: each channel now writes `scores/<chan>.json` *last*, so its presence
means "this channel finished completely". `--resume` skips such channels and refuses to resume if the config
changed. A test checks that a resumed run gives the same thresholds as an uninterrupted one. The final run took 61.6 min.
**Reading the results:**
* Validation flag rate 0.5% → test FPR median 0.7–6.3%, mean up to 25%. A threshold is only as good as
  the assumption that test looks like validation.
* The "any overlap" event recall rewards trigger-happy detectors. Always read it next to event FP.
* A post-hoc sensitivity table can show that a threshold is badly placed, but using it to pick the threshold
  would be tuning on test labels.
* The sensitivity analysis itself had a flaw: multiplying a threshold is meaningless for bounded scores
  (Isolation Forest gave a fake event F1 of 0.90 at x0.5). It was caught on inspection and IF is now excluded.
  **Lesson:** sanity-check every number that looks too good, especially in diagnostics.
**Explain independently:** why the LSTM has the best PR-AUC *and* the highest FPR at the same time.

## 13. Independent review and what it changed

**Files:** `src/models/thresholds.py` (`calibration_is_degenerate`), `src/evaluation/metrics.py` (`macro_auc`;
warm-up excluded from AUC), `src/evaluation/evaluate.py` (dual reporting, KS statistic), `EXPERIMENTS.md`.
**What the reviewer found.** The reviewer recomputed everything from the artifacts, and all headline numbers reproduced exactly. But:
* 19 constant-training channels had "calibrated" thresholds of ~0 and produced almost half of all false positives.
  The pipeline reported a number where no calibration existed. **Lesson:** a function that always returns a
  value hides the cases where the value is meaningless. Return a status, and report results with and without those cases.
* The pooled normalised-score AUC (0.753 vs ~0.60) tripled the LSTM's real ranking lead (macro 0.677 vs 0.635) and
  reversed the baseline/IF order. **Lesson:** dividing by a per-channel constant does not make ranks comparable
  across channels. Average per-channel metrics and give an interval.
* "Distribution shift explains the false alarms" was a story, not a measurement. It was replaced by a per-channel KS
  statistic, which gives moderate support for the learned models and none for the baseline.
**Explain independently:** why macro and pooled AUC can rank two methods differently on the same predictions.

## 14. Release preparation: what checking from a fresh clone found

**Files:** `src/evaluation/calibration.py`, `src/models/persistence.py`, `src/inference/predict.py`, `app/dashboard.py`,
`src/models/lstm.py`, `SHIP_PLAN.md`.
* **A fresh environment finds what the dev environment hides.** Installing from `requirements.txt` into a new venv
  and running the tests on a copy without generated files exposed two problems. Three dashboard pages crashed
  without the gitignored scores, and `scipy` was only installed by accident (transitively).
* **A "fix" can change results silently.** The first D5 fix (seed before building the LSTM) changed the dropout
  stream, and a re-run of two channels no longer matched the saved model. Seeding both before construction and
  before training fixed the bug *and* reproduced the published run bit-identically. **Always re-run a reference
  case after touching anything random.**
* **Measure a suspicion before acting on it.** Small validation sets (D4) were suspected of driving the results.
  The bootstrap showed the aggregate effect is small (block bootstrap: LSTM F1 [0.375, 0.389]), so the explanation lies elsewhere.
  The *first* bootstrap resampled points independently and understated the spread about 3×, because the scores are
  autocorrelated (lag-1 ≈ 0.8–0.98). The final audit caught it. **Lesson:** a bootstrap must respect the dependence
  structure of the data, and a resampled extreme quantile cannot go above the sample maximum.
* **Leave-channels-out is how to use labels honestly for calibration.** A rule chosen on other channels and applied
  to held-out channels gives a fair estimate. Choosing it on the evaluated channels would not.
  Pre-register the grid and objective, and report a grid-edge choice instead of widening the grid afterwards.
* **Don't edit code under a running job.** The seed-1 run had already imported `src/utils/config.py` when a new
  helper was added to it. Seventy-four minutes later its final evaluation step imported the *new* `evaluate.py`,
  which expected that helper, and crashed with `ImportError`. Training outputs were complete, so only the
  evaluation was re-run (logged in its `run.log`). **Lesson:** a long job runs a mix of the code it loaded at
  start and code it imports later. Freeze the tree, or work in a separate checkout, while experiments run.
* **Inference must carry the caveats the evaluation knows about.** A threshold flagged uncalibratable in the
  report was silently used by `predict`. The flag now travels with the model.

---

Results and their interpretation: `EXPERIMENTS.md`. Next study steps: the "Files to study first" list in `README.md`.
