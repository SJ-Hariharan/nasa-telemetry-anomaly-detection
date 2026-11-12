# Time-series thinking: ordering, windows and leakage

Tabular ML assumes rows are independent and identically distributed (i.i.d.). Telemetry violates
both: neighbouring points are related, and behaviour changes over time. Most design decisions in
this project follow from taking that seriously.

---

## 1. Stationarity

**Problem.** A detector learns "normal" from the past and applies it to the future. That only works
if the future resembles the past.

**Concept.** A series is (weakly) *stationary* if its mean, variance and autocorrelation do not
depend on time $t$. Telemetry is usually only *locally* stationary: stable within an operating mode,
then a command switches modes.

**Worked example** (seeded NumPy): i.i.d. noise — first-half mean $-0.006$, second-half $-0.048$
(≈ same). A random walk (cumulative sum of the same kind of noise) — half means $-5.16$ vs $-24.84$:
non-stationary; a global mean is meaningless for it.

**In the code.** Rolling windows (`RollingZScoreDetector`, the `roll_*` features in
`isolation_features`) re-estimate "normal" locally, which tolerates slow drift.
`ValueScaler` is global, but it is only an affine rescaling, not a model of normality.

---

## 2. Autocorrelation

**Concept.** Correlation of a series with a lagged copy of itself:

$$\rho(k) = \frac{\mathrm{Cov}(x_t, x_{t-k})}{\mathrm{Var}(x_t)}$$

$k$ = lag in timesteps. $\rho(1)\approx1$ means "the next value is almost the current value".

**Worked example** (`pd.Series.autocorr`): a sine with period 50 → $\rho(1)=0.992$, $\rho(12)=0.079$
(quarter period), $\rho(25)=-1.0$ (half period). White noise → $\rho(k)\approx0$ at all lags.

**Why it matters.** (1) High autocorrelation is what makes *forecasting* possible (the LSTM) and what
makes lag features informative (`lag_0..lag_9`). (2) It breaks i.i.d. assumptions: a random
train/test split puts near-duplicate neighbours on both sides (section 5).

---

## 3. Distribution shift

**Concept.** $P_{\text{train}}(x) \ne P_{\text{test}}(x)$. In this benchmark it is visible directly:
test values were min-max scaled using **test** min/max, and some MSL channels exceed the
documented $[-1,1]$ range in test (M-6 reaches 258). `validate_dataset` warns about each such channel.

**Consequences here.** A detector calibrated on validation may flag a whole new-but-nominal regime
in test (false alarms), or a threshold may be too loose. Per-channel thresholds and rolling (local)
statistics reduce, but cannot remove, this risk. Results by channel: see EXPERIMENTS.md.

---

## 4. Chronological vs random splits

**Problem.** We need data to fit, data to tune (early stopping, thresholds), and data to evaluate.

**Concept.** The benchmark gives a train file and a later test file per channel. Inside train, the
project splits **chronologically**:

```python
# src/preprocessing/scaling.py
def chronological_split_index(n, val_fraction):
    split = int(round(n * (1.0 - val_fraction)))   # [0, split) fit, [split, n) validate
```

**Worked example.** P-1 has 2,872 train rows; `val_fraction: 0.2` → split 2298: rows 0–2297 fit,
2298–2871 validate. (`tests/test_data_and_features.py::test_chronological_split`: 100 → 80.)

**Why not random?** With $\rho(1)\approx1$, a random split puts $x_{t}$ in fit and $x_{t+1}$ in
validation; validation then measures memorisation of neighbours, not generalisation to the future.
A chronological split mimics deployment: train on the past, judge on later data.

**Alternatives.** Expanding-window / rolling-origin cross-validation (several chronological folds;
more robust, more compute); a gap ("purge") between fit and validation to avoid boundary overlap.

---

## 5. Sliding windows (`src/features/windows.py`)

The file's rule: **the score for timestep $t$ may use $x_0..x_t$ only, never $x_{t+1}..$.**

### 5.1 `forecast_windows` (LSTM)

Input $X_i = x_{t-W}..x_{t-1}$ (all features: value + commands), target $y_i = x_t$ (value only),
with $t = $ `target_idx[i]` $= i + W$.

```python
X = sliding_window_view(array, window, axis=0)[:-1].transpose(0, 2, 1)
target_idx = np.arange(window, n)
y = array[target_idx, VALUE_COL]
```

`sliding_window_view` returns a **view** (no copy): $n-W+1$ windows; the last one is dropped
because it has no next value to predict.

**Worked example.** Values $[10,11,12,13,14]$, $W=3$ → `X = [[10,11,12],[11,12,13]]`,
`y = [13,14]`, `target_idx = [3,4]`. `target_idx` is how `LSTMDetector.predict` writes each
prediction back to *its* timestep (`pred[t_idx] = ...`), so the error at $t$ is compared with the
label at $t$. Positions $0..W-1$ stay `NaN`.

### 5.2 `isolation_features` (Isolation Forest)

Features at $t$ describe the trailing window **ending at $t$** ($x_{t-W+1}..x_t$) — the current
point is included because it is the point being judged. Rows with any missing feature are warm-up:
the first `stat_window - 1` rows (rolling stats need `stat_window` points; lags need fewer, enforced
by `stat_window >= lags`).

**Contrast:** the baseline window *excludes* $x_t$ (a reference for comparing $x_t$); the IF window
*includes* $x_t$ (a description of the current local shape). Both are causal.

---

## 6. Warm-up positions

Every detector returns `NaN` where history is insufficient: baseline first `window` (100) points,
IF first `stat_window - 1` (49), LSTM first `window` (100). `apply_threshold` maps them to normal:

```python
pred = (np.nan_to_num(s, nan=-np.inf) > threshold).astype(np.int8)
```

So an anomaly in the first ~100 test points cannot be detected — an honest cost, and identical for
training-time evaluation and for `src/inference/predict.py` (`warmup_length`).

### Why test is not concatenated to train

Prepending the end of train to test would remove warm-up. The project does not: the benchmark
treats train and test as separate files (not guaranteed to be contiguous in time), each channel's
test file is scored on its own, and the saved-model inference path scores any new CSV exactly the
same way — `tests/test_integration.py::test_end_to_end_pipeline` checks that inference reproduces
the pipeline's test scores. A side effect: the LSTM EWMA restarts at the beginning of test.

---

## 7. Leakage and how each type is prevented

*Leakage* = information that would not be available at prediction time influences the model or its
evaluation, making reported results optimistic.

| Type | What it looks like | Prevention in this code | Test |
|---|---|---|---|
| Look-ahead (feature) | a feature at $t$ uses $x_{t+1}$ (e.g. centred rolling mean) | trailing windows only; `shift(1)` in the baseline; `forecast_windows` inputs end at $t-1$ | `test_forecast_windows_alignment` (`(X[:, -1, 0] < y).all()`), `test_isolation_features_are_causal` (changing rows ≥150 leaves features <150 identical) |
| Look-ahead (training) | model trained on windows whose target lies in validation | `LSTMDetector.fit`: `tr = t_idx < split_idx`; IF fits on `feats[:split_idx]` | covered by causality tests above |
| Scaler leakage | mean/std computed on validation or test | `ValueScaler().fit(train[:split])` in `process_channel` | `test_scaler_uses_only_fit_data` |
| Threshold leakage | thresholds or hyper-parameters tuned on test labels | threshold from `validation_scores(det, train_s, split)` (train only); labels joined last | `tests/test_integration.py::test_thresholds_do_not_depend_on_test_labels` |

The threshold test is a *metamorphic* test: it runs training, rewrites the test labels in
`labeled_anomalies.csv`, runs again, and asserts every threshold is identical. If any code path
read test labels, the thresholds would change.

Subtle legitimate case (comment in `LSTMDetector.fit`): a validation window may *look back* into
the fit region — that is past data, available in deployment, so it is not leakage.

**Leakage we cannot remove.** The dataset authors min-max scaled with test min/max before
publication (documented in `src/preprocessing/scaling.py`). Our train-fitted z-scaling neither adds
nor removes it.

## Interview checkpoint

1. **Why is a random split wrong for telemetry?** Strong autocorrelation makes adjacent points nearly
   identical, so validation scores reflect memorisation and are optimistic.
2. **What does `target_idx` guarantee?** That each prediction/error is stored at the timestep it
   predicts, so it is compared with the right label.
3. **Why does the baseline window exclude $x_t$ but the IF features include it?** The baseline
   compares $x_t$ against a reference; IF describes the current local pattern, which must contain $x_t$.
4. **How would you prove thresholds don't use test labels?** Change test labels, retrain, assert
   thresholds are unchanged (`test_thresholds_do_not_depend_on_test_labels`).
5. **What is the cost of not concatenating train and test?** The first `window` test points are
   unscorable (predicted normal).

## Exercises

1. Change `lstm.window` and `baseline.window` to 50 in `configs/smoke.yaml` and rerun
   `python -m src.cli all --config configs/smoke.yaml`. Predict first: how many leading NaN scores in
   `reports/runs/smoke/scores/P-1.parquet`?
2. Write a "leaky" version of the baseline using `v.rolling(W, center=True)` in a scratch script and
   check that `test_isolation_features_are_causal`'s trick (modify the future) changes past scores.
3. Compute `pd.Series(np.load("data/raw/telemanom/train/P-1.npy")[:, 0]).autocorr(k)` for
   k = 1, 10, 100. What does it tell you about how easy one-step forecasting is?
4. Temporarily set `val_fraction: 0.5` in a copy of the smoke config. Predict how `split_idx` in
   `training_summary.json` changes and what that does to LSTM training-window counts.
5. Explain in two sentences why the LSTM validation windows near `split_idx` are not leakage.
