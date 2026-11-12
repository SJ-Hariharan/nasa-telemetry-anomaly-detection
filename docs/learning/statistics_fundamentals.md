# Statistics fundamentals, as used in this project

Every detector here ends in the same statistical question: *is this number unusually large compared
with what we saw during normal operation?* This document builds the tools for answering it and
shows exactly where each one lives in the code.

---

## 1. Mean and standard deviation

**Problem.** We need a "typical value" and a "typical spread" for a channel.

**Concept.** For values $x_1,\dots,x_n$:

$$\mu = \frac{1}{n}\sum_{i=1}^n x_i \qquad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n (x_i-\mu)^2}$$

$\mu$ = mean (centre), $\sigma$ = population standard deviation (average distance from the centre,
in the same units as $x$). Dividing by $n-1$ instead of $n$ (`ddof=1`) gives the unbiased *sample*
estimate; for windows of 50–100 points the difference is ~1%. The project uses `ddof=0`
everywhere: `np.std` default in `ValueScaler.fit`, explicit `std(ddof=0)` in the rolling code.

**Worked example.** $x = [1,2,3,4,10]$ → $\mu = 4$, $\sigma = \sqrt{(9+4+1+0+36)/5} = \sqrt{10} = 3.162$
(`ddof=1` would give 3.536). Note how the single 10 dominates $\sigma$ — mean and std are **not
robust** to outliers.

**Alternatives.** Median and MAD (median absolute deviation) are robust: one outlier barely moves them.

---

## 2. Z-scores

**Problem.** "0.3 above average" means nothing without knowing the spread.

**Concept.** Express distance in units of spread: $z = (x-\mu)/\sigma$. $z=3$ means "three standard
deviations above the mean", whatever the channel's units.

**Worked example.** With all five points above, $z(10) = (10-4)/3.162 = 1.90$ — not alarming,
because the 10 inflated its own $\sigma$. Using only the *other* four points, $\mu=2.5$,
$\sigma=1.118$, and $z(10) = 7.5/1.118 = 6.71$ — clearly unusual. This is exactly why the baseline
excludes $x_t$ from its own reference window (section 6).

---

## 3. The Gaussian assumption

**Concept.** If data are Gaussian (normal), $P(|z|>3) \approx 0.27\%$ and $P(|z|>4) \approx 0.006\%$,
so "$|z|>3$ ⇒ anomaly" has a known false-alarm rate.

**Why it fails for telemetry.** Spacecraft telemetry is often: flat for long stretches
($\sigma\approx0$), step-like (mode switches), periodic, or heavy-tailed. Then the 0.27% figure is
meaningless. That is why this project **does not use a fixed z-threshold like 3**; it calibrates a
per-channel threshold from that channel's own validation scores (section 9). The
`RollingZScoreDetector` docstring states the assumption and its violation explicitly.

---

## 4. Quantiles

**Problem.** We want a threshold that a known fraction of normal scores stays below, without
assuming a distribution.

**Concept.** The $q$-quantile $Q_q$ is the value below which a fraction $q$ of the data lie. NumPy's
default (`np.quantile`, "linear" method) sorts the data $s_{(1)}\le\dots\le s_{(n)}$ and
interpolates at position $h=(n-1)q$ (0-based): $Q_q = s_{(\lfloor h\rfloor)} + (h-\lfloor h\rfloor)(s_{(\lfloor h\rfloor+1)} - s_{(\lfloor h\rfloor)})$.

**Worked example.** Scores $1,2,\dots,200$: $h = 199 \times 0.995 = 198.005$ → $Q_{0.995} = 199 + 0.005 = 199.005$.
Exactly one score (200) exceeds it: 0.5% of 200. Quantiles are *distribution-free*: they work for
skewed or heavy-tailed scores.

---

## 5. Rolling (moving-window) statistics

**Problem.** A channel's normal level drifts over time; a single global mean is wrong in most places.

**Concept.** Compute mean and std over the last $W$ points only:

$$\mu_t = \frac{1}{W}\sum_{i=t-W}^{t-1} x_i,\qquad \sigma_t = \sqrt{\frac{1}{W}\sum_{i=t-W}^{t-1}(x_i-\mu_t)^2}$$

$W$ = window length (`baseline.window: 100` in `configs/default.yaml`). Here the window covers
$t-W..t-1$ (past only). The first $W$ positions have no full window → no score (warm-up).

**Trade-off.** Small $W$ adapts fast but is noisy; large $W$ is stable but slow to follow real
level changes (and slow to "forget" an anomaly that entered the window).

---

## 6. The baseline: past-only rolling z-score (`src/models/detectors.py`)

$$\text{score}_t = \frac{|x_t - \mu_t|}{\max(\sigma_t,\ \text{min\_std})}$$

```python
def score(self, array_scaled):
    v = pd.Series(array_scaled[:, VALUE_COL].astype(np.float64))
    past = v.shift(1).rolling(self.window, min_periods=self.window)  # x_{t-W}..x_{t-1}
    z = (v - past.mean()).abs() / past.std(ddof=0).clip(lower=self.min_std)
    return z.to_numpy()
```

`shift(1)` moves the series one step forward, so the window ending at row $t$ contains
$x_{t-W}..x_{t-1}$. `min_periods=self.window` makes the first $W$ scores `NaN`.

**Why past-only?** (a) The spike would inflate its own $\sigma_t$ (section 2: 1.90 vs 6.71);
(b) it is causal — usable in real time.

**Worked example** (`window=4`): $x = [1,2,3,4,10,4]$ → scores `[nan, nan, nan, nan, 6.708, 0.241]`.
At $t=5$, the window $[2,3,4,10]$ now *contains the spike*, so $\sigma_5$ is large and the return to 4
looks normal. A big outlier in the window temporarily desensitises the detector (*masking*).

### The `min_std` floor

If the past window is perfectly flat, $\sigma_t = 0$ and any tiny change gives $z=\infty$.
`clip(lower=self.min_std)` with `min_std: 0.01`. Example (`window=3`): past $[1,1,1]$, then
$x=1.05$ → $0.05/0.01 = 5.0$ instead of infinity. The floor is in *scaled* units (train-std units
after `ValueScaler`), so it means "changes smaller than 1% of the channel's training std are never
dramatic". For constant-train channels the scaled unit is the original unit (section 7).

---

## 7. `ValueScaler` (`src/preprocessing/scaling.py`)

**Problem.** Channels have different scales; the LSTM and the shared `min_std` need comparable units.

**Concept.** Standardise the value column: $x' = (x - \mu_{\text{fit}})/\sigma_{\text{fit}}$, with
$\mu_{\text{fit}}, \sigma_{\text{fit}}$ computed **only on the fit part** of train (`train[:split]`).
Command columns (0/1) are left untouched.

```python
def fit(self, fit_array):
    values = fit_array[:, VALUE_COL]
    self.mean = float(values.mean())
    std = float(values.std())
    self.std = std if std >= self.MIN_STD else 1.0     # MIN_STD = 1e-3
```

**Constant-channel fallback.** 19 channels are constant in train. With $\sigma_{\text{fit}}=0$,
dividing would turn a test change of 0.001 into $\infty$-like inputs. Using $\sigma=1$ keeps the
original units (just shifted by the mean). Worked example
(`tests/test_data_and_features.py::test_scaler_constant_channel_keeps_units`): fit on all $-1$ →
mean $-1$, std 1; test value 0 → $(0-(-1))/1 = 1.0$.

**Why fit only on the fit part?** `test_scaler_uses_only_fit_data`: fit on $[0, 2]$ → mean 1, std 1;
an extreme test value 100 becomes 99 and does **not** change the statistics. Fitting on test (or
validation) would leak information about the future into the inputs.

**Alternatives.** Min-max scaling (sensitive to one extreme value), robust scaling (median/IQR).

---

## 8. EWMA: smoothing LSTM errors (`src/models/lstm.py`)

**Problem.** One-step forecast errors $e_t = |x_t - \hat x_t|$ are spiky; single noisy errors cause
false alarms, while a real anomaly produces a *sustained* run of large errors.

**Concept.** Exponentially weighted moving average: recent errors weigh more, older ones decay
geometrically. With smoothing factor $\alpha = 2/(\text{span}+1)$:

$$\text{EWMA}_t = \frac{\sum_{i=0}^{t} (1-\alpha)^i\, e_{t-i}}{\sum_{i=0}^{t}(1-\alpha)^i}$$

(pandas' default `adjust=True` form; the recursive form $s_t = \alpha e_t + (1-\alpha)s_{t-1}$ is
`adjust=False` and differs only near the start). `smoothing_span: 30` → $\alpha = 2/31 \approx 0.065$.
It is **causal**: $\text{EWMA}_t$ uses $e_0..e_t$ only.

**Worked example** (span 3, $\alpha=0.5$): $e = [0,0,0,4,0,0]$ →
`[0, 0, 0, 2.133, 1.032, 0.508]`. At $t=3$: $4/(1+0.5+0.25+0.125) = 2.133$. The single spike is
halved and decays; a sustained error run would keep the EWMA high.

```python
err = np.abs(array_scaled[:, VALUE_COL] - self.predict(array_scaled))
return pd.Series(err).ewm(span=self.smoothing_span, ignore_na=True).mean().where(~np.isnan(err)).to_numpy()
```

`.where(~np.isnan(err))` keeps warm-up positions `NaN`, so they are never flagged.

**Alternatives.** Simple moving average (equal weights, hard cut-off); Telemanom's original uses EWMA
plus a dynamic, non-parametric threshold.

---

## 9. Quantile thresholds (`src/models/thresholds.py`)

$$\text{thr} = \text{margin}\times Q_q(\text{validation scores}),\quad q = 0.995,\ \text{margin}=1.0$$

Validation scores = scores on the last 20% of the **train** file (`val_fraction: 0.2`), assumed mostly
nominal. If validation and test behave alike, about 0.5% of normal test points will be flagged.

```python
s = s[np.isfinite(s)]                       # drop warm-up NaNs
if method == "quantile":
    thr = np.quantile(s, quantile)
elif method == "mean_std":
    thr = s.mean() + k_std * s.std()
thr = float(thr * margin)
return thr if thr > 0 else float(np.finfo(np.float32).eps)
```

**Worked example.** Validation scores `[0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.9, 0.18, 0.22, 0.12]`:
`quantile=0.9` → 0.36 (only 0.9 exceeds it). `mean_std` with `k_std=2` → $0.262 + 2(0.220) = 0.702$ —
note how the single 0.9 inflates $\sigma$, the same non-robustness as in section 1.
A constant validation signal gives $Q=0$; the `eps` guard ($\approx1.19\times10^{-7}$) prevents
"every non-zero test score is anomalous" from being literally `> 0`.

**Limitations.** If validation contains an anomaly or is unrepresentative of test (regime change),
the threshold is wrong — too high or too low. Results: see EXPERIMENTS.md (including the threshold
sensitivity table produced by `src/evaluation/evaluate.py`).

## Interview checkpoint

1. **Why is the z-score of an outlier smaller when the outlier is included in the statistics?** It
   inflates both $\mu$ and especially $\sigma$ (squared deviation), shrinking its own $z$.
2. **Why a quantile threshold instead of $|z|>3$?** Scores are not Gaussian; a quantile gives a known
   flag rate on validation without assuming a distribution.
3. **What does `min_std` protect against?** Division by ~0 on flat windows, which would make tiny
   changes look infinitely anomalous.
4. **Why does `ValueScaler` use std = 1 for constant channels?** Their fit std is 0; keeping original
   units avoids exploding test values.
5. **What does EWMA span 30 trade off?** Fewer false alarms from single noisy errors vs. a delay and
   damping for very short anomalies.

## Exercises

1. Predict, then compute with `RollingZScoreDetector(window=4).score(...)`, the scores for
   `[0,0,0,0,1,0,0,0,0]` with `min_std=0.01` and with `min_std=1.0`.
2. In `configs/smoke.yaml`, set `threshold: {quantile: 0.95}` and rerun
   `python -m src.cli all --config configs/smoke.yaml`. Predict first: more or fewer false alarms?
3. Compute `pd.Series([0,0,0,4,4,4,0]).ewm(span=3).mean()` by hand for the first four values, then check.
4. Using `reports/data_validation.json`, pick a constant-train channel and explain what its baseline
   score equals at the first test point that differs from the train value.
5. Switch `threshold.method` to `mean_std` in a copy of the smoke config. Which channels' thresholds
   change most (`training_summary.json`), and why?
