# Isolation Forest, as used in `IsolationForestDetector`

Code: `src/models/detectors.py` (`IsolationForestDetector`), features in
`src/features/windows.py` (`isolation_features`), hyper-parameters in `configs/default.yaml`
(`isolation_forest:` section).

---

## 1. The isolation principle

**Problem.** Density- or distance-based detectors (kNN, LOF) need distances between many points —
expensive, and fragile in high dimensions. We want a fast, assumption-light detector for a
17-feature tabular description of each timestep.

**Why it works.** Anomalies are *few and different*. If you cut the data with random splits, a point
far from the crowd ends up alone after only a few cuts; a point inside a dense cluster needs many.

**Intuition first (1-D).** Points $\{1,2,3,4,100\}$. Pick a random cut uniformly in $[1,100]$.
With probability $(100-4)/(100-1) = 0.97$ the cut lands between 4 and 100 and isolates 100
**in one step**. Isolating 2 needs cuts on both sides of it, both inside short intervals — the first
cut lands in $(2,3)$ with probability only $1/99 \approx 0.01$.

---

## 2. Random partitioning: an isolation tree

**Concept.** An isolation tree (iTree) is built on a subsample by recursion:

1. pick a feature $q$ uniformly at random;
2. pick a split value $p$ uniformly in $[\min q, \max q]$ of the points at this node;
3. send points with $x_q < p$ left, the rest right; recurse until a node holds one point (or all
   points are identical, or a depth limit $\lceil\log_2\psi\rceil$ is reached).

No labels, no impurity criterion — splits are random. An *Isolation Forest* is many such trees
(`n_estimators: 200` in default, 50 in smoke), each on its own random subsample.

**Path length** $h(x)$ = number of edges from the root to the leaf where $x$ ends. Short path ⇒
easy to isolate ⇒ anomalous. When a leaf is hit at the depth limit with $m>1$ points, $c(m)$ is added
to estimate the remaining unbuilt depth.

---

## 3. The anomaly score

**Problem.** Raw path lengths depend on sample size; we want a normalised score.

**Concept.** Normalise the average path length by $c(n)$, the *average path length of an
unsuccessful search in a binary search tree* of $n$ points (the expected depth of a random point):

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n},\qquad H(i) \approx \ln i + 0.5772156649$$

$$s(x, n) = 2^{-\,E[h(x)]/c(n)}$$

- $n$ = subsample size per tree ($\psi$, `max_samples`),
- $H(i)$ = $i$-th harmonic number ($0.5772\ldots$ is the Euler–Mascheroni constant),
- $E[h(x)]$ = mean path length of $x$ over all trees,
- $s \in (0,1]$.

Reading the score: $E[h]\to 0 \Rightarrow s\to1$ (anomaly); $E[h]=c(n) \Rightarrow s=0.5$
(typical); $E[h]\gg c(n) \Rightarrow s\to0$ (deep inside a cluster).

**Worked example** (verified against sklearn's `_average_path_length`): $c(256) = 10.2448$.
- $E[h]=2$ → $s=2^{-2/10.24}=0.873$ (strong anomaly)
- $E[h]=4$ → $s=0.763$
- $E[h]=10.24$ → $s=0.500$
- $E[h]=16$ → $s=0.339$ (very normal)

Edge case: the formula gives $c(2)=0.154$, but sklearn (and the original paper) define $c(2)=1$
and $c(1)=0$.

---

## 4. Subsampling: `max_samples=256`

**Why small subsamples help.** Each tree sees only $\psi=256$ points drawn without replacement.
(a) *Speed*: tree size depends on $\psi$, not on the training length. (b) *Swamping* (normals near
anomalies are mis-flagged) and *masking* (a dense group of anomalies hides itself) both decrease
when each tree sees a sparser sample — clustered anomalies are less likely to be drawn together.
256 is the value recommended in the original paper; scores barely improve beyond it.

For a typical channel (P-1: 2,298 fit rows, 2,249 after the 49-row warm-up) each of the 200 trees
sees a different 256-row subsample, and together they cover the whole fit period. The shortest train
file has only 312 rows (→ 250 fit rows, ~201 complete feature rows); there sklearn automatically lowers
`max_samples` to the number of rows (with a warning), so $c(n)$ uses that smaller $n$.

---

## 5. Contamination — and why this project ignores it for thresholds

**Concept.** `contamination` is sklearn's assumed fraction of anomalies in the **training** data. It
only sets `offset_`, which `predict()` and `decision_function()` use to say inlier/outlier. With
`"auto"`, `offset_ = -0.5` (i.e. $s>0.5$ ⇒ outlier).

**Verified:** on the same toy data, `contamination="auto"` and `contamination=0.2` give *identical*
`score_samples`; only `offset_` changes ($-0.5$ vs $-0.503$).

**This project** passes `contamination: auto` but never calls `predict()`. It uses the continuous
score and calibrates a per-channel threshold on validation scores (99.5th percentile, see
`src/models/thresholds.py`), exactly like the other two detectors. Reasons: (1) the true anomaly rate
in train is unknown; (2) a fixed $s>0.5$ cut ignores per-channel differences; (3) the same
calibration rule for all methods makes the comparison fair.

---

## 6. sklearn's sign convention and why the code negates it

`IsolationForest.score_samples(X)` returns $-s(x,\psi)$: **higher = more normal** (sklearn's
convention: "bigger is better/inlier"). Toy data $\{1,2,3,4,100\}$ → `score_samples`
`[-0.446, -0.359, -0.351, -0.418, -0.730]`; negated → `[0.446, 0.359, 0.351, 0.418, 0.730]`: the
outlier 100 has the highest value, and the edge points 1 and 4 are slightly higher than the centre.

```python
def score(self, array_scaled):
    feats, _ = isolation_features(array_scaled, self.lags, self.stat_window)
    scores = np.full(len(feats), np.nan)
    ok = np.isfinite(feats).all(axis=1)
    if ok.any():
        scores[ok] = -self.model.score_samples(feats[ok])
    return scores
```

Negation makes "higher = more anomalous" consistent with the baseline and the LSTM, so
`calibrate_threshold`/`apply_threshold` (which test `score > thr`) work unchanged. The negated value
is exactly the paper's $s(x,\psi)$. Rows with NaN features (warm-up) stay `NaN`.

---

## 7. Feature design: turning a time series into a table

**Problem.** Isolation Forest sees each row independently — it has no notion of time. A spike at $t$
is only "different" if the row describes $x_t$ *relative to its recent past*.

**Concept.** `isolation_features(array, lags=10, stat_window=50)` builds, for every $t$, from the
trailing window ending at $t$:

| Feature | Formula | Captures |
|---|---|---|
| `lag_0..lag_9` | $x_t, x_{t-1}, \dots, x_{t-9}$ | local shape |
| `roll_mean`, `roll_std` | mean, std (`ddof=0`) of $x_{t-49..t}$ | local level, spread |
| `roll_min`, `roll_max` | min, max of $x_{t-49..t}$ | local range |
| `dev_from_mean` | $x_t - \text{roll\_mean}_t$ | departure from local level |
| `diff_1` | $x_t - x_{t-1}$ | sudden jumps |
| `cmd_activity` | number of command flags set in the last 50 steps | operational context |

That is 10 + 4 + 2 + 1 = 17 features (16 if the array has no command columns). The first
`stat_window - 1 = 49` rows are warm-up.

**Worked example** (`lags=2`, `stat_window=3`), values $[0,1,2,3,10,5]$, one command column
$[0,1,0,0,1,0]$; row $t=4$ (the jump to 10):

```
lag_0=10  lag_1=3  roll_mean=5  roll_std=3.559  roll_min=2  roll_max=10
dev_from_mean=5  diff_1=7  cmd_activity=1
```

`diff_1=7` and `dev_from_mean=5` are far outside what rows $t=2,3$ show (1 and 1), so random splits on
those features isolate this row quickly.

**Fit only on the fit part.** Because every feature at $t$ looks backwards,

```python
feats, self.feature_names = isolation_features(train_scaled, self.lags, self.stat_window)
fit_rows = feats[:split_idx]                         # rows < split never see validation data
fit_rows = fit_rows[np.isfinite(fit_rows).all(axis=1)]
```

Causality is tested by `tests/test_data_and_features.py::test_isolation_features_are_causal`.

---

## 8. Strengths and weaknesses

**Strengths**
- Linear-time training on small subsamples; no distance matrix; few hyper-parameters.
- No distributional assumption (unlike the Gaussian-ish baseline).
- Multivariate: can flag a combination (e.g. normal value but unusual `roll_std` and `cmd_activity`).

**Weaknesses**
- **Axis-parallel splits.** Each cut is on one feature. Anomalies that are only unusual along a
  *diagonal* (a correlation break between features) are isolated slowly, and the score map shows
  rectangular artefacts. Extended Isolation Forest (random hyperplanes) addresses this.
- **No temporal memory beyond engineered features.** The model only knows the last 10 values and
  50-step statistics. A slow drift over 500 steps, or a pattern tied to a command issued 200 steps
  earlier, is invisible (the LSTM is the project's answer to this).
- **Masking.** Many similar anomalies form their own small cluster and stop being "few". In
  telemetry, a long anomalous plateau quickly fills the 50-step rolling window and its rows start
  to look self-consistent.
- **Swamping.** Normal points close to anomalies get shorter paths and higher scores.
- **Correlated, redundant features.** Ten lag features of a smooth signal are nearly identical;
  random feature choice then over-samples "level" information relative to `diff_1` or `roll_std`.
- **Scores near the boundary are noisy** with few trees; 200 trees stabilise $E[h(x)]$.

**Alternatives.** Extended IF; One-Class SVM; LOF; autoencoders / forecasting models (the LSTM here).
Results for this detector: see EXPERIMENTS.md.

## Interview checkpoint

1. **What does $s=0.5$ mean?** $E[h(x)]=c(n)$: the point is as hard to isolate as a typical point.
2. **Why is `score_samples` negated?** sklearn returns higher-is-normal; the project needs
   higher-is-anomalous for a common thresholding rule. The negated value is the paper's $s$.
3. **Does `contamination` affect the scores here?** No. It only changes `offset_` used by
   `predict`; the project calibrates its own validation-quantile threshold.
4. **Why `max_samples=256` rather than all data?** Faster trees and less masking/swamping; more data
   per tree adds little.
5. **How does a stateless model detect a temporal anomaly?** Only through trailing-window features
   (lags, rolling stats, `diff_1`, `cmd_activity`) that encode recent history into each row.

## Exercises

1. In `configs/smoke.yaml`, change `isolation_forest.stat_window` to 20 and rerun
   `python -m src.cli all --config configs/smoke.yaml`. Predict first: how many leading NaN IF scores
   per channel, and will short spikes become more or less visible?
2. Set `lags: 60` with `stat_window: 50` and run. Predict the error message before you run it
   (`isolation_features` raises when `stat_window < lags`).
3. In a scratch script, fit `IsolationForest` on 2-D points on the line $y=x$ plus one point at
   $(1,-1)$. Compare its score with a point at $(3,3)$ beyond the cloud's end. What does this say
   about axis-parallel splits?
4. Load a saved model bundle for P-1 and print `get_params()["feature_names"]`. Count the features
   and match each to a row in the table in section 7.
5. Compute $c(n)$ for $n = 64, 256, 1024$ with the formula above, then explain why a point with
   $E[h]=6$ gets a higher score under $n=1024$ than under $n=64$.
