# Anomaly detection: framing, scores, thresholds and decisions

## 1. What counts as an anomaly?

**Problem.** "Find the faults" is not a well-defined ML task until we say what kind of deviation we
are looking for.

**Concept.** Three textbook types:

| Type | Definition | Telemetry example |
|---|---|---|
| Point | one value is extreme regardless of context | a single spike to $-3$ |
| Contextual | a value is normal in general but wrong *here* (given time, mode, commands) | a voltage that is fine in mode A but not right after command B |
| Collective | each point is plausible, the *sequence* is not | a sensor freezing at a plausible value; a shifted oscillation |

The benchmark's `class` column uses only **`point`** and **`contextual`**; its "contextual" events
often behave collectively (sustained level shifts, changed patterns). `load_labels` keeps one class
per (merged) sequence in `classes`.

**Worked example** (`tests/conftest.py` synthetic data): a sine wave with a 40-point level shift
(`test[600:640, 0] += 2.0`, labelled contextual) and one spike (`test[850, 0] = -3.0`, point).
Labels `[[600, 639], [850, 850]]` → 40 + 1 = 41 anomalous points
(`test_end_to_end_pipeline` asserts `table["label"].sum() == 41`).

**Important.** In this project an anomaly is a **statistically unusual observation relative to
learned normal behaviour**, not a confirmed fault. A planned manoeuvre can be unusual; a slowly
developing fault can look normal. Detections are prompts for an engineer to look, not diagnoses.

---

## 2. Unsupervised vs supervised framing

**Problem.** Labels are scarce: 104 anomaly sequences across 81 channels, all in **test**. Train has
no labels.

**Concept.**
- *Supervised*: learn $P(\text{anomaly}\mid x)$ from labelled examples. Needs many labelled anomalies
  and assumes future anomalies look like past ones.
- *Unsupervised / semi-supervised (novelty detection)*: learn only what "normal" looks like (train
  is assumed mostly nominal) and flag departures. Can catch never-seen failure modes.

This project is the second kind: every detector fits on the train file only. Test labels are used
**only for evaluation**. Since ~12.5% of test points are labelled anomalous, a classifier that says
"always normal" would score ~87.5% accuracy while finding nothing — hence precision/recall/F1 and
event metrics in `src/evaluation/metrics.py`.

---

## 3. Scores vs decisions

**Concept.** A detector outputs a continuous **score** $s_t$ (higher = more unusual). A **decision**
needs a threshold $\tau$: $\hat y_t = \mathbb{1}[s_t > \tau]$. Separating them lets us:
- compare detectors threshold-free (ROC-AUC, PR-AUC in `threshold_free`, computed per channel and
  macro-averaged; a pooled variant on $s_t/\tau$ is kept as a secondary figure);
- change the operating point without retraining (`SENSITIVITY_MULTIPLIERS` in
  `src/evaluation/evaluate.py` multiplies thresholds by 0.5…5).

Scores are **not probabilities**: a baseline score of 6 and an IF score of 0.6 are on different
scales (`src/models/detectors.py` docstring).

---

## 4. The shared detector interface

```python
det.fit(train_scaled, split_idx)   # learn only from train rows [0, split_idx)
scores = det.score(array_scaled)   # one score per timestep, NaN during warm-up
```

| Detector | `fit` | `score` |
|---|---|---|
| `RollingZScoreDetector` | nothing to learn | past-window z-score |
| `IsolationForestDetector` | forest on `isolation_features(...)[:split_idx]` | negated `score_samples` |
| `LSTMDetector` | forecaster on windows with target `< split_idx`, early-stopped on the rest | EWMA of $|x_t-\hat x_t|$ |

**Why it matters.** `process_channel`, threshold calibration, evaluation, persistence
(`DetectorBundle`) and inference (`detect`) treat all three identically. Adding a fourth detector
means implementing these two methods (plus `get_params`) and registering it in `build_detector`
(`src/pipeline.py`) and in `METHODS`/save-load logic (`src/models/persistence.py`). Passing the **whole** train array with
`split_idx` (instead of only the fit part) lets validation windows legitimately look back into the
fit region. Each detector also exposes `warmup` (history steps it needs before it can score).

---

## 5. Per-channel threshold calibration on validation

**Problem.** One global threshold cannot work: channels differ in noise level, and scores differ
across detectors.

**Concept.** For each (channel, method):

$$\tau = \text{margin}\cdot Q_{q}\big(\{s_t : t \in [\text{split}, n_{\text{train}})\}\big),\qquad q=0.995,\ \text{margin}=1$$

i.e. the 99.5th percentile of that detector's scores on the validation part of **train**, assumed
nominal. Expected flag rate on nominal data ≈ 0.5%. The pipeline records the realised validation
flag rate as `val_flag_rate` in `training_summary.json`.

```python
val_scores = validation_scores(det, train_s, split)  # = det.score(train_s[split - warmup:])[warmup:]
thr = calibrate_threshold(val_scores, **cfg["threshold"])
scores = det.score(test_s)
pred = apply_threshold(scores, thr, post["min_event_length"], post["merge_gap"])
```

**A subtle bug this project fixed.** The first version used `det.score(train_s)[split:]`. For the LSTM,
the EWMA smoothing then ran continuously from the fit part into validation, so early validation scores
contained decayed *in-sample* errors (small, because the model was trained on them). The test file, by
contrast, is scored on its own, with smoothing starting fresh. Calibration and test scoring were therefore
done under different conditions, which nudged thresholds down. `validation_scores` (`src/pipeline.py`)
now scores a slice starting exactly `warmup` steps before the split, which mirrors how test scoring
starts. `tests/test_models_and_metrics.py::test_validation_scores_use_test_like_conditions` perturbs old
fit-part data and asserts validation scores do not change. Lesson: *calibration data must be scored by
exactly the same procedure as the data the threshold will be applied to.*

**Worked example.** Validation scores `[0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.9, 0.18, 0.22, 0.12]`
with `quantile=0.9` → $\tau=0.36$. A test score of 0.5 is flagged; 0.3 is not.

### Why test labels must never tune thresholds

If you pick $\tau$ to maximise test F1, the reported F1 is the *best case over thresholds* on that
exact data — optimistic, and not reproducible on a new mission where labels don't exist. In
deployment you only have nominal history, which is exactly what validation simulates. Enforced by
`tests/test_integration.py::test_thresholds_do_not_depend_on_test_labels`.

**Alternatives.** Telemanom's non-parametric dynamic threshold (on smoothed errors, with pruning);
POT/extreme-value theory (fit a generalised Pareto tail); fixed contamination rate. **Limitations.**
An anomaly inside validation raises $\tau$; a regime change between validation and test causes
floods of false alarms or misses.

---

## 6. Post-processing (`apply_threshold` in `src/models/thresholds.py`)

**Problem.** Raw point decisions flicker: an anomaly produces `1,1,0,1,1` and noise produces
isolated `1`s. Operators want *events*.

**Concept.** Two knobs, applied in order:
1. `merge_gap` $g$: two predicted runs separated by $\le g$ normal points are joined.
2. `min_event_length` $L$: predicted runs shorter than $L$ points are removed.

**Worked example** (threshold 1.0, `runs` gives inclusive `(start, end)`):

```
scores          [nan, .5, 2, .1, 2, 2, .3, .2, 2, .1]
raw             [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]      runs (2,2) (4,5) (8,8)
merge_gap=1     [0, 0, 1, 1, 1, 1, 0, 0, 1, 0]      gap of 1 at index 3 filled
min_len=2       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]      single points removed
both            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]      merge first, so (2..5) survives
```

Defaults (`configs/default.yaml`): `min_event_length: 1`, `merge_gap: 0` → post-processing is off
and results reflect the raw detector. Turning it on trades latency and short-event recall for fewer
false-alarm events.

---

## 7. Evaluation, briefly

`src/evaluation/metrics.py` reports point-wise and event-wise metrics (an event is detected if any
predicted point falls inside it; a false-alarm event is a predicted run overlapping no label). It
explicitly avoids "point-adjust", which inflates point F1. Numbers: see EXPERIMENTS.md.

## Interview checkpoint

1. **Why is accuracy misleading here?** ~12.5% of test points are anomalous; "always normal" gets
   high accuracy with zero recall.
2. **Why separate scores and thresholds?** Threshold-free comparison (AUC), and the operating point
   can be changed without retraining.
3. **Why calibrate per channel?** Noise levels and score scales differ by sensor and detector.
4. **What goes wrong if you tune τ on test labels?** Optimistic, non-reproducible metrics; no labels
   exist in deployment.
5. **Is a flagged point a fault?** No — it is statistically unusual relative to training behaviour;
   it needs engineering confirmation.

## Exercises

1. Set `postprocess: {merge_gap: 5, min_event_length: 3}` in a copy of `configs/smoke.yaml` and rerun
   `python -m src.cli all --config <copy>`. Predict the effect on event false alarms and on
   single-point (`point`) anomalies before looking.
2. Open `reports/runs/smoke/training_summary.json`: compare each method's `val_flag_rate` with the
   0.5% the quantile implies. Why can it differ (hint: ties, few validation points)?
3. Implement a fourth detector (e.g. global z-score using the fit-part mean/std) with the same
   `fit`/`score` interface in a scratch file and score one channel with it.
4. Using `apply_threshold`, construct a score sequence where `merge_gap=1` then `min_event_length=3`
   gives a different result than applying them in the opposite order.
5. From `reports/runs/smoke/scores/T-1.parquet`, plot `baseline_score / threshold` and mark where
   `label == 1`. Are flagged points inside or before the labelled interval?
