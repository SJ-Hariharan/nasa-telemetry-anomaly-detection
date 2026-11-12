# Evaluating anomaly detectors — exactly as this project does it

Code referenced: `src/evaluation/metrics.py` (`point_counts`, `point_metrics`, `event_counts`, `event_metrics`,
`threshold_free`), `src/evaluation/evaluate.py` (`evaluate_run`), `src/models/thresholds.py`.
Every worked example below was computed by calling the project's own functions with `.venv/bin/python`.
Real results: see `EXPERIMENTS.md` (not quoted here).

Dataset context: 81 labelled channels, 104 anomaly sequences (after merging P-2's duplicate label rows), and roughly 12.5% of test timesteps anomalous overall — far less in many individual channels.

**Running example** (20 test timesteps, labelled sequences `[[4, 7], [15, 15]]`):
```
index : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
y_true: 0 0 0 0 1 1 1 1 0 0 0  0  0  0  0  1  0  0  0  0
y_pred: 0 0 0 0 0 0 1 1 1 0 0  0  1  0  0  0  0  0  0  0
```

---

## 1. Confusion matrix and point-wise metrics

**Problem.** "How good is the detector?" has two different failure modes: false alarms and missed anomalies.
**Why.** Operators care about both, at very different costs; a single number hides which one you have.

**Concept.** Each timestep is a sample. TP = predicted 1 & label 1; FP = predicted 1 & label 0; TN = 0 & 0; FN = 0 & 1.

| Metric | Formula | Question it answers |
|---|---|---|
| Precision | $TP/(TP+FP)$ | Of the alarms, how many were real? |
| Recall (TPR) | $TP/(TP+FN)$ | Of the anomalous points, how many did we flag? |
| F1 | $2PR/(P+R)$ | Harmonic mean — low if either is low |
| FPR | $FP/(FP+TN)$ | Fraction of normal points falsely flagged |
| FNR | $FN/(FN+TP)=1-\text{recall}$ | Fraction of anomalous points missed |
| Accuracy | $(TP+TN)/N$ | Fraction correct — misleading here (§2) |

**Example.** `point_counts(y_true, y_pred)` → `{tp: 2, fp: 2, tn: 13, fn: 3}` (TP at 6, 7; FP at 8, 12; FN at 4, 5, 15).
`point_metrics` → precision $2/4=0.5$, recall $2/5=0.4$, F1 $=2(0.5)(0.4)/0.9=0.444$, FPR $2/15=0.133$, FNR $0.6$, accuracy $15/20=0.75$.

**In the code.**
```python
return {"tp": int((y_true & y_pred).sum()), "fp": int((~y_true & y_pred).sum()),
        "tn": int((~y_true & ~y_pred).sum()), "fn": int((y_true & ~y_pred).sum())}
```
`_safe_div` returns 0.0 when a denominator is 0 (e.g. precision of a detector that never fires) — a convention, not a measurement.

**Limitation.** Point-wise metrics weight long anomalies heavily: one 2,000-point anomaly counts 2,000× more than a 1-point spike. Also, the EWMA tail of the LSTM score produces FP points right after true events.

---

## 2. Why accuracy misleads under imbalance

**Example.** 1,000 points, 125 anomalous (12.5%), detector says "normal" always:
`point_metrics(point_counts(y, zeros))` → accuracy **0.875**, recall 0, F1 0. In a channel with 2% anomalies, the same useless detector scores accuracy 0.98.
Accuracy is dominated by TN, which is abundant and uninformative. The project still reports it (in `comparison.csv`) so you can *see* it mislead; `tests/test_models_and_metrics.py::test_accuracy_misleads_on_imbalance` encodes this.

**Interview checkpoint (§1–2)**
- *Which metric would an operator with alarm fatigue watch?* Precision / FPR (and event FP count).
- *Why can FPR look tiny while precision is poor?* FPR divides by the huge number of normal points; even FPR = 1% on 10⁵ normal points is 1,000 false alarm points.
- *Why report FNR if it equals 1 − recall?* Convenience; mission teams talk about "missed anomalies".

---

## 3. Event-wise evaluation

**Problem.** An operator sees *incidents*, not timesteps. Detecting any part of an anomaly usually suffices to start an investigation.
**Why.** Gives each labelled sequence equal weight regardless of length, and counts false alarms as incidents.

**Concept** (as in `event_counts`, following Hundman et al. 2018):
- event **TP**: a labelled sequence with ≥ 1 predicted point inside it (inclusive bounds);
- event **FN**: a labelled sequence with no predicted point;
- event **FP**: a contiguous predicted run (`runs(y_pred)`) that overlaps **no** labelled sequence;
- **latency**: index of the first predicted point inside the event, counted from its start (0 = detected at onset).
Event precision $=TP/(TP+FP)$, event recall $=TP/(TP+FN)$, event F1 as usual; `median_latency` over detected events.

**Example.** Same arrays, `event_counts([[4, 7], [15, 15]], y_pred)` → `{event_tp: 1, event_fn: 1, event_fp: 1, latencies: [2]}`.
Event [4,7] hit first at index 6 ⇒ latency $6-4=2$. Run [6,8] overlaps a label (not FP, although point 8 was a point FP). Run [12,12] overlaps nothing ⇒ 1 FP event. Event [15,15] missed.
`event_metrics` → precision 0.5, recall 0.5, F1 0.5, median latency 2.0. Note how this differs from point F1 0.444.

**In the code.**
```python
for s, e in sequences:
    truth[s : e + 1] = True
    hits = np.flatnonzero(y_pred[s : e + 1])
    if hits.size: tp += 1; latencies.append(int(hits[0]))
    else:         fn += 1
fp = sum(1 for s, e in runs(y_pred) if not truth[s : e + 1].any())
```
**Alternatives.** Range-based precision/recall (Tatbul et al. 2018) that rewards overlap size and penalises fragmentation; time-tolerant windows; NAB scoring.
**Limitations.** "Any overlap" is lenient: one lucky point inside a 5,000-step anomaly counts as full detection. A predicted run that overlaps a label is never a false alarm even if it extends far beyond it. One false alarm that flickers into 3 runs counts as 3 FP events (`min_event_length`/`merge_gap` in `postprocess` exist for this but default to 1/0).

---

## 4. Why "point-adjust" is NOT used

**Concept.** Point-adjust (common in some papers): if any point of a true segment is detected, mark the **whole** segment as detected, then compute point metrics.
**Example.** 200 points, one labelled segment [50,149]; detector flags only 60 and 180.
Raw point metrics: precision 0.5, recall 0.01, F1 **0.020**. After point-adjust: TP=100, precision 0.990, recall 1.0, F1 **0.995**.
A detector that hit 1 of 100 anomalous points is reported as near-perfect. Even random scores can look strong under point-adjust, which is why this project keeps point-wise metrics raw and reports event-wise metrics separately — two honest views instead of one inflated one.

**Interview checkpoint (§3–4)**
- *Event recall is high but point recall is low — what does that mean?* You touch most incidents but flag only a small part of each (common with smoothed scores on long anomalies).
- *Why is point-adjust misleading?* It credits a whole segment for one hit, rewarding sparse, lucky detections in long segments.
- *What does latency add?* How early the alarm comes — critical for operations, invisible in F1.

---

## 5. Aggregating over channels: micro averaging

**Concept.** `evaluate_run` sums TP/FP/TN/FN (and event counts, latencies) over channels **first**, then computes metrics (`_sum` then `point_metrics`). That is *micro* averaging: every timestep (or event) counts equally. *Macro* averaging would compute F1 per channel and average, so every channel counts equally.

**Example.** Channel A: `{tp: 9, fp: 0, tn: 90, fn: 1}` (F1 0.947). Channel B (one anomalous point, missed, one FP): `{tp: 0, fp: 1, tn: 8, fn: 1}` (F1 0).
Micro F1 = 0.857; macro F1 = 0.474. Neither is "right": micro reflects total workload, macro reveals channels where the method fails. `per_channel.csv` lets you compute macro views; spacecraft-level micro results are in `metrics.json` under `SMAP` / `MSL`.

**In the code.**
```python
r = {"overall": {**point_metrics(_sum(counts[m]["point"], P_KEYS)),
                 **event_metrics(_sum(counts[m]["event"], E_KEYS)), ...
```

---

## 6. ROC-AUC and PR-AUC on normalised scores

**Problem.** F1 judges one threshold. We also want to know how well the raw scores *rank* anomalous above normal points.
**Concept.** ROC-AUC = probability that a random anomalous point scores higher than a random normal one (0.5 = random). PR-AUC (`average_precision_score`) summarises the precision–recall curve; its random baseline equals the anomaly rate, so it is more informative under imbalance.

**Why normalise.** Each channel's scores live on its own scale (and the three detectors' scores are not comparable either). Pooling raw scores lets loud channels dominate the ranking. `evaluate_run` divides by the channel threshold: $\tilde s_t = s_t/\text{thr}_c$, so 1.0 means "at that channel's decision boundary" everywhere. Warm-up NaNs become 0.

**Example.** Channel A scores `[1, 2, 3, 6]`, thr 4; channel B `[10, 20, 30, 60]`, thr 40; the last point of each is anomalous.
Each channel alone ranks perfectly (AUC 1.0). `threshold_free` on pooled **raw** scores: ROC-AUC **0.75**, PR-AUC 0.70 (B's normal points outrank A's anomaly). On pooled **normalised** scores: ROC-AUC 1.0, PR-AUC 1.0.

**Limitations.** Normalised AUC depends on the calibrated thresholds; channels with an `eps` threshold (constant validation scores) produce huge normalised scores. AUC also counts points, inheriting the long-anomaly weighting of §1.

**Interview checkpoint (§5–6)**
- *Why is PR-AUC preferred over ROC-AUC with rare positives?* ROC's FPR axis is diluted by many TNs; PR focuses on alarms.
- *Why divide by the threshold, not z-score each channel's test scores?* Z-scoring with test statistics would use test data; the threshold comes from validation only.
- *Micro vs macro — which hides a failing channel?* Micro, if the channel is small.

---

## 7. Thresholds, and the post-hoc sensitivity analysis

**Concept.** Thresholds are calibrated per channel on the validation part of the **train** file (`calibrate_threshold`, 0.995-quantile by default). Test labels are joined to the score table only after all models have predicted (`src/pipeline.py`), and `tests/test_integration.py::test_thresholds_do_not_depend_on_test_labels` checks that changing test labels leaves thresholds unchanged.
Example: validation scores $0.01, 0.02, \dots, 2.00$ ⇒ `calibrate_threshold` → 1.99005 (= `np.quantile(v, 0.995)`).

`evaluate_run` also recomputes metrics with every threshold multiplied by $k\in\{0.5,0.75,1,1.5,2,3,5\}$ (`SENSITIVITY_MULTIPLIERS` → `sensitivity.csv`, `figures/threshold_sensitivity.png`).
**Why it must not pick the threshold.** It uses test labels. Choosing the best $k$ would tune on the test set, making the reported numbers optimistic and not repeatable on new data. It is a diagnostic: "how fragile is our precision/recall trade-off?"
**Alternative (legitimate).** Tune $k$ or $q$ on a separate labelled validation set or with cross-validation over channels — never on the test labels you report.

---

## 8. Per-class recall and a common evaluation set

- **Per-class event recall.** Labels carry a class (`point` or `contextual`). `events.csv` has one row per (channel, method, event); `event_recall_by_class` in `metrics.json` = mean of `detected` by method and class. Merged P-2 intervals keep their first class (`merge_intervals`).
- **Same channel set.** `common = [c for c ... if all(... status == "ok" for m in methods)]`: a channel enters the comparison only if **all** methods succeeded on it; excluded channels are logged and listed in `metrics.json`. Otherwise a method that crashed on hard channels would look better.
- **By spacecraft.** SMAP and MSL are aggregated separately as well (different feature counts and behaviour).

**Interview checkpoint (§7–8)**
- *Your colleague picks k=1.5 because sensitivity.csv shows the best F1 there. Problem?* Test-set tuning; results no longer estimate performance on unseen data.
- *Why require all methods to succeed on a channel?* Comparable denominators; avoids survivorship bias.
- *Why break recall down by class?* Point spikes and contextual anomalies need different detectors; a single recall hides that.

---

## Exercises

1. Recreate the running example in Python with `point_counts`, `point_metrics`, `event_counts`, `event_metrics` and confirm every number above. Then set `y_pred[15] = 1` and predict the change before running it.
2. After `python -m src.cli all --config configs/smoke.yaml`, open `reports/runs/smoke/per_channel.csv` and compute macro F1 per method; compare with the micro F1 in `comparison.csv`.
3. Implement point-adjust in a scratch script (not in `src/`) and apply it to `reports/runs/smoke/scores/P-1.parquet`. How much does point F1 rise for each method?
4. Using `false_alarms.csv`, count how many FP events are shorter than 5 points. What would `postprocess.min_event_length: 5` do, and why must that choice be justified without test labels?
5. Write a short paragraph explaining why `threshold_free` sets NaN warm-up scores to 0, and whether that choice favours any method.
