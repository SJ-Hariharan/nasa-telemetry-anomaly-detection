# Experiments

Every number here was produced by the code in this repository on the real benchmark data. Nothing is
estimated or copied from papers. Raw outputs: `reports/runs/<run>/`.

## Run log

| Run | Config | Channels | Purpose | Status |
|---|---|---|---|---|
| `smoke` | `configs/smoke.yaml` | 3 (P-1, T-1, M-6) | End-to-end check, tiny LSTM (2 epochs) | Done, ~30 s, run twice: bit-identical |
| P-1 convergence check | 30 epochs / batch 64 vs ≤ 60 epochs / batch 128 / lr 2e-3 | 1 | Was the LSTM undertrained? | First setting still improving at its last epoch (val MSE 0.155). Second converged (0.142) and became the default |
| `full` (attempt 1) | default | 81 | First full run | **Discarded.** Stopped after the internal review found the LSTM validation-scoring mismatch (see LEARNING_LOG §11) |
| `full` (attempt 2) | default | 56/81 | Corrected code | Interrupted when the agent session ended (no error). Outputs deleted. Resume support added |
| **`full`** (final) | `configs/default.yaml`, seed 42 | **81** | Reported results | **Done.** Training took 3,698 s (61.6 min, CPU), with 0 failures. Evaluation took ~15 s |

The final run used one seed and one configuration. Hyper-parameters were not searched. Thresholds were
fixed from validation data before any test metric was computed.

## Setup (final run)

* 81 channels: 54 SMAP + 27 MSL. T-10 is excluded (it has no labels), and P-2's two label rows are merged. There are 104 labelled events (61 "point", 43 "contextual").
* Per channel: chronological 80/20 split of the train file, a scaler fitted on the first 80%, and one model of each type.
  The threshold is the 99.5% quantile of that channel's validation scores.
* Rolling z-score: W = 100 past points. Isolation Forest: 200 trees on 17 trailing features (10 lags, rolling stats over 50).
  LSTM: W = 100, 2×64, dropout 0.3, Adam lr 2e-3, batch 128, ≤ 60 epochs, patience 6, ≤ 4,000 strided training
  windows, value + command inputs, score = EWMA(span 30) of |forecast error|.
* LSTM convergence: the median best epoch was 14. Only 2 of 81 channels were still improving near the 60-epoch cap.

## Results: test set, micro-aggregated over 81 channels

| Method | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Events TP / FN / FP | ROC-AUC | PR-AUC | Median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.301 | 0.195 | 0.237 | 0.065 | 0.029 | 0.827 | 0.056 | 86 / 18 / 2,881 | 0.603 | 0.221 | 37 steps |
| Isolation Forest | 0.268 | 0.299 | 0.283 | 0.117 | 0.034 | 0.692 | 0.065 | 72 / 32 / 2,025 | 0.590 | 0.183 | 27 steps |
| **LSTM forecaster** | 0.269 | **0.688** | **0.387** | 0.267 | **0.052** | **0.933** | **0.099** | 97 / 7 / 1,764 | **0.753** | **0.317** | **16 steps** |

Base rate: 12.5% of test points are labelled anomalous. A detector that flags everything would get point
precision 0.125, so all three methods beat it, but not by much on precision.

Macro-average point F1 (mean over channels): baseline 0.145, IF 0.166, LSTM 0.339.

### By spacecraft

| Method | SMAP F1 | SMAP event recall | SMAP event FP | MSL F1 | MSL event recall | MSL event FP |
|---|---|---|---|---|---|---|
| Rolling z-score | 0.253 | 0.868 | 1,883 | 0.156 | 0.750 | 998 |
| Isolation Forest | 0.312 | 0.662 | 1,659 | 0.166 | 0.750 | 366 |
| LSTM | 0.407 | 0.956 | 1,605 | 0.242 | 0.889 | 159 |

### Event recall by anomaly class

| Class | Rolling z-score | Isolation Forest | LSTM |
|---|---|---|---|
| point (61) | 0.918 | 0.705 | 0.984 |
| contextual (43) | 0.698 | 0.674 | 0.860 |

## Interpretation

1. **The LSTM is the strongest detector on every threshold-free and recall metric.** It has ROC-AUC 0.753 vs ~0.60,
   PR-AUC 0.317 vs ≤ 0.221, detects 97/104 events, and detects them earliest. Learning the channel's dynamics *and* its
   response to commands ranks anomalous timesteps better than a local z-score or a static feature vector.
   It misses 7 events, 6 of them contextual (e.g. M-1's 1,141-step event).
2. **Precision is the main weakness, for every method.** Thresholds flag ~0.5% of validation points by
   construction: the median validation flag rate is 0.52–0.54%. On test, though, the **median channel FPR is
   0.7% (baseline), 1.7% (IF) and 6.3% (LSTM), while the mean is 7.8% / 13.6% / 25.2%.** A minority of
   channels behave differently in test than in their short training history: 29 channels have LSTM FPR > 20%.
   This is distribution shift, and it produces thousands of short false-alarm events. Those FP events are
   concentrated: the 10 worst channels account for 60–78% of each method's false alarms (baseline: P-14 alone has 791).
3. **Event recall is lenient; event precision is not.** The event metrics follow the "any overlap" rule. A detector
   that fires often gets high event recall cheaply, which is why the baseline has 0.827 event recall but 0.029 event precision.
   Read event recall together with event FP and latency.
4. **Isolation Forest is not better than the simple baseline here.** It has higher point F1 (0.283 vs 0.237) but
   lower event recall and a lower AUC. Its scores are also fragmented: 48% of its false-alarm runs are a single timestep.
5. **Point F1 and event F1 rank methods the same way here (LSTM > IF > baseline) but differ by 4×.**
   That gap is why both are reported.

### Post-hoc threshold sensitivity (diagnostic only, NOT used for any choice)

Each calibrated threshold was multiplied by k ∈ {0.5 … 5} and test metrics recomputed (`sensitivity.csv`).
For the LSTM, raising every threshold 3× would raise event F1 from 0.099 to 0.225 and point F1 from 0.387
to 0.430, at the cost of event recall (0.933 → 0.606). This shows that **the 99.5% quantile is too low for
this benchmark's test distribution.** Choosing k from this table would mean tuning on test labels, so the
reported results stay at k = 1. A legitimate fix would choose the quantile or k on held-out *channels* (see limitations).
Isolation Forest is excluded from this analysis: its scores lie in a narrow band around 0.5, so x0.5 flags
every point (one giant "event" that overlaps every label: a meaningless event F1 of 0.90) and x1.5 flags none.

### Error analysis

* **False alarms:** `false_alarms.csv` lists every predicted run with no overlap. LSTM false-alarm runs are longer
  (15% single-point vs 55% baseline / 48% IF) because of EWMA smoothing. They cluster on channels whose test
  regimes are absent from training (F-1, E-9, T-1, E-8, E-2).
* **Misses:** `events.csv` gives every labelled event with detected / latency / peak score per method. The LSTM's 7 misses are
  G-1, D-11, F-3, M-1, M-4 and T-13 (×2). They are mostly long contextual events where the signal stays in a familiar range.
* **Timelines:** `reports/runs/full/figures/timeline_*.png` and the dashboard's Anomaly analysis page show
  scores, thresholds, predictions and labels together for any channel.

## Comparison with the literature (qualitative only)

Hundman et al. (2018) report much higher event-level precision on this benchmark. Their pipeline differs:
nonparametric dynamic thresholds, anomaly pruning, 250-step windows, 10-step forecasts, larger models, and a
different event-scoring protocol. **These results are not a replication and should not be compared number-for-number.**
The ranking (LSTM forecaster > simple baselines on recall/AUC) is consistent with the paper's message.

## Reproduce

```bash
python -m src.cli all --config configs/default.yaml           # ~62 min CPU; add --resume after an interruption
python -m src.cli evaluate --config configs/default.yaml      # re-evaluate from saved scores (~15 s)
```
