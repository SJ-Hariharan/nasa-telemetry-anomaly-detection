# Architecture

## Pipeline

```
python -m src.cli download     src/data/download.py      pinned mirror -> data/raw/telemanom (SHA-256 verified)
python -m src.cli validate     src/data/validation.py    -> reports/data_validation.json
notebooks/exploratory_analysis.ipynb                     -> reports/figures/eda_*.png
python -m src.cli train        src/pipeline.py           per channel:
   load (src/data/loader.py)
   chronological fit/validation split of TRAIN + ValueScaler (src/preprocessing/scaling.py)
   features / windows (src/features/windows.py)
   detectors (src/models/detectors.py, src/models/lstm.py)  fit on fit part
   threshold calibration on validation scores (src/models/thresholds.py)
   score + threshold TEST                                    -> reports/runs/<run>/scores/<chan>.parquet
   save bundles (src/models/persistence.py)                  -> models/<run>/<method>/<chan>/
python -m src.cli evaluate     src/evaluation/evaluate.py + metrics.py
                                                          -> metrics.json, comparison.csv/.md, per_channel.csv,
                                                             events.csv, false_alarms.csv, sensitivity.csv, figures/
streamlit run app/dashboard.py reads only the files above (+ models for inference)
python -m src.cli predict      src/inference/predict.py  saved bundle + new CSV -> scores/predictions
```

## Responsibilities

| Layer | Module | Knows about |
|---|---|---|
| Acquisition | `src/data/download.py` | URLs, hashes. Nothing about ML. |
| Loading & labels | `src/data/loader.py` | File layout, label format, interval merge. |
| Validation | `src/data/validation.py` | Data assumptions; never modifies data. |
| Preprocessing | `src/preprocessing/scaling.py` | Chronological split, train-only scaling. |
| Features | `src/features/windows.py` | Causal windows / tabular features. |
| Models | `src/models/detectors.py`, `src/models/lstm.py` | `fit(train_scaled, split_idx)` / `score(array)`. No labels. |
| Decisions | `src/models/thresholds.py` | Validation-quantile threshold, post-processing. |
| Persistence | `src/models/persistence.py` | Bundle = scaler + threshold + weights + params. |
| Orchestration | `src/pipeline.py` | Order of steps; labels attached last. |
| Evaluation | `src/evaluation/` | Labels, metrics, error analysis, figures. |
| Inference | `src/inference/predict.py` | Input validation + scoring with a saved bundle. |
| Presentation | `app/dashboard.py` | Reads artifacts; no training. |

## Key design choices

* **One small interface for all detectors** (`fit`/`score`/`get_params`). Training, calibration,
  evaluation and inference never branch on the model type except in persistence. That is the only
  place where formats differ (`.pt` vs `.joblib` vs nothing).
* **Per-channel models.** Channels have different units, dynamics and command sets (25 vs 55 features).
  This follows Telemanom. The cost is 81 × 3 small models instead of one big one.
* **Labels enter late.** `src/pipeline.py` attaches the label column after every model has been fitted,
  calibrated and has predicted. `tests/test_integration.py::test_thresholds_do_not_depend_on_test_labels`
  changes the test labels and asserts that thresholds do not move.
* **Artifacts as the contract.** The dashboard and evaluation read Parquet/JSON/CSV files, so you can
  re-evaluate or redesign the UI without retraining.
* **Configuration, not code edits.** All numbers are in `configs/*.yaml`, and every run saves the
  resolved config as `config_used.json` next to its results.

## What is deliberately absent

No experiment tracker, database, web API, Docker or hyper-parameter search framework. For 81 channels
on a laptop they would add complexity without improving the science. See `docs/inference_and_deployment.md`
(learning folder) for what a production system would need.
