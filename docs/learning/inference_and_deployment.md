# Inference, persistence, reproducibility and testing — and why this is not a production system

Code referenced: `src/models/persistence.py` (`DetectorBundle`, `load_bundle`), `src/inference/predict.py`
(`frame_to_array`, `detect`, `warmup_length`, `detect_from_csv`, `MAX_ROWS`), `src/cli.py` (`main`),
`app/dashboard.py`, `src/utils/reproducibility.py`, `src/data/download.py`, `tests/`.
Examples verified with `.venv/bin/python`. Model-quality numbers: see `EXPERIMENTS.md`.

---

## 1. Decoupling training from inference

**Problem.** Training 81 channels × 3 methods takes a long time; scoring new data should take seconds and must not retrain anything.
**Why.** If inference re-fits anything (scaler, threshold, model) on incoming data, the decision boundary silently moves — an anomaly-filled input would "normalise" itself.

**Concept.** Training (`src/pipeline.py::process_channel`) produces an artifact per (method, channel); inference (`src/inference/predict.py`) only *reads* it. The contract is the tiny detector interface in `src/models/detectors.py`: `fit(train_scaled, split_idx)` at training time, `score(array_scaled)` at inference time.

**Example.** A saved bundle with scaler mean 0.2, std 0.05 transforms an incoming value 0.3 to $(0.3-0.2)/0.05=2.0$ — whatever the incoming file's own mean is. Command columns pass through unchanged (`ValueScaler.transform` only touches column 0).

**Interview checkpoint**
- *Why not refit the scaler on the new file?* A file that is mostly anomalous would be re-centred and its anomaly hidden; also scores would not be comparable to the calibrated threshold.
- *What is the minimum the inference side must know?* Scaler stats, threshold, model weights/params, feature count, post-processing rules — exactly `meta.json` + weights.

---

## 2. The model bundle (`src/models/persistence.py`)

```
models/<run_name>/<method>/<chan_id>/
    meta.json     method, channel, scaler {mean, std}, threshold, n_features, params, postprocess
                  (+ best_epoch, history for lstm)
    model.pt      LSTM weights (state_dict only)          [lstm]
    model.joblib  fitted IsolationForest                  [isolation_forest]
```
The baseline has no weights: its "model" is its parameters (`window`, `min_std`) in `meta.json`.

**Why `state_dict` + `weights_only=True`.** A `state_dict` is just named tensors; the architecture is rebuilt from code (`LSTMForecaster(n_in, hidden_size, num_layers, dropout)`) and `meta.json` params:
```python
det.model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
det.model.eval()
```
`weights_only=True` restricts unpickling to tensors and primitive containers, so a tampered `model.pt` cannot execute code. `map_location="cpu"` lets a GPU-trained model load anywhere.

**The pickle caveat.** `joblib.load(path / "model.joblib")` *is* pickle: loading a malicious file runs arbitrary code. Rule from the module docstring: only load model directories this project produced. Safer alternatives: ONNX export (`skl2onnx`) or `skops.io` with an explicit trusted-types list.

**Limitations.** No schema version in `meta.json`; if `LSTMForecaster` changes, old `model.pt` files fail to load (strict `load_state_dict`) — which is at least a loud failure. No checksum of artifacts.

**Interview checkpoint**
- *Why save `n_features`?* To rebuild the input layer (25 vs 55) and validate incoming data shape.
- *Why is `torch.save(model)` (whole module) worse than a state_dict?* It pickles class references — brittle across refactors and unsafe to load.
- *What does `weights_only=True` protect against?* Code execution via pickle `__reduce__` payloads.

---

## 3. Scoring new data (`src/inference/predict.py`)

**Validation rules in `frame_to_array(df, n_features)`:**
1. a `value` column is required, else `ValueError`;
2. at most `MAX_ROWS = 500_000` rows (protects the dashboard from huge uploads);
3. `value` must be fully numeric and finite (`pd.to_numeric(errors="coerce")` then `isfinite`);
4. command columns: either **all** of `cmd_0..cmd_{n_features-2}` or **none**; if present they must be 0/1;
5. none present ⇒ zeros + a warning (models trained with commands see a different input distribution);
6. values outside $[-1.5, 1.5]$ ⇒ warning (the benchmark data are pre-scaled to $[-1,1]$ by the dataset authors using **test** min/max — some channels, e.g. M-6, exceed that range anyway).

**Example.** `frame_to_array(DataFrame({"value": [0.1, 0.2, 2.0]}), 3)` → array of shape (3, 3) with zero command columns and **two** warnings (no `cmd_*`; values outside range). `{"value": [0.1], "cmd_0": [1]}` with `n_features=3` → `ValueError: Expected all of cmd_0..cmd_1 or none; got 1`. `{"value": ["a"]}` → `ValueError: 'value' column contains missing or non-numeric entries`.

**Warm-up.** `warmup_length` = steps that cannot be scored. With `configs/default.yaml`: baseline 100 (`window`), Isolation Forest 49 (`stat_window - 1`), LSTM 100 (`window`). `detect` rejects inputs with ≤ warm-up rows.

**`detect` itself:**
```python
scores = bundle.detector.score(bundle.scaler.transform(array))      # training scaler, reused
pred = apply_threshold(scores, bundle.threshold, bundle.postprocess.get("min_event_length", 1),
                       bundle.postprocess.get("merge_gap", 0))       # training threshold, reused
```
It returns `timestep, value, score, threshold, normalized_score, is_anomaly, scored`.
**Example** (baseline bundle, `window=100`, threshold 5.0, a 260-step small sine with a 0.9 spike at t=200): rows 0–99 have NaN score, `scored=False`, `is_anomaly=0`; row 200 has score 13.1, normalised 2.6, `is_anomaly=1`; it is the only flagged row.

**Limitation.** Each call scores a whole series from scratch; there is no streaming state (hidden state, EWMA) carried between calls, so the first `warmup` points of every request are unscored.

---

## 4. The CLI (`src/cli.py`)

```
python -m src.cli predict --run smoke --method lstm --channel P-1 --input my.csv [--output out.csv]
```
`main` builds `models_dir = resolve_path(cfg["output"]["models_dir"]) / args.run`, calls `detect_from_csv` (load bundle → `frame_to_array` → print warnings → `detect`), and prints how many rows were flagged plus the threshold; without `--output` it prints the first 20 flagged rows. Missing arguments ⇒ `parser.error(...)`. Other commands: `download`, `validate`, `train`, `evaluate`, `all` (train + evaluate).

---

## 5. The Streamlit dashboard (`app/dashboard.py`)

- **Reads only saved artifacts:** `reports/data_validation.json`, `reports/runs/<run>/` (scores parquet, `metrics.json`, CSVs), `models/<run>/` for the Inference page, and raw train files for context plots. It never trains.
- **Caching:** `@st.cache_data` on `available_runs`, `load_run`, `load_scores`, `load_train_values` — Streamlit reruns the whole script on every widget interaction, so caching avoids re-reading parquet files. Caveat: if a run is re-generated while the app is open, cached data is stale until the cache is cleared.
- **Inference page:** same `load_bundle` → `frame_to_array` → `detect` path as the CLI; errors become `st.error` messages rather than tracebacks.
- **Honest labelling:** a `DISCLAIMER` ("Historical benchmark analysis only…") on several pages; the simulated input shows `st.warning("SIMULATION: a synthetic sine wave with an injected spike and level shift. Not NASA data.")`; a smoke run gets "metrics are not meaningful"; the sensitivity chart says it is "NOT for choosing k".

---

## 6. Reproducibility

| Ingredient | Mechanism |
|---|---|
| Code | the repository (`src/`) |
| Config | `configs/*.yaml`; `inherit:` for overrides; the merged config is saved as `reports/runs/<run>/config_used.json` |
| Seeds | `set_seed(cfg["seed"])` per channel & method; `use_deterministic_algorithms(True, warn_only=True)` |
| Data | `HF_REVISION` (pinned Hugging Face mirror commit) + SHA-256 check of every file in `src/data/download.py` |
| Environment | `requirements.txt` |

**Caveat.** `requirements.txt` gives *minimum* versions (`torch>=2.2`), not exact pins, so a fresh install may differ from the environment that produced a run. For exact reproduction, record `pip freeze` alongside a run. Bit-exact PyTorch results across machines are not guaranteed (see `set_seed` docstring).

---

## 7. Testing strategy (`tests/`)

| Level | Examples | What it protects |
|---|---|---|
| Unit | `test_forecast_windows_alignment`, `test_scaler_uses_only_fit_data`, `test_point_metrics_hand_example`, `test_frame_to_array_validation`, `test_detect_rejects_too_short_input` | alignment, leakage rules, metric formulas, input validation |
| Model sanity | `test_lstm_forward_shape_and_learning` (val loss at best epoch < epoch 0), `test_baseline_is_causal_and_detects_spike` | the learning loop actually learns; causality |
| Round-trip | `test_bundle_roundtrip_gives_identical_scores` (all 3 methods; `assert_allclose(..., rtol=1e-5, equal_nan=True)`) | save → load gives the same scores |
| Integration | `test_end_to_end_pipeline` on a synthetic dataset from `tests/conftest.py` (sine + injected level shift [600, 639] + spike at 850) | train → evaluate → CSV predict reproduces pipeline scores/predictions |
| Leakage | `test_thresholds_do_not_depend_on_test_labels` | test labels never influence thresholds |
| UI | `test_dashboard.py`: `AppTest.from_file(...)` renders every page headlessly, runs inference on the simulated signal | the app doesn't crash; skipped if no evaluated run exists |

Synthetic data is used only to exercise code paths offline — never reported as a result. Run with `.venv/bin/python -m pytest -q`.

**Interview checkpoint**
- *Why a round-trip test with tolerance instead of exact equality?* It is defensive: weights load bit-exactly, but floating-point results can differ in the last bits across code paths (the integration test compares CSV-predicted scores with the pipeline's using `rtol=1e-4`).
- *Why an integration test on synthetic data rather than the real benchmark?* Fast, offline, deterministic, and anomalies are known by construction.
- *What does the dashboard test not check?* That plots are correct or readable — only that pages render without exceptions.

---

## 8. What a real production deployment would need — and why this is not one

This project analyses a **historical, labelled benchmark**. A real spacecraft-operations system would additionally need:
- **Streaming inference** with persistent per-channel state (LSTM window buffer, EWMA), handling of gaps, out-of-order and late data, and timestamps (the benchmark has only indices).
- **Monitoring**: alarm rates per channel, score distributions, latency, failures; alerting on the detector itself.
- **Drift detection**: compare incoming value/score distributions against training (e.g. PSI, KS tests); new mission phases change "normal".
- **Retraining policy**: when, on which verified-nominal data, with what review; avoid training on unlabelled anomalies.
- **Versioning**: model registry, schema-versioned `meta.json`, data/config/code hashes per model, rollback.
- **Security**: no pickle from untrusted sources, signed artifacts, access control.
- **Human factors**: operator feedback loop, alarm triage, explanation of each alarm, documented false-alarm budget.
- **Validation**: performance requirements agreed with domain experts, not benchmark F1.
None of these exist here by design; the dashboard states it "does not monitor any spacecraft".

---

## Exercises

1. Run `python -m src.cli all --config configs/smoke.yaml`, then inspect `models/smoke/lstm/P-1/meta.json`. Which fields would you need to add for a production model registry?
2. Export the P-1 test array to CSV (`value, cmd_0..cmd_23`) in a scratch folder and run `python -m src.cli predict --config configs/smoke.yaml --run smoke --method lstm --channel P-1 --input <csv> --output <out.csv>`. Verify the scores equal `lstm_score` in `reports/runs/smoke/scores/P-1.parquet`.
3. Feed the same CSV with all `cmd_*` columns dropped. What warning appears, and how do the scores change?
4. Run `.venv/bin/python -m pytest tests/test_models_and_metrics.py -k roundtrip -q`. Then explain why `load_bundle` must call `det.model.eval()`.
5. Sketch (in prose) a drift check comparing the mean/std of incoming `value` to the saved scaler stats, and decide what should happen when it fires.
