# NASA Spacecraft Telemetry Anomaly Detection Using Machine Learning and LSTM Networks

Unsupervised anomaly detection on the public **NASA SMAP & MSL telemetry benchmark** (Hundman et al., KDD 2018,
"Telemanom"). The project compares three detectors (a rolling z-score baseline, Isolation Forest and an LSTM
forecaster) under one leakage-controlled pipeline. It includes point- and event-level evaluation, error analysis
and an interactive Streamlit dashboard.

> **Scope:** historical benchmark analysis. It does not monitor real spacecraft or run in NASA systems.
> An "anomaly" is a statistically unusual observation, not a confirmed fault.

## Results (full run: 81 channels, thresholds calibrated without labels)

| Method | Point precision | Point recall | Point F1 | Event recall | Event F1 | False-alarm events | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.301 | 0.195 | 0.237 | 0.827 (86/104) | 0.056 | 2,881 | 0.603 | 0.221 |
| Isolation Forest | 0.268 | 0.299 | 0.283 | 0.692 (72/104) | 0.065 | 2,025 | 0.590 | 0.183 |
| **LSTM forecaster** | 0.269 | **0.688** | **0.387** | **0.933 (97/104)** | **0.099** | **1,764** | **0.753** | **0.317** |

The LSTM ranks anomalies best and catches 97 of 104 labelled events (median latency 16 steps). Its alarms
are imprecise, though, like every method's here: test data drifts away from the short training histories,
so thresholds that flag 0.5% of validation points flag far more on test. Details, per-spacecraft numbers,
error analysis and a post-hoc threshold-sensitivity study: [`EXPERIMENTS.md`](EXPERIMENTS.md).
Single seed and no hyper-parameter search. **Not a replication of the Telemanom paper.**

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.cli download                              # ~170 MB, SHA-256 verified, idempotent
python -m src.cli validate                              # -> reports/data_validation.json
python -m src.cli all --config configs/smoke.yaml       # ~30 s end-to-end check (3 channels, tiny model)
python -m src.cli all --config configs/default.yaml     # full experiment (~1 h on CPU; --resume after interruption)
streamlit run app/dashboard.py                          # dashboard at http://localhost:8501
pytest -q                                               # tests
```

Inference on new data with a saved model (CSV with a `value` column, optional `cmd_0..` columns):

```bash
python -m src.cli predict --run full --method lstm --channel P-1 --input my_telemetry.csv --output scored.csv
```

## How it works

```
download → validate → EDA → chronological split of TRAIN (80% fit / 20% validation) → train-only scaling
→ windows / features → {rolling z-score | Isolation Forest | LSTM forecaster} → anomaly scores
→ per-channel threshold = 99.5% quantile of VALIDATION scores → predictions on TEST
→ point + event metrics, error analysis → dashboard / saved models → inference
```

| Method | Score at timestep t |
|---|---|
| Rolling z-score | distance of x_t from the mean of the previous 100 points, in rolling standard deviations |
| Isolation Forest | how easily random trees isolate the trailing-window feature vector at t |
| LSTM | smoothed absolute error of a one-step-ahead forecast made from the previous 100 timesteps (value + commands) |

Test labels are attached only after every model has predicted. A test changes the labels and checks that
no threshold moves.

## Repository layout

```
src/data/            download (pinned + verified), loader, validation
src/preprocessing/   chronological split, train-only scaler
src/features/        causal windows and Isolation Forest features
src/models/          detectors, LSTM, thresholds, persistence
src/evaluation/      metrics (point/event/AUC), evaluation + error analysis + figures
src/inference/       input validation + scoring with saved models
src/pipeline.py      per-channel training/scoring orchestration
src/cli.py           command-line entry point
app/dashboard.py     Streamlit + Plotly dashboard
configs/             default.yaml (full), smoke.yaml (quick)
notebooks/           exploratory_analysis.ipynb (executed)
reports/             data_validation.json, figures/, runs/<run>/ (metrics, tables, figures)
models/              saved per-channel detectors (generated)
tests/               unit, integration, dashboard tests
docs/                architecture, methodology, dataset, limitations, learning/ (9 teaching notes)
```

## Status

| Component | State |
|---|---|
| Download (pinned, SHA-256 verified), validation, EDA notebook | Implemented, run on real data |
| Preprocessing, three detectors, thresholds, evaluation, error analysis | Implemented, full run on 81 channels |
| Persistence + inference (`predict` CLI, dashboard upload) | Implemented, round-trip tested |
| Streamlit dashboard (5 pages) | Implemented, render-tested headless and checked in a browser |
| Resume after interruption (`--resume`) | Implemented, tested |
| Tests | Unit + integration + dashboard (`pytest -q`) |
| Multiple seeds / hyper-parameter search / cross-channel models | **Not done** |
| Nonparametric dynamic thresholding (Telemanom) | **Not implemented** |
| GPU training, real-time streaming ingestion | **Not implemented** (offline CPU only) |

## Files to study first (suggested reading order)

1. `docs/dataset.md`, then `notebooks/exploratory_analysis.ipynb`: what the data is and why it is tricky.
2. `src/data/loader.py`: file format, inclusive label intervals, the P-2 merge.
3. `src/preprocessing/scaling.py` and `src/features/windows.py`: leakage control and the alignment convention.
4. `src/models/detectors.py`: the shared interface; the baseline and Isolation Forest.
5. `src/models/lstm.py`: forecaster, training loop, early stopping, error → score.
6. `src/models/thresholds.py`: from scores to decisions.
7. `src/pipeline.py`: how the pieces are ordered (labels last).
8. `src/evaluation/metrics.py`, then `evaluate.py`: point vs event metrics, error analysis.
9. `src/models/persistence.py`, then `src/inference/predict.py`: saving, loading, safe inference.
10. `app/dashboard.py`; then `tests/` to see each guarantee pinned down.

Teaching notes: `docs/learning/` · decisions and mistakes: `LEARNING_LOG.md` · results: `EXPERIMENTS.md`.

## References

* K. Hundman et al. *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
* Telemanom: https://github.com/khundman/telemanom · Data mirror: https://huggingface.co/datasets/lxr11111/telemanom
* F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation Forest.* ICDM 2008.
* S. Hochreiter, J. Schmidhuber. *Long Short-Term Memory.* Neural Computation 1997.

Code: MIT (see `LICENSE`). Data: Caltech, BSD-3-clause-style terms, downloaded separately.
