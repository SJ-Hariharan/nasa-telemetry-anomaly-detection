# NASA Spacecraft Telemetry Anomaly Detection Using Machine Learning and LSTM Networks

Unsupervised anomaly detection on the public **NASA SMAP & MSL telemetry benchmark** (Hundman et al., KDD 2018,
"Telemanom"). The project compares three detectors (a rolling z-score baseline, Isolation Forest and an LSTM
forecaster) under one leakage-controlled pipeline. It includes point- and event-level evaluation, error analysis
and an interactive Streamlit dashboard.

> **Scope:** historical benchmark analysis. It does not monitor real spacecraft or run in NASA systems.
> An "anomaly" is a statistically unusual observation, not a confirmed fault.

## Results (full run: 81 channels, thresholds calibrated without labels)

| Method | Point precision | Point recall | Point F1 | Event recall | Event F1 | False-alarm events | ROC-AUC (macro) | PR-AUC (macro) |
|---|---|---|---|---|---|---|---|---|
| Rolling z-score | 0.301 | 0.195 | 0.237 | 0.827 (86/104) | 0.056 | 2,881 | 0.590 | 0.260 |
| Isolation Forest | 0.268 | 0.299 | 0.283 | 0.692 (72/104) | 0.065 | 2,025 | 0.635 | 0.292 |
| **LSTM forecaster** | 0.269 | **0.688** | **0.387** | **0.933 (97/104)** | **0.099** | **1,764** | **0.677** | **0.456** |

The LSTM catches 97 of 104 labelled events (median latency 16 steps) and has the best PR-AUC. Its ROC-AUC lead
over Isolation Forest is within the bootstrap intervals. Every method raises far too many alarms. 22 channels
cannot be calibrated at all (constant training signal, threshold ≈ 0), and they account for 52% (baseline) and
44% (LSTM) of the point false positives. On the 59 calibratable channels, LSTM point F1 is 0.425. Validation-to-test score shift (KS statistic)
explains part of the rest. Details, both tables, error analysis and a post-hoc sensitivity study: [`EXPERIMENTS.md`](EXPERIMENTS.md).
No hyper-parameter search. **Not a replication of the Telemanom paper.**

**Calibration studies** (on saved scores, no retraining; `reports/runs/full/calibration/`):
* *Threshold uncertainty.* Every threshold comes from a small validation set (fewer than 5 expected exceedances
  per channel). A block bootstrap of those sets moves the aggregate results by about 0.01–0.05 in F1 (LSTM F1
  [0.375, 0.389]). That is real but small next to the precision gap. The ranges are lower bounds (see EXPERIMENTS).
* *Leave-channels-out calibration.* A threshold rule chosen on other channels (5 folds, pre-specified rule
  family and objective) and applied to held-out channels cuts LSTM false-alarm events from 1,764 to 332 and lifts
  event F1 from 0.099 to 0.231 across all 81 channels. Event recall falls from 0.933 to 0.548. On the 59 calibratable
  channels the gain is smaller (0.077 → 0.150), and the LSTM is not always the best method there. The chosen rule
  sits at the edge of the grid. This is an estimate for new channels, not a re-tuned headline number.

**Seed variance** (seeds 42, 1, 2; `reports/seed_summary.json`). LSTM point F1 is 0.390 ± 0.009 and macro
ROC-AUC 0.682 ± 0.007, so point-level results are stable. Event-level results are not: false-alarm events range
1,764–2,897 and event F1 0.063–0.099. The seed-42 run in the table above is the most favourable of the three on
event F1 and false alarms. The leave-channels-out improvement holds on every seed.

## Quick start

Three paths, from fastest to most complete. You need Python 3.11 and about 2 GB of disk for the dependencies.

### 1. First run: offline demo on simulated data (~2 minutes including install, no NASA data, no network after install)

```bash
git clone https://github.com/SJ-Hariharan/nasa-telemetry-anomaly-detection.git
cd nasa-telemetry-anomaly-detection
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt for the exact versions used for the reported runs

python -m src.cli demo                 # ~10-30 s: synthetic data -> validate -> train -> evaluate -> inference
streamlit run app/dashboard.py         # opens http://localhost:8501
```

In the dashboard, choose **`demo (SIMULATED demo)`** in the sidebar and open each page: telemetry with injected
anomalies, scores and thresholds, method comparison, and inference on a new sample (*Demo sample input*). From the
command line:

```bash
python -m src.cli predict --run demo --method lstm --channel Z-1 --input reports/runs/demo/sample_input.csv
```

> **The demo is not evidence about NASA data.** It generates 4 synthetic channels with 8 hand-made, deliberately
> easy anomalies, then runs the project's *real* detectors, thresholds, metrics and inference code on them with
> small settings (`configs/demo.yaml`). It shows how the pipeline works, nothing more. Every demo page carries a red
> *SIMULATED DEMO — NOT NASA BENCHMARK RESULTS* banner. Demo outputs go only to `data/demo/`,
> `reports/runs/demo/` and `models/demo/` (all gitignored). A rerun replaces them and never touches the NASA runs.

### 2. Read the real NASA results (no extra work)

The repository already contains the summary reports of the real runs (`full`, `full_seed1`, `full_seed2`). In the
dashboard, choose `full` for the **Overview** and **Model comparison** pages; the numbers and their caveats are in
[`EXPERIMENTS.md`](EXPERIMENTS.md). The per-channel pages and inference for these runs need path 3.

### 3. Reproduce the NASA benchmark (~170 MB download, ~1 hour on a laptop CPU)

```bash
python -m src.cli download                              # ~170 MB, SHA-256 verified, idempotent
python -m src.cli validate                              # -> reports/data_validation.json
python -m src.cli all --config configs/smoke.yaml       # ~30 s end-to-end check on NASA data (3 channels, tiny model)
python -m src.cli all --config configs/default.yaml     # full experiment (~1 h on CPU; add --resume after an interruption)
python -m src.cli calibration --run full                # threshold uncertainty + leave-channels-out (~15 s)
pytest -q                                               # tests (~1-2 min, offline)
```

Optional seed-variance runs (~1–2 h each on CPU, resumable, written to new run directories):

```bash
python -m src.cli all --config configs/seed1.yaml --resume
python -m src.cli all --config configs/seed2.yaml --resume
python -m src.cli seeds --runs full full_seed1 full_seed2   # -> reports/seed_summary.json
```

**Git hooks (one-time setup, recommended for contributors):**

```bash
git config core.hooksPath .githooks
```

* `pre-commit` (instant) rejects whitespace errors, likely secrets (`.env`, keys), data and model files
  (`data/raw/`, `*.npy`, `*.pt`, `*.joblib`, generated scores) and files over 5 MB. It also checks that staged Python
  files compile.
* `pre-push` (~20–60 s) runs the full test suite and aborts the push if anything fails.
* Plain shell scripts, no framework. Bypass deliberately with `--no-verify`.

**What a fresh clone contains.** The repository includes every NASA run's summary reports (metrics, tables,
figures, calibration studies). It does not include the raw data, the per-channel score tables or the saved models,
which are generated (~170 MB, ~18 MB and ~186 MB). Without them the dashboard's Overview and Model comparison pages
work for the NASA runs; the Telemetry explorer, Anomaly analysis and Inference pages say what is missing and give the
exact command (the demo, or `download` + training). The dashboard's **Start here** page lists what each run has.

Inference on new data with a saved model (CSV with a `value` column, optional `cmd_0..` columns):

```bash
python -m src.cli predict --run full --method lstm --channel P-1 --input my_telemetry.csv --output scored.csv
```

Inference reuses the saved scaler and threshold (never refits). It rejects malformed input with a message and exit
code 2: a missing `value` column, non-finite values, non-0/1 commands, partial command columns, too few rows,
more than 500,000 rows, or a path-like channel or run name. It warns when commands are missing, when values fall
outside the benchmark range, and when the channel's threshold is not a real calibration (constant training signal).
Saved Isolation Forest models are joblib (pickle) files: only load `models/` directories this project produced.

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
src/evaluation/      metrics (point/event/AUC), evaluation + error analysis + figures,
                     calibration studies (threshold bootstrap, leave-channels-out, seed summary)
src/inference/       input validation + scoring with saved models
src/pipeline.py      per-channel training/scoring orchestration
src/cli.py           command-line entry point
src/demo.py          offline demo: synthetic telemetry generator + end-to-end run (SIMULATED, not NASA data)
app/dashboard.py     Streamlit + Plotly dashboard
configs/             default.yaml (full), smoke.yaml (quick), seed1.yaml / seed2.yaml (seed variance),
                     demo.yaml (offline demo on simulated data)
notebooks/           exploratory_analysis.ipynb (executed)
reports/             data_validation.json, figures/, runs/<run>/ (metrics, tables, figures)
models/              saved per-channel detectors (generated)
tests/               unit, integration, dashboard tests
.githooks/           pre-commit (fast hygiene checks) and pre-push (test suite) hooks
docs/                architecture, methodology, dataset, limitations, learning/ (9 teaching notes)
```

## Status

| Component | State |
|---|---|
| Download (pinned, SHA-256 verified), validation, EDA notebook | Implemented, run on real data |
| Preprocessing, three detectors, thresholds, evaluation, error analysis | Implemented, full run on 81 channels |
| Persistence + inference (`predict` CLI, dashboard upload) | Implemented, round-trip tested |
| Streamlit dashboard (Start here + 5 pages) | Implemented, render-tested headless and checked in a browser |
| Offline first-run demo on simulated data (`python -m src.cli demo`) | Implemented; tested offline in a clean copy (deterministic, labelled SIMULATED, real runs untouched) |
| Resume after interruption (`--resume`) | Implemented, tested |
| Tests | Unit + integration + dashboard (`pytest -q`) |
| Calibration status, macro AUC with CIs, validation-vs-test shift (KS) | Implemented, tested (review fixes D1–D3, D6) |
| Threshold-uncertainty bootstrap, leave-channels-out calibration | Implemented on saved scores, tested |
| Seed variance (seeds 42, 1, 2) | Done: 3 complete 81-channel runs, `python -m src.cli seeds` |
| Hyper-parameter search / cross-channel models | **Not done** |
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
