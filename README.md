# NASA SMAP/MSL Telemetry Anomaly Detection

This project runs unsupervised per-channel anomaly detectors on the public Telemanom benchmark: 81 labelled channels from the SMAP satellite and MSL Curiosity rover, with 104 labelled anomaly sequences after merging P-2's overlapping rows. It compares a rolling z-score, Isolation Forest, a reference LSTM forecaster with a validation-quantile threshold, the same saved LSTM models with Telemanom's dynamic threshold (`lstm_dynamic`, an ablation with no retraining), and a Telemanom-aligned LSTM with dynamic threshold (`telemanom`). Each method is scored with point and event metrics on the test split.

Historical, anonymised data, not a live NASA feed. The project has no NASA affiliation, is not a fault-diagnosis system, and is not an exact reproduction of the paper.

## Results

All 81 channels; thresholds and settings fixed without test labels; mean ± std over 3 seeds; full table in `reports/benchmark/benchmark.md`.

| Method | Event F1 | Events found (of 104) | False-alarm events | Event precision | Point F1 | Point recall | Median decision latency (samples) |
|---|---|---|---|---|---|---|---|
| Telemanom-aligned LSTM + dynamic threshold (`telemanom`) | 0.733 ± 0.022 | 70 ± 2 | 17 ± 3 | 0.805 ± 0.028 | 0.335 ± 0.019 | 0.234 ± 0.014 | 94 |
| Reference LSTM models + dynamic threshold (ablation) (`lstm_dynamic`) | 0.716 ± 0.028 | 64 ± 2 | 11 ± 3 | 0.851 ± 0.039 | 0.322 ± 0.009 | 0.220 ± 0.007 | 121 |
| Reference LSTM, static threshold (`lstm`) | 0.080 ± 0.018 | 97 ± 1 | 2314 ± 567 | 0.042 ± 0.010 | 0.390 ± 0.009 | 0.699 ± 0.009 | 19 |
| Isolation Forest, static threshold (`isolation_forest`) | 0.062 ± 0.003 | 72 ± 1 | 2133 ± 94 | 0.033 ± 0.002 | 0.302 ± 0.021 | 0.325 ± 0.025 | 27 |
| Rolling z-score, static threshold (`baseline`) | 0.056 ± 0.000 | 86 ± 0 | 2881 ± 0 | 0.029 ± 0.000 | 0.237 ± 0.000 | 0.195 ± 0.000 | 37 |

**Best under the protocol: the Telemanom-aligned method** (event F1 0.733 ± 0.022). Its lead over the ablation `lstm_dynamic` (0.716 ± 0.028) is smaller than the seed spread, so the improvement comes from the **dynamic threshold**, not the new network: re-scoring the *same saved* reference LSTM models with it cuts false-alarm events from 2314 to 11 (mean over seeds). The trade-off is plain: the dynamic methods find 64-70 of 104 events instead of 97, have lower point recall and point F1, and alert later (median ~94-121 samples vs 19). If missing events is costlier than false alarms, the reference LSTM's recall is the better operating point; nothing here is evidence of operational fitness.

## Quick start (no download, ~2 minutes)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt    # exact pins: requirements-lock.txt
streamlit run app/dashboard.py
```

The app works on a fresh clone from committed artifacts. It never trains or downloads anything. Saved model weights are not committed, so Inference needs a local run on a fresh clone.

* `reports/benchmark/`: cross-run comparison.
* `reports/showcase/`: telemetry, scores and predictions for 7 channels, picked by a fixed rule. Saved models for 2 channels are written locally by `showcase` (gitignored, not committed).
* `reports/runs/<run>/`: metrics, events, false alarms and calibration for each run.

Dashboard pages: Overview, Compare methods, Channel timeline, Detections & errors, Historical replay, Inference and Reproduce.

* **Historical replay** is labelled "Historical replay — not live telemetry". It feeds archived samples in order; the dynamic threshold releases decisions per 70-sample batch.
* **Inference** takes an uploaded CSV with a `value` column and optional `cmd_0..` 0/1 columns, or an archived segment.

| Symptom | Fix |
|---|---|
| `streamlit: command not found` | Activate the venv (`source .venv/bin/activate`) or run `python -m streamlit run app/dashboard.py`. |
| Port busy | `streamlit run app/dashboard.py --server.port 8502` |

`python -m src.cli demo` runs the pipeline on synthetic data in ~30 s; it is labelled SIMULATED and not NASA results.

## Reproduce

1. **Get the data, then validate it.**

   ```bash
   python -m src.cli download                            # verified downloader, ~166 MB on disk, SHA-256 checked against a pinned mirror
   python -m src.cli download --from-dir /path/to/copy   # or import an existing copy (e.g. an attached Kaggle dataset); each file is checked against src/data/checksums.json
   python -m src.cli validate
   ```

2. **Run the tests.** About 5 minutes on CPU. The notebook test is opt-in.

   ```bash
   python -m pytest -q
   RUN_NOTEBOOK=1 python -m pytest tests/test_notebook.py
   ```

3. **Smoke run.** 3 channels, tiny model, ~2 minutes. Metrics are not meaningful.

   ```bash
   python -m src.cli all --config configs/telemanom_smoke.yaml
   ```

4. **Reference run.** About 1 hour on CPU. Seeds: `configs/seed1.yaml`, `configs/seed2.yaml`.

   ```bash
   python -m src.cli all --config configs/default.yaml --resume
   ```

5. **Telemanom-aligned run.** This is the slowest step: several CPU-hours for 81 channels (measured: about 7 single-thread process-hours per seed, about 2 h wall-clock with 4 parallel shards on an 8-core Apple CPU). It can be split across processes, then collected and evaluated. Seeds: `configs/telemanom_seed1.yaml`, `configs/telemanom_seed2.yaml`.

   ```bash
   # optional: train 4 shards in parallel first (each skips channels already finished)
   for i in 0 1 2 3; do python -m src.cli train --config configs/telemanom.yaml --shard $i/4 --resume & done; wait
   # collect every channel (trains any that are missing) and evaluate
   python -m src.cli all --config configs/telemanom.yaml --resume
   ```

6. **Ablation (no training).** Needs the saved models from the reference run. Also `lstm_dynamic_seed1.yaml` and `lstm_dynamic_seed2.yaml`.

   ```bash
   python -m src.cli derive --config configs/lstm_dynamic.yaml
   ```

7. **Calibration studies, comparison and showcase.** `compare` writes `reports/benchmark/`; `showcase` writes `reports/showcase/`.

   ```bash
   python -m src.cli calibration --run full        # also --run telemanom, --run lstm_dynamic
   python -m src.cli compare
   python -m src.cli showcase
   ```

8. **Inference from saved models.**

   ```bash
   python -m src.cli predict --run telemanom --method telemanom --channel P-1 --input my.csv
   ```

New experiments always use new run names. The committed reference runs (`full`, `full_seed1`, `full_seed2`) are never overwritten.

## Colab / Kaggle

`notebooks/run_pipeline.ipynb` runs the real CLI. It finds an attached Kaggle copy (checksum-verified) or uses the downloader. `SMOKE=True` is the default; set `SMOKE=False` for a resumable full run. Outputs are zipped to `/kaggle/working`, copied to Google Drive if it is mounted, or written to `outputs/`. The notebook has been executed locally in smoke mode only; it has NOT been run on Colab or Kaggle.

## GPU

Set `device: auto` (or `cuda`) under `telemanom:` or `lstm:` in a config. If CUDA is absent, the run falls back to CPU with a warning. Scoring always runs on CPU, so results do not depend on where training ran.

GPU training has not been verified and no speed-up is claimed; the benchmark trains 81 small per-channel models, so a GPU may not help much.

## How it works

Per channel:

1. The train file is split chronologically 80/20 into fit and validation parts. The scaler is fit on the fit part only.
2. The detectors are trained on the fit part.
3. Static thresholds are the 99.5% quantile of validation scores.
4. The dynamic threshold is Telemanom's label-free nonparametric threshold with pruning (p = 0.13), computed on the series being judged.
5. Test labels are joined only for evaluation.

Metrics: micro point precision, recall and F1, plus event metrics. An event is found if any flagged point overlaps it. A false-alarm event is a flagged run that overlaps no label. No point-adjust is applied.

See [`docs/methodology.md`](docs/methodology.md), [`docs/limitations.md`](docs/limitations.md), [`docs/dataset.md`](docs/dataset.md), [`docs/architecture.md`](docs/architecture.md) and [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Limitations

* The dataset is min-max scaled by its authors using test-set statistics. All methods share this property.
* Labels are approximate: they are historical expert annotations derived from incident reports.
* Telemanom's constants were chosen by the paper's authors on this benchmark. They were not tuned here, but they may favour that method slightly.
* The event metric counts any overlap as a hit, which is lenient. Long flagged segments score well under it, so point metrics are reported alongside.
* The dynamic threshold needs up to 69 future samples (batch release) and about 2,100 samples before its first decision.
* 22 channels (19 with a constant training signal) cannot get a calibrated static threshold.
* These are offline benchmark results, not operational evidence.

## Data source, citation and licences

* **Data:** NASA SMAP and MSL telemetry released with the Telemanom benchmark. Values are anonymised and min-max
  scaled by the dataset authors (using test-set statistics, a known benchmark property; see
  [`docs/limitations.md`](docs/limitations.md)).
* **Labels:** historical expert annotations derived from incident reports (`labeled_anomalies.csv`). Boundaries are
  approximate; T-10 has no labels; P-2's duplicate rows are merged.
* **Provenance:** the original S3 archive now returns HTTP 403, so the downloader uses the Hugging Face mirror
  `lxr11111/telemanom` pinned to one commit, verifies every file's SHA-256, and cross-checks the labels against
  the canonical repository ([`docs/dataset.md`](docs/dataset.md)).
* **Cite:** K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Söderström. *Detecting Spacecraft Anomalies
  Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018 ([arXiv:1802.04431](https://arxiv.org/abs/1802.04431)).
  Code: [github.com/khundman/telemanom](https://github.com/khundman/telemanom).
* **Licences:** this project's code is MIT ([`LICENSE`](LICENSE)). The data is © Caltech under BSD-3-clause-style
  terms and is downloaded separately. The only data committed are the 7 test-file excerpts in `reports/showcase/` used by
  the offline app.

## Repository layout

```
app/                     Streamlit dashboard
src/
  cli.py                 command-line entry point
  pipeline.py            per-channel training and scoring
  replay.py              historical replay
  showcase.py            showcase export
  data/                  download, loader, validation
  models/                detectors, lstm, dynamic_threshold, decision, thresholds, persistence
  evaluation/            evaluate, metrics, calibration, compare
  inference/             input validation and scoring with saved models
configs/                 run configurations
reports/                 benchmark/, showcase/, runs/<run>/
notebooks/               exploratory_analysis.ipynb (EDA), run_pipeline.ipynb
tests/                   unit, integration and dashboard tests
docs/                    methodology, dataset, architecture, limitations
```
