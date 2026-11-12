# Project context (for future sessions and reviewers)

**What:** unsupervised anomaly detection on the NASA SMAP/MSL Telemanom benchmark. It compares a rolling
z-score baseline, Isolation Forest and an LSTM forecaster, with a Streamlit dashboard. Historical analysis only.

**Status:** see the "Status" table in `README.md` (implemented / tested / not done).

## Decisions already made (don't relitigate without a reason)

| Decision | Reason | Where |
|---|---|---|
| Data from HF mirror `lxr11111/telemanom` @ `d93d47d7…`, SHA-256 verified | Original S3 is 403; Kaggle needs credentials | `src/data/download.py` |
| Labels from GitHub `khundman/telemanom` (identical to mirror) | Canonical source | same |
| P-2 duplicate rows merged by interval union; T-10 excluded (unlabelled) | Benchmark quirks found by validation | `src/data/loader.py`, `validation.py` |
| Chronological 80/20 split of the TRAIN file; test never used for fitting | Leakage control | `src/preprocessing/scaling.py` |
| z-scaling on fit part; std = 1 fallback for constant channels (19 channels) | Avoid divide-by-zero blow-ups | same |
| Per-channel models and thresholds | Heterogeneous channels; follows Telemanom | `src/pipeline.py` |
| Threshold = 99.5% quantile of validation scores | No labels; fixed a priori | `src/models/thresholds.py` |
| LSTM: W 100, 2×64, dropout 0.3, Adam 2e-3, batch 128, ≤ 60 epochs, patience 6, ≤ 4k windows | CPU budget; converged on P-1 (val MSE 0.142) | `configs/default.yaml` |
| Point + event metrics, no point-adjust, micro aggregation | Honest evaluation | `src/evaluation/` |

## Environment notes

* `.venv` (Python 3.11). Data in `data/raw/telemanom` (~170 MB). The disk was ~4 GB free during the build.
* In this sandboxed agent environment, git init, pip, downloads and Jupyter kernels needed the sandbox
  disabled (proxy SSL / truncation / port binding). A normal terminal needs nothing special.
* Full run: `python -m src.cli all --config configs/default.yaml` took 61.6 min on this laptop's CPU (LSTM ≈ 99% of it).
  Use `--resume` after an interruption: finished channels are skipped, and a changed config is refused.
  The smoke run takes ~30 s.

## Results (final full run)

LSTM: point F1 0.387, event recall 97/104, ROC-AUC 0.753. Baseline F1 0.237, IF 0.283. All methods have low
precision because of train→test drift. See `EXPERIMENTS.md`.

## Open questions / next steps

See "Remaining limitations" in `README.md` and the questions in `FABLE_REVIEW_BRIEF.md`.
