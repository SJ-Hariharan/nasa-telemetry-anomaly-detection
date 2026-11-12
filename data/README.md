# data/

* `raw/telemanom/`: the benchmark exactly as downloaded (never modified):
  `train/*.npy`, `test/*.npy`, `labeled_anomalies.csv`, `MANIFEST.json` (source, pinned revision, SHA-256 per file).
* `processed/`: reserved for derived data. The pipeline currently keeps derived arrays in memory
  (scaling and windowing take milliseconds) and writes per-run outputs to `reports/runs/<run>/`
  instead, so this folder is normally empty.

Neither folder is committed to git. Recreate with:

```bash
python -m src.cli download   # ~170 MB, verified, skips files already present
python -m src.cli validate   # writes reports/data_validation.json
```

Provenance, license and quirks: see `docs/dataset.md`.
