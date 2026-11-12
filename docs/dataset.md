# Dataset: NASA SMAP & MSL telemetry anomaly benchmark (Telemanom)

## Provenance

| Item | Value |
|---|---|
| Paper | Hundman, Constantinou, Laporte, Colwell, Söderström. *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*. KDD 2018. doi:10.1145/3219819.3219845 (arXiv:1802.04431) |
| Original code | https://github.com/khundman/telemanom (Apache-2.0) |
| Data license | Copyright Caltech, BSD-3-clause style redistribution terms (`LICENSE.txt` in the mirror) |
| Original download | `s3-us-west-2.amazonaws.com/telemanom/data.zip`. **Returns HTTP 403 as of 2026-11-11.** The current README points to Kaggle (needs an API key). |
| Source used | Hugging Face mirror `lxr11111/telemanom`, **pinned commit `d93d47d794b7ca2ca860b9157dbfbe29ad8e08d6`**, original `.npy` files under `data/data/{train,test}/` |
| Integrity | Each of the 164 `.npy` files is verified against the SHA-256 in the mirror's Git-LFS metadata (`data/raw/telemanom/MANIFEST.json`). |
| Labels | `labeled_anomalies.csv` downloaded from the canonical GitHub repo. It is **byte-identical** to the mirror copy (SHA-256 `057ce2d6…7539`). |

Why the mirror is trustworthy enough: it contains the unchanged original `.npy` files plus Parquet
conversions, and its conversion script is published. Its label file matches the original repository
exactly. Remaining risk: we cannot hash-compare the `.npy` files against the (now unavailable) S3 zip.

Reproduce: `python -m src.cli download`. This is idempotent: verified files are skipped, and `--force` re-downloads.

## Structure

* `train/<chan>.npy`, `test/<chan>.npy`: float arrays `(timesteps, features)`.
  Column 0 = telemetry value. Other columns = one-hot command indicators: 24 for SMAP (25 features), 54 for MSL (55).
* `labeled_anomalies.csv`: `chan_id, spacecraft, anomaly_sequences, class, num_values`.
  `anomaly_sequences` are inclusive `[start, end]` indices into the **test** array.
* Intended use: learn from `train` (treated as nominal, unlabelled) and detect labelled anomalies in `test`.
  We respect this split. The train file is further split *chronologically* (last 20% = validation).

## Verified facts (from `reports/data_validation.json` and the EDA notebook)

| Fact | Value |
|---|---|
| Label rows / unique channels | 82 / **81** (P-2 appears twice) |
| SMAP / MSL channels | 54 / 27 |
| Anomaly sequences | 105 published, **104** after merging P-2's overlapping duplicates |
| Train / test points | 196,321 / 509,555 |
| Test points labelled anomalous | 63,738 (12.5%) |
| Non-finite values | none |
| Test length == `num_values` | all channels |

## Quirks found and how they are handled

| Quirk | Handling |
|---|---|
| P-2 listed twice: `[[5350, 6575]]` and `[[5300, 6420]]` | Interval union → `[[5300, 6575]]` (`merge_intervals`); reported as a validation warning |
| `T-10` has train/test files but no label row | Excluded from evaluation. Missing labels ≠ "no anomalies" |
| 19 channels have a **constant** value in train | `ValueScaler` uses std = 1 for them. Any change in test looks novel (EDA §3) |
| 11 MSL channels exceed the documented [-1, 1] range in test (M-6 reaches 258) | Kept as published, documented. Contradicts the "scaled with test min/max" claim for those channels |
| Values were scaled by the authors using the **test** min/max | Unremovable leakage in the benchmark; see `docs/limitations.md` |
| Channel IDs and values are anonymised | No physical units or subsystem semantics beyond the ID letter |

The paper reports 429,735 SMAP + 66,709 MSL "telemetry values evaluated" (496,444). Our test-point count
(509,555) is a plain array-length sum over the 81 labelled channels. We reproduced the paper's numbers
exactly as `Σ (test_len − 260)` per label row: 250 history steps (`l_s`) + 10 prediction steps,
with P-2 counted twice. SMAP: 435,826 + 8,209 − 55·260 = 429,735. MSL: 73,729 − 27·260 = 66,709.
Our metrics count every test point instead (warm-up points without enough history are predicted normal).
