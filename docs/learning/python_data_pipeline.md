# The Python data pipeline: from a URL to a per-channel score table

This document follows one byte of telemetry from the internet to the score table that the
evaluation reads. Every section points at the real code.

```
python -m src.cli download   -> src/data/download.py   -> data/raw/telemanom/{train,test}/*.npy + labels + MANIFEST.json
python -m src.cli validate   -> src/data/validation.py -> reports/data_validation.json
python -m src.cli train      -> src/pipeline.py        -> models/<run>/... + reports/runs/<run>/scores/<chan>.parquet
python -m src.cli evaluate   -> src/evaluation/evaluate.py (reads the parquet tables)
```

---

## 1. Reproducible download (`src/data/download.py`)

**Problem.** The dataset's original S3 bucket now returns HTTP 403 (see the module docstring), so we
download from a Hugging Face mirror. A mirror can change, files can arrive truncated, and the
network can fail halfway through 160+ files.

**Why it matters.** If two people "download the dataset" and get different bytes, their results
are not comparable. Silent corruption is worse than a crash.

**Concept.** Three guarantees:

1. *Pinned revision* — the URL contains a fixed commit hash, not a branch name:
   `HF_REVISION = "d93d47d794b7ca2ca860b9157dbfbe29ad8e08d6"`. A branch moves; a commit does not.
2. *Content verification* — a cryptographic hash $h = \mathrm{SHA256}(\text{bytes})$ is a 256-bit
   fingerprint. Changing a single bit changes $h$ completely, so "hash matches the published
   hash" ⇒ "bytes are identical" (with overwhelming probability).
3. *Idempotency* — running the command twice gives the same end state, and the second run is cheap.

**Worked example.** `hashlib.sha256(b"abc").hexdigest()` starts with `ba7816bf8f01cfea`. Change
`"abc"` to `"abd"` and the digest is unrecognisably different — there is no "close" hash.

**How the code does it.**

```python
# download_dataset(): skip files that are already present AND verified
if target.exists() and target.stat().st_size == entry["size"] and sha256_of(target) == entry["sha256"]:
    continue
data = _fetch(HF_FILE_URL.format(path=entry["remote_path"]))
if hashlib.sha256(data).hexdigest() != entry["sha256"]:
    raise RuntimeError(f"Checksum mismatch for {entry['remote_path']} - refusing to save it.")
```

- The expected hashes come from the mirror's Git-LFS metadata (`build_manifest`) and are written to
  `MANIFEST.json`, so later runs reuse the manifest instead of re-listing the mirror.
- The size check runs first because it is free; hashing a file costs a full read.
- The file is written only *after* verification, so a bad download never lands on disk.
- `_fetch` retries up to 4 attempts with linear backoff (`time.sleep(2 * attempt)` → 2 s, 4 s, 6 s)
  and catches `OSError` and `http.client.HTTPException` (which covers `URLError` and
  `IncompleteRead`, i.e. truncated transfers).
- Labels come from the canonical GitHub repo and are compared byte-for-byte with the mirror's copy
  (`labels_match_mirror` in the manifest).

**Alternatives.** DVC / `pooch` (hash-pinned dataset registries); Kaggle API (needs credentials);
committing the data to git (too large, licence questions).

**Limitations.** GitHub `master` is a moving target, so the label file is pinned **by content**:
`LABELS_SHA256` in `download.py`. The existing file is re-hashed on every run. If upstream changes,
the pinned mirror copy is used, and if neither matches, the download stops instead of silently using new labels.

---

## 2. Loading arrays and labels (`src/data/loader.py`)

### 2.1 The `.npy` layout

Each channel has `train/<chan>.npy` and `test/<chan>.npy` of shape `(timesteps, features)`:

| column | meaning |
|---|---|
| 0 (`VALUE_COL`) | the telemetry value (pre-scaled to $[-1,1]$ by the dataset authors) |
| 1.. | one-hot "command" flags: 24 for SMAP (25 features total), 54 for MSL (55 total) |

```python
def load_channel(raw_dir, chan_id, split):
    ...
    # allow_pickle=False: .npy files from the internet must never execute code on load.
    return np.load(path, allow_pickle=False).astype(np.float32)
```

**Why `allow_pickle=False`?** A `.npy` file may contain an *object* array serialised with
`pickle`. Unpickling can call arbitrary Python callables — loading a malicious file could run code
on your machine. Numeric arrays never need pickle, so disallowing it costs nothing and closes the hole.

### 2.2 Parsing `labeled_anomalies.csv`

The CSV stores lists as strings: `"[[5350, 6575], [5300, 6420]]"`.

```python
df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)  # safe: literals only
df["classes"] = df["class"].str.strip("[]").str.split(",").apply(lambda xs: [x.strip() for x in xs])
```

**`ast.literal_eval` vs `eval`.** `eval` executes any expression — `eval("__import__('os').system(...)")`
would run a shell command. `literal_eval` only accepts Python *literals* (numbers, strings, lists,
tuples, dicts, …). Verified: `ast.literal_eval("__import__('os').getcwd()")` raises `ValueError`.
The `class` column (`[point, contextual]`) has unquoted words, so it is not a valid literal and is
parsed with string operations instead.

### 2.3 Inclusive intervals → a label vector

`[start, end]` are **both inclusive** indices into the *test* array. `label_vector` therefore writes
`y[start : end + 1] = 1` (Python slices exclude their end).

Worked example (`tests/test_data_and_features.py::test_label_vector_is_inclusive`):
`label_vector(6, [[1, 2], [4, 4]])` → `[0, 1, 1, 0, 1, 0]`. Off-by-one here would shift every label
and silently reduce every metric. Out-of-range intervals raise `ValueError` instead of being clipped.

### 2.4 Merging P-2's duplicate rows

The published CSV lists channel P-2 twice: `[5350, 6575]` and `[5300, 6420]`. `load_labels` groups
by `chan_id` and calls `merge_intervals`, which sorts by start and unions overlapping *or touching*
intervals (`s <= merged[-1][1] + 1`):

```
sorted: [5300, 6420], [5350, 6575]  ->  5350 <= 6421, so merge  ->  [[5300, 6575]]
```

That is why the project counts **104** anomaly sequences rather than the published 105. The merged
interval keeps the class of the earlier-starting interval. `n_label_rows` records that two rows were
merged, and `validate_dataset` reports it as a warning — nothing is silently dropped.

---

## 3. Validation before modelling (`src/data/validation.py`)

**Problem.** A model trained on broken data produces confident nonsense.

`validate_dataset` separates **errors** (pipeline assumptions violated → `ok: false`, CLI exits 1)
from **warnings** (surprises worth recording). Checks include:

- errors: non-finite values, train/test feature mismatch, test length ≠ `num_values`, label
  intervals outside the test range, duplicate rows that disagree on spacecraft/length;
- warnings: duplicate label rows merged (P-2), files with no label row (T-10 has data but no labels
  → excluded), command columns not strictly 0/1, test values outside $[-1,1]$ (some MSL channels,
  M-6 reaches 258), channels constant in train (19 of them).

Note the philosophy: repeated rows are *recorded*, not removed — flat telemetry is real behaviour.

---

## 4. Configuration (`src/utils/config.py`, `configs/*.yaml`)

**Problem.** Hard-coded hyperparameters scattered across files make a run impossible to reproduce
or compare.

**Concept.** Every result-affecting number lives in `configs/default.yaml` ("reproducible from
code + config + seed"). `configs/smoke.yaml` says `inherit: default.yaml` and overrides only what
differs. `_deep_merge` merges *recursively*, so overriding `lstm.hidden_size` keeps all other `lstm`
keys:

```yaml
# smoke.yaml                          # effective lstm section
lstm:                                 # window: 100        (from default)
  hidden_size: 16                     # hidden_size: 16    (override)
  num_layers: 1                       # num_layers: 1      (override)
  epochs: 2                           # ... dropout, patience, etc. from default
```

`yaml.safe_load` is used (the YAML equivalent of `literal_eval`: no arbitrary object construction).
Relative paths are resolved against `PROJECT_ROOT` by `resolve_path`, so configs are not
machine-specific. `run_training` also writes `config_used.json` next to the results.

---

## 5. The CLI (`src/cli.py`)

One entry point with sub-commands (`download | validate | train | evaluate | all | predict`).
Heavy imports (torch via `src.pipeline`) happen *inside* the branch that needs them, so
`python -m src.cli download` does not import PyTorch. `validate` returns exit code 1 when the report
has errors, which lets shell scripts or CI stop early.

---

## 6. The training loop (`src/pipeline.py`)

`run_training` loops over channels; `process_channel` does, for one channel:

```python
split = chronological_split_index(len(train), cfg["split"]["val_fraction"])
scaler = ValueScaler().fit(train[:split])                 # statistics from fit part ONLY
train_s, test_s = scaler.transform(train), scaler.transform(test)
...
det = build_detector(method, cfg).fit(train_s, split)
val_scores = validation_scores(det, train_s, split)  # validation = last part of TRAIN, scored like test
thr = calibrate_threshold(val_scores, **cfg["threshold"])
scores = det.score(test_s)
...
# Labels are joined only now, after every model has been fitted, calibrated and has predicted.
table["label"] = label_vector(len(test), label_row.anomaly_sequences)
```

**Why per channel?** Each channel is a different sensor with its own scale and dynamics; the
benchmark (and Telemanom) models channels independently. It also keeps memory bounded.

**Why join labels last?** It makes it structurally obvious that nothing above that line can read
test labels. `tests/test_integration.py::test_thresholds_do_not_depend_on_test_labels` checks it.

**Failure handling.** A failing method on one channel is logged and recorded as
`{"status": "failed"}` in `training_summary.json`'s `failures` list — the run continues but the
failure is never hidden. `set_seed(cfg["seed"])` is called per channel *and* method so results do
not depend on channel order.

**Raw vs processed separation.** `data/raw/` holds verified, untouched downloads. Everything derived
(scaled arrays, scores, models) goes to `models/` and `reports/runs/<run_name>/`. You can always
delete derived outputs and regenerate them; you can never "un-modify" raw data. The scaler is
recomputed on the fly and saved inside the model bundle (`DetectorBundle`), not baked into files.

---

## Alternatives and limitations (whole pipeline)

- Alternatives: a workflow tool (Make, DVC, Prefect) to cache stages; Parquet instead of `.npy`;
  Hydra/OmegaConf instead of a 20-line config loader.
- Limitations: the loop is sequential over channels (no parallelism); the data was min-max scaled by
  the dataset authors using **test** min/max — a benchmark leakage this pipeline cannot remove.

## Interview checkpoint

1. **Why verify a SHA-256 when the download "succeeded"?** HTTP success does not prove the bytes are
   complete or unchanged; the hash proves identity with the published file.
2. **What makes `download` idempotent?** Each file is skipped if its size and hash already match the
   manifest; a re-run downloads nothing and ends in the same state.
3. **Why `literal_eval` instead of `eval`?** It parses literals only, so a crafted CSV cannot execute code.
4. **The labels say `[4, 4]`. How many points are anomalous?** One — intervals are inclusive.
5. **Why are labels attached at the very end of `process_channel`?** So no fitting, scoring or
   thresholding step can depend on them, and the test proves it.

## Exercises

1. Run `python -m src.cli download` twice. Predict what the second run logs, then check
   ("0 downloaded, N already present and verified").
2. In a Python shell, call `merge_intervals([[1, 2], [3, 4], [6, 7]], ["a", "b", "c"])`. Predict the
   result first (hint: touching intervals merge, a gap of one point does not).
3. Create `configs/my.yaml` with `inherit: smoke.yaml` and `run_name: my`, change only
   `baseline.window`, and print `load_config("configs/my.yaml")["baseline"]`. Which keys survive?
4. Run `python -m src.cli validate` and find, in `reports/data_validation.json`, the P-2 merge
   warning and the list of constant-train channels.
5. Read `tests/test_data_and_features.py::test_duplicate_label_rows_are_merged_and_reported` and
   explain why the merged S-1 interval starts at 595.
