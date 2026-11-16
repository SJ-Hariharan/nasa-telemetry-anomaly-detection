# Ship plan: nasa-telemetry-anomaly-detection v1.0.0 (release candidate)

**Release-prep start (2026-11-15 22:05):** branch `master`, HEAD `e5719e1`, clean working tree, no local
`AGENTS.md`/`CLAUDE.md`, no lint configuration. Nothing has ever been pushed.

**Current state:** HEAD is still `e5719e1`. All release-prep changes are **uncommitted** in the working tree
(modified docs/code/tests, plus new `SHIP_PLAN.md`, `src/evaluation/calibration.py`, `configs/seed{1,2}.yaml`,
`reports/runs/full/calibration/`, `reports/runs/full_seed{1,2}/`). Seed runs: see "Run status" below.
Generated, gitignored artifacts touched: `models/full/*/*/meta.json` gained a `calibratable` key (weights, scaler and
threshold untouched and asserted unchanged). `reports/runs/full/` committed files are byte-identical to HEAD.

## Audit findings: verified against the code

Each was reproduced or checked against the current code, not taken from the review on trust.

| ID | Finding | Classification | Action | Evidence / regression test |
|---|---|---|---|---|
| D1 | Degenerate (~0) thresholds on constant-training channels | Defect, fixed in `e5719e1` | Kept: flagged + all/calibratable reporting | `test_degenerate_calibration_is_flagged`, integration test |
| D2 | Pooled normalised AUC misranks methods | Defect, fixed in `e5719e1` | Kept: macro AUC + bootstrap CI primary | `test_macro_auc_mean_and_bootstrap_interval` |
| D3 | "Distribution shift" asserted without evidence | Reporting defect, fixed in `e5719e1` | Kept: per-channel KS statistic | `per_channel.csv: ks_val_test` |
| D4 | 99.5% quantile from 62–862 validation points | **Confirmed**: all 81 channels have < 5 expected exceedances | **Measured**: bootstrap threshold uncertainty propagated to the headline metrics | `src/evaluation/calibration.py`, `test_calibration_uncertainty_contains_point_estimate` |
| D5 | `LSTMDetector(seed=)` did not fix the weights outside the pipeline | **Confirmed** (same seed gave different weights) | **Fixed**: seeded before construction and before training; existing run still reproduces bit-identically (T-9, C-2 re-run) | `test_lstm_seed_alone_controls_weights` |
| D6 | Warm-up rows counted in AUC | Fixed in `e5719e1` | Kept | `test_threshold_free_ignores_warmup_nans` |
| D7 | Path-like `--channel`/`--run` reach a joblib (pickle) load; CSV read fully before the row limit | **Confirmed** (`'../lstm/P-1'` accepted) | **Fixed**: strict name validation, bounded CSV read, CLI exits 2 with a message | `test_load_bundle_rejects_path_like_channel_ids`, `test_read_telemetry_csv_bounds_rows`, `test_cli_predict_rejects_bad_names_and_input` |
| B1 | Event-FP counts not comparable (only LSTM smoothed) | Methodological limitation | Documented | `docs/limitations.md` |
| B2 | Early stopping and calibration share the validation window | Methodological limitation | Documented | `docs/limitations.md` |
| N1 | Absolute home-directory paths in logs (`reports/runs/full/run.log`, committed) | **Confirmed** | Code fixed (logs use project-relative paths). New seed `run.log`s had the project prefix replaced by `<repo>` before their first commit. The committed `full/run.log` and history are **not rewritten** (owner instruction) | `display_path` in `src/utils/config.py` |
| N2–N7 | Fresh-clone dashboard crash; silent uncalibrated inference; undeclared `scipy`; broken doc link; `seeds` output path; hooks | Fixed with tests (see FABLE_REVIEW_BRIEF §13) | | |
| Final audit | A1–A5, B1–B6, C1–C4 (no Critical/High) | Fixed or disclosed (FABLE_REVIEW_BRIEF §14) | | |
| R1 | Single seed | Limitation | **Two more full runs** (seeds 1, 2), new run dirs, resumable | `configs/seed1.yaml`, `configs/seed2.yaml`, `python -m src.cli seeds` |
| R2 | Leave-channels-out calibration (review's top next experiment) | Optional, highest value | **Implemented** on saved scores, pre-specified protocol, held-out-label independence tested | `test_lco_rule_ignores_heldout_labels` |

## Rules for this work

* The `full` run (seed 42) is never overwritten. New runs go to `full_seed1`, `full_seed2`.
* No threshold, rule or hyper-parameter is chosen from the labels of the channels it is evaluated on.
  The leave-channels-out rule for each fold is chosen from the other folds' channels only, with a rule family and
  objective fixed before running. Its grid was **not** widened after seeing results.
* Nothing is committed, pushed or published without the owner's approval. (Approved 2026-11-16: two commits under
  `SJ-Hariharan`, push to `github.com/SJ-Hariharan/nasa-telemetry-anomaly-detection` if it is a safe fast-forward.)

## Acceptance criteria

1. Full test suite passes. Smoke run passes from a fresh clone of the working tree.
2. `full` metrics re-derived from saved artifacts match the reported numbers.
3. Saved-model inference reproduces evaluation scores and flags. Malformed input is rejected with clear errors.
4. Dashboard renders every page (headless AppTest + browser check) and shows numbers equal to `metrics.json`.
5. README, EXPERIMENTS, methodology, limitations, review brief and learning log agree with the code and with each other.
6. No secrets, machine-specific paths in new files, temporary files or large generated files in the diff.
7. Final independent (Fable) review of the release candidate, with every finding verified or addressed.

## Run status

| Run | Seed | State | Evidence |
|---|---|---|---|
| `full` | 42 | Complete (unchanged; committed files byte-identical to `e5719e1`) | 81/81, 0 failures, 61.6 min |
| `full_seed1` | 1 | **Complete** | 81/81, 0 failures, 73.9 min recorded compute. The automatic evaluation crashed (`ImportError`, code edited mid-run) and was re-run alone; logged in its `run.log` |
| `full_seed2` | 2 | **Complete** | 81/81, 0 failures, 106.4 min recorded compute. Killed by a machine reboot after 72/81 (2026-11-16 01:07); resumed 11:10 with `--resume` (9 channels, 219 s); evaluated without error |

Seed summary: `reports/seed_summary.json`. Calibration studies exist for all three runs (`reports/runs/<run>/calibration/`).

## Checks log (exact commands and outcomes)

| Check | Command | Outcome |
|---|---|---|
| Unit + integration + dashboard tests | `pytest -q` (repo venv), also via `.githooks/pre-push` | **59 passed** (8 s), 2026-11-16, after the final-audit fixes |
| Git hooks | throwaway clone with `core.hooksPath .githooks`: trailing whitespace, `.env`, `.pt` file, Python syntax error, clean change; pre-push with a deliberately failing test | 4 bad commits rejected (exit 1), clean commit accepted; pre-push aborted on the failing test (exit 1) and passed on the real tree |
| Seed runs | `python -m src.cli all --resume --config configs/seed{1,2}.yaml`; `python -m src.cli seeds --runs full full_seed1 full_seed2` | Both complete (see Run status); baseline identical across seeds (sanity check) |
| Clean environment install | fresh `python3.11 -m venv` in a temp copy of the working tree (tracked + new files, `.gitignore` respected), `pip install -r requirements.txt` | Installed; torch 2.15.0, scikit-learn 1.9.1, numpy 2.4.6, pandas 3.1.1, streamlit 1.67.0, scipy 1.17.1 (= lock file) |
| Tests in the clean environment | `pytest -q` in the temp copy (no scores/models present) | First run: **3 failed** (dashboard pages crashed without gitignored scores: finding N2). After fix: **57 passed** |
| Data validation (clean env) | `python -m src.cli validate` | `ok: true`, 81 channels, 0 errors, 14 warnings (known quirks) |
| Smoke pipeline (clean env) | `python -m src.cli all --config configs/smoke.yaml` | 3/3 channels, 0 failures, training 4.9 s |
| `full` metrics re-derived independently | script recomputing labels from `labeled_anomalies.csv`, predictions from saved scores and thresholds, point/event counts and macro AUC with scikit-learn | All headline metrics for all 3 methods equal `metrics.json` (tolerance 1e-9) |
| `full` committed files intact | `git diff --stat HEAD -- reports/runs/full` | No changes (only the new untracked `calibration/` folder) |
| D5 fix does not change results | re-ran T-9 and C-2 with `process_channel` into a temp dir | Thresholds equal, LSTM weights bit-identical to `models/full` |
| Inference round trip | `python -m src.cli predict --run full --method lstm --channel M-6 --input <M-6 test as CSV>` | Scores and flags equal the run's table; calibration warning printed |
| Malformed inference input | same command with NaN value, cmd=3, partial cmd columns, 50 rows, no `value`, binary garbage, `--run ../x`, `--channel ../P-1` | Each rejected with a message; exit code 2 (checked for NaN and bad run name) |
| Calibration studies | `python -m src.cli calibration --run full` | Done in ~15 s; outputs in `reports/runs/full/calibration/` |
| Dashboard (browser) | Streamlit on localhost, in-app browser | Overview table equals `metrics.json`; calibratable section lists the 22 channels; Inference on M-6 shows the uncalibrated-threshold warning, 411 flags = run table; Telemetry explorer renders; no console errors |
| Lint / type check | not configured in the repository | Not run |
