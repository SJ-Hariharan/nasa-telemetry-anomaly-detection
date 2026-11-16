# Release notes

## Unreleased (on master, not tagged)

**Dashboard: "Spacecraft Telemetry Anomaly Lab".**
* **NASA results first.** The app opens on the real `full` run from committed reports, with seed error bars, the
  caveats directly under the headline table, and per-mission and per-channel views. It works from a fresh clone with
  no download.
* **Six pages:** Overview, Channel explorer, Events & errors, Models & calibration, Inference, Data &
  reproducibility. A mission filter (SMAP/MSL) and a run selector apply across pages.
* **Provenance labels** on every view: committed report, downloaded NASA source data, or local artifact.
* **One-click verified download and validation, and background reproduction runs** with live progress. They never
  start on page load, the full run needs a confirmation, and outputs go to new `reproduce*` runs.
* **Explorer:** real train/test telemetry with labelled intervals, *Jump to event*, and score and threshold overlays
  when scores exist.
* **Optional synthetic demo** hidden unless enabled, and labelled SIMULATED on every page when shown.
* **Fixes:**
  * downloads now verify TLS with `certifi` (python.org macOS Python lacked CA certificates);
  * cached loaders are keyed by the repository root;
  * marker-overflow flags are visible on log axes;
  * a background run's status is correct after it exits.

## v1.0.0 (release candidate, not yet tagged)

First complete release of **nasa-telemetry-anomaly-detection**: an offline, leakage-controlled comparison of a
rolling z-score baseline, Isolation Forest and an LSTM forecaster on the public NASA SMAP/MSL (Telemanom)
benchmark. Historical benchmark analysis only; it does not monitor any spacecraft.

### Results (full run, seed 42, 81 channels, thresholds calibrated without labels)

| Method | Point F1 | Event recall | Event F1 | False-alarm events | ROC-AUC (macro) | PR-AUC (macro) |
|---|---|---|---|---|---|---|
| Rolling z-score | 0.237 | 86/104 | 0.056 | 2,881 | 0.590 | 0.260 |
| Isolation Forest | 0.283 | 72/104 | 0.065 | 2,025 | 0.635 | 0.292 |
| LSTM forecaster | 0.387 | 97/104 | 0.099 | 1,764 | 0.677 | 0.456 |

Over three seeds (42, 1, 2): LSTM point F1 0.390 ± 0.009, macro ROC-AUC 0.682 ± 0.007, event F1 0.080 ± 0.018
(false-alarm events 1,764–2,897). Seed 42, shown above, is the most favourable seed on event F1 and false alarms.
Leave-channels-out calibration, an estimate for unseen channels, cuts LSTM false-alarm events to 332–434 and
raises event F1 to 0.195–0.231 across seeds (all 81 channels), at the cost of event recall (0.52–0.56). Gains are
smaller on the 59 calibratable channels, and the rule choices sit at the edge of the pre-specified grid.

### Included
* Pinned, SHA-256-verified data download; validation; executed EDA notebook.
* Three detectors behind one interface; per-channel validation-quantile thresholds; resumable training.
* Point, event and macro threshold-free metrics, reported on all 81 channels and on the 59 calibratable channels.
* Calibration studies on saved scores: threshold-estimation bootstrap and leave-channels-out rule selection.
* Seed-variance runs (seeds 42, 1, 2) and `python -m src.cli seeds`.
* Saved-model inference (CLI and dashboard) with input validation and calibration warnings.
* Streamlit dashboard; unit, integration and dashboard tests; teaching documentation.

### Fixed since the first independent review
D1 uncalibratable channels flagged · D2 macro AUC with CIs · D3 shift claim replaced by a KS measurement ·
D5 LSTM seed self-sufficient (results unchanged) · D6 warm-up excluded from AUC · D7 name validation and bounded
CSV reads · fresh-clone dashboard crash · inference warning for uncalibrated thresholds · `scipy` declared.

### Known limitations
Low precision and many false alarms for every method at the default threshold; 22 uncalibratable channels; no
hyper-parameter search; simplified Telemanom; benchmark test-informed scaling. See `docs/limitations.md`.
