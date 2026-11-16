"""Inference: score NEW telemetry with a saved detector bundle.

Decoupled from training: it needs only ``models/<run>/<method>/<chan>/`` (scaler statistics,
threshold, weights). The same scaler and threshold that were fitted at training time are reused -
never refitted on the incoming data (that would silently shift the decision boundary).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.persistence import DetectorBundle, load_bundle
from src.models.thresholds import apply_threshold

MAX_ROWS = 500_000  # protects the dashboard from accidentally huge uploads


def warmup_length(bundle: DetectorBundle) -> int:
    """Number of initial timesteps that cannot be scored (not enough history)."""
    return int(bundle.detector.warmup)


def frame_to_array(df: pd.DataFrame, n_features: int) -> tuple[np.ndarray, list[str]]:
    """Validate a telemetry table and convert it to the (n, n_features) array the model expects.

    Required: a numeric ``value`` column. Optional: ``cmd_0..cmd_{n_features-2}`` 0/1 columns.
    If command columns are absent they are assumed inactive (0) and a warning is returned,
    because models that use commands will then see a different input distribution.
    """
    warnings: list[str] = []
    if "value" not in df.columns:
        raise ValueError("Input must contain a 'value' column")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Input has {len(df)} rows; limit is {MAX_ROWS}")
    values = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("'value' column contains missing or non-numeric entries")
    arr = np.zeros((len(df), n_features), dtype=np.float32)
    arr[:, 0] = values
    cmd_cols = [f"cmd_{i}" for i in range(n_features - 1)]
    present = [c for c in cmd_cols if c in df.columns]
    if present and len(present) != len(cmd_cols):
        raise ValueError(f"Expected all of cmd_0..cmd_{n_features - 2} or none; got {len(present)}")
    if present:
        cmds = df[cmd_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        if not np.isin(cmds, (0, 1)).all():
            raise ValueError("Command columns must contain only 0/1")
        arr[:, 1:] = cmds
    elif n_features > 1:
        warnings.append("No cmd_* columns supplied: command flags assumed 0 (inactive).")
    if values.min() < -1.5 or values.max() > 1.5:
        warnings.append("Values fall well outside the benchmark's [-1, 1] range; scores may be unreliable.")
    return arr, warnings


def detect(bundle: DetectorBundle, array: np.ndarray) -> pd.DataFrame:
    """Score raw (unscaled, benchmark-units) telemetry with a fitted bundle."""
    if array.ndim != 2 or array.shape[1] != bundle.n_features:
        raise ValueError(f"Expected shape (n, {bundle.n_features}), got {array.shape}")
    w = warmup_length(bundle)
    if len(array) <= w:
        raise ValueError(f"Need more than {w} timesteps for {bundle.method} (warm-up), got {len(array)}")
    scores = bundle.detector.score(bundle.scaler.transform(array))
    pred = apply_threshold(scores, bundle.threshold, bundle.postprocess.get("min_event_length", 1),
                           bundle.postprocess.get("merge_gap", 0))
    return pd.DataFrame({"timestep": np.arange(len(array)), "value": array[:, 0], "score": scores,
                         "threshold": bundle.threshold, "normalized_score": scores / bundle.threshold,
                         "is_anomaly": pred, "scored": np.isfinite(scores)})


def read_telemetry_csv(source) -> pd.DataFrame:
    """Read at most MAX_ROWS + 1 rows, so an oversized file is rejected without loading all of it."""
    df = pd.read_csv(source, nrows=MAX_ROWS + 1)
    if len(df) > MAX_ROWS:
        raise ValueError(f"Input has more than {MAX_ROWS} rows")
    return df


def detect_from_csv(models_dir: Path, method: str, chan_id: str, csv_path) -> pd.DataFrame:
    bundle = load_bundle(models_dir, method, chan_id)
    arr, warnings = frame_to_array(read_telemetry_csv(csv_path), bundle.n_features)
    if bundle.calibration_warning():
        warnings.append(bundle.calibration_warning())
    for w in warnings:
        print(f"warning: {w}")
    return detect(bundle, arr)
