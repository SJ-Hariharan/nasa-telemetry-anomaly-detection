"""Time-series window and feature construction.

Alignment convention used everywhere in the project (memorise this one):

    The score for timestep t may use observations x[0..t] only (never x[t+1..]).

* ``forecast_windows``: input = x[t-W .. t-1] (W past rows), target = value at t.
  The returned ``target_idx`` array says which timestep each window predicts, so errors
  are written back to the correct position and compared with the label at that same t.
* ``isolation_features``: features describe x[t-W+1 .. t] (the trailing window ending AT t),
  so the current point is included - it is the point being judged.
Positions without enough history get no window / NaN features ("warm-up").
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.data.loader import VALUE_COL


def forecast_windows(array: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, target_idx) for one-step-ahead forecasting.

    array: (n, f). Returns X: (n - window, window, f) as a read-only view (no copy),
    y: (n - window,), target_idx: (n - window,) with target_idx[i] = i + window.

    Small example: values [a,b,c,d], window=2 ->
        X[0]=[a,b] -> y[0]=c (t=2);  X[1]=[b,c] -> y[1]=d (t=3)
    """
    n = len(array)
    if n <= window:
        raise ValueError(f"Need more than {window} timesteps, got {n}")
    # sliding_window_view over axis 0 gives (n - window + 1, f, window); drop the last window,
    # which would have no target, and move the time axis before the feature axis.
    X = sliding_window_view(array, window, axis=0)[:-1].transpose(0, 2, 1)
    target_idx = np.arange(window, n)
    y = array[target_idx, VALUE_COL]
    return X, y, target_idx


def isolation_features(array: np.ndarray, lags: int, stat_window: int) -> tuple[np.ndarray, list[str]]:
    """Per-timestep tabular features for Isolation Forest (rows with NaN = warm-up).

    Features at t:
      lag_0..lag_{k-1}  raw values x_t, x_{t-1}, ... (local shape)
      roll_mean/std/min/max over the trailing ``stat_window`` (local level and spread)
      dev_from_mean = x_t - roll_mean, diff_1 = x_t - x_{t-1} (sudden jumps)
      cmd_activity = number of command flags set in the trailing window (operational context)
    """
    import pandas as pd

    if stat_window < lags:
        raise ValueError("stat_window must be >= lags")
    v = pd.Series(array[:, VALUE_COL].astype(np.float64))
    cols: dict[str, pd.Series] = {f"lag_{k}": v.shift(k) for k in range(lags)}
    roll = v.rolling(stat_window, min_periods=stat_window)
    cols["roll_mean"] = roll.mean()
    cols["roll_std"] = roll.std(ddof=0)
    cols["roll_min"] = roll.min()
    cols["roll_max"] = roll.max()
    cols["dev_from_mean"] = v - cols["roll_mean"]
    cols["diff_1"] = v.diff()
    if array.shape[1] > 1:
        cmd_count = pd.Series(array[:, 1:].sum(axis=1))
        cols["cmd_activity"] = cmd_count.rolling(stat_window, min_periods=stat_window).sum()
    feats = pd.DataFrame(cols)
    return feats.to_numpy(dtype=np.float32), list(feats.columns)
