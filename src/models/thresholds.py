"""Threshold calibration and turning scores into anomaly predictions.

A score only ranks timesteps; deciding "anomalous or not" needs a threshold. We calibrate one
threshold PER CHANNEL using ONLY the validation part of the train file (assumed mostly nominal,
and never labelled). Test labels are never used here - tuning on them would make the reported
test metrics optimistic and unrepeatable on new data.

    quantile:  thr = margin * Quantile_q(validation scores)
               e.g. q = 0.995 -> about 0.5% of nominal validation points would be flagged.
    mean_std:  thr = margin * (mean + k * std) of validation scores.
"""
from __future__ import annotations

import numpy as np


def calibrate_threshold(val_scores: np.ndarray, method: str = "quantile", quantile: float = 0.995,
                        margin: float = 1.0, k_std: float = 4.0, **_ignored) -> float:
    s = np.asarray(val_scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("No finite validation scores - cannot calibrate a threshold")
    if method == "quantile":
        thr = np.quantile(s, quantile)
    elif method == "mean_std":
        thr = s.mean() + k_std * s.std()
    else:
        raise ValueError(f"Unknown threshold method {method!r}")
    thr = float(thr * margin)
    # A constant validation signal can give thr == 0, which would flag every non-zero test score.
    return thr if thr > 0 else float(np.finfo(np.float32).eps)


def runs(binary: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1s as inclusive (start, end) pairs. [0,1,1,0,1] -> [(1,2),(4,4)]"""
    b = np.concatenate([[0], np.asarray(binary, dtype=np.int8), [0]])
    d = np.diff(b)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def apply_threshold(scores: np.ndarray, threshold: float, min_event_length: int = 1, merge_gap: int = 0) -> np.ndarray:
    """Binary predictions. Warm-up positions (NaN score) are predicted normal (0)."""
    s = np.asarray(scores, dtype=np.float64)
    pred = (np.nan_to_num(s, nan=-np.inf) > threshold).astype(np.int8)
    if merge_gap > 0:
        r = runs(pred)
        for (s1, e1), (s2, _e2) in zip(r, r[1:]):
            if s2 - e1 - 1 <= merge_gap:
                pred[e1 + 1 : s2] = 1
    if min_event_length > 1:
        for s0, e0 in runs(pred):
            if e0 - s0 + 1 < min_event_length:
                pred[s0 : e0 + 1] = 0
    return pred
