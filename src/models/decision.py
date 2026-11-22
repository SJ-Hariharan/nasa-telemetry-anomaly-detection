"""Turn detector scores into yes/no predictions - one function shared by training, inference and replay.

Two threshold modes:
* ``static``  (baseline, isolation_forest, lstm): one per-channel threshold calibrated on the validation part of
  the train file (``src.models.thresholds``).
* ``dynamic`` (telemanom, lstm_dynamic): Telemanom's label-free nonparametric threshold computed from the smoothed
  errors of the series being judged (``src.models.dynamic_threshold``). Scores are converted to benchmark units
  first (``unit_scale`` = the scaler's std) because the method has absolute 0.05 rules defined in those units.
"""
from __future__ import annotations

import numpy as np

from src.models.dynamic_threshold import dynamic_threshold
from src.models.thresholds import apply_threshold

DYNAMIC_METHODS = ("telemanom", "lstm_dynamic")


def threshold_mode(method: str) -> str:
    return "dynamic" if method in DYNAMIC_METHODS else "static"


def postprocess(pred: np.ndarray, post: dict | None) -> np.ndarray:
    post = post or {}
    return apply_threshold(np.asarray(pred, dtype=np.float64), 0.5, post.get("min_event_length", 1),
                           post.get("merge_gap", 0))


def decide(method: str, scores: np.ndarray, raw_values: np.ndarray, *, threshold: float | None = None,
           unit_scale: float = 1.0, post: dict | None = None, dynamic_cfg: dict | None = None,
           decided_until: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (decision_scores, predictions, per-position threshold).

    For static methods ``decision_scores`` are the scores unchanged and the threshold is constant. For dynamic
    methods the scores are expressed in benchmark units and the threshold is the window epsilon (NaN = warm-up).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if threshold_mode(method) == "static":
        if threshold is None:
            raise ValueError(f"{method} needs a calibrated threshold")
        post = post or {}
        pred = apply_threshold(scores, threshold, post.get("min_event_length", 1), post.get("merge_gap", 0))
        return scores, pred, np.full(scores.shape, float(threshold))
    s = scores * float(unit_scale)
    pred, eps = dynamic_threshold(s, raw_values, dynamic_cfg, decided_until=decided_until)
    return s, postprocess(pred, post), eps
