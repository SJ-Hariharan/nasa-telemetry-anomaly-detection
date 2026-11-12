"""Leakage-safe scaling and chronological splitting.

Rules enforced here:
1. Statistics (mean, std) come ONLY from the fitting part of the train file.
2. The validation part is the *last* chunk of the train file (chronological, no shuffling), so the
   model never learns from observations that come after the ones it is validated on.
3. The test file is only ever transformed, never used to fit anything.

Caveat we cannot undo: the benchmark authors already min-max scaled each channel to [-1, 1]
using the TEST set's min/max (stated in the Telemanom README). Our z-scaling is a further
affine transform fitted on train data; it does not add new leakage, but it cannot remove the
original one either. See docs/limitations.md.
"""
from __future__ import annotations

import numpy as np

from src.data.loader import VALUE_COL


def chronological_split_index(n: int, val_fraction: float) -> int:
    """Index where validation starts: points [0, split) fit, [split, n) validate."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    split = int(round(n * (1.0 - val_fraction)))
    if split < 1 or split >= n:
        raise ValueError(f"Series of length {n} too short for val_fraction={val_fraction}")
    return split


class ValueScaler:
    """Z-score the telemetry value column; command columns (0/1) are left untouched."""

    MIN_STD = 1e-3

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std)

    def fit(self, fit_array: np.ndarray) -> "ValueScaler":
        values = fit_array[:, VALUE_COL]
        self.mean = float(values.mean())
        # 19 benchmark channels are exactly constant in train (std = 0). Dividing by ~0 would turn
        # tiny test changes into astronomically large inputs, so we keep the original units instead.
        std = float(values.std())
        self.std = std if std >= self.MIN_STD else 1.0
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        out = np.array(array, dtype=np.float32, copy=True)
        out[:, VALUE_COL] = (out[:, VALUE_COL] - self.mean) / self.std
        return out

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "ValueScaler":
        return cls(d["mean"], d["std"])
