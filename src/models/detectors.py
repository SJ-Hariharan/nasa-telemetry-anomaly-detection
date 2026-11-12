"""The three anomaly detectors, behind one small shared interface.

Every detector works on ONE channel that has already been scaled with ``ValueScaler``:

    det.fit(train_scaled, split_idx)   # learn only from train rows [0, split_idx)
    scores = det.score(array_scaled)   # one score per timestep, NaN during warm-up

Higher score = more unusual. Scores are NOT probabilities; a per-channel threshold
(``src.models.thresholds``) turns them into yes/no predictions.
The interface is intentionally tiny so training, calibration, evaluation and inference
can treat all three methods identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data.loader import VALUE_COL
from src.features.windows import isolation_features


class RollingZScoreDetector:
    """Statistical baseline: how many rolling standard deviations is x_t from its recent past?

        score_t = |x_t - mean(x_{t-W..t-1})| / max(std(x_{t-W..t-1}), min_std)

    The window EXCLUDES x_t, otherwise a spike would inflate its own reference std.
    Assumes the recent past is locally stationary and roughly Gaussian - which telemetry
    often violates (step changes, periodic patterns, flat lines).
    """

    name = "baseline"

    def __init__(self, window: int = 100, min_std: float = 0.01):
        self.window = int(window)
        self.min_std = float(min_std)

    def fit(self, train_scaled: np.ndarray, split_idx: int) -> "RollingZScoreDetector":
        return self  # nothing to learn; the rolling window *is* the model

    def score(self, array_scaled: np.ndarray) -> np.ndarray:
        v = pd.Series(array_scaled[:, VALUE_COL].astype(np.float64))
        past = v.shift(1).rolling(self.window, min_periods=self.window)  # x_{t-W}..x_{t-1}
        z = (v - past.mean()).abs() / past.std(ddof=0).clip(lower=self.min_std)
        return z.to_numpy()

    @property
    def warmup(self) -> int:
        """Leading timesteps that get no score (not enough history)."""
        return self.window

    def get_params(self) -> dict:
        return {"window": self.window, "min_std": self.min_std}


class IsolationForestDetector:
    """Isolation Forest on trailing-window features (see ``isolation_features``).

    Anomalies are 'few and different', so random axis-aligned splits isolate them in fewer
    steps. sklearn's ``score_samples`` is high for normal points; we negate it so that
    higher = more anomalous, consistent with the other detectors.
    """

    name = "isolation_forest"

    def __init__(self, lags: int = 10, stat_window: int = 50, n_estimators: int = 200,
                 max_samples: int | str = 256, contamination="auto", seed: int = 42):
        self.lags, self.stat_window = int(lags), int(stat_window)
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                     contamination=contamination, random_state=seed, n_jobs=-1)
        self.feature_names: list[str] = []

    def fit(self, train_scaled: np.ndarray, split_idx: int) -> "IsolationForestDetector":
        feats, self.feature_names = isolation_features(train_scaled, self.lags, self.stat_window)
        # Features at t only look backwards, so rows < split_idx never see validation data.
        fit_rows = feats[:split_idx]
        fit_rows = fit_rows[np.isfinite(fit_rows).all(axis=1)]
        if len(fit_rows) < 10:
            raise ValueError("Too few complete feature rows to fit Isolation Forest")
        self.model.fit(fit_rows)
        return self

    def score(self, array_scaled: np.ndarray) -> np.ndarray:
        feats, _ = isolation_features(array_scaled, self.lags, self.stat_window)
        scores = np.full(len(feats), np.nan)
        ok = np.isfinite(feats).all(axis=1)
        if ok.any():
            scores[ok] = -self.model.score_samples(feats[ok])
        return scores

    @property
    def warmup(self) -> int:
        return self.stat_window - 1

    def get_params(self) -> dict:
        return {"lags": self.lags, "stat_window": self.stat_window, "feature_names": self.feature_names}
