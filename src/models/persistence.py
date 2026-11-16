"""Save / load a fitted per-channel detector bundle.

Layout:  models/<run_name>/<method>/<chan_id>/
            meta.json     method, channel, scaler stats, threshold, hyper-parameters
            model.pt      LSTM weights (state_dict only)          [lstm]
            model.joblib  fitted IsolationForest                  [isolation_forest]

Security: joblib/pickle files can execute code when loaded, so only load model directories that
this project produced. LSTM weights are loaded with ``weights_only=True`` (no code execution).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import torch

from src.models.detectors import IsolationForestDetector, RollingZScoreDetector
from src.models.lstm import LSTMDetector, LSTMForecaster
from src.preprocessing.scaling import ValueScaler

METHODS = ("baseline", "isolation_forest", "lstm")
CHANNEL_ID = re.compile(r"^[A-Z]-\d{1,3}$")      # benchmark channel ids: P-1, M-6, T-13 ...
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def check_name(value: str, pattern: re.Pattern, what: str) -> str:
    """Reject names that would escape the models directory (e.g. '../x') before they become a path.
    Loading a joblib file executes code, so the location must never be attacker-chosen."""
    if not isinstance(value, str) or not pattern.fullmatch(value) or ".." in value:
        raise ValueError(f"Invalid {what}: {value!r}")
    return value


@dataclass
class DetectorBundle:
    """Everything needed to score new data for one channel with one method."""

    method: str
    chan_id: str
    detector: object
    scaler: ValueScaler
    threshold: float
    n_features: int
    postprocess: dict
    calibratable: bool | None = None     # False: constant training signal / no-spread validation scores; None: unknown

    def calibration_warning(self) -> str | None:
        if self.calibratable is False:
            return (f"The {self.method} threshold for {self.chan_id} is NOT calibrated: the channel's training signal is "
                    "constant (or its validation scores have no spread), so the threshold is ~0 and almost any change is "
                    "flagged. Treat these flags as 'differs from training', not as calibrated anomalies.")
        if self.calibratable is None:
            return "Calibration status was not recorded for this model; check the run's training_summary.json."
        return None

    def save(self, models_dir: Path) -> Path:
        out = Path(models_dir) / self.method / self.chan_id
        out.mkdir(parents=True, exist_ok=True)
        meta = {"method": self.method, "chan_id": self.chan_id, "scaler": self.scaler.to_dict(),
                "threshold": self.threshold, "n_features": self.n_features,
                "params": self.detector.get_params(), "postprocess": self.postprocess,
                "calibratable": self.calibratable}
        if self.method == "lstm":
            torch.save(self.detector.model.state_dict(), out / "model.pt")
            meta["best_epoch"] = self.detector.best_epoch
            meta["history"] = self.detector.history
        elif self.method == "isolation_forest":
            joblib.dump(self.detector.model, out / "model.joblib")
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        return out


def load_bundle(models_dir: Path, method: str, chan_id: str) -> DetectorBundle:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    check_name(chan_id, CHANNEL_ID, "channel id")
    path = Path(models_dir) / method / chan_id
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No saved {method} model for channel {chan_id} under {models_dir}")
    meta = json.loads(meta_path.read_text())
    params = meta["params"]
    if method == "baseline":
        det = RollingZScoreDetector(**params)
    elif method == "isolation_forest":
        det = IsolationForestDetector(lags=params["lags"], stat_window=params["stat_window"])
        det.model = joblib.load(path / "model.joblib")
        det.feature_names = params["feature_names"]
    else:
        det = LSTMDetector(**params)
        n_in = meta["n_features"] if det.use_commands else 1
        det.model = LSTMForecaster(n_in, det.hidden_size, det.num_layers, det.dropout)
        det.model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
        det.model.eval()
    return DetectorBundle(method, chan_id, det, ValueScaler.from_dict(meta["scaler"]),
                          float(meta["threshold"]), int(meta["n_features"]), meta.get("postprocess", {}),
                          meta.get("calibratable"))
