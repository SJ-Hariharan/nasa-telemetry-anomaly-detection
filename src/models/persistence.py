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
from dataclasses import dataclass
from pathlib import Path

import joblib
import torch

from src.models.detectors import IsolationForestDetector, RollingZScoreDetector
from src.models.lstm import LSTMDetector, LSTMForecaster
from src.preprocessing.scaling import ValueScaler

METHODS = ("baseline", "isolation_forest", "lstm")


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

    def save(self, models_dir: Path) -> Path:
        out = Path(models_dir) / self.method / self.chan_id
        out.mkdir(parents=True, exist_ok=True)
        meta = {"method": self.method, "chan_id": self.chan_id, "scaler": self.scaler.to_dict(),
                "threshold": self.threshold, "n_features": self.n_features,
                "params": self.detector.get_params(), "postprocess": self.postprocess}
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
                          float(meta["threshold"]), int(meta["n_features"]), meta.get("postprocess", {}))
