"""Shared fixtures: a tiny synthetic dataset in the exact benchmark layout.

The synthetic data is ONLY for testing code paths offline; it is never reported as a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _series(n: int, rng: np.random.Generator, n_cmd: int) -> np.ndarray:
    t = np.arange(n)
    value = 0.5 * np.sin(2 * np.pi * t / 50) + 0.02 * rng.standard_normal(n)
    cmds = (rng.random((n, n_cmd)) < 0.01).astype(np.float32)
    return np.column_stack([value, cmds]).astype(np.float32)


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    (raw / "train").mkdir(parents=True)
    (raw / "test").mkdir()
    rows = ["chan_id,spacecraft,anomaly_sequences,class,num_values"]
    for chan, sc, n_cmd in (("S-1", "SMAP", 24), ("M-1", "MSL", 54)):
        train, test = _series(1500, rng, n_cmd), _series(1000, rng, n_cmd)
        test[600:640, 0] += 2.0      # injected level shift  (contextual-like)
        test[850, 0] = -3.0          # injected spike        (point)
        np.save(raw / "train" / f"{chan}.npy", train)
        np.save(raw / "test" / f"{chan}.npy", test)
        rows.append(f'{chan},{sc},"[[600, 639], [850, 850]]","[contextual, point]",1000')
    (raw / "labeled_anomalies.csv").write_text("\n".join(rows) + "\n")
    return raw


@pytest.fixture
def tiny_config(synthetic_raw_dir: Path, tmp_path: Path) -> dict:
    from src.utils.config import load_config

    cfg = load_config("configs/smoke.yaml")
    cfg["run_name"] = "pytest"
    cfg["data"].update(raw_dir=str(synthetic_raw_dir), channels="all")
    cfg["output"] = {"models_dir": str(tmp_path / "models"), "reports_dir": str(tmp_path / "reports")}
    cfg["lstm"].update(window=30, epochs=3, max_train_windows=300)
    cfg["baseline"]["window"] = 30
    return cfg
