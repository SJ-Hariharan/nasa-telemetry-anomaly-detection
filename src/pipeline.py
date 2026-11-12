"""Training + scoring pipeline (per channel, per method).

For each channel:
  1. load train/test arrays and labels
  2. chronological split of TRAIN into fit / validation      (src.preprocessing.scaling)
  3. fit ValueScaler on the fit part only, transform everything
  4. for each method: fit on fit part -> score validation -> calibrate threshold
     -> score test -> threshold -> predictions                (no test label is read until here)
  5. save the model bundle + a per-channel score table for evaluation and the dashboard

Labels are attached to the score table only as a final column for the evaluation step.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import VALUE_COL, label_vector, load_channel, load_labels, select_channels
from src.models.detectors import IsolationForestDetector, RollingZScoreDetector
from src.models.lstm import LSTMDetector
from src.models.persistence import METHODS, DetectorBundle
from src.models.thresholds import apply_threshold, calibrate_threshold, calibration_is_degenerate
from src.preprocessing.scaling import ValueScaler, chronological_split_index
from src.utils.config import resolve_path
from src.utils.reproducibility import get_logger, set_seed

log = get_logger("pipeline")


def build_detector(method: str, cfg: dict):
    if method == "baseline":
        return RollingZScoreDetector(**cfg["baseline"])
    if method == "isolation_forest":
        return IsolationForestDetector(**cfg["isolation_forest"], seed=cfg["seed"])
    if method == "lstm":
        return LSTMDetector(**cfg["lstm"], seed=cfg["seed"])
    raise ValueError(f"Unknown method {method!r}")


def validation_scores(det, train_scaled: np.ndarray, split: int) -> np.ndarray:
    """Score the validation part [split, n) under the SAME conditions as the test file.

    The test file is scored on its own: each detector starts with exactly ``warmup`` steps of history
    and any smoothing (LSTM EWMA) starts fresh. Scoring the whole train file instead would let EWMA
    carry in-sample (fit-part) errors across the boundary, lowering early validation scores and
    therefore the threshold. So we score the slice that starts ``warmup`` steps before the split.
    """
    start = split - det.warmup
    if start < 0:
        raise ValueError(f"Fit part ({split}) shorter than detector warm-up ({det.warmup})")
    scores = det.score(train_scaled[start:])[det.warmup:]
    assert len(scores) == len(train_scaled) - split
    return scores


def run_paths(cfg: dict) -> tuple[Path, Path]:
    return (resolve_path(cfg["output"]["models_dir"]) / cfg["run_name"],
            resolve_path(cfg["output"]["reports_dir"]) / cfg["run_name"])


def process_channel(chan_id: str, label_row, raw_dir: Path, cfg: dict, methods: list[str],
                    models_dir: Path) -> tuple[pd.DataFrame, dict]:
    train = load_channel(raw_dir, chan_id, "train")
    test = load_channel(raw_dir, chan_id, "test")
    split = chronological_split_index(len(train), cfg["split"]["val_fraction"])
    scaler = ValueScaler().fit(train[:split])                 # statistics from fit part ONLY
    train_s, test_s = scaler.transform(train), scaler.transform(test)

    table = pd.DataFrame({"timestep": np.arange(len(test)), "value": test[:, VALUE_COL]})
    info: dict = {"chan_id": chan_id, "spacecraft": label_row.spacecraft, "train_len": len(train),
                  "test_len": len(test), "split_idx": split, "constant_train": scaler.constant, "methods": {}}
    val_table = pd.DataFrame({"timestep": np.arange(split, len(train))})   # validation scores, kept for diagnostics
    post = cfg["postprocess"]
    for method in methods:
        set_seed(cfg["seed"])                                  # same seed per channel & method
        t0 = time.perf_counter()
        try:
            det = build_detector(method, cfg).fit(train_s, split)
            val_scores = validation_scores(det, train_s, split)  # validation = last part of TRAIN
            thr = calibrate_threshold(val_scores, **cfg["threshold"])
            scores = det.score(test_s)
        except Exception as exc:  # report, never hide: the run summary lists every failure
            log.error("%s/%s failed: %s", chan_id, method, exc)
            info["methods"][method] = {"status": "failed", "error": repr(exc)}
            continue
        pred = apply_threshold(scores, thr, post["min_event_length"], post["merge_gap"])
        table[f"{method}_score"] = scores
        table[f"{method}_pred"] = pred
        if method == "lstm":  # forecast in original units, for explanation plots
            table["lstm_forecast"] = det.predict(test_s) * scaler.std + scaler.mean
        DetectorBundle(method, chan_id, det, scaler, thr, train.shape[1], post).save(models_dir)
        val_table[f"{method}_score"] = val_scores
        degenerate = calibration_is_degenerate(val_scores, thr)
        if degenerate:
            log.warning("%s/%s: validation scores have no spread - threshold %.3g is NOT a calibration", chan_id, method, thr)
        m_info = {"status": "ok", "threshold": thr, "seconds": round(time.perf_counter() - t0, 2),
                  "val_flag_rate": float(np.mean(np.nan_to_num(val_scores, nan=-np.inf) > thr)),
                  "calibratable": not degenerate, "val_len": int(len(val_scores))}
        if method == "lstm":
            m_info.update(best_epoch=det.best_epoch, epochs_run=len(det.history["val_loss"]),
                          best_val_mse=float(min(det.history["val_loss"])))
        info["methods"][method] = m_info
    # Labels are joined only now, after every model has been fitted, calibrated and has predicted.
    table["label"] = label_vector(len(test), label_row.anomaly_sequences)
    return table, info, val_table


def run_training(cfg: dict, methods: list[str] | None = None, resume: bool = False) -> Path:
    """Train/score every selected channel. ``resume=True`` skips channels already completed by an
    interrupted run with the IDENTICAL config (results are deterministic per channel, so this is
    equivalent to a full rerun)."""
    methods = methods or cfg.get("methods", list(METHODS))
    set_seed(cfg["seed"])
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    labels = load_labels(raw_dir).set_index("chan_id", drop=False)
    channels = select_channels(labels, cfg["data"]["channels"], cfg["data"]["spacecraft"])
    models_dir, reports_dir = run_paths(cfg)
    (reports_dir / "scores").mkdir(parents=True, exist_ok=True)
    cfg_path = reports_dir / "config_used.json"
    if resume and cfg_path.exists() and json.loads(cfg_path.read_text()) != json.loads(json.dumps(cfg)):
        raise ValueError("Cannot resume: config differs from the interrupted run's config_used.json")
    cfg_path.write_text(json.dumps(cfg, indent=1))

    infos = []
    t_start = time.perf_counter()
    for i, chan in enumerate(channels, 1):
        info_path = reports_dir / "scores" / f"{chan}.json"
        if resume and info_path.exists() and (reports_dir / "scores" / f"{chan}.parquet").exists():
            infos.append(json.loads(info_path.read_text()))
            log.info("[%d/%d] %s already complete - skipped (resume)", i, len(channels), chan)
            continue
        table, info, val_table = process_channel(chan, labels.loc[chan], raw_dir, cfg, methods, models_dir)
        table.to_parquet(reports_dir / "scores" / f"{chan}.parquet", index=False)
        val_table.to_parquet(reports_dir / "scores" / f"{chan}.val.parquet", index=False)
        info_path.write_text(json.dumps(info, indent=1))   # written last: marks the channel complete
        infos.append(info)
        times = {m: v.get("seconds") for m, v in info["methods"].items()}
        log.info("[%d/%d] %s done %s", i, len(channels), chan, times)
    summary = {"run_name": cfg["run_name"], "methods": methods, "n_channels": len(channels),
               "total_seconds": round(time.perf_counter() - t_start, 1),
               "sum_method_seconds": round(sum(v.get("seconds") or 0 for c in infos for v in c["methods"].values()), 1),
               "channels": infos,
               "failures": [(c["chan_id"], m) for c in infos for m, v in c["methods"].items()
                            if v["status"] != "ok"]}
    (reports_dir / "training_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("Training finished in %.1fs; failures: %s", summary["total_seconds"], summary["failures"] or "none")
    return reports_dir
