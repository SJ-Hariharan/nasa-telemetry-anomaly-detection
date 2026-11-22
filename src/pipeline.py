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
from src.models.decision import decide, threshold_mode
from src.models.lstm import InsufficientDataError, LSTMDetector, TelemanomDetector
from src.models.persistence import TRAINABLE, DetectorBundle
from src.models.thresholds import calibrate_threshold, calibration_is_degenerate
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
    if method == "telemanom":
        return TelemanomDetector(**cfg["telemanom"], seed=cfg["seed"])
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


def method_inputs(method: str, cfg: dict, base_scaler: ValueScaler, train, test, train_s, test_s):
    """Scaler + inputs for one method. ``raw_units: true`` (telemanom) keeps the benchmark's own [-1, 1] units,
    as in the reference implementation, so its absolute settings (min_delta, the 0.05 rules) mean the same thing."""
    if not (cfg.get(method) or {}).get("raw_units", False):
        return base_scaler, train_s, test_s
    ident = ValueScaler(0.0, 1.0)
    ident.constant = base_scaler.constant
    return ident, ident.transform(train), ident.transform(test)


def run_paths(cfg: dict) -> tuple[Path, Path]:
    return (resolve_path(cfg["output"]["models_dir"]) / cfg["run_name"],
            resolve_path(cfg["output"]["reports_dir"]) / cfg["run_name"])


def process_channel(chan_id: str, label_row, raw_dir: Path, cfg: dict, methods: list[str],
                    models_dir: Path) -> tuple[pd.DataFrame, dict]:
    train = load_channel(raw_dir, chan_id, "train")
    test = load_channel(raw_dir, chan_id, "test")
    split = chronological_split_index(len(train), cfg["split"]["val_fraction"])
    base_scaler = ValueScaler().fit(train[:split])            # statistics from fit part ONLY
    train_s0, test_s0 = base_scaler.transform(train), base_scaler.transform(test)

    table = pd.DataFrame({"timestep": np.arange(len(test)), "value": test[:, VALUE_COL]})
    info: dict = {"chan_id": chan_id, "spacecraft": label_row.spacecraft, "train_len": len(train),
                  "test_len": len(test), "split_idx": split, "constant_train": base_scaler.constant, "methods": {}}
    val_table = pd.DataFrame({"timestep": np.arange(split, len(train))})   # validation scores, kept for diagnostics
    post = cfg["postprocess"]
    for method in methods:
        set_seed(cfg["seed"])                                  # same seed per channel & method
        t0 = time.perf_counter()
        scaler, train_s, test_s = method_inputs(method, cfg, base_scaler, train, test, train_s0, test_s0)
        try:
            det = build_detector(method, cfg).fit(train_s, split)
            val_scores = validation_scores(det, train_s, split)  # validation = last part of TRAIN
            scores = det.score(test_s)
            dyn = cfg.get("dynamic_threshold")
            if threshold_mode(method) == "static":
                thr = calibrate_threshold(val_scores, **cfg["threshold"])
                _, pred, _ = decide(method, scores, test[:, VALUE_COL], threshold=thr, post=post)
                val_pred = (np.nan_to_num(val_scores, nan=-np.inf) > thr)
            else:   # label-free dynamic threshold on the series being judged (validation and test separately)
                val_scores, val_pred, _ = decide(method, val_scores, train[split:, VALUE_COL],
                                                 unit_scale=scaler.std, post=post, dynamic_cfg=dyn)
                scores, pred, eps = decide(method, scores, test[:, VALUE_COL], unit_scale=scaler.std, post=post,
                                           dynamic_cfg=dyn)
                table[f"{method}_threshold"] = eps
                thr = float(np.nanmedian(eps)) if np.isfinite(eps).any() else float("nan")
        except InsufficientDataError as exc:
            # Too little training data to build the model: the method raises no alarm on this channel. The channel
            # stays in the evaluation (its events count as missed) so every method is scored on the same channels.
            log.warning("%s/%s: %s - no model, no alarms", chan_id, method, exc)
            table[f"{method}_score"] = np.nan
            table[f"{method}_pred"] = np.zeros(len(test), dtype=np.int8)
            val_table[f"{method}_score"] = np.nan
            info["methods"][method] = {"status": "ok", "insufficient_data": True, "threshold": float("nan"),
                                       "threshold_mode": threshold_mode(method), "calibratable": False,
                                       "val_flag_rate": 0.0, "val_len": int(len(train) - split),
                                       "seconds": round(time.perf_counter() - t0, 2), "note": str(exc)}
            continue
        except Exception as exc:  # report, never hide: the run summary lists every failure
            log.error("%s/%s failed: %s", chan_id, method, exc)
            info["methods"][method] = {"status": "failed", "error": repr(exc)}
            continue
        table[f"{method}_score"] = scores
        table[f"{method}_pred"] = pred
        if method in ("lstm", "telemanom"):  # forecast in original units, for explanation plots
            table[f"{method}_forecast"] = det.predict(test_s) * scaler.std + scaler.mean
            val_table[f"{method}_forecast"] = (det.predict(train_s[split - det.warmup:])[det.warmup:]
                                               * scaler.std + scaler.mean)
            val_table["value"] = train[split:, VALUE_COL]
        val_table[f"{method}_score"] = val_scores
        degenerate = calibration_is_degenerate(val_scores, thr)
        if degenerate:
            log.warning("%s/%s: validation scores have no spread - threshold %.3g is NOT a calibration", chan_id, method, thr)
        DetectorBundle(method, chan_id, det, scaler, thr, train.shape[1], post,
                       calibratable=not (degenerate or scaler.constant),
                       dynamic=dyn if threshold_mode(method) == "dynamic" else None).save(models_dir)
        m_info = {"status": "ok", "threshold": thr, "threshold_mode": threshold_mode(method),
                  "seconds": round(time.perf_counter() - t0, 2),
                  "val_flag_rate": float(np.mean(val_pred)),
                  "calibratable": not degenerate, "val_len": int(len(val_scores))}
        if method in ("lstm", "telemanom"):
            m_info.update(best_epoch=det.best_epoch, epochs_run=len(det.history["val_loss"]),
                          best_val_mse=float(min(det.history["val_loss"])), device=det.device_requested, train_device=det.train_device)
        info["methods"][method] = m_info
    # Labels are joined only now, after every model has been fitted, calibrated and has predicted.
    table["label"] = label_vector(len(test), label_row.anomaly_sequences)
    return table, info, val_table


def parse_shard(text: str | None) -> tuple[int, int] | None:
    """'2/4' -> (2, 4): this process handles channels whose position % 4 == 2."""
    if not text:
        return None
    i, n = (int(x) for x in text.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"Bad shard {text!r}: expected I/N with 0 <= I < N")
    return i, n


def run_training(cfg: dict, methods: list[str] | None = None, resume: bool = False,
                 shard: tuple[int, int] | None = None) -> Path:
    """Train/score every selected channel. ``resume=True`` skips channels already completed by an
    interrupted run with the IDENTICAL config (results are deterministic per channel, so this is
    equivalent to a full rerun).

    ``shard=(i, n)`` processes only every n-th channel (several processes can share one run directory, each writing
    its own channels). A shard does not write training_summary.json; run once more with ``resume=True`` and no shard
    to collect all channels."""
    methods = methods or cfg.get("methods", ["baseline", "isolation_forest", "lstm"])
    unknown = set(methods) - set(TRAINABLE)
    if unknown:
        raise ValueError(f"Cannot train {sorted(unknown)}; trainable methods: {TRAINABLE}")
    set_seed(cfg["seed"])
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    labels = load_labels(raw_dir).set_index("chan_id", drop=False)
    channels = select_channels(labels, cfg["data"]["channels"], cfg["data"]["spacecraft"])
    models_dir, reports_dir = run_paths(cfg)
    (reports_dir / "scores").mkdir(parents=True, exist_ok=True)
    cfg_path = reports_dir / "config_used.json"
    if (resume or shard) and cfg_path.exists() and json.loads(cfg_path.read_text()) != json.loads(json.dumps(cfg)):
        raise ValueError("Cannot resume: config differs from the interrupted run's config_used.json")
    cfg_path.write_text(json.dumps(cfg, indent=1))

    infos = []
    t_start = time.perf_counter()
    for i, chan in enumerate(channels, 1):
        if shard and (i - 1) % shard[1] != shard[0]:
            continue
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
    if shard:
        log.info("Shard %d/%d finished in %.1fs (no summary written: run again with --resume to collect)",
                 shard[0], shard[1], time.perf_counter() - t_start)
        return reports_dir
    summary = {"run_name": cfg["run_name"], "methods": methods, "n_channels": len(channels),
               "total_seconds": round(time.perf_counter() - t_start, 1),
               "sum_method_seconds": round(sum(v.get("seconds") or 0 for c in infos for v in c["methods"].values()), 1),
               "channels": infos,
               "failures": [(c["chan_id"], m) for c in infos for m, v in c["methods"].items()
                            if v["status"] != "ok"]}
    (reports_dir / "training_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("Training finished in %.1fs; failures: %s", summary["total_seconds"], summary["failures"] or "none")
    return reports_dir


def run_derived(cfg: dict, resume: bool = False) -> Path:
    """``lstm_dynamic``: re-score a finished run's SAVED LSTM models with Telemanom's dynamic threshold.

    No training happens: the reference LSTM weights, scalers and validation split are reused unchanged, only the error
    smoothing and the decision rule change (``cfg['lstm_dynamic']``, ``cfg['dynamic_threshold']``). This isolates
    the effect of the thresholding procedure from the effect of the forecaster. The re-computed forecasts are checked
    against the source run's stored forecasts (``forecast_max_abs_diff``) to prove the models are the same.
    """
    from src.models.persistence import load_bundle

    src_run = cfg["derived_from"]
    models_root, reports_root = resolve_path(cfg["output"]["models_dir"]), resolve_path(cfg["output"]["reports_dir"])
    src_reports, src_models = reports_root / src_run, models_root / src_run
    src_summary = json.loads((src_reports / "training_summary.json").read_text())
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    labels = load_labels(raw_dir).set_index("chan_id", drop=False)
    out = reports_root / cfg["run_name"]
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=1))
    over, dyn, post = cfg.get("lstm_dynamic", {}), cfg.get("dynamic_threshold"), cfg["postprocess"]
    infos, t_start = [], time.perf_counter()
    for c in src_summary["channels"]:
        chan = c["chan_id"]
        info_path = out / "scores" / f"{chan}.json"
        if resume and info_path.exists():
            infos.append(json.loads(info_path.read_text()))
            continue
        info = {k: c[k] for k in ("chan_id", "spacecraft", "train_len", "test_len", "split_idx", "constant_train")}
        info["methods"] = {}
        if c["methods"].get("lstm", {}).get("status") != "ok":
            info["methods"]["lstm_dynamic"] = {"status": "failed", "error": "source lstm failed"}
        else:
            t0 = time.perf_counter()
            b = load_bundle(src_models, "lstm", chan)
            det = b.detector
            det.smoothing_span = int(over.get("smoothing_span", det.smoothing_span))
            det.warmup_fill = int(over.get("warmup_fill", 0))
            train, test = load_channel(raw_dir, chan, "train"), load_channel(raw_dir, chan, "test")
            split = c["split_idx"]
            train_s, test_s = b.scaler.transform(train), b.scaler.transform(test)
            val_raw, test_raw = validation_scores(det, train_s, split), det.score(test_s)
            val_scores, val_pred, _ = decide("lstm_dynamic", val_raw, train[split:, VALUE_COL],
                                             unit_scale=b.scaler.std, post=post, dynamic_cfg=dyn)
            scores, pred, eps = decide("lstm_dynamic", test_raw, test[:, VALUE_COL], unit_scale=b.scaler.std,
                                       post=post, dynamic_cfg=dyn)
            forecast = det.predict(test_s) * b.scaler.std + b.scaler.mean
            stored = pd.read_parquet(src_reports / "scores" / f"{chan}.parquet", columns=["lstm_forecast"])["lstm_forecast"]
            diff = float(np.nanmax(np.abs(forecast - stored.to_numpy())))
            table = pd.DataFrame({"timestep": np.arange(len(test)), "value": test[:, VALUE_COL],
                                  "lstm_dynamic_threshold": eps, "lstm_dynamic_score": scores,
                                  "lstm_dynamic_pred": pred, "lstm_dynamic_forecast": forecast,
                                  "label": label_vector(len(test), labels.loc[chan].anomaly_sequences)})
            pd.DataFrame({"timestep": np.arange(split, len(train)), "lstm_dynamic_score": val_scores}).to_parquet(
                out / "scores" / f"{chan}.val.parquet", index=False)
            table.to_parquet(out / "scores" / f"{chan}.parquet", index=False)
            thr = float(np.nanmedian(eps)) if np.isfinite(eps).any() else float("nan")
            info["methods"]["lstm_dynamic"] = {
                "status": "ok", "threshold": thr, "threshold_mode": "dynamic",
                "seconds": round(time.perf_counter() - t0, 2), "val_flag_rate": float(np.mean(val_pred)),
                "calibratable": not calibration_is_degenerate(val_scores, thr), "val_len": int(len(val_scores)),
                "forecast_max_abs_diff": diff}
        info_path.write_text(json.dumps(info, indent=1))
        infos.append(info)
    summary = {"run_name": cfg["run_name"], "methods": ["lstm_dynamic"], "derived_from": src_run,
               "n_channels": len(infos), "total_seconds": round(time.perf_counter() - t_start, 1),
               "channels": infos,
               "failures": [(c["chan_id"], m) for c in infos for m, v in c["methods"].items() if v["status"] != "ok"]}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
    worst = max((v.get("forecast_max_abs_diff", 0) for c in infos for v in c["methods"].values()), default=0)
    log.info("Derived run %s from %s: %d channels, max forecast difference vs source %.2e",
             cfg["run_name"], src_run, len(infos), worst)
    return out
