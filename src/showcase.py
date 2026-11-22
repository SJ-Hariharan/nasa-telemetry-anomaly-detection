"""Build the small committed artifact the app uses on a fresh clone (no data download, no training).

    python -m src.cli showcase        ->  reports/showcase/{index.json, channels/<chan>.parquet, models/...}

Contents: for a handful of channels, the archived test telemetry, labels and every method's scores, thresholds and
predictions from the seed-42 runs; plus saved telemanom and reference-LSTM models for two channels so the inference
page works offline. Channels are picked by a fixed rule (below), written to index.json with the reason - they
illustrate outcomes and are NOT a representative sample; the benchmark tables cover all 81 channels.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_channel
from src.utils.config import resolve_path
from src.utils.reproducibility import get_logger

log = get_logger("showcase")
SOURCES = {"baseline": "full", "isolation_forest": "full", "lstm": "full", "lstm_dynamic": "lstm_dynamic",
           "telemanom": "telemanom"}
MODEL_METHODS = {"telemanom": "telemanom", "lstm": "full"}   # method -> run whose saved models are copied
N_MODEL_CHANNELS = 2


def _per_channel(reports: Path, run: str, method: str) -> pd.DataFrame:
    pc = pd.read_csv(reports / run / "per_channel.csv")
    return pc[pc.method == method].set_index("chan_id")


def select_channels(reports: Path, primary: str = "telemanom") -> list[dict]:
    """Fixed rule: SMAP+MSL clean detections, the primary method's misses and false alarms, and the reference LSTM's
    worst false-alarm channels. Ties broken by channel id."""
    p = _per_channel(reports, SOURCES[primary], primary).reset_index().sort_values("chan_id")
    ref = _per_channel(reports, "full", "lstm").reset_index().sort_values("chan_id")
    picks: list[dict] = []

    def add(df, reason, k):
        for c in df["chan_id"]:
            if len([x for x in picks if x["reason"] == reason]) >= k:
                break
            if c not in {x["chan_id"] for x in picks}:
                picks.append({"chan_id": c, "reason": reason})

    for sc in ("SMAP", "MSL"):
        clean = p[(p.spacecraft == sc) & (p.event_fp == 0) & (p.event_tp > 0)].sort_values("event_tp", ascending=False, kind="stable")
        add(clean, f"{primary}: detected, no false alarm ({sc})", 1)
    add(p[p.event_fn > 0].sort_values("event_fn", ascending=False, kind="stable"), f"{primary}: missed event(s)", 2)
    add(p[p.event_fp > 0].sort_values("event_fp", ascending=False, kind="stable"), f"{primary}: false alarm(s)", 1)
    add(ref.sort_values("event_fp", ascending=False, kind="stable"), "reference lstm: most false alarms", 2)
    return picks


def build_showcase(cfg: dict, out: Path | None = None) -> Path:
    reports = resolve_path(cfg["output"]["reports_dir"])
    models = resolve_path(cfg["output"]["models_dir"])
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    out = out or reports.parent / "showcase"
    if out.exists():
        shutil.rmtree(out)
    (out / "channels").mkdir(parents=True)
    methods = [m for m, r in SOURCES.items() if (reports / r / "metrics.json").exists()]
    primary = next(m for m in ("telemanom", "lstm_dynamic", "lstm") if m in methods)
    picks = select_channels(reports, primary)
    thresholds, dyn_cfg = {}, None
    for pick in picks:
        chan = pick["chan_id"]
        test = load_channel(raw_dir, chan, "test")
        t = pd.DataFrame({"timestep": np.arange(len(test)), "value": test[:, 0].astype(np.float32)})
        for m in methods:
            run = SOURCES[m]
            s = pd.read_parquet(reports / run / "scores" / f"{chan}.parquet")
            if "label" not in t:
                t["label"] = s["label"].astype(np.int8)
            t[f"{m}_score"] = s[f"{m}_score"].astype(np.float32)
            t[f"{m}_pred"] = s[f"{m}_pred"].astype(np.int8)
            if f"{m}_threshold" in s:
                t[f"{m}_threshold"] = s[f"{m}_threshold"].astype(np.float32)
            info = {c["chan_id"]: c for c in json.loads((reports / run / "training_summary.json").read_text())["channels"]}
            thresholds.setdefault(chan, {})[m] = {"threshold": info[chan]["methods"][m]["threshold"],
                                                  "calibratable": info[chan]["methods"][m].get("calibratable")}
            if m in ("telemanom", "lstm_dynamic") and dyn_cfg is None:
                dyn_cfg = json.loads((reports / run / "config_used.json").read_text()).get("dynamic_threshold")
        for i in range(test.shape[1] - 1):           # command flags, so the inference sample is the real input
            t[f"cmd_{i}"] = test[:, i + 1].astype(np.int8)
        t.to_parquet(out / "channels" / f"{chan}.parquet", index=False, compression="zstd")
        pick.update(spacecraft=str(_per_channel(reports, "full", "lstm").loc[chan, "spacecraft"]), n=len(t),
                    n_commands=test.shape[1] - 1)
    model_chans = []
    def calibrated(c):
        return all(v.get("calibratable") is not False for v in thresholds[c].values())

    for sc in ("SMAP", "MSL"):                       # one calibratable channel per mission with committed models
        c = next((p["chan_id"] for p in picks if p["spacecraft"] == sc and calibrated(p["chan_id"])), None)
        if c:
            model_chans.append(c)
    for m, run in MODEL_METHODS.items():
        for c in model_chans[:N_MODEL_CHANNELS]:
            src = models / run / m / c
            if src.exists():
                shutil.copytree(src, out / "models" / m / c)
    index = {"note": "Channels chosen by a fixed rule to illustrate outcomes; NOT a representative sample.",
             "sources": {m: SOURCES[m] for m in methods}, "channels": picks, "thresholds": thresholds,
             "dynamic_threshold": dyn_cfg, "model_channels": model_chans[:N_MODEL_CHANNELS],
             "model_runs": {m: r for m, r in MODEL_METHODS.items() if m in methods}}
    (out / "index.json").write_text(json.dumps(index, indent=1, default=float))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log.info("Showcase: %d channels, %.1f MB in reports/%s", len(picks), size / 1e6, out.name)
    return out
