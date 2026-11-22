"""Historical replay: what a detector would have shown if archived samples arrived one at a time.

HISTORICAL REPLAY - NOT LIVE TELEMETRY. The samples are archived benchmark test data.

Honesty rules:
* Only information available at time t is used. Scores are causal (the score at i uses samples <= i), and for the
  dynamic threshold only windows that have finished by t are evaluated (``decided_until=t``), so a flag appears at
  the moment the method would really have released it - up to one 70-sample batch after the flagged sample.
* Labels are never used to decide anything; they are shown next to each alert *after the fact* ("hindsight").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.decision import DYNAMIC_METHODS, postprocess
from src.models.dynamic_threshold import dynamic_threshold, release_times
from src.models.thresholds import runs

REPLAY_LABEL = "Historical replay — not live telemetry"


@dataclass
class ReplayState:
    t: int                       # samples received so far
    decided: int                 # positions [0, decided) have a final decision
    pred: np.ndarray             # decisions known at time t (length t; undecided positions are 0)
    alerts: pd.DataFrame         # one row per alert: start, end, raised_at, delay, hindsight


def replay_state(table: pd.DataFrame, method: str, t: int, dynamic_cfg: dict | None = None,
                 post: dict | None = None) -> ReplayState:
    n = len(table)
    t = int(np.clip(t, 0, n))
    scores = table[f"{method}_score"].to_numpy(dtype=np.float64)
    labels = table["label"].to_numpy() if "label" in table else np.zeros(n, dtype=np.int8)
    if method in DYNAMIC_METHODS:
        pred_full, _ = dynamic_threshold(scores, table["value"].to_numpy(), dynamic_cfg, decided_until=t)
        pred_full = postprocess(pred_full, post)
        rel = release_times(scores, dynamic_cfg)
        decided = int(np.sum(rel < t))
    else:
        pred_full = table[f"{method}_pred"].to_numpy().copy()      # causal scores x fixed threshold: final at once
        pred_full[t:] = 0
        rel = np.arange(n)
        decided = t
    pred = pred_full[:t].astype(np.int8)
    rows = []
    for s, e in runs(pred):
        raised = int(rel[s])
        rows.append({"start": s, "end": e, "length": e - s + 1, "raised_at": raised, "delay": raised - s,
                     "hindsight": "overlaps a labelled anomaly" if labels[s : e + 1].any() else "false alarm"})
    return ReplayState(t, decided, pred, pd.DataFrame(rows, columns=["start", "end", "length", "raised_at", "delay",
                                                                      "hindsight"]))
