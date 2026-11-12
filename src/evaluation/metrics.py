"""Point-wise and event-wise evaluation.

Why not accuracy? About 12.5% of test points are anomalous overall, and far fewer in many channels.
A detector that always says "normal" scores ~87.5% accuracy while finding nothing.
Precision/recall/F1 and FPR/FNR describe the two kinds of mistakes separately.

Two views of the same predictions:
* point-wise: every timestep is one sample. Long anomalies dominate; one 2,000-point anomaly
  counts 2,000 times more than a 1-point spike.
* event-wise (as in Hundman et al. 2018): each labelled anomaly sequence is ONE event.
  - true event detected  (TP) if at least one predicted point falls inside it
  - true event missed    (FN) otherwise
  - false alarm event    (FP) = a contiguous predicted run that overlaps no labelled sequence
  This treats overlapping/adjacent anomalous timesteps as one incident instead of many
  independent ones. Note: "any overlap" is lenient - see docs/methodology.md. We do NOT use
  "point-adjust" (marking a whole true segment detected), which is known to inflate point F1.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.thresholds import runs


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def point_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: labels {y_true.shape} vs predictions {y_pred.shape}")
    return {"tp": int((y_true & y_pred).sum()), "fp": int((~y_true & y_pred).sum()),
            "tn": int((~y_true & ~y_pred).sum()), "fn": int((y_true & ~y_pred).sum())}


def point_metrics(c: dict) -> dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {**c, "precision": p, "recall": r, "f1": _safe_div(2 * p * r, p + r),
            "fpr": _safe_div(fp, fp + tn), "fnr": _safe_div(fn, fn + tp),
            "accuracy": _safe_div(tp + tn, tp + fp + tn + fn)}


def event_counts(sequences: list[list[int]], y_pred: np.ndarray) -> dict:
    """Event-level TP/FN/FP plus detection latency (points from event start to first hit)."""
    y_pred = np.asarray(y_pred).astype(bool)
    tp = fn = 0
    latencies: list[int] = []
    truth = np.zeros(len(y_pred), dtype=bool)
    for s, e in sequences:
        truth[s : e + 1] = True
        hits = np.flatnonzero(y_pred[s : e + 1])
        if hits.size:
            tp += 1
            latencies.append(int(hits[0]))
        else:
            fn += 1
    fp = sum(1 for s, e in runs(y_pred) if not truth[s : e + 1].any())
    return {"event_tp": tp, "event_fn": fn, "event_fp": fp, "latencies": latencies}


def event_metrics(c: dict) -> dict:
    tp, fn, fp = c["event_tp"], c["event_fn"], c["event_fp"]
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    lat = c.get("latencies", [])
    return {"event_tp": tp, "event_fn": fn, "event_fp": fp, "event_precision": p, "event_recall": r,
            "event_f1": _safe_div(2 * p * r, p + r),
            "median_latency": float(np.median(lat)) if lat else None}


def threshold_free(y_true: np.ndarray, norm_scores: np.ndarray) -> dict:
    """ROC-AUC and PR-AUC (average precision) of scores, independent of any threshold.

    ``norm_scores`` = score / channel threshold, so values from different channels are on a
    comparable scale (1.0 = that channel's decision boundary). Warm-up NaNs count as 0.
    """
    s = np.nan_to_num(np.asarray(norm_scores, dtype=np.float64), nan=0.0, posinf=1e12)
    y = np.asarray(y_true).astype(int)
    if y.min() == y.max():
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}
