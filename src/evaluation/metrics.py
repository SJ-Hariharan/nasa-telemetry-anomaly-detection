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


def event_counts(sequences: list[list[int]], y_pred: np.ndarray, release: np.ndarray | None = None) -> dict:
    """Event-level TP/FN/FP plus detection latency (points from event start to first hit).

    ``release[i]`` = the timestep at which the decision about position i was available (default: i itself). With
    it, ``decision_latencies`` = release[first hit] - event start, which includes any batching delay of the method.
    """
    y_pred = np.asarray(y_pred).astype(bool)
    tp = fn = 0
    latencies: list[int] = []
    decision: list[int] = []
    truth = np.zeros(len(y_pred), dtype=bool)
    for s, e in sequences:
        truth[s : e + 1] = True
        hits = np.flatnonzero(y_pred[s : e + 1])
        if hits.size:
            tp += 1
            latencies.append(int(hits[0]))
            decision.append(int(release[s + hits[0]] - s) if release is not None else int(hits[0]))
        else:
            fn += 1
    fp = sum(1 for s, e in runs(y_pred) if not truth[s : e + 1].any())
    return {"event_tp": tp, "event_fn": fn, "event_fp": fp, "latencies": latencies, "decision_latencies": decision}


def event_metrics(c: dict) -> dict:
    tp, fn, fp = c["event_tp"], c["event_fn"], c["event_fp"]
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    lat, dec = c.get("latencies", []), c.get("decision_latencies", [])
    return {"event_tp": tp, "event_fn": fn, "event_fp": fp, "event_precision": p, "event_recall": r,
            "event_f1": _safe_div(2 * p * r, p + r),
            "median_latency": float(np.median(lat)) if lat else None,
            "median_decision_latency": float(np.median(dec)) if dec else None}


def threshold_free(y_true: np.ndarray, norm_scores: np.ndarray) -> dict:
    """ROC-AUC and PR-AUC (average precision) of scores, independent of any threshold.

    Warm-up positions (NaN score) are EXCLUDED: counting them as score 0 would add thousands of
    trivially easy negatives. Returns None when only one class is present.

    Use per channel (raw scores). Pooling ``score / threshold`` across channels is also computed by
    the evaluation, but it is only a secondary figure: dividing by a per-channel threshold does not
    make scores from different channels rank-comparable (review finding D2).
    """
    s = np.asarray(norm_scores, dtype=np.float64)
    y = np.asarray(y_true).astype(int)
    ok = np.isfinite(s)
    s, y = np.clip(s[ok], None, 1e12), y[ok]
    if y.size == 0 or y.min() == y.max():
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def macro_auc(values: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of per-channel AUCs with a bootstrap-over-channels 95% interval (None entries skipped)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return {"mean": float(v.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(v.size)}
