"""Evaluate a finished run: metrics, comparison table, error analysis and figures.

Input : reports/runs/<run>/scores/*.parquet and training_summary.json (written by src.pipeline)
Output: metrics.json, comparison.csv/.md, per_channel.csv, events.csv, false_alarms.csv,
        sensitivity.csv and figures/*.png in the same run directory.

Aggregation is *micro*: TP/FP/TN/FN (and event counts) are summed over channels before computing
precision/recall, so every timestep (or event) counts equally regardless of its channel.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import (event_counts, event_metrics, point_counts, point_metrics,  # noqa: E402
                                    threshold_free)
from src.models.persistence import METHODS  # noqa: E402
from src.models.thresholds import apply_threshold, runs  # noqa: E402
from src.utils.reproducibility import get_logger  # noqa: E402

log = get_logger("evaluate")
SENSITIVITY_MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
# Multiplying a threshold only makes sense for scores on an open-ended scale starting at 0.
# Isolation Forest scores (-score_samples) live in a narrow band around 0.5, so x0.5 flags every
# point and x1.5 flags none; its rows are kept in sensitivity.csv but marked and left off the plot.
SENSITIVITY_NOT_COMPARABLE = {"isolation_forest"}


def _sum(dicts: list[dict], keys) -> dict:
    out = {k: sum(d[k] for d in dicts) for k in keys}
    if "latencies" in dicts[0]:
        out["latencies"] = [x for d in dicts for x in d["latencies"]]
    return out


def evaluate_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "training_summary.json").read_text())
    ch_info = {c["chan_id"]: c for c in summary["channels"]}
    seqs, classes = _load_label_meta(run_dir, list(ch_info))
    methods = [m for m in METHODS if m in summary["methods"]]
    # Compare methods only on channels where ALL of them succeeded (same evaluation set).
    common = [c for c, i in ch_info.items() if all(i["methods"].get(m, {}).get("status") == "ok" for m in methods)]
    excluded = sorted(set(ch_info) - set(common))
    if excluded:
        log.warning("Excluded from comparison (a method failed): %s", excluded)

    per_channel, event_rows, fa_rows = [], [], []
    pooled = {m: {"y": [], "s": []} for m in methods}
    counts = {m: {"point": [], "event": [], "SMAP": [], "MSL": [], "SMAP_ev": [], "MSL_ev": []} for m in methods}
    sens = {m: {k: {"p": [], "e": []} for k in SENSITIVITY_MULTIPLIERS} for m in methods}

    for chan in common:
        t = pd.read_parquet(run_dir / "scores" / f"{chan}.parquet")
        y, sc = t["label"].to_numpy(), ch_info[chan]["spacecraft"]
        for m in methods:
            thr = ch_info[chan]["methods"][m]["threshold"]
            pc, ec = point_counts(y, t[f"{m}_pred"]), event_counts(seqs[chan], t[f"{m}_pred"].to_numpy())
            counts[m]["point"].append(pc); counts[m]["event"].append(ec)
            counts[m][sc].append(pc); counts[m][f"{sc}_ev"].append(ec)
            pooled[m]["y"].append(y); pooled[m]["s"].append(t[f"{m}_score"].to_numpy() / thr)
            per_channel.append({"chan_id": chan, "spacecraft": sc, "method": m, "threshold": thr,
                                **point_metrics(pc), **event_metrics(ec)})
            # Post-hoc diagnostic ONLY: how would results move if the calibrated threshold were scaled?
            for k in SENSITIVITY_MULTIPLIERS:
                p = apply_threshold(t[f"{m}_score"].to_numpy(), thr * k)
                sens[m][k]["p"].append(point_counts(y, p)); sens[m][k]["e"].append(event_counts(seqs[chan], p))
            # Error analysis rows
            pred = t[f"{m}_pred"].to_numpy()
            norm = t[f"{m}_score"].to_numpy() / thr
            for (s, e), cls in zip(seqs[chan], classes[chan]):
                hits = np.flatnonzero(pred[s : e + 1])
                event_rows.append({"chan_id": chan, "spacecraft": sc, "method": m, "start": s, "end": e,
                                   "length": e - s + 1, "class": cls, "detected": bool(hits.size),
                                   "latency": int(hits[0]) if hits.size else None,
                                   "peak_norm_score": float(np.nanmax(norm[s : e + 1])) if np.isfinite(norm[s : e + 1]).any() else None})
            for s, e in runs(pred):
                if not y[s : e + 1].any():
                    fa_rows.append({"chan_id": chan, "spacecraft": sc, "method": m, "start": s, "end": e,
                                    "length": e - s + 1, "peak_norm_score": float(np.nanmax(norm[s : e + 1]))})

    P_KEYS, E_KEYS = ("tp", "fp", "tn", "fn"), ("event_tp", "event_fn", "event_fp")
    results = {}
    for m in methods:
        r = {"overall": {**point_metrics(_sum(counts[m]["point"], P_KEYS)),
                         **event_metrics(_sum(counts[m]["event"], E_KEYS)),
                         **threshold_free(np.concatenate(pooled[m]["y"]), np.concatenate(pooled[m]["s"]))}}
        for sc in ("SMAP", "MSL"):
            if counts[m][sc]:
                r[sc] = {**point_metrics(_sum(counts[m][sc], P_KEYS)), **event_metrics(_sum(counts[m][f"{sc}_ev"], E_KEYS))}
        r["sensitivity"] = [{"multiplier": k, "scale_comparable": m not in SENSITIVITY_NOT_COMPARABLE, **{f"point_{x}": v for x, v in point_metrics(_sum(sens[m][k]["p"], P_KEYS)).items()
                                                  if x in ("precision", "recall", "f1", "fpr")},
                             **{x: v for x, v in event_metrics(_sum(sens[m][k]["e"], E_KEYS)).items()
                                if x in ("event_precision", "event_recall", "event_f1")}}
                            for k in SENSITIVITY_MULTIPLIERS]
        results[m] = r

    events = pd.DataFrame(event_rows)
    comp = pd.DataFrame([{"method": m, **{k: results[m]["overall"][k] for k in
                          ("precision", "recall", "f1", "fpr", "fnr", "accuracy", "event_precision",
                           "event_recall", "event_f1", "event_tp", "event_fn", "event_fp", "roc_auc", "pr_auc")}}
                         for m in methods])
    by_class = (events.groupby(["method", "class"])["detected"].mean().unstack().round(3)
                if len(events) else pd.DataFrame())
    metrics = {"run_name": summary["run_name"], "n_channels_evaluated": len(common), "excluded_channels": excluded,
               "training_failures": summary["failures"], "results": results,
               "event_recall_by_class": by_class.to_dict() if len(by_class) else {},
               "comparison_markdown": _markdown(comp)}
    comp.to_csv(run_dir / "comparison.csv", index=False)
    (run_dir / "comparison.md").write_text(metrics["comparison_markdown"])
    pd.DataFrame(per_channel).to_csv(run_dir / "per_channel.csv", index=False)
    events.to_csv(run_dir / "events.csv", index=False)
    pd.DataFrame(fa_rows, columns=["chan_id", "spacecraft", "method", "start", "end", "length",
                                   "peak_norm_score"]).to_csv(run_dir / "false_alarms.csv", index=False)
    pd.DataFrame([{"method": m, **row} for m in methods for row in results[m]["sensitivity"]]).to_csv(
        run_dir / "sensitivity.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    _figures(run_dir, comp, results, methods, common)
    log.info("Evaluation written to %s", run_dir)
    return metrics


def _load_label_meta(run_dir: Path, chans: list[str]):
    from src.data.loader import load_labels
    from src.utils.config import resolve_path

    cfg = json.loads((run_dir / "config_used.json").read_text())
    labels = load_labels(resolve_path(cfg["data"]["raw_dir"])).set_index("chan_id")
    return ({c: labels.loc[c, "anomaly_sequences"] for c in chans},
            {c: labels.loc[c, "classes"] for c in chans})


def _markdown(comp: pd.DataFrame) -> str:
    cols = ["method", "precision", "recall", "f1", "fpr", "accuracy", "event_precision", "event_recall",
            "event_f1", "event_tp", "event_fn", "event_fp", "roc_auc", "pr_auc"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in comp.iterrows():
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _figures(run_dir: Path, comp: pd.DataFrame, results: dict, methods: list[str], chans: list[str]) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    # 1. metric comparison
    keys = ["precision", "recall", "f1", "event_precision", "event_recall", "event_f1"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ax.bar(np.arange(len(keys)) + i * width, [results[m]["overall"][k] for k in keys], width, label=m)
    ax.set_xticks(np.arange(len(keys)) + width * (len(methods) - 1) / 2, keys, rotation=20)
    ax.set_ylim(0, 1); ax.set_title(f"Test metrics ({len(chans)} channels, thresholds from validation)")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "metric_comparison.png", dpi=120); plt.close(fig)
    # 2. threshold sensitivity (post-hoc)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in [m for m in methods if m not in SENSITIVITY_NOT_COMPARABLE]:
        s = pd.DataFrame(results[m]["sensitivity"])
        axes[0].plot(s["point_recall"], s["point_precision"], "o-", label=m)
        axes[1].plot(s["multiplier"], s["event_f1"], "o-", label=m)
    axes[0].set_xlabel("point recall"); axes[0].set_ylabel("point precision"); axes[0].set_title("Precision vs recall as threshold scales")
    axes[1].set_xscale("log"); axes[1].set_xlabel("threshold multiplier (1 = calibrated)"); axes[1].set_ylabel("event F1")
    axes[1].set_title("Post-hoc sensitivity (NOT used for tuning)")
    fig.text(0.5, 0.01, "Isolation Forest omitted: its bounded score scale makes threshold multipliers meaningless",
             ha="center", fontsize=8)
    for a in axes:
        a.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "threshold_sensitivity.png", dpi=120); plt.close(fig)
    # 3. timelines for up to 4 channels
    for chan in chans[:4]:
        t = pd.read_parquet(run_dir / "scores" / f"{chan}.parquet")
        fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(11, 1.9 * (len(methods) + 1)), sharex=True)
        axes[0].plot(t["timestep"], t["value"], lw=0.6, color="k")
        axes[0].fill_between(t["timestep"], t["value"].min(), t["value"].max(), where=t["label"] == 1,
                             color="red", alpha=0.2, label="labelled anomaly")
        axes[0].set_title(f"{chan}: test telemetry"); axes[0].legend(loc="upper right")
        info = json.loads((run_dir / "training_summary.json").read_text())
        thr = {c["chan_id"]: c for c in info["channels"]}[chan]["methods"]
        for ax, m in zip(axes[1:], methods):
            ax.plot(t["timestep"], t[f"{m}_score"], lw=0.6)
            ax.axhline(thr[m]["threshold"], color="orange", ls="--", lw=1)
            ax.fill_between(t["timestep"], 0, 1, where=t[f"{m}_pred"] == 1, transform=ax.get_xaxis_transform(),
                            color="blue", alpha=0.15)
            ax.set_ylabel(m, fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / f"timeline_{chan}.png", dpi=110); plt.close(fig)
