"""Cross-run benchmark table: every method under the same evaluation protocol, with seed variation.

Input : reports/runs/<run>/{metrics.json, config_used.json, training_summary.json, calibration/calibration.json}
Output: reports/benchmark/benchmark.json, benchmark.md, runs.csv, operating_points.csv

Protocol (stated in benchmark.md and fixed before the new runs were made):
* all runs evaluate the same 81 channels with the same micro point and any-overlap event metrics, no point-adjust;
* each method's *headline* uses its predeclared settings - validation-quantile threshold (q = 0.995) for the static
  methods, the published Telemanom parameters (p = 0.13) for the dynamic ones. No test label chooses anything;
* the ranking criterion, chosen before looking at the new results, is mean event F1 over seeds; false alarms,
  recall and point F1 are always shown next to it;
* leave-channels-out (LCO) rows are a second, clearly labelled operating point (rule chosen on other channels' labels).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.decision import threshold_mode
from src.utils.reproducibility import get_logger

log = get_logger("compare")
DEFAULT_RUNS = ("full", "full_seed1", "full_seed2", "lstm_dynamic", "lstm_dynamic_seed1", "lstm_dynamic_seed2",
                "telemanom", "telemanom_seed1", "telemanom_seed2")
KEYS = ("precision", "recall", "f1", "fpr", "event_precision", "event_recall", "event_f1", "event_tp", "event_fn",
        "event_fp", "false_alarms_per_10k", "median_latency", "median_decision_latency", "roc_auc_macro",
        "pr_auc_macro")
PROTOCOL = """Protocol: same 81 channels and metrics for every method (micro point metrics; an event is found if any
predicted point overlaps it; a false-alarm event is a predicted run overlapping no label; no point-adjust).
Headline settings are predeclared: validation-quantile threshold (q = 0.995) for static methods, published Telemanom
parameters (p = 0.13) for dynamic ones. Ranking criterion, fixed in advance: mean event F1 over seeds.
Decision latency includes the dynamic threshold's batch delay (up to 69 points)."""
DESCRIPTION = {
    "baseline": "Rolling z-score, validation-quantile threshold",
    "isolation_forest": "Isolation Forest on trailing-window features, validation-quantile threshold",
    "lstm": "Reference LSTM forecaster (window 100, 2x64), validation-quantile threshold",
    "lstm_dynamic": "Same saved reference LSTM models + Telemanom dynamic threshold (ablation, no retraining)",
    "telemanom": "Telemanom-aligned LSTM (l_s 250, 2x80, 10-step) + Telemanom dynamic threshold",
}


def _num(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


COMMON_SUBSET_RUN = "full"   # its calibratable channels define ONE subset used for every method


def common_subset(reports_dir: Path) -> list[str] | None:
    """Channels with a calibratable static threshold in the reference run (fixed set, same for all methods)."""
    pc = Path(reports_dir) / COMMON_SUBSET_RUN / "per_channel.csv"
    if not pc.exists():
        return None
    d = pd.read_csv(pc)
    return sorted(set(d.chan_id) - set(d.loc[~d.calibratable.astype(bool), "chan_id"]))


def subset_metrics(per_channel: pd.DataFrame, method: str, chans: list[str]) -> dict:
    """Micro metrics over a channel subset, from the per-channel counts in per_channel.csv."""
    from src.evaluation.metrics import event_metrics, point_metrics

    d = per_channel[(per_channel.method == method) & per_channel.chan_id.isin(chans)]
    pm_ = point_metrics({k: int(d[k].sum()) for k in ("tp", "fp", "tn", "fn")})
    em = event_metrics({k: int(d[k].sum()) for k in ("event_tp", "event_fn", "event_fp")})
    n = pm_["tp"] + pm_["fp"] + pm_["tn"] + pm_["fn"]
    return {**pm_, **em, "n_channels": len(d), "false_alarms_per_10k": 1e4 * em["event_fp"] / n if n else None,
            "median_latency": None, "median_decision_latency": None,
            "roc_auc_macro": float(d["roc_auc"].mean()), "pr_auc_macro": float(d["pr_auc"].mean())}


def collect(reports_dir: Path, runs=DEFAULT_RUNS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, ops, checks = [], [], {}
    common = common_subset(reports_dir)
    seen: dict[tuple[str, int], str] = {}
    for run in runs:
        d = Path(reports_dir) / run
        if not (d / "metrics.json").exists():
            log.warning("Skipping %s: no metrics.json", run)
            continue
        met = json.loads((d / "metrics.json").read_text())
        cfg = json.loads((d / "config_used.json").read_text())
        summ = json.loads((d / "training_summary.json").read_text())
        seed = int(cfg["seed"])
        events = pd.read_csv(d / "events.csv") if (d / "events.csv").exists() else None
        for m, r in met["results"].items():
            key = (m, seed)
            if key in seen:      # e.g. baseline/IF re-run inside the telemanom run: a reproducibility check
                first = next(x for x in rows if x["method"] == m and x["seed"] == seed and x["subset"] == "overall")
                same = all(np.isclose(first[k], r["overall"][k], equal_nan=True) for k in ("f1", "event_f1", "event_fp"))
                checks[f"{m} seed {seed}: {seen[key]} vs {run}"] = bool(same)
                continue
            seen[key] = run
            ch = [c["methods"][m] for c in summ["channels"] if c["methods"].get(m, {}).get("status") == "ok"]
            if common:
                r = {**r, "calibratable": subset_metrics(pd.read_csv(d / "per_channel.csv"), m, common)}
            for subset in ("overall", "calibratable"):
                if r.get(subset):
                    if subset == "overall":
                        _backfill(r[subset], events, m, subset, met)
                    rows.append({"method": m, "run": run, "seed": seed, "subset": subset,
                                 "threshold_mode": threshold_mode(m),
                                 "n_channels": r[subset]["n_channels"],
                                 "n_uncalibratable": met["n_channels_evaluated"] - met["n_calibratable_channels"],
                                 "median_val_flag_rate": float(np.median([c["val_flag_rate"] for c in ch])) if ch else None,
                                 **{k: _num(r[subset].get(k)) for k in KEYS}})
            cal = d / "calibration" / "calibration.json"
            if cal.exists():
                c = json.loads(cal.read_text())
                src = c.get("leave_channels_out", {}).get(m) or c.get("dynamic_threshold_lco", {}).get(m)
                if src:
                    for name, label in (("default", "predeclared"), ("lco", "leave-channels-out")):
                        ops.append({"method": m, "run": run, "seed": seed, "operating_point": label,
                                    **{k: src["all"][name][k] for k in src["all"][name]}})
    return pd.DataFrame(rows), pd.DataFrame(ops), checks


def _backfill(res: dict, events: pd.DataFrame | None, method: str, subset: str, met: dict) -> None:
    """Runs evaluated before these fields existed (the committed reference runs): derive them from the same run's
    committed counts and events.csv. Static methods decide immediately, so decision latency = latency."""
    n = sum(res.get(k, 0) for k in ("tp", "fp", "tn", "fn"))
    if res.get("false_alarms_per_10k") is None and n:
        res["false_alarms_per_10k"] = 1e4 * res["event_fp"] / n
    if res.get("median_decision_latency") is None and events is not None and threshold_mode(method) == "static":
        e = events[(events.method == method) & events.detected]
        if subset == "calibratable":
            e = e[~e.chan_id.isin(met.get("uncalibratable_channels", {}))]
        if len(e):
            res["median_latency"] = res.get("median_latency") or float(e["latency"].median())
            res["median_decision_latency"] = float(e["latency"].median())


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    o = df[df.subset == "overall"]
    agg = o.groupby("method")[list(KEYS)].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{k}_{s}" for k, s in agg.columns]
    agg["n_seeds"] = o.groupby("method")["seed"].nunique()
    agg["threshold_mode"] = o.groupby("method")["threshold_mode"].first()
    agg["median_val_flag_rate"] = o.groupby("method")["median_val_flag_rate"].median()
    agg["description"] = [DESCRIPTION.get(m, m) for m in agg.index]
    return agg.sort_values("event_f1_mean", ascending=False).reset_index()


def _pm(row, k, fmt="{:.3f}"):
    m, s = row[f"{k}_mean"], row[f"{k}_std"]
    if m is None or not np.isfinite(m):
        return "-"
    return fmt.format(m) + ("" if not np.isfinite(s) or row["n_seeds"] < 2 else " ± " + fmt.format(s))


def write_benchmark(reports_dir: Path, out_dir: Path, runs=DEFAULT_RUNS) -> dict:
    df, ops, checks = collect(reports_dir, runs)
    if df.empty:
        raise FileNotFoundError("No evaluated runs found - run training + evaluation first")
    summ = summarise(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "runs.csv", index=False)
    ops.to_csv(out_dir / "operating_points.csv", index=False)
    lines = ["# Benchmark comparison (NASA SMAP/MSL, 81 channels, test labels used for scoring only)", "",
             PROTOCOL,
             "", "## Headline (predeclared settings, mean ± std over seeds)", "",
             "| Method | Threshold | Seeds | Event F1 | Event recall | Events found | Event precision | False-alarm events "
             "| False alarms / 10k pts | Point F1 | Point precision | Point recall | Median decision latency (pts) | "
             "PR-AUC (macro) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r['method']} | {r['threshold_mode']} | {r['n_seeds']} | {_pm(r, 'event_f1')} | "
                     f"{_pm(r, 'event_recall')} | {_pm(r, 'event_tp', '{:.0f}')}/104 | {_pm(r, 'event_precision')} | "
                     f"{_pm(r, 'event_fp', '{:.0f}')} | {_pm(r, 'false_alarms_per_10k', '{:.1f}')} | {_pm(r, 'f1')} | "
                     f"{_pm(r, 'precision')} | {_pm(r, 'recall')} | {_pm(r, 'median_decision_latency', '{:.0f}')} | "
                     f"{_pm(r, 'pr_auc_macro')} |")
    lines += ["", "Methods: " + "; ".join(f"**{m}** - {DESCRIPTION.get(m, m)}" for m in summ["method"]), ""]
    if not ops.empty:
        lines += ["## Operating points (seed-42 run of each method; LCO = rule chosen on other channel folds)", "",
                  "| Method | Operating point | Event recall | Event precision | Event F1 | False-alarm events | Point F1 |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in ops[ops.seed == 42].iterrows():
            lines.append(f"| {r['method']} | {r['operating_point']} | {r['event_recall']:.3f} | "
                         f"{r['event_precision']:.3f} | {r['event_f1']:.3f} | {int(r['event_fp'])} | {r['f1']:.3f} |")
        lines.append("")
    if checks:
        lines += ["## Reproducibility checks", ""] + [f"* {k}: {'identical' if v else 'DIFFERENT'}" for k, v in checks.items()]
    (out_dir / "benchmark.md").write_text("\n".join(lines) + "\n")
    result = {"runs": sorted(df["run"].unique().tolist()), "summary": summ.to_dict("records"),
              "reproducibility_checks": checks,
              "ranking_criterion": "mean event F1 over seeds at predeclared settings (no test-label tuning)"}
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=1, default=float))
    log.info("Benchmark written to reports/%s", out_dir.name)
    return result
