"""Calibration studies on a finished run (no retraining; uses the saved validation and test scores).

1. ``calibration_uncertainty`` (review finding D4)
   Every per-channel threshold is a 99.5% quantile of only 62-862 validation scores, i.e. it rests on
   fewer than 5 expected exceedances. We bootstrap the validation points of each channel, recompute its
   threshold, re-apply it to the (unchanged) test scores and recompute the headline metrics. The spread
   of the metrics across bootstrap replicates is the uncertainty that comes from estimating thresholds on
   small samples. No label is used to compute or choose any threshold here; labels only score the result.

2. ``leave_channels_out`` (the review's recommended next experiment)
   Question: is the default rule (99.5% quantile) a good *rule* for channels it has never seen, and would
   a different rule from a small pre-specified family generalise better? Protocol (fixed before running):
     * rule family, all computed from a channel's own validation scores only:
           thr = median + m * (Q_q - median),   q in QUANTILES, m in MARGINS
       (m = 1 reproduces the run's threshold; the form is location-invariant so it is meaningful for
       Isolation Forest scores, which live in a narrow band around 0.5)
     * K = 5 folds of channels, stratified by spacecraft, fixed seed
     * for each fold: pick the rule that maximises micro EVENT F1 on the other 4 folds' channels
       (ties -> smaller m, then smaller q), then apply it to the held-out fold's channels
     * report the held-out predictions of all folds together, next to the default rule
   The held-out fold's labels never influence the rule applied to it (``test_lco_rule_ignores_heldout_labels``).
   Labels of the *other* channels are used, as in any cross-validation; the result is therefore an estimate
   of how a rule chosen this way would perform on new channels, not a re-tuned headline number.

3. ``seed_summary``: mean / std / min / max of headline metrics over runs that differ only in the seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import event_counts, event_metrics, point_counts, point_metrics
from src.models.persistence import METHODS
from src.utils.reproducibility import get_logger

log = get_logger("calibration")
QUANTILES = (0.99, 0.995, 0.999, 1.0)
MARGINS = (1.0, 1.5, 2.0, 3.0, 5.0)
DEFAULT_RULE = (0.995, 1.0)
HEADLINE = ("precision", "recall", "f1", "fpr", "event_precision", "event_recall", "event_f1")


# ---------------------------------------------------------------- data access
def load_channel_records(run_dir: Path) -> list[dict]:
    """One record per channel: labels, event sequences, and per-method validation/test scores."""
    from src.evaluation.evaluate import _load_label_meta

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "training_summary.json").read_text())
    methods = [m for m in METHODS if m in summary["methods"]]
    infos = [c for c in summary["channels"] if all(c["methods"].get(m, {}).get("status") == "ok" for m in methods)]
    seqs, _ = _load_label_meta(run_dir, [c["chan_id"] for c in infos])
    records = []
    for c in infos:
        chan = c["chan_id"]
        val_path = run_dir / "scores" / f"{chan}.val.parquet"
        if not val_path.exists():
            raise FileNotFoundError(f"{val_path} missing: rerun training with the current pipeline")
        t, v = pd.read_parquet(run_dir / "scores" / f"{chan}.parquet"), pd.read_parquet(val_path)
        records.append({"chan_id": chan, "spacecraft": c["spacecraft"], "y": t["label"].to_numpy(),
                        "seqs": seqs[chan], "calibratable": not c.get("constant_train", False) and
                        all(c["methods"][m].get("calibratable", True) for m in methods),
                        "threshold": {m: c["methods"][m]["threshold"] for m in methods},
                        "test": {m: t[f"{m}_score"].to_numpy() for m in methods},
                        "val": {m: v[f"{m}_score"].to_numpy() for m in methods}})
    return records


# ---------------------------------------------------------------- shared helpers
def rule_threshold(val_scores: np.ndarray, q: float, m: float) -> float:
    s = np.asarray(val_scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    med, qv = float(np.median(s)), float(np.quantile(s, q))
    thr = qv + (m - 1.0) * (qv - med)           # = median + m (Q - median); written so m = 1 returns Q exactly
    return thr if thr > 0 else float(np.finfo(np.float32).eps)      # same floor as calibrate_threshold


def _counts(rec: dict, method: str, thr: float) -> tuple[dict, dict]:
    pred = np.nan_to_num(rec["test"][method], nan=-np.inf) > thr
    return point_counts(rec["y"], pred), event_counts(rec["seqs"], pred.astype(np.int8))


def micro_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    pc = {k: sum(p[k] for p, _ in pairs) for k in ("tp", "fp", "tn", "fn")}
    ec = {k: sum(e[k] for _, e in pairs) for k in ("event_tp", "event_fn", "event_fp")}
    out = {**point_metrics(pc), **event_metrics(ec)}
    return {k: out[k] for k in HEADLINE + ("event_tp", "event_fn", "event_fp")}


# ---------------------------------------------------------------- 1. threshold-estimation uncertainty
def block_bootstrap_indices(n: int, n_boot: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap: resample contiguous blocks (wrapping at the end) so that the strong
    autocorrelation of validation scores (median lag-1 0.80-0.98) is preserved. block_len = 1 is the iid bootstrap."""
    L = max(1, min(block_len, n))
    k = -(-n // L)                                                     # blocks per replicate (ceil)
    starts = rng.integers(0, n, size=(n_boot, k))
    return ((starts[:, :, None] + np.arange(L)) % n).reshape(n_boot, -1)[:, :n]


def calibration_uncertainty(records: list[dict], methods: list[str], n_boot: int = 100, seed: int = 0,
                            quantile: float = 0.995, block_len: int = 50) -> dict:
    """Bootstrap of each channel's threshold. Note: a resampled 99.5% quantile can never exceed the sample
    maximum, so the intervals are one-sided near the top and should be read as lower bounds on the uncertainty."""
    out = {}
    for m in methods:
        rng = np.random.default_rng([seed, *m.encode()])   # keyed by method NAME: independent of which others run
        per_channel, reps = [], []
        boot_thr = {}
        for r in records:
            v = r["val"][m][np.isfinite(r["val"][m])]
            idx = block_bootstrap_indices(v.size, n_boot, block_len, rng)
            t = np.quantile(v[idx], quantile, axis=1)
            t = np.where(t > 0, t, np.finfo(np.float32).eps)
            boot_thr[r["chan_id"]] = t
            lo, hi = np.percentile(t, [5, 95])
            per_channel.append({"chan_id": r["chan_id"], "method": m, "val_len": int(v.size),
                                "expected_exceedances": float(v.size * (1 - quantile)),
                                "threshold": r["threshold"][m], "thr_p05": float(lo), "thr_p95": float(hi),
                                "rel_width": float((hi - lo) / abs(r["threshold"][m])) if r["threshold"][m] else None})
        for b in range(n_boot):
            reps.append(micro_metrics([_counts(r, m, boot_thr[r["chan_id"]][b]) for r in records]))
        rep = pd.DataFrame(reps)
        out[m] = {"block_len": block_len, "point_estimate": micro_metrics([_counts(r, m, r["threshold"][m]) for r in records]),
                  "interval_90": {k: [float(rep[k].quantile(0.05)), float(rep[k].quantile(0.95))] for k in HEADLINE},
                  "per_channel": per_channel}
    return out


# ---------------------------------------------------------------- 2. leave-channels-out calibration
def make_folds(records: list[dict], k: int = 5, seed: int = 0) -> list[list[int]]:
    """Stratified by spacecraft: each fold gets a proportional share of SMAP and MSL channels."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for sc in sorted({r["spacecraft"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["spacecraft"] == sc]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return folds


def select_rule(records: list[dict], method: str, objective: str = "event_f1") -> tuple[float, float]:
    """Best (q, m) on the given channels; ties -> smaller m, then smaller q. Uses only these channels' labels."""
    best, best_val = None, -np.inf
    for m_ in MARGINS:
        for q in QUANTILES:
            score = micro_metrics([_counts(r, method, rule_threshold(r["val"][method], q, m_)) for r in records])[objective]
            if score > best_val + 1e-12:
                best, best_val = (q, m_), score
    return best


def leave_channels_out(records: list[dict], methods: list[str], k: int = 5, seed: int = 0,
                       objective: str = "event_f1") -> dict:
    folds = make_folds(records, k, seed)
    out = {"protocol": {"k": k, "seed": seed, "objective": objective, "quantiles": QUANTILES,
                        "margins": MARGINS, "default_rule": DEFAULT_RULE,
                        "folds": [[records[i]["chan_id"] for i in f] for f in folds]}}
    for m in methods:
        chosen, held = [], {}
        for f, fold in enumerate(folds):
            train = [records[i] for i in range(len(records)) if i not in set(fold)]
            q, m_ = select_rule(train, m, objective)
            chosen.append({"fold": f, "q": q, "m": m_})
            for i in fold:
                held[i] = (q, m_)
        res = {}
        for subset, keep in (("all", lambda r: True), ("calibratable", lambda r: r["calibratable"])):
            idx = [i for i, r in enumerate(records) if keep(r)]
            res[subset] = {
                "n_channels": len(idx),
                "lco": micro_metrics([_counts(records[i], m, rule_threshold(records[i]["val"][m], *held[i])) for i in idx]),
                "default": micro_metrics([_counts(records[i], m, records[i]["threshold"][m]) for i in idx])}
        out[m] = {"chosen_rules": chosen, **res}
    return out


# ---------------------------------------------------------------- 3. seed variance
def seed_summary(run_dirs: list[Path]) -> dict:
    rows = []
    for d in run_dirs:
        met = json.loads((Path(d) / "metrics.json").read_text())
        seed = json.loads((Path(d) / "config_used.json").read_text())["seed"]
        for m, r in met["results"].items():
            for subset in ("overall", "calibratable"):
                if r.get(subset):
                    rows.append({"run": Path(d).name, "seed": seed, "method": m, "subset": subset,
                                 **{k: r[subset][k] for k in HEADLINE + ("event_tp", "event_fp", "roc_auc_macro", "pr_auc_macro")}})
    df = pd.DataFrame(rows)
    keys = list(HEADLINE) + ["event_tp", "event_fp", "roc_auc_macro", "pr_auc_macro"]
    agg = df.groupby(["subset", "method"])[keys].agg(["mean", "std", "min", "max"])
    return {"runs": [Path(d).name for d in run_dirs], "seeds": sorted(df["seed"].unique().tolist()),
            "per_run": df.to_dict("records"),
            "summary": {f"{s}/{m}": {k: {stat: float(agg.loc[(s, m), (k, stat)]) for stat in ("mean", "std", "min", "max")}
                                      for k in keys} for s, m in agg.index}}


# ---------------------------------------------------------------- reporting
def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def run_calibration_studies(run_dir: Path, n_boot: int = 100) -> dict:
    run_dir = Path(run_dir)
    records = load_channel_records(run_dir)
    methods = [m for m in METHODS if m in records[0]["test"]]
    unc = calibration_uncertainty(records, methods, n_boot=n_boot)
    lco = leave_channels_out(records, methods)
    out_dir = run_dir / "calibration"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame([row for m in methods for row in unc[m]["per_channel"]]).to_csv(out_dir / "threshold_bootstrap.csv", index=False)
    result = {"n_boot": n_boot, "uncertainty": {m: {k: v for k, v in unc[m].items() if k != "per_channel"} for m in methods},
              "leave_channels_out": lco}
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=1, default=float))
    lines = [f"# Calibration studies: run `{run_dir.name}`", "",
             f"## Threshold-estimation uncertainty ({n_boot} circular block-bootstrap resamples, block length 50, "
             "of each channel's validation scores; one-sided near the top, read as lower bounds)", "",
             "| Method | Metric | Point estimate | 90% interval |", "|---|---|---|---|"]
    for m in methods:
        for k in ("precision", "recall", "f1", "event_f1"):
            lo, hi = unc[m]["interval_90"][k]
            lines.append(f"| {m} | {k} | {unc[m]['point_estimate'][k]:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "## Leave-channels-out calibration (held-out folds, rule chosen on the other folds by event F1)", "",
              "| Method | Channels | Rule | Precision | Recall | F1 | FPR | Event precision | Event recall | Event F1 | Event FP |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        for subset in ("all", "calibratable"):
            for name in ("default", "lco"):
                r = lco[m][subset][name]
                lines.append(f"| {m} | {subset} ({lco[m][subset]['n_channels']}) | "
                             f"{'default q=0.995, m=1' if name == 'default' else 'LCO-selected'} | "
                             + " | ".join(_fmt(r[k]) for k in ("precision", "recall", "f1", "fpr", "event_precision",
                                                                "event_recall", "event_f1", "event_fp")) + " |")
    lines += ["", "Rules chosen per fold (q, m):", ""]
    for m in methods:
        lines.append(f"* {m}: " + ", ".join(f"fold {c['fold']}: ({c['q']}, {c['m']})" for c in lco[m]["chosen_rules"]))
    (out_dir / "calibration.md").write_text("\n".join(lines) + "\n")
    from src.utils.config import display_path

    log.info("Calibration studies written to %s", display_path(out_dir))
    return result
