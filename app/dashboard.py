"""Spacecraft Telemetry Anomaly Lab: demo app for the historical NASA SMAP/MSL (Telemanom) benchmark study.

Launch from the repository root:  streamlit run app/dashboard.py

Everything the app shows is read from files; it never downloads data or trains a model.
  committed   reports/benchmark/   cross-run comparison of all methods (python -m src.cli compare)
              reports/showcase/    a few channels' telemetry, scores and predictions + 2 channels' saved models
                                   (python -m src.cli showcase), so a fresh clone works offline
              reports/runs/<run>/  per-run metrics, events, false alarms, calibration studies
  local only  reports/runs/<run>/scores/, models/<run>/, data/raw/  (generated / downloaded on this machine)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import lab_theme as T  # noqa: E402
from src.inference.predict import detect, frame_to_array, read_telemetry_csv, warmup_length  # noqa: E402
from src.models.decision import DYNAMIC_METHODS  # noqa: E402
from src.models.persistence import CHANNEL_ID, METHODS, TRAINABLE, load_bundle  # noqa: E402
from src.models.thresholds import runs as runs_of  # noqa: E402
from src.replay import REPLAY_LABEL, replay_state  # noqa: E402

REPORTS = ROOT / "reports"
RUNS_DIR, BENCH_DIR, SHOW_DIR = REPORTS / "runs", REPORTS / "benchmark", REPORTS / "showcase"
MODELS_DIR, NASA_RAW = ROOT / "models", ROOT / "data" / "raw" / "telemanom"
PAGES = ["Overview", "Compare methods", "Channel timeline", "Detections & errors", "Historical replay",
         "Inference", "Reproduce"]
# Seed-42 run holding each method's per-channel results.
METHOD_RUN = {"baseline": "full", "isolation_forest": "full", "lstm": "full", "lstm_dynamic": "lstm_dynamic",
              "telemanom": "telemanom"}
LABELS, COLORS = T.METHOD_LABELS, T.METHOD_COLORS
SHORT = {"baseline": "z-score", "isolation_forest": "IForest", "lstm": "LSTM", "lstm_dynamic": "LSTM+dyn",
         "telemanom": "Telemanom"}
DISCLAIMER = ("Historical, offline analysis of the public NASA SMAP/MSL benchmark (Telemanom, Hundman et al., KDD "
              "2018). Not affiliated with NASA, not a live feed, not operational software. Flags are statistical "
              "anomalies, not diagnosed faults.")

st.set_page_config(page_title="Telemetry Anomaly Lab · NASA SMAP/MSL", layout="wide")
st.markdown(T.CSS, unsafe_allow_html=True)


# ============================================================================== data access (read-only, cached)
def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _csv(p: Path):
    try:
        return pd.read_csv(p)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


# Cache keys include the repository root so test copies of the app never share cached files.
@st.cache_data
def _bench(root: str) -> dict:
    b = Path(root) / "reports" / "benchmark"
    return {"json": _json(b / "benchmark.json"), "runs": _csv(b / "runs.csv"), "ops": _csv(b / "operating_points.csv")}


def bench() -> dict:
    return _bench(str(ROOT))


@st.cache_data
def _run(root: str, run: str) -> dict:
    d = Path(root) / "reports" / "runs" / run
    return {"metrics": _json(d / "metrics.json") or {}, "config": _json(d / "config_used.json") or {},
            "summary": _json(d / "training_summary.json") or {"channels": []},
            "events": _csv(d / "events.csv"), "false_alarms": _csv(d / "false_alarms.csv"),
            "per_channel": _csv(d / "per_channel.csv"), "sensitivity": _csv(d / "sensitivity.csv"),
            "calibration": _json(d / "calibration" / "calibration.json"),
            "p_curve": _csv(d / "calibration" / "dynamic_p_curve.csv")}


def run_data(run: str) -> dict:
    return _run(str(ROOT), run)


@st.cache_data
def _showcase_index(root: str):
    return _json(Path(root) / "reports" / "showcase" / "index.json")


def showcase() -> dict | None:
    return _showcase_index(str(ROOT))


@st.cache_data
def _parquet(path: str):
    return pd.read_parquet(path) if Path(path).exists() else None


def showcase_table(chan: str) -> pd.DataFrame | None:
    return _parquet(str(SHOW_DIR / "channels" / f"{chan}.parquet"))


def local_table(chan: str, methods: list[str]) -> pd.DataFrame | None:
    """Merge per-method score tables of the seed-42 runs generated on this machine (gitignored)."""
    out = None
    for m in methods:
        t = _parquet(str(RUNS_DIR / METHOD_RUN[m] / "scores" / f"{chan}.parquet"))
        if t is None:
            continue
        cols = [c for c in (f"{m}_score", f"{m}_pred", f"{m}_threshold") if c in t]
        if out is None:
            out = t[["timestep", "value", "label"]].copy()
        out[cols] = t[cols]
    return out


def local_channels() -> list[str]:
    d = RUNS_DIR / "full" / "scores"
    return sorted((p.stem for p in d.glob("*.parquet") if not p.stem.endswith(".val")), key=chan_key) if d.exists() else []


def chan_key(c: str):
    letter, _, num = c.partition("-")
    return (letter, int(num) if num.isdigit() else 0)


def method_threshold(chan: str, m: str, table: pd.DataFrame) -> np.ndarray | float | None:
    if f"{m}_threshold" in table:
        return table[f"{m}_threshold"].to_numpy()
    idx = showcase() or {}
    thr = idx.get("thresholds", {}).get(chan, {}).get(m, {}).get("threshold")
    if thr is None:
        info = {c["chan_id"]: c for c in run_data(METHOD_RUN[m])["summary"]["channels"]}.get(chan, {})
        thr = info.get("methods", {}).get(m, {}).get("threshold")
    return thr


def dyn_cfg() -> dict | None:
    return (showcase() or {}).get("dynamic_threshold") or run_data("telemanom")["config"].get("dynamic_threshold")


def table(df: pd.DataFrame, fmt: dict | None = None, **kw):
    cfg = {c: st.column_config.NumberColumn(format=f) for c, f in (fmt or {}).items() if c in df.columns}
    st.dataframe(df, width="stretch", column_config=cfg, hide_index=True, **kw)


def pm(mean, std, fmt="{:.3f}") -> str:
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return "—"
    return fmt.format(mean) + ("" if std is None or not np.isfinite(std) else f" ± {fmt.format(std)}")


# ============================================================================== charts
def add_bands(fig, intervals, color, name, row, opacity=0.25, limit=150, y=None):
    intervals = list(intervals)
    if not intervals:
        return
    if len(intervals) <= limit:
        for i, (s, e) in enumerate(intervals):
            fig.add_vrect(x0=s - 0.5, x1=e + 0.5, fillcolor=color, opacity=opacity, line_width=0, layer="below",
                          row=row, col=1, exclude_empty_subplots=False, **({"annotation_text": name, "annotation_position": "top left",
                                              "annotation_font_color": T.TEXT} if i == 0 and name else {}))
    else:   # many short runs: one marker trace keeps the browser fast
        xs = np.concatenate([np.arange(s, e + 1) for s, e in intervals])
        fig.add_trace(go.Scattergl(x=xs, y=np.full(len(xs), 0.5 if y is None else y), mode="markers", name=name,
                                   marker=dict(color=color, size=4, symbol="line-ns-open")), row=row, col=1)


def timeline_figure(t: pd.DataFrame, chan: str, methods: list[str], score_method: str | None, x_range=None,
                    title="") -> go.Figure:
    """Row 1: telemetry + labelled intervals. One strip per method: its predicted intervals (and labels, faint).
    Last row: the chosen method's score and threshold (dynamic methods: the window epsilon over time)."""
    n_rows = 1 + len(methods) + (1 if score_method else 0)
    heights = [0.42] + [0.07] * len(methods) + ([0.3] if score_method else [])
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=heights)
    x = t["timestep"].to_numpy()
    labels = runs_of(t["label"].to_numpy())
    fig.add_trace(go.Scattergl(x=x, y=t["value"], mode="lines", name="telemetry", line=dict(color=T.TRACE, width=0.9)),
                  row=1, col=1)
    add_bands(fig, labels, T.LABEL, "labelled anomaly", 1, 0.3)
    for i, m in enumerate(methods, start=2):
        add_bands(fig, labels, T.LABEL, "", i, 0.12)
        pred = runs_of(t[f"{m}_pred"].to_numpy())
        add_bands(fig, pred, COLORS[m], f"{LABELS[m]} flags", i, 0.85)
        fig.update_yaxes(showticklabels=False, title_text=SHORT[m], title_font_size=10, row=i, col=1,
                         range=[0, 1])
    if score_method:
        r = n_rows
        s = t[f"{score_method}_score"].to_numpy(dtype=float)
        fig.add_trace(go.Scattergl(x=x, y=s, mode="lines", name=f"{LABELS[score_method]} score",
                                   line=dict(color=COLORS[score_method], width=1)), row=r, col=1)
        thr = method_threshold(chan, score_method, t)
        if isinstance(thr, np.ndarray):
            fig.add_trace(go.Scattergl(x=x, y=thr, mode="lines", name="dynamic threshold (epsilon)",
                                       line=dict(color=T.THRESH, dash="dash", width=1.3)), row=r, col=1)
        elif thr is not None:
            fig.add_hline(y=thr, line=dict(color=T.THRESH, dash="dash", width=1.3), row=r, col=1,
                          annotation_text=f"threshold {thr:.3g}", annotation_font_color=T.THRESH)
        positive = s[np.isfinite(s) & (s > 0)]
        if positive.size and positive.max() / max(positive.min(), 1e-12) > 1e3:
            fig.update_yaxes(type="log", row=r, col=1)
        fig.update_yaxes(title_text="score", row=r, col=1)
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_xaxes(title_text="sample index (the benchmark has no timestamps)", row=n_rows, col=1)
    fig.update_layout(height=300 + 46 * len(methods) + (190 if score_method else 0), title=title,
                      hovermode="x unified", showlegend=True)
    if x_range:
        fig.update_xaxes(range=x_range)
    return fig


# ============================================================================== shared UI
def timeline_source(key: str, methods_needed=None) -> tuple[str, list[str], callable] | None:
    """Pick between the committed showcase and full local scores (if this machine has them)."""
    idx, local = showcase(), local_channels()
    opts = []
    if idx:
        opts.append(f"Committed showcase ({len(idx['channels'])} channels)")
    if local:
        opts.append(f"Local run scores ({len(local)} channels)")
    if not opts:
        st.info("No telemetry to plot: the committed showcase (`reports/showcase/`) is missing and this machine has no "
                "local scores. Recreate the showcase with `python -m src.cli showcase` after a full run.")
        return None
    src = st.radio("Telemetry source", opts, horizontal=True, key=f"{key}_src",
                   help="The showcase is committed so the app works on a fresh clone. Local scores exist after "
                        "running the pipeline on this machine and cover every channel.")
    if src.startswith("Committed"):
        reason = {c["chan_id"]: c["reason"] for c in idx["channels"]}
        chans = sorted(reason, key=chan_key)
        st.caption("Showcase channels were picked by a fixed rule to illustrate outcomes (reason shown per channel); "
                   "they are not a representative sample.")
        return "showcase", chans, (lambda c: showcase_table(c)), reason
    return "local", local, (lambda c: local_table(c, list(METHOD_RUN))), {}


def methods_available(t: pd.DataFrame) -> list[str]:
    return [m for m in METHODS if f"{m}_pred" in t]


# ============================================================================== pages
def page_overview():
    st.markdown(T.hero(
        "NASA SMAP & MSL · Telemanom benchmark (KDD 2018) · historical data",
        "Spacecraft Telemetry Anomaly Lab",
        "Unsupervised detectors that learn each telemetry channel's normal behaviour from its training file and flag "
        "departures in the test file. 81 expert-labelled channels from the SMAP satellite and the Curiosity rover "
        "(105 labelled anomaly sequences, 104 after merging P-2's overlapping rows).",
        [("Historical benchmark · not a live feed", "hist"), ("Results: committed reports", "report")]),
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(T.callout("What this project shows",
                          "A leakage-safe pipeline (no test label chooses a model, threshold or setting), five "
                          "detectors compared under one protocol, an ablation that separates the model from the "
                          "thresholding rule, seed variation, and calibration diagnostics."), unsafe_allow_html=True)
    c2.markdown(T.callout("What it does not claim",
                          "No NASA affiliation or live data; no fault diagnosis; not an exact reproduction of the "
                          "Telemanom paper (differences are listed in docs/methodology.md); the dataset authors scaled "
                          "each channel with test-set min/max, which every method shares."), unsafe_allow_html=True)
    b = bench()
    if not b["json"]:
        st.info("The cross-run benchmark (`reports/benchmark/`) is missing. Create it with `python -m src.cli compare`.")
        return
    summ = pd.DataFrame(b["json"]["summary"])
    best = summ.iloc[0]
    second = summ.iloc[1] if len(summ) > 1 else None
    ref = summ.set_index("method").loc["lstm"] if "lstm" in set(summ.method) else None
    st.markdown("## Headline result")
    st.caption("Mean ± std over seeds. Predeclared settings, thresholds chosen without test labels. Ranking criterion "
               "fixed in advance: event F1. An event counts as found if any flagged sample overlaps it; a false-alarm "
               "event is a flagged run that overlaps no label.")
    rows = []
    for _, r in summ.iterrows():
        rows.append({"Method": LABELS.get(r["method"], r["method"]), "Threshold": r["threshold_mode"],
                     "Seeds": int(r["n_seeds"]),
                     "Event F1": pm(r["event_f1_mean"], r["event_f1_std"]),
                     "Events found (of 104)": pm(r["event_tp_mean"], r["event_tp_std"], "{:.0f}"),
                     "False-alarm events": pm(r["event_fp_mean"], r["event_fp_std"], "{:.0f}"),
                     "Event precision": pm(r["event_precision_mean"], r["event_precision_std"]),
                     "Point F1": pm(r["f1_mean"], r["f1_std"]),
                     "Median decision latency (samples)": pm(r["median_decision_latency_mean"], None, "{:.0f}")})
    table(pd.DataFrame(rows))
    if ref is not None and best["method"] != "lstm":
        st.markdown(T.callout(
            f"Best under the protocol: {LABELS.get(best['method'], best['method'])}",
            f"Event F1 {best['event_f1_mean']:.3f} vs {ref['event_f1_mean']:.3f} for the reference LSTM. It raises "
            f"{best['event_fp_mean']:.0f} false-alarm events instead of {ref['event_fp_mean']:.0f}, but finds "
            f"{best['event_tp_mean']:.0f} of 104 events instead of {ref['event_tp_mean']:.0f}: fewer false alarms "
            "bought with lower recall. See Compare methods for every operating point."
            + (f" Based on {int(best['n_seeds'])} seed(s)." if best["n_seeds"] < 3 else "")
            + (f" The runner-up, {LABELS.get(second['method'], second['method'])} ({second['event_f1_mean']:.3f} ± "
               f"{second['event_f1_std']:.3f}), is within the seed spread, so the two are not clearly separated."
               if second is not None and best["event_f1_mean"] - second["event_f1_mean"]
               <= max(best["event_f1_std"], second["event_f1_std"]) else "")), unsafe_allow_html=True)
    runs = b["runs"]
    o = runs[runs.subset == "overall"] if runs is not None else None
    if o is not None and len(o):
        fig = go.Figure()
        for m in [m for m in METHODS if m in set(o.method)]:
            d = o[o.method == m]
            fig.add_trace(go.Scatter(x=d["event_fp"], y=d["event_recall"], mode="markers", name=LABELS[m],
                                     marker=dict(color=COLORS[m], size=12, symbol=T.METHOD_SYMBOLS[m],
                                                 line=dict(color=T.BG, width=1)),
                                     text=[f"seed {s}" for s in d["seed"]],
                                     hovertemplate="%{text}<br>false-alarm events %{x}<br>event recall %{y:.3f}"
                                                   "<extra>" + LABELS[m] + "</extra>"))
        fig.update_layout(height=400, title="Recall vs false alarms (one marker per seed)",
                          xaxis=dict(title="false-alarm events on the test set (log)", type="log"),
                          yaxis=dict(title="event recall", range=[0, 1.02]))
        st.plotly_chart(fig, width="stretch", config=T.PLOT_CONFIG)
        st.caption(f"Up and to the left is better. Best method: {best['event_fp_mean']:.0f} false-alarm events "
                   f"({best['false_alarms_per_10k_mean']:.2f} per 10,000 test samples) for {best['event_tp_mean']:.0f} "
                   "found events. A historical benchmark cannot show whether that is operationally acceptable.")


def page_compare():
    st.markdown("# Compare methods")
    b = bench()
    if b["runs"] is None:
        st.info("Run `python -m src.cli compare` to build `reports/benchmark/`.")
        return
    subset = st.radio("Channels", ["overall", "calibratable"], horizontal=True, key="cmp_subset",
                      format_func={"overall": "All 81 channels",
                                   "calibratable": "59 channels with a calibratable static threshold (same set for every method)"}.get)
    r = b["runs"][b["runs"].subset == subset]
    keys = ["event_f1", "event_recall", "event_precision", "event_tp", "event_fp", "false_alarms_per_10k", "f1",
            "precision", "recall", "fpr", "median_decision_latency", "roc_auc_macro", "pr_auc_macro"]
    agg = r.groupby("method")[keys].agg(["mean", "std"])
    rows = []
    for m in [m for m in METHODS if m in agg.index]:
        a = agg.loc[m]
        fmt = {"event_tp": "{:.0f}", "event_fp": "{:.0f}", "false_alarms_per_10k": "{:.1f}",
               "median_decision_latency": "{:.0f}"}
        rows.append({"Method": LABELS[m], "Seeds": int(r[r.method == m].seed.nunique()),
                     **{k: pm(a[(k, "mean")], a[(k, "std")], fmt.get(k, "{:.3f}")) for k in keys}})
    st.markdown("### All metrics (mean ± std over seeds)")
    table(pd.DataFrame(rows).rename(columns={
        "event_f1": "Event F1", "event_recall": "Event recall", "event_precision": "Event precision",
        "event_tp": "Events found", "event_fp": "False-alarm events", "false_alarms_per_10k": "FA / 10k samples",
        "f1": "Point F1", "precision": "Point precision", "recall": "Point recall", "fpr": "Point FPR",
        "median_decision_latency": "Median decision latency", "roc_auc_macro": "ROC-AUC (macro)",
        "pr_auc_macro": "PR-AUC (macro)"}))
    st.caption("Decision latency = samples from an event's start until the method could raise the alert (the dynamic "
               "threshold releases decisions in batches of 70, which is included). ROC/PR-AUC are threshold-free, "
               "averaged per channel.")
    with st.expander("Per-seed rows"):
        table(r[["method", "run", "seed", "event_f1", "event_recall", "event_precision", "event_tp", "event_fp", "f1",
                 "median_decision_latency"]], {k: "%.3f" for k in ("event_f1", "event_recall", "event_precision", "f1")})

    st.markdown("## Operating points")
    st.caption("Predeclared = the settings fixed before the run. Leave-channels-out (LCO) = for each fold of channels, "
               "the rule that maximised event F1 on the *other* folds' labels; the held-out fold's labels never choose "
               "its own rule. Static methods vary the validation quantile/margin; dynamic methods vary the pruning "
               "parameter p.")
    ops = b["ops"]
    if ops is not None and len(ops):
        o42 = ops[ops.seed == 42]
        fig = go.Figure()
        for m in [m for m in METHODS if m in set(o42.method)]:
            d = o42[o42.method == m].set_index("operating_point")
            fig.add_trace(go.Scatter(x=d["event_fp"], y=d["event_recall"], mode="lines+markers+text",
                                     name=LABELS[m], text=["predeclared" if i == "predeclared" else "LCO"
                                                           for i in d.index],
                                     textposition="top center", line=dict(color=COLORS[m]),
                                     marker=dict(symbol=T.METHOD_SYMBOLS[m], size=11)))
        fig.update_layout(height=380, xaxis=dict(title="false-alarm events (log)", type="log"),
                          yaxis=dict(title="event recall", range=[0, 1.02]),
                          title="Moving from the predeclared setting to the LCO-selected one (seed 42)")
        st.plotly_chart(fig, width="stretch", config=T.PLOT_CONFIG)
        table(o42[["method", "operating_point", "event_recall", "event_precision", "event_f1", "event_fp", "f1"]],
              {k: "%.3f" for k in ("event_recall", "event_precision", "event_f1", "f1")})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Static threshold: scale the calibrated threshold")
        sens = run_data("full")["sensitivity"]
        if sens is not None and len(sens):
            fig = go.Figure()
            for m in ("baseline", "lstm"):
                d = sens[sens.method == m]
                fig.add_trace(go.Scatter(x=d["multiplier"], y=d["event_f1"], mode="lines+markers", name=LABELS[m],
                                         line=dict(color=COLORS[m])))
            fig.update_layout(height=300, xaxis=dict(title="threshold multiplier (1 = calibrated)", type="log"),
                              yaxis=dict(title="event F1"))
            st.plotly_chart(fig, width="stretch", config=T.PLOT_CONFIG)
            st.caption("Post-hoc diagnostic on test labels; not used to choose anything.")
    with c2:
        st.markdown("#### Dynamic threshold: pruning parameter p")
        fig, any_ = go.Figure(), False
        for m in DYNAMIC_METHODS:
            pc = run_data(METHOD_RUN[m])["p_curve"]
            if pc is not None and len(pc):
                any_ = True
                fig.add_trace(go.Scatter(x=pc["p"], y=pc["event_f1"], mode="lines+markers", name=LABELS[m],
                                         line=dict(color=COLORS[m]), text=pc["event_fp"],
                                         hovertemplate="p %{x}<br>event F1 %{y:.3f}<br>false alarms %{text}"))
        if any_:
            fig.add_vline(x=0.13, line=dict(color=T.THRESH, dash="dash"), annotation_text="published p = 0.13")
            fig.update_layout(height=300, xaxis_title="p (higher = prune more)", yaxis_title="event F1")
            st.plotly_chart(fig, width="stretch", config=T.PLOT_CONFIG)
            st.caption("Post-hoc diagnostic on test labels; not used to choose anything.")
        else:
            st.caption("Run `python -m src.cli calibration --run telemanom` to create this curve.")

    st.markdown("## Calibration status")
    rows = []
    for m in [m for m in METHODS if (RUNS_DIR / METHOD_RUN[m] / "metrics.json").exists()]:
        rd = run_data(METHOD_RUN[m])
        ch = [c["methods"].get(m, {}) for c in rd["summary"]["channels"]]
        ok = [c for c in ch if c.get("status") == "ok"]
        pc = rd["per_channel"]
        ks = pc[pc.method == m]["ks_val_test"] if pc is not None and "ks_val_test" in pc else pd.Series(dtype=float)
        rows.append({"Method": LABELS[m], "Run": METHOD_RUN[m], "Threshold rule":
                     "dynamic (label-free, from the judged series)" if m in DYNAMIC_METHODS else
                     "static (99.5% quantile of validation scores)",
                     "Channels with degenerate calibration": sum(1 for c in ok if c.get("calibratable") is False),
                     "Median validation flag rate": float(np.median([c["val_flag_rate"] for c in ok])) if ok else None,
                     "Median test false-positive rate": float(pc[pc.method == m]["fpr"].median())
                     if pc is not None else None,
                     "Median KS(validation, test scores)": float(ks.median()) if len(ks) else None})
    table(pd.DataFrame(rows), {"Median validation flag rate": "%.4f", "Median test false-positive rate": "%.4f",
                               "Median KS(validation, test scores)": "%.2f"})
    st.caption("Why the reference LSTM raises so many false alarms (seed 42): its thresholds flag a median 0.54% of "
               "validation points, but the median test false-positive rate is 6.3%, so test-period errors are wider "
               "than the short validation segment suggested. 59% of its false-alarm runs peak below 1.5x the threshold "
               "(borderline crossings) and 38% last 5 samples or fewer (flicker); 10 channels produce 63% of them. "
               "The dynamic threshold adapts epsilon to recent errors, requires a clear gap to normal errors (pruning) "
               "and merges nearby flags. Dynamic validation flag rates are high because validation segments are "
               "shorter than one 2,100-sample window and flags are padded by 100 samples.")
    checks = b["json"].get("reproducibility_checks") if b["json"] else None
    if checks:
        st.markdown("**Reproducibility checks:** " + "; ".join(f"{k}: {'identical' if v else 'different'}"
                                                              for k, v in checks.items()))


def page_timeline():
    st.markdown("# Channel timeline")
    src = timeline_source("tl")
    if not src:
        return
    kind, chans, loader, reason = src
    c1, c2 = st.columns([1, 2])
    chan = c1.selectbox("Channel", chans, key=f"tl_chan_{kind}",
                        format_func=lambda c: f"{c} · {reason[c]}" if c in reason else c)
    t = loader(chan)
    if t is None:
        st.warning(f"No scores for {chan}.")
        return
    avail = methods_available(t)
    default = [m for m in ("lstm", "lstm_dynamic", "telemanom") if m in avail] or avail[:2]
    methods = c2.multiselect("Methods", avail, default=default, format_func=LABELS.get, key=f"tl_methods_{kind}")
    score_m = st.radio("Score panel", ["(none)"] + methods, horizontal=True, index=len(methods),
                       format_func=lambda m: m if m == "(none)" else LABELS[m], key=f"tl_score_{kind}")
    n = len(t)
    lo, hi = st.slider("Sample range", 0, n - 1, (0, n - 1), key=f"tl_range_{kind}_{chan}")
    st.plotly_chart(timeline_figure(t, chan, methods, None if score_m == "(none)" else score_m, [lo, hi],
                                    f"{chan} · test file"), width="stretch", config=T.PLOT_CONFIG)
    rows = []
    labels = runs_of(t["label"].to_numpy())
    for m in methods:
        pred = t[f"{m}_pred"].to_numpy()
        found = sum(bool(pred[s : e + 1].any()) for s, e in labels)
        fa = sum(1 for s, e in runs_of(pred) if not t["label"].to_numpy()[s : e + 1].any())
        rows.append({"Method": LABELS[m], "Labelled events": len(labels), "Found": found, "Missed": len(labels) - found,
                     "False-alarm events": fa, "Flagged samples": int(pred.sum())})
    if rows:
        table(pd.DataFrame(rows))


def page_events():
    st.markdown("# Detections & errors")
    avail = [m for m in METHODS if (RUNS_DIR / METHOD_RUN[m] / "events.csv").exists()]
    if not avail:
        st.info("No per-event reports found under `reports/runs/`.")
        return
    pref = next((x for x in ("telemanom", "lstm_dynamic", "lstm") if x in avail), avail[0])
    m = st.radio("Method", avail, index=avail.index(pref), horizontal=True,
                 format_func=LABELS.get, key="ev_method")
    rd = run_data(METHOD_RUN[m])
    ev, fa = rd["events"], rd["false_alarms"]
    ev, fa = ev[ev.method == m], fa[fa.method == m]
    c = st.columns(4)
    c[0].metric("Labelled events", len(ev))
    c[1].metric("Detected", int(ev.detected.sum()))
    c[2].metric("Missed", int((~ev.detected).sum()))
    c[3].metric("False-alarm events", len(fa))
    st.caption(f"Seed-42 run `{METHOD_RUN[m]}`, all 81 channels.")
    by = ev.groupby("class")["detected"].agg(["sum", "count"]).reset_index()
    by["recall"] = by["sum"] / by["count"]
    st.markdown("**Recall by labelled anomaly class:** " + ", ".join(
        f"{r['class']} {int(r['sum'])}/{int(r['count'])}" for _, r in by.iterrows()))
    kinds = {"True detections": ev[ev.detected], "Missed events": ev[~ev.detected], "False alarms": fa}
    kind = st.radio("Show", list(kinds), horizontal=True, key="ev_kind")
    d = kinds[kind].copy()
    if kind == "False alarms":
        d = d.sort_values(["length", "peak_norm_score"], ascending=False)
        cols = ["chan_id", "spacecraft", "start", "end", "length", "peak_norm_score"]
    else:
        cols = ["chan_id", "spacecraft", "start", "end", "length", "class", "latency", "decision_latency",
                "peak_norm_score"]
    cols = [c for c in cols if c in d]
    if kind == "False alarms" and len(d):
        st.caption(f"{len(d)} false-alarm runs; median length {d.length.median():.0f} samples; "
                   f"{(d.length <= 5).mean():.0%} are 5 samples or shorter. Top channels: " +
                   ", ".join(f"{k} ({v})" for k, v in d.chan_id.value_counts().head(5).items()))
    table(d[cols].reset_index(drop=True), {"peak_norm_score": "%.2f"}, height=260)
    # Zoomed example for channels whose scores are available (showcase or local)
    idx = showcase() or {}
    plot_chans = set(local_channels()) or {c["chan_id"] for c in idx.get("channels", [])}
    d = d[d.chan_id.isin(plot_chans)]
    if not len(d):
        st.caption("Plots need scores: the committed showcase covers a few channels; local runs cover all.")
        return
    opts = [f"{r.chan_id} [{r.start}–{r.end}]" for r in d.itertuples()]
    pick = st.selectbox("Plot an example (channels with available scores)", opts, key=f"ev_pick_{m}_{kind}")
    row = d.iloc[opts.index(pick)]
    t = local_table(row.chan_id, [m]) if local_channels() else showcase_table(row.chan_id)
    if t is None or f"{m}_pred" not in t:
        st.caption("No scores for this channel/method.")
        return
    pad = max(300, int(row.length))
    rng = [max(0, int(row.start) - pad), min(len(t) - 1, int(row.end) + pad)]
    st.plotly_chart(timeline_figure(t, row.chan_id, [m], m, rng, f"{row.chan_id} · {kind.lower()[:-1]}"),
                    width="stretch", config=T.PLOT_CONFIG)


def page_replay():
    st.markdown("# Historical replay")
    st.markdown(f'<div class="replay-banner">{REPLAY_LABEL.upper()}</div>', unsafe_allow_html=True)
    st.markdown("Archived test samples are fed to the detector in order. Only data received so far is used: scores are "
                "causal, and the dynamic threshold releases decisions once per 70-sample batch, so an alert can appear "
                "up to 69 samples after the sample it flags. Labels are hindsight, shown only to judge each alert.")
    idx = showcase()
    if not idx:
        st.info("The replay uses the committed showcase (`reports/showcase/`), which is missing. "
                "Recreate it with `python -m src.cli showcase`.")
        return
    reason = {c["chan_id"]: c["reason"] for c in idx["channels"]}
    c1, c2 = st.columns(2)
    chan = c1.selectbox("Channel", list(reason), key="rp_chan",
                        format_func=lambda c: f"{c} · {reason[c]}")
    t = showcase_table(chan)
    avail = methods_available(t)
    pref = next((x for x in ("telemanom", "lstm_dynamic", "lstm") if x in avail), avail[0])
    m = c2.selectbox("Detector", avail, index=avail.index(pref), format_func=LABELS.get, key="rp_method")
    n, step = len(t), 70
    key = f"rp_t_{chan}"
    if key not in st.session_state:
        st.session_state[key] = n // 3
    b = st.columns([1, 1, 1, 1, 3])
    if b[0].button("⟲ Restart", key="rp_restart"):
        st.session_state[key] = min(n, 300)
    if b[1].button("+1 batch", key="rp_step"):
        st.session_state[key] = min(n, st.session_state[key] + step)
    if b[2].button("+10 batches", key="rp_step10"):
        st.session_state[key] = min(n, st.session_state[key] + 10 * step)
    play = b[3].button("▶ Play 15 s", key="rp_play")
    hindsight = b[4].toggle("Show hindsight labels", value=True, key="rp_labels")
    st.slider("Samples received", 0, n, key=key, step=1)
    holder = st.empty()

    def render(tt: int):
        s = replay_state(t, m, tt, dyn_cfg())
        with holder.container():
            seen = t.iloc[:tt]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.04)
            fig.add_trace(go.Scattergl(x=seen["timestep"], y=seen["value"], mode="lines", name="received telemetry",
                                       line=dict(color=T.TRACE, width=0.9)), row=1, col=1)
            if hindsight:
                add_bands(fig, [(a, min(e, tt - 1)) for a, e in runs_of(t["label"].to_numpy()) if a < tt],
                          T.LABEL, "labelled (hindsight)", 1, 0.25)
            add_bands(fig, runs_of(s.pred), COLORS[m], "alert", 1, 0.55)
            fig.add_trace(go.Scattergl(x=seen["timestep"], y=seen[f"{m}_score"], mode="lines", name="score",
                                       line=dict(color=COLORS[m], width=1)), row=2, col=1)
            if s.decided < tt:
                for row in (1, 2):
                    fig.add_vrect(x0=s.decided, x1=tt, fillcolor=T.MUTED, opacity=0.18, line_width=0, row=row, col=1,
                                  **({"annotation_text": "awaiting batch", "annotation_position": "top right"}
                                     if row == 1 else {}))
            fig.update_xaxes(range=[0, n], title_text="sample index", row=2, col=1)
            fig.update_layout(height=430, title=f"[{REPLAY_LABEL}] {chan} · {LABELS[m]} · {tt:,}/{n:,} samples")
            st.plotly_chart(fig, width="stretch", config=T.PLOT_CONFIG, key=f"rp_fig_{tt}")
            k = st.columns(3)
            k[0].metric("Samples received", f"{tt:,}")
            k[1].metric("Alerts raised", len(s.alerts))
            k[2].metric("Alerts that were false alarms (hindsight)", int((s.alerts.hindsight == "false alarm").sum())
                        if len(s.alerts) else 0)
            if len(s.alerts):
                table(s.alerts.rename(columns={"raised_at": "raised at sample", "delay": "delay (samples)"}))
            else:
                st.caption("No alerts yet.")

    if play:
        start = st.session_state[key]
        frames = list(range(start, n + 1, max(step, (n - start) // 30 or step)))[:31] or [n]
        for tt in frames:
            render(tt)
            time.sleep(0.4)
    else:
        render(st.session_state[key])


@st.cache_data(max_entries=8, show_spinner=False)
def _score(models_dir: str, method: str, chan: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    bundle = load_bundle(Path(models_dir), method, chan)
    arr, warns = frame_to_array(df, bundle.n_features)
    return detect(bundle, arr), warns


def model_sources() -> dict[str, Path]:
    out = {}
    if (SHOW_DIR / "models").exists():
        out["Showcase models (local)"] = SHOW_DIR / "models"
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and any((d / m).exists() for m in TRAINABLE):
                out[f"Local run: {d.name}"] = d
    return out


def page_inference():
    st.markdown("# Inference with a saved model")
    st.markdown("Score a telemetry CSV with a detector saved at training time. Its scaler and threshold rule are "
                "reused, never refitted. " + DISCLAIMER)
    srcs = model_sources()
    if not srcs:
        st.info("No saved models on this machine (model weights are not committed). Create them with a run, e.g. "
                "`python -m src.cli all --config configs/telemanom_smoke.yaml` (about 2 minutes, writes "
                "`models/telemanom_smoke/`), or a full run followed by `python -m src.cli showcase`.")
        return
    c1, c2, c3 = st.columns(3)
    src = c1.selectbox("Models", list(srcs), key="inf_src")
    root = srcs[src]
    methods = [m for m in TRAINABLE if (root / m).exists()]
    m = c2.selectbox("Method", methods, index=methods.index("telemanom") if "telemanom" in methods else 0,
                     format_func=LABELS.get, key="inf_method")
    chans = sorted((p.name for p in (root / m).iterdir() if CHANNEL_ID.fullmatch(p.name)), key=chan_key)
    chan = c3.selectbox("Channel", chans, key="inf_chan")
    try:
        bundle = load_bundle(root, m, chan)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return
    if bundle.calibration_warning():
        st.warning(bundle.calibration_warning())
    st.caption(f"Input: a `value` column (benchmark-scaled units, about [-1, 1]) and optionally `cmd_0..cmd_"
               f"{bundle.n_features - 2}` (0/1 command flags). More than {warmup_length(bundle)} rows. Threshold rule: "
               f"{'dynamic (label-free, computed on your series)' if m in DYNAMIC_METHODS else 'fixed per-channel threshold'}.")
    sample = showcase_table(chan)
    opts = (["Archived test segment of this channel (showcase)"] if sample is not None else []) + ["Upload CSV"]
    how = st.radio("Input", opts, horizontal=True, key="inf_how")
    cmd_cols = [f"cmd_{i}" for i in range(bundle.n_features - 1)]
    df = None
    if how == "Upload CSV":
        up = st.file_uploader("Telemetry CSV", type="csv", key="inf_upload")
        tmpl = pd.DataFrame({"value": [0.0] * 3, **{c: [0] * 3 for c in cmd_cols}})
        st.download_button("Download CSV template", tmpl.to_csv(index=False), "telemetry_template.csv")
        if up is not None:
            try:
                df = read_telemetry_csv(up)
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                st.error(f"Could not read the CSV: {exc}")
    else:
        df = sample[["value"] + [c for c in cmd_cols if c in sample]]
        st.download_button("Download this input as CSV", df.to_csv(index=False), f"{chan}_test_input.csv")
    if df is None:
        st.caption("Choose an input to score.")
        return
    try:
        with st.spinner("Scoring…"):
            out, warns = _score(str(root), m, chan, df)
        for w in warns:
            st.warning(w)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return
    c = st.columns(3)
    c[0].metric("Samples scored", f"{int(out['scored'].sum()):,}")
    c[1].metric("Flagged samples", f"{int(out['is_anomaly'].sum()):,}")
    c[2].metric("Flagged runs", len(runs_of(out["is_anomaly"].to_numpy())))
    show = pd.DataFrame({"timestep": out["timestep"], "value": out["value"], f"{m}_score": out["score"],
                         f"{m}_pred": out["is_anomaly"], f"{m}_threshold": out["threshold"],
                         "label": sample["label"] if (sample is not None and how != "Upload CSV") else 0})
    st.plotly_chart(timeline_figure(show, chan, [m], m, title=f"{chan} · scored input"
                                    + (" (labels shown: archived test segment)" if how != "Upload CSV" else "")),
                    width="stretch", config=T.PLOT_CONFIG)
    st.download_button("Download scored CSV", out.to_csv(index=False), f"{chan}_{m}_scores.csv")


def page_reproduce():
    st.markdown("# Reproduce")
    raw_ok = (NASA_RAW / "labeled_anomalies.csv").exists() and any((NASA_RAW / "test").glob("*.npy")) \
        if (NASA_RAW / "test").exists() else False
    st.markdown(T.chips(("NASA data: downloaded" if raw_ok else "NASA data: not downloaded",
                         "source" if raw_ok else "missing"),
                        (f"Local score tables: {len(local_channels())} channels", "local" if local_channels() else "missing"),
                        ("Benchmark artifact: present" if bench()["json"] else "Benchmark artifact: missing",
                         "report" if bench()["json"] else "missing")), unsafe_allow_html=True)
    st.markdown("The app never trains or downloads. Run these from the repository root (see README for timings):")
    st.code("""pip install -r requirements.txt
python -m src.cli download && python -m src.cli validate      # ~80 MB, checksummed
python -m pytest -q                                           # tests (~3 min)
python -m src.cli all --config configs/telemanom_smoke.yaml   # 3-channel smoke run (~2 min)
python -m src.cli all --config configs/default.yaml --resume  # reference run (~1 h CPU)
python -m src.cli all --config configs/telemanom.yaml --resume # Telemanom-aligned run (CPU; see README)
python -m src.cli derive --config configs/lstm_dynamic.yaml   # ablation from saved reference models
python -m src.cli calibration --run telemanom
python -m src.cli compare && python -m src.cli showcase
streamlit run app/dashboard.py""", language="bash")
    st.caption("New experiments always write to new run names; the committed reference runs are never overwritten. "
               "Optional synthetic demo without NASA data: `python -m src.cli demo` (clearly labelled SIMULATED).")


# ============================================================================== main
def main():
    sb = st.sidebar
    sb.markdown('<div class="lab-brand">Telemetry Anomaly <span>Lab</span></div>'
                '<div class="muted">NASA SMAP &amp; MSL · historical benchmark</div>', unsafe_allow_html=True)
    page = sb.radio("Navigate", PAGES, key="page")
    if sb.button("Reload files from disk"):
        st.cache_data.clear()
        st.rerun()
    sb.markdown("---")
    sb.caption(DISCLAIMER)
    {"Overview": page_overview, "Compare methods": page_compare, "Channel timeline": page_timeline,
     "Detections & errors": page_events, "Historical replay": page_replay, "Inference": page_inference,
     "Reproduce": page_reproduce}[page]()


main()
