"""Streamlit dashboard for the historical NASA SMAP/MSL telemetry anomaly-detection experiments.

Launch:  streamlit run app/dashboard.py

The dashboard never trains anything. It reads
  reports/data_validation.json            dataset summary
  reports/runs/<run>/...                  scores, metrics, error analysis (from `src.cli all`)
  models/<run>/<method>/<chan>/           saved detectors (Inference page)
  data/raw/telemanom/                     raw train telemetry for context plots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_channel  # noqa: E402
from src.inference.predict import detect, frame_to_array, warmup_length  # noqa: E402
from src.models.persistence import METHODS, load_bundle  # noqa: E402
from src.models.thresholds import runs  # noqa: E402

RUNS_DIR, MODELS_DIR, RAW_DIR = ROOT / "reports" / "runs", ROOT / "models", ROOT / "data" / "raw" / "telemanom"
LABELS = {"baseline": "Rolling z-score (baseline)", "isolation_forest": "Isolation Forest", "lstm": "LSTM forecaster"}
COLORS = {"baseline": "#7f7f7f", "isolation_forest": "#2a9d8f", "lstm": "#6c5ce7", "truth": "#e63946"}
DISCLAIMER = ("Historical benchmark analysis only. This app does not monitor any spacecraft. "
              "An 'anomaly' here is a statistically unusual observation, not a confirmed physical fault.")

st.set_page_config(page_title="Telemetry Anomaly Detection", layout="wide")


# ----------------------------------------------------------------------------------- data access
@st.cache_data
def available_runs() -> list[str]:
    return sorted(p.parent.name for p in RUNS_DIR.glob("*/metrics.json"))


@st.cache_data
def load_run(run: str) -> dict:
    d = RUNS_DIR / run
    summary = json.loads((d / "training_summary.json").read_text())
    return {"metrics": json.loads((d / "metrics.json").read_text()), "summary": summary,
            "config": json.loads((d / "config_used.json").read_text()),
            "channels": {c["chan_id"]: c for c in summary["channels"]},
            "comparison": pd.read_csv(d / "comparison.csv"), "per_channel": pd.read_csv(d / "per_channel.csv"),
            "events": pd.read_csv(d / "events.csv"), "false_alarms": pd.read_csv(d / "false_alarms.csv"),
            "sensitivity": pd.read_csv(d / "sensitivity.csv")}


@st.cache_data
def load_scores(run: str, chan: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS_DIR / run / "scores" / f"{chan}.parquet")


@st.cache_data
def load_train_values(chan: str) -> np.ndarray | None:
    try:
        return load_channel(RAW_DIR, chan, "train")[:, 0]
    except FileNotFoundError:
        return None


def ok_methods(run_data: dict, chan: str) -> list[str]:
    return [m for m in METHODS if run_data["channels"][chan]["methods"].get(m, {}).get("status") == "ok"]


def threshold_desc(cfg: dict) -> str:
    t = cfg["threshold"]
    if t["method"] == "quantile":
        return f"{t['quantile']:.3%} quantile (x{t['margin']})"
    return f"mean + {t['k_std']} std (x{t['margin']})"


# ----------------------------------------------------------------------------------- plotting
def add_intervals(fig: go.Figure, binary: np.ndarray, color: str, name: str, row_y=(0, 1), limit: int = 250):
    """Shade contiguous runs of 1s. Very many runs are drawn as one trace to keep the browser fast."""
    r = runs(binary)
    if not r:
        return
    if len(r) <= limit:
        for i, (s, e) in enumerate(r):
            fig.add_vrect(x0=s - 0.5, x1=e + 0.5, fillcolor=color, opacity=0.18, line_width=0,
                          layer="below", annotation_text=name if i == 0 else None, annotation_position="top left")
    else:
        y = np.where(binary == 1, 1.0, np.nan)
        fig.add_trace(go.Scatter(x=np.arange(len(binary)), y=y, mode="markers", name=f"{name} ({len(r)} runs)",
                                 marker=dict(color=color, size=3), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", range=[0, 1.05], visible=False))


def telemetry_figure(t: pd.DataFrame, method: str | None, title: str, x_range=None) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=t["timestep"], y=t["value"], mode="lines", name="telemetry",
                                 line=dict(color="#222", width=1)))
    if "lstm_forecast" in t and method == "lstm":
        fig.add_trace(go.Scattergl(x=t["timestep"], y=t["lstm_forecast"], mode="lines", name="LSTM forecast",
                                   line=dict(color=COLORS["lstm"], width=1, dash="dot")))
    add_intervals(fig, t["label"].to_numpy(), COLORS["truth"], "labelled anomaly")
    if method:
        add_intervals(fig, t[f"{method}_pred"].to_numpy(), COLORS[method], f"{method} prediction")
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="test timestep (index)", yaxis_title="value (benchmark-scaled)",
                      legend=dict(orientation="h", y=-0.25))
    if x_range:
        fig.update_xaxes(range=x_range)
    return fig


def score_figure(t: pd.DataFrame, method: str, thr: float, x_range=None) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=t["timestep"], y=t[f"{method}_score"], mode="lines", name="anomaly score",
                                 line=dict(color=COLORS[method], width=1)))
    fig.add_hline(y=thr, line=dict(color="orange", dash="dash"), annotation_text=f"threshold {thr:.3g}")
    add_intervals(fig, t["label"].to_numpy(), COLORS["truth"], "labelled anomaly")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="test timestep",
                      yaxis_title="score (log)", yaxis_type="log", showlegend=False)
    if x_range:
        fig.update_xaxes(range=x_range)
    return fig


def predicted_intervals(t: pd.DataFrame, method: str, thr: float) -> pd.DataFrame:
    score, label = t[f"{method}_score"].to_numpy(), t["label"].to_numpy()
    rows = []
    for s, e in runs(t[f"{method}_pred"].to_numpy()):
        seg = score[s : e + 1]
        rows.append({"start": s, "end": e, "length": e - s + 1, "peak score": float(np.nanmax(seg)),
                     "threshold": thr, "peak / threshold": float(np.nanmax(seg) / thr),
                     "overlaps labelled anomaly": bool(label[s : e + 1].any())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- pages
def page_overview(run: str, rd: dict):
    st.header("Overview")
    st.info(DISCLAIMER)
    val_path = ROOT / "reports" / "data_validation.json"
    if val_path.exists():
        s = json.loads(val_path.read_text())["summary"]
        c = st.columns(5)
        c[0].metric("Labelled channels", s["n_channels_labelled"])
        c[1].metric("SMAP / MSL", f"{s['by_spacecraft'].get('SMAP', 0)} / {s['by_spacecraft'].get('MSL', 0)}")
        c[2].metric("Train observations", f"{s['n_train_points']:,}")
        c[3].metric("Test observations", f"{s['n_test_points']:,}")
        c[4].metric("Labelled anomaly sequences", s["n_anomaly_sequences"])
    m = rd["metrics"]
    st.subheader(f"Run `{run}` - test-set results ({m['n_channels_evaluated']} channels)")
    if m["training_failures"]:
        st.warning(f"Training failures (excluded from comparison): {m['training_failures']}")
    comp = rd["comparison"].copy()
    comp["method"] = comp["method"].map(LABELS)
    st.dataframe(comp.set_index("method").style.format("{:.3f}", subset=[c for c in comp.columns if comp[c].dtype == float]),
                 use_container_width=True)
    st.caption("Point metrics treat every timestep as a sample; event metrics treat each labelled anomaly sequence "
               "as one event (detected if any predicted point overlaps it) and each predicted run that overlaps no "
               "label as one false alarm. Thresholds were calibrated on the validation part of the TRAIN files only.")
    with st.expander("Dataset provenance"):
        st.markdown("NASA SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory) telemetry with expert "
                    "anomaly labels from Hundman et al., KDD 2018 (Telemanom). Original `.npy` files obtained from a "
                    "pinned Hugging Face mirror and verified by SHA-256; labels cross-checked against the Telemanom "
                    "GitHub repository. See `docs/dataset.md`.")


def page_explorer(run: str, rd: dict):
    st.header("Telemetry explorer")
    chans = sorted(rd["channels"])
    c1, c2 = st.columns([1, 2])
    chan = c1.selectbox("Channel", chans, help="Channel IDs are anonymised; the letter is the subsystem type.")
    methods = ok_methods(rd, chan)
    method = c2.radio("Overlay predictions from", ["none"] + methods, horizontal=True,
                      format_func=lambda m: LABELS.get(m, "none"))
    t = load_scores(run, chan)
    info = rd["channels"][chan]
    lo, hi = st.slider("Test time range", 0, len(t) - 1, (0, len(t) - 1))
    st.plotly_chart(telemetry_figure(t, None if method == "none" else method,
                                     f"{chan} ({info['spacecraft']}) - test period", [lo, hi]), use_container_width=True)
    train = load_train_values(chan)
    with st.expander("Training-period telemetry (what the models learned 'normal' from)"):
        if train is None:
            st.write("Raw training file not available locally.")
        else:
            fig = go.Figure(go.Scattergl(y=train, mode="lines", line=dict(color="#555", width=1)))
            fig.add_vline(x=info["split_idx"], line=dict(color="orange", dash="dash"),
                          annotation_text="fit | validation split")
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="train timestep")
            st.plotly_chart(fig, use_container_width=True)
            if np.std(train) < 1e-3:
                st.warning("This channel is constant during training, so any change in the test period looks novel.")


def page_anomalies(run: str, rd: dict):
    st.header("Anomaly analysis")
    chans = sorted(rd["channels"])
    c1, c2 = st.columns(2)
    chan = c1.selectbox("Channel", chans, key="an_chan")
    methods = ok_methods(rd, chan)
    if not methods:
        st.error("No method succeeded on this channel.")
        return
    method = c2.selectbox("Method", methods, format_func=LABELS.get)
    t = load_scores(run, chan)
    thr = rd["channels"][chan]["methods"][method]["threshold"]
    st.markdown(f"**How to read this:** the score measures how unusual each timestep is for `{LABELS[method]}`. "
                f"The dashed line is the channel's threshold **{thr:.4g}**, set as the "
                f"{threshold_desc(rd['config'])} of scores on the *validation* part of "
                "the training data. Points above it are predicted anomalous. Red shading = expert labels.")
    preds = predicted_intervals(t, method, thr)
    sel_range = None
    if len(preds):
        idx = st.selectbox("Zoom to predicted interval", ["(whole series)"] + list(range(len(preds))),
                           format_func=lambda i: i if i == "(whole series)" else
                           f"#{i}: {preds.loc[i, 'start']}-{preds.loc[i, 'end']} (peak/thr {preds.loc[i, 'peak / threshold']:.1f})")
        if idx != "(whole series)":
            s, e = int(preds.loc[idx, "start"]), int(preds.loc[idx, "end"])
            pad = max(200, (e - s) * 2)
            sel_range = [max(0, s - pad), min(len(t) - 1, e + pad)]
    st.plotly_chart(telemetry_figure(t, method, f"{chan}: telemetry, labels and {method} predictions", sel_range),
                    use_container_width=True)
    st.plotly_chart(score_figure(t, method, thr, sel_range), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted anomaly intervals")
        if len(preds):
            st.dataframe(preds, use_container_width=True, height=280)
        else:
            st.write("No timestep exceeded the threshold.")
    with col2:
        st.subheader("Labelled anomalies vs prediction")
        ev = rd["events"]
        st.dataframe(ev[(ev.chan_id == chan) & (ev.method == method)][["start", "end", "length", "class", "detected",
                                                                         "latency", "peak_norm_score"]],
                     use_container_width=True)
        st.caption("latency = timesteps from labelled start to first predicted point; peak_norm_score = max score / "
                   "threshold inside the labelled window (< 1 means the method never crossed its threshold there).")


def page_comparison(run: str, rd: dict):
    st.header("Model comparison")
    comp = rd["comparison"]
    keys = st.multiselect("Metrics", ["precision", "recall", "f1", "fpr", "event_precision", "event_recall",
                                      "event_f1", "roc_auc", "pr_auc"],
                          default=["precision", "recall", "f1", "event_precision", "event_recall", "event_f1"])
    fig = go.Figure([go.Bar(name=LABELS[m], x=keys, y=[comp.set_index("method").loc[m, k] for k in keys],
                            marker_color=COLORS[m]) for m in comp["method"]])
    fig.update_layout(barmode="group", height=380, yaxis_range=[0, 1], margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    res = rd["metrics"]["results"]
    st.subheader("By spacecraft")
    rows = [{"method": LABELS[m], "spacecraft": sc, **{k: res[m][sc][k] for k in
            ("precision", "recall", "f1", "fpr", "event_precision", "event_recall", "event_f1")}}
            for m in res for sc in ("SMAP", "MSL") if sc in res[m]]
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)
    st.subheader("Event recall by anomaly class")
    st.dataframe(pd.DataFrame(rd["metrics"]["event_recall_by_class"]).rename(index=LABELS), use_container_width=True)
    st.subheader("Threshold sensitivity (post-hoc diagnostic)")
    st.caption("How metrics would change if every calibrated threshold were multiplied by k. Computed with test "
               "labels, so it is for understanding the precision/recall trade-off only - NOT for choosing k.")
    sens = rd["sensitivity"]
    if "scale_comparable" in sens:
        sens = sens[sens["scale_comparable"]]
        st.caption("Isolation Forest is omitted: its scores lie in a narrow band around 0.5, so multiplying its "
                   "threshold flags either every point or none.")
    fig = go.Figure([go.Scatter(x=sens[sens.method == m]["point_recall"], y=sens[sens.method == m]["point_precision"],
                                mode="lines+markers+text", text=sens[sens.method == m]["multiplier"],
                                textposition="top center", name=LABELS[m], line_color=COLORS[m])
                     for m in sens["method"].unique()])
    fig.update_layout(xaxis_title="point recall", yaxis_title="point precision", height=380, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Per-channel results")
    st.dataframe(rd["per_channel"][["chan_id", "spacecraft", "method", "precision", "recall", "f1", "event_tp",
                                    "event_fn", "event_fp"]].round(3), use_container_width=True, height=300)


def page_inference(run: str, rd: dict):
    st.header("Inference with a saved model")
    st.info("Scores new telemetry with a detector saved during training (its scaler and threshold are reused, "
            "never refitted). " + DISCLAIMER)
    run_models = MODELS_DIR / run
    chans = sorted(p.name for p in (run_models / "lstm").glob("*")) or sorted(rd["channels"])
    c1, c2, c3 = st.columns(3)
    chan = c1.selectbox("Channel model", chans, key="inf_chan")
    method = c2.selectbox("Method", METHODS, format_func=LABELS.get, key="inf_method")
    source = c3.radio("Input", ["Upload CSV", "Benchmark test file (historical)", "Simulated signal"])
    try:
        bundle = load_bundle(run_models, method, chan)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    st.caption(f"Model expects {bundle.n_features} columns: `value` + `cmd_0..cmd_{bundle.n_features - 2}` "
               f"(commands optional). Needs > {warmup_length(bundle)} rows of history.")
    df = None
    if source == "Upload CSV":
        up = st.file_uploader("CSV with a 'value' column", type="csv")
        template = pd.DataFrame({"value": [0.0] * 3, **{f"cmd_{i}": [0] * 3 for i in range(bundle.n_features - 1)}})
        st.download_button("Download CSV template", template.to_csv(index=False), "telemetry_template.csv")
        if up is not None:
            try:
                df = pd.read_csv(up)
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                st.error(f"Could not read CSV: {exc}")
    elif source == "Benchmark test file (historical)":
        try:
            arr = load_channel(RAW_DIR, chan, "test")
            df = pd.DataFrame(arr, columns=["value"] + [f"cmd_{i}" for i in range(arr.shape[1] - 1)])
        except FileNotFoundError:
            st.error("Raw data not found - run `python -m src.cli download`.")
    else:
        st.warning("SIMULATION: a synthetic sine wave with an injected spike and level shift. Not NASA data.")
        n = st.slider("Length", 300, 3000, 1000)
        rng = np.random.default_rng(0)
        v = 0.5 * np.sin(np.arange(n) / 30) + 0.02 * rng.standard_normal(n)
        v[int(n * 0.6)] += 1.5
        v[int(n * 0.8): int(n * 0.85)] += 0.6
        df = pd.DataFrame({"value": v})
    if df is None:
        st.write("Provide input to run inference.")
        return
    try:
        arr, warns = frame_to_array(df, bundle.n_features)
        for w in warns:
            st.warning(w)
        with st.spinner("Scoring..."):
            out = detect(bundle, arr)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return
    c = st.columns(3)
    c[0].metric("Rows scored", int(out["scored"].sum()))
    c[1].metric("Flagged anomalous", int(out["is_anomaly"].sum()))
    c[2].metric("Threshold", f"{bundle.threshold:.4g}")
    fig = go.Figure(go.Scattergl(x=out["timestep"], y=out["value"], mode="lines", name="value", line=dict(color="#222", width=1)))
    add_intervals(fig, out["is_anomaly"].to_numpy(), COLORS[method], "predicted anomaly")
    fig.update_layout(height=320, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    t = out.assign(label=0)
    t[f"{method}_score"] = out["score"]
    st.plotly_chart(score_figure(t, method, bundle.threshold), use_container_width=True)
    st.dataframe(out[out["is_anomaly"] == 1].head(500), use_container_width=True, height=250)
    st.download_button("Download results CSV", out.to_csv(index=False), f"{chan}_{method}_scores.csv")


def main():
    st.sidebar.title("Telemetry anomaly detection")
    st.sidebar.caption("NASA SMAP/MSL benchmark (Telemanom) - historical analysis")
    runs_ = available_runs()
    if not runs_:
        st.title("No evaluated runs found")
        st.write("Create one first:")
        st.code("python -m src.cli download\npython -m src.cli validate\npython -m src.cli all --config configs/smoke.yaml")
        return
    if st.sidebar.button("Reload run data", help="Clear cached files after re-running the pipeline"):
        st.cache_data.clear()
        runs_ = available_runs()
    run = st.sidebar.selectbox("Experiment run", runs_, index=runs_.index("full") if "full" in runs_ else 0)
    if run == "smoke":
        st.sidebar.warning("Smoke run: tiny model, 3 channels - metrics are not meaningful.")
    page = st.sidebar.radio("Page", ["Overview", "Telemetry explorer", "Anomaly analysis", "Model comparison", "Inference"])
    rd = load_run(run)
    {"Overview": page_overview, "Telemetry explorer": page_explorer, "Anomaly analysis": page_anomalies,
     "Model comparison": page_comparison, "Inference": page_inference}[page](run, rd)
    st.sidebar.markdown("---")
    st.sidebar.caption(DISCLAIMER)


main()
