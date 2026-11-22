"""Visual system for the dashboard: palette, Plotly template, CSS and small HTML helpers.

Path-free on purpose (the dashboard owns all file locations). Colours are chosen for contrast on the dark
background (text >= 4.5:1, chart marks >= 3:1) and so that method colours never collide with the label colour.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio

BG, PANEL, BORDER = "#0B1220", "#141E31", "#26324A"
TEXT, MUTED = "#E6EDF7", "#9AA8BF"
ACCENT = "#4FC3E8"
LABEL = "#FF6B6B"          # expert-labelled anomaly intervals
THRESH = "#FFB547"         # thresholds
TRACE, TRACE_TRAIN = "#D6E2F5", "#7F8DA6"
METHOD_COLORS = {"baseline": "#A0AEC4", "isolation_forest": "#2EC4B6", "lstm": "#B794FF", "lstm_dynamic": "#F28DD3",
                 "telemanom": "#6BD98A"}
METHOD_LABELS = {"baseline": "Rolling z-score", "isolation_forest": "Isolation Forest", "lstm": "LSTM (reference)",
                 "lstm_dynamic": "LSTM + dynamic threshold", "telemanom": "Telemanom-aligned"}
MISSION_COLORS = {"SMAP": "#4FC3E8", "MSL": "#F59E7B"}
# Redundant (non-colour) encodings so methods stay distinguishable for colour-blind readers and in greyscale.
METHOD_PATTERNS = {"baseline": "", "isolation_forest": "/", "lstm": "x", "lstm_dynamic": "\\", "telemanom": "."}
METHOD_SYMBOLS = {"baseline": "circle", "isolation_forest": "diamond", "lstm": "square", "lstm_dynamic": "triangle-up",
                  "telemanom": "star"}
PLOT_CONFIG = {"displaylogo": False, "displayModeBar": "hover"}

pio.templates["lab"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=13, color=TEXT),
    colorway=[METHOD_COLORS["baseline"], METHOD_COLORS["isolation_forest"], METHOD_COLORS["lstm"], ACCENT, THRESH],
    xaxis=dict(gridcolor="rgba(230,237,247,0.07)", zerolinecolor="rgba(230,237,247,0.15)", linecolor=BORDER,
               tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED))),
    yaxis=dict(gridcolor="rgba(230,237,247,0.07)", zerolinecolor="rgba(230,237,247,0.15)", linecolor=BORDER,
               tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED))),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT), orientation="h", y=-0.22, x=0),
    hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT)),
    margin=dict(l=8, r=8, t=36, b=8), title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left"),
))
pio.templates.default = "lab"

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h2 {{ font-size: 1.35rem !important; margin-top: 1.4rem !important; }}
h3 {{ font-size: 1.08rem !important; color: {TEXT}; }}
.lab-hero {{ border: 1px solid {BORDER}; border-radius: 14px; padding: 1.4rem 1.6rem 1.2rem;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(79,195,232,0.12), transparent 60%), {PANEL};
  margin-bottom: 1.1rem; }}
.lab-eyebrow {{ color: {ACCENT}; font-size: .78rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }}
.lab-title {{ font-size: 1.9rem; font-weight: 700; color: {TEXT}; margin: .25rem 0 .35rem; line-height: 1.2; }}
.lab-sub {{ color: {MUTED}; font-size: .98rem; max-width: 70ch; }}
.lab-chips {{ margin: .7rem 0 .9rem; display: flex; flex-wrap: wrap; gap: .4rem; }}
.chip {{ display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px; padding: .18rem .65rem;
  font-size: .78rem; font-weight: 600; border: 1px solid; line-height: 1.5; }}
.chip-report {{ color: #9FD9F0; border-color: rgba(79,195,232,.45); background: rgba(79,195,232,.10); }}
.chip-source {{ color: #8FE3B0; border-color: rgba(72,199,116,.45); background: rgba(72,199,116,.10); }}
.chip-local {{ color: #D2BEFF; border-color: rgba(183,148,255,.45); background: rgba(183,148,255,.10); }}
.chip-missing {{ color: {MUTED}; border-color: {BORDER}; background: transparent; border-style: dashed; }}
.chip-sim {{ color: #FFC2C2; border-color: rgba(255,107,107,.6); background: rgba(255,107,107,.14); }}
.chip-hist {{ color: #FFD9A0; border-color: rgba(255,181,71,.45); background: rgba(255,181,71,.10); }}
.callout {{ border: 1px solid {BORDER}; border-left: 3px solid {THRESH}; border-radius: 10px; padding: .75rem .95rem;
  background: {PANEL}; height: 100%; }}
.callout b {{ color: {TEXT}; }} .callout p {{ color: {MUTED}; margin: .25rem 0 0; font-size: .9rem; }}
.step {{ border: 1px solid {BORDER}; border-radius: 12px; padding: .9rem 1.05rem; background: {PANEL}; margin-bottom: .6rem; }}
.step .n {{ color: {ACCENT}; font-weight: 700; margin-right: .4rem; }}
.muted {{ color: {MUTED}; font-size: .9rem; }}
div[data-testid="stMetric"] {{ border: 1px solid {BORDER}; border-radius: 12px; padding: .7rem .9rem; background: {PANEL}; }}
div[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .82rem !important; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] .lab-brand {{ font-weight: 700; font-size: 1.05rem; color: {TEXT}; }}
section[data-testid="stSidebar"] .lab-brand span {{ color: {ACCENT}; }}
.replay-banner {{ border: 2px solid {THRESH}; background: rgba(255,181,71,.12); color: #FFE2B3; border-radius: 10px;
  padding: .6rem 1rem; font-weight: 700; margin-bottom: .8rem; letter-spacing: .02em; }}
.sim-banner {{ border: 2px solid {LABEL}; background: rgba(255,107,107,.14); color: #FFD6D6; border-radius: 10px;
  padding: .7rem 1rem; font-weight: 600; margin-bottom: .9rem; }}
@media (max-width: 640px) {{
  div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) {{ flex-direction: row !important;
    flex-wrap: wrap !important; gap: .5rem !important; }}
  div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) > div {{ min-width: calc(50% - .5rem) !important;
    flex: 1 1 calc(50% - .5rem) !important; width: calc(50% - .5rem) !important; }}
  .lab-title {{ font-size: 1.45rem; }} .lab-hero {{ padding: 1rem; }} .block-container {{ padding-top: 3.4rem; }}
}}
</style>
"""

CHIP_KIND = {"report": "chip-report", "source": "chip-source", "local": "chip-local", "missing": "chip-missing",
             "sim": "chip-sim", "hist": "chip-hist"}


def chip(text: str, kind: str = "report") -> str:
    return f'<span class="chip {CHIP_KIND[kind]}">{html.escape(text)}</span>'


def chips(*items: tuple[str, str]) -> str:
    return '<div class="lab-chips">' + "".join(chip(t, k) for t, k in items) + "</div>"


def hero(eyebrow: str, title: str, sub: str, chip_items=()) -> str:
    return (f'<div class="lab-hero"><div class="lab-eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="lab-title">{html.escape(title)}</div><div class="lab-sub">{sub}</div>'
            f'{chips(*chip_items) if chip_items else ""}</div>')


def callout(title: str, body: str) -> str:
    return f'<div class="callout"><b>{html.escape(title)}</b><p>{body}</p></div>'


def step(n, title: str, body: str) -> str:
    num = f'<span class="n">{html.escape(str(n))}</span>' if n is not None else ""
    return f'<div class="step">{num}<b>{html.escape(title)}</b><div class="muted">{body}</div></div>'
