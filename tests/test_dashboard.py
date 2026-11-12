"""Headless dashboard check: every page renders without exceptions for an existing run.

Skipped when no evaluated run exists (e.g. a fresh clone before `python -m src.cli all ...`).
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = sorted(p.parent.name for p in (ROOT / "reports" / "runs").glob("*/metrics.json"))
PAGES = ["Overview", "Telemetry explorer", "Anomaly analysis", "Model comparison", "Inference"]


@pytest.mark.skipif(not RUNS, reason="no evaluated run available")
@pytest.mark.parametrize("page", PAGES)
def test_dashboard_page_renders(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]
    if page == "Inference":
        # switch input to the clearly-labelled simulated signal and run inference on it
        radio = [r for r in at.radio if "Simulated signal" in r.options][0]
        radio.set_value("Simulated signal").run()
        assert not at.exception, [e.message for e in at.exception]
        assert any("Flagged anomalous" in m.label for m in at.metric)
