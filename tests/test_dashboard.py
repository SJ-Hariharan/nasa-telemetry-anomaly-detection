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
    """Also passes on a fresh clone without the gitignored scores/models, as long as the page explains that."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]
    if page in ("Telemetry explorer", "Anomaly analysis", "Inference") and any(
            "not in this checkout" in i.value for i in at.info):
        return          # fresh clone: generated scores/models absent -> the page must explain, not crash
    if page == "Inference":
        # switch input to the clearly-labelled simulated signal and run inference on it
        radio = [r for r in at.radio if "Simulated signal" in r.options][0]
        radio.set_value("Simulated signal").run()
        assert not at.exception, [e.message for e in at.exception]
        assert any("Flagged anomalous" in m.label for m in at.metric)


@pytest.mark.skipif(not RUNS, reason="no evaluated run available")
def test_dashboard_pages_explain_missing_generated_artifacts(tmp_path, monkeypatch):
    """Regression: a fresh clone has run reports but no per-channel scores or models; pages must not crash."""
    import shutil

    from streamlit.testing.v1 import AppTest

    fake = tmp_path / "repo"
    shutil.copytree(ROOT / "app", fake / "app")
    shutil.copytree(ROOT / "src", fake / "src")
    run = RUNS[0]
    src_run = ROOT / "reports" / "runs" / run
    shutil.copytree(src_run, fake / "reports" / "runs" / run, ignore=shutil.ignore_patterns("scores"))
    at = AppTest.from_file(str(fake / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.message for e in at.exception])
        if page in ("Telemetry explorer", "Anomaly analysis", "Inference"):
            assert any("not in this checkout" in i.value for i in at.info), page
