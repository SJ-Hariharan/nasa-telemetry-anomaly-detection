"""Headless dashboard tests (Streamlit AppTest).

Most tests run the app from a FAKE repository copy (app/, src/, committed reports only) so the outcome never depends
on what happens to be downloaded or trained on this machine. Raw telemetry used in the "data present" test is a tiny
fixture in the benchmark layout, only to exercise the code path.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Overview", "Channel explorer", "Events & errors", "Models & calibration", "Data & reproducibility",
         "Inference"]
HAS_FULL = (ROOT / "reports" / "runs" / "full" / "metrics.json").exists()
pytestmark = pytest.mark.skipif(not HAS_FULL, reason="committed full-run reports missing")


def _fake_repo(tmp: Path, calibration=True, seeds=True, demo=False) -> Path:
    """app + src + COMMITTED NASA reports only (no raw data, no scores, no models): a fresh clone."""
    repo = tmp / "repo"
    shutil.copytree(ROOT / "app", repo / "app")
    shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "reports" / "runs").mkdir(parents=True)
    shutil.copy(ROOT / "reports" / "data_validation.json", repo / "reports")
    if seeds:
        shutil.copy(ROOT / "reports" / "seed_summary.json", repo / "reports")
    ignore = shutil.ignore_patterns("scores", *(() if calibration else ("calibration",)))
    shutil.copytree(ROOT / "reports" / "runs" / "full", repo / "reports" / "runs" / "full", ignore=ignore)
    if demo:     # a simulated run: same file layout, config marked simulated
        d = repo / "reports" / "runs" / "demo"
        shutil.copytree(ROOT / "reports" / "runs" / "full", d, ignore=shutil.ignore_patterns("scores", "calibration"))
        cfg = json.loads((d / "config_used.json").read_text())
        cfg.update(run_name="demo", simulated=True)
        (d / "config_used.json").write_text(json.dumps(cfg))
    return repo


def _app(repo: Path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(repo / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _goto(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, (page, [e.message for e in at.exception])
    return at


def _text(at) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(i.value for i in at.info)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_in_this_checkout(page):
    _goto(_app(ROOT), page)


def test_default_route_is_the_nasa_overview_and_demo_is_hidden(tmp_path):
    at = _app(_fake_repo(tmp_path, demo=True))
    assert at.sidebar.radio(key="page").value == "Overview"
    assert at.sidebar.selectbox(key="run").value == "full"
    assert not any("demo" in o for o in at.sidebar.selectbox(key="run").options)     # hidden by default
    assert "Spacecraft Telemetry Anomaly Lab" in _text(at) and "committed report" in _text(at)
    assert not any("SIMULATED" in e.value for e in at.error)
    assert "0.387" in str(at.dataframe[0].value.loc["LSTM forecaster", "Point F1"])[:5]


def test_demo_is_only_shown_when_enabled_and_always_labelled(tmp_path):
    at = _app(_fake_repo(tmp_path, demo=True))
    at.toggle(key="show_demo").set_value(True).run()
    opts = at.sidebar.selectbox(key="run").options
    assert any("demo" in o and "SIMULATED" in o for o in opts)
    at.sidebar.selectbox(key="run").set_value("demo").run()
    for page in PAGES:
        _goto(at, page)
        assert any("SIMULATED DEMO" in e.value for e in at.error), page


def test_fresh_clone_explains_missing_data_and_models(tmp_path):
    at = _app(_fake_repo(tmp_path))
    _goto(at, "Channel explorer")
    txt = _text(at)
    assert "Telemetry: not downloaded" in txt and "python -m src.cli download" in txt
    assert "Model scores: not generated" in txt
    assert len(at.get("plotly_chart")) >= 1                     # committed label timeline still shown
    _goto(at, "Inference")
    assert "Saved models are created by training" in _text(at)
    _goto(at, "Data & reproducibility")
    assert any(b.label == "Download and verify NASA data" for b in at.button)
    assert [b for b in at.button if b.label == "Run data validation"][0].disabled


def test_downloaded_raw_data_is_plotted_with_source_label(tmp_path):
    repo = _fake_repo(tmp_path)
    raw = repo / "data" / "raw" / "telemanom"
    (raw / "train").mkdir(parents=True)
    (raw / "test").mkdir()
    info = json.loads((repo / "reports" / "data_validation.json").read_text())["channels"]["P-1"]
    rng = np.random.default_rng(0)
    for split, n in (("train", info["train_len"]), ("test", info["test_len"])):
        np.save(raw / split / "P-1.npy", np.column_stack([rng.standard_normal(n), np.zeros((n, 24))]).astype(np.float32))
    (raw / "labeled_anomalies.csv").write_text(
        'chan_id,spacecraft,anomaly_sequences,class,num_values\nP-1,SMAP,"[[2149, 2349]]",[contextual],8505\n')
    at = _goto(_app(repo), "Channel explorer")
    assert at.selectbox(key="ex_chan_All").value == "P-1"
    txt = _text(at)
    assert "Telemetry: downloaded NASA source data" in txt and "not downloaded" not in txt
    assert any(s.label == "Sample range" for s in at.slider)


def test_mission_filter_restricts_channels(tmp_path):
    at = _app(_fake_repo(tmp_path))
    at.sidebar.radio(key="mission").set_value("MSL").run()
    _goto(at, "Channel explorer")
    sc = json.loads((ROOT / "reports" / "data_validation.json").read_text())["channels"]
    opts = at.selectbox(key="ex_chan_MSL").options
    assert opts and all(sc[o.split(" · ")[0]]["spacecraft"] == "MSL" for o in opts)
    _goto(at, "Overview")
    assert "Mission: MSL" in _text(at) and "from NASA's MSL Curiosity rover" in _text(at)


def test_missing_optional_reports_do_not_break_pages(tmp_path):
    at = _app(_fake_repo(tmp_path, calibration=False, seeds=False))
    for page in PAGES:
        _goto(at, page)
    _goto(at, "Models & calibration")
    assert "Leave-channels-out" not in _text(at)


def test_no_runs_at_all_shows_recovery_steps(tmp_path):
    repo = tmp_path / "bare"
    shutil.copytree(ROOT / "app", repo / "app")
    shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    at = _app(repo)
    assert "No result reports were found" in _text(at)
    assert any("src.cli download" in c.value for c in at.code)
