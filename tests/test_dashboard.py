"""Headless dashboard tests (Streamlit AppTest).

Most tests run the app from a FAKE repository copy holding only what git commits (app/, src/, reports/ without
score tables, no models/, no data/): exactly what a reviewer gets from a fresh clone.
"""
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Overview", "Compare methods", "Channel timeline", "Detections & errors", "Historical replay", "Inference",
         "Reproduce"]
HAS_ARTIFACTS = (ROOT / "reports" / "benchmark" / "benchmark.json").exists() and \
    (ROOT / "reports" / "showcase" / "index.json").exists()
pytestmark = pytest.mark.skipif(not HAS_ARTIFACTS, reason="committed benchmark/showcase artifacts missing")
COMMITTED_RUNS = ("full", "full_seed1", "full_seed2", "lstm_dynamic", "lstm_dynamic_seed1", "lstm_dynamic_seed2",
                  "telemanom", "telemanom_seed1", "telemanom_seed2")


def _fake_repo(tmp: Path, benchmark=True, showcase=True, runs=True) -> Path:
    repo = tmp / "repo"
    shutil.copytree(ROOT / "app", repo / "app")
    shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "reports" / "runs").mkdir(parents=True)
    if benchmark:
        shutil.copytree(ROOT / "reports" / "benchmark", repo / "reports" / "benchmark")
    if showcase:
        shutil.copytree(ROOT / "reports" / "showcase", repo / "reports" / "showcase",
                        ignore=shutil.ignore_patterns("models"))          # weights are not committed
    if runs:
        for r in COMMITTED_RUNS:
            if (ROOT / "reports" / "runs" / r).exists():
                shutil.copytree(ROOT / "reports" / "runs" / r, repo / "reports" / "runs" / r,
                                ignore=shutil.ignore_patterns("scores", "run.log"))
    return repo


def _app(repo: Path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(repo / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _goto(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, (page, [e.message for e in at.exception])
    return at


def _text(at) -> str:
    return "\n".join([m.value for m in at.markdown] + [i.value for i in at.info] + [c.value for c in at.caption])


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_in_this_checkout(page):
    _goto(_app(ROOT), page)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    return _fake_repo(tmp_path_factory.mktemp("fresh"))


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_from_a_fresh_clone(fresh, page):
    _goto(_app(fresh), page)


def test_overview_states_claims_and_headline(fresh):
    at = _app(fresh)
    txt = _text(at)
    assert "What it does not claim" in txt and "not a live feed" in txt.lower()
    assert len(at.dataframe) >= 1 and "Event F1" in at.dataframe[0].value.columns
    assert len(at.get("plotly_chart")) >= 1


def test_no_control_starts_training_or_downloads(fresh):
    at = _app(fresh)
    for page in PAGES:
        _goto(at, page)
        for b in at.button:
            assert not re.search(r"\b(train|start|run)\b|download and|reproduc", b.label.lower()), b.label


def test_timeline_uses_committed_showcase(fresh):
    at = _goto(_app(fresh), "Channel timeline")
    assert at.radio(key="tl_src").value.startswith("Committed showcase")
    assert len(at.get("plotly_chart")) == 1 and len(at.dataframe) == 1


def test_replay_is_labelled_and_advances(fresh):
    at = _goto(_app(fresh), "Historical replay")
    assert "HISTORICAL REPLAY — NOT LIVE TELEMETRY" in _text(at)
    chan = at.selectbox(key="rp_chan").value
    before = at.session_state[f"rp_t_{chan}"]
    at.button(key="rp_step").click().run()
    assert not at.exception
    assert at.session_state[f"rp_t_{chan}"] == min(before + 70, at.slider[0].max)


def test_inference_explains_missing_models_on_a_fresh_clone(fresh):
    at = _goto(_app(fresh), "Inference")
    assert "model weights are not committed" in _text(at) and "telemanom_smoke" in _text(at)


@pytest.mark.skipif(not (ROOT / "reports" / "showcase" / "models").exists(), reason="local showcase models missing")
def test_inference_scores_local_showcase_model(tmp_path):
    repo = _fake_repo(tmp_path)
    shutil.copytree(ROOT / "reports" / "showcase" / "models", repo / "reports" / "showcase" / "models")
    at = _goto(_app(repo), "Inference")
    assert at.selectbox(key="inf_src").value == "Showcase models (local)"
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Samples scored"].replace(",", "")) > 1000


def test_missing_artifacts_explain_how_to_recreate(tmp_path):
    at = _app(_fake_repo(tmp_path, benchmark=False, showcase=False, runs=False))
    for page in PAGES:
        _goto(at, page)
    _goto(at, "Overview")
    assert "python -m src.cli compare" in _text(at)
    _goto(at, "Historical replay")
    assert "python -m src.cli showcase" in _text(at)
