"""The offline first-run demo (`python -m src.cli demo`).

Runs in a clean temporary copy of the repository (only tracked/unignored files: no NASA data, no models, no
generated scores) with networking disabled, exactly as a first-time user would after `git clone`.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SIM = "SIMULATED DEMO - NOT NASA BENCHMARK RESULTS"
NO_NETWORK = """
import socket
def _blocked(*a, **k):
    raise OSError("network disabled in this test")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
"""


# generated/local paths a fresh clone never has (mirrors .gitignore); used when the tests run outside a git checkout
_NOT_IN_CLONE = (".git/", ".venv/", "data/raw/telemanom/", "data/processed/", "data/demo/", "models/full",
                 "models/smoke", "models/demo", "reports/runs/demo/", ".pytest_cache/")


def _clean_copy(dest: Path) -> Path:
    git = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True)
    if git.returncode == 0:
        files = git.stdout.splitlines()
    else:                                    # e.g. a downloaded zip: walk the tree, skip what a clone would not have
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
        files = [f for f in files if not any(f.startswith(x) for x in _NOT_IN_CLONE)
                 and "/scores/" not in f and "__pycache__" not in f]
    for f in files:
        if (ROOT / f).is_file():
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dest / f)
    return dest


def _tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def _run_demo(repo: Path, offline_dir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(offline_dir), "MPLCONFIGDIR": str(offline_dir / "mpl")}
    return subprocess.run([sys.executable, "-m", "src.cli", "demo"], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def demo_repo(tmp_path_factory):
    """Clean checkout copy + no-network sitecustomize; the demo is run once here and reused by the tests."""
    base = tmp_path_factory.mktemp("demo")
    repo = _clean_copy(base / "repo")
    offline = base / "offline"
    offline.mkdir()
    (offline / "sitecustomize.py").write_text(NO_NETWORK)
    full_runs = sorted(p for p in (repo / "reports" / "runs").glob("full*") if p.is_dir())
    before = {p.name: _tree_hash(p) for p in full_runs}
    first = _run_demo(repo, offline)
    return {"repo": repo, "offline": offline, "first": first, "before": before}


def test_clean_copy_has_no_nasa_data_and_network_is_blocked(demo_repo):
    repo = demo_repo["repo"]
    assert not (repo / "data" / "raw" / "telemanom").exists() and not (repo / "models" / "full").exists()
    probe = subprocess.run([sys.executable, "-c", "import socket; socket.create_connection(('example.com', 80), 2)"],
                           env={**os.environ, "PYTHONPATH": str(demo_repo["offline"])}, capture_output=True, text=True)
    assert probe.returncode != 0 and "network disabled" in probe.stderr


def test_demo_runs_offline_and_is_labelled_simulated(demo_repo):
    r, repo = demo_repo["first"], demo_repo["repo"]
    assert r.returncode == 0, r.stderr[-3000:]
    assert SIM in r.stdout and "streamlit run app/dashboard.py" in r.stdout
    cfg = json.loads((repo / "reports/runs/demo/config_used.json").read_text())
    assert cfg["simulated"] is True and cfg["run_name"] == "demo"
    for d in ("data/demo/telemanom_synthetic", "reports/runs/demo", "models/demo"):
        assert SIM in (repo / d / "SIMULATED_DEMO").read_text()


def test_demo_exercises_the_real_pipeline(demo_repo):
    repo = demo_repo["repo"]
    chans = ["Z-1", "Z-2", "Z-3", "Z-4"]
    for m in ("baseline", "isolation_forest", "lstm"):
        for c in chans:
            assert (repo / "models/demo" / m / c / "meta.json").exists()            # saved by the pipeline
    run = repo / "reports/runs/demo"
    metrics = json.loads((run / "metrics.json").read_text())                      # written by evaluate_run
    assert metrics["n_channels_evaluated"] == 4 and metrics["training_failures"] == []
    lstm = metrics["results"]["lstm"]["overall"]
    assert lstm["event_tp"] + lstm["event_fn"] == 8                              # the 8 injected anomalies
    t = pd.read_parquet(run / "scores" / "Z-1.parquet")
    assert t["label"].sum() == 1 + 80 and {"lstm_score", "lstm_pred", "isolation_forest_score"} <= set(t.columns)
    assert json.loads((run / "data_validation.json").read_text())["ok"]
    out = pd.read_csv(run / "sample_output_lstm.csv")                             # written via detect_from_csv
    assert len(out) == 1200 and out["scored"].sum() == 1200 - 50


def test_demo_is_deterministic_and_rerun_safe(demo_repo):
    repo, run = demo_repo["repo"], demo_repo["repo"] / "reports/runs/demo"
    keep = {f: (run / f).read_bytes() for f in ("comparison.csv", "per_channel.csv", "events.csv",
                                                "sample_output_lstm.csv", "sample_output_isolation_forest.csv")}
    thr = {c["chan_id"]: {m: v["threshold"] for m, v in c["methods"].items()}
           for c in json.loads((run / "training_summary.json").read_text())["channels"]}
    again = _run_demo(repo, demo_repo["offline"])                                 # rerun replaces its own outputs
    assert again.returncode == 0, again.stderr[-3000:]
    assert {f: (run / f).read_bytes() for f in keep} == keep
    assert thr == {c["chan_id"]: {m: v["threshold"] for m, v in c["methods"].items()}
                   for c in json.loads((run / "training_summary.json").read_text())["channels"]}


def test_demo_does_not_touch_real_runs(demo_repo):
    repo = demo_repo["repo"]
    assert demo_repo["before"], "the clean copy should contain the committed full-run reports"
    after = {p.name: _tree_hash(p) for p in sorted((repo / "reports" / "runs").glob("full*")) if p.is_dir()}
    assert after == demo_repo["before"]


def test_demo_refuses_to_replace_folders_it_did_not_create(demo_repo):
    repo = demo_repo["repo"]
    marker = repo / "models/demo/SIMULATED_DEMO"
    saved = marker.read_text()
    marker.unlink()                                  # now models/demo looks like someone else's folder
    before = _tree_hash(repo / "models/demo")
    try:
        r = _run_demo(repo, demo_repo["offline"])
        assert r.returncode == 2 and "was not created by" in r.stderr
        assert _tree_hash(repo / "models/demo") == before                         # nothing deleted or changed
    finally:
        marker.write_text(saved)
