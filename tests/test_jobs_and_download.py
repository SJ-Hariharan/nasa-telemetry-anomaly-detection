"""Dashboard-triggered background runs (src/jobs.py) and the downloader's progress callback (offline)."""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from src.jobs import ALLOWED_CONFIGS, JobRefused, job_status, pid_alive, start_job


class _FakeProc:
    def __init__(self, pid):
        self.pid = pid


def test_only_local_reproduction_runs_can_be_started(tmp_path):
    for run in ("full", "smoke", "full_seed1", "../x"):
        with pytest.raises(JobRefused):
            start_job(tmp_path, run, popen=lambda *a, **k: _FakeProc(os.getpid()))
    assert set(ALLOWED_CONFIGS) == {"reproduce", "reproduce_smoke"}


def test_start_job_records_lock_and_refuses_duplicates(tmp_path):
    calls = []

    def popen(cmd, **kw):
        calls.append((cmd, kw))
        return _FakeProc(os.getpid())          # "alive" for the duration of the test

    job = start_job(tmp_path, "reproduce_smoke", popen=popen)
    cmd, kw = calls[0]
    assert cmd[1:] == ["-m", "src.cli", "all", "--resume", "--config", "configs/reproduce_smoke.yaml"]
    assert kw["cwd"] == tmp_path and kw["start_new_session"] is True
    assert json.loads((tmp_path / "reports/runs/reproduce_smoke/.job.json").read_text())["pid"] == job["pid"]
    assert job_status(tmp_path, "reproduce_smoke")["state"] == "running"
    with pytest.raises(JobRefused):
        start_job(tmp_path, "reproduce_smoke", popen=popen)
    assert len(calls) == 1


def test_job_status_states(tmp_path):
    assert job_status(tmp_path, "reproduce")["state"] == "none"
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    d = tmp_path / "reports/runs/reproduce"
    d.mkdir(parents=True)
    (d / ".job.json").write_text(json.dumps({"pid": p.pid, "started": "t"}))
    assert job_status(tmp_path, "reproduce")["state"] == "stopped"            # dead before finishing
    (d / "scores").mkdir()
    for i in range(81):
        (d / "scores" / f"X-{i}.json").write_text("{}")
    (d / "metrics.json").write_text("{}")
    st = job_status(tmp_path, "reproduce")
    assert st["state"] == "finished" and (st["done"], st["total"]) == (81, 81)


def test_exited_child_is_not_reported_running():
    """Regression: an exited, un-reaped child (zombie) must not count as alive."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    import time
    time.sleep(1.0)                      # child exits; we deliberately do not wait() -> zombie until reaped
    assert pid_alive(p.pid) is False


def test_download_reports_progress_for_every_file(tmp_path, monkeypatch):
    import src.data.download as dl

    files = []
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
        data = f"{split}-bytes".encode()
        (tmp_path / split / "X-1.npy").write_bytes(data)
        files.append({"split": split, "file": "X-1.npy", "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "remote_path": f"{split}/X-1.npy"})
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": files}))
    labels = b"chan_id,spacecraft,anomaly_sequences,class,num_values\n"
    (tmp_path / dl.LABELS_FILE).write_bytes(labels)
    monkeypatch.setattr(dl, "LABELS_SHA256", hashlib.sha256(labels).hexdigest())
    monkeypatch.setattr(dl, "_fetch", lambda *a, **k: pytest.fail("no network access expected"))
    seen = []
    dl.download_dataset(tmp_path, progress=lambda d, t, m: seen.append((d, t, m)))
    assert [(d, t) for d, t, _ in seen] == [(1, 2), (2, 2)] and all("verified" in m for *_, m in seen)


def test_downloads_verify_tls_with_certifi_bundle():
    """Regression: python.org macOS Python has no system CA store; downloads must still verify certificates."""
    import ssl

    import certifi

    from src.data.download import ssl_context

    ctx = ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert ctx.cert_store_stats()["x509_ca"] > 0          # a CA bundle was loaded
    assert certifi.where()
