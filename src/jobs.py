"""Background pipeline runs started from the dashboard (explicit user click only).

A run is a detached `python -m src.cli all --resume --config <cfg>` subprocess. It keeps going if the browser tab
closes, and `--resume` means an interrupted run continues where it stopped. Guards:
  * only the local-reproduction configs in ALLOWED_CONFIGS can be started (never the committed `full` / `smoke`);
  * a `.job.json` lock records the PID, so a second click while a run is alive is refused;
  * progress = number of per-channel completion files (`scores/<chan>.json`, written last) / expected channels.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ALLOWED_CONFIGS = {"reproduce": "configs/reproduce.yaml", "reproduce_smoke": "configs/reproduce_smoke.yaml"}
EXPECTED_CHANNELS = {"reproduce": 81, "reproduce_smoke": 3}


class JobRefused(RuntimeError):
    pass


def _job_file(root: Path, run: str) -> Path:
    return Path(root) / "reports" / "runs" / run / ".job.json"


def pid_alive(pid: int) -> bool:
    # If we started it (the dashboard process), reap it: an exited child stays a zombie that os.kill() still "finds".
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        return wpid == 0          # 0: our child, still running; pid: it has exited (now reaped)
    except ChildProcessError:
        pass                      # not our child (e.g. the dashboard was restarted): fall back to a signal check
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:          # exists but owned by someone else
        return True
    try:                             # an exited child that its parent has not reaped yet is a zombie: not running
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return not stat.strip().startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


def job_status(root: Path, run: str) -> dict:
    """{state: none|running|finished|failed|stopped, done, total, started, log_tail}"""
    root = Path(root)
    run_dir = root / "reports" / "runs" / run
    total = EXPECTED_CHANNELS.get(run, 0)
    done = len(list((run_dir / "scores").glob("*.json"))) if (run_dir / "scores").exists() else 0
    info = {"state": "none", "done": done, "total": total, "started": None, "log_tail": ""}
    jf = _job_file(root, run)
    if not jf.exists():
        if (run_dir / "metrics.json").exists():
            info["state"] = "finished"
        return info
    job = json.loads(jf.read_text())
    info["started"] = job.get("started")
    log = run_dir / "job.log"
    if log.exists():
        info["log_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
    if pid_alive(job["pid"]):
        info["state"] = "running"
    elif (run_dir / "metrics.json").exists() and done >= total:
        info["state"] = "finished"
    elif "Traceback" in info["log_tail"]:
        info["state"] = "failed"
    else:
        info["state"] = "stopped"          # e.g. machine slept/rebooted: start again to resume
    return info


def start_job(root: Path, run: str, popen=subprocess.Popen) -> dict:
    """Start the configured run in the background. Refuses unknown runs and duplicates of a live run."""
    if run not in ALLOWED_CONFIGS:
        raise JobRefused(f"Only {sorted(ALLOWED_CONFIGS)} can be started from the dashboard")
    root = Path(root)
    if job_status(root, run)["state"] == "running":
        raise JobRefused(f"Run '{run}' is already running")
    run_dir = root / "reports" / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "job.log", "a")
    log.write(f"\n=== started {time.strftime('%Y-%m-%d %H:%M:%S')} from the dashboard ===\n")
    log.flush()
    proc = popen([sys.executable, "-m", "src.cli", "all", "--resume", "--config", ALLOWED_CONFIGS[run]],
                 cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                 start_new_session=True, env={**os.environ, "MPLCONFIGDIR": str(run_dir / ".mpl")})
    job = {"pid": proc.pid, "config": ALLOWED_CONFIGS[run], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _job_file(root, run).write_text(json.dumps(job))
    return job
