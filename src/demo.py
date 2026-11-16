"""Offline first-run demo on SIMULATED telemetry (no NASA data, no network, ~30 s on a laptop CPU).

    python -m src.cli demo

What it does, using the project's real code paths (nothing is canned):
  1. generates 4 synthetic channels in the exact benchmark layout (train/test .npy + labeled_anomalies.csv)
     under data/demo/telemanom_synthetic/, with known injected anomalies in the test part
  2. validates them with src.data.validation.validate_dataset
  3. trains and scores the three detectors with src.pipeline.run_training (configs/demo.yaml)
  4. evaluates with src.evaluation.evaluate.evaluate_run
  5. scores a fresh synthetic sample with the saved models through src.inference.predict.detect_from_csv

Everything it writes is marked simulated: config_used.json has "simulated": true, and each output folder holds a
SIMULATED_DEMO marker file. THE RESULTS SAY NOTHING ABOUT NASA BENCHMARK PERFORMANCE OR REAL SPACECRAFT: the anomalies
are deliberately easy, hand-made patterns. The demo shows how the pipeline works end to end, nothing more.

Safety: the demo only ever writes the run named "demo" and data/demo/. On a rerun it replaces those folders only if
they carry its marker file; anything else under those names is left alone and the demo refuses to run.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT, display_path, load_config, resolve_path
from src.utils.reproducibility import get_logger

log = get_logger("demo")
DEMO_CONFIG = "configs/demo.yaml"
MARKER = "SIMULATED_DEMO"
LABEL = "SIMULATED DEMO - NOT NASA BENCHMARK RESULTS"
N_CMD = 4                     # command columns (the real benchmark has 24 or 54)
TRAIN_LEN, TEST_LEN = 2000, 2000


class DemoRefused(RuntimeError):
    """A folder the demo would replace exists but was not created by the demo."""


# ------------------------------------------------------------------------------------------- generator
def _commands(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((n, N_CMD)) < 0.01).astype(np.float32)


def _signal(kind: str, n: int, rng: np.random.Generator, cmds: np.ndarray, t0: int = 0) -> np.ndarray:
    t = np.arange(t0, t0 + n)
    noise = rng.standard_normal(n)
    if kind == "sine":                 # periodic signal + noise
        return 0.5 * np.sin(2 * np.pi * t / 60) + 0.03 * noise
    if kind == "commanded":            # flat, with a decaying step after each command on cmd_0
        v = np.zeros(n)
        level = 0.0
        for i in range(n):
            level = level * np.exp(-1 / 40) + (0.4 if cmds[i, 0] else 0.0)
            v[i] = level
        return v - 0.3 + 0.02 * noise
    if kind == "two_tone":             # slow + fast periodic components
        return 0.3 * np.sin(2 * np.pi * t / 200) + 0.2 * np.sin(2 * np.pi * t / 37) + 0.02 * noise
    if kind == "sawtooth":             # repeating ramp
        return 0.8 * ((t % 80) / 80 - 0.5) + 0.02 * noise
    raise ValueError(kind)


# kind, list of (start, end_inclusive, anomaly type, class) injected into the TEST part only
CHANNELS = {
    "Z-1": ("sine", [(700, 700, "spike", "point"), (1300, 1379, "level_shift", "contextual")]),
    "Z-2": ("commanded", [(900, 979, "uncommanded_step", "contextual"), (1600, 1600, "spike", "point")]),
    "Z-3": ("two_tone", [(1100, 1159, "noise_burst", "contextual"), (1600, 1699, "frequency_change", "contextual")]),
    "Z-4": ("sawtooth", [(800, 899, "stuck_sensor", "contextual"), (1500, 1500, "spike", "point")]),
}


def _inject(v: np.ndarray, kind: str, start: int, end: int, rng: np.random.Generator, t0: int) -> None:
    s = slice(start, end + 1)
    if kind == "spike":
        v[s] += 1.2 if v[start] < 0.2 else -1.2
    elif kind == "level_shift":
        v[s] += 0.6
    elif kind == "uncommanded_step":   # a step response with no command behind it
        v[s] += 0.4 * np.exp(-np.arange(end - start + 1) / 40)
    elif kind == "noise_burst":
        v[s] += 0.25 * rng.standard_normal(end - start + 1)
    elif kind == "frequency_change":
        t = np.arange(t0 + start, t0 + end + 1)
        v[s] = 0.3 * np.sin(2 * np.pi * t / 200) + 0.2 * np.sin(2 * np.pi * t / 12)
    elif kind == "stuck_sensor":
        v[s] = v[start]
    else:
        raise ValueError(kind)


def generate_demo_dataset(out_dir: Path, seed: int = 0) -> Path:
    """Write the synthetic benchmark-layout dataset. Deterministic for a given seed."""
    out_dir = Path(out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(exist_ok=True)
    rows = ["chan_id,spacecraft,anomaly_sequences,class,num_values"]
    for i, (chan, (kind, anomalies)) in enumerate(CHANNELS.items()):
        rng = np.random.default_rng([seed, i])
        cmd_tr, cmd_te = _commands(TRAIN_LEN, rng), _commands(TEST_LEN, rng)
        train = _signal(kind, TRAIN_LEN, rng, cmd_tr)
        test = _signal(kind, TEST_LEN, rng, cmd_te, t0=TRAIN_LEN)
        for start, end, a_kind, _ in anomalies:
            _inject(test, a_kind, start, end, rng, TRAIN_LEN)
        np.save(out_dir / "train" / f"{chan}.npy", np.column_stack([train, cmd_tr]).astype(np.float32))
        np.save(out_dir / "test" / f"{chan}.npy", np.column_stack([test, cmd_te]).astype(np.float32))
        seqs = [[s, e] for s, e, _, _ in anomalies]
        classes = ", ".join(c for *_, c in anomalies)
        rows.append(f'{chan},SIM,"{seqs}","[{classes}]",{TEST_LEN}')
    (out_dir / "labeled_anomalies.csv").write_text("\n".join(rows) + "\n")
    (out_dir / MARKER).write_text(f"{LABEL}\nGenerated by src/demo.py (seed {seed}). Not NASA data.\n")
    return out_dir


def sample_input(seed: int = 1) -> tuple[pd.DataFrame, list[list[int]]]:
    """A NEW synthetic Z-1-like sequence (not seen in training) for the inference step, with 2 injected anomalies."""
    rng = np.random.default_rng([seed, 99])
    n = 1200
    cmds = _commands(n, rng)
    v = _signal("sine", n, rng, cmds, t0=5000)
    injected = [[500, 500], [800, 859]]
    _inject(v, "spike", 500, 500, rng, 5000)
    _inject(v, "level_shift", 800, 859, rng, 5000)
    df = pd.DataFrame(np.column_stack([v, cmds]), columns=["value"] + [f"cmd_{i}" for i in range(N_CMD)])
    return df, injected


# ------------------------------------------------------------------------------------------- safety
def _prepare(path: Path) -> None:
    """Make ``path`` empty for the demo: remove it only if it is the demo's own (marker present)."""
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()) or path.name not in ("demo", "telemanom_synthetic"):
        raise DemoRefused(f"refusing to touch unexpected path {path}")
    if path.exists():
        if not (path / MARKER).exists():
            raise DemoRefused(f"{display_path(path)} exists but was not created by `python -m src.cli demo` "
                              f"(no {MARKER} marker). Move or rename it, then run the demo again.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / MARKER).write_text(f"{LABEL}\nCreated by `python -m src.cli demo`. Safe to delete.\n")


# ------------------------------------------------------------------------------------------- runner
def run_demo(config_path: str = DEMO_CONFIG) -> dict:
    from src.data.validation import validate_dataset
    from src.evaluation.evaluate import evaluate_run
    from src.inference.predict import detect_from_csv
    from src.pipeline import run_paths, run_training

    cfg = load_config(config_path)
    if not cfg.get("simulated") or cfg["run_name"] != "demo":
        raise DemoRefused(f"{config_path} must set simulated: true and run_name: demo")
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    models_dir, reports_dir = run_paths(cfg)
    for p in (raw_dir, models_dir, reports_dir):          # all checks first: nothing is touched if any fails
        if p.exists() and not (p / MARKER).exists():
            raise DemoRefused(f"{display_path(p)} exists but was not created by `python -m src.cli demo` "
                              f"(no {MARKER} marker). Move or rename it, then run the demo again.")
    for p in (raw_dir, models_dir, reports_dir):
        _prepare(p)

    print(f"\n=== {LABEL} ===\n1/5 generating synthetic telemetry -> {display_path(raw_dir)}")
    generate_demo_dataset(raw_dir, seed=cfg["seed"])
    print("2/5 validating it with the project's data checks")
    report = validate_dataset(raw_dir)
    if not report["ok"]:
        raise RuntimeError(f"synthetic data failed validation: {report['errors']}")
    print("3/5 training + scoring baseline, Isolation Forest and LSTM (real pipeline, small settings)")
    run_training(cfg)
    (reports_dir / "data_validation.json").write_text(json.dumps(report, indent=1))
    print("4/5 evaluating against the injected (known) anomalies")
    metrics = evaluate_run(reports_dir)
    print("5/5 scoring a new synthetic sample with the saved models (inference path)")
    df, injected = sample_input()
    df.to_csv(reports_dir / "sample_input.csv", index=False)
    (reports_dir / "sample_input_injected.json").write_text(json.dumps(
        {"channel_model": "Z-1", "injected_anomalies": injected, "note": LABEL}, indent=1))
    inference = {}
    for method in ("baseline", "isolation_forest", "lstm"):
        out = detect_from_csv(models_dir, method, "Z-1", reports_dir / "sample_input.csv")
        out.to_csv(reports_dir / f"sample_output_{method}.csv", index=False)
        flags = out["is_anomaly"].to_numpy()
        inference[method] = {"flagged": int(flags.sum()),
                             "injected_hit": [bool(flags[s : e + 1].any()) for s, e in injected]}
    (reports_dir / "sample_inference_summary.json").write_text(json.dumps(inference, indent=1))

    print(f"\n=== {LABEL} ===")
    print(metrics["comparison_markdown"].split("\n\n")[1] if "\n\n" in metrics["comparison_markdown"]
          else metrics["comparison_markdown"])
    print("\nInference on the new sample (channel model Z-1, 2 injected anomalies):")
    for m, r in inference.items():
        print(f"  {m:17s} flagged {r['flagged']:4d} timesteps; injected anomalies hit: {r['injected_hit']}")
    print(f"""
These numbers come from easy, hand-made synthetic anomalies. They demonstrate the workflow only and are NOT
evidence about NASA benchmark performance (see EXPERIMENTS.md for the real results).

Next steps:
  streamlit run app/dashboard.py      # pick the run 'demo (SIMULATED)' in the sidebar, then try every page
  python -m src.cli predict --run demo --method lstm --channel Z-1 --input {display_path(reports_dir / 'sample_input.csv')}
Real benchmark (needs ~170 MB download and ~1 h CPU): see README, Quick start step 3 ("Reproduce the NASA benchmark").
""")
    return {"metrics": metrics, "inference": inference, "reports_dir": reports_dir, "models_dir": models_dir}
