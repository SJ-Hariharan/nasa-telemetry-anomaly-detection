"""Integration tests: the full train -> evaluate -> load -> predict path on synthetic data."""
import json

import numpy as np
import pandas as pd

from src.cli import main as cli_main
from src.evaluation.evaluate import evaluate_run
from src.inference.predict import detect_from_csv
from src.pipeline import run_training


def test_end_to_end_pipeline(tiny_config, synthetic_raw_dir, tmp_path):
    reports = run_training(tiny_config)
    summary = json.loads((reports / "training_summary.json").read_text())
    assert summary["failures"] == []
    table = pd.read_parquet(reports / "scores" / "S-1.parquet")
    for m in ("baseline", "isolation_forest", "lstm"):
        assert {f"{m}_score", f"{m}_pred"} <= set(table.columns)
    assert table["label"].sum() == 41                     # 40-point shift + 1-point spike

    metrics = evaluate_run(reports)
    assert metrics["n_channels_evaluated"] == 3
    # The constant-training channel is reported but flagged, and the comparison is also given without it.
    assert set(metrics["uncalibratable_channels"]) == {"C-1"} and metrics["n_calibratable_channels"] == 2
    assert summary["channels"][2]["constant_train"] and not summary["channels"][2]["methods"]["baseline"]["calibratable"]
    for m in ("baseline", "isolation_forest", "lstm"):
        o, c = metrics["results"][m]["overall"], metrics["results"][m]["calibratable"]
        assert 0.0 <= o["f1"] <= 1.0 and o["event_tp"] + o["event_fn"] == 5 and c["event_tp"] + c["event_fn"] == 4
        assert c["n_channels"] == 2 and o["roc_auc_macro_ci"][0] <= o["roc_auc_macro"] <= o["roc_auc_macro_ci"][1]
    from src.models.persistence import load_bundle
    assert load_bundle(tmp_path / "models" / "pytest", "baseline", "C-1").calibratable is False
    assert load_bundle(tmp_path / "models" / "pytest", "lstm", "S-1").calibratable is True
    per = pd.read_csv(reports / "per_channel.csv")
    assert {"roc_auc", "pr_auc", "ks_val_test", "calibratable", "val_len"} <= set(per.columns)
    assert per["ks_val_test"].between(0, 1).all() and (reports / "scores" / "S-1.val.parquet").exists()
    # The injected anomalies are blatant: the simple baseline must catch at least one event.
    assert metrics["results"]["baseline"]["overall"]["event_tp"] >= 1
    for f in ("comparison.csv", "comparison_calibratable.csv", "events.csv", "false_alarms.csv", "sensitivity.csv", "figures/metric_comparison.png"):
        assert (reports / f).exists(), f

    # Inference from saved models reproduces the pipeline's test scores exactly.
    test = np.load(synthetic_raw_dir / "test" / "S-1.npy")
    csv = tmp_path / "new.csv"
    pd.DataFrame(test, columns=["value"] + [f"cmd_{i}" for i in range(test.shape[1] - 1)]).to_csv(csv, index=False)
    models_dir = tmp_path / "models" / "pytest"
    for m in ("baseline", "isolation_forest", "lstm"):
        out = detect_from_csv(models_dir, m, "S-1", csv)
        np.testing.assert_allclose(out["score"], table[f"{m}_score"], rtol=1e-4, equal_nan=True)
        assert (out["is_anomaly"].to_numpy() == table[f"{m}_pred"].to_numpy()).all()


def test_thresholds_do_not_depend_on_test_labels(tiny_config, synthetic_raw_dir):
    """Changing the TEST labels must not change any learned threshold or score."""
    r1 = json.loads((run_training(tiny_config, ["baseline", "isolation_forest"]) / "training_summary.json").read_text())
    csv = synthetic_raw_dir / "labeled_anomalies.csv"
    csv.write_text(csv.read_text().replace("[[600, 639], [850, 850]]", "[[10, 20], [30, 30]]"))
    r2 = json.loads((run_training(tiny_config, ["baseline", "isolation_forest"]) / "training_summary.json").read_text())
    for c1, c2 in zip(r1["channels"], r2["channels"]):
        for m in ("baseline", "isolation_forest"):
            assert c1["methods"][m]["threshold"] == c2["methods"][m]["threshold"]


def test_cli_validate(synthetic_raw_dir, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"data:\n  raw_dir: {synthetic_raw_dir}\n")
    out = tmp_path / "validation.json"
    monkeypatch.setattr("src.cli.VALIDATION_REPORT", str(out))   # don't overwrite the real report
    assert cli_main(["validate", "--config", str(cfg)]) == 0
    assert json.loads(out.read_text())["ok"]


def test_resume_skips_completed_channels_and_matches_full_run(tiny_config):
    full = json.loads((run_training(tiny_config, ["baseline", "isolation_forest"]) / "training_summary.json").read_text())
    reports = run_training(tiny_config, ["baseline", "isolation_forest"])
    (reports / "scores" / "M-1.json").unlink()            # simulate an interruption before M-1 finished
    resumed = json.loads((run_training(tiny_config, ["baseline", "isolation_forest"], resume=True)
                          / "training_summary.json").read_text())
    for a, b in zip(full["channels"], resumed["channels"]):
        assert {m: v["threshold"] for m, v in a["methods"].items()} == {m: v["threshold"] for m, v in b["methods"].items()}
    tiny_config["seed"] = 7                                  # a different config must refuse to resume
    import pytest
    with pytest.raises(ValueError):
        run_training(tiny_config, ["baseline"], resume=True)


def test_cli_predict_rejects_bad_names_and_input(tmp_path, capsys):
    import pytest

    csv = tmp_path / "x.csv"
    csv.write_text("value\n0.1\n")
    with pytest.raises(SystemExit):                       # argparse error for a path-like run name
        cli_main(["predict", "--run", "../models", "--method", "lstm", "--channel", "P-1", "--input", str(csv)])
    assert cli_main(["predict", "--run", "full", "--method", "lstm", "--channel", "../P-1", "--input", str(csv)]) == 2
    assert "Invalid channel id" in capsys.readouterr().err


def test_calibration_and_seed_studies_on_a_run(tiny_config):
    from src.evaluation.calibration import run_calibration_studies, seed_summary

    methods = ["baseline", "isolation_forest"]
    r1 = run_training(tiny_config, methods)
    evaluate_run(r1)
    out = run_calibration_studies(r1, n_boot=10)
    assert (r1 / "calibration" / "calibration.md").exists() and (r1 / "calibration" / "threshold_bootstrap.csv").exists()
    lco = out["leave_channels_out"]
    assert sum(len(f) for f in lco["protocol"]["folds"]) == 3
    assert lco["baseline"]["all"]["default"]["event_tp"] == json.loads((r1 / "metrics.json").read_text())[
        "results"]["baseline"]["overall"]["event_tp"]          # default rule == the run's own thresholds
    import shutil
    other = r1.parent / "pytest_seed7"
    shutil.copytree(r1, other)
    cfg = json.loads((other / "config_used.json").read_text()); cfg["seed"] = 7
    (other / "config_used.json").write_text(json.dumps(cfg))
    s = seed_summary([r1, other])
    assert s["seeds"] == [7, 42] and s["summary"]["overall/baseline"]["f1"]["std"] == 0.0
    # CLI writes the summary to <reports_dir>/../seed_summary.json (reports/seed_summary.json by default)
    import yaml
    cfg_file = r1.parent.parent / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({"data": {"raw_dir": str(r1.parent)},
                                        "output": {"reports_dir": str(r1.parent), "models_dir": str(r1.parent)}}))
    assert cli_main(["seeds", "--config", str(cfg_file), "--runs", r1.name, other.name]) == 0
    assert (r1.parent.parent / "seed_summary.json").exists() and not (r1.parent / "seed_summary.json").exists()
