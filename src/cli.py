"""Command-line entry point.

    python -m src.cli demo                           # START HERE: offline demo on SIMULATED data (~30 s, no download)
    python -m src.cli download                       # fetch + verify the benchmark (idempotent)
    python -m src.cli validate                       # data checks -> reports/data_validation.json
    python -m src.cli train    --config configs/smoke.yaml
    python -m src.cli evaluate --config configs/smoke.yaml
    python -m src.cli all      --config configs/default.yaml   # train + evaluate
    python -m src.cli predict  --run full --method lstm --channel P-1 --input my.csv
    python -m src.cli all      --config configs/telemanom.yaml   # Telemanom-aligned forecaster + dynamic threshold
    python -m src.cli derive   --config configs/lstm_dynamic.yaml # reference LSTM models + dynamic threshold (no training)
    python -m src.cli calibration --run full             # threshold uncertainty + leave-channels-out study
    python -m src.cli compare                            # cross-run benchmark -> reports/benchmark/
    python -m src.cli showcase                           # small committed app artifact -> reports/showcase/
    python -m src.cli seeds --runs full full_seed1 full_seed2   # mean/std over seed-variance runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import load_config, resolve_path
from src.utils.reproducibility import get_logger

log = get_logger("cli")
VALIDATION_REPORT = "reports/data_validation.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "validate", "train", "evaluate", "all", "predict",
                                                   "calibration", "seeds", "demo", "derive", "compare",
                                                   "showcase"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="re-download even if files verify")
    parser.add_argument("--from-dir", help="[download] import an existing local copy (e.g. /kaggle/input) after "
                                           "checking every file against the committed checksums")
    parser.add_argument("--methods", nargs="*", help="subset of methods to train")
    parser.add_argument("--resume", action="store_true", help="[train/all] skip channels finished by an interrupted run")
    parser.add_argument("--shard", help="[train] I/N: only every N-th channel starting at I (parallel processes)")
    parser.add_argument("--run", help="[predict] run name under models/")
    parser.add_argument("--method", help="[predict] baseline | isolation_forest | lstm | telemanom")
    parser.add_argument("--channel", help="[predict] channel id, e.g. P-1")
    parser.add_argument("--input", help="[predict] CSV with a 'value' column (+ optional cmd_* columns)")
    parser.add_argument("--output", help="[predict] where to write the result CSV")
    parser.add_argument("--runs", nargs="*", help="[seeds] run names under reports/runs/ that differ only in seed")
    parser.add_argument("--n-boot", type=int, default=100, help="[calibration] bootstrap resamples")
    args = parser.parse_args(argv)
    if args.command == "demo":                       # self-contained: its own config, no NASA data needed
        from src.demo import DemoRefused, run_demo

        try:
            run_demo()
        except DemoRefused as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0
    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg["data"]["raw_dir"])

    if args.command == "download":
        from src.data.download import download_dataset, find_dataset_dir, import_local_copy

        if args.from_dir:
            src_dir = find_dataset_dir(Path(args.from_dir))
            if src_dir is None:
                print(f"error: no benchmark copy found under {args.from_dir}", file=sys.stderr)
                return 2
            import_local_copy(src_dir, raw_dir)
        else:
            download_dataset(raw_dir, force=args.force)
    elif args.command == "validate":
        from src.data.validation import validate_dataset

        report = validate_dataset(raw_dir)
        out = resolve_path(VALIDATION_REPORT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print(json.dumps({k: report[k] for k in ("ok", "errors", "warnings", "summary")}, indent=1))
        return 0 if report["ok"] else 1
    elif args.command == "showcase":
        from src.showcase import build_showcase

        build_showcase(cfg)
    elif args.command == "compare":
        from src.evaluation.compare import DEFAULT_RUNS, write_benchmark

        reports = resolve_path(cfg["output"]["reports_dir"])
        write_benchmark(reports, reports.parent / "benchmark", args.runs or DEFAULT_RUNS)
        print((reports.parent / "benchmark" / "benchmark.md").read_text())
    elif args.command == "derive":
        from src.evaluation.evaluate import evaluate_run
        from src.pipeline import run_derived

        metrics = evaluate_run(run_derived(cfg, resume=args.resume))
        print(metrics["comparison_markdown"])
    elif args.command in ("train", "evaluate", "all"):
        from src.pipeline import run_paths

        if args.command in ("train", "all"):
            from src.pipeline import run_training

            from src.pipeline import parse_shard

            run_training(cfg, args.methods, resume=args.resume, shard=parse_shard(args.shard))
            if args.shard:
                return 0
        if args.command in ("evaluate", "all"):
            from src.evaluation.evaluate import evaluate_run

            metrics = evaluate_run(run_paths(cfg)[1])
            print(metrics["comparison_markdown"])
    elif args.command in ("calibration", "seeds"):
        from src.evaluation.calibration import run_calibration_studies, seed_summary
        from src.models.persistence import RUN_NAME, check_name

        reports = resolve_path(cfg["output"]["reports_dir"])
        try:
            names = [check_name(r, RUN_NAME, "run name") for r in ([args.run] if args.command == "calibration" else args.runs or [])]
        except ValueError as exc:
            parser.error(str(exc))
        if not names or (args.command == "seeds" and len(names) < 2):
            parser.error("calibration needs --run; seeds needs --runs with at least two runs")
        if args.command == "calibration":
            run_calibration_studies(reports / names[0], n_boot=args.n_boot)
            print((reports / names[0] / "calibration" / "calibration.md").read_text())
        else:
            out = seed_summary([reports / n for n in names])
            dest = reports.parent / "seed_summary.json"         # reports/seed_summary.json, beside (not inside) runs/
            dest.write_text(json.dumps(out, indent=1, default=float))
            print(f"wrote {dest.name} next to {reports.name}/")
            print(json.dumps(out["summary"], indent=1)[:4000])
    elif args.command == "predict":
        import pandas as pd

        from src.inference.predict import detect_from_csv

        if not all([args.run, args.method, args.channel, args.input]):
            parser.error("predict needs --run, --method, --channel and --input")
        from src.models.persistence import RUN_NAME, check_name

        try:
            models_dir = resolve_path(cfg["output"]["models_dir"]) / check_name(args.run, RUN_NAME, "run name")
        except ValueError as exc:
            parser.error(str(exc))
        try:
            result: pd.DataFrame = detect_from_csv(models_dir, args.method, args.channel, args.input)
        except (ValueError, FileNotFoundError) as exc:   # bad input -> clear message + exit code 2, no traceback
            print(f"error: {exc}", file=sys.stderr)
            if isinstance(exc, FileNotFoundError) and "saved" in str(exc):
                print("hint: saved models are generated files, not part of the repository. Recreate them with "
                      f"`python -m src.cli all --config <the run's config> --resume`, or try "
                      "`python -m src.cli demo` and then `--run demo --channel Z-1`.", file=sys.stderr)
            return 2
        if args.output:
            result.to_csv(args.output, index=False)
        n = int(result["is_anomaly"].sum())
        thr = result["threshold"]
        rule = (f"dynamic threshold, median epsilon {thr.median():.4g}" if result["threshold_mode"].iloc[0] == "dynamic"
                else f"threshold {thr.iloc[0]:.4g}")
        print(f"{len(result)} rows scored, {n} flagged anomalous ({rule}).")
        if not args.output:
            print(result[result["is_anomaly"] == 1].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
