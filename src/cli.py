"""Command-line entry point.

    python -m src.cli download                       # fetch + verify the benchmark (idempotent)
    python -m src.cli validate                       # data checks -> reports/data_validation.json
    python -m src.cli train    --config configs/smoke.yaml
    python -m src.cli evaluate --config configs/smoke.yaml
    python -m src.cli all      --config configs/default.yaml   # train + evaluate
    python -m src.cli predict  --run full --method lstm --channel P-1 --input my.csv
"""
from __future__ import annotations

import argparse
import json
import sys

from src.utils.config import load_config, resolve_path
from src.utils.reproducibility import get_logger

log = get_logger("cli")
VALIDATION_REPORT = "reports/data_validation.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "validate", "train", "evaluate", "all", "predict"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="re-download even if files verify")
    parser.add_argument("--methods", nargs="*", help="subset of methods to train")
    parser.add_argument("--resume", action="store_true", help="[train/all] skip channels finished by an interrupted run")
    parser.add_argument("--run", help="[predict] run name under models/")
    parser.add_argument("--method", help="[predict] baseline | isolation_forest | lstm")
    parser.add_argument("--channel", help="[predict] channel id, e.g. P-1")
    parser.add_argument("--input", help="[predict] CSV with a 'value' column (+ optional cmd_* columns)")
    parser.add_argument("--output", help="[predict] where to write the result CSV")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg["data"]["raw_dir"])

    if args.command == "download":
        from src.data.download import download_dataset

        download_dataset(raw_dir, force=args.force)
    elif args.command == "validate":
        from src.data.validation import validate_dataset

        report = validate_dataset(raw_dir)
        out = resolve_path(VALIDATION_REPORT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print(json.dumps({k: report[k] for k in ("ok", "errors", "warnings", "summary")}, indent=1))
        return 0 if report["ok"] else 1
    elif args.command in ("train", "evaluate", "all"):
        from src.pipeline import run_paths

        if args.command in ("train", "all"):
            from src.pipeline import run_training

            run_training(cfg, args.methods, resume=args.resume)
        if args.command in ("evaluate", "all"):
            from src.evaluation.evaluate import evaluate_run

            metrics = evaluate_run(run_paths(cfg)[1])
            print(metrics["comparison_markdown"])
    elif args.command == "predict":
        import pandas as pd

        from src.inference.predict import detect_from_csv

        if not all([args.run, args.method, args.channel, args.input]):
            parser.error("predict needs --run, --method, --channel and --input")
        models_dir = resolve_path(cfg["output"]["models_dir"]) / args.run
        result: pd.DataFrame = detect_from_csv(models_dir, args.method, args.channel, args.input)
        if args.output:
            result.to_csv(args.output, index=False)
        n = int(result["is_anomaly"].sum())
        print(f"{len(result)} rows scored, {n} flagged anomalous "
              f"(threshold {result['threshold'].iloc[0]:.4g}).")
        if not args.output:
            print(result[result["is_anomaly"] == 1].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
