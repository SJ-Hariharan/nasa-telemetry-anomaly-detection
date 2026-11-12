"""Validate the raw benchmark before any modelling.

A model trained on silently broken data produces confident nonsense, so we check the
assumptions the rest of the pipeline relies on and fail loudly (errors) or record
surprises (warnings). The result is saved to ``reports/data_validation.json``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.loader import VALUE_COL, load_channel, load_labels, read_raw_labels
from src.utils.reproducibility import get_logger

log = get_logger("validation")


def validate_dataset(raw_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    raw_labels = read_raw_labels(raw_dir)
    labels = load_labels(raw_dir)
    errors: list[str] = []
    warnings: list[str] = []
    channels: dict[str, dict] = {}

    dup = sorted(set(raw_labels["chan_id"][raw_labels["chan_id"].duplicated()]))
    for chan in dup:
        rows = raw_labels[raw_labels["chan_id"] == chan]
        if rows[["spacecraft", "num_values"]].nunique().max() > 1:
            errors.append(f"{chan}: duplicate label rows disagree on spacecraft/num_values")
        else:
            warnings.append(f"{chan}: {len(rows)} label rows {rows['anomaly_sequences'].tolist()} "
                            "merged into their interval union")
    for r in raw_labels.itertuples(index=False):  # within-row overlaps (before any merging)
        ordered = sorted(r.anomaly_sequences)
        if any(ordered[i][1] >= ordered[i + 1][0] for i in range(len(ordered) - 1)):
            warnings.append(f"{r.chan_id}: overlapping anomaly sequences within one label row")
    unlabelled = sorted({p.stem for p in (raw_dir / "train").glob("*.npy")} - set(labels["chan_id"]))
    if unlabelled:
        warnings.append(f"Telemetry files with no label row (excluded from evaluation): {unlabelled}")

    for row in labels.itertuples(index=False):
        info: dict = {"spacecraft": row.spacecraft}
        try:
            train = load_channel(raw_dir, row.chan_id, "train")
            test = load_channel(raw_dir, row.chan_id, "test")
        except FileNotFoundError as exc:
            errors.append(f"{row.chan_id}: missing file {exc}")
            continue
        info.update(train_len=len(train), test_len=len(test), n_features=train.shape[1])
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
            errors.append(f"{row.chan_id}: unexpected shapes {train.shape} / {test.shape}")
            continue
        for split, arr in (("train", train), ("test", test)):
            n_nan = int((~np.isfinite(arr)).sum())
            if n_nan:
                errors.append(f"{row.chan_id}/{split}: {n_nan} non-finite values")
            v = arr[:, VALUE_COL]
            info[f"{split}_value_min"] = float(v.min())
            info[f"{split}_value_max"] = float(v.max())
            # Consecutive identical rows are normal for telemetry; we record, not remove, them.
            info[f"{split}_repeated_rows_frac"] = float((np.diff(arr, axis=0) == 0).all(axis=1).mean())
            cmds = arr[:, 1:]
            if cmds.size and not np.isin(cmds, (0.0, 1.0)).all():
                warnings.append(f"{row.chan_id}/{split}: command columns are not strictly 0/1")
        if test.shape[0] != row.num_values:
            errors.append(f"{row.chan_id}: test length {test.shape[0]} != num_values {row.num_values}")
        seqs = row.anomaly_sequences
        for s, e in seqs:
            if not (0 <= s <= e < len(test)):
                errors.append(f"{row.chan_id}: anomaly [{s},{e}] outside test range")
        if len(seqs) != len(row.classes):
            warnings.append(f"{row.chan_id}: {len(seqs)} sequences but {len(row.classes)} class labels")
        info["anomalous_frac_test"] = float(sum(e - s + 1 for s, e in seqs) / len(test))
        if info["test_value_min"] < -1.0001 or info["test_value_max"] > 1.0001:
            warnings.append(f"{row.chan_id}: test values in [{info['test_value_min']:.3g}, "
                            f"{info['test_value_max']:.3g}], outside the documented [-1, 1] range")
        info["constant_train"] = bool(train[:, VALUE_COL].std() < 1e-3)
        channels[row.chan_id] = info

    const = sorted(c for c, i in channels.items() if i["constant_train"])
    if const:
        warnings.append(f"{len(const)} channels have a constant telemetry value in TRAIN: {const}")

    summary = {
        "n_channels_labelled": int(len(labels)),
        "n_channels_loaded": len(channels),
        "by_spacecraft": labels["spacecraft"].value_counts().to_dict(),
        "n_anomaly_sequences": int(labels["anomaly_sequences"].apply(len).sum()),
        "n_train_points": int(sum(c["train_len"] for c in channels.values())),
        "n_test_points": int(sum(c["test_len"] for c in channels.values())),
        "n_test_anomalous_points": int(sum(round(c["anomalous_frac_test"] * c["test_len"]) for c in channels.values())),
        "feature_counts": sorted({c["n_features"] for c in channels.values()}),
    }
    report = {"ok": not errors, "errors": errors, "warnings": warnings, "summary": summary, "channels": channels}
    log.info("Validation: %d channels, %d errors, %d warnings", len(channels), len(errors), len(warnings))
    return report
