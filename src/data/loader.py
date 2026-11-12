"""Load telemetry arrays and anomaly labels from the raw benchmark directory.

File format (from the Telemanom README and verified by ``src.data.validation``):
* ``train/<chan>.npy`` and ``test/<chan>.npy``: float array of shape (timesteps, features).
  Column 0 is the telemetry value. Columns 1.. are one-hot "command" indicators
  (24 for SMAP channels, 54 for MSL channels).
* ``labeled_anomalies.csv``: one row per channel. ``anomaly_sequences`` holds
  [start, end] index pairs into the **test** array; both ends are inclusive.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import LABELS_FILE

VALUE_COL = 0  # index of the telemetry value inside each .npy array


def read_raw_labels(raw_dir: Path) -> pd.DataFrame:
    """Read ``labeled_anomalies.csv`` exactly as published, parsing its string-encoded lists."""
    path = Path(raw_dir) / LABELS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m src.cli download` first.")
    df = pd.read_csv(path)
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)  # safe: literals only
    # "class" looks like "[point, contextual]" (unquoted words) so literal_eval can't parse it.
    df["classes"] = df["class"].str.strip("[]").str.split(",").apply(lambda xs: [x.strip() for x in xs])
    return df.drop(columns=["class"])


def merge_intervals(seqs: list[list[int]], classes: list[str]) -> tuple[list[list[int]], list[str]]:
    """Union overlapping/touching inclusive intervals; a merged interval keeps its first class.

    [[5350, 6575], [5300, 6420]] -> [[5300, 6575]]
    """
    merged: list[list[int]] = []
    merged_cls: list[str] = []
    for (s, e), c in sorted(zip(seqs, classes), key=lambda x: x[0][0]):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
            merged_cls.append(c)
    return merged, merged_cls


def load_labels(raw_dir: Path) -> pd.DataFrame:
    """One row per channel, with duplicate channel rows merged.

    The published file lists P-2 twice with overlapping ranges ([5350, 6575] and [5300, 6420]).
    We merge them (interval union) instead of silently keeping one; ``validate_dataset`` reports it.
    """
    df = read_raw_labels(raw_dir)
    rows = []
    for chan, g in df.groupby("chan_id", sort=False):
        seqs = [s for row in g["anomaly_sequences"] for s in row]
        classes = [c for row in g["classes"] for c in row]
        if len(seqs) != len(classes):  # can't pair classes with sequences reliably
            classes = ["unknown"] * len(seqs)
        seqs, classes = merge_intervals(seqs, classes)
        first = g.iloc[0]
        rows.append({"chan_id": chan, "spacecraft": first["spacecraft"], "anomaly_sequences": seqs,
                     "num_values": int(first["num_values"]), "classes": classes, "n_label_rows": len(g)})
    return pd.DataFrame(rows)


def select_channels(labels: pd.DataFrame, channels="all", spacecraft="all") -> list[str]:
    """Return channel ids filtered by config (``"all"`` or explicit list / spacecraft name)."""
    df = labels
    if spacecraft != "all":
        df = df[df["spacecraft"] == spacecraft]
    ids = df["chan_id"].tolist()
    if channels != "all":
        missing = set(channels) - set(ids)
        if missing:
            raise ValueError(f"Unknown channel ids requested: {sorted(missing)}")
        ids = [c for c in ids if c in set(channels)]
    return ids


def load_channel(raw_dir: Path, chan_id: str, split: str) -> np.ndarray:
    """Load one channel's array for ``split`` in {"train", "test"} as float32."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = Path(raw_dir) / split / f"{chan_id}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    # allow_pickle=False: .npy files from the internet must never execute code on load.
    return np.load(path, allow_pickle=False).astype(np.float32)


def label_vector(n: int, sequences: list[list[int]]) -> np.ndarray:
    """Convert [[start, end], ...] (inclusive) into a 0/1 vector of length ``n``.

    Example: n=6, sequences=[[1, 2], [4, 4]] -> [0, 1, 1, 0, 1, 0]
    """
    y = np.zeros(n, dtype=np.int8)
    for start, end in sequences:
        if not (0 <= start <= end < n):
            raise ValueError(f"Anomaly sequence [{start}, {end}] outside test length {n}")
        y[start : end + 1] = 1
    return y
