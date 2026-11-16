"""Configuration loading.

Configs are plain YAML dictionaries. A config may say ``inherit: other.yaml`` to start from
another file and override only some keys (used by ``configs/smoke.yaml``).
Paths in the config are relative to the project root, so nothing is machine-specific.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` updated recursively with ``override`` (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``inherit`` chains recursively.

    A relative ``path`` is tried against the current directory first, then the project root.
    """
    path = Path(path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path) if not path.exists() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def display_path(path) -> str:
    """Path relative to the project root for logs and reports (no machine-specific absolute paths)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return p.name


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
