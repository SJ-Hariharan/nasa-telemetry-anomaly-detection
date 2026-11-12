"""Logging and random-seed helpers shared by every module."""
from __future__ import annotations

import logging
import os
import random

import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single, concise console handler."""
    root = logging.getLogger("telemetry")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("TELEMETRY_LOG_LEVEL", "INFO"))
    return root.getChild(name)


def set_seed(seed: int) -> None:
    """Seed every random number generator we use (Python, NumPy, PyTorch).

    Note: bit-exact reproducibility of PyTorch on different hardware/OS versions is not
    guaranteed even with a seed; results should match closely on the same machine.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch is optional for data-only commands
        pass
