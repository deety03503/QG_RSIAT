"""Centralized Python, NumPy, and PyTorch random seeding."""

from __future__ import annotations

import logging
import random


def seed_everything(
    seed: int = 1993,
    *,
    rank: int = 0,
    deterministic: bool = False,
    logger: logging.Logger | None = None,
) -> int:
    if seed < 0 or rank < 0:
        raise ValueError("seed and rank must be non-negative")
    resolved_seed = seed + rank
    random.seed(resolved_seed)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to seed the data pipeline") from exc
    np.random.seed(resolved_seed)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to seed model initialization") from exc
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log = logger or logging.getLogger(__name__)
    log.info("Random generators seeded with %d (base=%d, rank=%d)", resolved_seed, seed, rank)
    return resolved_seed
