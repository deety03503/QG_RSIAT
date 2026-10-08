"""Optional callback-driven CUDA batch-size probe with a conservative fallback."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
import math
from typing import Any


def probe_batch_size(
    run_step: Callable[[int], Any],
    candidates: Iterable[int],
    *,
    fallback_batch: int,
    safety_fraction: float = 0.9,
    logger: logging.Logger | None = None,
) -> int:
    """Return the largest passing candidate, applying margin only after OOM.

    `run_step(batch_size)` owns model/input construction and one forward/backward.
    The probe does not run unless explicitly called by its runtime integrator.
    Non-OOM failures are logged and return the configured fallback rather than
    hiding or reclassifying them as an out-of-memory condition.
    """
    if fallback_batch < 1:
        raise ValueError("fallback_batch must be positive")
    if (
        isinstance(safety_fraction, bool)
        or not isinstance(safety_fraction, (int, float))
        or not math.isfinite(safety_fraction)
        or not 0 < safety_fraction <= 1
    ):
        raise ValueError("safety_fraction must be in (0, 1]")

    log = logger or logging.getLogger(__name__)
    raw_candidates = list(candidates)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in raw_candidates
    ):
        raise ValueError("all candidates must be positive integers")
    ordered = sorted(set(raw_candidates))
    if not ordered:
        raise ValueError("candidates must contain at least one positive batch size")

    try:
        import torch
    except ImportError as exc:
        log.warning("Batch probe unavailable because PyTorch could not be imported: %s", exc)
        return fallback_batch

    if not torch.cuda.is_available():
        return fallback_batch

    best: int | None = None
    oom_observed = False
    for batch_size in ordered:
        try:
            run_step(batch_size)
            best = batch_size
        except torch.cuda.OutOfMemoryError:
            oom_observed = True
            log.info("CUDA batch probe reached OOM at batch size %d", batch_size)
            break
        except Exception as exc:
            log.warning(
                "CUDA batch probe failed for a non-OOM reason; using fallback batch %d: %s",
                fallback_batch,
                exc,
            )
            best = None
            break
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                log.warning("Could not clear CUDA cache after batch probe: %s", exc)

    if best is None:
        if oom_observed:
            if ordered[0] == 1:
                raise RuntimeError("CUDA out of memory even at batch size 1; no safe batch was found")
            return max(1, min(fallback_batch, ordered[0] // 2))
        return fallback_batch
    safe_batch = (
        max(1, math.floor(best * safety_fraction))
        if oom_observed
        else best
    )
    return safe_batch
