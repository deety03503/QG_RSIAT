"""Logging context manager for elapsed wall-clock time."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def timed(
    label: str,
    *,
    logger: logging.Logger | None = None,
) -> Iterator[None]:
    log = logger or logging.getLogger(__name__)
    started = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s took %.3f seconds", label, time.perf_counter() - started)
