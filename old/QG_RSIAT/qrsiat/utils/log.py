"""Small logger factory that avoids duplicate handlers."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str = "qrsiat",
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(handler, "_qrsiat_stream", False) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler._qrsiat_stream = True  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        has_file = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == Path(resolved)
            for handler in logger.handlers
        )
        if not has_file:
            logger.addHandler(logging.FileHandler(path, encoding="utf-8"))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    return logger
