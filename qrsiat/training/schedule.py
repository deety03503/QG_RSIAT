"""Linear warm-up schedules for the RSIAT and QR-RSIAT objectives."""

from __future__ import annotations


def warmup_value(base: float, epoch: int, warmup_epochs: int) -> float:
    if base < 0 or epoch < 0 or warmup_epochs < 0:
        raise ValueError("base, epoch, and warmup_epochs must be non-negative")
    if warmup_epochs == 0:
        return float(base)
    return float(base) * min(1.0, epoch / warmup_epochs)

# Additional task-weight wrappers belong with the objective that owns them.
