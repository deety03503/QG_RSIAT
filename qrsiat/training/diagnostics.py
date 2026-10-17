"""Tensor-only helpers for low-synchronization epoch accounting."""

from __future__ import annotations

from typing import Any


def diagnostics_due(enabled: bool, is_main: bool, epoch: int, epochs: int) -> bool:
    return bool(enabled and is_main and epochs > 0 and epoch in {0, epochs - 1})


def accumulate_float64(total: Any, value: Any) -> Any:
    """Accumulate a scalar loss without copying it to the host."""
    total.add_(value.detach().to(dtype=total.dtype))
    return total


def rt_loss_components(output: dict[str, Any], beta: float) -> tuple[Any, dict[str, Any]]:
    """Return the legacy combined regularizer and detached log components."""
    components = {
        "loss_rs": output["loss_rs"],
        "loss_rel": output["loss_rel"],
        "loss_align": output["loss_align"] * beta,
        "loss_orth": output["loss_orth"],
    }
    total = (
        output["loss_rs"] + output["loss_rel"] + output["loss_align"] * beta
        + output["loss_orth"]
    )
    return total, {name: value.detach() for name, value in components.items()}
