"""Per-process runtime state derived from an explicit runtime plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware.plan import RuntimePlan
from .precision import autocast_context, create_grad_scaler


@dataclass(frozen=True)
class RuntimeContext:
    device: Any
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    amp_dtype: str
    scaler: Any | None

    def autocast(self) -> Any:
        return autocast_context(self.device, self.amp_dtype)


def create_runtime_context(
    plan: RuntimePlan,
    *,
    rank: int = 0,
    local_rank: int | None = None,
) -> RuntimeContext:
    """Resolve a process-local device; never assume a CUDA device exists."""
    if rank < 0 or rank >= plan.world_size:
        raise ValueError(f"rank {rank} is outside world_size {plan.world_size}")
    resolved_local_rank = rank if local_rank is None else local_rank
    if resolved_local_rank < 0:
        raise ValueError("local_rank must be non-negative")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to create RuntimeContext") from exc

    if plan.mode == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Runtime mode {plan.mode!r} requires CUDA, but CUDA is unavailable"
            )
        device_index = (
            resolved_local_rank if plan.mode == "ddp" else plan.device_index
        )
        if device_index is None:
            raise ValueError(f"Runtime plan mode {plan.mode!r} has no device index")
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Planned CUDA device {device_index} is not visible "
                f"(visible count: {torch.cuda.device_count()})"
            )
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)

    scaler = create_grad_scaler(plan.use_grad_scaler)
    return RuntimeContext(
        device=device,
        rank=rank,
        local_rank=resolved_local_rank,
        world_size=plan.world_size,
        is_main=rank == 0,
        amp_dtype=plan.amp_dtype,
        scaler=scaler,
    )
