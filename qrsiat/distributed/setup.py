"""Explicit process-group lifecycle for torchrun-launched jobs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class DistributedState:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    initialized: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    mode: str,
    *,
    timeout_minutes: int = 30,
    logger: logging.Logger | None = None,
) -> DistributedState:
    """Initialize only when invoked under torchrun; failures are actionable."""
    if mode != "ddp":
        return DistributedState()
    if timeout_minutes < 1:
        raise ValueError("timeout_minutes must be positive")

    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise RuntimeError("DDP requires an installed PyTorch distribution") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA; no CUDA device is available")
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "DDP must be launched with torchrun; missing environment variables: "
            + ", ".join(missing)
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2 or not 0 <= rank < world_size:
        raise RuntimeError(
            f"Invalid torchrun topology: rank={rank}, world_size={world_size}"
        )
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside the visible CUDA device range "
            f"[0, {torch.cuda.device_count()})"
        )

    torch.cuda.set_device(local_rank)
    log = logger or logging.getLogger(__name__)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    except Exception as first_error:
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.environ.get("NCCL_P2P_DISABLE") == "1":
            raise RuntimeError("NCCL process-group initialization failed") from first_error
        log.warning("NCCL initialization failed; retrying with NCCL_P2P_DISABLE=1: %s", first_error)
        os.environ["NCCL_P2P_DISABLE"] = "1"
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=timeout_minutes),
            )
        except Exception as retry_error:
            raise RuntimeError(
                "NCCL initialization failed on retry. Relaunch with --mode single "
                "or inspect CUDA/NCCL topology."
            ) from retry_error

    return DistributedState(rank, local_rank, world_size, True)


def cleanup_distributed() -> None:
    """Destroy an initialized process group; safe to call from finally blocks."""
    try:
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
