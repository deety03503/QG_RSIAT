"""Small collectives with single-process identity behavior."""

from __future__ import annotations

from typing import Any


def _distributed() -> tuple[Any, bool]:
    try:
        import torch.distributed as dist
    except ImportError:
        return None, False
    active = dist.is_available() and dist.is_initialized()
    return dist, active


def all_reduce_sum(value: Any) -> Any:
    dist, active = _distributed()
    if active:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def barrier() -> None:
    dist, active = _distributed()
    if active:
        dist.barrier()


def broadcast_object(value: Any, src: int = 0) -> Any:
    dist, active = _distributed()
    if not active:
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def gather_cat(tensor: Any, *, with_grad: bool = False) -> Any:
    """Gather equal-shaped rank tensors along dimension zero."""
    dist, active = _distributed()
    if not active:
        return tensor
    if with_grad:
        try:
            from torch.distributed.nn.functional import all_gather
        except ImportError as exc:
            raise RuntimeError(
                "Gradient-preserving gather requires "
                "torch.distributed.nn.functional.all_gather"
            ) from exc
        return __import__("torch").cat(all_gather(tensor), dim=0)
    import torch

    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)
