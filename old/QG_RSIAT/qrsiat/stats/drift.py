"""Feature-space drift estimation and prototype compensation."""

from __future__ import annotations

from typing import Any

from qrsiat.distributed.collectives import all_reduce_sum


def estimate_drift(
    old_features: Any,
    new_features: Any,
    old_prototypes: Any,
    *,
    sigma: float = 4.0,
    distributed: bool = False,
) -> Any:
    """Estimate per-class displacement using globally merged weighted sums."""
    import torch

    if old_features.shape != new_features.shape or old_features.ndim != 2:
        raise ValueError("old_features and new_features must share shape [N,D]")
    if old_prototypes.ndim != 2 or old_prototypes.shape[1] != old_features.shape[1]:
        raise ValueError("old_prototypes must have shape [C,D] matching the features")
    if old_prototypes.shape[0] < 1 or sigma <= 0:
        raise ValueError("old_prototypes and sigma must be positive")
    old = old_features.detach().to(torch.float64)
    new = new_features.detach().to(device=old.device, dtype=torch.float64)
    prototypes = old_prototypes.detach().to(device=old.device, dtype=torch.float64)
    delta = new - old
    distances = (old.unsqueeze(0) - prototypes.unsqueeze(1)).square().sum(dim=2)
    weights = torch.exp(-distances / (2 * sigma * sigma)) + 1e-5
    numerators = weights @ delta
    denominators = weights.sum(dim=1)
    if distributed:
        all_reduce_sum(numerators)
        all_reduce_sum(denominators)
    if torch.any(denominators == 0):
        empty = torch.nonzero(denominators == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Cannot estimate drift for classes with no samples: {empty}")
    return numerators / denominators[:, None]


def compensate_prototypes(prototypes: Any, drift: Any, class_count: int | None = None) -> Any:
    if prototypes.ndim != 2 or drift.ndim != 2 or prototypes.shape[1] != drift.shape[1]:
        raise ValueError("prototypes and drift must have compatible [C,D] shapes")
    count = prototypes.shape[0] if class_count is None else class_count
    if count < 0 or count > prototypes.shape[0] or count > drift.shape[0]:
        raise ValueError("class_count is outside prototype/drift range")
    result = prototypes.clone()
    result[:count] = result[:count] + drift[:count].to(result)
    return result
