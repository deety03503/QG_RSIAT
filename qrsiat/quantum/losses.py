"""Relational and selectively weighted orthogonality objectives."""

from __future__ import annotations

from typing import Any


def qrel_loss(current_kernel: Any, previous_kernel: Any) -> Any:
    if current_kernel.shape != previous_kernel.shape or current_kernel.ndim != 2:
        raise ValueError("relational kernels must have identical [B,B] shapes")
    return (current_kernel.float() - previous_kernel.detach().float()).square().mean()


def qorth_loss(
    sample_states: Any,
    prototype_states: Any,
    *,
    top_k: int = 3,
    temperature: float = 0.1,
    epsilon: float = 0.2,
) -> Any:
    from .simulator import fidelity_kernel

    if top_k < 1 or temperature <= 0 or epsilon < 0:
        raise ValueError("top_k and temperature must be positive; epsilon must be non-negative")
    fidelities = fidelity_kernel(sample_states, prototype_states)
    if fidelities.shape[1] == 0:
        return fidelities.sum() * 0.0
    k = min(top_k, fidelities.shape[1])
    values, indices = fidelities.topk(k, dim=1)
    selected_weights = torch_softmax(values / temperature, dim=1)
    penalties = (values - epsilon).relu()
    return (selected_weights * penalties).sum(dim=1).mean() + fidelities.sum() * 0.0


def torch_softmax(values: Any, dim: int) -> Any:
    import torch

    return torch.softmax(values.float(), dim=dim)
