from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def qrel_loss(k_current: torch.Tensor, k_old: torch.Tensor) -> torch.Tensor:
    """Quantum Relational Distillation Loss: off-diagonal MSE between current and teacher kernel matrices."""
    b = k_current.shape[0]
    if b <= 1:
        return (k_current.float() - k_old.float().detach()).square().mean()
    mask = ~torch.eye(b, dtype=torch.bool, device=k_current.device)
    return (k_current.float()[mask] - k_old.float().detach()[mask]).square().mean()


def qorth_kernel_loss(
    features: torch.Tensor,
    aligned_protos: torch.Tensor,
    kernel: nn.Module,
    top_k: int = 3,
    temperature: float = 0.1,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Selective Quantum-Weighted Orthogonality Loss over top-k nearest old prototypes."""
    if aligned_protos.numel() == 0 or aligned_protos.shape[0] == 0:
        return features.new_zeros(())

    sim = kernel(features.float(), aligned_protos.float().detach())
    k = min(max(int(top_k), 1), int(sim.shape[1]))
    values = sim.topk(k=k, dim=1).values.float()
    temp = max(float(temperature), 1e-6)
    weights = torch.softmax(values / temp, dim=1).detach()
    hinge = F.relu(values - float(epsilon))
    return (weights * hinge).sum(dim=1).mean()

