from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _root_fidelity(k: torch.Tensor) -> torch.Tensor:
    """Convert fidelity |<a|b>|^2 to overlap amplitude |<a|b>| (avoids the (o1+o2)^2 shrinkage of squared values)."""
    return k.float().clamp_min(1e-8).sqrt()


class _GradScale(torch.autograd.Function):
    """Identity in forward, multiplies the gradient by `scale` in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    return _GradScale.apply(x, scale)


def qrel_loss(k_current: torch.Tensor, k_old: torch.Tensor) -> torch.Tensor:
    """Quantum Relational Distillation Loss: off-diagonal MSE between current and teacher overlap amplitudes."""
    amp_cur = _root_fidelity(k_current)
    amp_old = _root_fidelity(k_old.detach())
    b = amp_cur.shape[0]
    if b <= 1:
        return (amp_cur - amp_old).square().mean()
    mask = ~torch.eye(b, dtype=torch.bool, device=amp_cur.device)
    return (amp_cur[mask] - amp_old[mask]).square().mean()


def qorth_kernel_loss(
    features: torch.Tensor,
    aligned_protos: torch.Tensor,
    kernel: nn.Module,
    top_k: int = 3,
    temperature: float = 0.1,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Selective Quantum-Weighted Orthogonality Loss over top-k nearest old prototypes.

    Uses the overlap amplitude |<a|b>| and a squared hinge, so the gradient is zero at the
    threshold and grows smoothly with the violation. Gradients flow into both arguments
    (the kernel itself is expected to be frozen by the caller).
    """
    if aligned_protos.numel() == 0 or aligned_protos.shape[0] == 0:
        return features.new_zeros(())

    sim = _root_fidelity(kernel(features.float(), aligned_protos.float()))
    k = min(max(int(top_k), 1), int(sim.shape[1]))
    values = sim.topk(k=k, dim=1).values.float()
    temp = max(float(temperature), 1e-6)
    weights = torch.softmax(values / temp, dim=1).detach()
    hinge = F.relu(values - float(epsilon)).square()
    return (weights * hinge).sum(dim=1).mean()
