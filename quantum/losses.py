from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _root_fidelity(k: torch.Tensor) -> torch.Tensor:
    """Convert fidelity |<a|b>|^2 to overlap amplitude |<a|b>| (avoids the (o1+o2)^2 shrinkage of squared values)."""
    return k.float().clamp_min(1e-8).sqrt()


def _overlap_amplitude(kernel: nn.Module, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Similarity on a common [0, 1] 'overlap' scale where 0 means orthogonal.

    - quantum: sqrt of state fidelity = |<psi_a|psi_b>|.
    - cosine: max(cos, 0) on raw features. PairwiseKernel's (cos + 1) / 2 is NOT used here because
      it maps orthogonal vectors to 0.5, which would make the hinge threshold epsilon meaningless.
    - others (rbf / mlp): kernel output as-is.
    """
    name = getattr(kernel, "name", "quantum")
    if name == "quantum":
        return _root_fidelity(kernel(a.float(), b.float()))
    if name == "cosine":
        left = F.normalize(a.float(), p=2, dim=1)
        right = F.normalize(b.float(), p=2, dim=1)
        return (left @ right.T).clamp_min(0.0)
    return kernel(a.float(), b.float()).float().clamp(0.0, 1.0)


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


def qrel_loss(
    k_current: torch.Tensor,
    k_old: torch.Tensor,
    top_k: int = 5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Top-K Quantum Relational Distillation Loss.

    Focuses relational alignment on each sample's top-k nearest neighbors (according to the
    frozen teacher `k_old`) weighted by a temperature-scaled softmax, combined with a relational
    KL term over the top-k neighborhood distribution so gradients are not squashed to ~1e-4.
    """
    amp_cur = _root_fidelity(k_current)
    amp_old = _root_fidelity(k_old.detach())
    b = amp_cur.shape[0]
    if b <= 1:
        return (amp_cur - amp_old).abs().mean()

    eye_mask = torch.eye(b, dtype=torch.bool, device=amp_cur.device)
    off_diag_old = amp_old.masked_fill(eye_mask, -1.0)

    k = min(max(int(top_k), 1), b - 1)
    top_vals_old, top_idx = off_diag_old.topk(k=k, dim=1)
    top_vals_cur = amp_cur.gather(dim=1, index=top_idx)

    temp = max(float(temperature), 1e-6)
    p_old = torch.softmax(top_vals_old / temp, dim=1).detach()
    log_p_cur = F.log_softmax(top_vals_cur / temp, dim=1)

    # 1. Weighted Smooth-L1 (scaled to L1 magnitude) on top-k overlap amplitudes
    diff = (top_vals_cur - top_vals_old).abs()
    loss_amp = (p_old * diff).sum(dim=1).mean()

    # 2. Temperature-scaled relational KL divergence over the top-k neighborhood
    loss_kl = F.kl_div(log_p_cur, p_old, reduction="batchmean") * temp

    return loss_amp + loss_kl


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
    threshold and grows smoothly with the violation. The kernel is expected to be frozen and
    `aligned_protos` to be detached by the caller (fixed anchors), so gradients flow only into `features`.
    """
    if aligned_protos.numel() == 0 or aligned_protos.shape[0] == 0:
        return features.new_zeros(())

    sim = _overlap_amplitude(kernel, features, aligned_protos)
    k = min(max(int(top_k), 1), int(sim.shape[1]))
    values = sim.topk(k=k, dim=1).values.float()
    temp = max(float(temperature), 1e-6)
    weights = torch.softmax(values / temp, dim=1).detach()
    hinge = F.relu(values - float(epsilon)).square()
    return (weights * hinge).sum(dim=1).mean()
