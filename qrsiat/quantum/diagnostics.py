"""Small, side-effect-free helpers for inspecting pairwise kernels."""

from __future__ import annotations

from typing import Any


def kernel_diagnostics(kernel_matrix: Any, labels: Any) -> dict[str, Any]:
    """Summarize within/between-class kernel values and centered alignment.

    The returned keys and quantiles follow Appendix A3 in ``agents/PROMT.md``.
    Diagonal entries are excluded from both pair distributions.
    """
    import torch

    matrix = kernel_matrix.detach().float()
    labels = labels.reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel_matrix must be square")
    if matrix.shape[0] != labels.numel():
        raise ValueError("kernel_matrix and labels must have the same length")
    n = matrix.shape[0]
    if n < 1:
        raise ValueError("kernel_matrix must contain at least one sample")

    eye = torch.eye(n, dtype=torch.bool, device=matrix.device)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    different = labels[:, None].ne(labels[None, :])
    quantiles = torch.tensor((0.05, 0.5, 0.95, 0.99), device=matrix.device)
    stats: dict[str, Any] = {
        "F_same": (
            torch.quantile(matrix[same], quantiles).tolist()
            if bool(same.any())
            else []
        ),
        "F_diff": (
            torch.quantile(matrix[different], quantiles).tolist()
            if bool(different.any())
            else []
        ),
    }
    target = labels[:, None].eq(labels[None, :]).to(dtype=matrix.dtype)
    centered_kernel = matrix - matrix.mean(dim=0, keepdim=True)
    centered_kernel = centered_kernel - centered_kernel.mean(dim=1, keepdim=True)
    centered_target = target - target.mean(dim=0, keepdim=True)
    centered_target = centered_target - centered_target.mean(dim=1, keepdim=True)
    stats["cka"] = (
        (centered_kernel * centered_target).sum()
        / (centered_kernel.norm() * centered_target.norm() + 1e-12)
    ).item()
    return stats
