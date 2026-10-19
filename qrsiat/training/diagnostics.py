"""Small, CPU-testable helpers for optional QR-RSIAT diagnostics."""

from __future__ import annotations

from typing import Any, Iterable


def create_epoch_metric_accumulator(device: Any) -> Any:
    """Create float64 sums for loss logging and exact sample counts."""
    import torch

    return torch.zeros(8, device=device, dtype=torch.float64)


def accumulate_step_metrics(
    accumulator: Any,
    loss: Any,
    loss_cls: Any,
    loss_rt: Any,
    components: Any,
    correct: Any,
) -> Any:
    """Add one batch without a device-to-host synchronization."""
    import torch

    accumulator.add_(torch.stack(
        [
            loss.detach().to(torch.float64),
            loss_cls.detach().to(torch.float64),
            loss_rt.detach().to(torch.float64),
            *components.detach().to(torch.float64),
            correct.detach().to(torch.float64),
        ]
    ))
    return accumulator


def kernel_diagnostics(kernel: Any, labels: Any, *, prefix: str = "kernel") -> dict[str, Any]:
    """Summarize within/between-class kernel values and centered alignment."""
    import torch

    matrix = kernel.detach().float()
    labels = labels.detach().reshape(-1)
    if matrix.ndim != 2 or matrix.shape != (labels.numel(), labels.numel()):
        raise ValueError("kernel must be square and match the number of labels")
    n = labels.numel()
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    same = (labels[:, None] == labels[None, :]) & off_diagonal
    different = labels[:, None] != labels[None, :]
    quantiles = torch.tensor([0.05, 0.5, 0.95, 0.99], device=matrix.device)

    def summarize(values: Any) -> list[float] | None:
        if values.numel() == 0:
            return None
        return [float(value) for value in torch.quantile(values, quantiles).cpu()]

    result: dict[str, Any] = {
        f"{prefix}_F_same_q05_q50_q95_q99": summarize(matrix[same]),
        f"{prefix}_F_diff_q05_q50_q95_q99": summarize(matrix[different]),
    }
    if n < 2:
        result[f"{prefix}_cka"] = None
        return result

    target = (labels[:, None] == labels[None, :]).to(matrix.dtype)
    centered_kernel = matrix - matrix.mean(0, keepdim=True) - matrix.mean(1, keepdim=True) + matrix.mean()
    centered_target = target - target.mean(0, keepdim=True) - target.mean(1, keepdim=True) + target.mean()
    denom = centered_kernel.norm() * centered_target.norm()
    result[f"{prefix}_cka"] = float(
        ((centered_kernel * centered_target).sum() / denom.clamp_min(1e-12)).cpu()
    )
    return result


def gradient_diagnostics(
    losses: dict[str, Any],
    reference_name: str,
    parameters: Iterable[Any],
) -> dict[str, Any]:
    """Return gradient norm ratios and cosines against one reference loss."""
    import torch

    params = [parameter for parameter in parameters if parameter.requires_grad]
    result: dict[str, Any] = {}
    if reference_name not in losses or not params:
        return result

    names = list(losses)
    gradients: dict[str, Any] = {}
    for index, name in enumerate(names):
        loss = losses[name]
        if not getattr(loss, "requires_grad", False):
            gradients[name] = None
            continue
        values = torch.autograd.grad(
            loss,
            params,
            retain_graph=index < len(names) - 1,
            allow_unused=True,
        )
        chunks = [value.detach().float().reshape(-1) for value in values if value is not None]
        gradients[name] = torch.cat(chunks) if chunks else None

    reference = gradients.get(reference_name)
    reference_norm = reference.norm() if reference is not None else None
    for name, gradient in gradients.items():
        if gradient is None:
            result[f"grad_{name}_norm"] = 0.0
            if reference_norm is not None:
                result[f"grad_{name}_norm_ratio_to_{reference_name}"] = 0.0
                result[f"grad_{name}_cosine_to_{reference_name}"] = 0.0
            continue
        norm = gradient.norm()
        result[f"grad_{name}_norm"] = float(norm.cpu())
        if reference_norm is not None:
            result[f"grad_{name}_norm_ratio_to_{reference_name}"] = float(
                (norm / reference_norm.clamp_min(1e-12)).cpu()
            )
            dot = torch.dot(gradient, reference)
            cosine = dot / (norm * reference_norm).clamp_min(1e-12)
            result[f"grad_{name}_cosine_to_{reference_name}"] = float(cosine.cpu())
    return result
