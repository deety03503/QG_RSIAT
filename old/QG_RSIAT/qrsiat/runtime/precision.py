"""Autocast and gradient-scaler helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def autocast_context(device: Any, amp_dtype: str, enabled: bool = True) -> Any:
    """Create autocast only for a supported low-precision CUDA execution plan."""
    if not enabled or getattr(device, "type", str(device)) != "cuda":
        return nullcontext()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CUDA autocast") from exc

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    try:
        dtype = dtype_by_name[amp_dtype]
    except KeyError as exc:
        if amp_dtype == "fp32":
            return nullcontext()
        raise ValueError(f"Unsupported autocast dtype: {amp_dtype!r}") from exc
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def full_precision_context(device: Any) -> Any:
    """Disable any enclosing autocast region for numerically sensitive work."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to control autocast") from exc
    device_type = getattr(device, "type", str(device))
    if device_type not in {"cpu", "cuda"}:
        return nullcontext()
    return torch.autocast(device_type=device_type, enabled=False)


def create_grad_scaler(enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for gradient scaling") from exc

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)
