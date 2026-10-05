"""Opt-in runtime optimizations with explicit eager/safe fallbacks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any


def configure_backend(plan: Any, logger: logging.Logger | None = None) -> None:
    """Apply backend flags from the plan; unsupported settings remain disabled."""
    log = logger or logging.getLogger(__name__)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to configure runtime backends") from exc

    if plan.mode == "cpu":
        return
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(plan.tf32)
        torch.backends.cudnn.allow_tf32 = bool(plan.tf32)
        torch.backends.cudnn.benchmark = bool(plan.cudnn_benchmark)
    except Exception as exc:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.benchmark = previous_benchmark
        log.warning("Backend optimization setup failed; restored previous flags: %s", exc)


def has_fused_attention() -> bool:
    try:
        import torch.nn.functional as functional
    except ImportError:
        return False
    return callable(getattr(functional, "scaled_dot_product_attention", None))


def make_compiled_callable(
    module: Any,
    *,
    enabled: bool = False,
    logger: logging.Logger | None = None,
) -> Callable[..., Any]:
    """Compile a call target lazily and permanently fall back to eager on error.

    The original module remains the owner of parameters/state_dict; callers
    should continue to register and optimize that original module.
    """
    log = logger or logging.getLogger(__name__)
    if not enabled:
        return module
    try:
        import torch

        compiled = torch.compile(module)
    except Exception as exc:
        log.warning("torch.compile setup failed; using eager execution: %s", exc)
        return module

    state = {"enabled": True}

    def call(*args: Any, **kwargs: Any) -> Any:
        if state["enabled"]:
            try:
                return compiled(*args, **kwargs)
            except Exception as exc:
                state["enabled"] = False
                log.warning("Compiled execution failed; retrying eagerly: %s", exc)
        return module(*args, **kwargs)

    return call


def enable_gradient_checkpointing(
    module: Any,
    *,
    enabled: bool = False,
    logger: logging.Logger | None = None,
) -> bool:
    """Enable a supported checkpointing hook or keep the eager safe path."""
    if not enabled:
        return False
    log = logger or logging.getLogger(__name__)
    for method_name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
        method = getattr(module, method_name, None)
        if callable(method):
            try:
                method()
                return True
            except Exception as exc:
                log.warning(
                    "Gradient checkpointing setup failed; leaving it disabled: %s",
                    exc,
                )
                return False
    log.warning("Module has no supported gradient-checkpointing hook; leaving it disabled.")
    return False


class FrozenParameterCast:
    """Keep original parameter storage available for an execution fallback."""

    def __init__(self, changed: list[tuple[Any, Any]]) -> None:
        self._changed = changed

    def restore(self) -> bool:
        if not self._changed:
            return False
        for parameter, original_cpu in reversed(self._changed):
            parameter.data = original_cpu.to(device=parameter.device)
        self._changed.clear()
        return True


def cast_frozen_parameters(
    module: Any,
    dtype: Any,
    *,
    enabled: bool = False,
    logger: logging.Logger | None = None,
) -> FrozenParameterCast | None:
    """Cast frozen parameters and retain storage for an execution fallback."""
    if not enabled:
        return None
    log = logger or logging.getLogger(__name__)
    changed: list[tuple[Any, Any]] = []
    try:
        for parameter in module.parameters():
            if not parameter.requires_grad and parameter.is_floating_point():
                original_cpu = parameter.data.detach().to(device="cpu", copy=True)
                changed.append((parameter, original_cpu))
                parameter.data = parameter.data.to(dtype=dtype)
    except Exception as exc:
        for parameter, original in reversed(changed):
            parameter.data = original.to(device=parameter.device)
        log.warning("Frozen-backbone cast failed; restored original dtypes: %s", exc)
        return None
    return FrozenParameterCast(changed) if changed else None
