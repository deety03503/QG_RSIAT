"""One guarded training epoch with AMP and optional accumulation."""

from __future__ import annotations

from typing import Any


def train_epoch(
    module: Any,
    loader: Any,
    optimizer: Any,
    context: Any,
    *,
    step_kwargs: dict[str, Any] | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")
    module.train()
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    steps = 0
    scaler = context.scaler
    for index, batch in enumerate(loader):
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise ValueError("training batches must contain (index, input, label)")
        _, inputs, labels = batch[:3]
        inputs = inputs.to(context.device, non_blocking=context.device.type == "cuda")
        labels = labels.to(context.device, non_blocking=context.device.type == "cuda")
        with context.autocast():
            outputs = module(inputs, labels, **(step_kwargs or {}))
            loss = outputs.get("loss")
            if loss is None or loss.ndim != 0 or not bool(loss.isfinite()):
                raise FloatingPointError(f"Invalid scalar training loss at batch {index}")
            scaled_loss = loss / grad_accum_steps
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()

        final_batch = index + 1 == len(loader)
        if (index + 1) % grad_accum_steps == 0 or final_batch:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                import torch

                torch.nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for name, value in outputs.items():
            if hasattr(value, "ndim") and value.ndim == 0:
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        steps += 1
    if steps == 0:
        raise RuntimeError("Training loader yielded no batches")
    return {name: value / steps for name, value in totals.items()}
