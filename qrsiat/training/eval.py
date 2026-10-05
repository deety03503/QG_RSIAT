"""Exact-count evaluation helpers for single-process and DDP runs."""

from __future__ import annotations

from typing import Any

from qrsiat.distributed.collectives import all_reduce_sum


def evaluate_accuracy(
    module: Any,
    loader: Any,
    device: Any,
    *,
    top_k: int = 5,
) -> dict[str, float | int]:
    import torch

    if top_k < 1:
        raise ValueError("top_k must be positive")
    was_training = module.training
    module.eval()
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    total = torch.zeros(1, dtype=torch.int64, device=device)
    actual_k = 0
    with torch.inference_mode():
        for batch in loader:
            if len(batch) < 3:
                raise ValueError("evaluation batches must contain (index, input, label)")
            _, inputs, labels = batch[:3]
            inputs = inputs.to(device, non_blocking=getattr(device, "type", "") == "cuda")
            labels = labels.to(device, non_blocking=getattr(device, "type", "") == "cuda")
            output = module(inputs)
            logits = output["logits"] if isinstance(output, dict) else output
            actual_k = min(top_k, int(logits.shape[1]))
            if actual_k < 1:
                raise ValueError("Model returned no class logits")
            predictions = logits.topk(actual_k, dim=1).indices
            correct += predictions.eq(labels[:, None]).any(dim=1).sum()
            total += labels.numel()
    all_reduce_sum(correct)
    all_reduce_sum(total)
    if int(total) == 0:
        raise RuntimeError("Evaluation loader yielded no samples")
    if was_training:
        module.train()
    return {
        f"top{actual_k}": float((100.0 * correct.float() / total).item()),
        "samples": int(total.item()),
        "top_k": actual_k,
    }
