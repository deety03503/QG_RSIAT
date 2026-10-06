"""DataLoader construction following a resolved runtime plan."""

from __future__ import annotations

from typing import Any


def create_data_loader(
    dataset: Any,
    plan: Any,
    *,
    training: bool,
    batch_size: int | None = None,
    sampler: Any | None = None,
    drop_last: bool | None = None,
    seed: int | None = None,
) -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to create a data loader") from exc

    batch = batch_size or (
        plan.per_device_batch if training else plan.eval_batch_size
    )
    if batch < 1:
        raise ValueError("batch_size must be positive")
    workers = int(plan.num_workers)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch,
        "shuffle": bool(training and sampler is None),
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(plan.pin_memory),
        "drop_last": bool(False if drop_last is None else drop_last),
        "generator": generator,
    }
    if workers:
        options["persistent_workers"] = bool(plan.persistent_workers)
        options["prefetch_factor"] = int(plan.prefetch_factor or 2)
    try:
        return DataLoader(**options)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not construct DataLoader with workers={workers}, "
            f"batch_size={batch}: {exc}"
        ) from exc
