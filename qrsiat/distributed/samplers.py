"""Distributed samplers that do not duplicate evaluation examples."""

from __future__ import annotations

import math
from typing import Any


class DistributedEvalSampler:
    """Partition dataset indices by rank without padding or duplication."""

    def __init__(
        self,
        dataset: Any,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise RuntimeError("DistributedEvalSampler requires PyTorch") from exc
        active = dist.is_available() and dist.is_initialized()
        self.dataset = dataset
        self.num_replicas = num_replicas or (dist.get_world_size() if active else 1)
        self.rank = rank if rank is not None else (dist.get_rank() if active else 0)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be within a positive num_replicas range")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return max(0, math.ceil((len(self.dataset) - self.rank) / self.num_replicas))
