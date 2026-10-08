"""Numerically stable, mergeable float64 class sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qrsiat.distributed.collectives import all_reduce_sum


@dataclass(frozen=True)
class ClassStatistics:
    counts: Any
    means: Any
    covariances: Any


class ClassStatisticsAccumulator:
    def __init__(self, num_classes: int, feature_dim: int, *, device: Any = "cpu") -> None:
        import torch

        if num_classes < 1 or feature_dim < 1:
            raise ValueError("num_classes and feature_dim must be positive")
        self.device = torch.device(device)
        self.counts = torch.zeros(num_classes, dtype=torch.float64, device=self.device)
        self.sums = torch.zeros(num_classes, feature_dim, dtype=torch.float64, device=self.device)
        self.cross_products = torch.zeros(
            num_classes, feature_dim, feature_dim, dtype=torch.float64, device=self.device
        )

    def update(self, features: Any, labels: Any) -> None:
        import torch

        if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
            raise ValueError("features must be [N,D] and labels must be [N]")
        if features.shape[1] != self.sums.shape[1]:
            raise ValueError(
                f"Expected feature dimension {self.sums.shape[1]}, got {features.shape[1]}"
            )
        values = features.detach().to(device=self.device, dtype=torch.float64)
        targets = labels.detach().to(device=self.device, dtype=torch.long)
        if targets.numel() and (
            int(targets.min()) < 0 or int(targets.max()) >= self.counts.numel()
        ):
            raise ValueError("Class label outside accumulator range")
        self.counts.index_add_(0, targets, torch.ones_like(targets, dtype=torch.float64))
        self.sums.index_add_(0, targets, values)
        for class_id in torch.unique(targets):
            mask = targets == class_id
            class_values = values[mask]
            self.cross_products[class_id] += class_values.T @ class_values

    def finalize(self, *, covariance_epsilon: float = 1e-3, distributed: bool = True) -> ClassStatistics:
        import torch

        if covariance_epsilon < 0:
            raise ValueError("covariance_epsilon must be non-negative")
        counts, sums, products = self.counts.clone(), self.sums.clone(), self.cross_products.clone()
        if distributed:
            all_reduce_sum(counts)
            all_reduce_sum(sums)
            all_reduce_sum(products)
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten()
        if missing.numel():
            raise ValueError(f"Cannot compute class statistics; no features for class(es) {missing.tolist()}")
        means = sums / counts[:, None]
        centered = products - counts[:, None, None] * means[:, :, None] * means[:, None, :]
        denominator = (counts - 1).clamp_min(1)
        covariances = centered / denominator[:, None, None]
        eye = torch.eye(means.shape[1], dtype=torch.float64, device=self.device)
        covariances = covariances + covariance_epsilon * eye
        return ClassStatistics(counts, means, covariances)
