"""Comparable pairwise kernels with a common [B,B] interface."""

from __future__ import annotations

from typing import Any

from torch import nn


class PairwiseKernel(nn.Module):
    def __init__(
        self,
        name: str = "cosine",
        *,
        input_dim: int = 768,
        n_qubits: int = 8,
        layers: int = 2,
        centering: bool = True,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if name not in {"quantum", "cosine", "rbf", "mlp"}:
            raise ValueError(f"Unsupported kernel {name!r}")
        self.name = name
        self.quantum = None
        self.projector = None
        if name == "quantum":
            from .feature_map import QuantumFeatureMap

            self.quantum = QuantumFeatureMap(input_dim, n_qubits, layers, centering=centering)
        elif name == "mlp":
            width = hidden_dim or input_dim
            self.projector = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.Linear(width, input_dim),
            )
        if name == "rbf":
            import torch

            self.register_buffer("gamma", torch.tensor(1.0 / input_dim))

    def __call__(self, features: Any, other: Any | None = None) -> Any:
        return super().__call__(features, other)

    def forward(self, features: Any, other: Any | None = None) -> Any:
        import torch
        import torch.nn.functional as functional

        right = features if other is None else other
        if features.ndim != 2 or right.ndim != 2 or features.shape[1] != right.shape[1]:
            raise ValueError("kernel inputs must have compatible [B,D] shapes")
        if self.name == "quantum":
            from .simulator import fidelity_kernel

            if self.quantum is None:
                raise RuntimeError("Quantum kernel was not initialized")
            states, _ = self.quantum(features)
            right_states = states if other is None else self.quantum(right)[0]
            return fidelity_kernel(states, right_states)
        if self.projector is not None:
            features = self.projector(features)
            right = features if other is None else self.projector(right)
        left_norm = functional.normalize(features.float(), dim=1)
        right_norm = left_norm if other is None else functional.normalize(right.float(), dim=1)
        cosine = left_norm @ right_norm.transpose(0, 1)
        if self.name in {"cosine", "mlp"}:
            return (cosine + 1.0) / 2.0
        left_sq = features.float().square().sum(dim=1, keepdim=True)
        right_sq = left_sq if other is None else right.float().square().sum(dim=1).unsqueeze(0)
        distances = (left_sq + right_sq - 2.0 * features.float() @ right.float().T).clamp_min(0.0)
        return torch.exp(-self.gamma * distances)


class KernelFactory:
    @staticmethod
    def create(name: str, **kwargs: Any) -> PairwiseKernel:
        return PairwiseKernel(name, **kwargs)
