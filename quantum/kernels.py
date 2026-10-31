from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from quantum.feature_map import QuantumFeatureMap
from quantum.simulator import fidelity_kernel


class PairwiseKernel(nn.Module):
    """Unified pairwise kernel interface returning similarity matrices in [0, 1]."""

    def __init__(
        self,
        name: str = "quantum",
        input_dim: int = 768,
        n_qubits: int = 8,
        layers: int = 2,
        centering: bool = True,
        cycle: bool = False,
    ) -> None:
        super().__init__()
        self.name = str(name).lower()
        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.layers = int(layers)
        self.centering = bool(centering)
        self.cycle = bool(cycle)

        if self.name == "quantum":
            self.quantum = QuantumFeatureMap(
                input_dim=self.input_dim,
                n_qubits=self.n_qubits,
                layers=self.layers,
                centering=self.centering,
                cycle=self.cycle,
            )
        elif self.name == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.GELU(),
                nn.Linear(self.input_dim, self.input_dim),
            )
        elif self.name in {"cosine", "rbf"}:
            pass
        else:
            raise ValueError(
                f"Unknown kernel type '{name}'. Supported: 'quantum', 'cosine', 'rbf', 'mlp'."
            )

    def forward(
        self,
        features: torch.Tensor,
        other: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.name == "quantum":
            if other is None:
                states, _ = self.quantum(features)
                return fidelity_kernel(states, states)

            center = None
            if self.quantum.centering and features.shape[0] > 1:
                center = features.float().mean(dim=0, keepdim=True).detach()
            left_states, _ = self.quantum(features, centering_mean=center)
            right_states, _ = self.quantum(other, centering_mean=center)
            return fidelity_kernel(left_states, right_states)

        if self.name == "cosine":
            left = F.normalize(features.float(), p=2, dim=1)
            right = left if other is None else F.normalize(other.float(), p=2, dim=1)
            cosine = left @ right.T
            return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)

        if self.name == "rbf":
            left = features.float()
            right = left if other is None else other.float()
            left_sq = left.square().sum(dim=1, keepdim=True)
            right_sq = right.square().sum(dim=1, keepdim=True).T
            dist_sq = (left_sq + right_sq - 2.0 * (left @ right.T)).clamp_min(0.0)
            return torch.exp(-dist_sq / float(left.shape[1])).clamp(0.0, 1.0)

        if self.name == "mlp":
            left = F.normalize(self.mlp(features.float()), p=2, dim=1)
            right = left if other is None else F.normalize(self.mlp(other.float()), p=2, dim=1)
            cosine = left @ right.T
            return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)

        raise RuntimeError(f"Unsupported kernel: {self.name}")

