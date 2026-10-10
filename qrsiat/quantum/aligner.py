"""Identity-initialized residual classical-quantum feature aligner."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class QHybridAligner(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        n_qubits: int = 8,
        layers: int = 2,
        *,
        centering: bool = True,
    ) -> None:
        super().__init__()
        from .simulator import RealStatevectorCircuit

        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.down = nn.Linear(input_dim, n_qubits * layers)
        self.circuit = RealStatevectorCircuit(n_qubits, layers)
        self.theta = nn.Parameter(torch.zeros(layers, n_qubits))
        self.up = nn.Linear(2 * n_qubits, input_dim, bias=False)
        nn.init.zeros_(self.up.weight)
        self.centering = centering
        self.n_qubits = n_qubits
        self.layers = layers
    def __call__(
        self, features: Any
    ) -> Any:
        return super().__call__(features)

    def forward(self, features: Any) -> Any:
        import torch

        if features.ndim != 2 or features.shape[1] != self.down.in_features:
            raise ValueError(f"features must have shape [B,{self.down.in_features}]")
        values = features.float()
        if self.centering:
            values = torch.nn.functional.normalize(values, p=2, dim=1)
        angles = self.down(values).reshape(-1, self.layers, self.n_qubits)
        angles = angles.tanh() * torch.pi
        state = self.circuit(angles, self.theta)
        readout = self.circuit.readout(state)
        return features + self.up(readout)
