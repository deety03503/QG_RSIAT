"""Projection from continuous embeddings to shallow circuit angles."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .simulator import RealStatevectorCircuit
from .centering import EMACenter


class QuantumFeatureMap(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        n_qubits: int = 8,
        layers: int = 2,
        *,
        centering: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.project = nn.Linear(input_dim, n_qubits * layers)
        self.simulator = RealStatevectorCircuit(n_qubits, layers)
        self.centering = centering
        self.center_ema = EMACenter(input_dim)
        self.n_qubits = n_qubits
        self.layers = layers
        self.theta = nn.Parameter(torch.zeros(layers, n_qubits))

    def __call__(
        self, features: Any, *, centering_mean: Any | None = None
    ) -> tuple[Any, Any]:
        return super().__call__(features, centering_mean=centering_mean)

    def forward(
        self, features: Any, *, centering_mean: Any | None = None
    ) -> tuple[Any, Any]:
        if features.ndim != 2 or features.shape[1] != self.project.in_features:
            raise ValueError(
                f"features must have shape [B,{self.project.in_features}]"
            )
        values = features.float()
        if self.centering:
            if centering_mean is None:
                if self.training:
                    centering_mean = self.center_ema.update(values)
                else:
                    centering_mean = self.center_ema.current(values)
            else:
                if centering_mean.ndim == 1:
                    centering_mean = centering_mean.unsqueeze(0)
                if tuple(centering_mean.shape) != (1, self.project.in_features):
                    raise ValueError(
                        f"centering_mean must have shape [{self.project.in_features}] "
                        f"or [1,{self.project.in_features}]"
                    )
                centering_mean = centering_mean.to(values)
            if centering_mean is not None:
                values = values - centering_mean
        angles = self.project(values).reshape(-1, self.layers, self.n_qubits)
        angles = angles.tanh() * 3.141592653589793
        state = self.simulator(angles, self.theta)
        return state, self.simulator.readout(state)
