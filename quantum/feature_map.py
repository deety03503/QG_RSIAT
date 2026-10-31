from __future__ import annotations

import math
import torch
from torch import nn

from quantum.simulator import RealStatevectorCircuit


def center_features(
    features: torch.Tensor,
    centering: bool = True,
    centering_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    values = features.float()
    if not centering:
        return values

    if centering_mean is not None:
        center = centering_mean.float().detach()
        if center.dim() == 1:
            center = center.unsqueeze(0)
        return values - center

    if values.shape[0] > 1:
        center = values.mean(dim=0, keepdim=True).detach()
        return values - center

    return values


def build_angle_encoder(input_dim: int, n_qubits: int, layers: int) -> nn.Sequential:
    out_dim = int(n_qubits) * int(layers)
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 128, bias=False),
        nn.LayerNorm(128),
        nn.ReLU(inplace=True),
        nn.Linear(128, out_dim, bias=False),
        nn.LayerNorm(out_dim),
    )


class QuantumFeatureMap(nn.Module):
    """Quantum angle encoder + real statevector simulator."""

    def __init__(
        self,
        input_dim: int = 768,
        n_qubits: int = 8,
        layers: int = 2,
        centering: bool = True,
        cycle: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.layers = int(layers)
        self.centering = bool(centering)
        self.cycle = bool(cycle)

        self.encoder = build_angle_encoder(self.input_dim, self.n_qubits, self.layers)
        self.theta = nn.Parameter(torch.zeros(self.layers, self.n_qubits))
        self.circuit = RealStatevectorCircuit(
            n_qubits=self.n_qubits,
            layers=self.layers,
            cycle=self.cycle,
        )

    def encode_angles(
        self,
        features: torch.Tensor,
        centering_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = center_features(
            features,
            centering=self.centering,
            centering_mean=centering_mean,
        )
        raw = self.encoder(values).reshape(values.shape[0], self.layers, self.n_qubits)
        return torch.tanh(raw) * math.pi

    def forward(
        self,
        features: torch.Tensor,
        *,
        centering_mean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = self.encode_angles(features, centering_mean=centering_mean)
        states = self.circuit(angles, self.theta)
        obs = self.circuit.readout(states)
        return states, obs

