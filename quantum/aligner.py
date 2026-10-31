from __future__ import annotations

import math
import torch
from torch import nn

from quantum.feature_map import build_angle_encoder, center_features
from quantum.simulator import RealStatevectorCircuit


class QHybridAligner(nn.Module):
    """Identity-initialized residual quantum-hybrid aligner."""

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
        self.up = nn.Linear(2 * self.n_qubits, self.input_dim, bias=False)
        # Orthogonal init keeps W_up^T != 0 so the circuit receives gradient from step 0;
        # the small residual gate keeps the initial correction near identity (||delta|| <= 0.1 * ||obs||).
        nn.init.orthogonal_(self.up.weight, gain=1.0)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.last_delta_norm = 0.0

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
    ) -> torch.Tensor:
        features_fp32 = features.float()
        angles = self.encode_angles(features_fp32, centering_mean=centering_mean)
        states = self.circuit(angles, self.theta)
        obs = self.circuit.readout(states)
        delta = self.up(obs) * self.res_scale
        self.last_delta_norm = float(delta.detach().norm(dim=1).mean().item())
        return features_fp32 + delta

