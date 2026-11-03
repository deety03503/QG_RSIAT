from __future__ import annotations

import math
import torch
from torch import nn

from quantum.feature_map import build_angle_encoder, center_features
from quantum.simulator import RealStatevectorCircuit


class QHybridAligner(nn.Module):
    """Enhanced identity-initialized residual quantum-hybrid aligner.

    Upgrades:
    - Entanglement-aware readout (single-qubit Pauli-X/Z + 2-qubit nearest-neighbor ZZ/XX correlators).
    - Non-linear bottleneck up-projection (hidden_dim=128) breaking the rank-16 linear subspace bottleneck.
    - Learnable residual scale starting near identity (initial_scale=0.1) with task-to-task reset support.
    """

    def __init__(
        self,
        input_dim: int = 768,
        n_qubits: int = 8,
        layers: int = 2,
        centering: bool = True,
        cycle: bool = False,
        hidden_dim: int = 128,
        entangled_readout: bool = True,
        initial_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.layers = int(layers)
        self.centering = bool(centering)
        self.cycle = bool(cycle)
        self.hidden_dim = int(hidden_dim)
        self.entangled_readout = bool(entangled_readout)

        self.encoder = build_angle_encoder(self.input_dim, self.n_qubits, self.layers)
        self.theta = nn.Parameter(torch.zeros(self.layers, self.n_qubits))
        self.circuit = RealStatevectorCircuit(
            n_qubits=self.n_qubits,
            layers=self.layers,
            cycle=self.cycle,
        )

        obs_dim = 4 * self.n_qubits if self.entangled_readout else 2 * self.n_qubits

        # Non-linear bottleneck up-projection expands rank from <= 16 to a rich non-linear manifold in R^768
        self.up = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_dim, bias=False),
        )
        # Orthogonal initialization keeps W_up^T != 0 so the circuit receives gradients from step 0
        nn.init.orthogonal_(self.up[2].weight, gain=1.0)

        # Fixed residual scale keeps initial correction near identity;
        # the non-linear MLP `up` (102k params) adapts the magnitude smoothly without scalar gradient explosion.
        self.res_scale = float(initial_scale)
        self.last_delta_norm = 0.0

    def reset_scale(self, scale: float = 0.1) -> None:
        """Reset residual scale to near-identity when entering a new incremental task."""
        self.res_scale = float(scale)

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
        obs = self.circuit.readout(states, include_correlators=self.entangled_readout)
        delta = self.up(obs) * self.res_scale
        self.last_delta_norm = float(delta.detach().norm(dim=1).mean().item())
        return features_fp32 + delta
