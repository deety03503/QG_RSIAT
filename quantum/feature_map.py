import math

import torch
from torch import nn

from quantum.simulator import statevector


class QuantumFeatureMap(nn.Module):
    def __init__(self, input_dim=768, n_qubits=8, n_layers=2, center=True):
        super().__init__()
        if input_dim < 1 or n_qubits < 1 or n_layers < 1:
            raise ValueError("input_dim, n_qubits and n_layers must be positive")
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.center = center
        self.angle_map = nn.Linear(input_dim, n_qubits * n_layers)

    def angles(self, features, center_mean=None):
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch, {self.input_dim}]"
            )
        angles = self.angle_map(features)
        if self.center:
            mean = angles.mean(dim=0, keepdim=True) if center_mean is None else center_mean
            angles = angles - mean
        angles = math.pi * torch.tanh(angles)
        return angles.reshape(-1, self.n_layers, self.n_qubits)

    def forward(self, features, center_mean=None):
        return statevector(self.angles(features, center_mean=center_mean))

    def readout(self, features, center_mean=None):
        states = self.forward(features, center_mean=center_mean)
        return self.z_readout(states)

    def z_readout(self, states):
        from quantum.simulator import z_expectations

        return z_expectations(states, self.n_qubits)
