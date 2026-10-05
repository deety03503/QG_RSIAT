import torch
from torch import nn


class QHybridAligner(nn.Module):
    def __init__(self, feature_map, feature_dim=768):
        super().__init__()
        self.feature_map = feature_map
        self.readout_up = nn.Linear(feature_map.n_qubits, feature_dim)
        nn.init.zeros_(self.readout_up.weight)
        nn.init.zeros_(self.readout_up.bias)

    def forward(self, features):
        readout = self.feature_map.readout(features)
        return features + self.readout_up(readout)
