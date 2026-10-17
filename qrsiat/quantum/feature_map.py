"""Projection from continuous embeddings to shallow circuit angles."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .simulator import RealStatevectorCircuit


class QuantumFeatureMap(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        n_qubits: int = 8,
        layers: int = 2,
        *,
        centering: bool = True,
        cycle: bool = False,
        input_norm: str = "none",
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if input_norm not in {"none", "layernorm", "l2"}:
            raise ValueError(f"Unsupported input_norm: {input_norm!r}")
        self.input_norm = input_norm
        self.project = nn.Linear(input_dim, n_qubits * layers)
        self.simulator = RealStatevectorCircuit(n_qubits, layers, cycle=cycle)
        self.centering = centering
        self.n_qubits = n_qubits
        self.layers = layers
        self.theta = nn.Parameter(torch.zeros(layers, n_qubits))
        self.register_buffer("bandwidth", torch.tensor(1.0, dtype=torch.float32))

    def normalize_input(self, features: torch.Tensor) -> torch.Tensor:
        values = features.float()
        if self.input_norm == "layernorm":
            return nn.functional.layer_norm(values, (self.project.in_features,))
        if self.input_norm == "l2":
            return nn.functional.normalize(values, p=2, dim=-1) * math.sqrt(self.project.in_features)
        return values

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
        values = self.normalize_input(features)
        if self.centering:
            if centering_mean is None:
                if values.shape[0] > 1:
                    centering_mean = values.mean(dim=0, keepdim=True)
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
        u = self.project(values).reshape(-1, self.layers, self.n_qubits)
        angles = (self.bandwidth * u).tanh() * 3.141592653589793
        state = self.simulator(angles, self.theta)
        return state, self.simulator.readout(state)


@torch.no_grad()
def calibrate_bandwidth(
    feature_map: Any,
    feats: Any,
    labels: Any | None = None,
    target: float = 0.1,
    grid: Any | None = None,
) -> float:
    """Select bandwidth gamma so median fidelity of pairs (different-class if labeled) is closest to target."""
    from .simulator import fidelity_kernel

    qfm = getattr(feature_map, "quantum", feature_map)
    if not hasattr(qfm, "bandwidth"):
        raise ValueError("feature_map must be QuantumFeatureMap or PairwiseKernel with quantum")

    n = feats.shape[0]
    if n < 2:
        return float(qfm.bandwidth.item())

    if grid is None:
        grid = torch.logspace(-4, 3, 71, device=feats.device)
    elif isinstance(grid, (list, tuple)):
        grid = torch.tensor(grid, device=feats.device, dtype=torch.float32)
    else:
        grid = grid.to(feats.device)

    mask = ~torch.eye(n, dtype=torch.bool, device=feats.device)
    if labels is not None:
        labels = labels.to(feats.device).reshape(-1)
        diff_mask = mask & (labels[:, None] != labels[None, :])
        if bool(diff_mask.any()):
            mask = diff_mask

    best, best_gap = None, float("inf")
    for gamma in grid.tolist():
        qfm.bandwidth.fill_(gamma)
        state, _ = qfm(feats)
        k = fidelity_kernel(state)
        med = k[mask].median().item()
        gap = abs(med - target)
        if gap < best_gap:
            best, best_gap = gamma, gap

    if best is not None:
        qfm.bandwidth.fill_(best)
    return float(qfm.bandwidth.item())
