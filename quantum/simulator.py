from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RealStatevectorCircuit(nn.Module):
    """Exact real-amplitude statevector quantum circuit simulator (RY + CNOT)."""

    def __init__(self, n_qubits: int = 8, layers: int = 2, cycle: bool = False) -> None:
        super().__init__()
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")

        self.n_qubits = int(n_qubits)
        self.layers = int(layers)
        self.cycle = bool(cycle)
        self.dim = 1 << self.n_qubits

        basis = torch.arange(self.dim, dtype=torch.long)

        zero_indices = []
        one_indices = []
        x_pair_indices = []
        z_signs = []

        for j in range(self.n_qubits):
            bit = 1 << j
            mask_zero = (basis & bit) == 0
            z_idx = basis[mask_zero]
            o_idx = z_idx | bit
            zero_indices.append(z_idx)
            one_indices.append(o_idx)
            x_pair_indices.append(basis ^ bit)
            z_signs.append(1.0 - 2.0 * ((basis & bit) != 0).float())

        self.register_buffer("zero_idx", torch.stack(zero_indices, dim=0), persistent=False)
        self.register_buffer("one_idx", torch.stack(one_indices, dim=0), persistent=False)
        self.register_buffer("x_pair_idx", torch.stack(x_pair_indices, dim=0), persistent=False)
        self.register_buffer("z_sign", torch.stack(z_signs, dim=0), persistent=False)

        cnot_perm = basis.clone()
        if self.n_qubits > 1:
            pairs = [(j, j + 1) for j in range(self.n_qubits - 1)]
            if self.cycle:
                pairs.append((self.n_qubits - 1, 0))
            for control, target in pairs:
                c_bit = 1 << control
                t_bit = 1 << target
                step_perm = torch.where((basis & c_bit) != 0, basis ^ t_bit, basis)
                cnot_perm = cnot_perm.index_select(0, step_perm)

        self.register_buffer("cnot_perm", cnot_perm, persistent=False)

    def forward(
        self,
        angles: torch.Tensor,
        trainable_angles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        angles = angles.float()
        if trainable_angles is not None:
            angles = angles + trainable_angles.float().unsqueeze(0)

        batch_size = angles.shape[0]
        state = torch.zeros(batch_size, self.dim, dtype=torch.float32, device=angles.device)
        state[:, 0] = 1.0

        for l in range(self.layers):
            for j in range(self.n_qubits):
                half_angle = angles[:, l, j] * 0.5
                c = torch.cos(half_angle).unsqueeze(1)
                s = torch.sin(half_angle).unsqueeze(1)
                s0 = state.index_select(1, self.zero_idx[j])
                s1 = state.index_select(1, self.one_idx[j])
                state = (
                    torch.zeros_like(state)
                    .index_copy(1, self.zero_idx[j], c * s0 - s * s1)
                    .index_copy(1, self.one_idx[j], s * s0 + c * s1)
                )
            state = state.index_select(1, self.cnot_perm)

        return state

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        state = state.float()
        paired = state[:, self.x_pair_idx]
        x_exp = (state.unsqueeze(1) * paired).sum(dim=-1)
        probs = state.square()
        z_exp = probs @ self.z_sign.t()
        return torch.cat([x_exp, z_exp], dim=1)


def fidelity_kernel(
    states: torch.Tensor,
    other: torch.Tensor | None = None,
    obs: torch.Tensor | None = None,
    other_obs: torch.Tensor | None = None,
) -> torch.Tensor:
    left = F.normalize(states.float(), p=2, dim=1)
    right = left if other is None else F.normalize(other.float(), p=2, dim=1)
    if obs is None:
        return (left @ right.T).square().clamp(0.0, 1.0)
    amp_sim = (left @ right.T).abs().clamp(0.0, 1.0)
    left_obs = F.normalize(obs.float(), p=2, dim=1)
    right_obs = left_obs if other_obs is None else F.normalize(other_obs.float(), p=2, dim=1)
    proj_sim = ((left_obs @ right_obs.T + 1.0) * 0.5).clamp(0.0, 1.0)
    return (0.5 * amp_sim + 0.5 * proj_sim).clamp(0.0, 1.0)

