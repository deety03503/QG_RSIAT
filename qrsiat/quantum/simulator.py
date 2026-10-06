"""Batched real-valued RY/CNOT statevector with Pauli-Z readout."""

from __future__ import annotations

from typing import Any


class RealStatevectorCircuit:
    def __init__(self, n_qubits: int, layers: int) -> None:
        if n_qubits < 1 or layers < 1:
            raise ValueError("n_qubits and layers must be positive")
        self.n_qubits = n_qubits
        self.layers = layers

    def __call__(self, angles: Any, trainable_angles: Any | None = None) -> Any:
        import torch

        if angles.ndim != 3 or angles.shape[1:] != (self.layers, self.n_qubits):
            raise ValueError(
                f"angles must have shape [B,{self.layers},{self.n_qubits}], "
                f"got {tuple(angles.shape)}"
            )
        if not angles.is_floating_point():
            raise TypeError("Circuit angles must be floating point")
        values = angles.float()
        if trainable_angles is not None:
            if tuple(trainable_angles.shape) != (self.layers, self.n_qubits):
                raise ValueError("trainable_angles has an incompatible shape")
            values = values + trainable_angles.float().unsqueeze(0)

        batch_size = values.shape[0]
        state = torch.zeros(batch_size, 1 << self.n_qubits, dtype=torch.float32, device=values.device)
        state[:, 0] = 1.0
        basis = torch.arange(1 << self.n_qubits, device=values.device)
        for layer in range(self.layers):
            for qubit in range(self.n_qubits):
                bit = 1 << qubit
                zero = basis[(basis & bit) == 0]
                one = basis[(basis & bit) != 0]
                cos = torch.cos(values[:, layer, qubit] / 2).unsqueeze(1)
                sin = torch.sin(values[:, layer, qubit] / 2).unsqueeze(1)
                state_zero = state.index_select(1, zero)
                state_one = state.index_select(1, one)
                state = (
                    torch.zeros_like(state)
                    .index_copy(1, zero, cos * state_zero - sin * state_one)
                    .index_copy(1, one, sin * state_zero + cos * state_one)
                )
            for control in range(self.n_qubits - 1):
                target = control + 1
                source = torch.where(
                    (basis & (1 << control)) != 0,
                    basis ^ (1 << target),
                    basis,
                )
                state = state.index_select(1, source)
        return state

    def readout(self, state: Any) -> Any:
        import torch

        if state.ndim != 2 or state.shape[1] != 1 << self.n_qubits:
            raise ValueError("state must have shape [B, 2**n_qubits]")
        basis = torch.arange(state.shape[1], device=state.device)
        outputs = []
        for qubit in range(self.n_qubits):
            bit = 1 << qubit
            z_sign = 1.0 - 2.0 * ((basis & bit) != 0).to(state.dtype)
            z_value = (state.square() * z_sign.unsqueeze(0)).sum(dim=1)
            outputs.append(z_value)
        return torch.stack(outputs, dim=1).clamp(-1.0, 1.0)


def fidelity_kernel(states: Any, other: Any | None = None) -> Any:
    import torch

    right = states if other is None else other
    if states.ndim != 2 or right.ndim != 2 or states.shape[1] != right.shape[1]:
        raise ValueError("state batches must be [B,S] and [C,S]")
    states = torch.nn.functional.normalize(states.float(), dim=1)
    right = torch.nn.functional.normalize(right.float(), dim=1)
    overlap = states @ right.transpose(0, 1)
    return overlap.square().clamp(0.0, 1.0)
