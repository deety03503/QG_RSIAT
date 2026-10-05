import torch


def _apply_ry(state, angles, qubit):
    shape = (*state.shape[:-1], *([2] * (angles.shape[-1])))
    amplitudes = state.reshape(shape).movedim(qubit + 1, -1)
    angle = angles[..., qubit]
    broadcast_shape = (angle.shape[0],) + (1,) * (amplitudes.ndim - 2)
    cosine = torch.cos(angle / 2).reshape(broadcast_shape)
    sine = torch.sin(angle / 2).reshape(broadcast_shape)
    zero, one = amplitudes.unbind(dim=-1)
    rotated = torch.stack(
        (cosine * zero - sine * one, sine * zero + cosine * one), dim=-1
    )
    return rotated.movedim(-1, qubit + 1).contiguous().reshape_as(state)


def _apply_cnot_chain(state, n_qubits):
    dimension = 1 << n_qubits
    indices = torch.arange(dimension, device=state.device)
    for control in range(n_qubits - 1):
        control_mask = 1 << (n_qubits - 1 - control)
        target_mask = 1 << (n_qubits - 2 - control)
        permutation = torch.where(
            (indices & control_mask) != 0, indices ^ target_mask, indices
        )
        state = state.index_select(-1, permutation)
    return state


def statevector(angles):
    """Simulate RY layers followed by a nearest-neighbor CNOT chain.

    Args:
        angles: Tensor shaped ``[batch, layers, qubits]``.

    Returns:
        Real normalized statevectors shaped ``[batch, 2**qubits]``.
    """
    if angles.ndim != 3:
        raise ValueError("angles must have shape [batch, layers, qubits]")
    if not angles.is_floating_point():
        raise TypeError("angles must use a floating-point dtype")
    _, layers, n_qubits = angles.shape
    if layers < 1 or n_qubits < 1:
        raise ValueError("layers and qubits must both be positive")
    angles = angles.to(dtype=torch.float32)
    state = angles.new_zeros((angles.shape[0], 1 << n_qubits))
    state[:, 0] = 1
    for layer in range(layers):
        for qubit in range(n_qubits):
            state = _apply_ry(state, angles[:, layer], qubit)
        state = _apply_cnot_chain(state, n_qubits)
    return state / state.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(state.dtype).tiny
    )


def fidelity(state_a, state_b):
    """Return pairwise squared overlaps for real statevectors."""
    if state_a.ndim != 2 or state_b.ndim != 2:
        raise ValueError("statevectors must be rank-2 tensors")
    if state_a.shape[1] != state_b.shape[1]:
        raise ValueError("statevectors must have matching state dimensions")
    state_a = state_a.to(dtype=torch.float32)
    state_b = state_b.to(dtype=torch.float32)
    overlaps = state_a @ state_b.transpose(0, 1)
    return overlaps.square().clamp(0.0, 1.0)


def z_expectations(state, n_qubits):
    """Measure Pauli-Z expectation for each qubit."""
    dimension = 1 << n_qubits
    if state.ndim != 2 or state.shape[-1] != dimension:
        raise ValueError("state dimension does not match n_qubits")
    indices = torch.arange(dimension, device=state.device)
    outputs = []
    probabilities = state.square()
    for qubit in range(n_qubits):
        bit_mask = 1 << (n_qubits - 1 - qubit)
        signs = torch.where(
            (indices & bit_mask) == 0,
            state.new_tensor(1.0),
            state.new_tensor(-1.0),
        )
        outputs.append((probabilities * signs).sum(dim=-1))
    return torch.stack(outputs, dim=-1)


def statevector_cost(n_qubits):
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    return 1 << n_qubits
