import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumStateSimulator(nn.Module):
    """
    Differentiable Quantum State Simulator for QG-RSIAT.
    Maps classical feature vectors z in R^d to pure quantum states |psi(z)> in C^(2^q)
    via a shallow Data Re-uploading Parameterized Quantum Circuit (PQC), computes
    pairwise quantum overlap amplitudes s(z_i, z_j) = |<psi(z_i)|psi(z_j)>|, and
    extracts 4q local Pauli expectations <O>_{psi(z)}.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_qubits: int = 8,
        num_layers: int = 2,
        eps_stab: float = 1e-8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.hilbert_dim = 1 << num_qubits
        self.obs_dim = 4 * num_qubits
        self.eps_stab = eps_stab

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, num_layers * num_qubits),
            nn.LayerNorm(num_layers * num_qubits),
        )

        self.var_theta = nn.Parameter(torch.zeros(num_layers, num_qubits))
        self.var_phi = nn.Parameter(torch.zeros(num_layers, num_qubits))
        self.data_scale_z = nn.Parameter(torch.ones(num_layers, num_qubits) * 0.5)

        nn.init.uniform_(self.var_theta, -0.1, 0.1)
        nn.init.uniform_(self.var_phi, -0.1, 0.1)

        self._init_quantum_buffers()

    def _init_quantum_buffers(self):
        q = self.num_qubits
        dim = self.hilbert_dim
        basis = torch.arange(dim, dtype=torch.long)

        bits = torch.stack(
            [(basis >> (q - 1 - k)) & 1 for k in range(q)], dim=1
        )  # (2^q, q)

        z_signs = 1.0 - 2.0 * bits.float()  # +1 for |0>, -1 for |1>, shape (2^q, q)
        zz_signs = z_signs * torch.roll(z_signs, shifts=-1, dims=1)  # (2^q, q)

        flip_indices = torch.stack(
            [basis ^ (1 << (q - 1 - k)) for k in range(q)], dim=0
        )  # (q, 2^q)

        cnot_bits = bits.clone()
        for k in range(q):
            target = (k + 1) % q
            cnot_bits[:, target] = cnot_bits[:, target] ^ cnot_bits[:, k]

        weights = (2 ** torch.arange(q - 1, -1, -1, dtype=torch.long)).unsqueeze(0)
        out_basis = (cnot_bits * weights).sum(dim=1)
        cnot_gather_perm = torch.argsort(out_basis)

        self.register_buffer("z_signs", z_signs)
        self.register_buffer("zz_signs", zz_signs)
        self.register_buffer("flip_indices", flip_indices)
        self.register_buffer("cnot_gather_perm", cnot_gather_perm)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode classical feature tensor z (B, d) into quantum state |psi(z)> (B, 2^q).
        """
        batch_size = z.size(0)
        q = self.num_qubits
        z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)

        angles = math.pi * torch.tanh(self.feature_proj(z_norm))
        angles = angles.view(batch_size, self.num_layers, q)

        psi = torch.zeros(
            (batch_size, self.hilbert_dim), dtype=torch.complex64, device=z.device
        )
        psi[:, 0] = 1.0 + 0.0j

        for layer_idx in range(self.num_layers):
            theta = angles[:, layer_idx, :] + self.var_theta[layer_idx].unsqueeze(0)
            phi = (
                angles[:, layer_idx, :] * self.data_scale_z[layer_idx].unsqueeze(0)
                + self.var_phi[layer_idx].unsqueeze(0)
            )

            for k in range(q):
                left_dim = 1 << k
                right_dim = 1 << (q - 1 - k)
                psi_view = psi.view(batch_size, left_dim, 2, right_dim)
                psi_0 = psi_view[:, :, 0, :]
                psi_1 = psi_view[:, :, 1, :]

                half_theta = 0.5 * theta[:, k].view(batch_size, 1, 1)
                half_phi = 0.5 * phi[:, k].view(batch_size, 1, 1)

                cos_t = torch.cos(half_theta)
                sin_t = torch.sin(half_theta)

                phase_neg = torch.complex(torch.cos(-half_phi), torch.sin(-half_phi))
                phase_pos = torch.conj(phase_neg)

                new_0 = phase_neg * (cos_t * psi_0 - sin_t * psi_1)
                new_1 = phase_pos * (sin_t * psi_0 + cos_t * psi_1)

                psi = torch.stack([new_0, new_1], dim=2).reshape(
                    batch_size, self.hilbert_dim
                )

            psi = psi[:, self.cnot_gather_perm]

        return psi

    def overlap_from_states(
        self, psi1: torch.Tensor, psi2: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute quantum state overlap amplitude s(z_i, z_j) = |<psi_i | psi_j>| in [0, 1].
        """
        if psi2 is None:
            psi2 = psi1
        inner_prod = torch.matmul(torch.conj(psi1), psi2.transpose(0, 1))
        overlap = torch.sqrt(
            inner_prod.real.square() + inner_prod.imag.square() + self.eps_stab
        )
        return torch.clamp(overlap, min=0.0, max=1.0)

    def overlap(self, z1: torch.Tensor, z2: torch.Tensor = None) -> torch.Tensor:
        """
        Encode features and compute pairwise quantum state overlap amplitude s(z_i, z_j).
        """
        psi1 = self.encode(z1)
        psi2 = psi1 if z2 is None else self.encode(z2)
        return self.overlap_from_states(psi1, psi2)

    def observables_from_state(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute local single-qubit and two-qubit Pauli expectations <O>_{psi(z)} in [-1, 1]^(B, 4q).
        """
        probs = psi.real.square() + psi.imag.square()  # (B, 2^q)
        exp_z = torch.matmul(probs, self.z_signs)  # (B, q)
        exp_zz = torch.matmul(probs, self.zz_signs)  # (B, q)

        psi_flipped = psi[:, self.flip_indices]  # (B, q, 2^q)
        prod = torch.conj(psi).unsqueeze(1) * psi_flipped  # (B, q, 2^q)

        exp_x = prod.real.sum(dim=-1)  # (B, q)
        exp_y = (prod.imag * self.z_signs.t().unsqueeze(0)).sum(dim=-1)  # (B, q)

        return torch.cat([exp_z, exp_x, exp_y, exp_zz], dim=-1)

    def observables(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode classical feature z and return local quantum observables <O>_{psi(z)}.
        """
        psi = self.encode(z)
        return self.observables_from_state(psi)

    def freeze(self):
        """
        Freeze all quantum kernel parameters after base task (t = 0).
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

