import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumStateSimulator(nn.Module):
    """
    Multi-Subspace Differentiable Quantum State Simulator for MS-QG-RSIAT.
    Decomposes classical feature vectors z in R^d into M orthogonal quantum
    subspaces (each a q-qubit Hilbert space C^(2^q)) via Stiefel-initialized
    projection and parallel Data Re-uploading Parameterized Quantum Circuits (PQCs).
    Computes multi-subspace quantum overlap amplitudes s_Q(z_i, z_j) in [0, 1]
    and extracts M * 4q local Pauli expectations <O>_{psi(z)} in [-1, 1].
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_qubits: int = 6,
        num_layers: int = 2,
        num_subspaces: int = 8,
        eps_stab: float = 1e-8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_subspaces = num_subspaces
        self.hilbert_dim = 1 << num_qubits
        self.obs_dim = num_subspaces * 4 * num_qubits
        self.eps_stab = eps_stab

        total_angles = num_subspaces * num_layers * num_qubits
        proj_linear = nn.Linear(input_dim, total_angles)
        nn.init.orthogonal_(proj_linear.weight, gain=1.0)
        nn.init.zeros_(proj_linear.bias)

        self.feature_proj = nn.Sequential(
            proj_linear,
            nn.LayerNorm(total_angles),
        )

        self.var_theta = nn.Parameter(
            torch.zeros(1, num_subspaces, num_layers, num_qubits)
        )
        self.var_phi = nn.Parameter(
            torch.zeros(1, num_subspaces, num_layers, num_qubits)
        )
        self.data_scale_z = nn.Parameter(
            torch.ones(1, num_subspaces, num_layers, num_qubits) * 0.5
        )

        nn.init.uniform_(self.var_theta, -0.05, 0.05)
        nn.init.uniform_(self.var_phi, -0.05, 0.05)

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
        Encode classical feature tensor z (B, d) into multi-subspace quantum states
        |psi(z)> of shape (B, M, 2^q).
        """
        batch_size = z.size(0)
        m = self.num_subspaces
        q = self.num_qubits
        bm = batch_size * m

        z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)
        angles = math.pi * torch.tanh(self.feature_proj(z_norm))
        angles = angles.view(batch_size, m, self.num_layers, q)

        theta_all = angles + self.var_theta
        phi_all = angles * self.data_scale_z + self.var_phi

        psi = torch.zeros(
            (bm, self.hilbert_dim), dtype=torch.complex64, device=z.device
        )
        psi[:, 0] = 1.0 + 0.0j

        for layer_idx in range(self.num_layers):
            theta = theta_all[:, :, layer_idx, :].reshape(bm, q)
            phi = phi_all[:, :, layer_idx, :].reshape(bm, q)

            for k in range(q):
                left_dim = 1 << k
                right_dim = 1 << (q - 1 - k)
                psi_view = psi.view(bm, left_dim, 2, right_dim)
                psi_0 = psi_view[:, :, 0, :]
                psi_1 = psi_view[:, :, 1, :]

                half_theta = 0.5 * theta[:, k].view(bm, 1, 1)
                half_phi = 0.5 * phi[:, k].view(bm, 1, 1)

                cos_t = torch.cos(half_theta)
                sin_t = torch.sin(half_theta)

                phase_neg = torch.complex(torch.cos(-half_phi), torch.sin(-half_phi))
                phase_pos = torch.conj(phase_neg)

                new_0 = phase_neg * (cos_t * psi_0 - sin_t * psi_1)
                new_1 = phase_pos * (sin_t * psi_0 + cos_t * psi_1)

                psi = torch.stack([new_0, new_1], dim=2).reshape(
                    bm, self.hilbert_dim
                )

            psi = psi[:, self.cnot_gather_perm]

        return psi.view(batch_size, m, self.hilbert_dim)

    def overlap_from_states(
        self, psi1: torch.Tensor, psi2: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute multi-subspace quantum state overlap amplitude s_Q(z_i, z_j) in [0, 1]
        from states psi1 (B1, M, 2^q) and psi2 (B2, M, 2^q).
        """
        if psi2 is None:
            psi2 = psi1
        if psi1.dim() == 2:
            psi1 = psi1.unsqueeze(1)
        if psi2.dim() == 2:
            psi2 = psi2.unsqueeze(1)

        p1 = torch.conj(psi1.permute(1, 0, 2))  # (M, B1, 2^q)
        p2 = psi2.permute(1, 2, 0)  # (M, 2^q, B2)
        inner_prod = torch.matmul(p1, p2)  # (M, B1, B2)

        overlap_m = torch.sqrt(
            inner_prod.real.square() + inner_prod.imag.square() + self.eps_stab
        )
        overlap = overlap_m.mean(dim=0)  # (B1, B2)
        return torch.clamp(overlap, min=0.0, max=1.0)

    def overlap(self, z1: torch.Tensor, z2: torch.Tensor = None) -> torch.Tensor:
        """
        Encode features and compute pairwise multi-subspace overlap amplitude s_Q(z_i, z_j).
        """
        psi1 = self.encode(z1)
        psi2 = psi1 if z2 is None else self.encode(z2)
        return self.overlap_from_states(psi1, psi2)

    def observables_from_state(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute local single-qubit and two-qubit Pauli expectations <O>_{psi(z)}
        in [-1, 1]^(B, M * 4q).
        """
        if psi.dim() == 2:
            psi = psi.unsqueeze(1)
        batch_size, m, dim = psi.shape
        bm = batch_size * m
        psi_flat = psi.reshape(bm, dim)

        probs = psi_flat.real.square() + psi_flat.imag.square()  # (B * M, 2^q)
        exp_z = torch.matmul(probs, self.z_signs)  # (B * M, q)
        exp_zz = torch.matmul(probs, self.zz_signs)  # (B * M, q)

        psi_flipped = psi_flat[:, self.flip_indices]  # (B * M, q, 2^q)
        prod = torch.conj(psi_flat).unsqueeze(1) * psi_flipped  # (B * M, q, 2^q)

        exp_x = prod.real.sum(dim=-1)  # (B * M, q)
        exp_y = (prod.imag * self.z_signs.t().unsqueeze(0)).sum(dim=-1)  # (B * M, q)

        obs_flat = torch.cat([exp_z, exp_x, exp_y, exp_zz], dim=-1)  # (B * M, 4q)
        return obs_flat.reshape(batch_size, m * 4 * self.num_qubits)

    def observables(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode classical feature z and return local quantum observables <O>_{psi(z)}.
        """
        psi = self.encode(z)
        return self.observables_from_state(psi)

    def encode_and_observables(self, z: torch.Tensor):
        """
        Encode classical feature z once and return both quantum state |psi(z)>
        and local Pauli expectations <O>_{psi(z)}.
        """
        psi = self.encode(z)
        obs = self.observables_from_state(psi)
        return psi, obs

    def freeze(self):
        """
        Freeze all quantum kernel parameters after base task (t = 0).
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
