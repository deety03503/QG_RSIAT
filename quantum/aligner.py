import torch
import torch.nn as nn
from quantum.simulator import QuantumStateSimulator


class QuantumHybridAligner(nn.Module):
    """
    Symmetric Quantum-Gated Residual Feature Aligner (Module 2 of MS-QG-RSIAT):
        h_fused = h_c(z) * (1 + alpha_q * g_q(<O>_{psi(z)})) + beta_q * g_q(<O>_{psi(z)})
        A(z) = z + tanh(Decoder(h_fused))
    Replaces AutoencoderSigmoid with a symmetric Tanh-bounded residual projector
    modulated by multi-subspace local quantum observables <O>_{psi(z)}, resolving
    both the positivity bias of Sigmoid and vanishing gradients under small ae_init_lr.
    """

    def __init__(
        self,
        input_dims: int = 768,
        code_dims: int = 128,
        quantum_sim: QuantumStateSimulator = None,
        alpha_init: float = 0.5,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.code_dims = code_dims
        self._quantum_sim_ref = (quantum_sim,)
        obs_dim = quantum_sim.obs_dim if quantum_sim is not None else 192

        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.GELU(),
            nn.Linear(64, code_dims),
            nn.GELU(),
        )

        self.quantum_gate = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.GELU(),
            nn.Linear(64, code_dims),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(code_dims, 64),
            nn.GELU(),
            nn.Linear(64, input_dims),
            nn.Tanh(),
        )

        self.alpha_q = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta_q = nn.Parameter(torch.tensor(0.1))

    @property
    def quantum_sim(self) -> QuantumStateSimulator:
        return self._quantum_sim_ref[0]

    def reset_parameters_for_task(self):
        """
        Keep warm-started parameters across incremental tasks to preserve
        accumulated drift manifold geometry under small ae_init_lr.
        """
        return

    def forward(self, z: torch.Tensor, obs: torch.Tensor = None) -> torch.Tensor:
        h_c = self.classical_encoder(z)

        if obs is None and self.quantum_sim is not None:
            with torch.no_grad():
                obs = self.quantum_sim.observables(z)

        if obs is not None:
            g_q = self.quantum_gate(obs)
            h_fused = h_c * (1.0 + self.alpha_q * g_q) + self.beta_q * g_q
        else:
            h_fused = h_c

        residual = self.decoder(h_fused)
        return z + residual
