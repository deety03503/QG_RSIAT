import torch
import torch.nn as nn
from quantum.simulator import QuantumStateSimulator


class QuantumHybridAligner(nn.Module):
    """
    Quantum-Hybrid Residual Feature Aligner (Module 2A of QG-RSIAT):
        A(z) = z + alpha * g(<O>_{psi(z)}, z)
    Replaces the classical AutoencoderSigmoid with an unconstrained residual MLP
    conditioned on local quantum observable expectations <O>_{psi(z)}, resolving
    the positivity bias of Sigmoid while preserving the identity mapping prior at
    the start of each incremental task.
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
        obs_dim = quantum_sim.obs_dim if quantum_sim is not None else 32

        self.quantum_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.GELU(),
            nn.Linear(64, code_dims),
            nn.GELU(),
        )

        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.GELU(),
            nn.Linear(64, code_dims),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(code_dims),
            nn.Linear(code_dims, 64),
            nn.GELU(),
            nn.Linear(64, input_dims),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.reset_parameters_for_task()

    @property
    def quantum_sim(self) -> QuantumStateSimulator:
        return self._quantum_sim_ref[0]

    def reset_parameters_for_task(self):
        """
        Initialize the final projection layer close to zero so that A(z) ≈ z
        (Identity mapping prior) at the start of incremental adaptation.
        """
        last_linear = self.decoder[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        q_sim = self.quantum_sim
        if q_sim is not None:
            obs = q_sim.observables(z)
            h_q = self.quantum_encoder(obs)
        else:
            h_q = 0.0

        h_c = self.classical_encoder(z)
        residual = self.decoder(h_c + h_q)
        return z + self.alpha * residual
