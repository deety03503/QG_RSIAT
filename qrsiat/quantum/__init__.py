"""Differentiable real-amplitude quantum-inspired training blocks."""

from .aligner import QHybridAligner
from .feature_map import QuantumFeatureMap
from .kernels import KernelFactory, PairwiseKernel
from .losses import qorth_loss, qrel_loss
from .simulator import RealStatevectorCircuit, fidelity_kernel

__all__ = [
    "KernelFactory",
    "PairwiseKernel",
    "QHybridAligner",
    "QuantumFeatureMap",
    "RealStatevectorCircuit",
    "fidelity_kernel",
    "qorth_loss",
    "qrel_loss",
]
