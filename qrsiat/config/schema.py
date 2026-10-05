"""Configuration schema for QR-RSIAT runtime and optional method features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class QRsiatConfig:
    seed: int | list[int] = 1993
    mode: str = "auto"
    aligner: str = "rae"
    lambda_qrel: float = 0.0
    pretrained_weights: str | None = None
    kernel: str = "cosine"
    orth: str = "plain"
    rs_kernel: str = "cosine"
    orth_top_k: int = 3
    orth_temperature: float = 0.1
    orth_epsilon: float = 0.2
    n_qubits: int = 8
    quantum_layers: int = 2
    quantum_centering: bool = True
    pair_gather: bool | None = None
    lr_scale_by_world: bool = False
    num_workers: int | None = None
    max_workers: int = 8
    prefetch_factor: int = 2
    use_compile: bool = False
    freeze_cast: bool = False
    grad_checkpointing: bool = False
    probe: bool | None = None
    smoke: bool = False
    output_dir: str = "./out"
    data_root: str | None = None
    resume: str = "auto"
    resume_from: str | None = None
    force_resume: bool = False
    time_budget_h: float | None = None

    def validate(self) -> "QRsiatConfig":
        string_fields = (
            "mode",
            "aligner",
            "kernel",
            "orth",
            "rs_kernel",
            "output_dir",
            "resume",
        )
        if any(not isinstance(getattr(self, name), str) for name in string_fields):
            raise ValueError("Mode, feature selectors, output_dir, and resume must be strings")
        integer_fields = (
            "n_qubits",
            "quantum_layers",
            "max_workers",
            "prefetch_factor",
            "orth_top_k",
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
            for name in integer_fields
        ):
            raise ValueError("Qubit, layer, worker, and prefetch settings must be integers")
        if self.num_workers is not None and (
            isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int)
        ):
            raise ValueError("num_workers must be an integer or None")
        for name in ("lambda_qrel", "orth_temperature", "orth_epsilon"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.time_budget_h is not None and (
            not isinstance(self.time_budget_h, (int, float))
            or isinstance(self.time_budget_h, bool)
        ):
            raise ValueError("time_budget_h must be numeric or None")
        if self.time_budget_h is not None and not math.isfinite(self.time_budget_h):
            raise ValueError("time_budget_h must be finite")
        bool_fields = (
            "quantum_centering",
            "lr_scale_by_world",
            "use_compile",
            "freeze_cast",
            "grad_checkpointing",
            "force_resume",
            "smoke",
        )
        if any(not isinstance(getattr(self, name), bool) for name in bool_fields):
            raise ValueError("Optimization and behavior flags must be booleans")
        for name in ("pair_gather", "probe"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or None")
        for name in ("data_root", "resume_from", "pretrained_weights"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")

        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, list)):
            raise ValueError("seed must be a non-negative integer or a list of seeds")
        seeds = self.seed if isinstance(self.seed, list) else [self.seed]
        if not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        ):
            raise ValueError("seed values must be non-negative integers")
        if self.mode not in {"auto", "cpu", "single", "ddp", "jobpar"}:
            raise ValueError(f"Unsupported mode: {self.mode!r}")
        if self.aligner not in {"rae", "qhybrid"}:
            raise ValueError(f"Unsupported aligner: {self.aligner!r}")
        if self.kernel not in {"quantum", "cosine", "rbf", "mlp"}:
            raise ValueError(f"Unsupported kernel: {self.kernel!r}")
        if self.orth not in {"plain", "qweighted"}:
            raise ValueError(f"Unsupported orth mode: {self.orth!r}")
        if self.rs_kernel not in {"cosine", "quantum"}:
            raise ValueError(f"Unsupported RS kernel: {self.rs_kernel!r}")
        if self.lambda_qrel < 0:
            raise ValueError("lambda_qrel must be non-negative")
        if self.orth_top_k < 1:
            raise ValueError("orth_top_k must be positive")
        if self.orth_temperature <= 0:
            raise ValueError("orth_temperature must be positive")
        if self.orth_epsilon < 0:
            raise ValueError("orth_epsilon must be non-negative")
        if self.lambda_qrel and self.aligner != "qhybrid":
            raise ValueError("lambda_qrel > 0 requires --aligner qhybrid")
        if self.orth == "qweighted" and self.aligner != "qhybrid":
            raise ValueError("--orth qweighted requires --aligner qhybrid")
        if self.n_qubits not in {4, 8, 12}:
            raise ValueError("n_qubits must be one of 4, 8, or 12")
        if self.quantum_layers not in {1, 2, 3}:
            raise ValueError("quantum_layers must be one of 1, 2, or 3")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers must be non-negative or None")
        if self.max_workers < 0:
            raise ValueError("max_workers must be non-negative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be at least 1")
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")
        if self.resume not in {"auto", "none"}:
            raise ValueError("resume must be 'auto' or 'none'")
        if self.time_budget_h is not None and self.time_budget_h <= 0:
            raise ValueError("time_budget_h must be positive or None")
        if self.data_root is not None and not self.data_root.strip():
            raise ValueError("data_root must not be empty or whitespace")
        if self.resume_from is not None and not self.resume_from.strip():
            raise ValueError("resume_from must not be empty or whitespace")
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "QRsiatConfig":
        allowed = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise ValueError(f"Unknown QRsiatConfig keys: {', '.join(unknown)}")
        return cls(**dict(values)).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
