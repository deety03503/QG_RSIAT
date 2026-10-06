"""Conservative runtime-mode, precision, and dataloader planning."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .detect import HardwareProfile


@dataclass(frozen=True)
class RuntimePlan:
    mode: str
    device_index: int | None
    world_size: int
    worker_slots: int
    amp_dtype: str
    use_grad_scaler: bool
    per_device_batch: int
    global_batch: int
    grad_accum_steps: int
    eval_batch_size: int
    num_workers: int
    prefetch_factor: int | None
    pin_memory: bool
    persistent_workers: bool
    tf32: bool
    cudnn_benchmark: bool
    use_compile: bool
    grad_checkpointing: bool
    quantum_dtype: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _precision(gpu_cc: int | None, bf16_ok: bool) -> tuple[str, bool, bool]:
    if gpu_cc is not None and gpu_cc >= 8 and bf16_ok:
        return "bf16", False, True
    if gpu_cc is not None and gpu_cc >= 7:
        return "fp16", True, False
    if gpu_cc is not None and gpu_cc >= 6:
        return "fp16", True, False
    return "fp32", False, False


def create_runtime_plan(
    profile: HardwareProfile,
    *,
    requested_mode: str = "auto",
    global_batch: int = 1,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    max_workers: int = 8,
    prefetch_factor: int = 2,
    queue_jobs: int = 1,
    use_compile: bool = False,
    grad_checkpointing: bool = False,
) -> RuntimePlan:
    """Plan execution without probing/model execution or silently forcing CUDA."""
    if requested_mode not in {"auto", "cpu", "single", "ddp", "jobpar"}:
        raise ValueError(f"Unsupported runtime mode: {requested_mode!r}")
    for name, value in (
        ("global_batch", global_batch),
        ("queue_jobs", queue_jobs),
        ("max_workers", max_workers),
        ("prefetch_factor", prefetch_factor),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if eval_batch_size is not None and (
        isinstance(eval_batch_size, bool) or not isinstance(eval_batch_size, int)
    ):
        raise ValueError("eval_batch_size must be an integer or None")
    if num_workers is not None and (
        isinstance(num_workers, bool) or not isinstance(num_workers, int)
    ):
        raise ValueError("num_workers must be an integer or None")
    if global_batch < 1 or (eval_batch_size is not None and eval_batch_size < 1):
        raise ValueError("Batch sizes must be positive integers")
    if max_workers < 0 or prefetch_factor < 1 or queue_jobs < 1:
        raise ValueError("Invalid worker, prefetch, or queue configuration")
    if num_workers is not None and num_workers < 0:
        raise ValueError("num_workers must be non-negative or None")

    warnings = list(profile.warnings)
    gpu_count = len(profile.gpus)
    requested = requested_mode

    if requested == "auto":
        if gpu_count == 0:
            mode = "cpu"
        elif gpu_count == 1:
            mode = "single"
        elif not profile.homogeneous_gpus:
            mode = "jobpar"
        elif not profile.nccl_available:
            mode = "single"
            warnings.append("NCCL is unavailable; selecting one GPU instead of DDP.")
        elif queue_jobs >= gpu_count:
            mode = "jobpar"
        else:
            mode = "ddp"
    else:
        mode = requested

    if mode == "cpu" and requested != "auto" and gpu_count:
        device_index = None
    elif mode == "cpu":
        device_index = None
    elif mode in {"single", "jobpar", "ddp"} and gpu_count == 0:
        raise RuntimeError(f"Runtime mode {mode!r} requires at least one CUDA device")
    else:
        device_index = max(
            profile.gpus,
            key=lambda gpu: gpu.free_mem if gpu.free_mem is not None else -1,
        ).index

    if mode == "ddp":
        if gpu_count < 2:
            raise RuntimeError("DDP requires at least two visible CUDA devices")
        if not profile.nccl_available:
            raise RuntimeError("DDP was requested but NCCL is unavailable")
        if not profile.homogeneous_gpus:
            raise RuntimeError("DDP requires homogeneous GPUs; select jobpar or single")
        world_size = gpu_count
    elif mode == "jobpar":
        world_size = 1
    else:
        world_size = 1

    is_cuda = mode != "cpu"
    selected_gpu = next(
        (gpu for gpu in profile.gpus if gpu.index == device_index),
        None,
    )
    if selected_gpu is None:
        amp_dtype, use_scaler, tf32 = "fp32", False, False
    else:
        amp_dtype, use_scaler, tf32 = _precision(
            selected_gpu.cc_major, selected_gpu.bf16_ok
        )
    if mode == "ddp" and profile.gpus:
        precision = [
            _precision(gpu.cc_major, gpu.bf16_ok)
            for gpu in profile.gpus
        ]
        if any(item[0] != precision[0][0] for item in precision):
            amp_dtype, use_scaler, tf32 = "fp32", False, False
            warnings.append("GPU precision capabilities differ; DDP precision is set to fp32.")
        else:
            amp_dtype, use_scaler = precision[0][0], precision[0][1]
            tf32 = all(item[2] for item in precision)

    worker_slots = gpu_count if mode == "jobpar" else 1
    per_device_batch = max(1, global_batch // world_size)
    grad_accum_steps = max(
        1,
        math.ceil(global_batch / (per_device_batch * world_size)),
    )
    if world_size > 1 and global_batch % world_size:
        warnings.append(
            "Global batch is not divisible by world size; per-rank batching/accumulation "
            "may change pairwise-loss semantics."
        )
    if grad_accum_steps > 1:
        warnings.append(
            "Gradient accumulation changes pairwise-loss semantics because pairs are "
            "formed per micro-batch."
        )

    available_workers = max(0, profile.cpu_count // world_size)
    workers = min(
        available_workers,
        max_workers,
        num_workers if num_workers is not None else max_workers,
    )
    workers = max(0, workers)
    return RuntimePlan(
        mode=mode,
        device_index=device_index,
        world_size=world_size,
        worker_slots=worker_slots,
        amp_dtype=amp_dtype,
        use_grad_scaler=use_scaler and amp_dtype == "fp16",
        per_device_batch=per_device_batch,
        global_batch=global_batch,
        grad_accum_steps=grad_accum_steps,
        eval_batch_size=eval_batch_size or global_batch,
        num_workers=workers,
        prefetch_factor=prefetch_factor if workers else None,
        pin_memory=is_cuda,
        persistent_workers=workers > 0,
        tf32=tf32,
        cudnn_benchmark=is_cuda,
        use_compile=use_compile,
        grad_checkpointing=grad_checkpointing,
        quantum_dtype="fp32",
        warnings=tuple(warnings),
    )
