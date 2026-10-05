"""Defensive discovery of host, software, and CUDA device capabilities."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPUProfile:
    index: int
    name: str
    total_mem: int | None
    free_mem: int | None
    cc_major: int | None
    cc_minor: int | None
    bf16_ok: bool
    tf32_ok: bool
    p2p_ok: bool


@dataclass(frozen=True)
class HardwareProfile:
    platform: str
    work_dir: str
    disk_free_gb: float | None
    cpu_count: int
    ram_gb: float | None
    torch_version: str | None
    cuda_version: str | None
    timm_version: str | None
    nccl_available: bool
    gpus: tuple[GPUProfile, ...]
    homogeneous_gpus: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except (AttributeError, OSError):
        pass
    return max(1, os.cpu_count() or 1)


def _read_memory_limit() -> int | None:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        try:
            value = path.read_text(encoding="ascii").strip()
            if value and value != "max":
                limit = int(value)
                if 0 < limit < (1 << 60):
                    return limit
        except (OSError, ValueError):
            continue
    return None


def _ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().total)
    except (ImportError, OSError, AttributeError):
        return None


def _platform_name() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").is_dir():
        return "colab"
    return "local"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _discover_torch(warnings: list[str]) -> tuple[Any | None, str | None, str | None, bool]:
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        warnings.append(f"PyTorch is not importable: {exc}")
        return None, None, None, False

    torch_version = str(torch.__version__)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        nccl_available = bool(torch.distributed.is_nccl_available())
    except (AttributeError, RuntimeError) as exc:
        warnings.append(f"Could not query NCCL availability: {exc}")
        nccl_available = False
    return torch, torch_version, str(cuda_version) if cuda_version else None, nccl_available


def _gpu_profiles(torch: Any, warnings: list[str]) -> tuple[GPUProfile, ...]:
    try:
        if not torch.cuda.is_available():
            return ()
        count = int(torch.cuda.device_count())
    except Exception as exc:
        warnings.append(f"CUDA device discovery failed; using CPU profile: {exc}")
        return ()

    profiles: list[GPUProfile] = []
    for index in range(count):
        name = f"cuda:{index}"
        total_mem = free_mem = None
        cc_major = cc_minor = None
        bf16_ok = tf32_ok = p2p_ok = False
        try:
            properties = torch.cuda.get_device_properties(index)
            name = str(properties.name)
            total_mem = int(properties.total_memory)
            cc_major = int(properties.major)
            cc_minor = int(properties.minor)
        except Exception as exc:
            warnings.append(f"Could not read properties for CUDA device {index}: {exc}")

        try:
            free_mem, observed_total = torch.cuda.mem_get_info(index)
            total_mem = int(observed_total)
        except Exception as exc:
            warnings.append(f"Could not read memory for CUDA device {index}: {exc}")

        try:
            with torch.cuda.device(index):
                bf16_ok = bool(torch.cuda.is_bf16_supported())
        except Exception as exc:
            warnings.append(f"Could not query bfloat16 support for CUDA device {index}: {exc}")

        tf32_ok = cc_major is not None and cc_major >= 8
        if count <= 1:
            p2p_ok = True
        else:
            try:
                p2p_ok = all(
                    bool(torch.cuda.can_device_access_peer(index, other))
                    for other in range(count)
                    if other != index
                )
            except Exception as exc:
                warnings.append(f"Could not query peer access for CUDA device {index}: {exc}")

        profiles.append(
            GPUProfile(
                index=index,
                name=name,
                total_mem=total_mem,
                free_mem=free_mem,
                cc_major=cc_major,
                cc_minor=cc_minor,
                bf16_ok=bf16_ok,
                tf32_ok=tf32_ok,
                p2p_ok=p2p_ok,
            )
        )
    return tuple(profiles)


def detect_hardware(work_dir: str | os.PathLike[str] = ".") -> HardwareProfile:
    """Collect a best-effort hardware snapshot; CUDA query failures do not abort."""
    warnings: list[str] = []
    resolved_work_dir = str(Path(work_dir).resolve())
    disk_free_gb: float | None
    try:
        disk_free_gb = shutil.disk_usage(resolved_work_dir).free / (1024**3)
    except OSError as exc:
        warnings.append(f"Could not inspect free disk space: {exc}")
        disk_free_gb = None

    ram = _ram_bytes()
    memory_limit = _read_memory_limit()
    if memory_limit is not None:
        ram = memory_limit if ram is None else min(ram, memory_limit)

    torch, torch_version, cuda_version, nccl_available = _discover_torch(warnings)
    gpus = _gpu_profiles(torch, warnings) if torch is not None else ()
    homogeneous = not gpus or all(
        (gpu.name, gpu.total_mem) == (gpus[0].name, gpus[0].total_mem)
        for gpu in gpus[1:]
    )

    return HardwareProfile(
        platform=_platform_name(),
        work_dir=resolved_work_dir,
        disk_free_gb=disk_free_gb,
        cpu_count=_cpu_count(),
        ram_gb=ram / (1024**3) if ram is not None else None,
        torch_version=torch_version,
        cuda_version=cuda_version,
        timm_version=_package_version("timm"),
        nccl_available=nccl_available,
        gpus=gpus,
        homogeneous_gpus=homogeneous,
        warnings=tuple(warnings),
    )
