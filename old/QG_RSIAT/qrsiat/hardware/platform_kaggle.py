"""Kaggle-specific paths and environment discovery."""

from __future__ import annotations

import os
from pathlib import Path


def is_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content").is_dir()


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").is_dir()


def kaggle_input_dir() -> Path | None:
    candidate = Path("/kaggle/input")
    return candidate if candidate.is_dir() else None


def kaggle_working_dir() -> Path | None:
    candidate = Path("/kaggle/working")
    return candidate if candidate.is_dir() else None


def kaggle_disk_free_bytes() -> int | None:
    working = kaggle_working_dir()
    if working is None:
        return None
    try:
        import shutil

        return shutil.disk_usage(working).free
    except OSError:
        return None


def cgroup_memory_limit_bytes() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            value = path.read_text(encoding="ascii").strip()
            if value and value != "max":
                limit = int(value)
                if 0 < limit < (1 << 60):
                    return limit
        except (OSError, ValueError):
            continue
    return None


def linux_memory_total_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None
