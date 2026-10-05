"""Serializable runtime reports for hardware and execution planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..hardware.detect import HardwareProfile
from ..hardware.plan import RuntimePlan
from .io import write_json_atomic


def build_runtime_report(
    profile: HardwareProfile,
    plan: RuntimePlan | None = None,
) -> dict[str, Any]:
    return {
        "hardware": profile.to_dict(),
        "runtime_plan": plan.to_dict() if plan is not None else None,
    }


def write_runtime_report(
    path: str | Path,
    profile: HardwareProfile,
    plan: RuntimePlan | None = None,
) -> None:
    write_json_atomic(path, build_runtime_report(profile, plan))
