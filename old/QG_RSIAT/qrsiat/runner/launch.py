"""Decide between one-GPU execution and torchrun process launch."""

from __future__ import annotations

from dataclasses import dataclass

from qrsiat.hardware.detect import HardwareProfile
from qrsiat.hardware.plan import RuntimePlan, create_runtime_plan


@dataclass(frozen=True)
class LaunchSelection:
    plan: RuntimePlan
    launch_command: tuple[str, ...]


def select_launch(
    profile: HardwareProfile,
    *,
    requested_mode: str = "auto",
    global_batch: int = 1,
    queue_jobs: int = 1,
    python_executable: str = "python",
    main_script: str = "main.py",
) -> LaunchSelection:
    plan = create_runtime_plan(
        profile,
        requested_mode=requested_mode,
        global_batch=global_batch,
        queue_jobs=queue_jobs,
    )
    if plan.mode == "ddp":
        command = (
            python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={plan.world_size}",
            main_script,
        )
    else:
        command = (python_executable, main_script)
    return LaunchSelection(plan, command)
