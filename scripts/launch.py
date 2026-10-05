"""Kaggle-aware entry point for direct runs and experiment queues."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrsiat.hardware.detect import detect_hardware
from qrsiat.runner.launch import select_launch
from qrsiat.runner.queue import run_experiment_queue


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue")
    parser.add_argument("--config", default="./exps/adapter_imageneta.json")
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--output_dir", "--out", dest="output_dir", default="./out")
    parser.add_argument("--time_budget_h", type=float)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    options, forwarded = parser.parse_known_args()

    if options.queue:
        _run_queue(
            options.queue,
            options=options,
            forwarded=forwarded,
        )
        return

    try:
        config = json.loads(Path(options.config).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read configuration {options.config}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Experiment config must contain a JSON object")
    if "experiments" in config:
        _run_queue(
            options.config,
            options=options,
            forwarded=forwarded,
        )
        return
    profile = detect_hardware(".")
    selection = select_launch(
        profile,
        requested_mode=options.mode,
        global_batch=int(config.get("batch_size", 1)),
        python_executable=sys.executable,
        main_script=str(Path(__file__).resolve().parents[1] / "main.py"),
    )
    resolved_mode = "ddp" if selection.plan.mode == "ddp" else selection.plan.mode
    command = [
        *selection.launch_command,
        "--config",
        str(Path(options.config).resolve()),
        "--mode",
        resolved_mode,
        "--output_dir",
        options.output_dir,
        *forwarded,
    ]
    if options.smoke:
        command.append("--smoke")
    if options.time_budget_h is not None:
        command.extend(("--time_budget_h", str(options.time_budget_h)))
    subprocess.run(command, check=True)


def _run_queue(queue_path: str, *, options: argparse.Namespace, forwarded: list[str]) -> None:
    profile = detect_hardware(".")
    if not profile.gpus:
        raise RuntimeError("Experiment queues require a GPU; use scripts/doctor.py on CPU")
    common_args = _filter_queue_args(forwarded)
    output_dir = Path(options.output_dir)
    if options.smoke:
        common_args.append("--smoke")
        output_dir = output_dir / "smoke-queue"
    run_experiment_queue(
        queue_path,
        output_dir=output_dir,
        gpu_count=len(profile.gpus),
        time_budget_h=options.time_budget_h,
        common_args=common_args,
    )


def _filter_queue_args(arguments: list[str]) -> list[str]:
    blocked = {
        "--config",
        "--mode",
        "--queue",
        "--out",
        "--output_dir",
        "--time_budget_h",
    }
    result: list[str] = []
    skip_value = False
    for item in arguments:
        if skip_value:
            skip_value = False
            continue
        if item in blocked:
            skip_value = True
            continue
        if any(item.startswith(f"{key}=") for key in blocked):
            continue
        result.append(item)
    return result


if __name__ == "__main__":
    main()
