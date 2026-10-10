"""Print hardware, runtime, dataset, and pretrained-weight diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrsiat.data.registry import discover_dataset
from qrsiat.data.weights import find_pretrained_checkpoint
from qrsiat.hardware.detect import detect_hardware
from qrsiat.hardware.plan import create_runtime_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--pretrained_weights")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--mode", choices=("auto", "cpu", "single", "ddp", "jobpar"), default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    profile = detect_hardware(".")
    plan = create_runtime_plan(
        profile,
        requested_mode=args.mode,
        global_batch=args.batch_size,
    )
    datasets: list[dict[str, Any]] = []
    for name in args.dataset:
        location = discover_dataset(name, data_root=args.data_root)
        datasets.append(
            {
                "name": location.name,
                "root": str(location.root),
                "train": str(location.train_dir) if location.train_dir else None,
                "test": str(location.test_dir) if location.test_dir else None,
                "is_cifar": location.is_cifar,
            }
        )
    weights = find_pretrained_checkpoint(explicit_path=args.pretrained_weights)
    report = {
        "hardware": profile.to_dict(),
        "runtime_plan": plan.to_dict(),
        "datasets": datasets,
        "pretrained_checkpoint": str(weights) if weights else None,
        "notes": [
            "Doctor does not initialize a model, download weights, probe batch sizes, or train.",
            "When no offline checkpoint is found, timm must have a valid cached checkpoint or network access.",
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
