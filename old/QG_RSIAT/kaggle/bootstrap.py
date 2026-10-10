"""Prepare attached datasets and optionally install non-PyTorch dependencies."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrsiat.data.registry import prepare_dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--data_root")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        requirements = Path(__file__).with_name("requirements-kaggle.txt")
        contents = requirements.read_text(encoding="utf-8").lower()
        if any(line.strip().split("=", 1)[0] in {"torch", "torchvision"} for line in contents.splitlines()):
            raise RuntimeError("Kaggle requirements must not install or replace torch/torchvision")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            check=True,
        )
    for dataset in args.dataset:
        resolved = prepare_dataset_root(dataset, data_root=args.data_root)
        print(f"Dataset {dataset}: {resolved.root}")


if __name__ == "__main__":
    main()
