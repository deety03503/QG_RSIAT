"""Aggregate seed-level QR-RSIAT result files into a CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./out")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    root = Path(args.out).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {root}")

    grouped: dict[str, list[float]] = {}
    for metrics_path in root.rglob("metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            value = float(payload["average_top1"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid metrics file {metrics_path}: {exc}") from exc
        experiment = payload.get("experiment_id", metrics_path.parent.parent.name)
        grouped.setdefault(str(experiment), []).append(value)

    destination = Path(args.csv) if args.csv else root / "summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("experiment", "seeds", "mean_top1", "std_top1"))
        for experiment, values in sorted(grouped.items()):
            writer.writerow(
                (
                    experiment,
                    len(values),
                    f"{statistics.mean(values):.4f}",
                    f"{statistics.pstdev(values):.4f}" if len(values) > 1 else "0.0000",
                )
            )
    print(f"Wrote {len(grouped)} experiment summary row(s) to {destination}")


if __name__ == "__main__":
    main()
