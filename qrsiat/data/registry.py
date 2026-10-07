"""Defensive resolution of RSIAT datasets on local disks and Kaggle."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from qrsiat.hardware.platform_kaggle import kaggle_input_dir


DATASET_DIRS = {
    "imageneta": "imagenet_a",
    "imagenet-a": "imagenet_a",
    "imagenetr": "imagenet_r",
    "imagenet-r": "imagenet_r",
    "cub": "cub",
    "vtab": "vtab",
    "omnibench": "omnibenchmark",
    "omnibenchmark": "omnibenchmark",
    "cifar224": "cifar224",
    "cifar100": "cifar224",
}
EXPECTED_CLASS_COUNTS = {
    "imageneta": 200,
    "imagenet-a": 200,
    "imagenetr": 200,
    "imagenet-r": 200,
    "cub": 200,
    "vtab": 50,
    "omnibench": 300,
    "omnibenchmark": 300,
}


@dataclass(frozen=True)
class DatasetLocation:
    name: str
    root: Path
    train_dir: Path | None
    test_dir: Path | None
    is_cifar: bool = False


def _dataset_key(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_DIRS:
        raise ValueError(
            f"Unsupported dataset {name!r}. Supported datasets: "
            + ", ".join(sorted(DATASET_DIRS))
        )
    return key


def _validate_imagefolder(name: str, root: Path) -> DatasetLocation:
    train_dir, test_dir = root / "train", root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset {name!r} must contain both 'train/' and 'test/' directories: {root}"
        )
    train_classes = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    test_classes = sorted(path.name for path in test_dir.iterdir() if path.is_dir())
    if not train_classes or not test_classes:
        raise FileNotFoundError(
            f"Dataset {name!r} has no class directories under {train_dir} or {test_dir}"
        )
    if train_classes != test_classes:
        missing_test = sorted(set(train_classes) - set(test_classes))
        missing_train = sorted(set(test_classes) - set(train_classes))
        raise ValueError(
            f"Dataset {name!r} train/test class mappings differ. "
            f"Missing in test={missing_test[:8]}, missing in train={missing_train[:8]}"
        )
    expected = EXPECTED_CLASS_COUNTS.get(_dataset_key(name))
    if expected is not None and len(train_classes) != expected:
        raise ValueError(
            f"Dataset {name!r} has {len(train_classes)} classes; RSIAT expects {expected}. "
            "Check that the attached dataset uses the original class split."
        )
    return DatasetLocation(name, root, train_dir, test_dir)


def discover_dataset(
    name: str,
    *,
    data_root: str | os.PathLike[str] | None = None,
    repository_root: str | os.PathLike[str] = ".",
    kaggle_input: str | os.PathLike[str] | None = None,
) -> DatasetLocation:
    """Find and validate a dataset without changing its class ordering."""
    key = _dataset_key(name)
    canonical = DATASET_DIRS[key]
    if key in {"cifar224", "cifar100"}:
        root = Path(data_root or Path(repository_root) / "data" / "datasets").expanduser()
        return DatasetLocation(name, root, None, None, is_cifar=True)

    candidates: list[Path] = []
    if data_root:
        base = Path(data_root).expanduser()
        candidates.extend((base / canonical, base / name, base))
    repo = Path(repository_root).expanduser()
    candidates.extend(
        (
            repo / "data" / "datasets" / canonical,
            repo / "data" / "datasets" / name,
            repo / "datasets" / canonical,
            repo / "datasets" / name,
        )
    )
    kaggle = Path(kaggle_input) if kaggle_input is not None else kaggle_input_dir()
    if kaggle is not None and kaggle.is_dir():
        safe_name = re.compile(re.escape(canonical), re.IGNORECASE)
        for dataset_mount in sorted(kaggle.iterdir()):
            if not dataset_mount.is_dir():
                continue
            if safe_name.fullmatch(dataset_mount.name.replace("_", "-")):
                candidates.extend((dataset_mount / canonical, dataset_mount / name, dataset_mount))
            candidates.extend(
                child for child in dataset_mount.iterdir()
                if child.is_dir() and safe_name.fullmatch(child.name.replace("_", "-"))
            )

    seen: set[Path] = set()
    errors: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not (resolved / "train").is_dir() or not (resolved / "test").is_dir():
            continue
        try:
            return _validate_imagefolder(name, resolved)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("Found dataset candidate(s), but validation failed: " + " | ".join(errors))
    raise FileNotFoundError(
        f"Dataset {name!r} was not found. Attach a Kaggle Dataset containing "
        f"{canonical}/train and {canonical}/test, or set DATA_ROOT/--data_root."
    )


def prepare_dataset_root(
    name: str,
    *,
    data_root: str | os.PathLike[str] | None = None,
    repository_root: str | os.PathLike[str] = ".",
) -> DatasetLocation:
    """Resolve data and create only the legacy-compatible symlink when needed."""
    location = discover_dataset(name, data_root=data_root, repository_root=repository_root)
    if location.is_cifar:
        location.root.mkdir(parents=True, exist_ok=True)
        return location

    expected = Path(repository_root) / "data" / "datasets" / DATASET_DIRS[_dataset_key(name)]
    if location.root.resolve() == expected.resolve():
        return location
    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists() or expected.is_symlink():
        if expected.resolve() == location.root.resolve():
            return location
        raise FileExistsError(
            f"Cannot link dataset {location.root} over existing path {expected}. "
            "Remove or relocate the existing path explicitly."
        )
    try:
        expected.symlink_to(location.root, target_is_directory=True)
    except OSError as exc:
        raise OSError(
            f"Could not create RSIAT dataset symlink {expected} -> {location.root}; "
            "place the dataset under data/datasets or enable symlink support."
        ) from exc
    return location
