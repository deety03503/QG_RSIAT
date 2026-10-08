"""Prepare CUB, ImageNet-A, or ImageNet-R as train/<class>/ and test/<class>/.

ImageNet-A and ImageNet-R are evaluation datasets, not training datasets. When
their source has no predefined split, this utility makes a reproducible,
stratified split for experiments and smoke tests. CUB uses its published
train/test assignments when the official metadata is present.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = {"cub": 200, "imageneta": 200, "imagenetr": 200}


def _split_indices(labels: list[int], test_fraction: float, seed: int) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)
    train: set[int] = set()
    test: set[int] = set()
    for indices in by_label.values():
        if len(indices) < 2:
            raise ValueError("Every class needs at least two images to create train and test splits")
        rng.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, round(len(indices) * test_fraction)))
        test.update(indices[:n_test])
        train.update(indices[n_test:])
    return train, test


def _class_directories(root: Path) -> dict[str, list[Path]]:
    """Find a class-folder tree, including one wrapper directory from tar files."""
    best: dict[str, list[Path]] = {}
    for candidate in (root, *sorted(p for p in root.iterdir() if p.is_dir())):
        classes = {
            class_dir.name: sorted(
                p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            for class_dir in candidate.iterdir()
            if class_dir.is_dir()
        }
        classes = {name: paths for name, paths in classes.items() if paths}
        if len(classes) > len(best):
            best = classes
    return best


def _prepare_cub_official(source: Path, output: Path) -> bool:
    """Use CUB_200_2011's published train_test_split.txt if present."""
    roots = [source, *sorted(p for p in source.rglob("CUB_200_2011") if p.is_dir())]
    for root in roots:
        image_root = root / "images"
        split_file = root / "train_test_split.txt"
        image_file = root / "images.txt"
        if not (image_root.is_dir() and split_file.is_file() and image_file.is_file()):
            continue
        image_relpaths: dict[int, str] = {}
        for line in image_file.read_text(encoding="utf-8").splitlines():
            image_id, relpath = line.split(maxsplit=1)
            image_relpaths[int(image_id)] = relpath
        split_ids: dict[int, int] = {}
        for line in split_file.read_text(encoding="utf-8").splitlines():
            image_id, is_train = line.split()
            split_ids[int(image_id)] = int(is_train)
        if set(image_relpaths) != set(split_ids):
            raise ValueError("CUB images.txt and train_test_split.txt contain different image IDs")
        for image_id, relpath in image_relpaths.items():
            split = "train" if split_ids[image_id] else "test"
            source_image = image_root / relpath
            if not source_image.is_file():
                raise FileNotFoundError(f"CUB image listed in metadata is missing: {source_image}")
            destination = output / split / Path(relpath).parent / Path(relpath).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination)
        return True
    return False


def _prepare_parquet(source: Path, output: Path, test_fraction: float, seed: int) -> bool:
    parquet_files = sorted((source / "data").glob("*.parquet"))
    if not parquet_files:
        return False
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Parquet sources require `pip install datasets`.") from exc

    dataset = load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")
    if "image" not in dataset.features or "label" not in dataset.features:
        raise ValueError(f"Expected image and label columns; found {list(dataset.features)}")
    labels = [int(label) for label in dataset["label"]]
    names = getattr(dataset.features["label"], "names", None)
    if not names:
        names = _class_names_from_metadata(source)
    if not names:
        # A numeric class folder preserves the exact label ID and its ordering.
        # Prefer names from dataset_infos.json whenever the repository provides it.
        names = [f"class_{index:03d}" for index in range(max(labels, default=-1) + 1)]
    if labels and (min(labels) < 0 or max(labels) >= len(names)):
        raise ValueError(
            f"Label IDs range from {min(labels)} to {max(labels)}, but only {len(names)} class names were found"
        )
    train_ids, test_ids = _split_indices(labels, test_fraction, seed)
    for split, indices in (("train", train_ids), ("test", test_ids)):
        for index in sorted(indices):
            image = dataset[index]["image"]
            if not hasattr(image, "convert"):
                raise TypeError(f"Dataset row {index} does not contain a PIL image")
            destination = output / split / names[labels[index]] / f"{index:06d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(destination, quality=95)
    return True


def _class_names_from_metadata(source: Path) -> list[str] | None:
    """Read Hugging Face ClassLabel names retained in dataset_infos.json."""
    metadata_path = source / "dataset_infos.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def find_names(value: object) -> list[str] | None:
        if isinstance(value, dict):
            label = value.get("label")
            if isinstance(label, dict) and isinstance(label.get("names"), list):
                names = label["names"]
                if all(isinstance(name, str) for name in names):
                    return names
            for child in value.values():
                found = find_names(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_names(child)
                if found:
                    return found
        return None

    return find_names(metadata)


def _prepare_class_folders(source: Path, output: Path, test_fraction: float, seed: int) -> bool:
    classes = _class_directories(source)
    if not classes:
        return False
    names = sorted(classes)
    paths = [path for name in names for path in classes[name]]
    labels = [label for label, name in enumerate(names) for _ in classes[name]]
    train_ids, test_ids = _split_indices(labels, test_fraction, seed)
    for split, indices in (("train", train_ids), ("test", test_ids)):
        for index in sorted(indices):
            name = names[labels[index]]
            source_image = paths[index]
            destination = output / split / name / source_image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(EXPECTED_CLASSES))
    parser.add_argument("--source", required=True, help="Downloaded archive, extracted folders, or HF repository")
    parser.add_argument("--output", required=True, help="Prepared dataset directory")
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test_fraction must be between 0 and 1")

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset source directory not found: {source}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    prepared = False
    if args.dataset == "cub":
        prepared = _prepare_cub_official(source, output)
    if not prepared:
        prepared = _prepare_parquet(source, output, args.test_fraction, args.seed)
    if not prepared:
        prepared = _prepare_class_folders(source, output, args.test_fraction, args.seed)
    if not prepared:
        output.rmdir()
        raise FileNotFoundError(
            f"Could not find CUB metadata, Parquet files, or class folders under {source}"
        )

    train_classes = {p.name for p in (output / "train").iterdir() if p.is_dir()}
    test_classes = {p.name for p in (output / "test").iterdir() if p.is_dir()}
    expected = EXPECTED_CLASSES[args.dataset]
    if len(train_classes) != expected or train_classes != test_classes:
        raise ValueError(
            f"Prepared {args.dataset} has {len(train_classes)} train and {len(test_classes)} test classes; "
            f"expected matching sets of {expected}. Check source dataset and split metadata."
        )
    print(f"Prepared {args.dataset}: {len(train_classes)} classes at {output}")


if __name__ == "__main__":
    main()
