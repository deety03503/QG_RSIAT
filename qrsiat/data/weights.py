"""Offline pretrained-checkpoint discovery with no random-weight fallback."""

from __future__ import annotations

import os
from pathlib import Path

from qrsiat.hardware.platform_kaggle import kaggle_input_dir

WEIGHT_SUFFIXES = {".pth", ".pt", ".bin", ".safetensors"}


def find_pretrained_checkpoint(
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    search_roots: tuple[str | os.PathLike[str], ...] | None = None,
    model_tokens: tuple[str, ...] = ("vit_base_patch16_224_in21k", "vit-b-16", "in21k"),
) -> Path | None:
    """Find a likely ViT-B/16-IN21K checkpoint; explicit paths fail fast."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint does not exist: {path}")
        if path.suffix.lower() not in WEIGHT_SUFFIXES:
            raise ValueError(f"Unsupported checkpoint extension: {path.suffix!r}")
        return path.resolve()

    matches: list[Path] = []
    if search_roots is None:
        kaggle_root = kaggle_input_dir()
        search_roots = (kaggle_root,) if kaggle_root is not None else ()
    normalized_tokens = tuple(token.lower().replace("_", "-") for token in model_tokens)
    for root_value in search_roots:
        root = Path(root_value).expanduser()
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            candidate = path.name.lower().replace("_", "-")
            standard_vit_token = "vit-base-patch16-224" in normalized_tokens
            is_wrong_pretraining = (
                standard_vit_token
                and "in21k" not in normalized_tokens
                and "in21k" in candidate
            )
            if not is_wrong_pretraining and any(token in candidate for token in normalized_tokens):
                matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda item: (len(item.parts), item.as_posix().lower()))
    return matches[0].resolve()
