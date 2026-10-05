"""Merge legacy experiment dictionaries with QR-RSIAT options."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schema import QRsiatConfig


def merge_config(
    legacy: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve all legacy keys and overlay validated QR-RSIAT defaults/options.

    Precedence is QR-RSIAT defaults < legacy experiment config < explicit
    overrides. Unknown keys from the legacy config are retained for the
    existing RSIAT learner; unknown keys in explicit overrides are rejected.
    """
    result = dict(legacy)
    config_keys = set(QRsiatConfig.__dataclass_fields__)
    explicit = dict(overrides or {})
    unknown = sorted(set(explicit) - config_keys)
    if unknown:
        raise ValueError(f"Unknown QR-RSIAT override keys: {', '.join(unknown)}")

    legacy_values = {key: legacy[key] for key in config_keys if key in legacy}
    merged_values = QRsiatConfig.from_mapping(legacy_values).to_dict()
    merged_values.update(explicit)
    validated = QRsiatConfig.from_mapping(merged_values).to_dict()
    result.update(validated)
    return result
