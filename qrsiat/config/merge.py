"""Merge legacy experiment dictionaries with QR-RSIAT options."""

from __future__ import annotations

from collections.abc import Mapping
import math
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
    policy = explicit.get("kernel_policy", legacy_values.get("kernel_policy", "legacy"))
    if policy in {"shared_frozen", "fixed_calibrated"}:
        if "kernel" in explicit or "rs_kernel" in explicit:
            selected_kernel = explicit.get("kernel", explicit.get("rs_kernel"))
            legacy_values["kernel"] = selected_kernel
            legacy_values["rs_kernel"] = selected_kernel
        elif "kernel" in legacy_values and "rs_kernel" not in legacy_values:
            legacy_values["rs_kernel"] = legacy_values["kernel"]
        elif "rs_kernel" in legacy_values and "kernel" not in legacy_values:
            legacy_values["kernel"] = legacy_values["rs_kernel"]
    elif "kernel" in legacy_values and "rs_kernel" not in legacy_values:
        legacy_values["rs_kernel"] = legacy_values["kernel"]
    elif "rs_kernel" in legacy_values and "kernel" not in legacy_values:
        legacy_values["kernel"] = legacy_values["rs_kernel"]
    merged_values = QRsiatConfig.from_mapping(legacy_values).to_dict()
    merged_values.update(explicit)
    if policy in {"shared_frozen", "fixed_calibrated"}:
        if "kernel" in explicit and "rs_kernel" not in explicit:
            merged_values["rs_kernel"] = explicit["kernel"]
        elif "rs_kernel" in explicit and "kernel" not in explicit:
            merged_values["kernel"] = explicit["rs_kernel"]
        elif "kernel" in legacy_values and "rs_kernel" not in legacy_values:
            merged_values["rs_kernel"] = merged_values["kernel"]
        elif "rs_kernel" in legacy_values and "kernel" not in legacy_values:
            merged_values["kernel"] = merged_values["rs_kernel"]
    elif "kernel" in explicit and "rs_kernel" not in explicit and "rs_kernel" not in legacy_values:
        merged_values["rs_kernel"] = explicit["kernel"]
    elif "rs_kernel" in explicit and "kernel" not in explicit and "kernel" not in legacy_values:
        merged_values["kernel"] = explicit["rs_kernel"]
    if merged_values["rs_kernel"] is None:
        merged_values["rs_kernel"] = merged_values["kernel"]
    validated = QRsiatConfig.from_mapping(merged_values).to_dict()
    result.update(validated)
    beta = result.get("beta", 0.0)
    gamma = result.get("gamma", 0.0)
    for name, value in (("beta", beta), ("gamma", gamma)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Legacy {name} must be numeric for QR-RSIAT integration")
    if validated["aligner"] == "qhybrid" and not any(
        value > 0 for value in (beta, gamma)
    ):
        raise ValueError(
            "The qhybrid aligner has no active objective: set positive beta or gamma."
        )
    if validated["lambda_qrel"] > 0 and beta <= 0:
        raise ValueError("lambda_qrel > 0 requires a positive beta coefficient.")
    if validated["orth"] == "qweighted" and gamma <= 0:
        raise ValueError("--orth qweighted requires a positive gamma coefficient.")
    if validated["kernel_policy"] != "fixed_calibrated" and validated["kernel"] in {"quantum", "mlp"}:
        lambda_rs = result.get("lambda_rs", 0.0)
        if (
            isinstance(lambda_rs, bool)
            or not isinstance(lambda_rs, (int, float))
            or not math.isfinite(lambda_rs)
            or lambda_rs <= 0
        ):
            raise ValueError(
                "A trainable shared kernel requires a positive lambda_rs "
                "to train it during the base task."
            )
    return result
