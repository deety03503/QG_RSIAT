"""Atomic, configuration-checked task checkpoint management."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from qrsiat.utils.io import write_json_atomic


def configuration_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _trainable_state_dict(module: Any) -> dict[str, Any]:
    state = module.state_dict()
    trainable = {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    return {name: value for name, value in state.items() if name in trainable}


class CheckpointManager:
    def __init__(self, output_dir: str | os.PathLike[str], *, keep: int = 2) -> None:
        if keep < 1:
            raise ValueError("keep must be positive")
        self.output_dir = Path(output_dir)
        self.keep = keep
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        task: int,
        modules: dict[str, Any],
        config: dict[str, Any],
        *,
        optimizer: Any | None = None,
        metadata: dict[str, Any] | None = None,
        rank: int = 0,
    ) -> Path | None:
        import torch

        if rank != 0:
            return None
        if task < 0 or not modules:
            raise ValueError("task must be non-negative and modules must not be empty")
        digest = configuration_digest(config)
        payload = {
            "schema_version": 1,
            "task": task,
            "config_digest": digest,
            "config": config,
            "modules": {
                name: _trainable_state_dict(module)
                for name, module in modules.items()
                if module is not None
            },
            "rng_state": _rng_state(),
            "metadata": metadata or {},
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        destination = self.output_dir / f"task_{task:04d}.pt"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.output_dir
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary_name)
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        write_json_atomic(
            self.output_dir / "latest.json",
            {"checkpoint": destination.name, "task": task, "config_digest": digest},
        )
        checkpoints = sorted(
            self.output_dir.glob("task_*.pt"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for obsolete in checkpoints[self.keep :]:
            obsolete.unlink()
        return destination

    def find_latest(self, resume_from: str | os.PathLike[str] | None = None) -> Path | None:
        roots = [Path(resume_from).expanduser()] if resume_from else []
        roots.append(self.output_dir)
        for root in roots:
            if root.is_file() and root.suffix == ".pt":
                return root.resolve()
            latest = root / "latest.json"
            if latest.is_file():
                try:
                    descriptor = json.loads(latest.read_text(encoding="utf-8"))
                    checkpoint_name = descriptor["checkpoint"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid checkpoint pointer {latest}: {exc}") from exc
                if (
                    not isinstance(checkpoint_name, str)
                    or Path(checkpoint_name).name != checkpoint_name
                    or not checkpoint_name.startswith("task_")
                    or Path(checkpoint_name).suffix != ".pt"
                ):
                    raise ValueError(
                        f"Checkpoint pointer must name a task_*.pt file: {latest}"
                    )
                candidate = root / checkpoint_name
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint pointer references missing file: {candidate}"
                    )
                resolved_candidate = candidate.resolve()
                if resolved_candidate.parent != root.resolve():
                    raise ValueError(
                        f"Checkpoint pointer escapes its output directory: {latest}"
                    )
                return resolved_candidate
            checkpoints = sorted(root.glob("task_*.pt"), reverse=True) if root.is_dir() else []
            if checkpoints:
                return checkpoints[0].resolve()
        return None

    def inspect(
        self,
        config: dict[str, Any],
        *,
        resume_from: str | os.PathLike[str] | None = None,
        force: bool = False,
        map_location: Any = "cpu",
    ) -> dict[str, Any] | None:
        import torch

        checkpoint = self.find_latest(resume_from)
        if checkpoint is None:
            return None
        try:
            payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location=map_location)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported or malformed checkpoint: {checkpoint}")
        expected = configuration_digest(config)
        observed = payload.get("config_digest")
        if observed != expected and not force:
            raise ValueError(
                f"Checkpoint configuration mismatch for {checkpoint}: "
                f"saved={observed}, current={expected}. Use force_resume only if intentional."
            )
        return payload

    def restore_modules(
        self,
        payload: dict[str, Any],
        modules: dict[str, Any],
        *,
        optimizer: Any | None = None,
        restore_rng: bool = True,
    ) -> None:
        import torch

        saved_modules = payload.get("modules")
        expected_names = {key for key, value in modules.items() if value is not None}
        if not isinstance(saved_modules, dict) or set(saved_modules) != expected_names:
            raise ValueError("Checkpoint trainable-module set does not match the current run")
        for name, module in modules.items():
            if module is not None:
                incompatible = module.load_state_dict(saved_modules[name], strict=False)
                expected_trainable = {
                    key for key, parameter in module.named_parameters()
                    if parameter.requires_grad
                }
                missing_trainable = expected_trainable.intersection(incompatible.missing_keys)
                if missing_trainable:
                    raise ValueError(
                        f"Checkpoint is missing trainable parameters for {name}: "
                        + ", ".join(sorted(missing_trainable)[:8])
                    )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if restore_rng and "rng_state" in payload:
            _restore_rng_state(payload["rng_state"])

    def load(
        self,
        modules: dict[str, Any],
        config: dict[str, Any],
        *,
        resume_from: str | os.PathLike[str] | None = None,
        optimizer: Any | None = None,
        force: bool = False,
        map_location: Any = "cpu",
    ) -> dict[str, Any] | None:
        payload = self.inspect(
            config,
            resume_from=resume_from,
            force=force,
            map_location=map_location,
        )
        if payload is None:
            return None
        self.restore_modules(payload, modules, optimizer=optimizer)
        return payload
