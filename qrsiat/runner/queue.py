"""GPU job-parallel experiment queue with durable completion markers."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from qrsiat.utils.io import write_json_atomic


def _load_jobs(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load experiment queue {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        raise ValueError("Queue JSON must contain an 'experiments' list")
    base_config = payload.get("base_config")
    if not isinstance(base_config, str) or not base_config.strip():
        raise ValueError("Queue JSON must define a non-empty 'base_config'")
    jobs = payload["experiments"]
    seen_jobs: set[tuple[str, int]] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"Queue experiment at index {index} must be an object")
        if not isinstance(job.get("id"), str) or not job["id"].strip():
            raise ValueError(f"Queue experiment at index {index} needs a non-empty id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job["id"]):
            raise ValueError(
                f"Queue experiment id {job['id']!r} must contain only letters, "
                "digits, dots, underscores, and hyphens, and start with a letter or digit"
            )
        if not isinstance(job.get("overrides", {}), dict):
            raise ValueError(f"Queue experiment {job.get('id')} overrides must be an object")
        seeds = job.get("seeds", [1993])
        if not isinstance(seeds, list) or not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
        ):
            raise ValueError(f"Queue experiment {job['id']} seeds must be non-negative integers")
        for seed in seeds:
            job_key = (job["id"], seed)
            if job_key in seen_jobs:
                raise ValueError(f"Duplicate queue job {job['id']!r} with seed {seed}")
            seen_jobs.add(job_key)
    return path.parent / base_config, jobs


def _run_one(
    job: dict[str, Any],
    seed: int,
    base_config_path: Path,
    out_dir: Path,
    device_index: int,
    common_args: list[str],
) -> tuple[str, str]:
    job_id = f"{job['id']}_s{seed}"
    job_dir = out_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    done_marker = job_dir / "DONE"
    if done_marker.is_file():
        return job_id, "skipped"

    config_path = job_dir / "job-config.json"
    try:
        config = json.loads(base_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read base experiment config {base_config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Base experiment config must be an object: {base_config_path}")
    config.update(job.get("overrides", {}))
    config["seed"] = [seed]
    config["experiment_id"] = job["id"]
    write_json_atomic(config_path, config)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "main.py"),
        "--config",
        str(config_path.resolve()),
        "--mode",
        "single",
        "--output_dir",
        str(job_dir.resolve()),
        *common_args,
    ]
    try:
        subprocess.run(command, check=True, env=environment)
    except subprocess.CalledProcessError as exc:
        (job_dir / "FAILED.txt").write_text(
            f"Command failed with exit code {exc.returncode}: {command!r}\n",
            encoding="utf-8",
        )
        raise
    done_marker.write_text("completed\n", encoding="utf-8")
    return job_id, "completed"


def run_experiment_queue(
    queue_path: str | Path,
    *,
    output_dir: str | Path,
    gpu_count: int,
    time_budget_h: float | None = None,
    reserve_minutes: int = 5,
    common_args: list[str] | None = None,
) -> list[dict[str, str]]:
    if gpu_count < 1:
        raise ValueError("The experiment queue requires at least one visible GPU")
    if reserve_minutes < 0:
        raise ValueError("reserve_minutes must be non-negative")
    if time_budget_h is not None and time_budget_h <= 0:
        raise ValueError("time_budget_h must be positive or None")
    queue_file = Path(queue_path).expanduser().resolve()
    base_config, experiments = _load_jobs(queue_file)
    if not base_config.is_file():
        raise FileNotFoundError(f"Queue base config does not exist: {base_config}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    pending = [
        (experiment, seed)
        for experiment in experiments
        for seed in experiment.get("seeds", [1993])
        if not (destination / f"{experiment['id']}_s{seed}" / "DONE").is_file()
    ]
    started = time.monotonic()
    results: list[dict[str, str]] = []
    failures: list[BaseException] = []

    def worker(slot: int, item: tuple[dict[str, Any], int]) -> tuple[str, str]:
        experiment, seed = item
        return _run_one(
            experiment,
            seed,
            base_config,
            destination,
            slot,
            common_args or [],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:
        active: dict[concurrent.futures.Future[tuple[str, str]], int] = {}
        next_job = 0
        while next_job < len(pending) or active:
            while next_job < len(pending) and len(active) < gpu_count:
                if time_budget_h is not None:
                    remaining = time_budget_h * 3600 - (time.monotonic() - started)
                    if remaining <= reserve_minutes * 60:
                        break
                occupied_slots = set(active.values())
                slot = next(index for index in range(gpu_count) if index not in occupied_slots)
                future = executor.submit(worker, slot, pending[next_job])
                active[future] = slot
                next_job += 1
            if not active:
                break
            done, _ = concurrent.futures.wait(
                active,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                active.pop(future)
                try:
                    job_id, status = future.result()
                    results.append({"job": job_id, "status": status})
                except Exception as exc:
                    failures.append(exc)
            if failures:
                for future in active:
                    future.cancel()
                break

    write_json_atomic(
        destination / "queue-status.json",
        {
            "results": results,
            "pending_not_started": len(pending) - len(results) - len(failures),
            "failures": [str(error) for error in failures],
        },
    )
    if failures:
        raise RuntimeError(f"{len(failures)} queued experiment(s) failed") from failures[0]
    return results
