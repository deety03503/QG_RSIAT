"""RSIAT entry point with validated QR-RSIAT runtime configuration."""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

from qrsiat.config.merge import merge_config as merge_qr_config
from qrsiat.data.registry import prepare_dataset_root
from qrsiat.distributed.collectives import broadcast_object
from qrsiat.distributed.setup import cleanup_distributed, initialize_distributed
from qrsiat.hardware.detect import detect_hardware
from qrsiat.hardware.plan import create_runtime_plan
from qrsiat.runtime.context import create_runtime_context
from qrsiat.runtime.optimize import configure_backend
from qrsiat.utils.io import write_json_atomic
from qrsiat.utils.report import write_runtime_report
from trainer import RSIAT_train


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RSIAT or opt-in QR-RSIAT continual learning."
    )
    parser.add_argument("--config", default="./exps/adapter_imageneta.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("auto", "cpu", "single", "ddp", "jobpar"))
    parser.add_argument("--aligner", choices=("rae", "qhybrid"))
    parser.add_argument("--lambda_qrel", type=float)
    parser.add_argument("--kernel", choices=("quantum", "cosine", "rbf", "mlp"))
    parser.add_argument("--orth", choices=("plain", "qweighted"))
    parser.add_argument("--rs_kernel", choices=("cosine", "quantum"))
    parser.add_argument("--n_qubits", type=int)
    parser.add_argument("--quantum_layers", type=int)
    parser.add_argument(
        "--quantum_centering",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--pair_gather", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lr_scale_by_world", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--max_workers", type=int)
    parser.add_argument("--prefetch_factor", type=int)
    parser.add_argument("--use_compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze_cast", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grad_checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--probe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output_dir", "--out", dest="output_dir")
    parser.add_argument("--data_root")
    parser.add_argument("--pretrained_weights")
    parser.add_argument("--orth_top_k", type=int)
    parser.add_argument("--orth_temperature", type=float)
    parser.add_argument("--orth_epsilon", type=float)
    parser.add_argument("--resume", choices=("auto", "none"))
    parser.add_argument("--resume_from")
    parser.add_argument("--force_resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--time_budget_h", type=float)
    return parser.parse_args()


def load_json(settings_path: str | Path) -> dict[str, Any]:
    path = Path(settings_path)
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read experiment configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment configuration must be a JSON object: {path}")
    return payload


def merge_configs(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    raw = vars(args)
    overrides = {
        key: value
        for key, value in raw.items()
        if key != "config" and value is not None
    }
    return merge_qr_config(config, overrides)


def _runtime(args: dict[str, Any]) -> tuple[Any, Any, Any]:
    profile = detect_hardware(".")
    plan = create_runtime_plan(
        profile,
        requested_mode=args["mode"],
        global_batch=int(args.get("batch_size", 1)),
        eval_batch_size=int(args.get("batch_size", 1)),
        num_workers=args.get("num_workers"),
        max_workers=args["max_workers"],
        prefetch_factor=args["prefetch_factor"],
        use_compile=args["use_compile"],
        grad_checkpointing=args["grad_checkpointing"],
    )
    if plan.mode == "ddp":
        state = initialize_distributed("ddp")
        context = create_runtime_context(
            plan, rank=state.rank, local_rank=state.local_rank
        )
    else:
        context = create_runtime_context(plan)
    configure_backend(plan)
    return profile, plan, context


def main() -> None:
    options = parse_arguments()
    config = load_json(options.config)
    args = merge_configs(options, config)
    if args.get("time_budget_h") is not None:
        raise ValueError(
            "--time_budget_h is enforced by scripts/launch.py --queue only; "
            "direct training cannot safely interrupt inside a task."
        )
    if args["smoke"]:
        args["output_dir"] = str(Path(args["output_dir"]).expanduser() / "smoke")
    args["config_path"] = str(Path(options.config).resolve())
    out_dir = Path(args["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qrsiat")
    logger.setLevel(logging.INFO)
    context = None
    try:
        profile, plan, context = _runtime(args)
        if args.get("probe") is None:
            args["probe"] = profile.platform == "kaggle"
        args["runtime_context"] = context
        args["runtime_plan"] = plan
        if args.get("pair_gather") is None:
            args["pair_gather"] = plan.mode == "ddp"
        args["device"] = [context.device]
        run_config = {
            key: value
            for key, value in args.items()
            if key not in {
                "runtime_context", "runtime_plan", "checkpoint_manager", "device",
                "checkpoint_config", "run_config", "config_path",
            }
        }
        args["run_config"] = run_config
        args["checkpoint_config"] = {
            key: value
            for key, value in run_config.items()
            if key not in {
                "output_dir", "data_root", "resume", "resume_from",
                "force_resume", "time_budget_h",
            }
        }
        if context.is_main:
            write_runtime_report(out_dir / "hardware.json", profile, plan)
            write_json_atomic(out_dir / "plan.json", plan.to_dict())
            write_json_atomic(out_dir / "config.json", args["run_config"])
        if plan.mode == "cpu":
            raise RuntimeError(
                "Training on CPU is intentionally disabled for this ViT workload. "
                "Use scripts/doctor.py for CPU-only diagnostics, then run scripts/launch.py "
                "on a Kaggle GPU runtime."
            )
        configure_backend(plan, logger)
        dataset = args.get("dataset", "").lower()
        data_root = args.get("data_root")
        if dataset in {"cifar224", "cifar100"}:
            os.environ["RSIAT_CIFAR_ROOT"] = str(data_root or "./data/datasets")
        preparation = {"error": None, "root": None}
        if context.is_main:
            try:
                location = prepare_dataset_root(
                    dataset,
                    data_root=data_root,
                    repository_root=".",
                )
                preparation["root"] = str(location.root)
                if location.is_cifar:
                    prepared_cifar = Path(data_root or "./data/datasets") / "cifar224"
                    if (prepared_cifar / "train").is_dir() and (prepared_cifar / "test").is_dir():
                        os.environ["RSIAT_DATASET_ROOT"] = str(prepared_cifar.resolve())
                else:
                    os.environ["RSIAT_DATASET_ROOT"] = str(location.root)
            except Exception:
                preparation["error"] = traceback.format_exc()
        if context.world_size > 1:
            preparation = broadcast_object(preparation, src=0)
        if preparation["error"] is not None:
            raise RuntimeError(
                "Dataset preparation failed on rank 0:\n" + preparation["error"]
            )
        RSIAT_train(args)
    except Exception:
        if context is None and int(os.environ.get("RANK", "0")) != 0:
            pass
        elif context is None or context.is_main:
            try:
                (out_dir / "FAILED.txt").write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )
            except OSError as write_error:
                logger.error("Failed to write run failure report: %s", write_error)
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
