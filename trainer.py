import sys
import logging
import copy
import torch
from utils import model_factory
from data.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np
import torch.distributed as dist


def RSIAT_train(args):
    configured_seeds = args["seed"]
    seed_list = list(configured_seeds) if isinstance(configured_seeds, (list, tuple)) else [configured_seeds]
    if not seed_list:
        raise ValueError("seed must contain at least one run seed")
    configured_num_worker = args["num_worker"]
    distributed_initialized_here = _configure_distributed(args)

    try:
        sum_seed = 0.0
        for seed in seed_list:
            run_args = copy.deepcopy(args)
            run_args["seed"] = int(seed)
            run_args["num_worker"] = configured_num_worker
            sum_seed += _train(run_args)
        avg_seed = sum_seed / len(seed_list)
        if args["rank"] == 0:
            print('Average Seed Accuracy (CNN):', avg_seed)
            logging.info("Average Seed Accuracy (CNN): {}".format(avg_seed))
    finally:
        if distributed_initialized_here:
            dist.destroy_process_group()


def _configure_distributed(args):
    if not isinstance(args["num_worker"], int) or isinstance(
        args["num_worker"], bool
    ):
        raise ValueError("num_worker must be an integer GPU count")
    if args["num_worker"] < 0:
        raise ValueError("num_worker must be non-negative (0 selects CPU)")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", str(world_size))
    )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    elif world_size > 1:
        if args["num_worker"] != local_world_size:
            raise ValueError(
                f"num_worker ({args['num_worker']}) must match torchrun world size "
                f"per node ({local_world_size})"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun DDP currently requires CUDA GPUs")
        if not dist.is_nccl_available():
            raise RuntimeError(
                "Multi-GPU CUDA training requires a PyTorch build with NCCL support"
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds the {torch.cuda.device_count()} "
                "visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
    elif args["num_worker"] > 1:
        raise RuntimeError(
            "Multiple GPUs now use DDP. Launch with torchrun "
            "--standalone --nproc_per_node=<num_worker> main.py ..."
        )

    args["distributed"] = world_size > 1
    args["rank"] = rank
    args["world_size"] = world_size
    args["local_rank"] = local_rank
    return initialized_here

def _train(args):
    if not isinstance(args["num_worker"], int) or isinstance(args["num_worker"], bool):
        raise ValueError("num_worker must be an integer GPU count")
    if args["num_worker"] < 0:
        raise ValueError("num_worker must be non-negative (0 selects CPU)")
    if not isinstance(args["data_loader_workers"], int) or isinstance(
        args["data_loader_workers"], bool
    ):
        raise ValueError("data_loader_workers must be a non-negative integer")
    if args["data_loader_workers"] < 0:
        raise ValueError("data_loader_workers must be a non-negative integer")
    if args["lambda_qrel"] < 0:
        raise ValueError("lambda_qrel must be non-negative")
    if args["qbits"] not in (4, 8, 12):
        raise ValueError("qbits must be one of 4, 8, or 12")
    if args["q_layers"] not in (1, 2, 3):
        raise ValueError("q_layers must be one of 1, 2, or 3")
    if args["orth_topk"] < 1:
        raise ValueError("orth_topk must be positive")
    if args["orth_tau"] <= 0:
        raise ValueError("orth_tau must be positive")
    if args["aligner"] not in ("rae", "qhybrid"):
        raise ValueError(f"Unsupported aligner: {args['aligner']}")
    if args["orth"] not in ("plain", "qweighted"):
        raise ValueError(f"Unsupported orth mode: {args['orth']}")
    if args["kernel"] not in ("quantum", "cosine", "rbf", "mlp"):
        raise ValueError(f"Unsupported kernel: {args['kernel']}")
    if args["rs_kernel"] not in ("cosine", "quantum"):
        raise ValueError(f"Unsupported RS kernel: {args['rs_kernel']}")

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    is_main_process = args.get("rank", 0) == 0
    if is_main_process and not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    if is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=logfilename + ".log"),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.ERROR,
            handlers=[logging.NullHandler()],
            force=True,
        )
    if args.get("distributed"):
        dist.barrier()

    _set_random(args["seed"])
    _set_device(args)
    if is_main_process:
        print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["class_order_seed"],
        args["init_cls"],
        args["increment"],
    )
    model = model_factory.get_model(args["model_name"], args)

    if is_main_process:
        print()
    cnn_curve = {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        model.incremental_train(data_manager)
        quantum_aligner = getattr(model, "quantum_aligner", None)
        quantum_modules = (
            [quantum_aligner]
            if quantum_aligner is not None
            else [getattr(model, "quantum_feature_map", None)]
        )
        quantum_modules = [module for module in quantum_modules if module is not None]
        quantum_parameters = sum(
            parameter.numel()
            for module in quantum_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        logging.info("Trainable quantum parameters: {}".format(quantum_parameters))
        cnn_accy = model.eval_task()
        task_accuracy_summary = ", ".join(
            f"Task {task_index + 1}: {accuracy:.2f}%"
            for task_index, accuracy in enumerate(cnn_accy["task_accuracies"])
        )
        logging.info(
            "Task-wise accuracy after Task %s: %s",
            task + 1,
            task_accuracy_summary,
        )
        if is_main_process:
            print(
                f"Task-wise accuracy after Task {task + 1}: "
                f"{task_accuracy_summary}"
            )
        model.after_task()
        
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])


        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        if is_main_process:
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    return sum(cnn_curve["top1"])/len(cnn_curve["top1"])
       
def _set_device(args):
    if args.get("distributed"):
        local_rank = args["local_rank"]
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        args["device"] = [torch.device(f"cuda:{local_rank}")]
        return

    gpu_count = args["num_worker"]
    if gpu_count == 0:
        args["device"] = [torch.device("cpu")]
        logging.info(
            "PyTorch %s; using CPU; CUDA available=%s; visible GPUs=%d; "
            "CUDA_VISIBLE_DEVICES=%s",
            torch.__version__,
            torch.cuda.is_available(),
            torch.cuda.device_count(),
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        )
        return
    visible_gpu_count = torch.cuda.device_count()
    logging.info(
        "PyTorch %s; CUDA available=%s; visible GPUs=%d; CUDA_VISIBLE_DEVICES=%s",
        torch.__version__,
        torch.cuda.is_available(),
        visible_gpu_count,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("num_worker requests CUDA GPUs, but CUDA is unavailable")
    if gpu_count > visible_gpu_count:
        raise RuntimeError(
            f"num_worker requests {gpu_count} GPUs, but only "
            f"{visible_gpu_count} are visible to PyTorch"
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_ids = list(range(gpu_count))

    selected_device_ids = set(device_ids)
    for index in range(visible_gpu_count):
        properties = torch.cuda.get_device_properties(index)
        free_memory, total_memory = torch.cuda.mem_get_info(index)
        logging.info(
            "Visible GPU %s%s: %s, VRAM %.2f GiB total / %.2f GiB free",
            index,
            " (selected)" if index in selected_device_ids else "",
            properties.name,
            total_memory / (1024 ** 3),
            free_memory / (1024 ** 3),
        )

    args["device"] = [torch.device(f"cuda:{index}") for index in device_ids]


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
