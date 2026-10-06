import sys
import logging
import copy
import torch
from utils import model_factory
from data.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
from pathlib import Path
from qrsiat.utils.seed import seed_everything
from qrsiat.utils.io import write_json_atomic
from qrsiat.training.checkpoint import CheckpointManager


def RSIAT_train(args):
    seed_list = copy.deepcopy(args["seed"])
    if isinstance(seed_list, int):
        seed_list = [seed_list]

    sum_seed = 0.0
    for seed in seed_list:
        args["seed"] = seed
        runtime_context = args.get("runtime_context")
        if runtime_context is not None:
            args["device"] = [runtime_context.device]
        args["checkpoint_manager"] = CheckpointManager(
            Path(args.get("output_dir", "./out")) / args["dataset"] / f"seed_{seed}"
        )
        sum_seed += _train(args)
    avg_seed = sum_seed / max(1, len(seed_list))
    if args.get("runtime_context") is None or args["runtime_context"].is_main:
        print('Average Seed Accuracy (CNN):', avg_seed)
        logging.info("Average Seed Accuracy (CNN): {}".format(avg_seed))

def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
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
    context = args.get("runtime_context")
    handlers = [logging.StreamHandler(sys.stdout)]
    if context is None or context.is_main:
        handlers.insert(0, logging.FileHandler(filename=logfilename + ".log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=handlers,
    )

    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = model_factory.get_model(args["model_name"], args)
    start_task = model.resume_from_checkpoint(data_manager)

    print()    
    accuracy_history = list(args.get("accuracy_curve", []))
    cnn_curve = args.get("metric_curve", {"top1": accuracy_history})
    task_limit = min(data_manager.nb_tasks, 2) if args.get("smoke", False) else data_manager.nb_tasks
    for task in range(start_task, task_limit):
        if context is None or context.is_main:
            logging.info("All params: {}".format(count_parameters(model._network)))
            logging.info(
                "Trainable params: {}".format(count_parameters(model._network, True))
            )
        
        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        model.after_task()
        
        # save_path_base = f"ckpt/{args['prefix']}/{args['dataset']}/{args['init_cls']}_{args['increment']}"
        # if not os.path.exists(save_path_base):
        #     os.makedirs(save_path_base)
        # save_path = f"ckpt/{args['prefix']}/{args['dataset']}/{args['init_cls']}_{args['increment']}/task_{task}.pth"
        # torch.save(model._network.state_dict(), save_path)
        # logging.info(f"Saved model checkpoint: {save_path}")
     
        if context is None or context.is_main:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        topk_key = next(
            (key for key in cnn_accy if key.startswith("top") and key != "top1"),
            "top1",
        )
        cnn_curve[topk_key] = cnn_curve.get(topk_key, [])
        cnn_curve[topk_key].append(cnn_accy[topk_key])
        args["accuracy_curve"] = list(cnn_curve["top1"])
        args["metric_curve"] = cnn_curve
        model._save_task_checkpoint()


        if context is None or context.is_main:
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN {} curve: {}".format(topk_key, cnn_curve[topk_key]))

        average_accuracy = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        if args.get("runtime_context") is None or args["runtime_context"].is_main:
            print('Average Accuracy (CNN):', average_accuracy)
            logging.info("Average Accuracy (CNN): {}".format(average_accuracy))
            write_json_atomic(
                Path(args.get("output_dir", "./out")) / args["dataset"] / f"seed_{args['seed']}" / "metrics.json",
                {
                    "experiment_id": args.get("experiment_id", args["dataset"]),
                    "cnn_curve": cnn_curve,
                    "average_top1": average_accuracy,
                },
            )
    if not cnn_curve["top1"]:
        raise RuntimeError("No completed or resumed task produced top-1 accuracy")
    average_accuracy = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
    if context is None or context.is_main:
        write_json_atomic(
            Path(args.get("output_dir", "./out")) / args["dataset"] / f"seed_{args['seed']}" / "metrics.json",
            {
                "experiment_id": args.get("experiment_id", args["dataset"]),
                "cnn_curve": cnn_curve,
                "average_top1": average_accuracy,
            },
        )
    return average_accuracy
       
def _set_device(args):
    context = args.get("runtime_context")
    if context is not None:
        args["device"] = [context.device]
        return
    devices = args.get("device", [])
    if devices == -1 or devices == ["cpu"]:
        args["device"] = [torch.device("cpu")]
        return
    raise RuntimeError(
        "No RuntimeContext was provided. Start through main.py or scripts/launch.py "
        "so hardware and device selection are validated first."
    )


def _set_random(args):
    context = args.get("runtime_context")
    seed_everything(
        int(args.get("seed", 1993)),
        rank=context.rank if context is not None else 0,
        deterministic=True,
    )


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
