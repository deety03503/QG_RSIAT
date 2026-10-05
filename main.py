import json
import argparse
from trainer import RSIAT_train


def main():
    args = parse_arguments()
    config = load_json(args.config)
    merged_config = merge_configs(args, config)
    RSIAT_train(merged_config)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train RSIAT or QR-RSIAT with a JSON experiment config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/adapter_imageneta.json",
        help="JSON file of settings.",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--num_worker", type=int, default=None)
    parser.add_argument("--data_loader_workers", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--aligner", choices=("rae", "qhybrid"), default=None)
    parser.add_argument("--lambda_qrel", type=float, default=None)
    parser.add_argument(
        "--kernel", choices=("quantum", "cosine", "rbf", "mlp"), default=None
    )
    parser.add_argument(
        "--orth", choices=("plain", "qweighted"), default=None
    )
    parser.add_argument(
        "--rs_kernel", choices=("cosine", "quantum"), default=None
    )
    parser.add_argument("--qbits", type=int, default=None)
    parser.add_argument("--q_layers", type=int, default=None)
    parser.add_argument("--orth_topk", type=int, default=None)
    parser.add_argument("--orth_tau", type=float, default=None)
    parser.add_argument("--orth_epsilon", type=float, default=None)
    parser.add_argument(
        "--orth_use_drift_comp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--center_features",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def load_json(settings_path):
    with open(settings_path, encoding="utf-8") as data_file:
        param = json.load(data_file)
    return param


def merge_configs(args, config):
    merged_config = dict(config)
    cli_values = vars(args)
    for key, value in cli_values.items():
        if key == "config" or value is not None:
            merged_config[key] = value

    defaults = {
        "class_order_seed": 1993,
        "num_worker": 2,
        "data_loader_workers": 8,
        "eval_interval": 1,
        "use_amp": True,
        "aligner": "rae",
        "lambda_qrel": 0.0,
        "kernel": "quantum",
        "orth": "plain",
        "rs_kernel": "cosine",
        "qbits": 8,
        "q_layers": 2,
        "orth_topk": 5,
        "orth_tau": 0.1,
        "orth_epsilon": 0.5,
        "orth_use_drift_comp": True,
        "center_features": True,
    }
    for key, value in defaults.items():
        merged_config.setdefault(key, value)
    return merged_config

if __name__ == '__main__':
    main()