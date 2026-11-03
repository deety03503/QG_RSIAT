import json
import os
import argparse
from trainer import RSIAT_train


DEFAULT_QR_CONFIG = {
    "aligner": "rae",
    "lambda_qrel": 0.0,
    "kernel": "cosine",
    "orth": "plain",
    "rs_kernel": "cosine",
    "n_qubits": 8,
    "quantum_layers": 2,
    "quantum_centering": True,
    "cycle": False,
    "orth_top_k": 3,
    "orth_temperature": 0.1,
    "orth_epsilon": 0.2,
    "qrel_top_k": 5,
    "qrel_temperature": 0.1,
    "rs_loss_type": "supcon",
    "rs_temperature": 0.1,
    "rs_encoder_freeze": False,
    "qhybrid_hidden_dim": 128,
    "qhybrid_entangled_readout": True,
    "qhybrid_initial_scale": 0.1,
    "qhybrid_warm_start": False,
}


def _str2bool(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"yes", "true", "t", "y", "1"}:
        return True
    if val in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


def main():
    args = parse_arguments()
    config = load_json(args.config)
    merged_config = merge_configs(args, config)
    RSIAT_train(merged_config)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    """
    Please specify the corresponding JSON file!!!!
    """
    parser.add_argument('--config', type=str, default='./exps/adapter_imageneta.json',
                        help='Json file of settings.')

    # QR-RSIAT optional flags (default=None so CLI only overrides JSON when explicitly passed)
    parser.add_argument('--aligner', type=str, choices=['rae', 'qhybrid'], default=None,
                        help='Feature aligner for incremental tasks (rae or qhybrid).')
    parser.add_argument('--lambda_qrel', type=float, default=None,
                        help='Weight for Quantum Relational Distillation Loss.')
    parser.add_argument('--kernel', type=str, choices=['quantum', 'cosine', 'rbf', 'mlp'], default=None,
                        help='Pairwise kernel type for relational distillation.')
    parser.add_argument('--orth', type=str, choices=['plain', 'qweighted'], default=None,
                        help='Old-prototype orthogonality mode (plain or qweighted).')
    parser.add_argument('--rs_kernel', type=str, choices=['cosine', 'quantum'], default=None,
                        help='Kernel type for Base-task RS_Loss (cosine or quantum).')
    parser.add_argument('--n_qubits', type=int, default=None,
                        help='Number of qubits q in the quantum circuit.')
    parser.add_argument('--quantum_layers', type=int, default=None,
                        help='Number of circuit layers L.')
    parser.add_argument('--quantum_centering', type=_str2bool, nargs='?', const=True, default=None,
                        help='Enable/disable feature centering before the quantum angle encoder.')
    parser.add_argument('--no_quantum_centering', '--no-quantum-centering',
                        dest='quantum_centering', action='store_false',
                        help='Disable feature centering before the quantum angle encoder.')
    parser.add_argument('--cycle', type=_str2bool, nargs='?', const=True, default=None,
                        help='Enable/disable cyclic CNOT (q-1 -> 0) at the end of each layer.')
    parser.add_argument('--no_cycle', '--no-cycle',
                        dest='cycle', action='store_false',
                        help='Disable cyclic CNOT at the end of each layer.')
    parser.add_argument('--orth_top_k', type=int, default=None,
                        help='Top-k nearest old prototypes for qweighted orthogonality.')
    parser.add_argument('--orth_temperature', type=float, default=None,
                        help='Softmax temperature tau for qweighted orthogonality.')
    parser.add_argument('--orth_epsilon', type=float, default=None,
                        help='Hinge threshold epsilon for qweighted orthogonality.')
    parser.add_argument('--qrel_top_k', type=int, default=None,
                        help='Top-k nearest neighbors per sample for relational distillation.')
    parser.add_argument('--qrel_temperature', type=float, default=None,
                        help='Softmax temperature tau for top-k relational distillation.')
    parser.add_argument('--rs_margin', type=float, default=None,
                        help='Optional override for RS_Loss margin.')
    parser.add_argument('--rs_loss_type', type=str, choices=['supcon', 'hinge'], default=None,
                        help='Loss type for Base-task Representation Steering (supcon or hinge).')
    parser.add_argument('--rs_temperature', type=float, default=None,
                        help='Softmax temperature tau for base-task Q-SupCon loss.')
    parser.add_argument('--rs_encoder_freeze', type=_str2bool, nargs='?', const=True, default=None,
                        help='Freeze classical angle encoder of rs_kernel to avoid representation shortcut.')
    parser.add_argument('--qhybrid_hidden_dim', type=int, default=None,
                        help='Hidden dimension for QHybridAligner bottleneck up-projection (default: 128).')
    parser.add_argument('--qhybrid_entangled_readout', type=_str2bool, nargs='?', const=True, default=None,
                        help='Enable/disable 2-qubit correlators (ZZ & XX) in QHybridAligner readout.')
    parser.add_argument('--qhybrid_initial_scale', type=float, default=None,
                        help='Initial learnable residual scale for QHybridAligner (default: 0.1).')
    parser.add_argument('--qhybrid_warm_start', type=_str2bool, nargs='?', const=True, default=None,
                        help='Preserve and warm-start QHybridAligner across incremental tasks.')
    parser.add_argument('--qhybrid_lr', type=float, default=None,
                        help='Custom learning rate for QHybridAligner (defaults to init_lr).')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Optional experiment log prefix override.')
    return parser.parse_args()


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def validate_config(config):
    config["aligner"] = str(config.get("aligner", "rae")).lower()
    config["kernel"] = str(config.get("kernel", "cosine")).lower()
    config["orth"] = str(config.get("orth", "plain")).lower()
    config["rs_kernel"] = str(config.get("rs_kernel", "cosine")).lower()
    config["lambda_qrel"] = float(config.get("lambda_qrel", 0.0))
    config["n_qubits"] = int(config.get("n_qubits", 8))
    config["quantum_layers"] = int(config.get("quantum_layers", 2))
    config["quantum_centering"] = bool(config.get("quantum_centering", True))
    config["cycle"] = bool(config.get("cycle", False))
    config["orth_top_k"] = int(config.get("orth_top_k", 3))
    config["orth_temperature"] = float(config.get("orth_temperature", 0.1))
    config["orth_epsilon"] = float(config.get("orth_epsilon", 0.2))
    config["qrel_top_k"] = int(config.get("qrel_top_k", 5))
    config["qrel_temperature"] = float(config.get("qrel_temperature", 0.1))
    config["rs_loss_type"] = str(config.get("rs_loss_type", "supcon")).lower()
    config["rs_temperature"] = float(config.get("rs_temperature", 0.1))
    config["rs_encoder_freeze"] = bool(config.get("rs_encoder_freeze", False))
    config["qhybrid_hidden_dim"] = int(config.get("qhybrid_hidden_dim", 128))
    config["qhybrid_entangled_readout"] = bool(config.get("qhybrid_entangled_readout", True))
    config["qhybrid_initial_scale"] = float(config.get("qhybrid_initial_scale", 0.1))
    config["qhybrid_warm_start"] = bool(config.get("qhybrid_warm_start", False))

    if config["aligner"] not in {"rae", "qhybrid"}:
        raise ValueError(f"Unsupported aligner: {config['aligner']}")
    if config["kernel"] not in {"quantum", "cosine", "rbf", "mlp"}:
        raise ValueError(f"Unsupported kernel: {config['kernel']}")
    if config["orth"] not in {"plain", "qweighted"}:
        raise ValueError(f"Unsupported orth mode: {config['orth']}")
    if config["rs_kernel"] not in {"cosine", "quantum"}:
        raise ValueError(f"Unsupported rs_kernel: {config['rs_kernel']}")
    if config["rs_loss_type"] not in {"supcon", "hinge"}:
        raise ValueError(f"Unsupported rs_loss_type: {config['rs_loss_type']}")

    if config["rs_kernel"] == "quantum" and config["rs_loss_type"] == "hinge":
        rs_margin = float(config.get("rs_margin", 0.5))
        if not (0.0 <= rs_margin < 1.0):
            raise ValueError(
                f"Invalid rs_margin={rs_margin} for rs_kernel='quantum'. "
                "Because quantum state fidelity lies in [0, 1], rs_margin must satisfy "
                "0.0 <= rs_margin < 1.0 (e.g., --rs_margin 0.4), otherwise negative-pair "
                "repulsion in RS_Loss is identically zero."
            )


def merge_configs(args, config):
    merged_config = dict(DEFAULT_QR_CONFIG)
    merged_config.update(config)
    for key, value in vars(args).items():
        if value is not None:
            merged_config[key] = value
    validate_config(merged_config)
    return merged_config


if __name__ == '__main__':
    main()
