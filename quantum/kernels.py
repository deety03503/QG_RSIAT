import torch.nn.functional as F

from quantum.simulator import fidelity


def kernel_matrix(x, y=None, kind="quantum", feature_map=None, rbf_gamma=None):
    """Build a batch Gram/cross-kernel matrix.

    The MLP option is a tanh kernel over unit-normalized features, with unit
    bias. The RBF default uses gamma=1/input_dim.
    """
    y = x if y is None else y
    if x.ndim != 2 or y.ndim != 2 or x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must be rank-2 tensors with matching features")
    if kind == "quantum":
        if feature_map is None:
            raise ValueError("feature_map is required for the quantum kernel")
        combined = torch.cat((x, y), dim=0)
        states = feature_map(combined)
        state_x, state_y = states.split((x.shape[0], y.shape[0]), dim=0)
        return fidelity(state_x, state_y)
    if kind == "cosine":
        return F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).transpose(0, 1)
    if kind == "rbf":
        gamma = 1.0 / x.shape[-1] if rbf_gamma is None else rbf_gamma
        distances = torch.cdist(x, y).square()
        return torch.exp(-gamma * distances)
    if kind == "mlp":
        normalized_x = F.normalize(x, dim=-1)
        normalized_y = F.normalize(y, dim=-1)
        return torch.tanh(normalized_x @ normalized_y.transpose(0, 1) + 1.0)
    raise ValueError(f"unsupported kernel kind: {kind}")
