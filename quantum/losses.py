import torch
import torch.nn.functional as F

from quantum.kernels import kernel_matrix


def qrel_loss(features, old_features, kind="quantum", feature_map=None):
    current_kernel = kernel_matrix(
        features, kind=kind, feature_map=feature_map
    )
    with torch.no_grad():
        old_kernel = kernel_matrix(
            old_features, kind=kind, feature_map=feature_map
        )
    return (current_kernel - old_kernel).square().sum()


def qorth_loss(
    features,
    prototypes,
    feature_map,
    top_k=5,
    temperature=0.1,
    epsilon=0.5,
):
    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("features and prototypes must be rank-2 tensors")
    if prototypes.shape[0] == 0:
        return features.new_zeros(())
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    similarities = kernel_matrix(
        features, prototypes, kind="quantum", feature_map=feature_map
    )
    weights = F.softmax(similarities / temperature, dim=1)
    selected_count = min(top_k, prototypes.shape[0])
    selected_indices = similarities.topk(selected_count, dim=1).indices
    selected_similarities = similarities.gather(1, selected_indices)
    selected_weights = weights.gather(1, selected_indices)
    return (
        selected_weights * F.relu(selected_similarities - epsilon)
    ).sum(dim=1).mean()
