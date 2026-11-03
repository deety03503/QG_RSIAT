import torch
import torch.nn as nn
import torch.nn.functional as F
from quantum.simulator import QuantumStateSimulator


class QSupConLoss(nn.Module):
    """
    Module 1 (Q-RS): Supervised Contrastive Loss in Quantum Hilbert Space.
        L_QSupCon = - sum_i (1 / |P(i)|) sum_{p in P(i)}
                    log [ exp(s(z_i, z_p) / tau) / sum_{a != i} exp(s(z_i, z_a) / tau) ]
    """

    def __init__(self, temperature: float = 0.15):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        simulator: QuantumStateSimulator,
    ) -> torch.Tensor:
        batch_size = features.size(0)
        if batch_size <= 1:
            return features.sum() * 0.0

        sim_matrix = simulator.overlap(features)  # (B, B) in [0, 1]
        logits = sim_matrix / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask_same = torch.eq(labels, labels.t()).float().to(features.device)
        mask_self = torch.eye(batch_size, device=features.device)
        mask_pos = mask_same * (1.0 - mask_self)
        mask_non_self = 1.0 - mask_self

        logits_masked = logits.masked_fill(mask_self.bool(), -1e9)
        logits_max, _ = torch.max(logits_masked, dim=1, keepdim=True)
        logits_shifted = logits - logits_max.detach()

        exp_logits = torch.exp(logits_shifted) * mask_non_self
        log_prob = logits_shifted - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-8
        )

        pos_count = mask_pos.sum(dim=1)
        valid_anchors = pos_count > 0
        if not valid_anchors.any():
            return sim_matrix.sum() * 0.0

        mean_log_prob_pos = (
            (mask_pos * log_prob).sum(dim=1)[valid_anchors] / pos_count[valid_anchors]
        )
        return -mean_log_prob_pos.mean()


class SelectiveQuantumOrthLoss(nn.Module):
    """
    Module 2B (Q-Orth): Selective Top-K Orthogonality Loss in Quantum Hilbert Space.
        L_orth = sum_i sum_{c in TopK(i)} w_{ic} * max(0, s(A(z_i^old), p_hat_c) - eps)^2
    """

    def __init__(
        self,
        topk: int = 5,
        margin_eps: float = 0.15,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.topk = topk
        self.margin_eps = margin_eps
        self.temperature = temperature

    def forward(
        self,
        z_aligned_old: torch.Tensor,
        z_cur: torch.Tensor,
        old_protos_aligned: torch.Tensor,
        simulator: QuantumStateSimulator,
    ) -> torch.Tensor:
        if old_protos_aligned.size(0) == 0 or z_aligned_old.size(0) == 0:
            return z_aligned_old.sum() * 0.0

        psi_old_aligned = simulator.encode(z_aligned_old)
        psi_protos = simulator.encode(old_protos_aligned)
        s_aligned = simulator.overlap_from_states(psi_old_aligned, psi_protos)

        if z_cur is not None:
            psi_cur = simulator.encode(z_cur)
            s_cur = simulator.overlap_from_states(psi_cur, psi_protos.detach())
            s_ic = 0.5 * (s_aligned + s_cur)
        else:
            s_ic = s_aligned

        k = min(self.topk, s_ic.size(1))
        topk_vals, _ = torch.topk(s_ic, k=k, dim=1)

        weights = F.softmax(topk_vals.detach() / self.temperature, dim=1)
        penalty = F.relu(topk_vals - self.margin_eps).square()

        return (weights * penalty).sum(dim=1).mean()


class TopKQuantumRelLoss(nn.Module):
    """
    Module 3 (Top-K Q-Rel): Local Quantum Relational Knowledge Distillation Loss.
        p_{ij} = exp(s(z_i, z_j) / tau) / sum_{k in TopK(i)} exp(s(z_i, z_k) / tau)
        L_qrel = sum_i sum_{j in TopK(i)} p^old_{ij} |s^cur_{ij} - s^old_{ij}|
                 + tau * KL(p^old_i || p^cur_i)
    """

    def __init__(self, topk: int = 8, temperature: float = 0.15):
        super().__init__()
        self.topk = topk
        self.temperature = temperature

    def forward(
        self,
        z_cur: torch.Tensor,
        z_old: torch.Tensor,
        simulator: QuantumStateSimulator,
    ) -> torch.Tensor:
        batch_size = z_cur.size(0)
        if batch_size <= 1:
            return z_cur.sum() * 0.0

        with torch.no_grad():
            s_old = simulator.overlap(z_old)

        s_cur = simulator.overlap(z_cur)

        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=z_cur.device)
        s_old_masked = s_old.masked_fill(eye_mask, -1.0)

        k = min(self.topk, batch_size - 1)
        s_old_topk, topk_indices = torch.topk(s_old_masked, k=k, dim=1)
        s_cur_topk = torch.gather(s_cur, dim=1, index=topk_indices)

        p_old = F.softmax(s_old_topk / self.temperature, dim=1)
        log_p_old = F.log_softmax(s_old_topk / self.temperature, dim=1)
        log_p_cur = F.log_softmax(s_cur_topk / self.temperature, dim=1)

        amp_diff = (p_old * torch.abs(s_cur_topk - s_old_topk)).sum(dim=1).mean()
        kl_div = (p_old * (log_p_old - log_p_cur)).sum(dim=1).mean()

        return amp_diff + self.temperature * kl_div

