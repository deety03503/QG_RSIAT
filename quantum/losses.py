import torch
import torch.nn as nn
import torch.nn.functional as F
from quantum.simulator import QuantumStateSimulator


class QSupConLoss(nn.Module):
    """
    Module 3.1 (Dual Q-RS): Dual Quantum-Cosine Representation Steering Loss for Base Task (t = 0).
    Combines Quantum-Hardness Modulated Spherical Cosine Steering (L_RS^{Q-weighted})
    with Multi-Subspace Supervised Contrastive Learning in Quantum Hilbert Space (L_QSupCon).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        rs_margin: float = 0.5,
        temperature: float = 0.15,
        lambda_q: float = 0.25,
        hard_weight: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.rs_margin = rs_margin
        self.temperature = temperature
        self.lambda_q = lambda_q
        self.hard_weight = hard_weight

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        simulator: QuantumStateSimulator,
        return_components: bool = False,
    ):
        batch_size = features.size(0)
        if batch_size <= 1:
            zero_loss = features.sum() * 0.0
            if return_components:
                return zero_loss, {"loss_rs_cos": zero_loss, "loss_qsupcon": zero_loss}
            return zero_loss

        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(features_norm, features_norm.t())
        q_sim = simulator.overlap(features)  # (B, B) in [0, 1]

        labels_col = labels.contiguous().view(-1, 1)
        mask_same = torch.eq(labels_col, labels_col.t()).float().to(features.device)
        mask_self = torch.eye(batch_size, device=features.device)
        mask_pos = mask_same * (1.0 - mask_self)
        mask_neg = 1.0 - mask_same
        mask_non_self = 1.0 - mask_self

        q_det = q_sim.detach()
        w_pos = 1.0 + self.hard_weight * (1.0 - q_det)
        w_neg = 1.0 + self.hard_weight * q_det

        pos_loss = (w_pos * F.relu(1.0 - cos_sim) * mask_pos).sum() / (
            mask_pos.sum() + 1e-6
        )
        neg_loss = (w_neg * F.relu(cos_sim - self.rs_margin) * mask_neg).sum() / (
            mask_neg.sum() + 1e-6
        )
        loss_rs_cos = pos_loss + self.alpha * neg_loss

        logits = q_sim / self.temperature
        logits_masked = logits.masked_fill(mask_self.bool(), -1e9)
        logits_max, _ = torch.max(logits_masked, dim=1, keepdim=True)
        logits_shifted = logits - logits_max.detach()

        exp_logits = torch.exp(logits_shifted) * mask_non_self
        log_prob = logits_shifted - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-8
        )

        pos_count = mask_pos.sum(dim=1)
        valid_anchors = pos_count > 0
        if valid_anchors.any():
            mean_log_prob_pos = (
                (mask_pos * log_prob).sum(dim=1)[valid_anchors]
                / pos_count[valid_anchors]
            )
            loss_qsupcon = -mean_log_prob_pos.mean()
        else:
            loss_qsupcon = q_sim.sum() * 0.0

        total_loss = loss_rs_cos + self.lambda_q * loss_qsupcon
        if return_components:
            return total_loss, {
                "loss_rs_cos": loss_rs_cos,
                "loss_qsupcon": loss_qsupcon,
            }
        return total_loss


class SelectiveQuantumOrthLoss(nn.Module):
    """
    Module 3.2 (Q-Orth): Projector-Mediated Selective Quantum-Cosine Orthogonality Loss.
    Operates through the aligned projector space A(z^old) and A(mu_c) to avoid
    direct gradient tug-of-war on the shared adapter while prioritizing the Top-K
    highest-risk colliding old prototypes in multi-subspace Quantum Hilbert space.
    """

    def __init__(
        self,
        topk: int = 5,
        margin_eps: float = 0.15,
        temperature: float = 0.1,
        rho_sel: float = 0.5,
        lambda_q_orth: float = 0.5,
    ):
        super().__init__()
        self.topk = topk
        self.margin_eps = margin_eps
        self.temperature = temperature
        self.rho_sel = rho_sel
        self.lambda_q_orth = lambda_q_orth

    def forward(
        self,
        z_aligned_old: torch.Tensor,
        z_cur: torch.Tensor,
        old_protos_aligned: torch.Tensor,
        simulator: QuantumStateSimulator,
        psi_old_aligned: torch.Tensor = None,
        psi_protos: torch.Tensor = None,
        return_components: bool = False,
    ):
        if old_protos_aligned.size(0) == 0 or z_aligned_old.size(0) == 0:
            zero_loss = z_aligned_old.sum() * 0.0
            if return_components:
                return zero_loss, {"loss_global": zero_loss, "loss_selective": zero_loss}
            return zero_loss

        u_norm = F.normalize(z_aligned_old, p=2, dim=1, eps=1e-8)
        p_norm = F.normalize(old_protos_aligned, p=2, dim=1, eps=1e-8)
        cos_sim_ic = torch.matmul(u_norm, p_norm.t())  # (B, C_old)
        loss_global = cos_sim_ic.mean()

        if psi_old_aligned is None:
            psi_old_aligned = simulator.encode(z_aligned_old)
        if psi_protos is None:
            psi_protos = simulator.encode(old_protos_aligned)

        q_sim_ic = simulator.overlap_from_states(psi_old_aligned, psi_protos)

        k = min(self.topk, q_sim_ic.size(1))
        q_topk, topk_idx = torch.topk(q_sim_ic, k=k, dim=1)
        cos_topk = torch.gather(cos_sim_ic, dim=1, index=topk_idx)

        weights = F.softmax(q_topk.detach() / self.temperature, dim=1)
        sel_penalty = (
            F.relu(cos_topk)
            + self.lambda_q_orth * F.relu(q_topk - self.margin_eps).square()
        )
        loss_selective = (weights * sel_penalty).sum(dim=1).mean()

        total_loss = (1.0 - self.rho_sel) * loss_global + self.rho_sel * loss_selective
        if return_components:
            return total_loss, {
                "loss_global": loss_global,
                "loss_selective": loss_selective,
            }
        return total_loss


class TopKQuantumRelLoss(nn.Module):
    """
    Module 3.3 (Top-K Q-Rel): Aligned Local Quantum Relational Knowledge Distillation Loss.
    Preserves local neighborhood manifold geometry in Quantum Hilbert space between
    current features z^cur and aligned historical features sg(A(z^old)).
    """

    def __init__(self, topk: int = 8, temperature: float = 0.15):
        super().__init__()
        self.topk = topk
        self.temperature = temperature

    def forward(
        self,
        z_cur: torch.Tensor,
        z_target: torch.Tensor,
        simulator: QuantumStateSimulator,
        psi_cur: torch.Tensor = None,
        psi_target: torch.Tensor = None,
        return_components: bool = False,
    ):
        batch_size = z_cur.size(0)
        if batch_size <= 1:
            zero_loss = z_cur.sum() * 0.0
            if return_components:
                return zero_loss, {"amp_loss": zero_loss, "kl_loss": zero_loss}
            return zero_loss

        with torch.no_grad():
            if psi_target is None:
                psi_target = simulator.encode(z_target)
            s_target = simulator.overlap_from_states(psi_target.detach())

        if psi_cur is None:
            psi_cur = simulator.encode(z_cur)
        s_cur = simulator.overlap_from_states(psi_cur)

        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=z_cur.device)
        s_target_masked = s_target.masked_fill(eye_mask, -1.0)

        k = min(self.topk, batch_size - 1)
        s_target_topk, topk_indices = torch.topk(s_target_masked, k=k, dim=1)
        s_cur_topk = torch.gather(s_cur, dim=1, index=topk_indices)

        p_target = F.softmax(s_target_topk / self.temperature, dim=1)
        log_p_target = F.log_softmax(s_target_topk / self.temperature, dim=1)
        log_p_cur = F.log_softmax(s_cur_topk / self.temperature, dim=1)

        smooth_diff = F.smooth_l1_loss(
            s_cur_topk, s_target_topk, reduction="none", beta=0.05
        )
        amp_loss = (p_target * smooth_diff).sum(dim=1).mean()
        kl_div = (p_target * (log_p_target - log_p_cur)).sum(dim=1).mean()
        kl_loss = (self.temperature ** 2) * kl_div

        total_loss = amp_loss + kl_loss
        if return_components:
            return total_loss, {"amp_loss": amp_loss, "kl_loss": kl_loss}
        return total_loss
