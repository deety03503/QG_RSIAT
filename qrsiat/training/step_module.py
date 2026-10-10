"""Single forward boundary for trainable network and optional QR losses."""

from __future__ import annotations

import logging
from typing import Any

from qrsiat.runtime.precision import full_precision_context


try:
    import torch
    from torch import nn
except ImportError as exc:
    raise RuntimeError("PyTorch is required for QR-RSIAT training") from exc


class StepModule(nn.Module):
    """Wrap all trainable components participating in a training forward."""

    def __init__(
        self,
        network: nn.Module,
        *,
        rs_loss: nn.Module | None = None,
        old_network: nn.Module | None = None,
        aligner: nn.Module | None = None,
        relational_kernel: nn.Module | None = None,
        relational_kernel_teacher: nn.Module | None = None,
        orth_kernel: nn.Module | None = None,
        old_projector: nn.Module | None = None,
        mode: str = "rae",
        pair_gather: bool = False,
        use_compile: bool = False,
        frozen_cast_controller: Any | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"rae", "qhybrid"}:
            raise ValueError(f"Unsupported aligner mode: {mode!r}")
        self.network = network
        self.rs_loss = rs_loss
        self.old_network = old_network
        self.aligner = aligner
        self.relational_kernel = relational_kernel
        self.relational_kernel_teacher = relational_kernel_teacher
        self.orth_kernel = orth_kernel
        self.old_projector = old_projector
        self.mode = mode
        self.pair_gather = pair_gather
        from qrsiat.runtime.optimize import make_compiled_callable

        compiled_extract = make_compiled_callable(
            network.extract_vector,
            enabled=use_compile,
        )
        self._extract_features = compiled_extract
        if frozen_cast_controller is not None:
            def safe_extract(images: Any) -> Any:
                try:
                    return compiled_extract(images)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as exc:
                    if not frozen_cast_controller.restore():
                        raise
                    logging.getLogger(__name__).warning(
                        "Frozen-parameter cast failed during forward; restored original "
                        "dtype and retrying eagerly: %s",
                        exc,
                    )
                    return network.extract_vector(images)

            self._extract_features = safe_extract
        self._classify_features = make_compiled_callable(
            network.fc,
            enabled=use_compile,
        )
        if self.old_network is not None:
            self.old_network.requires_grad_(False)
            self.old_network.eval()
        if self.relational_kernel_teacher is not None:
            self.relational_kernel_teacher.requires_grad_(False)
            self.relational_kernel_teacher.eval()

    def train(self, mode: bool = True) -> "StepModule":
        super().train(mode)
        if self.old_network is not None:
            self.old_network.eval()
        if self.relational_kernel_teacher is not None:
            self.relational_kernel_teacher.eval()
        return self

    def forward(
        self,
        images: Any,
        labels: Any,
        *,
        class_start: int = 0,
        classification_loss: Any | None = None,
        old_features: Any | None = None,
        old_prototypes: Any | None = None,
        lambda_rs: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        top_k: int = 3,
        temperature: float = 0.1,
        orth_epsilon: float = 0.2,
        lambda_qrel: float = 1.0,
    ) -> dict[str, Any]:
        features = self._extract_features(images)
        output = self._classify_features(features)
        logits = output["logits"]
        selected_logits = logits[:, class_start:]
        selected_labels = labels - class_start
        if selected_logits.shape[1] < 1:
            raise ValueError("classification slice contains no output classes")
        if classification_loss is None:
            loss_cls = nn.functional.cross_entropy(selected_logits, selected_labels)
        else:
            loss_cls = classification_loss(selected_logits, selected_labels)

        zero = logits.sum() * 0.0
        for component in (
            self.rs_loss,
            self.aligner,
            self.relational_kernel,
            self.orth_kernel,
            self.old_projector,
        ):
            if component is not None:
                for parameter in component.parameters():
                    if parameter.requires_grad:
                        zero = zero + parameter.sum() * 0.0
        total = loss_cls
        loss_rs = zero
        loss_rel = zero
        loss_align = zero
        loss_orth = zero
        if lambda_rs and self.rs_loss is None:
            raise RuntimeError("lambda_rs is non-zero but StepModule has no RS loss")
        if self.rs_loss is not None and lambda_rs:
            rs_features, rs_labels = self._gather_pairs(features, labels)
            if getattr(self.rs_loss, "requires_full_precision", False):
                with full_precision_context(features.device):
                    loss_rs = float(lambda_rs) * self.rs_loss(rs_features.float(), rs_labels)
            else:
                loss_rs = float(lambda_rs) * self.rs_loss(rs_features, rs_labels)

        if self.old_network is not None and (beta or gamma or self.mode == "rae"):
            if old_features is None:
                with torch.inference_mode():
                    old_features = self.old_network.extract_vector(images)
            old_features = old_features.detach().clone().float()
            if self.mode == "rae":
                if self.old_projector is None or old_prototypes is None:
                    raise RuntimeError("RAE alignment requires a projector and saved prototypes")
                projected_old = self.old_projector(old_features)
                loss_align = nn.functional.mse_loss(features, projected_old)
                projected_prototypes = self.old_projector(old_prototypes.detach().to(features))
                normalized_features = nn.functional.normalize(projected_old, dim=1)
                normalized_prototypes = nn.functional.normalize(projected_prototypes, dim=1)
                loss_orth = (normalized_prototypes @ normalized_features.T).mean()
                loss_orth = float(gamma) * loss_orth
                total = total + float(beta) * loss_align + loss_orth
            else:
                if self.aligner is None:
                    raise RuntimeError("qhybrid mode requires an aligner")
                with full_precision_context(features.device):
                    projected_old = self.aligner(old_features)
                    loss_align = nn.functional.mse_loss(features.float(), projected_old.float())
                    if beta and lambda_qrel:
                        if self.relational_kernel is None:
                            raise RuntimeError(
                                "lambda_qrel is non-zero but no relational kernel is configured"
                            )
                        if self.relational_kernel_teacher is None:
                            raise RuntimeError(
                                "lambda_qrel requires a frozen relational-kernel teacher"
                            )
                        current_global, _ = self._gather_pairs(features, labels)
                        previous_global, _ = self._gather_pairs(old_features, labels)
                        current_kernel = self.relational_kernel(current_global.float())
                        with torch.no_grad():
                            previous_kernel = self.relational_kernel_teacher(
                                previous_global.float()
                            )
                        loss_rel = float(beta) * float(lambda_qrel) * (
                            (current_kernel - previous_kernel.detach()).square().mean()
                        )
                    if gamma and old_prototypes is None:
                        raise RuntimeError(
                            "Incremental orthogonality requires saved old prototypes"
                        )
                    if gamma:
                        if self.orth_kernel is not None:
                            centering_mean = self.aligner.center_ema.current(old_features)
                            aligned_prototypes = self.aligner(
                                old_prototypes.detach().float(),
                                centering_mean=centering_mean,
                            )
                            similarity = self.orth_kernel(
                                features.float(), aligned_prototypes.float()
                            )
                            if similarity.shape[1] == 0:
                                loss_orth = similarity.sum() * 0.0
                            else:
                                k = min(int(top_k), similarity.shape[1])
                                if k < 1 or temperature <= 0:
                                    raise ValueError("top_k and temperature must be positive")
                                values = similarity.topk(k, dim=1).values
                                weights = torch.softmax(values / temperature, dim=1)
                                loss_orth = float(gamma) * (
                                    weights * nn.functional.relu(values - orth_epsilon)
                                ).sum(dim=1).mean()
                        else:
                            centering_mean = self.aligner.center_ema.current(old_features)
                            projected_prototypes = self.aligner(
                                old_prototypes.detach().float(),
                                centering_mean=centering_mean,
                            )
                            normalized_features = nn.functional.normalize(projected_old, dim=1)
                            normalized_prototypes = nn.functional.normalize(projected_prototypes, dim=1)
                            loss_orth = float(gamma) * (
                                normalized_prototypes @ normalized_features.T
                            ).mean()
                total = total + float(beta) * loss_align + loss_orth
        elif any((beta, gamma)):
            if self.aligner is None:
                raise RuntimeError(
                    "Incremental QR losses require an aligner and frozen old network"
                )
        total = total + loss_rs + loss_rel
        return {
            "logits": logits,
            "features": features,
            "loss": total,
            "loss_cls": loss_cls,
            "loss_rs": loss_rs,
            "loss_rel": loss_rel,
            "loss_align": loss_align,
            "loss_orth": loss_orth,
        }

    def _gather_pairs(self, features: Any, labels: Any) -> tuple[Any, Any]:
        if not self.pair_gather:
            return features, labels
        from qrsiat.distributed.collectives import gather_cat

        try:
            import torch.distributed as dist
        except ImportError:
            return features, labels
        if not dist.is_available() or not dist.is_initialized():
            return features, labels
        return (
            gather_cat(features, with_grad=features.requires_grad),
            gather_cat(labels, with_grad=False),
        )
