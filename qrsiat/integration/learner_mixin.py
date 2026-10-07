"""Minimal bridge from the RSIAT learner to RuntimeContext and StepModule."""

from __future__ import annotations

from typing import Any


class QRsiatLearnerMixin:
    def build_step_module(
        self,
        network: Any,
        *,
        rs_loss: Any | None,
        old_network: Any | None,
        aligner: Any | None,
        relational_kernel: Any | None,
        relational_kernel_teacher: Any | None,
        orth_kernel: Any | None,
        old_projector: Any | None,
        mode: str,
        device: Any,
        runtime_context: Any | None,
        pair_gather: bool = False,
        use_compile: bool = False,
        frozen_cast_controller: Any | None = None,
    ) -> Any:
        from torch import nn
        from qrsiat.training.step_module import StepModule

        module = StepModule(
            network,
            rs_loss=rs_loss,
            old_network=old_network,
            aligner=aligner,
            relational_kernel=relational_kernel,
            relational_kernel_teacher=relational_kernel_teacher,
            orth_kernel=orth_kernel,
            old_projector=old_projector,
            mode=mode,
            pair_gather=pair_gather,
            use_compile=use_compile,
            frozen_cast_controller=frozen_cast_controller,
        ).to(device)
        if runtime_context is None or runtime_context.world_size == 1:
            return module
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("A multi-rank RuntimeContext requires an initialized process group")
        return nn.parallel.DistributedDataParallel(
            module,
            device_ids=[runtime_context.local_rank],
            output_device=runtime_context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
