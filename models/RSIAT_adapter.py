import copy
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import count_parameters, log_count_parameter, target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid
from qrsiat.quantum.aligner import QHybridAligner
from qrsiat.quantum.kernels import PairwiseKernel
from qrsiat.training.step_module import StepModule
from qrsiat.training.schedule import warmup_value
from qrsiat.training.diagnostics import gradient_diagnostics, kernel_diagnostics
from torch.utils.data.distributed import DistributedSampler
from qrsiat.distributed.samplers import DistributedEvalSampler
from qrsiat.data.loaders import create_data_loader
from qrsiat.integration.learner_mixin import QRsiatLearnerMixin
from qrsiat.stats.drift import estimate_drift
from qrsiat.hardware.probe import probe_batch_size
from qrsiat.runtime.optimize import cast_frozen_parameters, enable_gradient_checkpointing
from qrsiat.utils.io import write_json_atomic
import math
num_workers = 8

class Learner(QRsiatLearnerMixin, BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = SimpleVitNet(args, True)
        self._network.to(self._device)
        plan = args.get("runtime_plan")
        self._frozen_cast_controller = None
        if plan is not None and args.get("freeze_cast", False) and plan.amp_dtype in {
            "bf16", "fp16"
        }:
            cast_dtype = torch.bfloat16 if plan.amp_dtype == "bf16" else torch.float16
            self._frozen_cast_controller = cast_frozen_parameters(
                self._network.convnet,
                cast_dtype,
                enabled=True,
            )
        enable_gradient_checkpointing(
            self._network.convnet,
            enabled=bool(args.get("grad_checkpointing", False)),
        )
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None
        self.aligner_mode = args.get("aligner", "rae")
        self.qhybrid = None
        self.relational_kernel = None
        self.relational_kernel_teacher = None
        self.orth_kernel = None
        self.rs_kernel = None
        if self.aligner_mode == "qhybrid":
            self.qhybrid = QHybridAligner(
                n_qubits=args.get("n_qubits", 8),
                layers=args.get("quantum_layers", 2),
                centering=args.get("quantum_centering", True),
                cycle=args.get("cycle", False),
            )
        if args.get("lambda_qrel", 0.0) > 0:
            self.relational_kernel = PairwiseKernel(
                args.get("kernel", "cosine"),
                n_qubits=args.get("n_qubits", 8),
                layers=args.get("quantum_layers", 2),
                centering=args.get("quantum_centering", True),
                cycle=args.get("cycle", False),
            )
        if args.get("orth", "plain") == "qweighted":
            self.orth_kernel = PairwiseKernel(
                "quantum",
                n_qubits=args.get("n_qubits", 8),
                layers=args.get("quantum_layers", 2),
                centering=args.get("quantum_centering", True),
                cycle=args.get("cycle", False),
            )
        if args.get("rs_kernel", "cosine") == "quantum":
            self.rs_kernel = PairwiseKernel(
                "quantum",
                n_qubits=args.get("n_qubits", 8),
                layers=args.get("quantum_layers", 2),
                centering=args.get("quantum_centering", True),
                cycle=args.get("cycle", False),
            )
        self.rs_loss_func = RS_Loss(
            self.args["alpha"],
            self.args["rs_margin"],
            kernel=self.rs_kernel,
        )
        self.loss_cos = AngularPenaltySMLoss(
            loss_type="cosface",
            eps=1e-7,
            s=self.args["scale"],
            m=self.args["margin"],
        )
        self.old_ae = None
        self._step_module = None
        self._training_module = None
        self._batch_probe_completed = False
        self._diagnostic_batch = None

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network


    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                if self.args.get("smoke", False) and i >= 2:
                    break
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        if not embedding_list:
            return (
                torch.empty((0, self.feature_dim), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1
        
        if self._cur_task == 1 and self.aligner_mode == "rae":
            self.old_ae = AutoencoderSigmoid(input_dims=768, code_dims=self.args["ae_code_dims"])
            self.old_ae.to(self._device)
        if self._cur_task > 0 and self.aligner_mode == "qhybrid":
            # Each task aligns against a different frozen previous network, so
            # start the complete aligner afresh from its identity initialization.
            self.qhybrid = QHybridAligner(
                n_qubits=self.args.get("n_qubits", 8),
                layers=self.args.get("quantum_layers", 2),
                centering=self.args.get("quantum_centering", True),
                cycle=self.args.get("cycle", False),
            ).to(self._device)
        elif self.qhybrid is not None:
            self.qhybrid.to(self._device)
        if self.relational_kernel is not None:
            self.relational_kernel.to(self._device)
            if self._cur_task > 0:
                # Freeze the previous task's kernel as a stable target for this task.
                self.relational_kernel_teacher = copy.deepcopy(self.relational_kernel)
                self.relational_kernel_teacher.to(self._device)
                self.relational_kernel_teacher.requires_grad_(False)
                self.relational_kernel_teacher.eval()
        if self.orth_kernel is not None:
            self.orth_kernel.to(self._device)
        if self.rs_kernel is not None:
            self.rs_kernel.to(self._device)
            
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(data_manager.get_task_size(self._cur_task)*4)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        self._network_module_ptr = self._network
        self._build_step_module()
        self._maybe_probe_batch_size()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.task_sizes = list(data_manager._increments)
        if self.args.get("diagnostics", False):
            self._prepare_diagnostic_batch(data_manager)
        sampler = None
        eval_sampler = None
        context = self.args.get("runtime_context")
        plan = self.args.get("runtime_plan")
        batch_size = (
            plan.per_device_batch
            if plan is not None and context is not None and context.world_size > 1
            else self.batch_size
        )
        if context is not None and context.world_size > 1:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                drop_last=False,
            )
        self.train_loader = (
            create_data_loader(
                train_dataset,
                plan,
                training=True,
                batch_size=batch_size,
                sampler=sampler,
                seed=self.args.get("seed", 1993) + (context.rank if context else 0),
            )
            if plan is not None
            else DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = (
            create_data_loader(
                test_dataset,
                plan,
                training=False,
                batch_size=self.batch_size,
            )
            if plan is not None
            else DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        )

        feature_loader = None
        if self._cur_task >0:
            self._network.to(self._device)
            feature_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="test",
            )
            feature_sampler = (
                DistributedEvalSampler(
                    feature_dataset,
                    num_replicas=context.world_size,
                    rank=context.rank,
                )
                if context is not None and context.world_size > 1
                else None
            )
            feature_loader = (
                create_data_loader(
                    feature_dataset,
                    plan,
                    training=False,
                    batch_size=self.batch_size,
                    sampler=feature_sampler,
                )
                if plan is not None
                else DataLoader(
                    feature_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=8,
                )
            )
            train_embeddings_old, _ = self.extract_features(
                feature_loader, self.old_network_module_ptr, None
            )

        self._train(self.train_loader, self.test_loader)

      
        if self._cur_task >0:
            train_embeddings_new, _ = self.extract_features(
                feature_loader, self._network, None
            )
            old_class_mean = self._class_means[:self._known_classes]
            context = self.args.get("runtime_context")
            gap = estimate_drift(
                train_embeddings_old.to(self._device),
                train_embeddings_new.to(self._device),
                torch.as_tensor(old_class_mean, dtype=torch.float64, device=self._device),
                sigma=4.0,
                distributed=context is not None and context.world_size > 1,
            ).cpu().numpy()
            if self.args['ssca'] is True:
                old_class_mean +=gap
                self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        task_size = data_manager.get_task_size(self._cur_task)

        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            if not self.args.get("smoke", False):
                self._stage2_compact_classifier(task_size, self.args['ca_epochs'])

    def _build_step_module(self):
        old_network = self.old_network_module_ptr if self._cur_task > 0 else None
        self._training_module = self.build_step_module(
            self._network,
            rs_loss=self.rs_loss_func if self._cur_task == 0 else None,
            old_network=old_network,
            aligner=self.qhybrid,
            relational_kernel=self.relational_kernel,
            relational_kernel_teacher=self.relational_kernel_teacher,
            orth_kernel=self.orth_kernel,
            old_projector=self.old_ae,
            mode=self.aligner_mode,
            device=self._device,
            runtime_context=self.args.get("runtime_context"),
            pair_gather=bool(self.args.get("pair_gather", False)),
            use_compile=bool(self.args.get("use_compile", False)),
            frozen_cast_controller=self._frozen_cast_controller,
        )
        self._step_module = (
            self._training_module.module
            if hasattr(self._training_module, "module")
            else self._training_module
        )

    def _maybe_probe_batch_size(self):
        if not self.args.get("probe", False) or self._batch_probe_completed:
            return
        self._batch_probe_completed = True
        context = self.args.get("runtime_context")
        if context is None or context.device.type != "cuda":
            return
        if context.world_size > 1:
            logging.warning("Skipping per-process OOM probe under DDP to avoid collective stalls.")
            return
        requested = int(self.batch_size)
        candidates = sorted(
            {
                1,
                max(1, requested // 4),
                max(1, requested // 2),
                requested,
            }
        )

        def run_step(size):
            self._training_module.zero_grad(set_to_none=True)
            try:
                images = torch.zeros(size, 3, 224, 224, device=self._device)
                labels = torch.full(
                    (size,),
                    self._known_classes,
                    dtype=torch.long,
                    device=self._device,
                )
                kwargs = {
                    "class_start": self._known_classes,
                    "classification_loss": self.loss_cos,
                    "lambda_rs": self.args.get("lambda_rs", 0.0)
                    if self._cur_task == 0
                    else 0.0,
                    "beta": self.args.get("beta", 0.0)
                    if self._cur_task > 0
                    else 0.0,
                    "gamma": self.args.get("gamma", 0.0)
                    if self._cur_task > 0
                    else 0.0,
                    "lambda_qrel": self.args.get("lambda_qrel", 0.0),
                }
                if self._cur_task > 0:
                    kwargs["old_prototypes"] = torch.as_tensor(
                        self._class_means[: self._known_classes],
                        dtype=torch.float32,
                        device=self._device,
                    )
                with context.autocast():
                    output = self._training_module(images, labels, **kwargs)
                output["loss"].backward()
            finally:
                self._training_module.zero_grad(set_to_none=True)

        chosen = probe_batch_size(
            run_step,
            candidates,
            fallback_batch=requested,
        )
        self.batch_size = chosen
        plan = self.args.get("runtime_plan")
        if plan is not None:
            updated_plan = replace(
                plan,
                per_device_batch=chosen,
                global_batch=chosen,
                grad_accum_steps=1,
            )
            self.args["runtime_plan"] = updated_plan
            if context.is_main:
                output_dir = Path(self.args.get("output_dir", "./out"))
                write_json_atomic(output_dir / "plan.json", updated_plan.to_dict())
        logging.info("CUDA batch probe selected batch_size=%d (configured=%d)", chosen, requested)

    def _prepare_diagnostic_batch(self, data_manager):
        """Cache one deterministic, non-augmented training batch for diagnostics."""
        context = self.args.get("runtime_context")
        if context is not None and not context.is_main:
            return
        dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        batch_size = max(1, min(int(self.args.get("batch_size", 48)), 64))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        try:
            _, images, labels = next(iter(loader))
        except StopIteration as exc:
            raise RuntimeError("Cannot build diagnostics batch from an empty task") from exc
        self._diagnostic_batch = (images.detach().cpu(), labels.detach().cpu())

    @staticmethod
    def _module_parameter_norm(parameter):
        return float(parameter.detach().float().norm().cpu())

    def _feature_map_diagnostics(self, name, quantum_module, features):
        if quantum_module is None:
            return {}
        feature_map = getattr(quantum_module, "quantum", quantum_module)
        project = getattr(feature_map, "project", None)
        if project is None:
            project = getattr(feature_map, "down", None)
        if project is None:
            return {}
        values = features.detach().float()
        if getattr(feature_map, "centering", False) and values.shape[0] > 1:
            values = values - values.mean(dim=0, keepdim=True)
        with torch.no_grad():
            projected = project(values).float()
            result = {
                f"{name}_sigma_u": float(projected.std(unbiased=False).cpu()),
                f"{name}_tanh_saturation_fraction": float(
                    (projected.tanh().abs() > 0.99).float().mean().cpu()
                ),
                f"{name}_W_q_or_down_frobenius": self._module_parameter_norm(project.weight),
            }
            theta = getattr(feature_map, "theta", None)
            if theta is not None:
                result[f"{name}_theta_norm"] = self._module_parameter_norm(theta)
        return result

    def _run_diagnostics(self, phase, epoch, warmup_epoch):
        """Write one JSONL record; this path is opt-in and never steps the optimizer."""
        if not self.args.get("diagnostics", False) or self._diagnostic_batch is None:
            return
        context = self.args.get("runtime_context")
        if context is not None and not context.is_main:
            return
        images, labels = self._diagnostic_batch
        images = images.to(self._device)
        labels = labels.to(self._device)
        module_was_training = self._step_module.training
        self._step_module.eval()
        try:
            old_features = None
            if self._cur_task > 0 and self.old_network_module_ptr is not None:
                with torch.inference_mode():
                    old_features = self.old_network_module_ptr.extract_vector(images).float()
            _, _, _, output, components = self._compute_rt_loss(
                images,
                labels,
                epoch,
                warmup_epoch,
                module=self._step_module,
                pair_gather_override=False,
                old_features=old_features,
                capture_output=True,
            )
            adapter_parameters = [
                parameter
                for name, parameter in self._network.convnet.named_parameters()
                if parameter.requires_grad and "adapter" in name.lower()
            ]
            if not adapter_parameters:
                adapter_parameters = [
                    parameter
                    for parameter in self._network.convnet.parameters()
                    if parameter.requires_grad
                ]
            gradient_values = {"loss_cls": output["loss_cls"], **components}
            record = {
                "task": int(self._cur_task),
                "phase": phase,
                "epoch": int(epoch + 1),
                "batch_samples": int(labels.numel()),
                "loss_components": {
                    name: float(value.detach().float().cpu())
                    for name, value in gradient_values.items()
                },
                "gradient_diagnostics": gradient_diagnostics(
                    gradient_values, "loss_cls", adapter_parameters
                ),
            }

            features = output["features"].detach().float()
            centered = features - features.mean(dim=0, keepdim=True)
            normalized = F.normalize(features, dim=1)
            normalized_centered = normalized - normalized.mean(dim=0, keepdim=True)
            record["feature_rms_by_dimension"] = float(
                centered.square().mean().sqrt().cpu()
            )
            record["normalized_feature_rms_by_dimension"] = float(
                normalized_centered.square().mean().sqrt().cpu()
            )

            with torch.no_grad():
                rs_kernel = self.rs_loss_func.kernel
                rs_matrix = (
                    rs_kernel(normalized)
                    if rs_kernel is not None
                    else normalized @ normalized.T
                )
                record.update(kernel_diagnostics(rs_matrix, labels, prefix="rs"))
                if self.relational_kernel is not None:
                    record.update(
                        kernel_diagnostics(
                            self.relational_kernel(features), labels, prefix="qrel"
                        )
                    )
                record.update(
                    self._feature_map_diagnostics("rs", rs_kernel, normalized)
                )
                record.update(
                    self._feature_map_diagnostics(
                        "qrel", self.relational_kernel, features
                    )
                )
                record.update(
                    self._feature_map_diagnostics("qorth", self.orth_kernel, features)
                )
                if self.qhybrid is not None:
                    record.update(
                        self._feature_map_diagnostics("aligner", self.qhybrid, features)
                    )
                    record["aligner_W_up_frobenius"] = self._module_parameter_norm(
                        self.qhybrid.up.weight
                    )

                if old_features is not None:
                    record["feature_drift_relative_norm"] = float(
                        ((features - old_features).norm() / old_features.norm().clamp_min(1e-12)).cpu()
                    )
                    prototypes = torch.as_tensor(
                        self._class_means[: self._known_classes],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    if self.qhybrid is not None and prototypes.numel():
                        aligned_prototypes = self.qhybrid(
                            prototypes,
                            centering_mean=old_features.mean(dim=0, keepdim=True),
                        )
                        record["aligned_prototype_mean_distance"] = float(
                            (aligned_prototypes - prototypes).norm(dim=1).mean().cpu()
                        )
                    if self.orth_kernel is not None and prototypes.numel():
                        aligned_prototypes = self.qhybrid(
                            prototypes,
                            centering_mean=old_features.mean(dim=0, keepdim=True),
                        )
                        orth_values = self.orth_kernel(features, aligned_prototypes)
                        k = min(int(self.args.get("orth_top_k", 3)), orth_values.shape[1])
                        if k > 0:
                            top_values = orth_values.topk(k, dim=1).values
                            qs = torch.tensor(
                                [0.05, 0.5, 0.95, 0.99], device=top_values.device
                            )
                            record["qorth_G_topk_q05_q50_q95_q99"] = [
                                float(value)
                                for value in torch.quantile(top_values.flatten(), qs).cpu()
                            ]
                            epsilon = float(self.args.get("orth_epsilon", 0.2))
                            record["qorth_P_G_top1_gt_epsilon"] = float(
                                (top_values[:, 0] > epsilon).float().mean().cpu()
                            )
            path = (
                Path(self.args.get("output_dir", "./out"))
                / self.args["dataset"]
                / f"seed_{self.args.get('seed', 1993)}"
                / "diagnostics.jsonl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            logging.info(
                "Diagnostics written: task=%d phase=%s path=%s",
                self._cur_task,
                phase,
                path,
            )
        finally:
            self._step_module.train(module_was_training)

    def _save_task_checkpoint(self):
        manager = self.args.get("checkpoint_manager")
        if manager is None:
            return
        context = self.args.get("runtime_context")
        if context is not None and not context.is_main:
            return
        modules = {"network": self._network}
        if self.qhybrid is not None:
            modules["qhybrid"] = self.qhybrid
        if self.relational_kernel is not None:
            modules["relational_kernel"] = self.relational_kernel
        if self.orth_kernel is not None:
            modules["orth_kernel"] = self.orth_kernel
        if self.rs_kernel is not None:
            modules["rs_kernel"] = self.rs_kernel
        if self.old_ae is not None:
            modules["old_ae"] = self.old_ae
        manager.save(
            self._cur_task,
            modules,
            self.args.get("checkpoint_config", {}),
            metadata={
                "known_classes": self._known_classes,
                "total_classes": self._total_classes,
                "class_means": self._class_means.tolist(),
                "class_covariances": self._class_covs.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous(),
                "task_sizes": list(self.data_manager._increments),
                "class_order": list(self.data_manager._class_order),
                "accuracy_curve": list(self.args.get("accuracy_curve", [])),
                "metric_curve": self.args.get("metric_curve", {}),
            },
            rank=context.rank if context is not None else 0,
        )

    def resume_from_checkpoint(self, data_manager):
        manager = self.args.get("checkpoint_manager")
        if manager is None or self.args.get("resume", "auto") == "none":
            return 0
        config = self.args.get("checkpoint_config", {})
        payload = manager.inspect(
            config,
            resume_from=self.args.get("resume_from"),
            force=self.args.get("force_resume", False),
            map_location="cpu",
        )
        if payload is None:
            logging.info("No task checkpoint found; starting a fresh run.")
            return 0
        task = payload.get("task")
        metadata = payload.get("metadata")
        if not isinstance(task, int) or not isinstance(metadata, dict):
            raise ValueError("Checkpoint is missing task index or task metadata")
        increments = list(data_manager._increments)
        if task < 0 or task >= len(increments):
            raise ValueError(f"Checkpoint task {task} is outside the current task schedule")
        saved_sizes = metadata.get("task_sizes")
        # Checkpoints store the complete dataset task schedule, not only the
        # tasks completed at the time the checkpoint was written.
        if saved_sizes != increments:
            raise ValueError("Checkpoint task schedule differs from the current dataset split")
        expected_classes = sum(increments[: task + 1])
        if metadata.get("total_classes") != expected_classes:
            raise ValueError("Checkpoint class count does not match the current task schedule")
        current_order = list(data_manager._class_order)
        if metadata.get("class_order") != current_order:
            raise ValueError("Checkpoint class order differs from the current dataset protocol")

        for completed_task in range(task + 1):
            self._network.update_fc(increments[completed_task])
        self._cur_task = task
        self._known_classes = expected_classes
        self._total_classes = expected_classes
        if task > 0 and self.aligner_mode == "rae":
            self.old_ae = AutoencoderSigmoid(
                input_dims=768, code_dims=self.args["ae_code_dims"]
            ).to(self._device)
        modules = {"network": self._network}
        for name in ("qhybrid", "relational_kernel", "orth_kernel", "rs_kernel", "old_ae"):
            module = getattr(self, name, None)
            if module is not None:
                modules[name] = module
        manager.restore_modules(payload, modules, restore_rng=True)
        self._class_means = np.asarray(metadata["class_means"], dtype=np.float64)
        self._class_covs = torch.as_tensor(
            metadata["class_covariances"],
            dtype=torch.float32,
            device="cpu",
        )
        if self._class_means.shape != (expected_classes, self.feature_dim):
            raise ValueError("Checkpoint prototype array has an incompatible shape")
        expected_covariance_shape = (
            expected_classes,
            self.feature_dim,
            self.feature_dim,
        )
        if tuple(self._class_covs.shape) != expected_covariance_shape:
            raise ValueError("Checkpoint covariance array has an incompatible shape")
        if not np.isfinite(self._class_means).all() or not torch.isfinite(
            self._class_covs
        ).all():
            raise ValueError("Checkpoint class statistics contain non-finite values")
        self._network.to(self._device)
        self._old_network = self._network.copy().freeze()
        self.old_network_module_ptr = self._old_network
        accuracy_curve = metadata.get("accuracy_curve", [])
        if not isinstance(accuracy_curve, list) or len(accuracy_curve) != task + 1:
            raise ValueError("Checkpoint accuracy history is incomplete or malformed")
        self.args["accuracy_curve"] = accuracy_curve
        metric_curve = metadata.get("metric_curve")
        if not isinstance(metric_curve, dict):
            raise ValueError("Checkpoint metric history is missing or malformed")
        self.args["metric_curve"] = metric_curve
        logging.info("Resumed from completed task %d (%d classes).", task, expected_classes)
        return task + 1

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._training_module is None:
            raise RuntimeError("Training StepModule was not built")
        context = self.args.get("runtime_context")
        lr_scale = (
            context.world_size
            if self.args.get("lr_scale_by_world", False)
            and context is not None
            and context.world_size > 1
            else 1
        )
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            if self.args.get("smoke", False):
                self.tuned_epochs = max(1, min(2, self.tuned_epochs))
            param_groups = [
                {'params': self._network.convnet.blocks[-1].parameters(), 'lr': 0.01 * lr_scale,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.convnet.blocks[:-1].parameters(), 'lr': 0.01 * lr_scale,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.fc.parameters(), 'lr': 0.01 * lr_scale, 'weight_decay': self.args['weight_decay']}
            ]
            self._append_step_parameters(param_groups, lr_scale=lr_scale)

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr * lr_scale, weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(
                    self._trainable_step_parameters(),
                    lr=self.init_lr * lr_scale,
                    weight_decay=self.weight_decay,
                )
            else:
                raise ValueError(f"Unsupported optimizer {self.args['optimizer']!r}")
                
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr * lr_scale)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])
        else:
            self.tuned_epochs = self.args['inc_epochs']
            if self.args.get("smoke", False):
                self.tuned_epochs = max(1, min(2, self.tuned_epochs))
            param_groups = []
            param_groups.append(
                {'params': self._network.convnet.parameters(), 'lr': self.init_lr * lr_scale, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'params': self._network.fc.parameters(), 'lr': self.init_lr * lr_scale, 'weight_decay': self.weight_decay})
            if self.old_ae is not None:
                param_groups.append(
                    {'params': self.old_ae.parameters(), 'lr': self.args['ae_init_lr'] * lr_scale,
                     'weight_decay': self.args['ae_weight_decay']})
            self._append_step_parameters(param_groups, lr_scale=lr_scale)
            
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(
                    self._trainable_step_parameters(),
                    lr=self.init_lr * lr_scale,
                    weight_decay=self.weight_decay,
                )
            else:
                raise ValueError(f"Unsupported optimizer {self.args['optimizer']!r}")

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr * lr_scale)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])

    def _trainable_step_parameters(self):
        if self._step_module is None:
            raise RuntimeError("Training StepModule was not built")
        return [parameter for parameter in self._step_module.parameters() if parameter.requires_grad]

    def _append_step_parameters(self, groups, *, lr_scale=1):
        # Some legacy groups provide one-shot parameter generators. Materialize
        # them before checking identities so the optimizer still receives them.
        for group in groups:
            group["params"] = list(group["params"])
        existing = {
            id(parameter)
            for group in groups
            for parameter in group["params"]
        }
        extras = [
            parameter for parameter in self._trainable_step_parameters()
            if id(parameter) not in existing
        ]
        if extras:
            groups.append(
                {
                    "params": extras,
                    "lr": self.init_lr * lr_scale,
                    "weight_decay": self.weight_decay,
                }
            )

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(range(self.tuned_epochs))
        context = self.args.get("runtime_context")
        plan = self.args.get("runtime_plan")
        grad_accum_steps = (
            plan.grad_accum_steps
            if plan is not None and context is not None and context.world_size > 1
            else 1
        )
        if self.args.get("smoke", False):
            grad_accum_steps = 1
        
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self._training_module.train()
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            # float64 mirrors Python's old per-step ``.item()`` accumulation
            # while keeping the reduction on the device until epoch end.
            loss_sums = torch.zeros(8, device=self._device, dtype=torch.float64)
            total = 0
            optimizer.zero_grad()
            step_count = 0

            if self.args.get("diagnostics", False) and epoch == 0:
                self._run_diagnostics("epoch_start", epoch, warmup_epoch)

            training_started = time.perf_counter()
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args.get("smoke", False) and i >= 2:
                    break
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                scaler = context.scaler if context is not None else None
                remaining = len(train_loader) - (i // grad_accum_steps) * grad_accum_steps
                divisor = min(grad_accum_steps, remaining)
                if scaler is None:
                    (loss / divisor).backward()
                else:
                    scaler.scale(loss / divisor).backward()
                should_step = (i + 1) % grad_accum_steps == 0 or i + 1 == len(train_loader)
                if should_step:
                    optimizer_parameters = [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ]
                    parameters_with_grad = [
                        parameter
                        for parameter in optimizer_parameters
                        if parameter.grad is not None
                    ]
                    if not parameters_with_grad:
                        raise RuntimeError(
                            "Training produced no gradients for any optimizer parameter "
                            f"(task={self._cur_task}, epoch={epoch}, batch={i}, "
                            f"optimizer={type(optimizer).__name__}, "
                            f"loss_requires_grad={loss.requires_grad}). "
                            "Check that the trainable StepModule parameters are connected "
                            "to the selected loss."
                        )
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad()
                _, preds = torch.max(logits, dim=1)
                correct = preds.eq(targets.expand_as(preds)).sum()
                loss_sums += torch.stack(
                    [
                        loss.detach().to(torch.float64),
                        loss_c.detach().to(torch.float64),
                        loss_rt.detach().to(torch.float64),
                        *self._last_loss_components.to(torch.float64),
                        correct.detach().to(torch.float64),
                    ]
                )
                total += len(targets)
                step_count += 1
            if step_count == 0:
                raise RuntimeError("Training loader yielded no batches")
            train_speed = total / max(time.perf_counter() - training_started, 1e-9)
            scheduler.step()

            # One device-to-host transfer per epoch replaces the three scalar
            # loss reads and the per-step accuracy transfer in the hot loop.
            (loss_sum, loss_c_sum, loss_rt_sum, rs_sum, rel_sum, align_sum,
             orth_sum, correct_sum) = loss_sums.detach().cpu().tolist()
            train_acc = np.around(correct_sum * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                loss_sum / step_count,
                loss_c_sum / step_count,
                loss_rt_sum / step_count,
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
            prog_bar.set_postfix(speed=f"{train_speed:.1f} samples/s")
            logging.info(info)
            logging.info(
                "Losses_rt_components => RS %.6f, qrel %.6f, align %.6f, orth %.6f",
                rs_sum / step_count,
                rel_sum / step_count,
                align_sum / step_count,
                orth_sum / step_count,
            )
            if self.args.get("diagnostics", False) and epoch + 1 == self.tuned_epochs:
                self._run_diagnostics("epoch_end", epoch, warmup_epoch)

    def _inc_loss(self, features, features_old):
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = torch.from_numpy(self._class_means).float().to(self._device,non_blocking=True)
        protos = self.old_ae(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0]*similarity.shape[1])
        return self.args["beta"] * loss_align + self.args["gamma"] * loss_orth
        
    def _compute_rt_loss(
        self,
        inputs,
        targets,
        epoch=None,
        warmup_epoch=10,
        *,
        module=None,
        pair_gather_override=None,
        old_features=None,
        capture_output=False,
    ):
        if self._training_module is None:
            raise RuntimeError("Training StepModule was not initialized for this task")
        if self._cur_task == 0:
            lambda_rs = self.args["lambda_rs"] * min(
                1.0, epoch / max(1, warmup_epoch)
            )
            beta = gamma = 0.0
        elif self.aligner_mode == "qhybrid":
            beta = warmup_value(self.args.get("beta", 0.0), epoch, warmup_epoch)
            gamma = warmup_value(self.args.get("gamma", 0.0), epoch, warmup_epoch)
            lambda_rs = 0.0
        else:
            beta = float(self.args.get("beta", 0.0))
            gamma = float(self.args.get("gamma", 0.0))
            lambda_rs = 0.0
        prototypes = None
        if self._cur_task > 0:
            prototypes = torch.as_tensor(
                self._class_means[: self._known_classes],
                dtype=torch.float32,
                device=self._device,
            )
        context = self.args.get("runtime_context")
        autocast = context.autocast() if context is not None else torch.autocast("cpu", enabled=False)
        active_module = self._training_module if module is None else module
        original_pair_gather = None
        if pair_gather_override is not None and hasattr(active_module, "pair_gather"):
            original_pair_gather = active_module.pair_gather
            active_module.pair_gather = pair_gather_override
        try:
            with autocast:
                output = active_module(
                inputs,
                targets,
                class_start=self._known_classes,
                classification_loss=self.loss_cos,
                old_features=old_features,
                old_prototypes=prototypes,
                lambda_rs=lambda_rs,
                beta=beta,
                gamma=gamma,
                lambda_qrel=self.args.get("lambda_qrel", 0.0),
                top_k=self.args.get("orth_top_k", 3),
                temperature=self.args.get("orth_temperature", 0.1),
                orth_epsilon=self.args.get("orth_epsilon", 0.2),
            )
        finally:
            if original_pair_gather is not None:
                active_module.pair_gather = original_pair_gather
        loss_components = {
            "loss_rs": output["loss_rs"],
            "loss_rel": output["loss_rel"],
            "loss_align": output["loss_align"] * beta,
            "loss_orth": output["loss_orth"],
        }
        loss_rt = sum(loss_components.values())
        self._last_loss_components = torch.stack(
            [value.detach().float() for value in loss_components.values()]
        )
        result = (output["logits"], output["loss_cls"], loss_rt)
        if capture_output:
            return (*result, output, loss_components)
        return result
    
class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5, kernel=None):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin
        self.kernel = kernel
        self.requires_full_precision = kernel is not None

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        dot_prod = (
            self.kernel(features.float())
            if self.kernel is not None
            else torch.matmul(features, features.t())
        )

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - self.margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)

        return loss
