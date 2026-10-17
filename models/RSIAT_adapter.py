import copy
import json
import logging
import time
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
from qrsiat.quantum.feature_map import calibrate_bandwidth
from qrsiat.quantum.diagnostics import kernel_diagnostics
from qrsiat.runtime.precision import full_precision_context
from qrsiat.training.step_module import StepModule
from qrsiat.training.diagnostics import (
    accumulate_float64,
    diagnostics_due,
    rt_loss_components,
)
from qrsiat.training.schedule import warmup_value
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
        kernel_name = args.get("kernel", "cosine")
        rs_kernel_name = args.get("rs_kernel") or kernel_name
        if rs_kernel_name != kernel_name:
            raise ValueError(
                "The shared kernel requires 'kernel' and 'rs_kernel' to match"
            )
        if args.get("orth", "plain") == "qweighted" and kernel_name != "quantum":
            raise ValueError("--orth qweighted requires the shared kernel to be quantum")
        self.kernel = PairwiseKernel(
            kernel_name,
            n_qubits=args.get("n_qubits", 8),
            layers=args.get("quantum_layers", 2),
            centering=args.get("quantum_centering", True),
            cycle=args.get("cycle", False),
            input_norm=args.get("input_norm", "none"),
        )
        if self.aligner_mode == "qhybrid":
            self.qhybrid = QHybridAligner(
                n_qubits=args.get("n_qubits", 8),
                layers=args.get("quantum_layers", 2),
                centering=args.get("quantum_centering", True),
                cycle=args.get("cycle", False),
            )
        self.rs_loss_func = RS_Loss(
            self.args["alpha"],
            self.args["rs_margin"],
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
        self._last_components = {}

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
            # Each task has a different teacher, so restart the near-identity aligner.
            self.qhybrid = QHybridAligner(
                n_qubits=self.args.get("n_qubits", 8),
                layers=self.args.get("quantum_layers", 2),
                centering=self.args.get("quantum_centering", True),
                cycle=self.args.get("cycle", False),
            ).to(self._device)
        elif self.qhybrid is not None:
            self.qhybrid.to(self._device)
        self.kernel.to(self._device)
        if self._cur_task > 0:
            self.kernel.requires_grad_(False)
            self.kernel.eval()
            
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

        if self.args.get("diagnostics", False) and self._is_diagnostics_rank():
            diagnostics_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="test",
            )
            diagnostics_loader = DataLoader(
                diagnostics_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                generator=torch.Generator().manual_seed(
                    int(self.args.get("seed", 1993)) + self._cur_task
                ),
            )
            _, diagnostic_images, diagnostic_labels = next(iter(diagnostics_loader))
            self._diagnostic_batch = (
                diagnostic_images.to(self._device),
                diagnostic_labels.to(self._device),
            )
            reference_network = (
                self.old_network_module_ptr
                if self._cur_task > 0 else self._network
            )
            self._diagnostic_reference_features = self._features_for_diagnostics(
                self._diagnostic_batch[0], reference_network
            )
        else:
            self._diagnostic_batch = None
            self._diagnostic_reference_features = None

        self._maybe_calibrate_kernel(data_manager)

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
            kernel=self.kernel,
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
        modules["kernel"] = self.kernel
        if self.old_ae is not None:
            modules["old_ae"] = self.old_ae
        manager.save(
            self._cur_task,
            modules,
            self.args.get("checkpoint_config", {}),
            full_state_modules={"kernel"},
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
        saved_modules = payload.get("modules")
        if isinstance(saved_modules, dict) and "kernel" not in saved_modules:
            legacy_kernel = saved_modules.get("rs_kernel")
            if legacy_kernel is None and any(
                parameter.requires_grad for parameter in self.kernel.parameters()
            ):
                raise ValueError(
                    "This legacy checkpoint has no base-task RS kernel to restore. "
                    "Start a new run to train the shared kernel on the base task."
                )
            if legacy_kernel is not None:
                expected_state = self.kernel.state_dict()
                unexpected_keys = set(legacy_kernel) - set(expected_state)
                incompatible_shapes = [
                    name
                    for name, value in legacy_kernel.items()
                    if name in expected_state
                    and tuple(value.shape) != tuple(expected_state[name].shape)
                ]
                if unexpected_keys or incompatible_shapes:
                    raise ValueError(
                        "The legacy RS kernel architecture does not match the "
                        "configured shared kernel; start a new run."
                    )
            payload = dict(payload)
            payload["modules"] = {
                name: state
                for name, state in saved_modules.items()
                if name not in {"relational_kernel", "orth_kernel", "rs_kernel"}
            }
            payload["modules"]["kernel"] = legacy_kernel or {}
            logging.info("Migrating the legacy base-task RS kernel to the shared kernel.")
        for name in ("qhybrid", "kernel", "old_ae"):
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
            epoch_started = time.perf_counter()
            self._network.train()
            self._training_module.train()
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            losses = torch.zeros((), dtype=torch.float64, device=self._device)
            losses_c = torch.zeros_like(losses)
            losses_rt = torch.zeros_like(losses)
            component_sums = {
                name: torch.zeros_like(losses)
                for name in ("loss_rs", "loss_rel", "loss_align", "loss_orth")
            }
            correct = torch.zeros((), dtype=torch.long, device=self._device)
            total = 0
            optimizer.zero_grad()
            step_count = 0

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
                accumulate_float64(losses, loss)
                accumulate_float64(losses_c, loss_c)
                accumulate_float64(losses_rt, loss_rt)
                for name in component_sums:
                    accumulate_float64(component_sums[name], self._last_components[name])
                _, preds = torch.max(logits, dim=1)
                correct.add_(preds.eq(targets.expand_as(preds)).sum())
                total += len(targets)
                step_count += 1
            if step_count == 0:
                raise RuntimeError("Training loader yielded no batches")
            epoch_values = torch.stack(
                (losses, losses_c, losses_rt, *component_sums.values(), correct.to(torch.float64))
            ).cpu().tolist()
            train_speed = total / max(time.perf_counter() - training_started, 1e-9)
            scheduler.step()

            losses_value, losses_c_value, losses_rt_value = epoch_values[:3]
            component_values = dict(zip(component_sums, epoch_values[3:7]))
            correct_value = int(epoch_values[7])
            train_acc = np.around(correct_value * 100 / total, decimals=2)

            diagnostic_elapsed = 0.0
            diagnostic_record = None
            if self._should_write_diagnostics(epoch):
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                diagnostic_started = time.perf_counter()
                diagnostic_record = self._collect_diagnostics(epoch, warmup_epoch)
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                diagnostic_elapsed = time.perf_counter() - diagnostic_started
                diagnostic_record["diagnostics_seconds"] = diagnostic_elapsed
            test_acc = self._compute_accuracy(self._network, test_loader)
            if diagnostic_record is not None:
                diagnostic_record["diagnostics_epoch_fraction"] = diagnostic_elapsed / max(
                    time.perf_counter() - epoch_started, 1e-9
                )
                self._append_diagnostic_record(diagnostic_record)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses_value / step_count,
                losses_c_value / step_count,
                losses_rt_value / step_count,
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
            prog_bar.set_postfix(speed=f"{train_speed:.1f} samples/s")
            if component_values:
                component_info = "Task {}, Epoch {}/{} => Loss_rs {:.6g}, Loss_rel {:.6g}, Loss_align {:.6g}, Loss_orth {:.6g}".format(
                    self._cur_task,
                    epoch + 1,
                    self.tuned_epochs,
                    *(component_values[name] / step_count for name in component_sums),
                )
                logging.info(component_info)
        logging.info(info)

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
        
    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):
        output, beta = self._forward_training_step(
            inputs, targets, epoch, warmup_epoch, self._training_module
        )
        loss_rt, self._last_components = rt_loss_components(output, beta)
        return output["logits"], output["loss_cls"], loss_rt

    def _forward_training_step(self, inputs, targets, epoch, warmup_epoch, module):
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
        with autocast:
            output = module(
                inputs,
                targets,
                class_start=self._known_classes,
                classification_loss=self.loss_cos,
                old_prototypes=prototypes,
                lambda_rs=lambda_rs,
                beta=beta,
                gamma=gamma,
                lambda_qrel=self.args.get("lambda_qrel", 0.0),
                top_k=self.args.get("orth_top_k", 3),
                temperature=self.args.get("orth_temperature", 0.1),
                orth_epsilon=self.args.get("orth_epsilon", 0.2),
            )
        return output, beta

    def _is_diagnostics_rank(self):
        context = self.args.get("runtime_context")
        return context is None or context.is_main

    def _should_write_diagnostics(self, epoch):
        return diagnostics_due(
            self.args.get("diagnostics", False),
            self._is_diagnostics_rank(),
            epoch,
            self.tuned_epochs,
        )

    def _maybe_calibrate_kernel(self, data_manager):
        if not self.args.get("kernel_calibration", False):
            return
        if self.kernel is None or self.kernel.quantum is None:
            return
        context = self.args.get("runtime_context")
        if context is not None and not context.is_main:
            return
        calib_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(
                int(self.args.get("seed", 1993)) + self._cur_task
            ),
        )
        all_features = []
        all_labels = []
        network = self.old_network_module_ptr if self._cur_task > 0 else self._network
        was_training = network.training
        network.eval()
        try:
            with torch.no_grad(), full_precision_context(self._device):
                for step, (_, images, labels) in enumerate(calib_loader):
                    if step >= 2:
                        break
                    images = images.to(self._device)
                    feats = network.extract_vector(images)
                    all_features.append(feats.float())
                    all_labels.append(labels.to(self._device))
        finally:
            network.train(was_training)
        if not all_features:
            return
        feats = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        target = float(self.args.get("kernel_target_median", 0.1))
        best_gamma = calibrate_bandwidth(
            self.kernel.quantum, feats, labels=labels, target=target
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(self.kernel.quantum.bandwidth, src=0)
        from qrsiat.quantum.simulator import fidelity_kernel
        with torch.no_grad(), full_precision_context(self._device):
            state, _ = self.kernel.quantum(feats)
            k = fidelity_kernel(state)
            n = feats.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=feats.device)
            diff_mask = mask & (labels[:, None] != labels[None, :])
            if bool(diff_mask.any()):
                mask = diff_mask
            med_f = k[mask].median().item() if bool(mask.any()) else 0.0
        logging.info(
            "Task {} Kernel Bandwidth Calibration: gamma={:.6g}, median_F={:.4f} (target={:.4f})".format(
                self._cur_task, best_gamma, med_f, target
            )
        )

    def _features_for_diagnostics(self, images, network=None):
        network = self._network if network is None else network
        was_training = network.training
        network.eval()
        context = self.args.get("runtime_context")
        autocast = context.autocast() if context is not None else torch.autocast("cpu", enabled=False)
        try:
            with torch.no_grad(), autocast:
                features = network.extract_vector(images).detach().float()
        finally:
            network.train(was_training)
        return features

    def _collect_diagnostics(self, epoch, warmup_epoch):
        images, labels = self._diagnostic_batch
        was_training = self._network.training
        self._network.eval()
        self._step_module.eval()
        try:
            output, beta = self._forward_training_step(
                images, labels, epoch, warmup_epoch, self._step_module
            )
            features = output["features"].float()
            centered = features - features.mean(dim=0, keepdim=True)
            normalized = F.normalize(features, dim=1)
            normalized_centered = normalized - normalized.mean(dim=0, keepdim=True)
            result = {
                "task": int(self._cur_task),
                "epoch": int(epoch + 1),
                "feature_rms_raw_centered": centered.square().mean().sqrt().item(),
                "feature_rms_normalized_centered": normalized_centered.square().mean().sqrt().item(),
            }
            result.update(self._quantum_input_stats(features, normalized))

            rs_features = features if getattr(self.kernel, "input_norm", "none") != "none" else normalized
            rs_matrix = self.kernel(rs_features)
            qrel_matrix = self.kernel(features)
            result["kernel_rs"] = kernel_diagnostics(rs_matrix, labels)
            result["kernel_qrel"] = kernel_diagnostics(qrel_matrix, labels)
            result.update(self._qorth_diagnostics(features))
            result.update(self._quantum_parameter_norms())
            result["bandwidth"] = (
                float(self.kernel.quantum.bandwidth.item())
                if self.kernel.quantum is not None
                else None
            )
            current_features = features.detach()
            reference = self._diagnostic_reference_features
            result["feature_drift_from_previous_task"] = (
                (current_features - reference).norm()
                / (reference.norm() + 1e-12)
            ).item()
            result["gradient_diagnostics"] = self._loss_gradient_diagnostics(
                output["loss_cls"],
                {
                    "loss_rs": output["loss_rs"],
                    "loss_rel": output["loss_rel"],
                    "loss_align": output["loss_align"] * beta,
                    "loss_orth": output["loss_orth"],
                },
            )
            return result
        finally:
            self._network.train(was_training)
            self._step_module.train(was_training)

    def _quantum_input_stats(self, features, normalized):
        stats = {}
        quantum = self.kernel.quantum
        if quantum is None:
            stats.update({
                "sigma_u_raw": None,
                "tanh_saturation_raw": None,
                "sigma_u_normalized": None,
                "tanh_saturation_normalized": None,
            })
            return stats
        for name, values in (("raw", features), ("normalized", normalized)):
            if quantum.centering:
                values = values - values.mean(dim=0, keepdim=True)
            with torch.no_grad():
                pre_tanh = quantum.project(values.float())
                stats[f"sigma_u_{name}"] = pre_tanh.std(unbiased=False).item()
                stats[f"tanh_saturation_{name}"] = (
                    pre_tanh.tanh().abs() > 0.99
                ).float().mean().item()
        return stats

    def _qorth_diagnostics(self, features):
        result = {
            "qorth_G_topk": [],
            "qorth_top1_gt_epsilon_probability": None,
            "aligned_prototype_distance": None,
        }
        if self._cur_task == 0 or self._class_means is None:
            return result
        prototypes = torch.as_tensor(
            self._class_means[: self._known_classes],
            dtype=torch.float32,
            device=features.device,
        )
        if self.aligner_mode == "qhybrid" and self.qhybrid is not None:
            with torch.no_grad():
                old_features = self._features_for_diagnostics(
                    self._diagnostic_batch[0], self.old_network_module_ptr
                )
                mean = old_features.mean(dim=0, keepdim=True)
                aligned = self.qhybrid(prototypes, centering_mean=mean)
            similarity = self.kernel(features, aligned)
        elif self.old_ae is not None:
            with torch.no_grad():
                old_features = self._features_for_diagnostics(
                    self._diagnostic_batch[0], self.old_network_module_ptr
                )
                projected_features = self.old_ae(old_features)
                aligned = self.old_ae(prototypes)
                similarity = (
                    F.normalize(aligned, dim=1)
                    @ F.normalize(projected_features, dim=1).transpose(0, 1)
                ).transpose(0, 1)
        else:
            return result
        if aligned.shape == prototypes.shape:
            result["aligned_prototype_distance"] = (
                (aligned - prototypes).norm(dim=1).mean().item()
            )
        if similarity.shape[1] == 0:
            return result
        k = min(int(self.args.get("orth_top_k", 3)), similarity.shape[1])
        values = similarity.topk(k, dim=1).values
        result["qorth_G_topk"] = values.detach().cpu().tolist()
        epsilon = float(self.args.get("orth_epsilon", 0.2))
        result["qorth_top1_gt_epsilon_probability"] = (
            (values[:, 0] > epsilon).float().mean().item()
        )
        return result

    def _quantum_parameter_norms(self):
        quantum = self.kernel.quantum
        aligner = self.qhybrid
        return {
            "W_q_fro": (
                quantum.project.weight.detach().float().norm().item()
                if quantum is not None else None
            ),
            "theta_norm": (
                quantum.theta.detach().float().norm().item()
                if quantum is not None else None
            ),
            "theta_aligner_norm": (
                aligner.theta.detach().float().norm().item()
                if aligner is not None else None
            ),
            "W_q_aligner_fro": (
                aligner.down.weight.detach().float().norm().item()
                if aligner is not None else None
            ),
            "W_up_fro": (
                aligner.up.weight.detach().float().norm().item()
                if aligner is not None else None
            ),
        }

    def _loss_gradient_diagnostics(self, classification_loss, components):
        parameters = [
            parameter for parameter in self._network.convnet.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            return {}
        losses = {"loss_cls": classification_loss, **components}
        gradients = {}
        for index, (name, loss) in enumerate(losses.items()):
            grads = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=index < len(losses) - 1,
                allow_unused=True,
            )
            gradients[name] = grads
        base = gradients["loss_cls"]
        base_norm = self._gradient_norm(base, parameters[0].device)
        result = {}
        for name, grads in gradients.items():
            norm = self._gradient_norm(grads, parameters[0].device)
            dot = sum(
                (left.float() * right.float()).sum()
                for left, right in zip(base, grads)
                if left is not None and right is not None
            )
            result[name] = {
                "norm": norm.item(),
                "norm_ratio_to_loss_cls": (norm / (base_norm + 1e-12)).item(),
                "cosine_to_loss_cls": (
                    (dot / (base_norm * norm + 1e-12)).item()
                    if name != "loss_cls" else 1.0
                ),
            }
        return result

    @staticmethod
    def _gradient_norm(gradients, device=None):
        squared = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
        if not squared:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(squared).sum().sqrt()

    def _append_diagnostic_record(self, record):
        output_dir = Path(self.args.get("output_dir", "./out"))
        seed = self.args.get("seed", 1993)
        if isinstance(seed, (list, tuple)):
            seed = seed[0]
        path = output_dir / self.args["dataset"] / f"seed_{seed}" / "diagnostics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    
class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5, kernel=None):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin
        self.kernel = kernel
        self.requires_full_precision = kernel is not None

    def forward(self, features, labels, *, kernel=None):
        device = features.device
        pairwise_kernel = self.kernel if kernel is None else kernel
        input_norm = getattr(pairwise_kernel, "input_norm", "none") if pairwise_kernel is not None else "none"
        if pairwise_kernel is not None and input_norm != "none":
            kernel_features = features.float()
        else:
            kernel_features = F.normalize(features, p=2, dim=1).float()
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        dot_prod = (
            pairwise_kernel(kernel_features)
            if pairwise_kernel is not None
            else torch.matmul(kernel_features, kernel_features.t())
        )

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - self.margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)

        return loss
