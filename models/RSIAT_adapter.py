import copy
import logging
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
from quantum.aligner import QHybridAligner
from quantum.kernels import PairwiseKernel
from quantum.losses import qorth_kernel_loss, qrel_loss
import math
num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None

        # QR-RSIAT configuration flags (defaults preserve 100% original RSIAT baseline)
        self.aligner_type = str(args.get("aligner", "rae")).lower()
        self.lambda_qrel = float(args.get("lambda_qrel", 0.0))
        self.kernel_type = str(args.get("kernel", "cosine")).lower()
        self.orth_type = str(args.get("orth", "plain")).lower()
        self.rs_kernel_type = str(args.get("rs_kernel", "cosine")).lower()
        self.n_qubits = int(args.get("n_qubits", 8))
        self.quantum_layers = int(args.get("quantum_layers", 2))
        self.quantum_centering = bool(args.get("quantum_centering", True))
        self.cycle = bool(args.get("cycle", False))
        self.orth_top_k = int(args.get("orth_top_k", 3))
        self.orth_temperature = float(args.get("orth_temperature", 0.1))
        self.orth_epsilon = float(args.get("orth_epsilon", 0.2))

        if self.rs_kernel_type == "quantum":
            self.rs_kernel = PairwiseKernel(
                name="quantum",
                input_dim=768,
                n_qubits=self.n_qubits,
                layers=self.quantum_layers,
                centering=self.quantum_centering,
                cycle=self.cycle,
            ).to(self._device)
            self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"], kernel=self.rs_kernel)
        else:
            self.rs_kernel = None
            self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"], kernel=None)

        self.old_ae = None
        self.qhybrid = None
        self.relational_kernel = None
        self.relational_kernel_teacher = None
        self.orth_kernel = None
        self._last_loss_stats = {
            "loss_rs": 0.0,
            "loss_align": 0.0,
            "loss_qrel": 0.0,
            "loss_orth": 0.0,
        }

    def _is_baseline_incremental(self) -> bool:
        return (
            self.aligner_type == "rae"
            and self.lambda_qrel == 0.0
            and self.orth_type == "plain"
        )

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def _prepare_incremental_modules(self):
        if self.aligner_type == "rae":
            if self.old_ae is None:
                self.old_ae = AutoencoderSigmoid(input_dims=768, code_dims=self.args["ae_code_dims"])
            self.old_ae.to(self._device)
        elif self.aligner_type == "qhybrid":
            self.qhybrid = QHybridAligner(
                input_dim=768,
                n_qubits=self.n_qubits,
                layers=self.quantum_layers,
                centering=self.quantum_centering,
                cycle=self.cycle,
            ).to(self._device)

        if self.lambda_qrel > 0.0:
            if self.relational_kernel is None:
                if self.rs_kernel is not None and self.kernel_type == "quantum":
                    self.relational_kernel = copy.deepcopy(self.rs_kernel).to(self._device)
                else:
                    self.relational_kernel = PairwiseKernel(
                        name=self.kernel_type,
                        input_dim=768,
                        n_qubits=self.n_qubits,
                        layers=self.quantum_layers,
                        centering=self.quantum_centering,
                        cycle=self.cycle,
                    ).to(self._device)
            else:
                self.relational_kernel.to(self._device)
            for p in self.relational_kernel.parameters():
                p.requires_grad_(True)

            self.relational_kernel_teacher = copy.deepcopy(self.relational_kernel).to(self._device)
            self.relational_kernel_teacher.eval()
            for p in self.relational_kernel_teacher.parameters():
                p.requires_grad_(False)

        if self.orth_type == "qweighted":
            if self.orth_kernel is None:
                self.orth_kernel = PairwiseKernel(
                    name="quantum",
                    input_dim=768,
                    n_qubits=self.n_qubits,
                    layers=self.quantum_layers,
                    centering=self.quantum_centering,
                    cycle=self.cycle,
                ).to(self._device)
            else:
                self.orth_kernel.to(self._device)
            for p in self.orth_kernel.parameters():
                p.requires_grad_(True)

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task == 1 and self.aligner_type == "rae":
            self.old_ae = AutoencoderSigmoid(input_dims=768, code_dims=self.args["ae_code_dims"])
            self.old_ae.to(self._device)

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(data_manager.get_task_size(self._cur_task)*4)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        self._network_module_ptr = self._network
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task > 0:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._network, None)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._cur_task > 0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network, None)
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
            if self.args['ssca'] is True:
                old_class_mean += gap
                self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        task_size = data_manager.get_task_size(self._cur_task)

        if self._cur_task > 0 and self.args['ca_epochs'] > 0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': self._network_module_ptr.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network_module_ptr.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network_module_ptr.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
            ]
            if self.rs_kernel_type == "quantum" and self.rs_kernel is not None:
                self.rs_kernel.to(self._device)
                rs_params = [p for p in self.rs_kernel.parameters() if p.requires_grad]
                if rs_params:
                    param_groups.append(
                        {'params': rs_params, 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
                    )

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(param_groups, lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])
        else:
            self.tuned_epochs = self.args['inc_epochs']
            self._prepare_incremental_modules()
            aux_lr = self.args.get('ae_init_lr', self.init_lr)
            aux_wd = self.args.get('ae_weight_decay', self.weight_decay)

            param_groups = []
            param_groups.append(
                {'params': self._network_module_ptr.convnet.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'params': self._network_module_ptr.fc.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})

            if self.aligner_type == "rae":
                param_groups.append(
                    {'params': self.old_ae.parameters(), 'lr': aux_lr, 'weight_decay': aux_wd})
            elif self.aligner_type == "qhybrid":
                param_groups.append(
                    {'params': self.qhybrid.parameters(), 'lr': aux_lr, 'weight_decay': aux_wd})

            if self.lambda_qrel > 0.0 and self.relational_kernel is not None:
                rel_params = [p for p in self.relational_kernel.parameters() if p.requires_grad]
                if rel_params:
                    param_groups.append(
                        {'params': rel_params, 'lr': aux_lr, 'weight_decay': aux_wd})

            if self.orth_type == "qweighted" and self.orth_kernel is not None:
                orth_params = [p for p in self.orth_kernel.parameters() if p.requires_grad]
                if orth_params:
                    param_groups.append(
                        {'params': orth_params, 'lr': aux_lr, 'weight_decay': aux_wd})

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(param_groups, lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])

    def _set_aux_train_mode(self):
        if self._cur_task == 0:
            if self.rs_kernel is not None:
                self.rs_kernel.train()
        else:
            if self.aligner_type == "rae" and self.old_ae is not None:
                self.old_ae.train()
            elif self.aligner_type == "qhybrid" and self.qhybrid is not None:
                self.qhybrid.train()
            if self.lambda_qrel > 0.0 and self.relational_kernel is not None:
                self.relational_kernel.train()
            if self.relational_kernel_teacher is not None:
                self.relational_kernel_teacher.eval()
            if self.orth_type == "qweighted" and self.orth_kernel is not None:
                self.orth_kernel.train()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(range(self.tuned_epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self._set_aux_train_mode()
            losses = 0.0
            losses_c, losses_rt = 0.0, 0.0
            losses_rs, losses_align, losses_qrel, losses_orth = 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                losses_rs += self._last_loss_stats["loss_rs"]
                losses_align += self._last_loss_stats["loss_align"]
                losses_qrel += self._last_loss_stats["loss_qrel"]
                losses_orth += self._last_loss_stats["loss_orth"]
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            num_batches = max(1, len(train_loader))
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, "
                "Loss_rs {:.3f}, Loss_align {:.3f}, Loss_qrel {:.3f}, Loss_orth {:.3f}, "
                "Train_accy {:.2f}, Test_accy {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / num_batches,
                losses_c / num_batches,
                losses_rt / num_batches,
                losses_rs / num_batches,
                losses_align / num_batches,
                losses_qrel / num_batches,
                losses_orth / num_batches,
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _inc_loss(self, features, features_old):
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = torch.from_numpy(self._class_means[:self._known_classes]).float().to(self._device, non_blocking=True)
        protos = self.old_ae(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0] * similarity.shape[1])
        self._last_loss_stats = {
            "loss_rs": 0.0,
            "loss_align": float(loss_align.detach().item()),
            "loss_qrel": 0.0,
            "loss_orth": float(loss_orth.detach().item()),
        }
        return self.args["beta"] * loss_align + self.args["gamma"] * loss_orth

    def _qr_inc_loss(self, features, features_old, epoch=None, warmup_epoch=10):
        features_old_detached = features_old.detach()

        # 1. Feature Alignment Loss
        if self.aligner_type == "qhybrid":
            aligned_old = self.qhybrid(features_old_detached)
        else:
            aligned_old = self.old_ae(features_old_detached)
        loss_align = nn.MSELoss()(features, aligned_old)

        # 2. Quantum Relational Distillation Loss
        if (
            self.lambda_qrel > 0.0
            and self.relational_kernel is not None
            and self.relational_kernel_teacher is not None
        ):
            k_cur = self.relational_kernel(features)
            with torch.no_grad():
                k_old = self.relational_kernel_teacher(features_old_detached)
            loss_qrel = qrel_loss(k_cur, k_old)
        else:
            loss_qrel = features.new_zeros(())

        # 3. Orthogonality Loss (plain or qweighted)
        protos = torch.from_numpy(self._class_means[:self._known_classes]).float().to(self._device, non_blocking=True)
        if self.orth_type == "qweighted":
            with torch.no_grad():
                if self.aligner_type == "qhybrid":
                    old_mean = features_old_detached.float().mean(dim=0, keepdim=True)
                    aligned_protos = self.qhybrid(protos, centering_mean=old_mean)
                else:
                    aligned_protos = self.old_ae(protos)
            loss_orth = qorth_kernel_loss(
                features=features,
                aligned_protos=aligned_protos,
                kernel=self.orth_kernel,
                top_k=self.orth_top_k,
                temperature=self.orth_temperature,
                epsilon=self.orth_epsilon,
            )
        else:
            if self.aligner_type == "qhybrid":
                old_mean = features_old_detached.float().mean(dim=0, keepdim=True).detach()
                aligned_protos = self.qhybrid(protos, centering_mean=old_mean)
            else:
                aligned_protos = self.old_ae(protos)
            aligned_protos_norm = F.normalize(aligned_protos, p=2, dim=1)
            aligned_old_norm = F.normalize(aligned_old, p=2, dim=1)
            similarity = torch.matmul(aligned_protos_norm, aligned_old_norm.t())
            loss_orth = similarity.sum() / (similarity.shape[0] * similarity.shape[1])

        # 4. Linear Warm-up & Total Incremental Loss
        if epoch is None:
            warmup_scale = 1.0
        else:
            warmup_scale = min(1.0, float(epoch) / float(max(1, warmup_epoch)))

        beta_e = self.args["beta"] * warmup_scale
        gamma_e = self.args["gamma"] * warmup_scale

        self._last_loss_stats = {
            "loss_rs": 0.0,
            "loss_align": float(loss_align.detach().item()),
            "loss_qrel": float(loss_qrel.detach().item()),
            "loss_orth": float(loss_orth.detach().item()),
        }

        return (
            beta_e * loss_align
            + beta_e * self.lambda_qrel * loss_qrel
            + gamma_e * loss_orth
        )

    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

        if self._cur_task == 0:
            lambda_rs = self.args["lambda_rs"] * min(1.0, epoch / max(1, warmup_epoch))
            rs_val = self.rs_loss_func(features, targets)
            loss_base = lambda_rs * rs_val
            self._last_loss_stats = {
                "loss_rs": float(loss_base.detach().item()),
                "loss_align": 0.0,
                "loss_qrel": 0.0,
                "loss_orth": 0.0,
            }
            return logits, loss_c, loss_base

        features_old = self.old_network_module_ptr.extract_vector(inputs).detach()
        if self._is_baseline_incremental():
            loss_inc = self._inc_loss(features, features_old)
        else:
            loss_inc = self._qr_inc_loss(features, features_old, epoch=epoch, warmup_epoch=warmup_epoch)
        return logits, loss_c, loss_inc


class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5, kernel=None):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin
        self.kernel = kernel

    def forward(self, features, labels):
        device = features.device
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        if self.kernel is None:
            features = F.normalize(features, p=2, dim=1)
            dot_prod = torch.matmul(features, features.t())
        else:
            dot_prod = self.kernel(features.float())

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - self.margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)

        return loss
