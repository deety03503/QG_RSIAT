import logging
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import log_count_parameter, seed_worker
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid
from quantum import QHybridAligner, QuantumFeatureMap, kernel_matrix, qorth_loss, qrel_loss

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

        self.logit_norm = None
        self.tuned_epochs = None
        self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"])
        self.old_ae = None
        self.quantum_feature_map = None
        self.quantum_aligner = None
        self._class_means_raw = None
        self._initialize_quantum_modules()

    def _initialize_quantum_modules(self):
        needs_quantum = (
            self.args.get("aligner", "rae") == "qhybrid"
            or (
                self.args.get("lambda_qrel", 0.0) > 0
                and self.args.get("kernel", "quantum") == "quantum"
            )
            or self.args.get("orth", "plain") == "qweighted"
            or self.args.get("rs_kernel", "cosine") == "quantum"
        )
        if not needs_quantum:
            return
        self.quantum_feature_map = QuantumFeatureMap(
            input_dim=self._network.feature_dim,
            n_qubits=self.args.get("qbits", 8),
            n_layers=self.args.get("q_layers", 2),
            center=self.args.get("center_features", True),
        ).to(self._device)
        if self.args.get("aligner", "rae") == "qhybrid":
            self.quantum_aligner = QHybridAligner(
                self.quantum_feature_map, feature_dim=self._network.feature_dim
            ).to(self._device)

    def after_task(self):
        self._known_classes = self._total_classes
        self._network = self._network_module()
        self._old_network = self._network.copy().freeze()
        self.old_network_module_ptr = self._old_network


    def extract_features(self, trainloader, model):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for batch in trainloader:
                (_, data, label) = batch
                data = data.to(self._device, non_blocking=True)
                outputs = model(data, return_features=True)
                embedding = outputs["features"]
                embedding_list.append(embedding.cpu())
                label_list.append(label)

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._network = self._network_module()
        self._cur_task += 1
        
        if self._cur_task == 1:
            if self.args.get("aligner", "rae") == "rae":
                self.old_ae = AutoencoderSigmoid(
                    input_dims=768, code_dims=self.args["ae_code_dims"]
                ).to(self._device)
            
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        train_generator = torch.Generator()
        train_generator.manual_seed(self.seed + self._cur_task)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            worker_init_fn=seed_worker,
            generator=train_generator,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self.num_worker > 0,
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        test_generator = torch.Generator()
        test_generator.manual_seed(self.seed + 10000 + self._cur_task)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            worker_init_fn=seed_worker,
            generator=test_generator,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self.num_worker > 0,
        )

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            device_ids = [device.index for device in self._multiple_gpus]
            self._network = nn.DataParallel(
                self._network, device_ids=device_ids, output_device=device_ids[0]
            )

      
        if self._cur_task >0:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(
                self.train_loader, self._network
            )

        self._train(self.train_loader, self.test_loader)
        
        self._network = self._network_module()

      
        if self._cur_task >0:
            train_embeddings_new, _ = self.extract_features(
                self.train_loader, self._network
            )
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
            if self.args['ssca'] is True:
                old_class_mean +=gap
                self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        task_size = data_manager.get_task_size(self._cur_task)

        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        network = self._network_module()
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': network.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': network.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
            ]
            if (
                self.quantum_feature_map is not None
                and self.args.get("rs_kernel", "cosine") == "quantum"
            ):
                param_groups.append({
                    'params': self.quantum_feature_map.parameters(),
                    'lr': self.init_lr,
                    'weight_decay': self.weight_decay,
                })

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                parameters = list(network.parameters())
                if (
                    self.quantum_feature_map is not None
                    and self.args.get("rs_kernel", "cosine") == "quantum"
                ):
                    parameters.extend(self.quantum_feature_map.parameters())
                optimizer = optim.AdamW(parameters, lr=self.init_lr, weight_decay=self.weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {self.args['optimizer']}")
                
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])
        else:
            self.tuned_epochs = self.args['inc_epochs']
            param_groups = []
            param_groups.append(
                {'params': network.convnet.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'params': network.fc.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            auxiliary_modules = []
            if self.old_ae is not None:
                auxiliary_modules.append(self.old_ae)
            if self.quantum_aligner is not None:
                auxiliary_modules.append(self.quantum_aligner)
            elif self.quantum_feature_map is not None:
                auxiliary_modules.append(self.quantum_feature_map)
            for module in auxiliary_modules:
                param_groups.append({
                    'params': module.parameters(),
                    'lr': self.args.get('ae_init_lr', self.init_lr),
                    'weight_decay': self.args.get('ae_weight_decay', self.weight_decay),
                })
            
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args['optimizer'] == 'adam':
                parameters = list(network.parameters())
                for module in auxiliary_modules:
                    parameters.extend(module.parameters())
                optimizer = optim.AdamW(
                    parameters, lr=self.init_lr, weight_decay=self.weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.args['optimizer']}")

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(
            range(self.tuned_epochs),
            desc=f"Task {self._cur_task + 1}",
            unit="epoch",
        )
        
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_c, losses_rt = 0.0, 0.0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                _, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                prog_bar.set_postfix(
                    batch=f"{i + 1}/{len(train_loader)}",
                    loss=f"{losses / (i + 1):.3f}",
                    refresh=False,
                )
            scheduler.step()

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                losses_c/len(train_loader),
                losses_rt/len(train_loader),
                test_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _alignment_module(self):
        return self.quantum_aligner if self.quantum_aligner is not None else self.old_ae

    def _warmup_weight(self, epoch):
        warmup_epochs = self.args["warmup_epoch"]
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, (epoch + 1) / warmup_epochs)

    def _inc_loss(self, features, features_old, epoch):
        aligner = self._alignment_module()
        if aligner is None:
            features_old_aligned = features_old
        else:
            features_old_aligned = aligner(features_old)
        loss_align = nn.MSELoss()(features, features_old_aligned)
        beta = self.args["beta"]
        gamma = self.args["gamma"]
        use_qr_warmup = (
            self.args.get("aligner", "rae") == "qhybrid"
            or self.args.get("lambda_qrel", 0.0) > 0
            or self.args.get("orth", "plain") == "qweighted"
        )
        warmup = self._warmup_weight(epoch) if use_qr_warmup else 1.0
        total_loss = beta * warmup * loss_align

        if self.args.get("lambda_qrel", 0.0) > 0:
            feature_map = (
                self.quantum_feature_map
                if self.args["kernel"] == "quantum"
                else None
            )
            loss_qrel = qrel_loss(
                features,
                features_old,
                kind=self.args["kernel"],
                feature_map=feature_map,
            )
            total_loss = total_loss + beta * warmup * self.args["lambda_qrel"] * loss_qrel

        if self.args.get("orth", "plain") == "qweighted":
            prototype_source = (
                self._class_means
                if self.args.get("orth_use_drift_comp", True)
                else self._class_means_raw
            )
            prototypes = torch.as_tensor(
                prototype_source[:self._known_classes],
                dtype=features.dtype,
                device=self._device,
            )
            if aligner is not None:
                prototypes = aligner(prototypes)
            loss_orth = qorth_loss(
                features,
                prototypes,
                feature_map=self.quantum_feature_map,
                top_k=self.args.get("orth_topk", 5),
                temperature=self.args.get("orth_tau", 0.1),
                epsilon=self.args.get("orth_epsilon", 0.5),
            )
            total_loss = total_loss + gamma * warmup * loss_orth
        else:
            features_old_norm = F.normalize(features_old_aligned, p=2, dim=1)
            prototypes = torch.as_tensor(
                self._class_means,
                dtype=features.dtype,
                device=self._device,
            )
            if aligner is not None:
                prototypes = aligner(prototypes)
            prototypes = F.normalize(prototypes, p=2, dim=1)
            similarity = torch.matmul(prototypes, features_old_norm.t())
            loss_orth = similarity.sum() / (
                similarity.shape[0] * similarity.shape[1]
            )
            total_loss = total_loss + gamma * loss_orth
        return total_loss
        
    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):
        loss_cos=AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        outputs = self._network(inputs, return_features=True)
        features = outputs["features"]
        logits = outputs["logits"]
        loss_c=loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

        if self._cur_task == 0:
            lambda_rs = self.args["lambda_rs"] * min(
                1.0, epoch / warmup_epoch
            ) if warmup_epoch > 0 else self.args["lambda_rs"]
            feature_map = (
                self.quantum_feature_map
                if self.args.get("rs_kernel", "cosine") == "quantum"
                else None
            )
            loss_base = lambda_rs * self.rs_loss_func(
                features, targets, feature_map=feature_map
            )
            return logits, loss_c, loss_base
        
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_inc = self._inc_loss(features, features_old, epoch)
        return logits, loss_c, loss_inc
    
class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin

    def forward(self, features, labels, feature_map=None):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        if feature_map is None:
            dot_prod = torch.matmul(features, features.t())
        else:
            fidelity = kernel_matrix(
                features, kind="quantum", feature_map=feature_map
            )
            dot_prod = 2.0 * fidelity - 1.0

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - self.margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)

        return loss