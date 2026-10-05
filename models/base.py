import copy
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import time
EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5
        runtime_context = args.get("runtime_context")
        self._device = runtime_context.device if runtime_context is not None else args["device"][0]
        self._multiple_gpus = []

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        return self._network.feature_dim


    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        context = getattr(self, "args", {}).get("runtime_context")
        distributed = context is not None and context.world_size > 1
        if distributed and context.rank != 0:
            torch.distributed.barrier()
            for tensor in self._network.fc.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)
            torch.distributed.barrier()
            return
        self._stage2_compact_classifier_local(task_size, ca_epochs)
        if distributed:
            torch.distributed.barrier()
            for tensor in self._network.fc.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)
            torch.distributed.barrier()

    def _stage2_compact_classifier_local(self, task_size, ca_epochs=5):
        for p in self._network.fc.parameters():
            p.requires_grad = True

        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.init_lr,
                           'weight_decay': self.weight_decay}]

        optimizer = optim.SGD(network_params, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)

        self._network.eval()

        for epoch in range(run_epochs):
            losses = 0.
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device) * (
                            0.9 + decay)

                cls_cov = self._class_covs[c_id].to(self._device)
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            inputs = sampled_data
            targets = sampled_label
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                # -stage two only use classifiers
                outputs = self._network.ca_forward(inp)
                logits = self.args['scale'] * outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task + 1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]

                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses / self._total_classes, test_acc)
            logging.info(info)


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        grouped = {k: float(v) for k, v in grouped.items()}
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        actual_k = min(self.topk, y_pred.shape[1])
        ret["top{}".format(actual_k)] = float(np.around(
            (y_pred.T == np.tile(y_true, (actual_k, 1))).sum() * 100 / len(y_true),
            decimals=2,
        ))

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            if getattr(self, "args", {}).get("smoke", False) and i >= 2:
                break
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        if total == 0:
            raise RuntimeError("Evaluation loader yielded no samples")
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for i, (_, inputs, targets) in enumerate(loader):
            if getattr(self, "args", {}).get("smoke", False) and i >= 2:
                break
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            actual_k = min(self.topk, int(outputs.shape[1]))
            if actual_k < 1:
                raise ValueError("Model returned no class logits")
            predicts = torch.topk(
                outputs, k=actual_k, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        if not y_pred:
            raise RuntimeError("Evaluation loader yielded no samples")
        return np.concatenate(y_pred), np.concatenate(y_true)


    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for index, (_, _inputs, _targets) in enumerate(loader):
            if getattr(self, "args", {}).get("smoke", False) and index >= 2:
                break
            _targets = _targets.numpy()
            _vectors = tensor2numpy(
                self._network.extract_vector(_inputs.to(self._device))
            )

            vectors.append(_vectors)
            targets.append(_targets)

        if not vectors:
            return (
                np.empty((0, self.feature_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        from qrsiat.distributed.samplers import DistributedEvalSampler
        from qrsiat.stats.accumulators import ClassStatisticsAccumulator

        context = getattr(self, "args", {}).get("runtime_context")
        plan = getattr(self, "args", {}).get("runtime_plan")
        new_class_count = self._total_classes - self._known_classes
        if new_class_count < 1:
            raise RuntimeError("No new classes are available for class-statistics extraction")
        accumulator = ClassStatisticsAccumulator(
            new_class_count,
            self.feature_dim,
            device=self._device,
        )
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            sampler = (
                DistributedEvalSampler(
                    idx_dataset,
                    num_replicas=context.world_size,
                    rank=context.rank,
                )
                if context is not None and context.world_size > 1
                else None
            )
            workers = plan.num_workers if plan is not None else 4
            loader_args = {
                "dataset": idx_dataset,
                "batch_size": batch_size,
                "shuffle": False,
                "sampler": sampler,
                "num_workers": workers,
            }
            if workers:
                loader_args["persistent_workers"] = bool(
                    plan.persistent_workers if plan is not None else False
                )
                loader_args["prefetch_factor"] = int(
                    plan.prefetch_factor if plan is not None and plan.prefetch_factor else 2
                )
            idx_loader = DataLoader(**loader_args)
            vectors, _ = self._extract_vectors(idx_loader)
            class_labels = torch.full(
                (len(vectors),),
                class_idx - self._known_classes,
                dtype=torch.long,
            )
            accumulator.update(
                torch.as_tensor(vectors, dtype=torch.float64),
                class_labels,
            )
        statistics = accumulator.finalize(
            covariance_epsilon=1e-3,
            distributed=context is not None and context.world_size > 1,
        )
        means = statistics.means.cpu().numpy()
        covariances = statistics.covariances.cpu()
        new_slice = slice(self._known_classes, self._total_classes)
        self._class_means[new_slice] = means
        self._class_covs[new_slice] = covariances
        if self._cur_task == 0:
            per_class_radius = [
                np.trace(covariances[index].numpy() + np.eye(self.feature_dim) * 1e-4)
                / self.feature_dim
                for index in range(new_class_count)
            ]
            self.radius = np.sqrt(np.mean(per_class_radius))
            print(self.radius)

    def displacement_cov(self, Y, class_mean, embedding_old, sigma):
        cov = None
        start_time = time.time()
        for _class in range(self._known_classes):
            loop_start_time = time.time()
            DY = self.cov_computation(Y, class_mean[_class])
            distance = np.sum((np.tile(Y[None, :, :], [1, 1, 1]) - np.tile(
                embedding_old[_class, None, :], [1, Y.shape[0], 1])) ** 2, axis=2)
            W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
            W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
            if cov is None:
                cov = np.sum(np.tile(W_norm[:, :, None, None], [
                    1, 1, DY.shape[1], DY.shape[2]]) * np.tile(DY[None, :, :, :], [W.shape[0], 1, 1, 1]), axis=1)
            else:
                displacement = np.sum(np.tile(W_norm[:, :, None, None], [
                    1, 1, DY.shape[1], DY.shape[2]]) * np.tile(DY[None, :, :, :], [W.shape[0], 1, 1, 1]), axis=1)
                cov = np.concatenate((cov, displacement))
            loop_end_time = time.time()
            print("single loop time: ", loop_end_time - loop_start_time)
        end_time = time.time()
        print("total loop time: ", end_time - start_time)

        cov = torch.tensor(cov)
        return cov

    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement
