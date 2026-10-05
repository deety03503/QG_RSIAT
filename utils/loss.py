import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=20, m=0):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        logits = wf.float()
        labels = labels.long()
        rows = torch.arange(labels.shape[0], device=labels.device)
        target = logits[rows, labels]

        if self.loss_type == 'cosface':
            target = target - self.m
        elif self.loss_type == 'arcface':
            target = torch.cos(
                torch.acos(torch.clamp(target, -1.0 + self.eps, 1.0 - self.eps))
                + self.m
            )
        elif self.loss_type == 'sphereface':
            target = torch.cos(
                self.m
                * torch.acos(torch.clamp(target, -1.0 + self.eps, 1.0 - self.eps))
            )

        logits = logits * self.s
        logits[rows, labels] = target * self.s
        return F.cross_entropy(logits, labels)