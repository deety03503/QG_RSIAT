"""Exponential moving average state for feature centering."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EMACenter(nn.Module):
    def __init__(self, dimension: int, decay: float = 0.99) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("centering EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.register_buffer("value", torch.zeros(1, dimension))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, features: Any) -> Any:
        mean = features.detach().float().mean(dim=0, keepdim=True)
        if self.initialized:
            self.value.mul_(self.decay).add_(mean.to(self.value), alpha=1.0 - self.decay)
        else:
            self.value.copy_(mean.to(self.value))
            self.initialized.fill_(True)
        return self.value.to(features)

    def current(self, features: Any) -> Any | None:
        if not self.initialized:
            return None
        return self.value.to(features)
