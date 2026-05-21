"""Exponential moving average of model parameters.

A standard, mostly-free SOTA trick. Maintain a shadow copy of the parameters
updated as ema = decay*ema + (1-decay)*current, then use the shadow copy for
evaluation. Typically gives +0.1-0.3 PSNR for free.
"""
from __future__ import annotations
import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(self._unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _unwrap(m: nn.Module) -> nn.Module:
        return m.module if hasattr(m, "module") else m

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = self._unwrap(model)
        d = self.decay
        for ep, sp in zip(self.ema.parameters(), src.parameters()):
            ep.mul_(d).add_(sp.detach(), alpha=1 - d)
        for eb, sb in zip(self.ema.buffers(), src.buffers()):
            eb.copy_(sb)

    def state_dict(self) -> dict:
        return {"ema": self.ema.state_dict(), "decay": self.decay}

    def load_state_dict(self, sd: dict) -> None:
        self.ema.load_state_dict(sd["ema"])
        self.decay = sd.get("decay", self.decay)
