"""Convex (softmax) fusion of two exposure-specific branches — Lemma 3."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvexFusion(nn.Module):
    """alpha_ue + alpha_oe = 1, alpha >= 0 -> fused feature lies in the convex
    hull of the two branches. No extrapolation possible (Lemma 3)."""

    def __init__(self, c: int):
        super().__init__()
        self.b_ue = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1),
        )
        self.b_oe = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1),
        )
        self.gate = nn.Conv2d(c, 2, 3, 1, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        u = self.b_ue(f)                                # shadow-recovery branch
        o = self.b_oe(f)                                # highlight-suppression branch
        a = F.softmax(self.gate(f), dim=1)              # (B, 2, H, W), sums to 1
        return a[:, 0:1] * u + a[:, 1:2] * o
