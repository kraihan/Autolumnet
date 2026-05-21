"""Perceptual (VGG-16) and structural-similarity losses."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def ssim_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - SSIM, computed with 3x3 mean filters in the [0,1] range."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = F.avg_pool2d(a, 3, 1, 1)
    mu_b = F.avg_pool2d(b, 3, 1, 1)
    va   = F.avg_pool2d(a * a, 3, 1, 1) - mu_a ** 2
    vb   = F.avg_pool2d(b * b, 3, 1, 1) - mu_b ** 2
    vab  = F.avg_pool2d(a * b, 3, 1, 1) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return 1.0 - s.mean()


class VGGPerceptual(nn.Module):
    """Sum of MSE between activations of three early VGG-16 blocks."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.VGG16_Weights.DEFAULT if pretrained else None
        vgg = torchvision.models.vgg16(weights=weights).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = (a - self.m) / self.s
        b = (b - self.m) / self.s
        loss = a.new_zeros(())
        for sl in self.slices:
            a = sl(a)
            b = sl(b)
            loss = loss + F.mse_loss(a, b)
        return loss
