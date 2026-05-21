"""Image-quality metrics that match the EMEF validation format."""
from __future__ import annotations
import math

import torch
import torch.nn.functional as F


def ssim_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    """SSIM using an 11x11 average filter, batched mean."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = F.avg_pool2d(a, 11, 1, 5)
    mu_b = F.avg_pool2d(b, 11, 1, 5)
    va   = F.avg_pool2d(a * a, 11, 1, 5) - mu_a ** 2
    vb   = F.avg_pool2d(b * b, 11, 1, 5) - mu_b ** 2
    vab  = F.avg_pool2d(a * b, 11, 1, 5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean().item())


def psnr_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(F.mse_loss(a, b).item())
    return 99.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)


def l1_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.l1_loss(a, b).item())
