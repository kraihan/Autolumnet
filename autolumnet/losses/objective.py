"""Loss bundle.  L_mono is intentionally absent — monotonicity is structural (Lemma 1)."""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .perceptual import VGGPerceptual, ssim_loss
from .sorted_w2 import sorted_w2_loss


@dataclass
class LossWeights:
    rec:    float = 1.0
    align:  float = 0.5
    perc:   float = 0.1
    smooth: float = 0.1
    ssim:   float = 0.2     # weight inside L_rec for SSIM term


def compute_loss(
    out: dict[str, torch.Tensor],
    gt_rgb: torch.Tensor,
    perc: VGGPerceptual,
    p_star: torch.Tensor,
    w: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Unified exposure-aware objective.

    L = w_rec * (L1 + w_ssim * (1-SSIM))
      + w_align  * sorted-W2( y_hat , P* )
      + w_perc   * sum_l || VGG_l(pred) - VGG_l(gt) ||^2
      + w_smooth * || grad r_y ||_2^2
    """
    pred = out["rgb"]
    l_rec   = F.l1_loss(pred, gt_rgb) + w.ssim * ssim_loss(pred, gt_rgb)
    l_align = sorted_w2_loss(out["y_hat"], p_star)
    l_perc  = perc(pred, gt_rgb)

    r = out["residual"]
    l_smooth = (
        (r[..., 1:, :] - r[..., :-1, :]).pow(2).mean()
        + (r[..., :, 1:] - r[..., :, :-1]).pow(2).mean()
    )

    total = w.rec * l_rec + w.align * l_align + w.perc * l_perc + w.smooth * l_smooth
    return total, {
        "rec":    float(l_rec.detach()),
        "align":  float(l_align.detach()),
        "perc":   float(l_perc.detach()),
        "smooth": float(l_smooth.detach()),
    }
