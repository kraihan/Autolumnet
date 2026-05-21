"""Distribution-matching loss based on 1-D Wasserstein-2 — Lemma 2.

Implements the EXACT 1-D W_2 distance between the empirical luminance
distribution of the prediction and samples from the target prior P*. Its
minimiser over the monotone family of tone curves is the OT map
T* = F_nu^{-1} o F_mu (Proposition 2).

Why sorting, not histograms: a hard histogram has zero gradient w.r.t. pixel
values, breaking gradient flow. Sorting is differentiable (gradients route to
the sorted values), gives the exact 1-D W_2 distance, and has no entropic bias.
"""
from __future__ import annotations
import torch


def sorted_w2_loss(y_pred: torch.Tensor, target_samples: torch.Tensor) -> torch.Tensor:
    """Sorted-sample squared-W2 between predicted luminance and target samples.

    Args:
        y_pred:         (B, 1, H, W) or (B, N) predicted luminance values in [0,1].
        target_samples: (M,) 1-D tensor of samples drawn from the target prior P*.
    Returns:
        Scalar tensor: mean squared distance between sorted prediction quantiles
        and target quantiles. Equals W_2^2 in 1-D up to a constant scale.
    """
    B = y_pred.shape[0]
    yp = y_pred.reshape(B, -1)
    n  = yp.shape[1]

    # quantile resampling of target to match prediction sample count
    q = (torch.arange(n, device=yp.device, dtype=yp.dtype) + 0.5) / n
    tgt_sorted = torch.sort(target_samples)[0]
    ti = torch.clamp((q * len(tgt_sorted)).long(), 0, len(tgt_sorted) - 1)
    tgt_q = tgt_sorted[ti].unsqueeze(0).expand(B, -1)

    # sorted prediction (gradient flows through sort to underlying values)
    yp_sorted, _ = torch.sort(yp, dim=1)
    return ((yp_sorted - tgt_q) ** 2).mean()
