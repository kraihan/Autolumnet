"""Monotone tone curve T_theta(y) — Lemma 1 + Proposition 1.

T_theta is built as the cumulative sum of softplus-normalised positive weights,
guaranteeing strict monotonicity *by construction*. The weights are predicted
from an IMAGE-LEVEL feature (global average pool), giving exactly one curve per
image — which is what makes spatial order preservation true (Proposition 1).

Do not change `weights` to come from spatial features. See Remark in §3.2 of
the corrected paper for why per-pixel parameters break the proof.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneToneCurve(nn.Module):
    """Strictly increasing piecewise-linear map [0,1] -> [0,1].

    Args:
        feat_dim: dimensionality of the image-level descriptor (e.g. 512 for ResNet-18).
        n_bins:   number of knots (more bins = finer curve, default 64 is plenty).
        eps:      positivity floor for the bin masses (must be > 0).
    """

    def __init__(self, feat_dim: int, n_bins: int = 64, eps: float = 1e-3):
        super().__init__()
        self.n_bins = n_bins
        self.eps = eps
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_bins),
        )

    def forward(self, global_feat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply T_theta pixelwise.

        Args:
            global_feat: (B, feat_dim) image-level descriptor.
            y:           (B, 1, H, W) input luminance in [0,1].
        Returns:
            (B, 1, H, W) corrected luminance in [0,1], same shape as `y`.
        """
        B = y.shape[0]

        # bin masses m_k >= eps > 0  (Lemma 1 requirement)
        w = F.softplus(self.head(global_feat)) + self.eps          # (B, K)
        w = w / w.sum(dim=1, keepdim=True)                          # mass = 1

        # knot values c_0=0, c_k=sum_{j<=k} w_j, c_K=1
        c = torch.cat(
            [torch.zeros(B, 1, device=y.device, dtype=y.dtype), torch.cumsum(w, dim=1)],
            dim=1,
        )

        # piecewise-linear interpolation
        K = self.n_bins
        yf = y.reshape(B, -1).clamp(0.0, 1.0)
        idx = torch.clamp((yf * K).floor().long(), 0, K - 1)
        c0 = torch.gather(c, 1, idx)
        c1 = torch.gather(c, 1, idx + 1)
        frac = (yf * K) - idx.float()
        out = c0 + frac * (c1 - c0)
        return out.reshape_as(y)
