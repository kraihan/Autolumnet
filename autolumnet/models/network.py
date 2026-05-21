"""ResNet-18 encoder, bounded-residual U-Net decoder, full AutoLumNet model."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .tone import MonotoneToneCurve
from .fusion import ConvexFusion


# Rec. 601 luminance weights
_RGB2Y = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [0,1] -> (B,1,H,W) in [0,1]."""
    return (rgb * _RGB2Y.to(rgb.device, rgb.dtype)).sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class ResNet18Encoder(nn.Module):
    """Shared encoder producing a 5-level feature pyramid (strides 2, 4, 8, 16, 32)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)   # /2,   64
        self.pool = m.maxpool
        self.l1, self.l2 = m.layer1, m.layer2               # /4 64,  /8 128
        self.l3, self.l4 = m.layer3, m.layer4               # /16 256,/32 512
        self.dims = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stem(x)
        f1 = self.l1(self.pool(f0))
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return [f0, f1, f2, f3, f4]


# --------------------------------------------------------------------------- #
# Decoder: bounded luminance residual + small chroma residual
# --------------------------------------------------------------------------- #
class _UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.fuse = ConvexFusion(cin)
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, 1, 1), nn.GELU(),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.fuse(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class BoundedResidualDecoder(nn.Module):
    """U-Net decoder. Produces:
        r_y in [-rho, +rho]    (luminance residual, bounded by Corollary 1)
        r_c in [-0.08, +0.08]  (chroma residual on R and B channels)
    """

    def __init__(self, dims: list[int], rho: float = 0.20):
        super().__init__()
        d0, d1, d2, d3, d4 = dims
        self.rho = rho
        self.up3 = _UpBlock(d4, d3, 256)
        self.up2 = _UpBlock(256, d2, 128)
        self.up1 = _UpBlock(128, d1, 64)
        self.up0 = _UpBlock(64,  d0, 32)
        self.res_y = nn.Conv2d(32, 1, 3, 1, 1)
        self.res_c = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, feats: list[torch.Tensor], hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        f0, f1, f2, f3, f4 = feats
        x = self.up3(f4, f3)
        x = self.up2(x,  f2)
        x = self.up1(x,  f1)
        x = self.up0(x,  f0)
        x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
        r_y = self.rho * torch.tanh(self.res_y(x))      # |r_y| <= rho (Corollary 1)
        r_c = 0.08    * torch.tanh(self.res_c(x))       # small chroma adjustment
        return r_y, r_c


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class AutoLumNet(nn.Module):
    """Bi-branch exposure-aware network for single-shot exposure correction.

    Forward returns a dict:
        rgb     : (B,3,H,W) corrected image in [0,1]
        y_in    : (B,1,H,W) input luminance
        y_tone  : (B,1,H,W) T_theta(Y_in)   -- after global monotone curve
        y_hat   : (B,1,H,W) clip(T_theta(Y) + r_y)  -- final luminance
        residual: (B,1,H,W) bounded luminance residual r_y, |r_y| <= rho
    """

    def __init__(self, pretrained: bool = True, n_bins: int = 64, rho: float = 0.20):
        super().__init__()
        self.enc  = ResNet18Encoder(pretrained)
        self.tone = MonotoneToneCurve(self.enc.dims[-1], n_bins=n_bins)
        self.dec  = BoundedResidualDecoder(self.enc.dims, rho=rho)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = rgb.shape[-2:]
        feats = self.enc(rgb)
        g = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)   # GLOBAL descriptor (Prop. 1)

        y_in   = rgb_to_luma(rgb)
        y_tone = self.tone(g, y_in)                          # T_theta(Y) -- monotone
        r_y, r_c = self.dec(feats, (H, W))

        y_hat = (y_tone + r_y).clamp(0.0, 1.0)               # the ONE definition

        # transfer luminance back to RGB, preserving chroma ratios
        gain = (y_hat + 1e-4) / (y_in + 1e-4)
        out = (rgb * gain).clamp(0.0, 1.0)

        # small per-channel chroma residual on R, B
        R = (out[:, 0:1] + r_c[:, 0:1]).clamp(0.0, 1.0)
        G =  out[:, 1:2]
        B = (out[:, 2:3] + r_c[:, 1:2]).clamp(0.0, 1.0)
        out = torch.cat([R, G, B], dim=1)

        return {"rgb": out, "y_in": y_in, "y_tone": y_tone,
                "y_hat": y_hat, "residual": r_y}
