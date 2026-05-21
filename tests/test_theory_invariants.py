"""Unit tests for the theory-critical guarantees.

If any of these fail, the model has silently lost one of the paper's proofs.
Run with:  pytest tests/
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from autolumnet.models import AutoLumNet, ConvexFusion, MonotoneToneCurve


def test_lemma1_strict_monotonicity():
    """T_theta must be strictly increasing for any inputs."""
    torch.manual_seed(0)
    curve = MonotoneToneCurve(feat_dim=32, n_bins=64, eps=1e-3).eval()
    for _ in range(5):
        g = torch.randn(2, 32)
        ramp = torch.linspace(0, 1, 4096).view(1, 1, 1, -1).expand(2, 1, 1, 4096)
        with torch.no_grad():
            t = curve(g, ramp)
        steps = t[..., 1:] - t[..., :-1]
        assert (steps > 0).all(), f"violation: min step = {steps.min().item()}"


def test_prop1_global_order_preserved():
    """One curve per image -> global luminance order preserved."""
    torch.manual_seed(0)
    curve = MonotoneToneCurve(feat_dim=32, n_bins=64).eval()
    g = torch.randn(1, 32)
    y = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        y_out = curve(g, y)
    y_flat   = y.reshape(-1)
    out_flat = y_out.reshape(-1)
    in_order  = torch.sign(y_flat[None, :] - y_flat[:, None])
    out_order = torch.sign(out_flat[None, :] - out_flat[:, None])
    assert torch.equal(in_order, out_order), "global pixel ordering changed"


def test_lemma3_convex_fusion_no_extrapolation():
    """Fused feature must lie in the convex hull of the two branch outputs."""
    torch.manual_seed(0)
    fusion = ConvexFusion(c=8).eval()
    f = torch.randn(2, 8, 4, 4)
    with torch.no_grad():
        out = fusion(f)
        u = fusion.b_ue(f)
        o = fusion.b_oe(f)
    lo = torch.minimum(u, o)
    hi = torch.maximum(u, o)
    assert (out >= lo - 1e-6).all() and (out <= hi + 1e-6).all()


def test_corollary1_bounded_residual():
    """Residual head must respect |r| <= rho."""
    torch.manual_seed(0)
    rho = 0.20
    net = AutoLumNet(pretrained=False, rho=rho).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = net(x)
    assert out["residual"].abs().max().item() <= rho + 1e-6, \
        f"residual exceeded rho: max={out['residual'].abs().max().item()}"


def test_outputs_in_unit_range():
    """All outputs (rgb, y_hat) must stay in [0,1]."""
    torch.manual_seed(0)
    net = AutoLumNet(pretrained=False).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = net(x)
    for k in ("rgb", "y_hat", "y_tone", "y_in"):
        v = out[k]
        assert 0.0 <= v.min().item() and v.max().item() <= 1.0, \
            f"{k} out of [0,1]: min={v.min().item()}, max={v.max().item()}"


def test_sorted_w2_loss_drops_when_pred_matches_target():
    """sorted_w2_loss must be lower when prediction matches the target distribution."""
    from autolumnet.losses import sorted_w2_loss
    torch.manual_seed(0)
    target = torch.rand(10_000) * 0.5 + 0.25     # uniform in [0.25, 0.75]
    pred_good = torch.rand(1, 1, 64, 64) * 0.5 + 0.25
    pred_bad  = torch.ones(1, 1, 64, 64) * 0.99
    good = sorted_w2_loss(pred_good, target)
    bad  = sorted_w2_loss(pred_bad,  target)
    assert good < bad, f"sorted_w2 should be lower for matching pred: good={good}, bad={bad}"
