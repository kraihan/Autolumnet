# Theory ↔ Code map

This is the corrected AutoLumNet, faithful to the rewritten Section 3 of the paper.
Each proven property is enforced by a specific line of code, and is verified by a
unit test in `tests/test_theory_invariants.py`.

## The five proven properties

### Lemma 1 — Strict monotonicity of the tone curve

`T_theta(y) = ∫_0^y m(t) dt`, with `m(t) > 0`, is strictly increasing.

**Code**: `autolumnet/models/tone.py`
```python
w = F.softplus(self.head(global_feat)) + self.eps   # m_k >= eps > 0
w = w / w.sum(dim=1, keepdim=True)                   # normalize
c = torch.cat([zeros, torch.cumsum(w, dim=1)], dim=1)  # T = ∫m
```

**Test**: `test_lemma1_strict_monotonicity` — pushes a 4096-step ramp through
the curve and asserts every successive step is > 0.

### Proposition 1 — Global order and spatial-extremum preservation

Applying *one* strictly-increasing curve to *every* pixel preserves the pairwise
order of pixels and the set of spatial extrema.

**Code**: `autolumnet/models/network.py`
```python
g = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)   # GLOBAL descriptor
y_tone = self.tone(g, y_in)                          # one curve / image
```

The `adaptive_avg_pool2d(..., 1)` is the load-bearing line. If you replace it
with a 1×1 conv (per-pixel features), Proposition 1 fails — see the
counterexample in the corrected paper's Remark.

**Test**: `test_prop1_global_order_preserved` — verifies the sign matrix of all
pixel-pair differences is identical before and after the curve.

### Lemma 2 — 1-D OT map = monotone rearrangement

The optimal transport map from input luminance distribution `μ` to target
`ν=P*` is `T*(y) = F_ν^{-1}(F_μ(y))`, computed exactly by sorting both sample
sets and pairing them in order.

**Code**: `autolumnet/losses/sorted_w2.py`
```python
yp_sorted, _ = torch.sort(yp, dim=1)
tgt_q        = tgt_sorted[ti].expand_as(yp_sorted)
return ((yp_sorted - tgt_q) ** 2).mean()
```

This is exactly `W_2^2(μ_emp, ν_emp)` in 1-D, and its gradient flows through
the `sort` op to the underlying pixel values — so minimising it pushes the
predicted distribution onto `P*`.

**Test**: `test_sorted_w2_loss_drops_when_pred_matches_target` — confirms the
loss is lower on a prediction whose distribution matches the target.

### Lemma 3 — Convex (softmax) fusion stays in the hull

`α_ue + α_oe = 1, α_ue, α_oe ≥ 0` ⇒ fused feature lies in
`[min(U,O), max(U,O)]`; no extrapolation possible.

**Code**: `autolumnet/models/fusion.py`
```python
a = F.softmax(self.gate(f), dim=1)            # sums to 1, non-negative
return a[:, 0:1] * u + a[:, 1:2] * o          # convex combination
```

**Test**: `test_lemma3_convex_fusion_no_extrapolation` — checks every output
element of `ConvexFusion` is between the pointwise min and max of the two
branches.

### Corollary 1 — Bounded residual (Trap T3 in code)

`|r_θ(p)| ≤ ρ` is required for the order-preservation sufficient condition to
hold at most pixels.

**Code**: `autolumnet/models/network.py`
```python
r_y = self.rho * torch.tanh(self.res_y(x))   # |r_y| <= rho
```

`tanh ∈ [-1, 1]` × `rho` enforces the bound structurally; the network cannot
ever output a residual larger than `ρ` regardless of what it learns.

**Test**: `test_corollary1_bounded_residual` — forwards a batch through the
full model and asserts `out["residual"].abs().max() <= rho`.

## What we do *not* claim

The corrected theory is honest about three things:

1. **Highlight clipping is not invertible.** When `Y(p) = 1` from saturation,
   the curve cannot distinguish those pixels. The bounded residual then does
   *prior-based restoration*, not inversion. This is acknowledged in §3.1 of
   the rewritten paper and is the reason the over-exposed test case in the
   toy demo cannot reach W_2 = 0.

2. **Local order preservation is conditional**, not unconditional. The
   sufficient condition in Corollary 1 holds at neighbouring pixels iff
   `T(Y(q)) - T(Y(p)) > |r(p) - r(q)|`. We encourage this with a smoothness
   loss on `r_θ` and a small `ρ = 0.20`, but we don't prove it for every pair.

3. **The mid-gray prior is a modelling choice**, not a theorem. We default to
   a truncated Gaussian centred at 0.5 (`p_star_mean: 0.50` in the YAML),
   which suits SICE. For scenes that are legitimately not centred at mid-gray
   (snow, night, low-key portraits) override it or estimate it from data.

## The deleted penalty

The original paper had `L_mono = Σ max(0, ε - a(p))` — a soft penalty to
encourage positive parameters. It is **absent** from this implementation by
design. Monotonicity is structural (Lemma 1), so the penalty would be
redundant. The absence is itself a statement of the corrected theory.
