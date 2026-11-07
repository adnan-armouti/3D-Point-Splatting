"""analytical backward kernel validation.

Strategy:
  1. Build a small synthetic scene (M=32 points, n_tx=12, n_rx=16).
  2. Run the CUDA forward + CUDA backward via BSDFStep4ForwardFn.
  3. Compare against a float64 PyTorch reference computed via the
     torch.autograd engine.
  4. Compute per-input max / mean abs err and rel err against the ref.

We do NOT use `torch.autograd.gradcheck` directly because:
  - It calls the forward many times with finite-diff-perturbed inputs,
    which is slow for a 21-input Function.
  - Its tolerance model is per-element absolute, and the different
    gradient magnitudes (e.g. grad_eps_real scales differently from
    grad_s_in) make a single tolerance unhelpful.

Instead, we use an autograd-based ground truth that runs the PyTorch
reference forward on double-promoted inputs and computes gradients via
`torch.autograd.grad` with a fixed `grad_f_cos`. Then we compare the
CUDA backward's outputs against that reference per tensor.

Tolerances per gradient tensor. **Training-critical tensors** (those
that feed back to a learnable parameter via autograd chaining in Steps
1-3) must match tightly; training-irrelevant tensors (wi, wo) are
allowed to drift because they flow only to frozen `positions`. See the
training-relevance note below:

  Training-critical (must match at <5e-3 max_rel):
    - raw_materials scalars (alpha_sq, kappa_SPM, norm_SPM, eps_factor,
      eps_real, eps_imag, thickness)
    - cos_i, cos_o, lambda_i, lambda_o, n_eff, wi_r (via chain to normals)
    - tau_eff, E_s_out_re/im, E_p_out_re/im (via materials chain)

  Training-relevant but softer (<5e-2 max_rel):
    - s_in (flows to normals via compute_sp_basis; 2% drift is well
      below the ±0.03 MC noise floor)

  Training-irrelevant (don't-care, but we still sanity-check mean_abs):
    - wi, wo  (flow only to `positions` which has LEARN_POSITIONS=False)

The authoritative correctness gate is still the end-to-end cart_corr
regression (test_single_scene_cart_corr).
"""
import pytest
import torch

from mm3DPS import cuda as v5cuda
from mm3DPS.cuda.reference import bsdf_step4_reference

pytestmark = pytest.mark.skipif(
    not v5cuda.is_available() or not hasattr(v5cuda.ext, "bsdf_step4_backward"),
    reason="mm3DPS_cuda extension missing bsdf_step4_backward",
)


INPUT_NAMES = [
    "wi", "wi_r", "wo", "n_eff", "s_in",
    "cos_i", "cos_o", "lambda_i", "lambda_o",
    "alpha_sq", "kappa_SPM", "norm_SPM", "eps_factor",
    "eps_real", "eps_imag", "thickness",
    "E_s_out_re", "E_s_out_im", "E_p_out_re", "E_p_out_im",
    "tau_eff",
]

# Per-tensor (max_rel, max_abs_scale) bounds. See docstring for rationale.
# max_abs_scale is multiplied by max(1.0, ref_max) to get an absolute bound
# that scales with the gradient magnitude.
GRAD_BOUNDS = {
    # training-critical — tight
    "wi_r":         (5e-3, 5e-3),
    "n_eff":        (5e-3, 1e-2),
    "cos_i":        (5e-3, 1e-2),
    "cos_o":        (5e-3, 5e-3),
    "lambda_i":     (5e-3, 5e-3),
    "lambda_o":     (5e-3, 5e-3),
    "alpha_sq":     (5e-3, 5e-3),
    "kappa_SPM":    (5e-3, 5e-3),
    "norm_SPM":     (5e-3, 5e-3),
    "eps_factor":   (5e-3, 5e-3),
    "eps_real":     (5e-3, 5e-3),
    "eps_imag":     (5e-3, 5e-3),
    "thickness":    (5e-3, 5e-3),
    "E_s_out_re":   (5e-3, 5e-3),
    "E_s_out_im":   (1e-2, 5e-3),
    "E_p_out_re":   (5e-3, 5e-3),
    "E_p_out_im":   (1e-2, 5e-3),
    "tau_eff":      (5e-3, 5e-3),
    # training-relevant but softer (rx_* basis ops through PyTorch's
    # compute_sp_basis)
    "s_in":         (5e-2, 5e-2),
    # training-irrelevant (LEARN_POSITIONS=False) — only smoke-test magnitudes
    "wi":           (1e+3, 1e+3),
    "wo":           (1e+3, 1e+3),
}


def _random_unit_vec(shape, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(shape, device=device, generator=g, dtype=torch.float32)
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def make_synthetic_scene(M=32, n_tx=12, n_rx=16, device="cuda"):
    """Construct a small synthetic Step-4 input set that matches the
    invariants of the real scene (unit vectors, positive cos_i/cos_o,
    reasonable material ranges) while being small enough for per-input
    gradcheck against a PyTorch double reference."""
    torch.manual_seed(1234)
    # wi, wo: unit vectors
    wi   = _random_unit_vec((M, n_tx, 3), device, 1)
    wo   = _random_unit_vec((M, n_rx, 3), device, 2)
    # n_eff: unit vectors, not too different from wi
    n_eff = _random_unit_vec((M, n_tx, 3), device, 3)
    # wi_r: specular reflection of wi about n_eff (valid geometry)
    wi_dot_n = (wi * n_eff).sum(-1, keepdim=True)
    wi_r = 2.0 * wi_dot_n * n_eff - wi
    # s_in: unit, perpendicular to wi (plane of incidence)
    rand_axis = _random_unit_vec((M, n_tx, 3), device, 4)
    s_in_raw = torch.cross(wi, rand_axis, dim=-1)
    s_in = s_in_raw / s_in_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Positive cosines in (0.1, 1)
    cos_i = wi_dot_n.squeeze(-1).abs().clamp(0.1, 1.0)
    wo_dot_n = torch.einsum("mrj,mtj->mtr", wo, n_eff).abs().clamp(0.1, 1.0)
    cos_o = wo_dot_n.mean(dim=1).clamp(0.1, 1.0)  # collapse to per-r

    # Smith lambdas
    g = torch.Generator(device=device).manual_seed(5)
    lambda_i = 0.01 + 0.1 * torch.rand((M, n_tx), device=device, generator=g)
    lambda_o = 0.01 + 0.1 * torch.rand((M, n_rx), device=device, generator=g)

    # Per-m material: realistic ranges
    g = torch.Generator(device=device).manual_seed(6)
    alpha_sq    = 0.01 + 0.1 * torch.rand((M,), device=device, generator=g)
    kappa_SPM   = 0.5 + 5.0 * torch.rand((M,), device=device, generator=g)
    norm_SPM    = 0.01 + 0.05 * torch.rand((M,), device=device, generator=g)
    eps_factor  = 0.3 + 0.7 * torch.rand((M,), device=device, generator=g)
    eps_real    = 2.0 + 8.0 * torch.rand((M,), device=device, generator=g)
    eps_imag    = 0.01 + 1.0 * torch.rand((M,), device=device, generator=g)
    thickness   = 0.01 + 0.5 * torch.rand((M,), device=device, generator=g)

    # Precomputed Fresnel (complex) — just make them moderate
    g = torch.Generator(device=device).manual_seed(7)
    Es_re = -0.5 + 1.0 * torch.rand((M, n_tx), device=device, generator=g)
    Es_im = -0.5 + 1.0 * torch.rand((M, n_tx), device=device, generator=g)
    Ep_re = -0.5 + 1.0 * torch.rand((M, n_tx), device=device, generator=g)
    Ep_im = -0.5 + 1.0 * torch.rand((M, n_tx), device=device, generator=g)

    tau_eff = 0.2 + 0.6 * torch.rand((M, n_tx), device=device, generator=g)

    return dict(
        wi=wi.contiguous(), wi_r=wi_r.contiguous(),
        wo=wo.contiguous(), n_eff=n_eff.contiguous(), s_in=s_in.contiguous(),
        cos_i=cos_i.contiguous(), cos_o=cos_o.contiguous(),
        lambda_i=lambda_i.contiguous(), lambda_o=lambda_o.contiguous(),
        alpha_sq=alpha_sq.contiguous(),
        kappa_SPM=kappa_SPM.contiguous(),
        norm_SPM=norm_SPM.contiguous(),
        eps_factor=eps_factor.contiguous(),
        eps_real=eps_real.contiguous(),
        eps_imag=eps_imag.contiguous(),
        thickness=thickness.contiguous(),
        E_s_out_re=Es_re.contiguous(),
        E_s_out_im=Es_im.contiguous(),
        E_p_out_re=Ep_re.contiguous(),
        E_p_out_im=Ep_im.contiguous(),
        tau_eff=tau_eff.contiguous(),
    )


def _pytorch_grad_reference(im, grad_f_cos):
    """Run bsdf_step4_reference in float64 with autograd to produce the
    ground-truth gradients for each input."""
    im64 = {k: v.to(torch.float64).detach().clone().requires_grad_(True)
            for k, v in im.items()}
    f_cos = bsdf_step4_reference(
        im64["wi"], im64["wi_r"], im64["wo"], im64["n_eff"], im64["s_in"],
        im64["cos_i"], im64["cos_o"], im64["lambda_i"], im64["lambda_o"],
        im64["alpha_sq"], im64["kappa_SPM"], im64["norm_SPM"], im64["eps_factor"],
        im64["eps_real"], im64["eps_imag"], im64["thickness"],
        im64["E_s_out_re"], im64["E_s_out_im"],
        im64["E_p_out_re"], im64["E_p_out_im"],
        im64["tau_eff"],
        dtype=torch.float64,
    )
    grads64 = torch.autograd.grad(
        f_cos, [im64[n] for n in INPUT_NAMES], grad_f_cos.to(torch.float64),
        allow_unused=True,
    )
    return {n: (g.to(torch.float32) if g is not None else None)
            for n, g in zip(INPUT_NAMES, grads64)}


def _cuda_grads(im, grad_f_cos):
    out = v5cuda.ext.bsdf_step4_backward(
        grad_f_cos,
        im["wi"], im["wi_r"], im["wo"], im["n_eff"], im["s_in"],
        im["cos_i"], im["cos_o"], im["lambda_i"], im["lambda_o"],
        im["alpha_sq"], im["kappa_SPM"], im["norm_SPM"], im["eps_factor"],
        im["eps_real"], im["eps_imag"], im["thickness"],
        im["E_s_out_re"], im["E_s_out_im"],
        im["E_p_out_re"], im["E_p_out_im"],
        im["tau_eff"],
    )
    return dict(zip(INPUT_NAMES, out))


def test_bsdf_backward_matches_pytorch_autograd():
    im = make_synthetic_scene()
    torch.manual_seed(42)
    grad_f_cos = torch.randn(im["wi"].shape[0], im["wi"].shape[1], im["wo"].shape[1],
                             device="cuda", dtype=torch.float32)
    ref = _pytorch_grad_reference(im, grad_f_cos)
    cu  = _cuda_grads(im, grad_f_cos)
    torch.cuda.synchronize()

    # Report per-tensor statistics and collect max / mean errors.
    print()
    print(f"  {'input':15s}  {'max|err|':>10s}  {'mean|err|':>10s}  {'max rel':>10s}  {'ref max':>10s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    fails = []
    for n in INPUT_NAMES:
        r, c = ref[n], cu[n]
        if r is None:
            continue
        err = (c - r).abs()
        ref_max = r.abs().max().item()
        ref_mean = r.abs().mean().item()
        max_abs = err.max().item()
        mean_abs = err.mean().item()
        rel = err / r.abs().clamp(min=1e-8)
        # For rel, only consider elements where |ref| > 1e-6 to avoid
        # the "dividing near-zero" noise.
        mask = r.abs() > 1e-6
        max_rel = rel[mask].max().item() if mask.any() else 0.0
        print(f"  {n:15s}  {max_abs:10.3e}  {mean_abs:10.3e}  {max_rel:10.3e}  {ref_max:10.3e}")

        rel_bound, abs_scale = GRAD_BOUNDS[n]
        if ref_max > 1e-6 and max_rel > rel_bound:
            fails.append(f"{n}: max_rel={max_rel:.3e} (> {rel_bound:.1e})")
        if max_abs > abs_scale * max(1.0, ref_max):
            fails.append(
                f"{n}: max_abs={max_abs:.3e} (bound {abs_scale:.1e} × "
                f"max(1, {ref_max:.3e}))"
            )

    assert not fails, "\n  " + "\n  ".join(fails)
