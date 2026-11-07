"""PyTorch reference implementations of the CUDA kernels.

These are used by the test suite to validate numerical equivalence. They
are NOT used in production — the real PyTorch path is the original code
in mm3DPS/rasterizer_factorized.py. This file exposes the Step 4
inner block in a form that takes the same inputs as the CUDA kernel, so
both paths can be compared head-to-head.

The reference intentionally mirrors the CUDA kernel's input/output
contract (not the original render_factorized structure) so that the test
can bit-compare them.
"""
import math
import torch

INV_PI = 1.0 / math.pi
WAVELENGTH = 3.9e-3
TWO_PI = 2.0 * math.pi


def _itu_slab_fresnel_dtype(eps_real, eps_imag, cos_theta_i, thickness, dtype=None):
    """itu_slab_fresnel that respects the input dtype.

    mm3DPS/bsdf_torch.py:itu_slab_fresnel hardcodes complex64 internally,
    which caps precision at ~1e-7 regardless of input dtype. This version
    honors float64 inputs and produces a complex128 result, so it can be
    used as the bit-accurate reference for validating the CUDA kernel.
    """
    # Decide complex dtype based on real dtype.
    if dtype is None:
        dtype = eps_real.dtype
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    cos_i = cos_theta_i.clamp(1e-6, 1.0)
    sin_sq = 1.0 - cos_i ** 2
    eta = torch.complex(eps_real.to(dtype), -eps_imag.to(dtype))
    a = torch.sqrt(eta - sin_sq.to(cdtype))
    cos_c = cos_i.to(cdtype)
    r_te = (cos_c - a) / (cos_c + a + 1e-10)
    r_tm = (eta * cos_c - a) / (eta * cos_c + a + 1e-10)
    q = (TWO_PI / WAVELENGTH) * thickness.to(cdtype) * a
    ej2q = torch.exp(-2j * q)
    R_TE = r_te * (1.0 - ej2q) / (1.0 - r_te ** 2 * ej2q + 1e-10)
    R_TM = r_tm * (1.0 - ej2q) / (1.0 - r_tm ** 2 * ej2q + 1e-10)
    return R_TE, R_TM


def bsdf_step4_reference(
    wi, wi_r, wo, n_eff, s_in,
    cos_i, cos_o, lambda_i, lambda_o,
    alpha_sq, kappa_SPM, norm_SPM, eps_factor,
    eps_real_m, eps_imag_m, thickness_m,
    E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
    tau_eff, return_intermediates=False, dtype=None,
):
    """PyTorch Step-4 reference.

    Args:
        dtype: If not None, promote all inputs to this dtype before
            computing. Use `torch.float64` to get a bit-accurate reference
            for validating the CUDA kernel against a precision floor
            tighter than float32 ULP.
    """
    if dtype is not None:
        wi = wi.to(dtype); wi_r = wi_r.to(dtype); wo = wo.to(dtype)
        n_eff = n_eff.to(dtype); s_in = s_in.to(dtype)
        cos_i = cos_i.to(dtype); cos_o = cos_o.to(dtype)
        lambda_i = lambda_i.to(dtype); lambda_o = lambda_o.to(dtype)
        alpha_sq = alpha_sq.to(dtype); kappa_SPM = kappa_SPM.to(dtype)
        norm_SPM = norm_SPM.to(dtype); eps_factor = eps_factor.to(dtype)
        eps_real_m = eps_real_m.to(dtype); eps_imag_m = eps_imag_m.to(dtype)
        thickness_m = thickness_m.to(dtype)
        E_s_out_re = E_s_out_re.to(dtype); E_s_out_im = E_s_out_im.to(dtype)
        E_p_out_re = E_p_out_re.to(dtype); E_p_out_im = E_p_out_im.to(dtype)
        tau_eff = tau_eff.to(dtype)
    """PyTorch reference for the fused Step-4 BSDF kernel.

    Inputs exactly mirror `mm3DPS_cuda.bsdf_step4_forward`. Output:
    f_cos (M, n_tx, n_rx) float32.
    """
    # Use the dtype-respecting slab fresnel so dtype=torch.float64 really
    # produces a complex128 reference. mm3DPS/bsdf_torch.py's itu_slab_fresnel
    # is hardcoded to complex64 and would otherwise cap our reference at
    # ~1e-7 regardless of input dtype.
    _slab = _itu_slab_fresnel_dtype
    device = wi.device
    M, n_tx, _ = wi.shape
    n_rx = wo.shape[1]

    # --- KA lobe (half-vector GGX) ---
    wo_dot_wi = torch.einsum('mrj,mtj->mtr', wo, wi)             # (M, n_tx, n_rx)
    wo_dot_n = torch.einsum('mrj,mtj->mtr', wo, n_eff)
    h_num = wo_dot_n + cos_i.unsqueeze(-1)
    h_len = torch.sqrt((2.0 + 2.0 * wo_dot_wi).clamp(min=1e-10))
    h_dot_n = (h_num / h_len).clamp(min=0.0)

    a_sq = alpha_sq[:, None, None]
    denom_ndf = (h_dot_n ** 2 * (a_sq - 1.0) + 1.0) ** 2
    D_KA = a_sq / (math.pi * denom_ndf.clamp(min=1e-20))
    G_KA = 1.0 / (1.0 + lambda_i.unsqueeze(-1) + lambda_o.unsqueeze(-2)).clamp(min=1e-10)
    f_KA = D_KA * G_KA / (4.0 * cos_i.unsqueeze(-1) * cos_o.unsqueeze(-2)).clamp(min=1e-10)

    # --- SPM lobe ---
    cos_dev = torch.einsum('mrj,mtj->mtr', wo, wi_r).clamp(-1.0, 1.0)
    f_SPM = norm_SPM[:, None, None] * torch.exp(kappa_SPM[:, None, None] * (cos_dev - 1.0))
    f_SPM = f_SPM * eps_factor[:, None, None]

    # --- Jones macro (rx_pol = (0, 0, 1)) ---
    rx_pol = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=s_in.dtype)
    # rx_s = dot(rx_pol, s_in) = s_in[..., 2]
    rx_s = s_in[..., 2]                                            # (M, n_tx)
    # p_out = cross(s_in, wo); rx_p = dot(wo, cross(rx_pol, s_in)) / |cross(s_in,wo)|
    rx_pol_cross_s = torch.cross(rx_pol.expand_as(s_in), s_in, dim=-1)
    rx_p_numer = torch.einsum('mrj,mtj->mtr', wo, rx_pol_cross_s)  # (M, n_tx, n_rx)
    s_cross_wo_sq = 1.0 - torch.einsum('mrj,mtj->mtr', wo, s_in) ** 2
    p_out_norm = torch.sqrt(s_cross_wo_sq.clamp(min=1e-12))
    rx_p = rx_p_numer / p_out_norm

    # E_rx = E_s * rx_s + E_p * rx_p  (rx_s per (m, t), rx_p per (m, t, r))
    E_rx_re = E_s_out_re.unsqueeze(-1) * rx_s.unsqueeze(-1) + E_p_out_re.unsqueeze(-1) * rx_p
    E_rx_im = E_s_out_im.unsqueeze(-1) * rx_s.unsqueeze(-1) + E_p_out_im.unsqueeze(-1) * rx_p
    R_jones_macro = (E_rx_re ** 2 + E_rx_im ** 2).clamp(0.0, 1.0)

    # --- Jones h (microfacet Jones at the half-vector) ---
    wi_e = wi.unsqueeze(2).expand(-1, -1, n_rx, -1).contiguous()   # (M, n_tx, n_rx, 3)
    wo_e = wo.unsqueeze(1).expand(-1, n_tx, -1, -1).contiguous()

    hv = wi_e + wo_e
    hv_len = hv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    hv = hv / hv_len
    cos_h = (wi_e * hv).sum(-1).clamp(min=1e-6)

    s_h_raw = torch.cross(wi_e, hv, dim=-1)
    s_h_norm = s_h_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    s_h = s_h_raw / s_h_norm

    p_h_in_raw = torch.cross(s_h, wi_e, dim=-1)
    p_h_in = p_h_in_raw / p_h_in_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    p_h_out_raw = torch.cross(s_h, wo_e, dim=-1)
    p_h_out = p_h_out_raw / p_h_out_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Projections (tx_pol = rx_pol = (0, 0, 1))
    tx_s_h    = s_h[..., 2]
    tx_p_h_in = p_h_in[..., 2]
    rx_s_h    = s_h[..., 2]
    rx_p_h_o  = p_h_out[..., 2]

    # Per-path slab Fresnel at cos_h
    cos_h_flat = cos_h.reshape(-1)
    eps_r_flat = eps_real_m[:, None, None].expand(-1, n_tx, n_rx).reshape(-1)
    eps_i_flat = eps_imag_m[:, None, None].expand(-1, n_tx, n_rx).reshape(-1)
    thick_flat = thickness_m[:, None, None].expand(-1, n_tx, n_rx).reshape(-1)
    R_TE_h, R_TM_h = _slab(eps_r_flat, eps_i_flat, cos_h_flat, thick_flat)
    r_s_h = R_TE_h.reshape(M, n_tx, n_rx)
    r_p_h = R_TM_h.reshape(M, n_tx, n_rx)

    _cdtype = r_s_h.dtype
    E_s_out_h = r_s_h * tx_s_h.to(_cdtype)
    E_p_out_h = r_p_h * tx_p_h_in.to(_cdtype)
    E_rx_h = (E_s_out_h * rx_s_h.to(_cdtype)
              + E_p_out_h * rx_p_h_o.to(_cdtype))
    R_jones_h = (E_rx_h.real ** 2 + E_rx_h.imag ** 2).clamp(0.0, 1.0)

    # --- Blend ---
    tau_ext = tau_eff.unsqueeze(-1)
    f_coh = tau_ext * R_jones_h * f_KA + (1.0 - tau_ext) * R_jones_macro * f_SPM
    f_cos = f_coh * cos_i.unsqueeze(-1)
    if return_intermediates:
        return f_cos, dict(
            f_KA=f_KA, f_SPM=f_SPM,
            R_jones_h=R_jones_h, R_jones_macro=R_jones_macro,
            cos_h=cos_h, cos_dev=cos_dev, h_dot_n=h_dot_n,
            s_h=s_h, p_h_in=p_h_in, p_h_out=p_h_out,
            r_s_h=r_s_h, r_p_h=r_p_h,
        )
    return f_cos
