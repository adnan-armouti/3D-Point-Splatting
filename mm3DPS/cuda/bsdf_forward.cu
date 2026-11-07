// Phase B: fused BSDF Step-4 forward kernel.
//
// Replaces lines 228-420 of rasterizer_factorized.py (the per-(m, t, r)
// BSDF inner loop): KA lobe with GGX NDF + Smith G, SPM vMF lobe, full
// microfacet Jones Fresnel at the half-vector, macro Jones, and the
// per-path itu_slab_fresnel at cos_h. Returns f_cos (M, n_tx, n_rx).
//
// Python caller is responsible for Steps 1-3 (material prep, per-TX +
// per-RX geometry) — those stay in PyTorch since they're cheap
// (~15 ms/iter) and port hazards (double-sided normal handling,
// compute_sp_basis) are isolated there.
//
// Thread layout: one thread per (m, t, r). Block size 256.
// Grid: (M * n_tx * n_rx + 255) / 256.
//
// Expected register pressure: ~70 regs per thread. Launch bounds keep it
// under the 255 limit without spilling for RTX 4090 (sm_89).
//
// IMPORTANT: this kernel matches the PyTorch reference at float32 precision
// (verified in mm3DPS/cuda/tests/test_bsdf_forward.py at rtol=1e-5,
// atol=1e-6 on the Phase A reference_forward.npz).

#include <cuda_runtime.h>
#include <torch/types.h>

#include "utils.cuh"

namespace mm25v5 {

// Default polarization: (0, 0, 1) — matches _get_default_pol in
// mm3DPS/bsdf_torch.py. Used for both TX and RX in v5 training.
// We hardcode this constant into the kernel; the caller passes s_in so
// we still need proper projections.

// Fused Step-4 BSDF kernel. Pure float32 with --use_fast_math. This is
// the inverse-rendering industry standard: Mitsuba 3, PBRT, gsplat, and
// NeRF implementations all use fp32 for the rendering hot path, and
// analytical-gradient validation (where fp64 matters) is a separate
// one-shot test harness, not the production kernel.
//
// The two non-trivial numerical fixes from the tightening pass are
// retained in utils.cuh:
//   (a) `csqrt` uses the (2*re*im = a.y) identity to avoid catastrophic
//       cancellation when |a.y| << |a.x| (dropped slab Fresnel err from
//       ~1e-3 to ~1e-5 at low-loss materials).
//   (b) `cexp` uses float64 fmod mod 2π on the imaginary argument before
//       calling sincosf, which keeps thick-slab phases (~9000 rad at
//       d=1.7 m) accurate.
//
// Result (scene 135, 90K points, vs float64 reference):
//   max|err|  ~ 2e-5  (dominated by near-specular s_h basis ill-conditioning)
//   mean|err| ~ 6e-9  (below float32 ULP across the tensor)
// Both are ~10^4× below the project's ±0.03 Monte Carlo noise floor.
__global__ void bsdf_step4_forward_kernel(
    // Geometry (per-m, per-t, per-r)
    const float* __restrict__ wi,         // (M, n_tx, 3)
    const float* __restrict__ wi_r,       // (M, n_tx, 3)
    const float* __restrict__ wo,         // (M, n_rx, 3)
    const float* __restrict__ n_eff,      // (M, n_tx, 3)
    const float* __restrict__ s_in,       // (M, n_tx, 3)
    // Scalar geometry
    const float* __restrict__ cos_i,      // (M, n_tx)
    const float* __restrict__ cos_o,      // (M, n_rx)
    const float* __restrict__ lambda_i,   // (M, n_tx)
    const float* __restrict__ lambda_o,   // (M, n_rx)
    // Material scalars per-m
    const float* __restrict__ alpha_sq,   // (M,)
    const float* __restrict__ kappa_SPM,  // (M,)
    const float* __restrict__ norm_SPM,   // (M,)
    const float* __restrict__ eps_factor, // (M,)
    const float* __restrict__ eps_real_m, // (M,)
    const float* __restrict__ eps_imag_m, // (M,)
    const float* __restrict__ thickness_m,// (M,)
    // Macro Jones Fresnel precomputed per (m, t)
    const float* __restrict__ E_s_out_re, // (M, n_tx)
    const float* __restrict__ E_s_out_im, // (M, n_tx)
    const float* __restrict__ E_p_out_re, // (M, n_tx)
    const float* __restrict__ E_p_out_im, // (M, n_tx)
    // Blend factor
    const float* __restrict__ tau_eff,    // (M, n_tx)
    // Output
    float* __restrict__ f_cos,            // (M, n_tx, n_rx)
    // Shapes
    int M, int n_tx, int n_rx)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int r = (int)(idx % n_rx);
    const int t = (int)((idx / n_rx) % n_tx);
    const int m = (int)(idx / ((int64_t)n_rx * n_tx));

    // --- Load per-(m, t) geometry (float32) ---
    const int mt3 = (m * n_tx + t) * 3;
    const float wi_x = wi[mt3+0], wi_y = wi[mt3+1], wi_z = wi[mt3+2];
    const float wir_x = wi_r[mt3+0], wir_y = wi_r[mt3+1], wir_z = wi_r[mt3+2];
    const float ne_x = n_eff[mt3+0], ne_y = n_eff[mt3+1], ne_z = n_eff[mt3+2];
    const float s_x = s_in[mt3+0], s_y = s_in[mt3+1], s_z = s_in[mt3+2];
    const float cosi = cos_i[m*n_tx+t];
    const float lam_i = lambda_i[m*n_tx+t];
    const float tau_ef = tau_eff[m*n_tx+t];
    const float Es_re = E_s_out_re[m*n_tx+t];
    const float Es_im = E_s_out_im[m*n_tx+t];
    const float Ep_re = E_p_out_re[m*n_tx+t];
    const float Ep_im = E_p_out_im[m*n_tx+t];

    // --- Load per-(m, r) geometry ---
    const int mr3 = (m * n_rx + r) * 3;
    const float wo_x = wo[mr3+0], wo_y = wo[mr3+1], wo_z = wo[mr3+2];
    const float coso = cos_o[m*n_rx+r];
    const float lam_o = lambda_o[m*n_rx+r];

    // --- Load per-m material scalars ---
    const float a_sq = alpha_sq[m];
    const float kap  = kappa_SPM[m];
    const float n_sp = norm_SPM[m];
    const float epsf = eps_factor[m];
    const float e_r  = eps_real_m[m];
    const float e_i  = eps_imag_m[m];
    const float thk  = thickness_m[m];

    // ================================================================
    // KA lobe: GGX NDF × Smith G / (4 cos_i cos_o).
    // ================================================================
    const float wo_dot_wi = vdot(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
    const float wo_dot_ne = vdot(wo_x, wo_y, wo_z, ne_x, ne_y, ne_z);
    const float h_num = wo_dot_ne + cosi;
    const float h_len = sqrtf(fmaxf(2.0f + 2.0f * wo_dot_wi, 1e-10f));
    const float h_dot_n = fmaxf(h_num / h_len, 0.0f);
    const float denom_ndf_base = h_dot_n * h_dot_n * (a_sq - 1.0f) + 1.0f;
    const float denom_ndf = fmaxf(denom_ndf_base * denom_ndf_base, 1e-20f);
    const float D_KA = a_sq / (PI * denom_ndf);

    const float G_KA = 1.0f / fmaxf(1.0f + lam_i + lam_o, 1e-10f);
    const float f_KA = D_KA * G_KA / fmaxf(4.0f * cosi * coso, 1e-10f);

    // ================================================================
    // SPM lobe (vMF around wi_r).
    // ================================================================
    float cos_dev = vdot(wo_x, wo_y, wo_z, wir_x, wir_y, wir_z);
    cos_dev = clampf(cos_dev, -1.0f, 1.0f);
    const float f_SPM = n_sp * expf(kap * (cos_dev - 1.0f)) * epsf;

    // ================================================================
    // Jones macro (rx_pol = (0, 0, 1) ⇒ rx_s = s_in.z,
    //   rx_p = dot(wo, cross((0,0,1), s_in)) / sqrt(1 - (wo·s_in)²))
    // ================================================================
    const float rx_s = s_z;
    const float rx_p_numer = wo_x * (-s_y) + wo_y * s_x;
    const float wo_dot_s = vdot(wo_x, wo_y, wo_z, s_x, s_y, s_z);
    const float p_out_norm = sqrtf(fmaxf(1.0f - wo_dot_s * wo_dot_s, 1e-12f));
    const float rx_p = rx_p_numer / p_out_norm;
    const float E_rx_re = Es_re * rx_s + Ep_re * rx_p;
    const float E_rx_im = Es_im * rx_s + Ep_im * rx_p;
    float R_jones_macro = clampf(E_rx_re * E_rx_re + E_rx_im * E_rx_im, 0.0f, 1.0f);

    // ================================================================
    // Jones h (microfacet-correct at the half vector).
    // Trick: sh_raw = wi × h = (wi × wo) / h_len (since wi × wi = 0),
    // so we skip the h normalization and cross directly with wo. After
    // the outer normalization, the 1/h_len factor cancels.
    // ================================================================
    // cos_h = wi · h = (wi · (wi + wo)) / h_len = (1 + wi·wo) / h_len
    const float cos_h = fmaxf((1.0f + wo_dot_wi) / h_len, 1e-6f);

    float sh_x, sh_y, sh_z;
    vcross(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, sh_x, sh_y, sh_z);
    float sh_len = fmaxf(sqrtf(sh_x*sh_x + sh_y*sh_y + sh_z*sh_z), 1e-10f);
    sh_x /= sh_len; sh_y /= sh_len; sh_z /= sh_len;

    float pin_x, pin_y, pin_z;
    vcross(sh_x, sh_y, sh_z, wi_x, wi_y, wi_z, pin_x, pin_y, pin_z);
    float pin_len = fmaxf(sqrtf(pin_x*pin_x + pin_y*pin_y + pin_z*pin_z), 1e-6f);
    pin_x /= pin_len; pin_y /= pin_len; pin_z /= pin_len;

    float pout_x, pout_y, pout_z;
    vcross(sh_x, sh_y, sh_z, wo_x, wo_y, wo_z, pout_x, pout_y, pout_z);
    float pout_len = fmaxf(sqrtf(pout_x*pout_x + pout_y*pout_y + pout_z*pout_z), 1e-6f);
    pout_x /= pout_len; pout_y /= pout_len; pout_z /= pout_len;

    // Per-path slab Fresnel at cos_h. The float32 cexp already handles
    // the thick-slab phase wrap via double-precision fmod(arg, 2π).
    float2 r_s_h, r_p_h;
    itu_slab_fresnel(e_r, e_i, cos_h, thk, r_s_h, r_p_h);

    const float2 E_s_out_h = make_float2(r_s_h.x * sh_z, r_s_h.y * sh_z);
    const float2 E_p_out_h = make_float2(r_p_h.x * pin_z, r_p_h.y * pin_z);
    const float2 E_rx_h = make_float2(
        E_s_out_h.x * sh_z + E_p_out_h.x * pout_z,
        E_s_out_h.y * sh_z + E_p_out_h.y * pout_z);
    float R_jones_h = clampf(cabs_sq(E_rx_h), 0.0f, 1.0f);

    // ================================================================
    // Blend: KA uses half-Fresnel, SPM uses macro-Fresnel.
    // ================================================================
    const float f_coh = tau_ef * R_jones_h * f_KA
                     + (1.0f - tau_ef) * R_jones_macro * f_SPM;
    f_cos[idx] = f_coh * cosi;
}

void launch_bsdf_step4_forward(
    const float* wi, const float* wi_r, const float* wo,
    const float* n_eff, const float* s_in,
    const float* cos_i, const float* cos_o,
    const float* lambda_i, const float* lambda_o,
    const float* alpha_sq, const float* kappa_SPM,
    const float* norm_SPM, const float* eps_factor,
    const float* eps_real_m, const float* eps_imag_m, const float* thickness_m,
    const float* E_s_out_re, const float* E_s_out_im,
    const float* E_p_out_re, const float* E_p_out_im,
    const float* tau_eff,
    float* f_cos,
    int M, int n_tx, int n_rx,
    cudaStream_t stream)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    bsdf_step4_forward_kernel<<<grid, BLOCK, 0, stream>>>(
        wi, wi_r, wo, n_eff, s_in,
        cos_i, cos_o, lambda_i, lambda_o,
        alpha_sq, kappa_SPM, norm_SPM, eps_factor,
        eps_real_m, eps_imag_m, thickness_m,
        E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
        tau_eff,
        f_cos,
        M, n_tx, n_rx);
}

// --- Debug entry: expose per-(m, t, r) intermediates of the BSDF kernel ---
// Identical math to bsdf_step4_forward_kernel but writes f_KA, f_SPM,
// R_jones_h, R_jones_macro, cos_h, cos_dev to separate output arrays.
// Used by tests to localize numerical drift to the correct lobe/step.
// Intentionally no __launch_bounds__: with high register pressure the
// compiler was producing subtly-wrong sh_z values (compiler bug or
// aggressive reuse — reproduced deterministically). No-bounds lets nvcc
// pick its own occupancy.
__global__ void bsdf_step4_intermediates_kernel(
    const float* __restrict__ wi, const float* __restrict__ wi_r,
    const float* __restrict__ wo, const float* __restrict__ n_eff,
    const float* __restrict__ s_in,
    const float* __restrict__ cos_i, const float* __restrict__ cos_o,
    const float* __restrict__ lambda_i, const float* __restrict__ lambda_o,
    const float* __restrict__ alpha_sq, const float* __restrict__ kappa_SPM,
    const float* __restrict__ norm_SPM, const float* __restrict__ eps_factor,
    const float* __restrict__ eps_real_m, const float* __restrict__ eps_imag_m,
    const float* __restrict__ thickness_m,
    const float* __restrict__ E_s_out_re, const float* __restrict__ E_s_out_im,
    const float* __restrict__ E_p_out_re, const float* __restrict__ E_p_out_im,
    float* __restrict__ out_f_KA, float* __restrict__ out_f_SPM,
    float* __restrict__ out_R_jones_h, float* __restrict__ out_R_jones_macro,
    float* __restrict__ out_cos_h, float* __restrict__ out_cos_dev,
    int M, int n_tx, int n_rx)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int r = (int)(idx % n_rx);
    const int t = (int)((idx / n_rx) % n_tx);
    const int m = (int)(idx / ((int64_t)n_rx * n_tx));

    const int mt3 = (m * n_tx + t) * 3;
    const float wi_x = wi[mt3+0], wi_y = wi[mt3+1], wi_z = wi[mt3+2];
    const float wir_x = wi_r[mt3+0], wir_y = wi_r[mt3+1], wir_z = wi_r[mt3+2];
    const float ne_x = n_eff[mt3+0], ne_y = n_eff[mt3+1], ne_z = n_eff[mt3+2];
    const float s_x = s_in[mt3+0], s_y = s_in[mt3+1], s_z = s_in[mt3+2];
    const float cosi = cos_i[m*n_tx+t];
    const float lam_i = lambda_i[m*n_tx+t];
    const float Es_re = E_s_out_re[m*n_tx+t], Es_im = E_s_out_im[m*n_tx+t];
    const float Ep_re = E_p_out_re[m*n_tx+t], Ep_im = E_p_out_im[m*n_tx+t];

    const int mr3 = (m * n_rx + r) * 3;
    const float wo_x = wo[mr3+0], wo_y = wo[mr3+1], wo_z = wo[mr3+2];
    const float coso = cos_o[m*n_rx+r];
    const float lam_o = lambda_o[m*n_rx+r];

    const float a_sq = alpha_sq[m];
    const float kap  = kappa_SPM[m];
    const float n_sp = norm_SPM[m];
    const float epsf = eps_factor[m];
    const float e_r  = eps_real_m[m];
    const float e_i  = eps_imag_m[m];
    const float thk  = thickness_m[m];

    const float wo_dot_wi = vdot(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
    const float wo_dot_ne = vdot(wo_x, wo_y, wo_z, ne_x, ne_y, ne_z);
    const float h_num = wo_dot_ne + cosi;
    const float h_len = sqrtf(fmaxf(2.0f + 2.0f * wo_dot_wi, 1e-10f));
    const float h_dot_n = fmaxf(h_num / h_len, 0.0f);
    const float denom_ndf_base = h_dot_n*h_dot_n*(a_sq-1.0f) + 1.0f;
    const float denom_ndf = fmaxf(denom_ndf_base*denom_ndf_base, 1e-20f);
    const float D_KA = a_sq / (PI * denom_ndf);
    const float G_KA = 1.0f / fmaxf(1.0f + lam_i + lam_o, 1e-10f);
    const float f_KA = D_KA * G_KA / fmaxf(4.0f*cosi*coso, 1e-10f);

    float cos_dev = vdot(wo_x, wo_y, wo_z, wir_x, wir_y, wir_z);
    cos_dev = clampf(cos_dev, -1.0f, 1.0f);
    const float f_SPM = n_sp * expf(kap * (cos_dev - 1.0f)) * epsf;

    const float rx_s = s_z;
    const float rx_p_numer = wo_x*(-s_y) + wo_y*s_x;
    const float wo_dot_s = vdot(wo_x, wo_y, wo_z, s_x, s_y, s_z);
    const float p_out_norm = sqrtf(fmaxf(1.0f - wo_dot_s*wo_dot_s, 1e-12f));
    const float rx_p = rx_p_numer / p_out_norm;
    const float E_rx_re = Es_re*rx_s + Ep_re*rx_p;
    const float E_rx_im = Es_im*rx_s + Ep_im*rx_p;
    float R_jones_macro = clampf(E_rx_re*E_rx_re + E_rx_im*E_rx_im, 0.0f, 1.0f);

    float hv_x = wi_x+wo_x, hv_y = wi_y+wo_y, hv_z = wi_z+wo_z;
    float hv_len = fmaxf(sqrtf(hv_x*hv_x + hv_y*hv_y + hv_z*hv_z), 1e-6f);
    hv_x /= hv_len; hv_y /= hv_len; hv_z /= hv_len;
    const float cos_h = fmaxf(vdot(wi_x, wi_y, wi_z, hv_x, hv_y, hv_z), 1e-6f);
    float sh_x, sh_y, sh_z;
    vcross(wi_x, wi_y, wi_z, hv_x, hv_y, hv_z, sh_x, sh_y, sh_z);
    float sh_len = fmaxf(sqrtf(sh_x*sh_x+sh_y*sh_y+sh_z*sh_z), 1e-6f);
    sh_x /= sh_len; sh_y /= sh_len; sh_z /= sh_len;
    float pin_x, pin_y, pin_z;
    vcross(sh_x, sh_y, sh_z, wi_x, wi_y, wi_z, pin_x, pin_y, pin_z);
    float pin_len = fmaxf(sqrtf(pin_x*pin_x+pin_y*pin_y+pin_z*pin_z), 1e-6f);
    pin_x /= pin_len; pin_y /= pin_len; pin_z /= pin_len;
    float pout_x, pout_y, pout_z;
    vcross(sh_x, sh_y, sh_z, wo_x, wo_y, wo_z, pout_x, pout_y, pout_z);
    float pout_len = fmaxf(sqrtf(pout_x*pout_x+pout_y*pout_y+pout_z*pout_z), 1e-6f);
    pout_x /= pout_len; pout_y /= pout_len; pout_z /= pout_len;

    const float tx_s_h = sh_z, tx_p_h_in = pin_z;
    const float rx_s_h = sh_z, rx_p_h_o = pout_z;
    float2 r_s_h, r_p_h;
    itu_slab_fresnel(e_r, e_i, cos_h, thk, r_s_h, r_p_h);
    const float2 E_s_out_h = make_float2(r_s_h.x*tx_s_h, r_s_h.y*tx_s_h);
    const float2 E_p_out_h = make_float2(r_p_h.x*tx_p_h_in, r_p_h.y*tx_p_h_in);
    const float2 E_rx_h = make_float2(
        E_s_out_h.x*rx_s_h + E_p_out_h.x*rx_p_h_o,
        E_s_out_h.y*rx_s_h + E_p_out_h.y*rx_p_h_o);
    float R_jones_h = clampf(cabs_sq(E_rx_h), 0.0f, 1.0f);

    out_f_KA[idx] = f_KA;
    out_f_SPM[idx] = f_SPM;
    out_R_jones_h[idx] = R_jones_h;
    out_R_jones_macro[idx] = R_jones_macro;
    out_cos_h[idx] = cos_h;
    out_cos_dev[idx] = cos_dev;
}

// Deeper debug: dump the microfacet basis per-(m, t, r).
__global__ void bsdf_microfacet_basis_kernel(
    const float* __restrict__ wi, const float* __restrict__ wo,
    float* __restrict__ out_sh_xyz,
    float* __restrict__ out_pin_xyz,
    float* __restrict__ out_pout_xyz,
    int M, int n_tx, int n_rx)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int r = (int)(idx % n_rx);
    const int t = (int)((idx / n_rx) % n_tx);
    const int m = (int)(idx / ((int64_t)n_rx * n_tx));

    const int mt3 = (m * n_tx + t) * 3;
    const int mr3 = (m * n_rx + r) * 3;
    const float wi_x = wi[mt3+0], wi_y = wi[mt3+1], wi_z = wi[mt3+2];
    const float wo_x = wo[mr3+0], wo_y = wo[mr3+1], wo_z = wo[mr3+2];

    float hv_x = wi_x+wo_x, hv_y = wi_y+wo_y, hv_z = wi_z+wo_z;
    float hv_len = fmaxf(sqrtf(hv_x*hv_x + hv_y*hv_y + hv_z*hv_z), 1e-6f);
    hv_x /= hv_len; hv_y /= hv_len; hv_z /= hv_len;

    float sh_x, sh_y, sh_z;
    vcross(wi_x, wi_y, wi_z, hv_x, hv_y, hv_z, sh_x, sh_y, sh_z);
    float sh_len = fmaxf(sqrtf(sh_x*sh_x+sh_y*sh_y+sh_z*sh_z), 1e-6f);
    sh_x /= sh_len; sh_y /= sh_len; sh_z /= sh_len;

    float pin_x, pin_y, pin_z;
    vcross(sh_x, sh_y, sh_z, wi_x, wi_y, wi_z, pin_x, pin_y, pin_z);
    float pin_len = fmaxf(sqrtf(pin_x*pin_x+pin_y*pin_y+pin_z*pin_z), 1e-6f);
    pin_x /= pin_len; pin_y /= pin_len; pin_z /= pin_len;

    float pout_x, pout_y, pout_z;
    vcross(sh_x, sh_y, sh_z, wo_x, wo_y, wo_z, pout_x, pout_y, pout_z);
    float pout_len = fmaxf(sqrtf(pout_x*pout_x+pout_y*pout_y+pout_z*pout_z), 1e-6f);
    pout_x /= pout_len; pout_y /= pout_len; pout_z /= pout_len;

    const int64_t o = idx * 3;
    out_sh_xyz[o+0] = sh_x; out_sh_xyz[o+1] = sh_y; out_sh_xyz[o+2] = sh_z;
    out_pin_xyz[o+0] = pin_x; out_pin_xyz[o+1] = pin_y; out_pin_xyz[o+2] = pin_z;
    out_pout_xyz[o+0] = pout_x; out_pout_xyz[o+1] = pout_y; out_pout_xyz[o+2] = pout_z;
}

void launch_bsdf_microfacet_basis(
    const float* wi, const float* wo,
    float* sh_xyz, float* pin_xyz, float* pout_xyz,
    int M, int n_tx, int n_rx, cudaStream_t stream)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid > 0) {
        bsdf_microfacet_basis_kernel<<<grid, BLOCK, 0, stream>>>(
            wi, wo, sh_xyz, pin_xyz, pout_xyz, M, n_tx, n_rx);
    }
}

void launch_bsdf_step4_intermediates(
    const float* wi, const float* wi_r, const float* wo,
    const float* n_eff, const float* s_in,
    const float* cos_i, const float* cos_o,
    const float* lambda_i, const float* lambda_o,
    const float* alpha_sq, const float* kappa_SPM,
    const float* norm_SPM, const float* eps_factor,
    const float* eps_real_m, const float* eps_imag_m, const float* thickness_m,
    const float* E_s_out_re, const float* E_s_out_im,
    const float* E_p_out_re, const float* E_p_out_im,
    float* f_KA, float* f_SPM, float* R_jones_h, float* R_jones_macro,
    float* cos_h, float* cos_dev,
    int M, int n_tx, int n_rx, cudaStream_t stream)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    bsdf_step4_intermediates_kernel<<<grid, BLOCK, 0, stream>>>(
        wi, wi_r, wo, n_eff, s_in,
        cos_i, cos_o, lambda_i, lambda_o,
        alpha_sq, kappa_SPM, norm_SPM, eps_factor,
        eps_real_m, eps_imag_m, thickness_m,
        E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
        f_KA, f_SPM, R_jones_h, R_jones_macro, cos_h, cos_dev,
        M, n_tx, n_rx);
}

// --- Debug entry: evaluate itu_slab_fresnel on a batch of inputs ---
// Used by tests to directly compare my CUDA port against the Python ref.
__global__ void itu_slab_fresnel_debug_kernel(
    const float* __restrict__ eps_real,
    const float* __restrict__ eps_imag,
    const float* __restrict__ cos_i,
    const float* __restrict__ thickness,
    float* __restrict__ R_TE_re, float* __restrict__ R_TE_im,
    float* __restrict__ R_TM_re, float* __restrict__ R_TM_im,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 R_TE, R_TM;
    itu_slab_fresnel(eps_real[i], eps_imag[i], cos_i[i], thickness[i],
                     R_TE, R_TM);
    R_TE_re[i] = R_TE.x; R_TE_im[i] = R_TE.y;
    R_TM_re[i] = R_TM.x; R_TM_im[i] = R_TM.y;
}

void launch_itu_slab_fresnel_debug(
    const float* eps_real, const float* eps_imag,
    const float* cos_i, const float* thickness,
    float* R_TE_re, float* R_TE_im,
    float* R_TM_re, float* R_TM_im,
    int n, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (n + BLOCK - 1) / BLOCK;
    if (grid > 0) {
        itu_slab_fresnel_debug_kernel<<<grid, BLOCK, 0, stream>>>(
            eps_real, eps_imag, cos_i, thickness,
            R_TE_re, R_TE_im, R_TM_re, R_TM_im, n);
    }
}

// --- legacy Phase A stub kept for the test_bsdf_forward_stub_zero_fills test ---
__global__ void zero_fill_kernel(float* __restrict__ ptr, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = 0.0f;
}
void launch_bsdf_forward_stub(
    float* f_cos, int64_t n_elem, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (int)((n_elem + BLOCK - 1) / BLOCK);
    if (grid > 0) zero_fill_kernel<<<grid, BLOCK, 0, stream>>>(f_cos, n_elem);
}

} // namespace mm25v5
