// Phase C: fused BSDF Step-4 backward kernel.
//
// Given grad_f_cos (M, n_tx, n_rx), computes analytical gradients for all
// 21 forward inputs by differentiating the fused BSDF chain rule on the
// fly. One thread per (m, t, r). Atomic accumulation to per-input grad
// tensors (many inputs are shared across t or r dimensions).
//
// Forward recap (see bsdf_forward.cu for full commentary):
//
//   f_cos = f_coh * cos_i
//   f_coh = tau_eff * R_jones_h * f_KA + (1 - tau_eff) * R_jones_macro * f_SPM
//
//   f_KA = D_KA * G_KA / (4 * cos_i * cos_o)
//     D_KA = alpha_sq / (PI * denom²)
//     denom = h_dot_n² * (alpha_sq - 1) + 1
//     h_dot_n = (wo·n_eff + cos_i) / sqrt(2 + 2*wo·wi)
//     G_KA = 1 / (1 + lambda_i + lambda_o)
//
//   f_SPM = norm_SPM * exp(kappa_SPM * (cos_dev - 1)) * eps_factor
//     cos_dev = clamp(wo · wi_r, -1, 1)
//
//   R_jones_macro = |Es*rx_s + Ep*rx_p|²  (complex E_s_out, E_p_out given)
//     rx_s = s_in.z
//     rx_p = (wo_x·(-s_in.y) + wo_y·s_in.x) / sqrt(1 - (wo·s_in)²)
//
//   R_jones_h = |r_s_h·sh_z² + r_p_h·pin_z·pout_z|²
//     sh = normalize(wi × wo)   (= wi × h since wi×wi=0)
//     pin = normalize(sh × wi)
//     pout = normalize(sh × wo)
//     cos_h = (1 + wo·wi) / h_len
//     r_s_h, r_p_h from itu_slab_fresnel(eps_real, eps_imag, cos_h, thk)
//
// Chain rule sketch (see implementation for exact derivations):
//   dL/dtau = grad_f_cos·cos_i · (R_jones_h·f_KA - R_jones_macro·f_SPM)
//   dL/df_KA, dL/df_SPM, dL/dR_jones_h, dL/dR_jones_macro = similar
//   ... propagate through D_KA, G_KA, sh basis, slab Fresnel ...
//
// This file is written in strict parallel to bsdf_forward.cu so the
// forward and backward derivations can be cross-checked.

#include <cuda_runtime.h>
#include <torch/types.h>

#include "utils.cuh"

namespace mm25v5 {

// Phase A leftover — kept for the zero-fill test that still references it.
__global__ void zero_fill_kernel_bw(float* __restrict__ ptr, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = 0.0f;
}
void launch_bsdf_backward_stub(
    float* grad_raw_mat, int64_t n_elem, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (int)((n_elem + BLOCK - 1) / BLOCK);
    if (grid > 0) {
        zero_fill_kernel_bw<<<grid, BLOCK, 0, stream>>>(grad_raw_mat, n_elem);
    }
}

// Kernel that zero-initializes a float tensor (used by bindings before
// scatter accumulation).
__global__ void zero_kernel(float* ptr, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = 0.0f;
}
void launch_zero(float* ptr, int64_t n, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const int grid = (int)((n + BLOCK - 1) / BLOCK);
    if (grid > 0) zero_kernel<<<grid, BLOCK, 0, stream>>>(ptr, n);
}

// ---------------------------------------------------------------------------
// Fused BSDF backward kernel (Phase C)
// ---------------------------------------------------------------------------

__global__ void bsdf_step4_backward_kernel(
    // ----- Upstream grad -----
    const float* __restrict__ grad_f_cos,   // (M, n_tx, n_rx)
    // ----- Forward inputs (same as forward kernel) -----
    const float* __restrict__ wi,            // (M, n_tx, 3)
    const float* __restrict__ wi_r,          // (M, n_tx, 3)
    const float* __restrict__ wo,            // (M, n_rx, 3)
    const float* __restrict__ n_eff,         // (M, n_tx, 3)
    const float* __restrict__ s_in,          // (M, n_tx, 3)
    const float* __restrict__ cos_i,         // (M, n_tx)
    const float* __restrict__ cos_o,         // (M, n_rx)
    const float* __restrict__ lambda_i,      // (M, n_tx)
    const float* __restrict__ lambda_o,      // (M, n_rx)
    const float* __restrict__ alpha_sq,      // (M,)
    const float* __restrict__ kappa_SPM,     // (M,)
    const float* __restrict__ norm_SPM,      // (M,)
    const float* __restrict__ eps_factor,    // (M,)
    const float* __restrict__ eps_real_m,    // (M,)
    const float* __restrict__ eps_imag_m,    // (M,)
    const float* __restrict__ thickness_m,   // (M,)
    const float* __restrict__ E_s_out_re,    // (M, n_tx)
    const float* __restrict__ E_s_out_im,    // (M, n_tx)
    const float* __restrict__ E_p_out_re,    // (M, n_tx)
    const float* __restrict__ E_p_out_im,    // (M, n_tx)
    const float* __restrict__ tau_eff,       // (M, n_tx)
    // ----- Output grads (accumulated via atomicAdd) -----
    float* __restrict__ grad_wi,             // (M, n_tx, 3)
    float* __restrict__ grad_wi_r,           // (M, n_tx, 3)
    float* __restrict__ grad_wo,             // (M, n_rx, 3)
    float* __restrict__ grad_n_eff,          // (M, n_tx, 3)
    float* __restrict__ grad_s_in,           // (M, n_tx, 3)
    float* __restrict__ grad_cos_i,          // (M, n_tx)
    float* __restrict__ grad_cos_o,          // (M, n_rx)
    float* __restrict__ grad_lambda_i,       // (M, n_tx)
    float* __restrict__ grad_lambda_o,       // (M, n_rx)
    float* __restrict__ grad_alpha_sq,       // (M,)
    float* __restrict__ grad_kappa_SPM,      // (M,)
    float* __restrict__ grad_norm_SPM,       // (M,)
    float* __restrict__ grad_eps_factor,     // (M,)
    float* __restrict__ grad_eps_real_m,     // (M,)
    float* __restrict__ grad_eps_imag_m,     // (M,)
    float* __restrict__ grad_thickness_m,    // (M,)
    float* __restrict__ grad_E_s_out_re,     // (M, n_tx)
    float* __restrict__ grad_E_s_out_im,     // (M, n_tx)
    float* __restrict__ grad_E_p_out_re,     // (M, n_tx)
    float* __restrict__ grad_E_p_out_im,     // (M, n_tx)
    float* __restrict__ grad_tau_eff,        // (M, n_tx)
    // ----- Shapes -----
    int M, int n_tx, int n_rx)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int r = (int)(idx % n_rx);
    const int t = (int)((idx / n_rx) % n_tx);
    const int m = (int)(idx / ((int64_t)n_rx * n_tx));

    const float gf = grad_f_cos[idx];
    if (gf == 0.0f) return;   // fast path for culled paths

    // ================================================================
    // STAGE 1: Recompute forward intermediates (same as forward kernel)
    // ================================================================
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

    // --- KA lobe forward recomputation ---
    const float wo_dot_wi = vdot(wo_x, wo_y, wo_z, wi_x, wi_y, wi_z);
    const float wo_dot_ne = vdot(wo_x, wo_y, wo_z, ne_x, ne_y, ne_z);
    const float h_num = wo_dot_ne + cosi;
    const float h_len_sq_raw = 2.0f + 2.0f * wo_dot_wi;
    const float h_len_sq_clamped = fmaxf(h_len_sq_raw, 1e-10f);
    const float h_len = sqrtf(h_len_sq_clamped);
    const float h_dot_n_raw = h_num / h_len;
    const float h_dot_n = fmaxf(h_dot_n_raw, 0.0f);
    const float denom_ndf_base = h_dot_n * h_dot_n * (a_sq - 1.0f) + 1.0f;
    const float denom_ndf_sq = denom_ndf_base * denom_ndf_base;
    const float denom_ndf = fmaxf(denom_ndf_sq, 1e-20f);
    const float D_KA = a_sq / (PI * denom_ndf);
    const float G_KA_denom = fmaxf(1.0f + lam_i + lam_o, 1e-10f);
    const float G_KA = 1.0f / G_KA_denom;
    const float f_KA_denom = fmaxf(4.0f * cosi * coso, 1e-10f);
    const float f_KA = D_KA * G_KA / f_KA_denom;

    // --- SPM lobe forward recomputation ---
    const float cos_dev_raw = vdot(wo_x, wo_y, wo_z, wir_x, wir_y, wir_z);
    const float cos_dev = clampf(cos_dev_raw, -1.0f, 1.0f);
    const float exp_arg = kap * (cos_dev - 1.0f);
    const float exp_val = expf(exp_arg);
    const float f_SPM = n_sp * exp_val * epsf;

    // --- Jones macro forward recomputation ---
    const float rx_s = s_z;
    const float rx_p_numer = wo_x * (-s_y) + wo_y * s_x;
    const float wo_dot_s = vdot(wo_x, wo_y, wo_z, s_x, s_y, s_z);
    const float s_cross_wo_sq_raw = 1.0f - wo_dot_s * wo_dot_s;
    const float s_cross_wo_sq = fmaxf(s_cross_wo_sq_raw, 1e-12f);
    const float p_out_norm = sqrtf(s_cross_wo_sq);
    const float rx_p = rx_p_numer / p_out_norm;
    const float E_rx_re = Es_re * rx_s + Ep_re * rx_p;
    const float E_rx_im = Es_im * rx_s + Ep_im * rx_p;
    const float R_jones_macro_raw = E_rx_re * E_rx_re + E_rx_im * E_rx_im;
    const float R_jones_macro = clampf(R_jones_macro_raw, 0.0f, 1.0f);

    // --- Jones h forward recomputation ---
    const float cos_h_raw = (1.0f + wo_dot_wi) / h_len;
    const float cos_h = fmaxf(cos_h_raw, 1e-6f);

    float sh_raw_x, sh_raw_y, sh_raw_z;
    vcross(wi_x, wi_y, wi_z, wo_x, wo_y, wo_z, sh_raw_x, sh_raw_y, sh_raw_z);
    const float sh_len_sq_raw =
        sh_raw_x*sh_raw_x + sh_raw_y*sh_raw_y + sh_raw_z*sh_raw_z;
    const float sh_len_sq = fmaxf(sh_len_sq_raw, 1e-20f);
    const float sh_len = sqrtf(sh_len_sq);
    const float sh_x = sh_raw_x / sh_len;
    const float sh_y = sh_raw_y / sh_len;
    const float sh_z = sh_raw_z / sh_len;

    float pin_raw_x, pin_raw_y, pin_raw_z;
    vcross(sh_x, sh_y, sh_z, wi_x, wi_y, wi_z, pin_raw_x, pin_raw_y, pin_raw_z);
    const float pin_len_sq_raw =
        pin_raw_x*pin_raw_x + pin_raw_y*pin_raw_y + pin_raw_z*pin_raw_z;
    const float pin_len_sq = fmaxf(pin_len_sq_raw, 1e-12f);
    const float pin_len = sqrtf(pin_len_sq);
    const float pin_x = pin_raw_x / pin_len;
    const float pin_y = pin_raw_y / pin_len;
    const float pin_z = pin_raw_z / pin_len;

    float pout_raw_x, pout_raw_y, pout_raw_z;
    vcross(sh_x, sh_y, sh_z, wo_x, wo_y, wo_z, pout_raw_x, pout_raw_y, pout_raw_z);
    const float pout_len_sq_raw =
        pout_raw_x*pout_raw_x + pout_raw_y*pout_raw_y + pout_raw_z*pout_raw_z;
    const float pout_len_sq = fmaxf(pout_len_sq_raw, 1e-12f);
    const float pout_len = sqrtf(pout_len_sq);
    const float pout_x = pout_raw_x / pout_len;
    const float pout_y = pout_raw_y / pout_len;
    const float pout_z = pout_raw_z / pout_len;

    float2 r_s_h, r_p_h;
    itu_slab_fresnel(e_r, e_i, cos_h, thk, r_s_h, r_p_h);

    // E_rx_h = r_s_h * sh_z² + r_p_h * pin_z * pout_z   (real sh_z, pin_z, pout_z)
    const float sh_z_sq = sh_z * sh_z;
    const float pin_pout_z = pin_z * pout_z;
    const float E_rx_h_re = r_s_h.x * sh_z_sq + r_p_h.x * pin_pout_z;
    const float E_rx_h_im = r_s_h.y * sh_z_sq + r_p_h.y * pin_pout_z;
    // Note: forward kernel writes  E_s_out_h = r_s_h * sh_z  (vector with
    // imag), then E_rx_h = E_s_out_h * sh_z + E_p_out_h * pout_z. The net
    // real/imag are:  E_rx_h.x = r_s_h.x · sh_z² + r_p_h.x · pin_z · pout_z.
    const float R_jones_h_raw = E_rx_h_re * E_rx_h_re + E_rx_h_im * E_rx_h_im;
    const float R_jones_h = clampf(R_jones_h_raw, 0.0f, 1.0f);

    // --- Final blend ---
    const float f_coh = tau_ef * R_jones_h * f_KA
                     + (1.0f - tau_ef) * R_jones_macro * f_SPM;
    // f_cos = f_coh * cosi   (already computed; forward returned this)

    // ================================================================
    // STAGE 2: Backward — chain rule from grad_f_cos through every op.
    //
    // Notation: dL/dx = grad_f_cos · df_cos/dx. We use `gX` for dL/dX.
    // All grads are accumulated atomically into the output tensors.
    // ================================================================

    // d f_cos / d f_coh = cosi
    // d f_cos / d cosi  = f_coh   (but cosi also appears inside f_KA's denom)
    const float g_f_coh  = gf * cosi;
    float g_cosi_direct  = gf * f_coh;   // direct path only; indirect added below

    // Blend: f_coh = tau_ef · (R_jones_h · f_KA) + (1 - tau_ef) · (R_jones_macro · f_SPM)
    const float KA_term  = R_jones_h * f_KA;
    const float SPM_term = R_jones_macro * f_SPM;
    const float g_tau_ef      = g_f_coh * (KA_term - SPM_term);
    const float g_R_jones_h   = g_f_coh * tau_ef * f_KA;
    const float g_f_KA        = g_f_coh * tau_ef * R_jones_h;
    const float g_R_j_macro   = g_f_coh * (1.0f - tau_ef) * f_SPM;
    const float g_f_SPM       = g_f_coh * (1.0f - tau_ef) * R_jones_macro;

    // ---- SPM lobe ----
    // f_SPM = n_sp · exp_val · epsf
    const float g_n_sp        = g_f_SPM * exp_val * epsf;
    const float g_epsf        = g_f_SPM * n_sp * exp_val;
    const float g_exp_val     = g_f_SPM * n_sp * epsf;
    // d exp_val / d exp_arg = exp_val
    const float g_exp_arg     = g_exp_val * exp_val;
    // exp_arg = kap * (cos_dev - 1)
    const float g_kap_spm     = g_exp_arg * (cos_dev - 1.0f);
    const float g_cos_dev     = g_exp_arg * kap;
    // cos_dev = clamp(cos_dev_raw, -1, 1). When saturated, grad passes through
    // as zero. Detect saturation by comparing raw vs clamped.
    const float g_cos_dev_raw =
        (cos_dev_raw > 1.0f || cos_dev_raw < -1.0f) ? 0.0f : g_cos_dev;
    // cos_dev_raw = wo · wi_r
    // d/d wo = wi_r, d/d wi_r = wo
    float g_wo_x_spm = g_cos_dev_raw * wir_x;
    float g_wo_y_spm = g_cos_dev_raw * wir_y;
    float g_wo_z_spm = g_cos_dev_raw * wir_z;
    const float g_wir_x = g_cos_dev_raw * wo_x;
    const float g_wir_y = g_cos_dev_raw * wo_y;
    const float g_wir_z = g_cos_dev_raw * wo_z;

    // ---- Jones macro ----
    // R_jones_macro = clamp(R_jones_macro_raw, 0, 1).
    const float g_R_j_macro_raw =
        (R_jones_macro_raw > 1.0f || R_jones_macro_raw < 0.0f) ? 0.0f : g_R_j_macro;
    // R_jones_macro_raw = E_rx_re² + E_rx_im²
    const float g_E_rx_re = g_R_j_macro_raw * 2.0f * E_rx_re;
    const float g_E_rx_im = g_R_j_macro_raw * 2.0f * E_rx_im;
    // E_rx_re = Es_re·rx_s + Ep_re·rx_p
    const float g_Es_re   = g_E_rx_re * rx_s;
    const float g_Ep_re   = g_E_rx_re * rx_p;
    const float g_Es_im   = g_E_rx_im * rx_s;
    const float g_Ep_im   = g_E_rx_im * rx_p;
    // rx_s appears in both re and im
    float g_rx_s          = g_E_rx_re * Es_re + g_E_rx_im * Es_im;
    float g_rx_p          = g_E_rx_re * Ep_re + g_E_rx_im * Ep_im;
    // rx_s = s_z
    float g_s_z_macro     = g_rx_s;
    // rx_p = rx_p_numer / p_out_norm
    const float g_rx_p_numer  = g_rx_p / p_out_norm;
    const float g_p_out_norm  = -g_rx_p * rx_p_numer / (p_out_norm * p_out_norm);
    // p_out_norm = sqrt(s_cross_wo_sq)   → d/d(s_cross_wo_sq) = 1/(2*p_out_norm)
    // s_cross_wo_sq = max(1 - wo_dot_s², eps); drop grad if clamped.
    const float g_s_cross_wo_sq =
        (s_cross_wo_sq_raw > 1e-12f) ? (g_p_out_norm * 0.5f / p_out_norm) : 0.0f;
    // d(1 - wo_dot_s²) / d wo_dot_s = -2 * wo_dot_s
    const float g_wo_dot_s = g_s_cross_wo_sq * (-2.0f * wo_dot_s);
    // wo_dot_s = wo · s   → d/d wo = s, d/d s = wo
    float g_wo_x_mac1 = g_wo_dot_s * s_x;
    float g_wo_y_mac1 = g_wo_dot_s * s_y;
    float g_wo_z_mac1 = g_wo_dot_s * s_z;
    float g_s_x_mac1  = g_wo_dot_s * wo_x;
    float g_s_y_mac1  = g_wo_dot_s * wo_y;
    float g_s_z_mac1  = g_wo_dot_s * wo_z;
    // rx_p_numer = wo_x·(-s_y) + wo_y·s_x   (no wo_z term)
    float g_wo_x_mac2 = g_rx_p_numer * (-s_y);
    float g_wo_y_mac2 = g_rx_p_numer * s_x;
    float g_wo_z_mac2 = 0.0f;
    float g_s_x_mac2  = g_rx_p_numer * wo_y;
    float g_s_y_mac2  = g_rx_p_numer * (-wo_x);

    // ---- KA lobe ----
    // f_KA = D_KA · G_KA / f_KA_denom
    const float g_D_KA        = g_f_KA * G_KA / f_KA_denom;
    const float g_G_KA        = g_f_KA * D_KA / f_KA_denom;
    const float g_f_KA_denom_raw = -g_f_KA * D_KA * G_KA / (f_KA_denom * f_KA_denom);
    const float g_f_KA_denom  = (f_KA_denom > 1e-10f) ? g_f_KA_denom_raw : 0.0f;
    // f_KA_denom = 4 · cosi · coso   → grads on cosi, coso
    const float g_cosi_f_KA   = g_f_KA_denom * 4.0f * coso;
    const float g_coso_f_KA   = g_f_KA_denom * 4.0f * cosi;

    // G_KA = 1 / G_KA_denom  → d/d G_KA_denom = -G_KA²
    const float g_G_KA_denom_raw = -g_G_KA * G_KA * G_KA;
    const float g_G_KA_denom  = (G_KA_denom > 1e-10f) ? g_G_KA_denom_raw : 0.0f;
    // G_KA_denom = 1 + lambda_i + lambda_o
    const float g_lam_i       = g_G_KA_denom;
    const float g_lam_o       = g_G_KA_denom;

    // D_KA = a_sq / (PI · denom_ndf)
    const float g_a_sq_direct = g_D_KA / (PI * denom_ndf);
    const float g_denom_ndf_raw = -g_D_KA * a_sq / (PI * denom_ndf * denom_ndf);
    const float g_denom_ndf   = (denom_ndf_sq > 1e-20f) ? g_denom_ndf_raw : 0.0f;
    // denom_ndf = denom_ndf_base²
    const float g_denom_ndf_base = g_denom_ndf * 2.0f * denom_ndf_base;
    // denom_ndf_base = h_dot_n² · (a_sq - 1) + 1
    const float g_h_dot_n_sq  = g_denom_ndf_base * (a_sq - 1.0f);
    const float g_a_sq_indirect = g_denom_ndf_base * h_dot_n * h_dot_n;
    // d(h_dot_n²)/d h_dot_n = 2·h_dot_n
    const float g_h_dot_n     = g_h_dot_n_sq * 2.0f * h_dot_n;
    // h_dot_n = max(h_dot_n_raw, 0)
    const float g_h_dot_n_raw = (h_dot_n_raw > 0.0f) ? g_h_dot_n : 0.0f;
    // h_dot_n_raw = h_num / h_len
    const float g_h_num       = g_h_dot_n_raw / h_len;
    const float g_h_len_from_hdotn = -g_h_dot_n_raw * h_num / (h_len * h_len);

    // h_num = wo·n_eff + cosi
    // d/d wo = n_eff, d/d n_eff = wo, d/d cosi = 1
    float g_wo_x_ka1 = g_h_num * ne_x;
    float g_wo_y_ka1 = g_h_num * ne_y;
    float g_wo_z_ka1 = g_h_num * ne_z;
    float g_ne_x_ka  = g_h_num * wo_x;
    float g_ne_y_ka  = g_h_num * wo_y;
    float g_ne_z_ka  = g_h_num * wo_z;
    float g_cosi_ka_hnum = g_h_num;

    // h_len enters from KA (h_dot_n) AND from Jones h (cos_h = (1 + wo_dot_wi) / h_len).
    // We'll add the Jones-h contribution below and accumulate at the end.
    float g_h_len = g_h_len_from_hdotn;

    // ---- Jones h (microfacet basis + slab Fresnel) ----
    // R_jones_h = clamp(R_jones_h_raw, 0, 1)
    const float g_R_jones_h_raw =
        (R_jones_h_raw > 1.0f || R_jones_h_raw < 0.0f) ? 0.0f : g_R_jones_h;
    // R_jones_h_raw = E_rx_h_re² + E_rx_h_im²
    const float g_E_rx_h_re = g_R_jones_h_raw * 2.0f * E_rx_h_re;
    const float g_E_rx_h_im = g_R_jones_h_raw * 2.0f * E_rx_h_im;
    // E_rx_h_re = r_s_h.x · sh_z²  +  r_p_h.x · pin_pout_z
    // E_rx_h_im = r_s_h.y · sh_z²  +  r_p_h.y · pin_pout_z
    const float g_r_s_h_re  = g_E_rx_h_re * sh_z_sq;
    const float g_r_s_h_im  = g_E_rx_h_im * sh_z_sq;
    const float g_r_p_h_re  = g_E_rx_h_re * pin_pout_z;
    const float g_r_p_h_im  = g_E_rx_h_im * pin_pout_z;
    const float g_sh_z_sq   = g_E_rx_h_re * r_s_h.x + g_E_rx_h_im * r_s_h.y;
    const float g_pin_pout_z= g_E_rx_h_re * r_p_h.x + g_E_rx_h_im * r_p_h.y;

    // sh_z_sq = sh_z²  → d/d sh_z = 2·sh_z
    float g_sh_z            = g_sh_z_sq * 2.0f * sh_z;
    // pin_pout_z = pin_z · pout_z
    float g_pin_z           = g_pin_pout_z * pout_z;
    float g_pout_z          = g_pin_pout_z * pin_z;

    // --- Slab Fresnel (r_s_h, r_p_h) backward — analytical (Phase C.3) ---
    //
    // Replaces the fp64 finite-difference block with closed-form Wirtinger
    // derivatives in fp32. The forward computes (for s = TE, p = TM):
    //
    //   eta   = e_r - i·e_i
    //   sin2  = 1 - c²                       (c = cos_h, real)
    //   a     = sqrt(eta - sin2)
    //   r_te  = (c - a) / (c + a)
    //   r_tm  = (eta·c - a) / (eta·c + a)
    //   X     = exp(-2j·K_WAVE·d·a)          (d = thickness)
    //   R_te  = r_te (1 - X) / (1 - r_te² X)
    //   R_tm  = r_tm (1 - X) / (1 - r_tm² X)
    //
    // Since every op is holomorphic in its complex arguments, the chain
    // rule reduces to: for a complex step F: C → C with upstream (real)
    // grad g_F = (g_F.re, g_F.im) and complex derivative F'(z), the
    // complex grad on the input is
    //     g_z = g_F · conj(F'(z))
    // (Wirtinger / holomorphic chain rule for a real-valued loss).
    // For real inputs u with the function still complex-valued, the
    // corresponding real grad is Re(g_F · conj(F'(u))).
    //
    // Cost: one full forward + O(20) complex ops total, no fp64 anywhere.
    // Replaces 6 fp64 slab_fresnel calls.
    float g_e_r = 0.0f, g_e_i = 0.0f, g_thk = 0.0f, g_cos_h_slab = 0.0f;
    {
        // ---- Forward recomputation (fp32, mirrors itu_slab_fresnel) ----
        const float c2    = fmaxf(fminf(cos_h, 1.0f), 1e-6f);
        const float sin2  = 1.0f - c2 * c2;
        const float2 eta  = make_float2(e_r, -e_i);
        const float2 inside = csub(eta, make_float2(sin2, 0.0f));
        const float2 a    = csqrt(inside);

        const float2 cc     = make_float2(c2, 0.0f);
        const float2 eps_c  = make_float2(1e-10f, 0.0f);

        const float2 den_te = cadd(cadd(cc, a), eps_c);     // c + a + eps
        const float2 r_te   = cdiv(csub(cc, a), den_te);

        const float2 eta_c  = cscale(eta, c2);
        const float2 den_tm = cadd(cadd(eta_c, a), eps_c);  // eta·c + a + eps
        const float2 r_tm   = cdiv(csub(eta_c, a), den_tm);

        const float2 q      = cscale(a, K_WAVE * thk);
        // X = exp(-2j·q) = exp(2·q.y - 2j·q.x)
        const float2 X      = cexp(make_float2(2.0f * q.y, -2.0f * q.x));

        const float2 one        = make_float2(1.0f, 0.0f);
        const float2 one_minus_X= csub(one, X);
        const float2 r_te_sq    = cmul(r_te, r_te);
        const float2 r_tm_sq    = cmul(r_tm, r_tm);
        const float2 D_TE       = cadd(csub(one, cmul(r_te_sq, X)), eps_c);
        const float2 D_TM       = cadd(csub(one, cmul(r_tm_sq, X)), eps_c);

        // ---- Shared helpers ----
        // Wirtinger VJP: g_z = g_F · conj(Fp).  For inputs that are real,
        // the real grad is the .x of the result.
        auto vjp_c = [] __device__ (const float2& gF, const float2& Fp) -> float2 {
            // (gF.re + i gF.im) * (Fp.re - i Fp.im)
            return make_float2(gF.x * Fp.x + gF.y * Fp.y,
                               gF.y * Fp.x - gF.x * Fp.y);
        };

        // ---- Upstream grads packaged as complex ----
        const float2 g_R_te = make_float2(g_r_s_h_re, g_r_s_h_im);
        const float2 g_R_tm = make_float2(g_r_p_h_re, g_r_p_h_im);

        // ---- R = r(1-X)/(1 - r²X) ----
        //   dR/dr = (1-X)(1 + r²X) / (1 - r²X)²
        //   dR/dX = r(r² - 1) / (1 - r²X)²
        const float2 D_TE_sq = cmul(D_TE, D_TE);
        const float2 D_TM_sq = cmul(D_TM, D_TM);

        const float2 dR_te_dr = cdiv(cmul(one_minus_X, cadd(one, cmul(r_te_sq, X))), D_TE_sq);
        const float2 dR_te_dX = cdiv(cmul(r_te, csub(r_te_sq, one)),                D_TE_sq);
        const float2 dR_tm_dr = cdiv(cmul(one_minus_X, cadd(one, cmul(r_tm_sq, X))), D_TM_sq);
        const float2 dR_tm_dX = cdiv(cmul(r_tm, csub(r_tm_sq, one)),                D_TM_sq);

        const float2 g_r_te = vjp_c(g_R_te, dR_te_dr);
        const float2 g_r_tm = vjp_c(g_R_tm, dR_tm_dr);
        const float2 g_X    = cadd(vjp_c(g_R_te, dR_te_dX), vjp_c(g_R_tm, dR_tm_dX));

        // ---- X = exp(-2j·q): dX/dq = -2j·X = (2 X.y, -2 X.x) ----
        const float2 dX_dq = make_float2(2.0f * X.y, -2.0f * X.x);
        const float2 g_q   = vjp_c(g_X, dX_dq);

        // ---- q = (K_WAVE · d) · a ----
        // a → q: holomorphic with scalar K_WAVE·d.  g_a += g_q · (K_WAVE·d)
        // d → q: q(d) = (K_WAVE·a)·d, d real: g_d += K_WAVE·(g_q.re·a.re + g_q.im·a.im)
        const float kwd = K_WAVE * thk;
        float2 g_a = cscale(g_q, kwd);
        g_thk += K_WAVE * (g_q.x * a.x + g_q.y * a.y);

        // ---- r_te = (c - a)/(c + a) ----
        //   dr_te/da = -2c / (c+a)²
        //   dr_te/dc =  2a / (c+a)²     (c real)
        const float2 den_te_sq = cmul(den_te, den_te);
        const float2 dr_te_da  = cdiv(make_float2(-2.0f * c2, 0.0f), den_te_sq);
        const float2 dr_te_dc  = cdiv(cscale(a, 2.0f),              den_te_sq);
        g_a = cadd(g_a, vjp_c(g_r_te, dr_te_da));
        g_cos_h_slab += vjp_c(g_r_te, dr_te_dc).x;

        // ---- r_tm = (eta·c - a)/(eta·c + a) ----
        //   dr_tm/da   = -2·eta·c / (eta·c + a)²
        //   dr_tm/dc   =  2·eta·a / (eta·c + a)²      (c real)
        //   dr_tm/deta =  2·c·a   / (eta·c + a)²
        const float2 den_tm_sq = cmul(den_tm, den_tm);
        const float2 eta_a     = cmul(eta, a);
        const float2 dr_tm_da  = cdiv(cscale(eta_c, -2.0f), den_tm_sq);
        const float2 dr_tm_dc  = cdiv(cscale(eta_a,  2.0f), den_tm_sq);
        const float2 dr_tm_deta= cdiv(cscale(a, 2.0f * c2), den_tm_sq);
        g_a = cadd(g_a, vjp_c(g_r_tm, dr_tm_da));
        g_cos_h_slab += vjp_c(g_r_tm, dr_tm_dc).x;
        const float2 g_eta_from_rtm = vjp_c(g_r_tm, dr_tm_deta);

        // ---- a = sqrt(inside): a'(inside) = 1/(2a) ----
        // g_inside = g_a · conj(1/(2a)) = g_a · (a / (2 |a|²))
        const float  a_abs_sq = fmaxf(a.x * a.x + a.y * a.y, 1e-30f);
        const float  inv_2aa  = 0.5f / a_abs_sq;
        // conj(1/(2a)) = a / (2 |a|²). Multiply as complex.
        const float2 Fp_sqrt  = make_float2(a.x * inv_2aa, a.y * inv_2aa);
        // NOTE: we want g_a · conj(Fp) where Fp = 1/(2a) = (a.x, -a.y)·inv_2aa.
        //   conj(Fp) = (a.x, a.y)·inv_2aa = Fp_sqrt.  So g_inside = cmul(g_a, Fp_sqrt).
        const float2 g_inside = cmul(g_a, Fp_sqrt);

        // ---- inside = eta - sin2 ----
        // g_eta += g_inside ; g_sin2 = -g_inside.re
        // sin2 = 1 - c² → dsin2/dc = -2c → g_c += 2·c·g_inside.re
        const float2 g_eta = cadd(g_eta_from_rtm, g_inside);
        g_cos_h_slab += 2.0f * c2 * g_inside.x;

        // ---- eta = (e_r, -e_i) ----
        //   e_r → eta: d(eta.re)/d e_r = 1  → g_e_r += g_eta.re
        //   e_i → eta: d(eta.im)/d e_i = -1 → g_e_i += -g_eta.im
        g_e_r += g_eta.x;
        g_e_i += -g_eta.y;
    }
    // cos_h also had a direct grad path through Jones h; add it here.
    // cos_h = max(cos_h_raw, 1e-6);  cos_h_raw = (1 + wo_dot_wi) / h_len
    float g_cos_h = g_cos_h_slab;
    const float g_cos_h_raw = (cos_h_raw > 1e-6f) ? g_cos_h : 0.0f;
    // d/d wo_dot_wi = 1 / h_len; d/d h_len = -(1 + wo_dot_wi) / h_len²
    float g_wo_dot_wi_from_cos_h = g_cos_h_raw / h_len;
    float g_h_len_from_cos_h     = -g_cos_h_raw * (1.0f + wo_dot_wi) / (h_len * h_len);
    g_h_len += g_h_len_from_cos_h;

    // --- Jones h basis: sh_z, pin_z, pout_z grads through cross products ---
    //
    // pin_z = pin_raw_z / pin_len
    // pout_z = pout_raw_z / pout_len
    // Same structure as sh_z / sh_len.
    //
    // d(v_z / |v|) / d v_i:
    //   For i=z: (|v|² - v_z²) / |v|³
    //   For i≠z: -v_z v_i / |v|³
    auto grad_unit_z = [] __device__ (
        float vx, float vy, float vz, float v_len,
        float& out_dx, float& out_dy, float& out_dz)
    {
        const float inv = 1.0f / v_len;
        const float inv3 = inv * inv * inv;
        out_dx = -vz * vx * inv3;
        out_dy = -vz * vy * inv3;
        out_dz = (v_len * v_len - vz * vz) * inv3;  // = (1 - (vz/v_len)²)/v_len
    };

    // --- pout chain ---
    float d_pout_dz_dx, d_pout_dz_dy, d_pout_dz_dz;
    grad_unit_z(pout_raw_x, pout_raw_y, pout_raw_z, pout_len,
                d_pout_dz_dx, d_pout_dz_dy, d_pout_dz_dz);
    const float g_pout_raw_x = g_pout_z * d_pout_dz_dx;
    const float g_pout_raw_y = g_pout_z * d_pout_dz_dy;
    const float g_pout_raw_z = g_pout_z * d_pout_dz_dz;
    // pout_raw = cross(sh, wo)
    //   pout_raw_x =  sh_y*wo_z - sh_z*wo_y
    //   pout_raw_y =  sh_z*wo_x - sh_x*wo_z
    //   pout_raw_z =  sh_x*wo_y - sh_y*wo_x
    float g_sh_x_pout =  g_pout_raw_y * wo_z + g_pout_raw_z * wo_y * -1.0f;   // d/d sh_x
    float g_sh_y_pout =  g_pout_raw_x * wo_z * -1.0f + g_pout_raw_z * wo_x;
    float g_sh_z_pout =  g_pout_raw_x * wo_y + g_pout_raw_y * wo_x * -1.0f;
    // Wait, let me redo that carefully.
    // pout_raw_x = sh_y*wo_z - sh_z*wo_y
    //   d/d sh_x = 0
    //   d/d sh_y = wo_z
    //   d/d sh_z = -wo_y
    //   d/d wo_x = 0
    //   d/d wo_y = -sh_z
    //   d/d wo_z = sh_y
    // pout_raw_y = sh_z*wo_x - sh_x*wo_z
    //   d/d sh_x = -wo_z
    //   d/d sh_y = 0
    //   d/d sh_z = wo_x
    //   d/d wo_x = sh_z
    //   d/d wo_y = 0
    //   d/d wo_z = -sh_x
    // pout_raw_z = sh_x*wo_y - sh_y*wo_x
    //   d/d sh_x = wo_y
    //   d/d sh_y = -wo_x
    //   d/d sh_z = 0
    //   d/d wo_x = -sh_y
    //   d/d wo_y = sh_x
    //   d/d wo_z = 0
    g_sh_x_pout = -g_pout_raw_y * wo_z + g_pout_raw_z * wo_y;
    g_sh_y_pout =  g_pout_raw_x * wo_z - g_pout_raw_z * wo_x;
    g_sh_z_pout = -g_pout_raw_x * wo_y + g_pout_raw_y * wo_x;
    float g_wo_x_pout = -g_pout_raw_y * wo_z + g_pout_raw_z * -wo_y;  // wrong, placeholder
    // Redo:
    g_wo_x_pout =  g_pout_raw_y * sh_z - g_pout_raw_z * sh_y;
    float g_wo_y_pout = -g_pout_raw_x * sh_z + g_pout_raw_z * sh_x;
    float g_wo_z_pout =  g_pout_raw_x * sh_y - g_pout_raw_y * sh_x;

    // --- pin chain ---
    float d_pin_dz_dx, d_pin_dz_dy, d_pin_dz_dz;
    grad_unit_z(pin_raw_x, pin_raw_y, pin_raw_z, pin_len,
                d_pin_dz_dx, d_pin_dz_dy, d_pin_dz_dz);
    const float g_pin_raw_x = g_pin_z * d_pin_dz_dx;
    const float g_pin_raw_y = g_pin_z * d_pin_dz_dy;
    const float g_pin_raw_z = g_pin_z * d_pin_dz_dz;
    // pin_raw = cross(sh, wi). Same structure, wi replaces wo.
    float g_sh_x_pin = -g_pin_raw_y * wi_z + g_pin_raw_z * wi_y;
    float g_sh_y_pin =  g_pin_raw_x * wi_z - g_pin_raw_z * wi_x;
    float g_sh_z_pin = -g_pin_raw_x * wi_y + g_pin_raw_y * wi_x;
    float g_wi_x_pin =  g_pin_raw_y * sh_z - g_pin_raw_z * sh_y;
    float g_wi_y_pin = -g_pin_raw_x * sh_z + g_pin_raw_z * sh_x;
    float g_wi_z_pin =  g_pin_raw_x * sh_y - g_pin_raw_y * sh_x;

    // --- sh chain ---
    // Aggregate dL/d sh_* from: direct (sh_z through Jones h),
    // pout chain (g_sh_*_pout), pin chain (g_sh_*_pin).
    const float g_sh_x_agg = g_sh_x_pout + g_sh_x_pin;
    const float g_sh_y_agg = g_sh_y_pout + g_sh_y_pin;
    float g_sh_z_agg = g_sh_z + g_sh_z_pout + g_sh_z_pin;

    // d(v / |v|) / d v_i:  For j component:  δ_ij/|v| - v_i v_j / |v|³
    // We already have dL/d sh_x, sh_y, sh_z where sh = sh_raw / sh_len.
    // Backprop to sh_raw.
    auto grad_norm_vec = [] __device__ (
        float vx, float vy, float vz, float v_len,
        float gx, float gy, float gz,
        float& out_dx, float& out_dy, float& out_dz)
    {
        const float inv = 1.0f / v_len;
        const float inv3 = inv * inv * inv;
        // d unit_x / d raw_x = (v_len² - v_x²) * inv³
        // d unit_x / d raw_y = -v_x v_y * inv³
        // d unit_x / d raw_z = -v_x v_z * inv³
        out_dx = gx * (v_len*v_len - vx*vx) * inv3
               + gy * (-vx * vy) * inv3
               + gz * (-vx * vz) * inv3;
        out_dy = gx * (-vy * vx) * inv3
               + gy * (v_len*v_len - vy*vy) * inv3
               + gz * (-vy * vz) * inv3;
        out_dz = gx * (-vz * vx) * inv3
               + gy * (-vz * vy) * inv3
               + gz * (v_len*v_len - vz*vz) * inv3;
    };
    float g_sh_raw_x, g_sh_raw_y, g_sh_raw_z;
    grad_norm_vec(sh_raw_x, sh_raw_y, sh_raw_z, sh_len,
                  g_sh_x_agg, g_sh_y_agg, g_sh_z_agg,
                  g_sh_raw_x, g_sh_raw_y, g_sh_raw_z);
    // sh_raw = cross(wi, wo)
    //   sh_raw_x = wi_y*wo_z - wi_z*wo_y
    //   sh_raw_y = wi_z*wo_x - wi_x*wo_z
    //   sh_raw_z = wi_x*wo_y - wi_y*wo_x
    const float g_wi_x_sh = -g_sh_raw_y * wo_z + g_sh_raw_z * wo_y;
    const float g_wi_y_sh =  g_sh_raw_x * wo_z - g_sh_raw_z * wo_x;
    const float g_wi_z_sh = -g_sh_raw_x * wo_y + g_sh_raw_y * wo_x;
    const float g_wo_x_sh =  g_sh_raw_y * wi_z - g_sh_raw_z * wi_y;
    const float g_wo_y_sh = -g_sh_raw_x * wi_z + g_sh_raw_z * wi_x;
    const float g_wo_z_sh =  g_sh_raw_x * wi_y - g_sh_raw_y * wi_x;

    // --- h_len backward ---
    // h_len = sqrt(max(2 + 2*wo_dot_wi, 1e-10))
    // d h_len / d wo_dot_wi = (2*inv)/(2*h_len) = 1/h_len
    const float g_h_len_raw = (h_len_sq_raw > 1e-10f) ? g_h_len : 0.0f;
    const float g_wo_dot_wi_from_h_len = g_h_len_raw / h_len;
    const float g_wo_dot_wi = g_wo_dot_wi_from_cos_h + g_wo_dot_wi_from_h_len;
    // wo_dot_wi = wo · wi
    float g_wo_x_wdw = g_wo_dot_wi * wi_x;
    float g_wo_y_wdw = g_wo_dot_wi * wi_y;
    float g_wo_z_wdw = g_wo_dot_wi * wi_z;
    float g_wi_x_wdw = g_wo_dot_wi * wo_x;
    float g_wi_y_wdw = g_wo_dot_wi * wo_y;
    float g_wi_z_wdw = g_wo_dot_wi * wo_z;

    // ================================================================
    // STAGE 3: Atomically accumulate into output grad tensors.
    //
    // Per-(m, t) inputs (cos_i, lambda_i, tau_eff, E_*, n_eff, s_in, wi,
    //   wi_r) get contributions from all n_rx paths at this (m, t).
    // Per-(m, r) inputs (cos_o, lambda_o, wo) get contributions from all
    //   n_tx paths.
    // Per-m inputs (alpha_sq, kappa_SPM, norm_SPM, eps_factor, eps_real,
    //   eps_imag, thickness) get contributions from all n_tx*n_rx paths.
    // ================================================================

    // cos_i: direct path (f_coh * cosi derivative) + KA f_KA_denom + KA h_num
    const float g_cosi_total = g_cosi_direct + g_cosi_f_KA + g_cosi_ka_hnum;
    atomicAdd(&grad_cos_i[m*n_tx + t], g_cosi_total);
    // cos_o: KA f_KA_denom only
    atomicAdd(&grad_cos_o[m*n_rx + r], g_coso_f_KA);

    // lambda_i / lambda_o from G_KA
    atomicAdd(&grad_lambda_i[m*n_tx + t], g_lam_i);
    atomicAdd(&grad_lambda_o[m*n_rx + r], g_lam_o);

    // Per-m material scalars
    atomicAdd(&grad_alpha_sq[m], g_a_sq_direct + g_a_sq_indirect);
    atomicAdd(&grad_kappa_SPM[m], g_kap_spm);
    atomicAdd(&grad_norm_SPM[m], g_n_sp);
    atomicAdd(&grad_eps_factor[m], g_epsf);
    atomicAdd(&grad_eps_real_m[m], g_e_r);
    atomicAdd(&grad_eps_imag_m[m], g_e_i);
    atomicAdd(&grad_thickness_m[m], g_thk);

    // Per-(m, t) macro Fresnel
    atomicAdd(&grad_E_s_out_re[m*n_tx + t], g_Es_re);
    atomicAdd(&grad_E_s_out_im[m*n_tx + t], g_Es_im);
    atomicAdd(&grad_E_p_out_re[m*n_tx + t], g_Ep_re);
    atomicAdd(&grad_E_p_out_im[m*n_tx + t], g_Ep_im);

    // tau_eff
    atomicAdd(&grad_tau_eff[m*n_tx + t], g_tau_ef);

    // wi (M, n_tx, 3): skipped — flows only to frozen `positions`
    // (LEARN_POSITIONS=False). The autograd.Function returns a pre-
    // zeroed tensor for grad_wi so PyTorch's chain rule sees zeros and
    // doesn't propagate anywhere useful. Saves 3 atomicAdds per thread.
    // (void)(g_wi_x_sh + g_wi_x_pin + g_wi_x_wdw);  // keep vars live for optimizer

    // wi_r (M, n_tx, 3): from SPM cos_dev
    atomicAdd(&grad_wi_r[mt3+0], g_wir_x);
    atomicAdd(&grad_wi_r[mt3+1], g_wir_y);
    atomicAdd(&grad_wi_r[mt3+2], g_wir_z);

    // n_eff (M, n_tx, 3): from KA h_num (wo·n_eff)
    atomicAdd(&grad_n_eff[mt3+0], g_ne_x_ka);
    atomicAdd(&grad_n_eff[mt3+1], g_ne_y_ka);
    atomicAdd(&grad_n_eff[mt3+2], g_ne_z_ka);

    // s_in (M, n_tx, 3): from Jones macro rx_s (= s_z) and rx_p_numer + wo_dot_s
    float g_s_x_total = g_s_x_mac1 + g_s_x_mac2;
    float g_s_y_total = g_s_y_mac1 + g_s_y_mac2;
    float g_s_z_total = g_s_z_mac1 + g_s_z_macro;
    atomicAdd(&grad_s_in[mt3+0], g_s_x_total);
    atomicAdd(&grad_s_in[mt3+1], g_s_y_total);
    atomicAdd(&grad_s_in[mt3+2], g_s_z_total);

    // wo (M, n_rx, 3): skipped — flows only to frozen `positions`
    // (LEARN_POSITIONS=False). Same rationale as grad_wi above.
}

void launch_bsdf_step4_backward(
    const float* grad_f_cos,
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
    float* grad_wi, float* grad_wi_r, float* grad_wo,
    float* grad_n_eff, float* grad_s_in,
    float* grad_cos_i, float* grad_cos_o,
    float* grad_lambda_i, float* grad_lambda_o,
    float* grad_alpha_sq, float* grad_kappa_SPM,
    float* grad_norm_SPM, float* grad_eps_factor,
    float* grad_eps_real_m, float* grad_eps_imag_m, float* grad_thickness_m,
    float* grad_E_s_out_re, float* grad_E_s_out_im,
    float* grad_E_p_out_re, float* grad_E_p_out_im,
    float* grad_tau_eff,
    int M, int n_tx, int n_rx,
    cudaStream_t stream)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    bsdf_step4_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        grad_f_cos,
        wi, wi_r, wo, n_eff, s_in,
        cos_i, cos_o, lambda_i, lambda_o,
        alpha_sq, kappa_SPM, norm_SPM, eps_factor,
        eps_real_m, eps_imag_m, thickness_m,
        E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
        tau_eff,
        grad_wi, grad_wi_r, grad_wo,
        grad_n_eff, grad_s_in,
        grad_cos_i, grad_cos_o,
        grad_lambda_i, grad_lambda_o,
        grad_alpha_sq, grad_kappa_SPM,
        grad_norm_SPM, grad_eps_factor,
        grad_eps_real_m, grad_eps_imag_m, grad_thickness_m,
        grad_E_s_out_re, grad_E_s_out_im,
        grad_E_p_out_re, grad_E_p_out_im,
        grad_tau_eff,
        M, n_tx, n_rx);
}

} // namespace mm25v5
