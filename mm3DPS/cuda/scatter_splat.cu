// Phase D: range-profile scatter splat kernels.
//
// Two kernels live here:
//
//   1. scatter_splat_kernel (Phase A, unchanged): basic atomic scatter
//      used by the unit test, and as the inner op of the generic
//      scatter_splat Python binding. Takes precomputed contrib tensors.
//
//   2. step5_fused_forward/backward_kernel (Phase D): fused
//      range-profile splatting. Takes (w_full, phi_carrier, n_peak,
//      psf_table), computes the carrier phasor, walks the SPREAD
//      offsets, looks up the PSF inline, and scatters directly into
//      rp_real/rp_imag. Avoids materializing the 540 MB contrib_real/
//      contrib_imag/flat_idx tensors that the PyTorch path allocates
//      every iteration. Backward gathers grad_rp at each offset and
//      accumulates per-path into grad_w (no atomics — each thread
//      writes its own output slot).

#include <cuda_runtime.h>
#include <torch/types.h>

#include "utils.cuh"

namespace mm25v5 {

// ---------------------------------------------------------------------------
// Phase A: basic atomic scatter (contrib tensors precomputed)
// ---------------------------------------------------------------------------

__global__ void scatter_splat_kernel_phaseA(
    const float* __restrict__ contrib_real,
    const float* __restrict__ contrib_imag,
    const int64_t* __restrict__ flat_idx,
    float* __restrict__ rp_real,
    float* __restrict__ rp_imag,
    int64_t n_items,
    int64_t out_size)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n_items) return;
    int64_t idx = flat_idx[i];
    if (idx < 0 || idx >= out_size) return;
    atomicAdd(rp_real + idx, contrib_real[i]);
    atomicAdd(rp_imag + idx, contrib_imag[i]);
}

void launch_scatter_splat(
    const float* contrib_real, const float* contrib_imag,
    const int64_t* flat_idx,
    float* rp_real, float* rp_imag,
    int64_t n_items, int64_t out_size, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (int)((n_items + BLOCK - 1) / BLOCK);
    if (grid > 0) {
        scatter_splat_kernel_phaseA<<<grid, BLOCK, 0, stream>>>(
            contrib_real, contrib_imag, flat_idx,
            rp_real, rp_imag, n_items, out_size);
    }
}

// ---------------------------------------------------------------------------
// Phase D: fused Step-5 splat forward
// ---------------------------------------------------------------------------

__global__ void step5_fused_forward_kernel(
    const float* __restrict__ w_full,       // (M, n_tx, n_rx)
    const float* __restrict__ phi_carrier,  // (M, n_tx, n_rx)
    const float* __restrict__ n_peak,       // (M, n_tx, n_rx)
    const float* __restrict__ psf_real,     // (spread, n_grid)
    const float* __restrict__ psf_imag,     // (spread, n_grid)
    float* __restrict__ rp_real,            // (n_tx, n_rx, K)
    float* __restrict__ rp_imag,            // (n_tx, n_rx, K)
    int M, int n_tx, int n_rx, int K,
    int spread, int n_grid,
    float w_threshold)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const float w = w_full[idx];
    if (w < w_threshold) return;  // culled path

    const int r = (int)(idx % n_rx);
    const int t = (int)((idx / n_rx) % n_tx);
    // m not needed — output is per-(t, r, bin)

    const float phi = phi_carrier[idx];
    const float np_peak = n_peak[idx];

    const int n_floor = (int)floorf(np_peak);
    const float n_frac = np_peak - (float)n_floor;

    float s, c;
    sincosf(phi, &s, &c);
    const float carrier_real = w * c;
    const float carrier_imag = w * s;

    // PSF table fractional index + linear interp weight
    const float idx_f = n_frac * (float)n_grid;
    int idx_lo = (int)floorf(idx_f);
    if (idx_lo < 0)         idx_lo = 0;
    if (idx_lo >= n_grid)   idx_lo = n_grid - 1;
    int idx_hi = idx_lo + 1;
    if (idx_hi >= n_grid)   idx_hi = n_grid - 1;
    const float frac = idx_f - (float)idx_lo;
    const float inv_frac = 1.0f - frac;

    const int base = (t * n_rx + r) * K;
    const int half_spread = spread >> 1;

    #pragma unroll
    for (int d = 0; d < 15; ++d) {
        if (d >= spread) break;
        const int psf_off = d * n_grid;
        const float pr_lo = psf_real[psf_off + idx_lo];
        const float pr_hi = psf_real[psf_off + idx_hi];
        const float pi_lo = psf_imag[psf_off + idx_lo];
        const float pi_hi = psf_imag[psf_off + idx_hi];
        const float psf_r = pr_lo * inv_frac + pr_hi * frac;
        const float psf_i = pi_lo * inv_frac + pi_hi * frac;

        const float contrib_re = carrier_real * psf_r - carrier_imag * psf_i;
        const float contrib_im = carrier_real * psf_i + carrier_imag * psf_r;

        int bin = (n_floor + (d - half_spread)) % K;
        if (bin < 0) bin += K;

        atomicAdd(&rp_real[base + bin], contrib_re);
        atomicAdd(&rp_imag[base + bin], contrib_im);
    }
}

void launch_step5_fused_forward(
    const float* w_full, const float* phi_carrier, const float* n_peak,
    const float* psf_real, const float* psf_imag,
    float* rp_real, float* rp_imag,
    int M, int n_tx, int n_rx, int K,
    int spread, int n_grid, float w_threshold,
    cudaStream_t stream)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    step5_fused_forward_kernel<<<grid, BLOCK, 0, stream>>>(
        w_full, phi_carrier, n_peak, psf_real, psf_imag,
        rp_real, rp_imag, M, n_tx, n_rx, K, spread, n_grid, w_threshold);
}

// ---------------------------------------------------------------------------
// Phase D: fused Step-5 splat backward
// ---------------------------------------------------------------------------
//
// For the backward we only need grad_w_full: phi_carrier and n_peak are
// always detached in the Python Step-5 path (`detach_phase=True` and
// n_peak is explicitly detached). Computing only grad_w means:
//   - No atomics (each thread owns its output slot)
//   - Simpler chain rule (no grad on phi, no grad on the PSF table)
//
// Chain rule:
//   contrib_re = w · (c · psf_r − s · psf_i)    with c=cos(phi), s=sin(phi)
//   contrib_im = w · (c · psf_i + s · psf_r)
//   ∂contrib_re/∂w = c·psf_r − s·psf_i
//   ∂contrib_im/∂w = c·psf_i + s·psf_r
//   ∂L/∂w += grad_rp_re · ∂contrib_re/∂w + grad_rp_im · ∂contrib_im/∂w
//           summed over all SPREAD offsets.

__global__ void step5_fused_backward_kernel(
    const float* __restrict__ grad_rp_real, // (n_tx, n_rx, K)
    const float* __restrict__ grad_rp_imag,
    const float* __restrict__ w_full,
    const float* __restrict__ phi_carrier,
    const float* __restrict__ n_peak,
    const float* __restrict__ psf_real,
    const float* __restrict__ psf_imag,
    float* __restrict__ grad_w,             // (M, n_tx, n_rx)
    int M, int n_tx, int n_rx, int K,
    int spread, int n_grid,
    float w_threshold)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const float w = w_full[idx];
    if (w < w_threshold) {
        grad_w[idx] = 0.0f;
        return;
    }

    const int r = (int)(idx % n_rx);
    const int t = (int)((idx / n_rx) % n_tx);
    const float phi = phi_carrier[idx];
    const float np_peak = n_peak[idx];
    const int n_floor = (int)floorf(np_peak);
    const float n_frac = np_peak - (float)n_floor;

    float s, c;
    sincosf(phi, &s, &c);

    const float idx_f = n_frac * (float)n_grid;
    int idx_lo = (int)floorf(idx_f);
    if (idx_lo < 0)         idx_lo = 0;
    if (idx_lo >= n_grid)   idx_lo = n_grid - 1;
    int idx_hi = idx_lo + 1;
    if (idx_hi >= n_grid)   idx_hi = n_grid - 1;
    const float frac = idx_f - (float)idx_lo;
    const float inv_frac = 1.0f - frac;

    const int base = (t * n_rx + r) * K;
    const int half_spread = spread >> 1;

    float g_w = 0.0f;

    #pragma unroll
    for (int d = 0; d < 15; ++d) {
        if (d >= spread) break;
        const int psf_off = d * n_grid;
        const float pr_lo = psf_real[psf_off + idx_lo];
        const float pr_hi = psf_real[psf_off + idx_hi];
        const float pi_lo = psf_imag[psf_off + idx_lo];
        const float pi_hi = psf_imag[psf_off + idx_hi];
        const float psf_r = pr_lo * inv_frac + pr_hi * frac;
        const float psf_i = pi_lo * inv_frac + pi_hi * frac;

        int bin = (n_floor + (d - half_spread)) % K;
        if (bin < 0) bin += K;

        const float g_rp_re = grad_rp_real[base + bin];
        const float g_rp_im = grad_rp_imag[base + bin];

        // ∂contrib_re/∂w = c·psf_r − s·psf_i
        // ∂contrib_im/∂w = c·psf_i + s·psf_r
        const float dre_dw = c * psf_r - s * psf_i;
        const float dim_dw = c * psf_i + s * psf_r;
        g_w += g_rp_re * dre_dw + g_rp_im * dim_dw;
    }
    grad_w[idx] = g_w;
}

void launch_step5_fused_backward(
    const float* grad_rp_real, const float* grad_rp_imag,
    const float* w_full, const float* phi_carrier, const float* n_peak,
    const float* psf_real, const float* psf_imag,
    float* grad_w,
    int M, int n_tx, int n_rx, int K,
    int spread, int n_grid, float w_threshold,
    cudaStream_t stream)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    step5_fused_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        grad_rp_real, grad_rp_imag,
        w_full, phi_carrier, n_peak, psf_real, psf_imag,
        grad_w, M, n_tx, n_rx, K, spread, n_grid, w_threshold);
}

} // namespace mm25v5
