"""CRP / ADC product-agnostic evaluation with per-range-bin azimuth-DC phase
correction.

Motivation
----------
The renderer produces a complex range profile (CRP) per (TX, RX) pair. The
absolute phase of the rendered CRP is not constrained by the (magnitude-only)
training loss — only relative phase across virtual antennas at each range bin
is meaningful, since that is what the azimuth FFT integrates to form the RA
image.  This module corrects for the unknown absolute phase per range bin by
reading the GT azimuth-DC bin (which captures the broadside spatial mean) and
rotating the predicted CRP at that range to match.  After correction, the
inverse range FFT gives an ADC stream directly comparable to GT.

Pipeline (per scene, all-numpy, no Hann, no zero-padding for invertibility):

    GT ADC (TX,RX,K) -> txrx_to_vx -> CRP_gt (n_vx, K)
    Rendered RP (TX,RX,K) -> txrx_to_vx -> CRP_pred (n_vx, K)
    RA_gt   = FFT(CRP_gt,   n=n_vx, axis=0)        # no shifts, no Hann
    RA_pred = FFT(CRP_pred, n=n_vx, axis=0)
    phi_star[r] = arg( RA_gt[0, r] * conj(RA_pred[0, r]) )
    RA_pred_corr = RA_pred * exp(-1j * phi_star)[None, :]
    CRP_pred_corr = IFFT(RA_pred_corr, n=n_vx, axis=0)
    ADC_pred_corr = IFFT(CRP_pred_corr, n=K, axis=1)
    ADC_gt        = IFFT(CRP_gt, n=K, axis=1)      # for metric symmetry

Both eval domains are then compared:

    CRP: complex correlation, magnitude correlation, normalized MSE
    ADC: same metrics, plus I/Q overlay traces for the qualitative figure

A round-trip sanity test verifies that GT ADC -> CRP -> RA -> CRP -> ADC
recovers the original ADC up to floating-point precision.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from mm3DPS.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT
from mm3DPS.data.ra_utils import txrx_to_virtual_array_numpy

# 86-element azimuth slice (elevation row 0) — matches the trained virtual array
N_VX = 86
K_RANGE = 256
# Default near-field mask: range bins [0:NEAR_FIELD_BINS] are dominated by
# TX-RX coupling (≤ ~0.6 m at 0.0432 m/bin) and are zeroed in both GT and
# pred CRPs before metric computation. The same masked CRP is then
# inverse-range-FFT'd to give a near-field-suppressed ADC for the time-
# domain metric.
NEAR_FIELD_BINS = 15


# ---------------------------------------------------------------------------
# Core: phase correction
# ---------------------------------------------------------------------------

def adc_txrx_to_vx_crp(adc_txrx: np.ndarray,
                         apply_range_hann: bool = True) -> np.ndarray:
    """ADC (TX=12, RX=16, K=256) complex -> CRP (n_vx=86, K=256) complex.

    Maps the (TX, RX) grid into the 86-element virtual array (elevation 0)
    and applies the range FFT.

    `apply_range_hann` (default True): apply a range-axis Hann window
    BEFORE the range FFT — this matches the trainer's loss-domain GT
    pipeline (`mm3DPS/data/ra_utils.py:adc_to_ra_complex` does the same).
    The renderer's `rp_complex` is emitted *without* a Hann (it's a CRP
    forward-model output, not a windowed FFT of an ADC), so the renderer
    learned during training to emit a CRP that matches the Hann-windowed
    GT CRP. Applying Hann to GT in eval therefore puts GT in the same
    domain the renderer was trained against.

    Set False for the (alternative) "raw FFT" CRP, which is exactly
    invertible via IFFT but is *not* the loss-domain GT.
    """
    vx = txrx_to_virtual_array_numpy(adc_txrx)            # (86, 256) complex
    if apply_range_hann:
        hann = np.hanning(K_RANGE).astype(np.float64)
        vx = vx * hann[None, :]
    crp = np.fft.fft(vx, n=K_RANGE, axis=-1)              # range FFT
    return crp


def rp_txrx_to_vx_crp(rp_txrx: np.ndarray) -> np.ndarray:
    """Rendered range-profile (TX=12, RX=16, K=256) complex -> CRP (86, 256).

    Renderer output is already a complex range profile (post-range-FFT in the
    forward model), so we only need the (TX, RX) -> virtual-array remap.
    """
    return txrx_to_virtual_array_numpy(rp_txrx)           # (86, 256)


def phase_correct_per_range(
    crp_pred: np.ndarray, crp_gt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-range-bin azimuth-DC phase correction.

    For each range bin r:
        phi*[r] = arg(  sum_v CRP_gt[v, r]  *  conj(sum_v CRP_pred[v, r])  )
                = arg(  RA_gt[0, r]         *  conj(RA_pred[0, r])         )

    where RA = FFT(CRP, n=n_vx, axis=0) and bin 0 is the azimuth-DC bin
    (the spatial mean across virtual antennas, i.e. the broadside beam).

    Returns
    -------
    crp_pred_corr : (n_vx, K) complex
    phi_star      : (K,) float, the per-range correction phase in radians
    """
    assert crp_pred.shape == crp_gt.shape == (N_VX, K_RANGE), (
        f"shape mismatch: pred={crp_pred.shape} gt={crp_gt.shape}")

    ra_gt = np.fft.fft(crp_gt, n=N_VX, axis=0)
    ra_pred = np.fft.fft(crp_pred, n=N_VX, axis=0)
    # Per-range cross-product at the azimuth-DC bin
    cross = ra_gt[0, :] * np.conj(ra_pred[0, :])          # (K,)
    phi_star = np.angle(cross)                             # (K,)
    # Rotate pred TOWARD gt: multiplying by exp(+j phi*) aligns the
    # azimuth-DC phase of pred with that of gt at each range bin.
    rot = np.exp(+1j * phi_star)[None, :]                  # (1, K)
    ra_pred_corr = ra_pred * rot
    crp_pred_corr = np.fft.ifft(ra_pred_corr, n=N_VX, axis=0)
    return crp_pred_corr, phi_star


def crp_to_adc(crp: np.ndarray) -> np.ndarray:
    """Inverse range FFT: CRP (n_vx, K) -> ADC (n_vx, K)."""
    return np.fft.ifft(crp, n=K_RANGE, axis=-1)


def apply_hann_az(crp_or_adc: np.ndarray) -> np.ndarray:
    """Apply the trainer's azimuth Hann window across the virtual-array axis.

    The trainer's `range_profile_to_ra` (mm3DPS/train_gaussian.py) and
    `adc_to_ra_complex` (mm3DPS/data/ra_utils.py) both multiply the (n_vx, K)
    array by `Hann(n_vx)[:, None]` before the azimuth FFT. So the renderer's
    loss only ever saw Hann-weighted edge VAs (≈0 weight) — those edge cells
    are essentially unconstrained by training and predict noise.

    For CRP/ADC metric evaluation to match the trainer's loss domain, we
    multiply BOTH GT and pred by the same Hann_az window on the VA axis.
    Without this, edge-VA noise dominates per-VA Pearson and pushes |CRP|
    correlation artificially low (verified empirically: 0.62 → 0.72 on
    seq_1_frame_185 train frame F181, matching polar |RA| Pearson 0.77).

    Note: Hann_az preserves phase per-element (it's a positive real scalar
    per VA), so phase corrections (R, VR) are unaffected — they're computed
    on the unweighted CRP and applied before this windowing.
    """
    h = np.hanning(N_VX).astype(np.float64)
    return crp_or_adc * h[:, None]


def phase_correct_va_range(
    crp_pred: np.ndarray, crp_gt: np.ndarray, n_iters: int = 3,
    start_bin: int = NEAR_FIELD_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint per-VA $\\alpha[v]$ + per-range $\\beta[r]$ LS phase correction.

    The total correction has rank-1 (separable) phase structure
    `correction[v, r] = α[v] + β[r]` — the calibration nuisance subspace.
    Both arrays are estimated by alternating closed-form LS on the residual:

        α[v] = arg(Σ_{r ≥ start_bin} CRP_GT[v, r] · conj(CRP_pred[v, r]))
        β[r] = arg(Σ_v             CRP_GT[v, r] · conj(CRP_pred[v, r]))

    α[v] sums **only over valid range bins** (≥ `start_bin`) so the near-
    field bins we ultimately mask out can't dominate the LS estimate (the
    coupling at bins 0..14 is the strongest signal in the radar response
    and would otherwise bias α toward a fit that is irrelevant to the
    post-mask metric). β[r] is per-bin so the near-field bins simply get
    their own corrections, which are subsequently masked out.

    Two or three alternating passes suffice (the subspaces are orthogonal —
    α only varies in v, β only in r). Total DOFs removed: n_vx + K = 86 + 256
    = 342, vs 22 016 phase-content DOFs across the (v, r) grid.

    Physical motivation:
      - α[v] models the per-channel RF calibration drift (TX/RX chain
        phase that the renderer's factory antenna patterns don't carry).
      - β[r] models per-range timing / ADC sample-zero offset, identical
        across all virtual antennas at a given range bin.

    Returns
    -------
    crp_pred_corrected : (n_vx, K) complex — pred with joint correction applied.
    alpha              : (n_vx,) float    — total per-VA correction phase.
    beta               : (K,) float       — total per-range correction phase.
    """
    assert crp_pred.shape == crp_gt.shape == (N_VX, K_RANGE)
    pred = crp_pred.copy()
    alpha_total = np.zeros(N_VX, dtype=np.float64)
    beta_total = np.zeros(K_RANGE, dtype=np.float64)

    for _ in range(n_iters):
        # β[r] step: per-range LS (proper inner product across virtual array)
        # Rotate pred by exp(+j arg(<gt, pred>)) at each r to align with gt.
        cross_r = np.sum(crp_gt * np.conj(pred), axis=0)         # (K,)
        beta_step = np.angle(cross_r)
        pred = pred * np.exp(+1j * beta_step[None, :])
        beta_total += beta_step

        # α[v] step: per-VA LS over VALID range bins only (skip near-field)
        cross_v = np.sum(crp_gt[:, start_bin:] *
                          np.conj(pred[:, start_bin:]), axis=1)  # (n_vx,)
        alpha_step = np.angle(cross_v)
        pred = pred * np.exp(+1j * alpha_step[:, None])
        alpha_total += alpha_step

    return pred, alpha_total, beta_total


def near_field_mask_crp(crp: np.ndarray,
                          start_bin: int = NEAR_FIELD_BINS) -> np.ndarray:
    """Zero out range bins [0:start_bin] of CRP — kills TX-RX coupling.

    Returns a copy. Apply identically to GT and pred so the comparison is
    consistent. The same masked CRP is then IFFT_range'd to obtain the
    near-field-suppressed ADC.
    """
    out = crp.copy()
    out[:, :start_bin] = 0
    return out


# ---------------------------------------------------------------------------
# Round-trip sanity test
# ---------------------------------------------------------------------------

def round_trip_sanity_test(adc_txrx: np.ndarray) -> dict:
    """Verify ADC -> CRP -> RA -> CRP -> ADC recovers the original.

    Checks (with no Hann, no zero-pad, no shifts):
        adc_vx          = txrx_to_vx(adc_txrx)
        crp             = FFT_range(adc_vx)
        ra              = FFT_az(crp, n=86)
        crp_recovered   = IFFT_az(ra, n=86)
        adc_recovered   = IFFT_range(crp_recovered)

    Returns max abs error at each step.
    """
    adc_vx = txrx_to_virtual_array_numpy(adc_txrx)         # (86, 256)
    crp = np.fft.fft(adc_vx, n=K_RANGE, axis=-1)
    ra = np.fft.fft(crp, n=N_VX, axis=0)
    crp_back = np.fft.ifft(ra, n=N_VX, axis=0)
    adc_back = np.fft.ifft(crp_back, n=K_RANGE, axis=-1)

    crp_err = float(np.max(np.abs(crp - crp_back)))
    crp_rel = crp_err / max(float(np.max(np.abs(crp))), 1e-30)
    adc_err = float(np.max(np.abs(adc_vx - adc_back)))
    adc_rel = adc_err / max(float(np.max(np.abs(adc_vx))), 1e-30)

    return {
        "crp_max_abs_err": crp_err,
        "crp_max_rel_err": crp_rel,
        "adc_max_abs_err": adc_err,
        "adc_max_rel_err": adc_rel,
        "crp_max_mag": float(np.max(np.abs(crp))),
        "adc_max_mag": float(np.max(np.abs(adc_vx))),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------





def _minmax(arr: np.ndarray) -> np.ndarray:
    """Independent min-max normalization to [0, 1] — matches the canonical
    `compute_cartesian_ra_metrics` in `mm3DPS/data/ra_utils.py`.
    """
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-30:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr.astype(np.float64) - mn) / (mx - mn)


def _compute_magnitude_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """RA-style metrics on |pred| and |gt| — invariant to phase correction.

    Independent min-max normalization of |GT| and |pred| to [0, 1], then
    Pearson, MSE, RMSE, PSNR, SSIM (all matching `compute_cartesian_ra_metrics`).
    """
    abs_pred = np.abs(pred).astype(np.float64)
    abs_gt = np.abs(gt).astype(np.float64)
    pred_norm = _minmax(abs_pred)
    gt_norm = _minmax(abs_gt)

    mag_corr = float(np.corrcoef(pred_norm.ravel(), gt_norm.ravel())[0, 1])
    if not np.isfinite(mag_corr):
        mag_corr = 0.0
    mag_mse = float(np.mean((pred_norm - gt_norm) ** 2))
    mag_rmse = float(np.sqrt(mag_mse))
    mag_psnr = (float(10.0 * np.log10(1.0 / mag_mse))
                if mag_mse > 0 else float("inf"))
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        win = min(7, pred_norm.shape[0], pred_norm.shape[1])
        if win % 2 == 0:
            win -= 1
        win = max(win, 3)
        mag_ssim = float(ssim_fn(gt_norm, pred_norm,
                                   data_range=1.0, win_size=win))
    except Exception:
        mag_ssim = float("nan")

    return {
        "mag_corr": mag_corr,
        "mag_psnr": mag_psnr,
        "mag_ssim": mag_ssim,
        "mag_mse": mag_mse,
        "mag_rmse": mag_rmse,
    }


def _phase_correct_va_only(crp_pred: np.ndarray, crp_gt: np.ndarray,
                            start_bin: int = NEAR_FIELD_BINS
                            ) -> np.ndarray:
    """Per-VA $\\alpha[v]$ only (V-mode) phase correction.

    α[v] = arg(Σ_{r ≥ start_bin} CRP_GT[v, r] · conj(CRP_pred[v, r]))

    86 DOFs removed. Models per-channel RF calibration drift. Symmetric to
    `phase_correct_per_range` (R mode) which models per-range timing.
    """
    cross_v = np.sum(crp_gt[:, start_bin:] *
                      np.conj(crp_pred[:, start_bin:]), axis=1)
    alpha = np.angle(cross_v)
    return crp_pred * np.exp(+1j * alpha[:, None])


def _compute_alt_phase_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Family of alternative phase-fidelity metrics for headline comparison.

    All scale-invariant by construction. Reported on the *same* signal the
    caller passes in (any phase correction is the caller's responsibility).

    Returns
    -------
    rho_per_va_mean      : mean of per-VA |ρ| (each row independently
                           absorbs an optimal complex scale, so per-VA
                           rotations are NOT penalized; per-range rotations
                           ARE).
    rho_per_range_mean   : symmetric — mean of per-range-bin |ρ| across
                           the range axis. Per-range rotations not
                           penalized; per-VA rotations are.
    phase_coherence      : magnitude-weighted ⟨cos(arg(pred · conj(gt)))⟩,
                           in [-1, 1]. Random-phase reference: 0. Equivalent
                           to the real-projection complex correlation
                           Re⟨pred, gt⟩ / (||pred|| ||gt||) but with |gt|²
                           magnitude weighting (focuses on signal-bearing
                           cells, not noise floor).
    """
    out = {}
    abs_gt = np.abs(gt)

    # Per-VA |ρ| mean (axis 0 = VA, axis 1 = range / fast-time)
    if pred.ndim >= 2 and pred.shape[0] >= 2:
        rho_v = []
        for v in range(pred.shape[0]):
            p = pred[v].ravel()
            g = gt[v].ravel()
            denom = np.linalg.norm(p) * np.linalg.norm(g)
            if denom > 1e-30:
                rho_v.append(float(np.abs(np.vdot(g, p)) / denom))
        out["rho_per_va_mean"] = float(np.mean(rho_v)) if rho_v else float("nan")
    else:
        out["rho_per_va_mean"] = float("nan")

    # Per-range |ρ| mean (across range / fast-time axis)
    if pred.ndim >= 2 and pred.shape[-1] >= 2:
        rho_r = []
        for r in range(pred.shape[-1]):
            p = pred[..., r].ravel()
            g = gt[..., r].ravel()
            denom = np.linalg.norm(p) * np.linalg.norm(g)
            if denom > 1e-30:
                rho_r.append(float(np.abs(np.vdot(g, p)) / denom))
        out["rho_per_range_mean"] = float(np.mean(rho_r)) if rho_r else float("nan")
    else:
        out["rho_per_range_mean"] = float("nan")

    # Magnitude-weighted phase coherence (mean cos(Δφ) weighted by |gt|²).
    # In [-1, 1]; 1 = perfect phase agreement, 0 = random.
    if abs_gt.size and float(abs_gt.max()) > 1e-30:
        phase_diff = np.angle(pred * np.conj(gt))
        weights = abs_gt.astype(np.float64) ** 2
        w_sum = float(weights.sum())
        if w_sum > 1e-30:
            out["phase_coherence"] = float(
                (weights * np.cos(phase_diff)).sum() / w_sum)
        else:
            out["phase_coherence"] = float("nan")
    else:
        out["phase_coherence"] = float("nan")

    return out


def _compute_diffphase_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Differential-phase metrics: phase-invariant *relative* phase fidelity.

    Why these and not |ρ| / σ_φ?
      The training loss is magnitude-only: per-(VA, range) absolute phase is
      unconstrained by design, and is unrecoverable up to a per-bin global
      rotation. Strict complex metrics (|ρ|, σ_φ, PSNR_ℂ) penalize that
      unrecoverable per-bin phase. Differential phase across an axis cancels
      any per-bin rotation along the *orthogonal* axis, so it isolates the
      relative-phase content that is physically meaningful for downstream
      products (peak localization, beamforming).

    Definitions (for a complex 2D tensor x of shape (V, K)):
      Δφ^(r) [v, r] = arg(x[v, r+1] · conj(x[v, r]))    range-derivative phase
      Δφ^(v) [v, r] = arg(x[v+1, r] · conj(x[v, r]))    VA-derivative phase

    Invariances:
      Δφ^(r) is invariant to any per-VA rotation of x (calibration α[v]).
      Δφ^(v) is invariant to any per-range rotation of x (sample-zero β[r]).
      Together they are invariant to the rank-1 calibration nuisance
      subspace that strict |ρ| / σ_φ would otherwise penalize.

    Metric: magnitude-weighted RMSE of Δφ_pred − Δφ_gt, wrapped into (-π, π].
    Weights are the joint magnitude product |gt[i]| · |gt[j]| so noise-only
    bins are down-weighted (a target-domain signal-aware error).

    Returns degrees. Random-phase reference: a uniform Δφ ∈ Uniform(-π, π]
    yields RMSE = π/√3 rad ≈ 103.9° (same as σ_φ).
    """
    def _wrap_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    abs_gt = np.abs(gt)
    out = {}

    # Range-derivative phase (axis -1)
    if pred.shape[-1] >= 2:
        d_pred_r = np.angle(pred[..., 1:] * np.conj(pred[..., :-1]))
        d_gt_r   = np.angle(gt[..., 1:]   * np.conj(gt[..., :-1]))
        diff_r = _wrap_pi(d_pred_r - d_gt_r)
        w_r = abs_gt[..., 1:] * abs_gt[..., :-1]
        w_sum = float(w_r.sum())
        if w_sum > 1e-30:
            rmse_rad = float(np.sqrt((w_r * diff_r ** 2).sum() / w_sum))
            out["dphi_r_coherence"] = float(
                (w_r * np.cos(diff_r)).sum() / w_sum)
        else:
            rmse_rad = float("nan")
            out["dphi_r_coherence"] = float("nan")
        out["dphi_r_rmse_deg"] = float(np.degrees(rmse_rad))
    else:
        out["dphi_r_rmse_deg"] = float("nan")
        out["dphi_r_coherence"] = float("nan")

    # VA-derivative phase (axis 0)
    if pred.ndim >= 2 and pred.shape[0] >= 2:
        d_pred_v = np.angle(pred[1:, :] * np.conj(pred[:-1, :]))
        d_gt_v   = np.angle(gt[1:, :]   * np.conj(gt[:-1, :]))
        diff_v = _wrap_pi(d_pred_v - d_gt_v)
        w_v = abs_gt[1:, :] * abs_gt[:-1, :]
        w_sum = float(w_v.sum())
        if w_sum > 1e-30:
            rmse_rad = float(np.sqrt((w_v * diff_v ** 2).sum() / w_sum))
            out["dphi_v_coherence"] = float(
                (w_v * np.cos(diff_v)).sum() / w_sum)
        else:
            rmse_rad = float("nan")
            out["dphi_v_coherence"] = float("nan")
        out["dphi_v_rmse_deg"] = float(np.degrees(rmse_rad))
    else:
        out["dphi_v_rmse_deg"] = float("nan")
        out["dphi_v_coherence"] = float("nan")

    return out


def _compute_complex_only_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Complex-valued metrics (depend on phase correction).

    Image-style (used in the CRP table):
      - complex_corr : |<gt, pred>| / (||gt|| ||pred||)             scale-invariant
      - complex_mse  : MSE on unit-peak-normalized complex tensors  in [0, 1]
      - complex_psnr : 10·log10(1 / complex_mse)                    image-domain

    Radar-standard time-series metrics (used in the ADC table):
      - rec_snr_db   : -10·log10(||pred-gt||² / ||gt||²)
                       a.k.a. Reconstruction SNR / NMSE in dB.
                       Energy-normalized, scale-aware. Matches Sionna RT,
                       NeRF², compressed-sensing radar conventions.
      - nmse         : ||pred-gt||² / ||gt||²                        linear NMSE
      - phase_rmse_deg : magnitude-weighted RMS phase error in degrees,
                       weights = |gt|² (noise-only bins down-weighted).

    The renderer's prediction may have an arbitrary global complex scale
    relative to GT (different scene-level RCS scaling). For metrics that
    depend on residual energy we first match it via the optimal scalar
    gain `c_opt = conj(<gt, pred>) / <pred, pred>`, which minimizes
    `||c_opt·pred - gt||²`. |ρ| is scale-invariant by construction.
    """
    p_flat = pred.ravel()
    g_flat = gt.ravel()

    # Complex correlation |ρ| (invariant to global complex gain)
    inner = np.vdot(g_flat, p_flat)                  # <gt, pred> = sum conj(gt)*pred
    g_norm = np.linalg.norm(g_flat)
    p_norm = np.linalg.norm(p_flat)
    cmag = float(np.abs(inner) / (g_norm * p_norm + 1e-30))

    # Optimal scalar gain to align pred to gt (minimizes ||c·pred − gt||²).
    # df/dc̄ = c·||pred||² − <pred, gt> = 0 ⇒ c_opt = <pred, gt>/||pred||²
    #                                       = conj(<gt, pred>) / ||pred||²
    pp = float(np.vdot(p_flat, p_flat).real)
    if pp > 1e-30:
        c_opt = np.conj(inner) / pp
        pred_aligned = pred * c_opt
    else:
        pred_aligned = pred

    # Image-style PSNR_C / MSE_C on unit-peak-normalized complex (kept for
    # CRP reporting). Use the gain-matched prediction so it's not penalized
    # for global RCS scale.
    p_peak = float(np.max(np.abs(pred_aligned))) if pred_aligned.size else 0.0
    g_peak = float(np.max(np.abs(gt))) if gt.size else 0.0
    if p_peak > 1e-30 and g_peak > 1e-30:
        pred_uc = pred_aligned / p_peak
        gt_uc = gt / g_peak
        complex_mse = float(np.mean(np.abs(pred_uc - gt_uc) ** 2))
        complex_psnr = (float(10.0 * np.log10(1.0 / complex_mse))
                         if complex_mse > 0 else float("inf"))
    else:
        complex_mse = float("nan")
        complex_psnr = float("nan")

    # Magnitude-weighted phase RMSE (degrees), with phase wrap into (-π, π]
    abs_gt = np.abs(gt)
    if abs_gt.size and float(abs_gt.max()) > 1e-30:
        phase_diff = np.angle(pred_aligned * np.conj(gt))   # principal value
        weights = (abs_gt.astype(np.float64)) ** 2          # |gt|² weighting
        w_sum = float(weights.sum())
        if w_sum > 1e-30:
            phase_rmse_rad = float(np.sqrt(
                (weights * phase_diff ** 2).sum() / w_sum))
        else:
            phase_rmse_rad = float("nan")
    else:
        phase_rmse_rad = float("nan")
    phase_rmse_deg = float(np.degrees(phase_rmse_rad))

    return {
        "complex_corr": cmag,
        "complex_psnr": complex_psnr,
        "complex_mse": complex_mse,
        "phase_rmse_deg": phase_rmse_deg,
    }




# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Eval result for a single (scene, frame) pair."""
    frame: int
    crp: dict
    adc: dict
    sanity: dict
    # Arrays kept only when the caller needs them for figures (test frame).
    crp_pred_corr: Optional[np.ndarray] = None
    crp_gt: Optional[np.ndarray] = None
    adc_pred_corr: Optional[np.ndarray] = None
    adc_gt: Optional[np.ndarray] = None
    phi_star: Optional[np.ndarray] = None


@dataclass
class SceneResult:
    name: str
    test_frame: int
    test: FrameResult                    # held-out test view
    train: List[FrameResult]             # 8 train views


def evaluate_frame(
    frame: int,
    gt_adc_path: str,
    rendered_rp_path: str,
    held_out_loop: int = 0,
    start_bin: int = NEAR_FIELD_BINS,
    keep_arrays: bool = False,
) -> FrameResult:
    """Evaluate a single (scene, frame): load GT + rendered, phase-correct,
    mask near-field, compute metrics.

    `start_bin`: range bins [0:start_bin] are zeroed in BOTH GT and pred CRPs
    after phase correction (TX-RX coupling rejection). Set to 0 to disable.
    The masked CRPs are then IFFT_range'd to give near-field-suppressed ADCs.
    CRP metrics are computed only over bins [start_bin:] for both signals.
    `keep_arrays`: when True, retain the masked CRPs/ADCs on the result
    so the figure pipeline can read them. Set False for train frames to
    keep memory bounded.
    """
    # GT raw ADC: (n_loops, n_rx, n_tx, K) complex128
    gt_raw = np.load(gt_adc_path)
    assert gt_raw.ndim == 4, f"unexpected GT shape: {gt_raw.shape}"
    gt_adc_txrx = gt_raw[held_out_loop].transpose(1, 0, 2).astype(np.complex128)

    # Rendered CRP: (12, 16, 256) complex64
    rp_pred_txrx = np.load(rendered_rp_path).astype(np.complex128)
    assert rp_pred_txrx.shape == (12, 16, 256), (
        f"unexpected rendered shape: {rp_pred_txrx.shape}")

    sanity = round_trip_sanity_test(gt_adc_txrx)

    crp_gt_full = adc_txrx_to_vx_crp(gt_adc_txrx)
    crp_pred_raw = rp_txrx_to_vx_crp(rp_pred_txrx)

    # ── Two phase-correction modes (both reported side-by-side) ──
    # Mode R: per-range β only (azimuth-DC bin scheme; user's earlier choice)
    crp_pred_R_full, phi_star = phase_correct_per_range(crp_pred_raw,
                                                          crp_gt_full)
    # Mode VR: per-VA α + per-range β joint LS (calibration removal)
    crp_pred_VR_full, alpha_VR, beta_VR = phase_correct_va_range(
        crp_pred_raw, crp_gt_full, start_bin=start_bin)

    # ── Near-field mask (TX-RX coupling) applied to GT and both pred CRPs ──
    crp_gt_m = near_field_mask_crp(crp_gt_full, start_bin=start_bin)
    crp_pred_R_m = near_field_mask_crp(crp_pred_R_full, start_bin=start_bin)
    crp_pred_VR_m = near_field_mask_crp(crp_pred_VR_full, start_bin=start_bin)

    # ── Apply Hann_az on the VA axis to put metric in trainer's loss domain ──
    # The trainer's loss is on |FFT_az(Hann_az × CRP)|, so edge VAs (Hann≈0)
    # are unconstrained by training and predict noise. Without this windowing
    # those noisy edge cells drag |CRP| Pearson down even though |RA| is
    # unaffected. Apply identically to GT and pred for fair comparison.
    crp_gt_e = apply_hann_az(crp_gt_m)
    crp_pred_R_e = apply_hann_az(crp_pred_R_m)
    crp_pred_VR_e = apply_hann_az(crp_pred_VR_m)

    # ── ADCs from the (Hann_az × masked) CRPs ──
    adc_gt_vx = crp_to_adc(crp_gt_e)
    adc_pred_R_vx = crp_to_adc(crp_pred_R_e)
    adc_pred_VR_vx = crp_to_adc(crp_pred_VR_e)

    # ── Magnitude metrics ──
    # |CRP| is invariant to phase correction (rotation preserves magnitude
    # pointwise) — pick either pred. |ADC| IS NOT invariant: IFFT_range
    # mixes phase and magnitude so a different CRP phase pattern produces
    # a different |ADC|. We compute |ADC| on the VR-corrected ADC since
    # VR is the recommended mode (the magnitude after calibration removal).
    crp_mag = _compute_magnitude_metrics(
        crp_pred_VR_e[:, start_bin:], crp_gt_e[:, start_bin:])
    adc_mag = _compute_magnitude_metrics(adc_pred_VR_vx, adc_gt_vx)

    # ── Differential-phase metrics (phase-invariant relative phase) ──
    # Computed on the *uncorrected* (raw) CRP/ADC: differential phase is
    # invariant to per-VA and per-range rotations by construction, so the
    # R / VR calibration removal would have no effect. Using the raw signal
    # is the cleanest framing for the paper: "no calibration, no penalty
    # on absolute per-bin phase, only relative-phase fidelity".
    crp_pred_raw_e = apply_hann_az(near_field_mask_crp(crp_pred_raw,
                                                        start_bin=start_bin))
    adc_pred_raw_vx = crp_to_adc(crp_pred_raw_e)
    crp_dphi = _compute_diffphase_metrics(
        crp_pred_raw_e[:, start_bin:], crp_gt_e[:, start_bin:])
    adc_dphi = _compute_diffphase_metrics(adc_pred_raw_vx, adc_gt_vx)

    # ── Alternative phase-fidelity metrics (per-VA, per-range, coherence) ──
    crp_alt = _compute_alt_phase_metrics(
        crp_pred_raw_e[:, start_bin:], crp_gt_e[:, start_bin:])
    adc_alt = _compute_alt_phase_metrics(adc_pred_raw_vx, adc_gt_vx)

    # ── Per-VA-only ("V" mode) phase correction for headline-comparison ──
    # 86 DOFs removed (vs 256 for R, 342 for VR). Symmetric to R.
    crp_pred_V_full = _phase_correct_va_only(
        crp_pred_raw, crp_gt_full, start_bin=start_bin)
    crp_pred_V_e = apply_hann_az(near_field_mask_crp(
        crp_pred_V_full, start_bin=start_bin))
    adc_pred_V_vx = crp_to_adc(crp_pred_V_e)

    # ── Complex metrics under each correction mode (kept for back-compat) ──
    crp_raw = _compute_complex_only_metrics(
        crp_pred_raw_e[:, start_bin:], crp_gt_e[:, start_bin:])
    crp_V = _compute_complex_only_metrics(
        crp_pred_V_e[:, start_bin:], crp_gt_e[:, start_bin:])
    crp_R = _compute_complex_only_metrics(
        crp_pred_R_e[:, start_bin:], crp_gt_e[:, start_bin:])
    crp_VR = _compute_complex_only_metrics(
        crp_pred_VR_e[:, start_bin:], crp_gt_e[:, start_bin:])
    adc_raw = _compute_complex_only_metrics(adc_pred_raw_vx, adc_gt_vx)
    adc_V = _compute_complex_only_metrics(adc_pred_V_vx, adc_gt_vx)
    adc_R = _compute_complex_only_metrics(adc_pred_R_vx, adc_gt_vx)
    adc_VR = _compute_complex_only_metrics(adc_pred_VR_vx, adc_gt_vx)

    crp_metrics = {
        **crp_mag,
        **crp_dphi,
        **crp_alt,
        **{f"{k}_raw": v for k, v in crp_raw.items()},
        **{f"{k}_V":   v for k, v in crp_V.items()},
        **{f"{k}_R":   v for k, v in crp_R.items()},
        **{f"{k}_VR":  v for k, v in crp_VR.items()},
    }
    adc_metrics = {
        **adc_mag,
        **adc_dphi,
        **adc_alt,
        **{f"{k}_raw": v for k, v in adc_raw.items()},
        **{f"{k}_V":   v for k, v in adc_V.items()},
        **{f"{k}_R":   v for k, v in adc_R.items()},
        **{f"{k}_VR":  v for k, v in adc_VR.items()},
    }

    out = FrameResult(frame=frame, crp=crp_metrics, adc=adc_metrics,
                       sanity=sanity)
    if keep_arrays:
        # Save the VR-corrected, Hann_az-weighted, near-field-masked arrays
        # as the canonical "complex output" — these are the exact arrays
        # the metrics were computed on, so figures show what's evaluated.
        out.crp_pred_corr = crp_pred_VR_e.astype(np.complex64)
        out.crp_gt = crp_gt_e.astype(np.complex64)
        out.adc_pred_corr = adc_pred_VR_vx.astype(np.complex64)
        out.adc_gt = adc_gt_vx.astype(np.complex64)
        out.phi_star = phi_star.astype(np.float32)
    return out


def evaluate_scene(
    name: str,
    test_frame: int,
    train_frames: List[int],
    held_out_loop: int = 0,
    start_bin: int = NEAR_FIELD_BINS,
    ours_dir: Optional[str] = None,
    run_tag_template: Optional[str] = None,
) -> SceneResult:
    """Evaluate one scene: test view + 8 train views."""
    F = test_frame
    test_gt = _scene_gt_adc_path(name, F)
    test_rp = _scene_rendered_rp_test(name, F, ours_dir, run_tag_template)
    test_res = evaluate_frame(F, test_gt, test_rp,
                                held_out_loop=held_out_loop,
                                start_bin=start_bin, keep_arrays=True)

    train_results = []
    for f in train_frames:
        gt_p = _scene_gt_adc_path(name, f)
        rp_p = _scene_rendered_rp_train(name, F, f, ours_dir, run_tag_template)
        if not (os.path.isfile(gt_p) and os.path.isfile(rp_p)):
            print(f"    [skip train frame {f}] missing GT or pred")
            continue
        r = evaluate_frame(f, gt_p, rp_p, held_out_loop=held_out_loop,
                            start_bin=start_bin, keep_arrays=False)
        train_results.append(r)

    return SceneResult(name=name, test_frame=F, test=test_res,
                        train=train_results)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    # Magnitude — RA-style on min-max-normalized |·| (invariant to phase corr)
    "mag_corr", "mag_psnr", "mag_ssim", "mag_mse", "mag_rmse",
    # Differential-phase (absolute-phase-invariant relative phase fidelity).
    # Both forms reported: RMSE in degrees and magnitude-weighted ⟨cos(Δφ)⟩
    # in [-1, 1] (random reference 0). The coherence form is what the paper
    # tables and figure overlays use for headline numbers.
    "dphi_r_rmse_deg", "dphi_v_rmse_deg",
    "dphi_r_coherence", "dphi_v_coherence",
    # Alternative phase-fidelity metrics (per-VA |ρ| mean, per-range |ρ|
    # mean, magnitude-weighted phase coherence) — for reporting comparison.
    "rho_per_va_mean", "rho_per_range_mean", "phase_coherence",
    # Complex with NO phase correction (raw)
    "complex_corr_raw", "complex_psnr_raw", "complex_mse_raw", "phase_rmse_deg_raw",
    # Complex under per-VA α only ("V") — 86 DOFs
    "complex_corr_V", "complex_psnr_V", "complex_mse_V", "phase_rmse_deg_V",
    # Complex under per-range β only ("R") — azimuth-DC bin correction (256 DOFs)
    "complex_corr_R", "complex_psnr_R", "complex_mse_R", "phase_rmse_deg_R",
    # Complex under per-VA α + per-range β ("VR") — joint LS calibration removal (342 DOFs)
    "complex_corr_VR", "complex_psnr_VR", "complex_mse_VR", "phase_rmse_deg_VR",
)


def _mean_metrics(frames: List[FrameResult]) -> dict:
    """Mean of each metric across a list of FrameResult."""
    out = {"crp": {}, "adc": {}}
    for k in _METRIC_KEYS:
        cv = [f.crp[k] for f in frames if np.isfinite(f.crp[k])]
        av = [f.adc[k] for f in frames if np.isfinite(f.adc[k])]
        out["crp"][k] = {"mean": float(np.mean(cv)) if cv else float("nan"),
                          "std":  float(np.std(cv))  if cv else float("nan")}
        out["adc"][k] = {"mean": float(np.mean(av)) if av else float("nan"),
                          "std":  float(np.std(av))  if av else float("nan")}
    return out


def aggregate(results: List[SceneResult]) -> dict:
    """Aggregate train and test metrics across scenes.

    For TEST: mean / std taken over the 6 scene-level test values.
    For TRAIN: mean / std taken over the 6 scene-level train means
      (so each scene contributes equally regardless of train-frame count).
    """
    test_frames = [r.test for r in results]
    # Per-scene train mean (single FrameResult-like dict per scene)
    train_scene_means = []
    for r in results:
        m = _mean_metrics(r.train)
        # Repackage into FrameResult so _mean_metrics can be reused
        train_scene_means.append(FrameResult(
            frame=-1,
            crp={k: m["crp"][k]["mean"] for k in _METRIC_KEYS},
            adc={k: m["adc"][k]["mean"] for k in _METRIC_KEYS},
            sanity={"crp_max_rel_err": 0.0, "adc_max_rel_err": 0.0},
        ))

    out = {
        "test": _mean_metrics(test_frames),
        "train": _mean_metrics(train_scene_means),
        "sanity": {},
    }
    for k in ("crp_max_rel_err", "adc_max_rel_err"):
        vals = [r.test.sanity[k] for r in results]
        out["sanity"][k] = {"mean": float(np.mean(vals)),
                             "max":  float(np.max(vals))}
    return out


def save_results(results: List[SceneResult], output_dir: str) -> None:
    """Write per-scene metrics + arrays + aggregate JSON.

    Per-scene structure:
      {
        "test_frame": int,
        "sanity": dict,
        "test": {"crp": {...}, "adc": {...}},
        "train_frames": [{"frame": F, "crp": {...}, "adc": {...}}, ...],
        "train_mean": {"crp": {...}, "adc": {...}}
      }
    """
    os.makedirs(output_dir, exist_ok=True)
    per_scene = {}
    for r in results:
        scene_dir = os.path.join(output_dir, r.name)
        os.makedirs(scene_dir, exist_ok=True)
        # Test-frame arrays for the figure pipeline
        if r.test.crp_pred_corr is not None:
            np.save(os.path.join(scene_dir, "crp_pred_corr.npy"),
                     r.test.crp_pred_corr)
            np.save(os.path.join(scene_dir, "crp_gt.npy"), r.test.crp_gt)
            np.save(os.path.join(scene_dir, "adc_pred_corr.npy"),
                     r.test.adc_pred_corr)
            np.save(os.path.join(scene_dir, "adc_gt.npy"), r.test.adc_gt)
            np.save(os.path.join(scene_dir, "phi_star.npy"), r.test.phi_star)

        train_mean = _mean_metrics(r.train)
        # Strip the std for the simple "mean" view
        train_mean_simple = {
            "crp": {k: train_mean["crp"][k]["mean"] for k in _METRIC_KEYS},
            "adc": {k: train_mean["adc"][k]["mean"] for k in _METRIC_KEYS},
        }
        per_scene[r.name] = {
            "test_frame": r.test_frame,
            "sanity": r.test.sanity,
            "test": {"crp": r.test.crp, "adc": r.test.adc},
            "train_frames": [
                {"frame": f.frame, "crp": f.crp, "adc": f.adc}
                for f in r.train
            ],
            "train_mean": train_mean_simple,
            "train_mean_std": train_mean,  # full mean+std view
        }

    agg = aggregate(results)
    out = {"per_scene": per_scene, "aggregate": agg}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Convenience: scene path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "..", ".."))


_DEFAULT_OURS_DIR = DEFAULT_OUTPUT_ROOT
_DATA_ROOT = DEFAULT_DATA_ROOT
_DEFAULT_RUN_TAG_TEMPLATE = (
    "{scene}_train8frames_1loops_test{test_frame}_loop0_pass2_N20000"
)


def _scene_run_dir(scene: str, test_frame: int,
                   ours_dir: Optional[str] = None,
                   run_tag_template: Optional[str] = None) -> str:
    """Resolve the per-scene 3DPS run directory.

    The defaults read ``output/training/<scene>_train8frames_..._N20000``.
    The CLI exposes the overrides as ``--ours_dir`` and ``--run_tag``.
    """
    od = ours_dir if ours_dir is not None else _DEFAULT_OURS_DIR
    tmpl = (run_tag_template if run_tag_template is not None
            else _DEFAULT_RUN_TAG_TEMPLATE)
    return os.path.join(od, tmpl.format(scene=scene, test_frame=test_frame))


def _scene_gt_adc_path(scene: str, frame: int) -> str:
    """data/<scene>/radar/cascaded_frame_<F>.npy"""
    return os.path.join(_DATA_ROOT, scene, "radar",
                         f"cascaded_frame_{frame}.npy")


def _scene_rendered_rp_test(scene: str, test_frame: int,
                             ours_dir: Optional[str] = None,
                             run_tag_template: Optional[str] = None) -> str:
    return os.path.join(
        _scene_run_dir(scene, test_frame, ours_dir, run_tag_template),
        "rendered_test_rp_complex.npy")


def _scene_rendered_rp_train(scene: str, test_frame: int,
                               train_frame: int,
                               ours_dir: Optional[str] = None,
                               run_tag_template: Optional[str] = None) -> str:
    return os.path.join(
        _scene_run_dir(scene, test_frame, ours_dir, run_tag_template),
        "train_frames", f"frame_{train_frame}", "rendered_rp_complex.npy")


def test_frame_of(scene: str) -> int:
    return int(scene.rsplit("_", 1)[-1])


# Paper scenes (name, test_frame, train_frames)
PAPER_SCENES_FULL = [
    ("seq_0_frame_135", 135, [131, 132, 133, 134, 136, 137, 138, 139]),
    ("seq_1_frame_185", 185, [181, 182, 183, 184, 186, 187, 188, 189]),
    ("seq_1_frame_438", 438, [434, 435, 436, 437, 439, 440, 441, 442]),
    ("seq_2_frame_105", 105, [101, 102, 103, 104, 106, 107, 108, 109]),
    ("seq_2_frame_160", 160, [156, 157, 158, 159, 161, 162, 163, 164]),
    ("seq_2_frame_300", 300, [296, 297, 298, 299, 301, 302, 303, 304]),
]
PAPER_SCENES = [s[0] for s in PAPER_SCENES_FULL]
PAPER_SCENES_INFO = {s[0]: (s[1], s[2]) for s in PAPER_SCENES_FULL}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=PAPER_SCENES)
    ap.add_argument("--output_dir", default=os.path.join(
        PROJECT_ROOT, "output", "crp_adc_eval"))
    ap.add_argument("--sanity_only", action="store_true",
                    help="Only run round-trip sanity test on GT (no rendered).")
    ap.add_argument("--start_bin", type=int, default=NEAR_FIELD_BINS,
                    help=f"Near-field mask: zero CRP bins [0:start_bin] in both"
                          f" GT and pred (default {NEAR_FIELD_BINS}; pass 0 to"
                          f" disable masking).")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                    help="Prepared data directory (default: <repo>/data).")
    ap.add_argument("--ours_dir", default=None,
                    help="Directory holding the per-scene 3DPS run directories "
                         "(default: <repo>/output/training).")
    ap.add_argument("--run_tag", default=None,
                    help="Run-tag template inside --ours_dir; supports "
                         "{scene} and {test_frame} placeholders. Default "
                         f"'{_DEFAULT_RUN_TAG_TEMPLATE}'.")
    args = ap.parse_args()
    global _DATA_ROOT
    _DATA_ROOT = args.data_root

    if args.sanity_only:
        print("=== Round-trip sanity test (GT only) ===")
        for scene in args.scenes:
            F = test_frame_of(scene)
            gt_path = _scene_gt_adc_path(scene, F)
            if not os.path.isfile(gt_path):
                print(f"  [skip] {scene}: GT not found at {gt_path}")
                continue
            gt_raw = np.load(gt_path)
            adc_txrx = gt_raw[0].transpose(1, 0, 2).astype(np.complex128)
            s = round_trip_sanity_test(adc_txrx)
            print(f"  {scene}: "
                  f"CRP rel_err={s['crp_max_rel_err']:.2e}  "
                  f"ADC rel_err={s['adc_max_rel_err']:.2e}  "
                  f"(CRP mag={s['crp_max_mag']:.2e}  "
                  f"ADC mag={s['adc_max_mag']:.2e})")
        return

    print(f"=== CRP/ADC eval (train+test, {len(args.scenes)} scenes,  "
          f"start_bin={args.start_bin}) ===")
    results = []
    for scene in args.scenes:
        if scene not in PAPER_SCENES_INFO:
            print(f"  [skip] {scene}: not in PAPER_SCENES_INFO "
                  f"(unknown train_frames)")
            continue
        F, train_frames = PAPER_SCENES_INFO[scene]
        # quick sanity check that test rendered CRP exists
        if not os.path.isfile(_scene_rendered_rp_test(scene, F,
                                                       args.ours_dir,
                                                       args.run_tag)):
            print(f"  [skip] {scene}: rendered test CRP not found")
            continue
        print(f"  evaluating {scene} (test={F}, train={train_frames})...")
        r = evaluate_scene(scene, F, train_frames,
                            start_bin=args.start_bin,
                            ours_dir=args.ours_dir,
                            run_tag_template=args.run_tag)
        tcrp, tadc = r.test.crp, r.test.adc
        print(f"    test  CRP: corr={tcrp['mag_corr']:.3f}  "
              f"PSNR={tcrp['mag_psnr']:.2f}  SSIM={tcrp['mag_ssim']:.3f}  "
              f"||  VR: |ρ|={tcrp['complex_corr_VR']:.3f}  "
              f"σ_φ={tcrp['phase_rmse_deg_VR']:.1f}°")
        print(f"    test  ADC: env corr={tadc['mag_corr']:.3f}  "
              f"||  VR: |ρ|={tadc['complex_corr_VR']:.3f}  "
              f"σ_φ={tadc['phase_rmse_deg_VR']:.1f}°")
        if r.train:
            tm = _mean_metrics(r.train)
            print(f"    train CRP (mean): corr="
                  f"{tm['crp']['mag_corr']['mean']:.3f}  "
                  f"PSNR={tm['crp']['mag_psnr']['mean']:.2f}  "
                  f"|| VR: |ρ|={tm['crp']['complex_corr_VR']['mean']:.3f}  "
                  f"σ_φ={tm['crp']['phase_rmse_deg_VR']['mean']:.1f}°")
            print(f"    train ADC (mean): env corr="
                  f"{tm['adc']['mag_corr']['mean']:.3f}  "
                  f"|| VR: |ρ|={tm['adc']['complex_corr_VR']['mean']:.3f}  "
                  f"σ_φ={tm['adc']['phase_rmse_deg_VR']['mean']:.1f}°")
        results.append(r)

    if results:
        save_results(results, args.output_dir)
        agg = aggregate(results)
        print("\n=== Aggregate (across {} scenes) ===".format(len(results)))
        for split in ("test", "train"):
            print(f"  --- {split} ---")
            for dom in ("crp", "adc"):
                m = agg[split][dom]
                if dom == "crp":
                    # Image-style report for CRP (PSNR/SSIM make sense here)
                    print(f"    CRP  |.|  corr={m['mag_corr']['mean']:.3f}  "
                          f"PSNR={m['mag_psnr']['mean']:.2f}  "
                          f"SSIM={m['mag_ssim']['mean']:.3f}  "
                          f"|| VR  |ρ|={m['complex_corr_VR']['mean']:.3f}  "
                          f"phRMSE={m['phase_rmse_deg_VR']['mean']:.1f}°")
                else:
                    # Time-series report for ADC (no PSNR/SSIM)
                    print(f"    ADC  envelope corr={m['mag_corr']['mean']:.3f}  "
                          f"|| VR  |ρ|={m['complex_corr_VR']['mean']:.3f}  "
                          f"phRMSE={m['phase_rmse_deg_VR']['mean']:.1f}°")
        print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
