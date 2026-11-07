"""Renderer-based cascade alignment (vendored from mm3DPS, CUDA backend).

Pose math helpers (``apply_2dof_to_config``, ``apply_4dof_to_config``)
and the grid-search / refinement control flow are preserved from the
original MC implementation. All rendering goes through the CUDA
backend via ``cascaded_renderer_cuda.AlignmentCtx`` — the MC renderer,
``mitsuba``, ``drjit``, and ``mm3DPS.renderer`` imports are removed.

Strategy per frame:
1. Build a CUDA ``AlignmentCtx`` (one-time: FPS'd model + GT RA).
2. Grid search over pose deltas; each candidate mutates ctx.rast pose
   in-place and renders via the CUDA kernel.
3. Nelder-Mead refinement over the same ctx.
4. ``save_aligned_config`` writes the final pose to a JSON config on disk.
"""

import os
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize

from mm3DPS.data.ra_utils import adc_to_ra_image, ra_polar_to_cartesian

from .cascaded_renderer_cuda import (
    AlignmentCtx, build_alignment_context, apply_pose_to_ctx,
    render_and_evaluate_cuda, destroy_alignment_context,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TX_PATTERN = os.path.join(PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy")
RX_PATTERN = os.path.join(PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy")

SCENES = [
    ("seq_0_frame_135", 135),
    ("seq_0_frame_390", 390),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]


# ============================================================
# Geometry helpers
# ============================================================

def _norm(v, eps=1e-9):
    return v / (np.linalg.norm(v) + eps)


def _axis_angle_to_matrix(w):
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    k = w / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def apply_2dof_to_config(base_config, delta_range_m, delta_azimuth_deg):
    """Apply 2DOF transform (range + azimuth) with ZERO rotation.

    Preserves the base-config boresight direction exactly. Returns
    ``(tx_pos_mm, rx_pos_mm, boresight)``.
    """
    tx_pos = np.array([t["pos_mm"] for t in base_config["tx_array"]], dtype=float) / 1000.0
    rx_pos = np.array([r["pos_mm"] for r in base_config["rx_array"]], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])
    n_tx = len(tx_pos)

    board_center = np.mean(all_pos, axis=0)
    boresight = _norm(np.array(base_config["tx_array"][0]["boresight"], dtype=float))

    rel_pos = all_pos - board_center

    range_offset = boresight * delta_range_m
    reference_range_m = 10.0
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)
    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight, z_world))
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])
    azimuth_offset = x_board * delta_azimuth_m
    translation = range_offset + azimuth_offset

    new_board_center = board_center + translation
    new_all_pos = rel_pos + new_board_center

    new_tx_pos_mm = new_all_pos[:n_tx] * 1000.0
    new_rx_pos_mm = new_all_pos[n_tx:] * 1000.0

    return new_tx_pos_mm, new_rx_pos_mm, boresight


def apply_4dof_to_config(base_config, delta_range_m, delta_azimuth_deg,
                         rotation_elev_deg, rotation_azim_deg):
    """Apply full 4DOF transform: range + azimuth + elevation-rot + azimuth-rot."""
    tx_pos = np.array([t["pos_mm"] for t in base_config["tx_array"]], dtype=float) / 1000.0
    rx_pos = np.array([r["pos_mm"] for r in base_config["rx_array"]], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])
    n_tx = len(tx_pos)

    board_center = np.mean(all_pos, axis=0)
    base_boresight = _norm(np.array(base_config["tx_array"][0]["boresight"], dtype=float))

    rot_elev_rad = np.radians(rotation_elev_deg)
    R_elev = _axis_angle_to_matrix(np.array([0.0, 0.0, rot_elev_rad]))
    boresight_after_elev = _norm(R_elev @ base_boresight)

    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight_after_elev, z_world))
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])

    rot_azim_rad = np.radians(rotation_azim_deg)
    R_azim = _axis_angle_to_matrix(x_board * rot_azim_rad)
    R_combined = R_azim @ R_elev

    new_boresight = _norm(R_combined @ base_boresight)

    rel_pos = all_pos - board_center
    rel_pos_rotated = (R_combined @ rel_pos.T).T

    range_offset = new_boresight * delta_range_m
    reference_range_m = 10.0
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)
    x_new = _norm(np.cross(new_boresight, z_world))
    if np.linalg.norm(x_new) < 1e-6:
        x_new = np.array([1.0, 0.0, 0.0])
    azimuth_offset = x_new * delta_azimuth_m
    translation = range_offset + azimuth_offset

    new_board_center = board_center + translation
    new_all_pos = rel_pos_rotated + new_board_center

    return new_all_pos[:n_tx] * 1000.0, new_all_pos[n_tx:] * 1000.0, new_boresight


# ============================================================
# RA processing helpers (kept for callers outside the alignment loop)
# ============================================================

def minmax_normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-30:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def adc_to_ra_cart(adc_ri, range_res):
    adc_t = torch.from_numpy(adc_ri).float()
    ra_polar = adc_to_ra_image(adc_t).numpy()
    return ra_polar_to_cartesian(ra_polar, range_res)


def compute_cart_corr(ra1_cart, ra2_cart):
    n1 = minmax_normalize(ra1_cart)
    n2 = minmax_normalize(ra2_cart)
    return float(np.corrcoef(n1.ravel(), n2.ravel())[0, 1])


def load_gt_ra_cart(gt_adc_path, range_res):
    gt_raw = np.load(gt_adc_path)
    if gt_raw.ndim == 4:
        gt_chirp0 = gt_raw[0].transpose(1, 0, 2)
    elif gt_raw.ndim == 3:
        gt_chirp0 = gt_raw
    else:
        raise ValueError(f"Unexpected GT shape: {gt_raw.shape}")
    gt_ri = np.stack([gt_chirp0.real, gt_chirp0.imag], axis=-1).astype(np.float32)
    return adc_to_ra_cart(gt_ri, range_res)


# ============================================================
# Render-and-evaluate via CUDA backend
# ============================================================

def render_and_evaluate(ctx, tx_pos_mm, rx_pos_mm, boresight,
                        recompute_active_mask=True):
    """CUDA-backed replacement for the MC ``render_and_evaluate``.

    Signature changed: the MC version took an ``renderer`` object that
    already had its pose set externally by ``update_renderer_antennas``.
    This version takes the pose directly and mutates ``ctx.rast`` in-place.
    """
    apply_pose_to_ctx(ctx, tx_pos_mm, rx_pos_mm, boresight)
    return render_and_evaluate_cuda(ctx, recompute_active_mask=recompute_active_mask)


# ============================================================
# Grid search + refinement
# ============================================================

def grid_search_2dof(ctx, base_config, ra_gt_cart=None,
                     range_min=-2.0, range_max=2.0, range_step=0.25,
                     az_min=-10.0, az_max=10.0, az_step=1.0,
                     verbose=True):
    """Grid search over (delta_range_m, delta_azimuth_deg).

    ``ra_gt_cart`` is unused — the CUDA context caches a GPU-side GT.
    Kept in the signature for API symmetry with pass-1.
    """
    range_grid = np.arange(range_min, range_max + range_step / 2, range_step)
    az_grid = np.arange(az_min, az_max + az_step / 2, az_step)
    total = len(range_grid) * len(az_grid)

    if verbose:
        print(f"  Grid search (2DOF): {len(range_grid)} range x {len(az_grid)} az = {total} evals")

    best_cc = -999.0
    best_dr = 0.0
    best_da = 0.0
    all_results = []
    count = 0
    t0 = time.time()

    for dr_m in range_grid:
        for da_deg in az_grid:
            count += 1
            tx_mm, rx_mm, bs = apply_2dof_to_config(base_config, dr_m, da_deg)
            cc = render_and_evaluate(ctx, tx_mm, rx_mm, bs)
            all_results.append((float(dr_m), float(da_deg), float(cc)))
            if cc > best_cc:
                best_cc = cc
                best_dr = dr_m
                best_da = da_deg

            if verbose and (count % 25 == 0 or count == total):
                elapsed = time.time() - t0
                rate = count / max(elapsed, 1e-6)
                eta = (total - count) / rate if rate > 0 else 0
                print(f"    [{count}/{total}] best CC={best_cc:.4f} "
                      f"(range={best_dr:.3f}m, az={best_da:.2f}°) "
                      f"[{elapsed:.1f}s, ~{eta:.1f}s left]")

    if verbose:
        print(f"  Grid search done in {time.time()-t0:.1f}s; best={best_cc:.4f} @ "
              f"range={best_dr:.3f}m az={best_da:.2f}°")
    return [best_dr, best_da], best_cc, all_results


def grid_search_4dof(ctx, base_config,
                     range_min=-2.0, range_max=2.0, range_step=0.5,
                     az_min=-10.0, az_max=10.0, az_step=2.0,
                     elev_rot_min=-5.0, elev_rot_max=5.0, elev_rot_step=1.25,
                     azim_rot_min=-5.0, azim_rot_max=5.0, azim_rot_step=1.25,
                     prior_weight=0.0,
                     verbose=True):
    """Grid search over 4 DOF: (delta_range, delta_azimuth, rot_elev, rot_azim).

    Only tractable with the CUDA backend. Default grid is 9×11×9×9 = 8019 cells
    (~24 s on a 4090). Tighten via caller if a prior allows it.

    ``prior_weight`` adds a penalty ``λ × ||δ||²`` where
    ``||δ||² = dr² + (da/10)² + de² + dz²`` (range in metres, angles in
    degrees, range scaled so its contribution matches degree-scale). The
    penalty is subtracted from the cart_corr objective:
        adjusted_score = cc - prior_weight × ||δ||²
    Passing ``prior_weight=0`` recovers the unpenalised grid search.
    When ``base_config`` is the trajectory-smoothed pose, ``δ=0`` is the
    prior itself, so this penalty pulls the optimiser back toward the
    smoothed trajectory.
    """
    rg = np.arange(range_min, range_max + range_step / 2, range_step)
    ag = np.arange(az_min, az_max + az_step / 2, az_step)
    eg = np.arange(elev_rot_min, elev_rot_max + elev_rot_step / 2, elev_rot_step)
    zg = np.arange(azim_rot_min, azim_rot_max + azim_rot_step / 2, azim_rot_step)
    total = len(rg) * len(ag) * len(eg) * len(zg)

    if verbose:
        print(f"  Grid search (4DOF): {len(rg)}×{len(ag)}×{len(eg)}×{len(zg)} = {total} evals"
              + (f"  [prior_weight={prior_weight}]" if prior_weight else ""))

    best_cc = -999.0           # raw cart_corr of the best-adjusted pose
    best_adj = -999.0          # adjusted score (cc - λ·||δ||²)
    best_params = [0.0, 0.0, 0.0, 0.0]
    all_results = []
    count = 0
    t0 = time.time()

    for dr_m in rg:
        for da_deg in ag:
            for de_deg in eg:
                for dz_deg in zg:
                    count += 1
                    tx_mm, rx_mm, bs = apply_4dof_to_config(
                        base_config, dr_m, da_deg, de_deg, dz_deg)
                    cc = render_and_evaluate(ctx, tx_mm, rx_mm, bs)
                    penalty = 0.0
                    if prior_weight:
                        penalty = prior_weight * (
                            dr_m * dr_m
                            + (da_deg / 10.0) ** 2
                            + de_deg * de_deg
                            + dz_deg * dz_deg
                        )
                    adj = cc - penalty
                    all_results.append((float(dr_m), float(da_deg),
                                        float(de_deg), float(dz_deg),
                                        float(cc), float(adj)))
                    if adj > best_adj:
                        best_adj = adj
                        best_cc = cc
                        best_params = [float(dr_m), float(da_deg), float(de_deg), float(dz_deg)]

                    if verbose and (count % 200 == 0 or count == total):
                        elapsed = time.time() - t0
                        rate = count / max(elapsed, 1e-6)
                        eta = (total - count) / rate if rate > 0 else 0
                        print(f"    [{count}/{total}] best adj={best_adj:.4f} cc={best_cc:.4f} "
                              f"(dr={best_params[0]:.2f} da={best_params[1]:.1f} "
                              f"de={best_params[2]:.1f} dz={best_params[3]:.1f}) "
                              f"[{elapsed:.1f}s, ~{eta:.1f}s left]")

    if verbose:
        print(f"  4DOF grid done in {time.time()-t0:.1f}s; "
              f"best adj={best_adj:.4f} cc={best_cc:.4f}")
    return best_params, best_cc, all_results


def refine_2dof(ctx, base_config, initial_2dof, max_evals=30, verbose=True):
    """Nelder-Mead refinement over 2DOF params. Returns (best_params, best_cc, n_evals)."""
    eval_count = [0]
    best_cc = [-999.0]
    best_params = [list(initial_2dof)]

    def objective(x):
        eval_count[0] += 1
        tx_mm, rx_mm, bs = apply_2dof_to_config(base_config, x[0], x[1])
        cc = render_and_evaluate(ctx, tx_mm, rx_mm, bs)
        if cc > best_cc[0]:
            best_cc[0] = cc
            best_params[0] = [float(x[0]), float(x[1])]
        if verbose:
            print(f"    refine2 {eval_count[0]:3d}: CC={cc:.4f}  "
                  f"range={x[0]:.3f}m az={x[1]:.2f}°")
        return -cc

    minimize(
        objective, x0=np.array(initial_2dof), method="Nelder-Mead",
        options={"maxiter": max_evals, "maxfev": max_evals + 5,
                 "xatol": 0.02, "fatol": 0.001, "adaptive": True})
    return best_params[0], best_cc[0], eval_count[0]


def refine_4dof(ctx, base_config, initial_4dof, max_evals=60,
                prior_weight=0.0, verbose=True):
    """Nelder-Mead refinement over 4DOF params with optional prior penalty.

    The penalty structure matches ``grid_search_4dof``:
        adjusted_score = cc - prior_weight × (dr² + (da/10)² + de² + dz²)
    Nelder-Mead minimises ``-adjusted_score``. Returns
    ``(best_params, best_cc, n_evals)`` where ``best_cc`` is the raw
    cart_corr at the adjusted-best pose.
    """
    eval_count = [0]
    best_cc = [-999.0]
    best_adj = [-999.0]
    best_params = [list(initial_4dof)]

    def objective(x):
        eval_count[0] += 1
        tx_mm, rx_mm, bs = apply_4dof_to_config(base_config, x[0], x[1], x[2], x[3])
        cc = render_and_evaluate(ctx, tx_mm, rx_mm, bs)
        penalty = 0.0
        if prior_weight:
            penalty = prior_weight * (
                x[0] * x[0] + (x[1] / 10.0) ** 2
                + x[2] * x[2] + x[3] * x[3]
            )
        adj = cc - penalty
        if adj > best_adj[0]:
            best_adj[0] = adj
            best_cc[0] = cc
            best_params[0] = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
        if verbose:
            print(f"    refine4 {eval_count[0]:3d}: CC={cc:.4f} adj={adj:.4f}  "
                  f"dr={x[0]:.3f} da={x[1]:.2f} de={x[2]:.2f} dz={x[3]:.2f}")
        return -adj

    minimize(
        objective, x0=np.array(initial_4dof), method="Nelder-Mead",
        options={"maxiter": max_evals, "maxfev": max_evals + 5,
                 "xatol": 0.02, "fatol": 0.001, "adaptive": True})
    return best_params[0], best_cc[0], eval_count[0]


# ============================================================
# Save aligned configs
# ============================================================

def save_aligned_config(base_config, delta_range_m, delta_azimuth_deg, output_path):
    """Apply 2DOF params (zero rotation) and save aligned config to disk."""
    tx_mm, rx_mm, boresight = apply_2dof_to_config(
        base_config, delta_range_m, delta_azimuth_deg)
    config = json.loads(json.dumps(base_config))
    for i, tx in enumerate(config["tx_array"]):
        tx["pos_mm"] = tx_mm[i].tolist()
        tx["boresight"] = boresight.tolist()
    for i, rx in enumerate(config["rx_array"]):
        rx["pos_mm"] = rx_mm[i].tolist()
        rx["boresight"] = boresight.tolist()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path


def save_aligned_config_4dof(base_config, params_4dof, output_path):
    """Apply 4DOF params and save aligned config to disk."""
    tx_mm, rx_mm, boresight = apply_4dof_to_config(base_config, *params_4dof)
    config = json.loads(json.dumps(base_config))
    for i, tx in enumerate(config["tx_array"]):
        tx["pos_mm"] = tx_mm[i].tolist()
        tx["boresight"] = boresight.tolist()
    for i, rx in enumerate(config["rx_array"]):
        rx["pos_mm"] = rx_mm[i].tolist()
        rx["boresight"] = boresight.tolist()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path


def save_aligned_config_from_pose(base_config, tx_pos_mm, rx_pos_mm,
                                  boresight, output_path):
    """Save an aligned config directly from explicit pose tensors (used by
    pass-2 when the winner is the trajectory-prior pose itself)."""
    config = json.loads(json.dumps(base_config))
    for i, tx in enumerate(config["tx_array"]):
        tx["pos_mm"] = list(map(float, tx_pos_mm[i]))
        tx["boresight"] = list(map(float, boresight))
    for i, rx in enumerate(config["rx_array"]):
        rx["pos_mm"] = list(map(float, rx_pos_mm[i]))
        rx["boresight"] = list(map(float, boresight))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path


# Re-export the CUDA context API so downstream callers can
# `from cascaded_renderer import AlignmentCtx, build_alignment_context, …`
__all__ = [
    'SCENES', 'TX_PATTERN', 'RX_PATTERN',
    '_norm', '_axis_angle_to_matrix',
    'apply_2dof_to_config', 'apply_4dof_to_config',
    'minmax_normalize', 'adc_to_ra_cart', 'compute_cart_corr',
    'load_gt_ra_cart',
    'render_and_evaluate',
    'grid_search_2dof', 'grid_search_4dof',
    'refine_2dof', 'refine_4dof',
    'save_aligned_config', 'save_aligned_config_4dof',
    'save_aligned_config_from_pose',
    'AlignmentCtx', 'build_alignment_context', 'apply_pose_to_ctx',
    'render_and_evaluate_cuda', 'destroy_alignment_context',
]
