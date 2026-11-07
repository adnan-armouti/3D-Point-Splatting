"""mm3DPS — c6 hemisphere integral training (point-based primitive).

Single training mode: visible-weighted FPS init + uniform hemisphere weights.

Pipeline:
  1. Load pcl.npy (1.5–2.5M points). Columns 0–2 are xyz, 3–5 are unit normals.
  2. FOV restrict (cos_bore > cos_bore_min, within radar range).
  3. RX-side ray-traced visibility against the Mitsuba scene (the ONLY mesh use).
  4. Cosine-hemisphere importance resample (prob ∝ cos_bore).
  5. FPS to target_n.
  6. Train per-point quaternions (→ surface normals) + 6 material params
     + global TX/RX antenna E/H planes. Each parameter group is gated by
     a LEARN_* flag (matches the differentiable renderer's convention). Positions are FROZEN by
     default — they come from the LiDAR pcl and stay fixed.

Primitive: a POINT, not a Gaussian. There is no scale and no opacity in
the model. The renderer reads only position, normal, and material per
point; the hemisphere weight on the active set is uniform = 1.0.

Ray tracing footprint (one-shot at init, freed before training loop):
  - One per-point RX-visibility ray test on the full FOV-filtered point
    cloud (~1.5M rays, batched). Filters out occluded points before FPS,
    so every point in the model is RX-visible by construction.
  - No TX shadow mask. The A/B test
    showed shadow OFF is at least as good as shadow ON (mean +0.0068,
    6/7 scenes improved or flat).
  - No reservoir sampler. No per-Gaussian post-FPS sanity-check ray test.
  - The renderer itself does ZERO ray tracing.

"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mm3DPS.data.ra_utils import adc_to_ra_complex, txrx_to_vx_chirps_torch

from mm3DPS.rasterizer import (
    reparameterize_torch,
    inverse_reparameterize_torch,
)

DEVICE = "cuda:0"

# =========================================================================
# Learnable parameter toggles (matches the differentiable renderer's LEARN_* pattern in train.py)
# =========================================================================
# Each flag controls (a) whether the underlying parameter has requires_grad
# and (b) whether it appears in the optimizer's param_groups. Flip a flag,
# and the parameter is frozen / unfrozen uniformly.
LEARN_POSITIONS = False    # frozen — pcl positions are treated as fixed reference
LEARN_NORMALS   = True     # quaternion → surface normal; refines noisy pcl normals
LEARN_MATERIALS = True     # 6 ITU/Cook-Torrance params per point
LEARN_PATTERNS  = False    # TX/RX antenna E/H planes — PERMANENT False per user direction (2026-04-13)

# When True (default after the raw-MSE plan), skip the differentiable renderer-trained antenna
# pattern injection at init and use factory patterns from the MMWCAS .npy
# files instead. the differentiable renderer-trained patterns absorb per-scene amplitude bias and
# break the cross-scene scale uniformity required for raw MSE. See
# 

# Default starting per-point material (ITU concrete).
# sigma_h lowered from 1e-4 (100 μm) to 5e-5 (50 μm) to stay inside the SPM
# validity regime (h_max_1 = 0.1/k ≈ 62 μm at 77 GHz). At 100 μm the
# enforce_spm_validity clamp routes grad through a constant, making sigma_h
# dead in backward — confirmed via gradient/drift/Fisher diagnostics.
ITU_CONCRETE = np.array([5.31, 0.0326, 5e-5, 5e-3, 0.5, 0.15], dtype=np.float32)


# =========================================================================
# Range profile -> RA image (azimuth FFT only, no range FFT)
# =========================================================================

# Module-level cache for the azimuth Hann window. Keyed by (length, device,
# complex dtype). The window depends only on the static array geometry, so
# computing it once per process is plenty.
_HANN_CACHE = {}


def _get_hann_window(num_vx, device, complex_dtype):
    key = (num_vx, str(device), complex_dtype)
    w = _HANN_CACHE.get(key)
    if w is None:
        # complex_dtype.real returns float32 if complex64, float64 if complex128
        real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
        w = torch.hann_window(num_vx, device=device, dtype=real_dtype).to(complex_dtype)[:, None]
        _HANN_CACHE[key] = w
    return w


def range_profile_to_ra(rp_real, rp_imag):
    """Convert complex range profiles to RA image via azimuth FFT only."""
    rp_ri = torch.stack([rp_real, rp_imag], dim=-1)              # (12, 16, K, 2)
    rp_ri = rp_ri.permute(1, 0, 2, 3)                            # (RX=16, TX=12, K, 2)
    rp_c = torch.complex(rp_ri[..., 0].contiguous(),
                          rp_ri[..., 1].contiguous())              # (16, 12, 256)
    rp_c = rp_c.unsqueeze(0)                                      # (1, 16, 12, 256)

    vx = txrx_to_vx_chirps_torch(rp_c)                            # (1, 7, 86, 256)
    ra = vx[0, 0, :, :]                                           # (86, 256)

    num_vx = ra.size(0)
    ra = ra * _get_hann_window(num_vx, ra.device, ra.dtype)
    ra = torch.fft.ifftshift(ra, dim=0)
    ra = torch.fft.fft(ra, n=128, dim=0)
    ra = ra[1:, :]
    ra = torch.fft.fftshift(ra, dim=0)
    return ra  # (127, 256) complex


def range_profile_to_ra_mag(rp_real, rp_imag):
    return range_profile_to_ra(rp_real, rp_imag).abs().float()


# =========================================================================
# GPU eval helpers (replace scipy griddata + numpy corrcoef)
# =========================================================================

def build_polar_to_cart_grid(n_az, n_range, range_res, grid_res=400, device=DEVICE):
    """Pre-compute the (grid_res-1, grid_res-1, 2) sample grid for grid_sample.

    Replicates the indexing of the differentiable renderer's ra_polar_to_cartesian (which uses
    scipy.griddata with linear interpolation) using F.grid_sample bilinear
    interpolation on a regular polar→cart grid.

    Polar grid layout (matches range_profile_to_ra output):
        shape   (n_az, n_range)        e.g. (127, 256)
        az bin i → sin(angle) = (i - n_az/2) * 2/(n_az+1)  ← linear in sin(angle)
        range bin j → r = j * range_res

    Cartesian grid:
        xi ∈ [-range_width, range_width]
        yi ∈ [0, range_depth]
        range_depth = n_range * range_res
        range_width = range_depth / 2

    For each (xi, yi):
        r       = sqrt(xi² + yi²)            → range bin
        sin_θ   = xi / r                     → az bin (since the polar grid
                                              is uniform in sin(angle))

    Returned grid is normalized to [-1, 1] for F.grid_sample with
    align_corners=True. The grid samples the polar image (interpreted as
    a 4D tensor of shape (1, 1, n_az, n_range)).

    The the differentiable renderer reference also (a) drops the last row/col, (b) flips the
    azimuth axis (`zi[:, ::-1]`). We replicate both: drop and flip applied
    to the grid construction so the resulting cart image lines up with
    the CPU reference.
    """
    range_depth = n_range * range_res
    range_width = range_depth / 2.0

    # the differentiable renderer uses linspace(grid_res) then drops the last col/row → (grid_res-1, grid_res-1)
    xi = torch.linspace(-range_width, range_width, grid_res, device=device)
    yi = torch.linspace(0.0, range_depth, grid_res, device=device)
    xi = xi[:-1]
    yi = yi[:-1]
    xx, yy = torch.meshgrid(xi, yi, indexing='xy')   # (grid_res-1, grid_res-1)

    # the differentiable renderer reverses the cart image along the azimuth (last) axis at the end:
    #   `return zi[:, ::-1]`
    # We bake the flip into the grid by flipping xx along its last axis.
    xx = torch.flip(xx, dims=[-1])

    r = torch.sqrt(xx * xx + yy * yy).clamp(min=1e-9)        # (h, w)
    sin_theta = (xx / r).clamp(-1.0, 1.0)                    # (h, w)

    # Polar grid az index from sin(theta).
    # the differentiable renderer's polar grid: bin k → sin(angle) = (-num_angle_bins/2 + 1 + k) * 2/num_angle_bins
    # where num_angle_bins = n_az + 1. For n_az=127 → num_angle_bins=128:
    #   bin 0   → sin = -126/128 ≈ -0.984
    #   bin 63  → sin = 0
    #   bin 126 → sin = +126/128 ≈ +0.984
    # Inverting: k = (n_az - 1)/2 + sin_theta * (n_az + 1)/2
    az_idx = (n_az - 1) / 2.0 + sin_theta * (n_az + 1) / 2.0

    # Polar grid range index: r_idx = r / range_res
    range_idx = r / range_res

    # Normalize to [-1, 1] for F.grid_sample with align_corners=True.
    # grid coords: (x in [-1,1] = column = range_idx, y in [-1,1] = row = az_idx)
    # F.grid_sample uses (x, y) order in the last dim where x is the WIDTH dim.
    # We treat the polar tensor as (1, 1, H=n_az, W=n_range), so:
    #   sample x ↔ range, sample y ↔ azimuth
    range_norm = 2.0 * range_idx / (n_range - 1) - 1.0
    az_norm    = 2.0 * az_idx    / (n_az - 1)    - 1.0

    grid = torch.stack([range_norm, az_norm], dim=-1)        # (h, w, 2)
    return grid.unsqueeze(0)                                 # (1, h, w, 2)


def polar_to_cart_torch(ra_polar, sample_grid):
    """GPU polar → cartesian via F.grid_sample.

    Args:
        ra_polar: (n_az, n_range) float tensor on GPU
        sample_grid: (1, h, w, 2) precomputed by build_polar_to_cart_grid

    Returns: (h, w) float tensor on GPU.
    """
    img = ra_polar.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W)
    out = F.grid_sample(
        img, sample_grid, mode='bilinear',
        padding_mode='zeros', align_corners=True)
    return out[0, 0]                                          # (h, w)


def cart_corr_torch(rend_cart, gt_cart_normalized):
    """GPU Pearson correlation between rendered cart RA and pre-normalized GT.

    Both inputs are 2D float tensors. The GT is expected to be already
    min-max normalized (cached once at init). The rendered tensor is
    normalized inside this function.
    """
    rend_min = rend_cart.min()
    rend_max = rend_cart.max()
    rend_range = (rend_max - rend_min).clamp(min=1e-30)
    rend_norm = (rend_cart - rend_min) / rend_range

    rend_flat = rend_norm.flatten()
    gt_flat = gt_cart_normalized.flatten()
    rend_mean = rend_flat.mean()
    gt_mean = gt_flat.mean()
    rend_centered = rend_flat - rend_mean
    gt_centered = gt_flat - gt_mean

    num = (rend_centered * gt_centered).sum()
    den = torch.sqrt(
        (rend_centered ** 2).sum() * (gt_centered ** 2).sum()
    ).clamp(min=1e-30)
    return num / den


def precompute_gt_loss_norm(gt_adc_ri, loss_type='mse_raw'):
    """One-shot computation of the GT tensor for the loss path.

    The GT does not change during training; the loss used to recompute it
    every iter (~5 ms wasted/iter). Call this once before the training loop
    and pass the result into compute_ra_loss_rp.

    For loss_type='mse'    : returns GT min-max normalized RA magnitude.
    For loss_type='pearson': same (Pearson is scale/shift invariant, so
                             normalization doesn't affect the result).
    For loss_type='mse_raw': returns raw GT RA magnitude (no normalization).
    """
    with torch.no_grad():
        ra_gt = adc_to_ra_complex(gt_adc_ri)
        ra_gt_mag = torch.abs(ra_gt)
        if loss_type == 'mse_raw':
            return ra_gt_mag.detach()
        mn, mx = ra_gt_mag.min(), ra_gt_mag.max()
        return ((ra_gt_mag - mn) / (mx - mn).clamp(min=1e-30)).detach()


def compute_ra_loss_rp(rp_real, rp_imag, gt_cached, loss_type='mse_raw'):
    """RA loss between rendered range profiles and the cached GT.

    loss_type:
      'mse'     — min-max-normalized MSE on RA magnitude (legacy)
      'pearson' — 1 - Pearson(rendered_flat, gt_flat); scale/shift invariant
      'mse_raw' — MSE on raw |RA| magnitude, no normalization. Requires the
                  forward-model scale to match GT scale (see the 100x C_radar
                  boost); otherwise the loss is dominated by the mismatch.
    """
    ra_rendered = range_profile_to_ra(rp_real, rp_imag)
    ra_rend_mag = torch.abs(ra_rendered)

    if loss_type == 'mse':
        mn = ra_rend_mag.min()
        mx = ra_rend_mag.max()
        rend_norm = (ra_rend_mag - mn) / (mx - mn).clamp(min=1e-30)
        loss = torch.mean((rend_norm - gt_cached) ** 2)
        return loss, {"ra_mse": loss.item()}
    elif loss_type == 'pearson':
        r_flat = ra_rend_mag.reshape(-1)
        g_flat = gt_cached.reshape(-1)
        r_mean = r_flat.mean()
        g_mean = g_flat.mean()
        r_c = r_flat - r_mean
        g_c = g_flat - g_mean
        num = (r_c * g_c).sum()
        den = torch.sqrt((r_c * r_c).sum() * (g_c * g_c).sum()).clamp(min=1e-30)
        corr = num / den
        loss = 1.0 - corr
        return loss, {"pearson_loss": loss.item()}
    elif loss_type == 'mse_raw':
        # Raw MSE on |RA| magnitude — depends on rendered scale matching GT
        # scale. Divide by the square of GT mean to bring the loss into an
        # O(1) range regardless of absolute scale (helps LR tuning) without
        # introducing any scale invariance into the gradient path.
        gt_scale = gt_cached.mean().detach().clamp(min=1e-30) ** 2
        loss = ((ra_rend_mag - gt_cached) ** 2).mean() / gt_scale
        return loss, {"ra_mse_raw": loss.item()}
    else:
        raise ValueError(f"Unknown loss_type {loss_type}")


# =========================================================================
# Point primitive model
# =========================================================================

class PointPrimitives(torch.nn.Module):
    """Point-based primitives for radar rendering.

    Per-point state: position(3) + quaternion(4) + raw_materials(6) = 13 floats.
    Each parameter's `requires_grad` is set from the LEARN_* flags above so that
    flipping a flag uniformly freezes/unfreezes the parameter for both autograd
    and the optimizer (matches the differentiable renderer's LEARN_* convention).

    Plus 1444 global antenna pattern values (TX/RX × E/H × 361 samples), gated
    by LEARN_PATTERNS in the training loop.
    """

    def __init__(self, N, device=DEVICE):
        super().__init__()
        self.device = device
        self.positions = torch.nn.Parameter(
            torch.zeros(N, 3, device=device), requires_grad=LEARN_POSITIONS)
        self.rotations = torch.nn.Parameter(
            torch.zeros(N, 4, device=device), requires_grad=LEARN_NORMALS)
        self.raw_materials = torch.nn.Parameter(
            torch.zeros(N, 6, device=device), requires_grad=LEARN_MATERIALS)

        with torch.no_grad():
            self.rotations[:, 0] = 1.0  # identity quaternion [w,x,y,z]

    @property
    def N(self):
        return self.positions.shape[0]

    def get_normals(self):
        """(N, 3) surface normals = third column of rotation matrix."""
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        nx = 2 * (x * z + w * y)
        ny = 2 * (y * z - w * x)
        nz = 1 - 2 * (x * x + y * y)
        return torch.stack([nx, ny, nz], dim=-1)


# =========================================================================
# Geometry helpers (no mesh required — pcl carries normals directly)
# =========================================================================

def _normals_to_quaternions(normals):
    """Convert (N, 3) unit normals to (N, 4) quaternions [w,x,y,z] that
    rotate the local +z axis onto the surface normal."""
    N = len(normals)
    nrm = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    n = normals / nrm

    # axis = cross(up=(0,0,1), n) = (-n_y, n_x, 0)
    axis = np.zeros((N, 3), dtype=np.float32)
    axis[:, 0] = -n[:, 1]
    axis[:, 1] = n[:, 0]

    axis_len = np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    axis = axis / axis_len

    dot = n[:, 2].copy()
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    half_angle = angle / 2.0

    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = np.cos(half_angle)
    quats[:, 1] = axis[:, 0] * np.sin(half_angle)
    quats[:, 2] = axis[:, 1] * np.sin(half_angle)
    quats[:, 3] = axis[:, 2] * np.sin(half_angle)

    parallel = dot > 0.9999
    quats[parallel] = [1, 0, 0, 0]
    anti = dot < -0.9999
    quats[anti] = [0, 1, 0, 0]
    return quats


def _farthest_point_sampling(pts, n_samples):
    """GPU farthest-point sampling.

    Sequentially picks n_samples points such that each new pick is the
    one farthest (Euclidean) from all already-picked points. Inherently
    sequential — no parallelism across iterations possible.

    Three optimizations vs the textbook implementation:
      1. Use squared distances throughout (avoids per-iter sqrt). Argmax of
         d² is the same as argmax of d.
      2. `torch.index_select` instead of fancy indexing `pts[idx]` for the
         single-row anchor lookup. Significantly faster on CUDA — fancy
         indexing dispatches a slower kernel for variable-length gathers.
      3. Skip the explicit `min_dist[idx] = 0` after each pick. The next
         (pts - pts[idx])² computation already produces a 0 at the picked
         row, and the torch.minimum() carries it through.

    Empirical (130K → 50K, RTX 4090): textbook 2.94 s → this version 1.77 s
    (~40% faster). Sample diversity unchanged (same average pairwise
    distance, same bounding-box coverage as the textbook variant).
    """
    N = pts.shape[0]
    if n_samples >= N:
        return torch.arange(N, device=pts.device)

    selected = torch.empty(n_samples, dtype=torch.long, device=pts.device)
    selected[0] = 0
    diff = pts - pts[0:1]
    min_dist_sq = (diff * diff).sum(-1)

    for j in range(1, n_samples):
        idx = torch.argmax(min_dist_sq)
        selected[j] = idx
        anchor = torch.index_select(pts, 0, idx.unsqueeze(0))
        diff = pts - anchor
        d_sq = (diff * diff).sum(-1)
        min_dist_sq = torch.minimum(min_dist_sq, d_sq)
    return selected


# =========================================================================
# Init-time ray tracing (the only ray tracing)
# =========================================================================

def _ray_test_visibility_batched(points_np, rx_center_np, mi_scene, chunk=200000):
    """Batched RX-side visibility ray test for the full point cloud.

    For each point, casts a ray from the point toward the RX array center.
    If the ray hits any mesh surface before reaching the RX, the point is
    occluded. This is the ONE ray-tracing call we keep — it filters
    the raw point cloud down to only RX-visible points before FPS.
    """
    N = len(points_np)
    visible = np.zeros(N, dtype=bool)
    eps = 1e-3
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        pts = points_np[i:j]
        delta = rx_center_np - pts
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)
        direction = delta / dist[:, None]
        origins = pts + eps * direction
        o = mi.Point3f(
            mi.Float(origins[:, 0]), mi.Float(origins[:, 1]), mi.Float(origins[:, 2]))
        d = mi.Vector3f(
            mi.Float(direction[:, 0]), mi.Float(direction[:, 1]), mi.Float(direction[:, 2]))
        rays = mi.Ray3f(o, d)
        rays.maxt = mi.Float(dist - 2 * eps)
        occluded = mi_scene.ray_test(rays)
        visible[i:j] = ~np.array(occluded)
    return visible


# =========================================================================
# FOV culling (no mesh, pure radar geometry)
# =========================================================================

def cull_gaussians(model, rast, cos_threshold=0.05):
    """Pre-filter Gaussians to only those visible to the radar (FOV only)."""
    with torch.no_grad():
        positions = model.positions
        normals = model.get_normals()

        radar_center = (rast.tx_positions.mean(0) + rast.rx_positions.mean(0)) / 2
        boresight = rast.tx_boresights.mean(0)
        boresight = boresight / boresight.norm().clamp(min=1e-6)

        to_radar = radar_center - positions
        dist = to_radar.norm(dim=-1)
        to_radar_dir = to_radar / dist.clamp(min=1e-6).unsqueeze(-1)

        cos_normal = torch.abs((normals * to_radar_dir).sum(-1))

        from_radar = -to_radar_dir
        cos_bore = (from_radar * boresight).sum(-1)

        max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))

        active = (
            (dist > 1.5) &
            (dist < max_range) &
            (cos_bore > 0.0) &
            (cos_normal > cos_threshold)
        )
        return active


# =========================================================================
# Init: visible-weighted FPS from raw point cloud
# =========================================================================

def init_visible_weighted(scene, rast, target_n=90000,
                          cos_bore_min=0.1761,
                          n_intermediate=None,
                          device=DEVICE,
                          return_pool=False,
                          enable_cull=True,
                          enable_occlusion=True,
                          enable_cosine_resample=True,
                          enable_fps=True):
    # Default: make the intermediate pool large enough that the cosine
    # importance resample step doesn't bottleneck FPS selection. The
    # n_resample = min(n_intermediate, 3*target_n) line then picks 3*target_n.
    if n_intermediate is None:
        n_intermediate = max(200000, 3 * target_n)
    """Initialize Gaussians via visible-weighted FPS over the raw point cloud.

    Pipeline:
      1. Load pcl.npy. Columns 0–2 are xyz, 3–5 are unit normals.
      2. FOV restrict by cos_bore_min and radar range.
      3. RX-side visibility ray test (the only mesh use at init).
      4. Cosine-hemisphere importance resample (prob ∝ cos_bore).
      5. FPS to target_n.
      6. Build the PointPrimitives model with normals from the pcl.

    Returns the PointPrimitives model. The caller is responsible for
    config/pattern_data loading.
    """
    pcl_path = rast._mesh_file.replace('scene/mesh.ply', 'scene/pcl.npy')
    if not os.path.exists(pcl_path):
        raise FileNotFoundError(
            f"pcl.npy not found at {pcl_path}. Training requires the point cloud "
            f"(with normals in columns 3–5).")

    pcl = np.load(pcl_path)
    if pcl.shape[1] < 6:
        raise ValueError(
            f"{pcl_path} has shape {pcl.shape}; expected at least 6 columns "
            f"(xyz + unit normals).")

    xyz_full = pcl[:, :3].astype(np.float32)
    nrm_full = pcl[:, 3:6].astype(np.float32)
    nrm_full = nrm_full / np.maximum(
        np.linalg.norm(nrm_full, axis=1, keepdims=True), 1e-12)

    print(f"  [init] Full pcl: {len(xyz_full)} points")

    # --- Step 1: FOV restrict ---
    # Ablation hook: `enable_cull=False` drops the azimuth
    # cone test and the range gate, keeping every point of the raw pcl that
    # passes the dist > 1.5 m near-field guard (kept unconditionally to
    # avoid d->0 singularities downstream).
    rx_center = rast.rx_positions.mean(dim=0).cpu().numpy()
    boresight = rast.tx_boresights.mean(dim=0).cpu().numpy()
    boresight = boresight / max(np.linalg.norm(boresight), 1e-8)

    delta = xyz_full - rx_center
    dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)
    dir_to_point = delta / dist[:, None]
    cos_bore_full = (dir_to_point * boresight).sum(axis=-1)

    max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))

    if enable_cull:
        fov_mask = (cos_bore_full > cos_bore_min) & (dist > 1.5) & (dist < max_range)
    else:
        fov_mask = dist > 1.5
    xyz_fov = xyz_full[fov_mask]
    nrm_fov = nrm_full[fov_mask]
    cos_bore_fov = cos_bore_full[fov_mask]
    print(f"  [init] After FOV: {len(xyz_fov)} points (enable_cull={enable_cull})")

    # --- Step 2: RX visibility ray tracing (the only mesh use) ---
    # Ablation hook: skip the Mitsuba ray cast, keep all
    # FOV-survivors. Saves ~2 s of init but lets back-of-wall / occluded
    # points pollute the optimisation pool.
    if enable_occlusion:
        rast.load_mi_scene()
        visible = _ray_test_visibility_batched(xyz_fov, rx_center, rast._mi_scene)
    else:
        visible = np.ones(len(xyz_fov), dtype=bool)
    xyz_vis = xyz_fov[visible]
    nrm_vis = nrm_fov[visible]
    cos_bore_vis = cos_bore_fov[visible]
    print(f"  [init] After RX visibility: {len(xyz_vis)} points (enable_occlusion={enable_occlusion})")

    if len(xyz_vis) < 100:
        raise RuntimeError(
            f"init: only {len(xyz_vis)} points survived visibility — "
            f"check the Mitsuba scene and FOV settings.")

    # --- Step 3: Cosine-hemisphere importance resample ---
    # Ablation hook: swap the cos_bore-proportional
    # importance weights for uniform weights (every visible point equally
    # likely to be drawn into the FPS pool).
    if enable_cosine_resample:
        weights = np.maximum(cos_bore_vis, 0.01)
    else:
        weights = np.ones_like(cos_bore_vis)
    probs = weights / weights.sum()
    n_resample = min(n_intermediate, 3 * target_n)
    rng = np.random.default_rng(42)
    sampled_idx = rng.choice(len(xyz_vis), size=n_resample, replace=True, p=probs)
    unique_idx = np.unique(sampled_idx)
    xyz_weighted = xyz_vis[unique_idx]
    nrm_weighted = nrm_vis[unique_idx]
    print(f"  [init] After cosine importance resample: {len(xyz_weighted)} points "
          f"(enable_cosine_resample={enable_cosine_resample})")

    # --- Step 4: FPS to target_n ---
    # Ablation hook: replace farthest-point sampling with
    # uniform random subsampling of the same `target_n` points (seed 42 for
    # reproducibility). Removes the spatial-coverage prior FPS provides.
    if len(xyz_weighted) > target_n:
        if enable_fps:
            pts_t = torch.from_numpy(xyz_weighted).to(device)
            sel = _farthest_point_sampling(pts_t, target_n).cpu().numpy()
        else:
            sel = rng.choice(len(xyz_weighted), size=target_n, replace=False)
        xyz = xyz_weighted[sel]
        normals = nrm_weighted[sel]
    else:
        xyz = xyz_weighted
        normals = nrm_weighted
        sel = None
    print(f"  [init] After FPS: {len(xyz)} points (enable_fps={enable_fps})")

    N = len(xyz)

    # --- Step 5: Build the model ---
    model = PointPrimitives(N, device=device)
    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(xyz).to(device))

        raw_default = inverse_reparameterize_torch(ITU_CONCRETE)
        model.raw_materials.copy_(
            torch.from_numpy(np.tile(raw_default, (N, 1))).to(device))

        quats = _normals_to_quaternions(normals)
        model.rotations.copy_(torch.from_numpy(quats).to(device))

    if return_pool:
        # Post-visibility pool (post-FOV + post-visibility ray test, NO
        # cosine resample, NO FPS). Size is typically several hundred
        # thousand points — the largest honest "radar-visible LiDAR"
        # candidate set. Used by S4 pool-kNN child spawning so children
        # are drawn from un-FPS-biased LiDAR neighbours of high-Fisher
        # parents (post-resample would bias toward cos_bore-aligned
        # points already close to the FPS-chosen training set).
        #
        # `pool_fps_sel` gives the indices into (xyz_vis, nrm_vis) that
        # map each of the target_n training slots to a real pool point.
        # Derivation:
        #   training pt i = xyz[i]
        #                 = xyz_weighted[sel[i]]   (sel: fps index → weighted)
        #                 = xyz_vis[unique_idx[sel[i]]]
        # so pool index for slot i = unique_idx[sel[i]].
        if len(xyz_weighted) > target_n:
            pool_fps_sel = unique_idx[sel].astype(np.int64)
        else:
            pool_fps_sel = unique_idx.astype(np.int64)
        return model, xyz_vis, nrm_vis, pool_fps_sel
    return model


# =========================================================================
# Renderer wrapper (uniform hemisphere weights)
# =========================================================================

def render_gaussians(model, rast, vertex_areas, active_mask=None,
                     shadow_mask=None, detach_phase=True,
                     positions_override=None,
                     use_cuda_kernels=True, psf_spread=None):
    """Range-profile splatting renderer wrapper.

    `vertex_areas` carries the precomputed per-point hemisphere weight
    (uniform = 1.0 for active points, 0.0 for inactive).

    `positions_override`: if provided, an (N, 3) tensor used as positions
    instead of `model.positions`. Used by MLP-A to inject pose-conditioned
    position deformations without mutating model state. Must have the
    same N as the model, before any active_mask is applied.

    Renders the full active set in one shot — no chunking. The dominant
    intermediate is the (M, n_tx, n_rx) BSDF tensor which at our scales
    (M ≈ 45K active, n_tx=12, n_rx=16, fp32) is ~35 MB. Total renderer
    working set is well under 1 GB on a 4090.
    """
    from mm3DPS.rasterizer_factorized import render_factorized

    positions = positions_override if positions_override is not None else model.positions
    normals = model.get_normals()
    raw_materials = model.raw_materials

    # Point-based primitive: no opacity multiply. vertex_areas is already
    # the binary 1.0/0.0 active mask for the hemisphere weight.
    areas = vertex_areas

    sm = None
    if shadow_mask is not None and active_mask is not None:
        sm = shadow_mask[active_mask]
    elif shadow_mask is not None:
        sm = shadow_mask

    if active_mask is not None:
        positions = positions[active_mask]
        normals = normals[active_mask]
        raw_materials = raw_materials[active_mask]
        areas = areas[active_mask]

    return render_factorized(
        positions, normals, areas, raw_materials, rast,
        reparameterize_torch, detach_phase=detach_phase,
        shadow_mask=sm,
        use_cuda_kernels=use_cuda_kernels,
        psf_spread=psf_spread)


# =========================================================================
# Training loop (LR schedule + grad clip helpers inlined)
# =========================================================================

def get_lr_scale(iteration, warmup_iters=5, warmup_factor=0.3,
                 total_iters=None, decay_start=100, min_lr_factor=0.01):
    """Linear warmup -> cosine decay to min_lr_factor."""
    if iteration < warmup_iters:
        return warmup_factor + (1.0 - warmup_factor) * (iteration / max(warmup_iters, 1))
    if total_iters is None or iteration < decay_start:
        return 1.0
    progress = (iteration - decay_start) / max(total_iters - decay_start, 1)
    progress = min(progress, 1.0)
    return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1 + math.cos(math.pi * progress))


def rms_clip_grad(param, max_rms):
    """Per-parameter RMS gradient clipping.

    Data-dependent Python branch on `rms > max_rms`. Not CUDA-graph
    compatible — use `rms_clip_grad_graph_safe` for graph-captured
    training loops.
    """
    if param.grad is None:
        return
    g = param.grad.data
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    rms = torch.sqrt(torch.mean(g ** 2))
    if rms > max_rms:
        g.mul_(max_rms / rms)
    param.grad.data = g


