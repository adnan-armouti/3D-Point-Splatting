"""CUDA backend for cascade alignment.

Replaces the MC rendering path in ``cascaded_renderer.py`` with the v5
CUDA-accelerated renderer. Each alignment context holds a frozen
scene (FPS'd point cloud + default ITU concrete materials) and a
mutable pose; grid search and refinement sweep candidate poses by
mutating the rast's TX/RX position and boresight tensors in-place.

Timing on 1× RTX 4090 (frame 135, 30k-point model, 4-DOF grid at
11×11×5×5 = 3025 cells):
    build_alignment_context :  ~6 s  (one-time per frame)
    render_and_evaluate_cuda: ~3 ms  (per candidate)
    full 3025-cell grid     :  ~9 s  (vs hours with the MC renderer)
"""

import os
import sys
import numpy as np
import torch

import mitsuba as mi
if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm3DPS.data.ra_utils import adc_to_ra_complex
from mm3DPS.data.io_utils import compute_range_res_from_cfg

from mm3DPS.rasterizer import Rasterizer
from mm3DPS.train_gaussian import (
    DEVICE,
    init_visible_weighted,
    cull_gaussians,
    render_gaussians,
    range_profile_to_ra_mag,
    build_polar_to_cart_grid,
    polar_to_cart_torch,
    cart_corr_torch,
)


# ---------------------------------------------------------------------------
# Alignment context
# ---------------------------------------------------------------------------

class AlignmentCtx:
    """Stateful alignment context. Holds the rast (mutable pose), the
    frozen scene model (positions + materials), and the precomputed GT
    for cart_corr.
    """
    def __init__(self, rast, model, vertex_areas, sample_grid,
                 gt_cart_norm, range_res, device, scene_name, frame_idx):
        self.rast = rast
        self.model = model
        self.vertex_areas = vertex_areas
        self.sample_grid = sample_grid
        self.gt_cart_norm = gt_cart_norm
        self.range_res = range_res
        self.device = device
        self.scene_name = scene_name
        self.frame_idx = frame_idx




def build_alignment_context(
    config_file,
    mesh_file=None,
    tx_pattern_file=None,
    rx_pattern_file=None,
    gt_adc_path=None,
    device=DEVICE,
    target_n=30000,
    chirp_idx=0,
    verbose=False,
):
    """Build a one-time alignment context for a single frame.

    Arguments default to paths derived from ``config_file``:
        mesh_file        = <scene>/scene/mesh.ply
        gt_adc_path      = <scene>/radar/cascaded_frame_<F>.npy (from config name)
        tx_pattern_file  = assets/antenna_pattern/MMWCAS/tx1_76.npy
        rx_pattern_file  = assets/antenna_pattern/MMWCAS/rx1_76.npy
    """
    scene_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
    scene_name = os.path.basename(scene_dir)

    if mesh_file is None:
        mesh_file = os.path.join(scene_dir, 'scene', 'mesh.ply')
    if tx_pattern_file is None:
        tx_pattern_file = os.path.join(
            PROJECT_ROOT, 'assets', 'antenna_pattern', 'MMWCAS', 'tx1_76.npy')
    if rx_pattern_file is None:
        rx_pattern_file = os.path.join(
            PROJECT_ROOT, 'assets', 'antenna_pattern', 'MMWCAS', 'rx1_76.npy')
    if gt_adc_path is None:
        # Parse frame from config filename: cascaded_frame_<F>.json or
        # cascaded_frame_<F>_aligned*.json
        base = os.path.basename(config_file)
        core = base.split('cascaded_frame_')[-1].split('_')[0].split('.')[0]
        frame_idx = int(core)
        gt_adc_path = os.path.join(
            scene_dir, 'radar', f'cascaded_frame_{frame_idx}.npy')
    else:
        base = os.path.basename(config_file)
        core = base.split('cascaded_frame_')[-1].split('_')[0].split('.')[0]
        frame_idx = int(core)

    rast = Rasterizer(
        config_file=config_file, mesh_file=mesh_file,
        tx_pattern_file=tx_pattern_file, rx_pattern_file=rx_pattern_file,
        device=device)

    if verbose:
        print(f'  [ctx] Rasterizer built from {os.path.basename(config_file)}')
    model = init_visible_weighted(scene_name, rast, target_n=target_n)
    rast.free_mi_scene()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=device)
    vertex_areas[active_mask] = 1.0

    range_res = compute_range_res_from_cfg(config_file)
    sample_grid = build_polar_to_cart_grid(127, 256, range_res, 400, device)

    gt_adc_np = np.load(gt_adc_path)
    if gt_adc_np.ndim == 4:
        if not (0 <= chirp_idx < gt_adc_np.shape[0]):
            raise ValueError(
                f'chirp_idx={chirp_idx} out of range for ADC of shape {gt_adc_np.shape}')
        gt_s = gt_adc_np[chirp_idx]
    else:
        gt_s = gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1).astype(np.float32)
    # (RX, TX, K, 2) -> (TX, RX, K, 2) to match render output layout
    gt_ri = gt_ri.transpose(1, 0, 2, 3)
    gt_adc_ri = torch.from_numpy(gt_ri).to(device)
    with torch.no_grad():
        ra_c = adc_to_ra_complex(gt_adc_ri)
        ra_polar = torch.abs(ra_c).float()
        gt_cart = polar_to_cart_torch(ra_polar, sample_grid)
        mn, mx = gt_cart.min(), gt_cart.max()
        gt_cart_norm = ((gt_cart - mn) / (mx - mn).clamp(min=1e-30)).detach()
    del gt_adc_ri, ra_c, ra_polar, gt_cart

    ctx = AlignmentCtx(
        rast=rast, model=model, vertex_areas=vertex_areas,
        sample_grid=sample_grid, gt_cart_norm=gt_cart_norm,
        range_res=range_res, device=device,
        scene_name=scene_name, frame_idx=frame_idx)
    return ctx


def apply_pose_to_ctx(ctx, tx_pos_mm, rx_pos_mm, boresight):
    """Update ctx.rast pose in-place. ``boresight`` is a single (3,) vector;
    it is broadcast to all TX and RX (matches pass-1 convention where every
    antenna in a frame shares the same boresight).
    """
    device = ctx.device
    with torch.no_grad():
        ctx.rast.tx_positions.copy_(
            torch.as_tensor(tx_pos_mm, dtype=torch.float32, device=device) / 1000.0)
        ctx.rast.rx_positions.copy_(
            torch.as_tensor(rx_pos_mm, dtype=torch.float32, device=device) / 1000.0)

        bore = torch.as_tensor(boresight, dtype=torch.float32, device=device)
        bore = bore / bore.norm().clamp(min=1e-8)
        n_tx = ctx.rast.tx_boresights.shape[0]
        n_rx = ctx.rast.rx_boresights.shape[0]
        ctx.rast.tx_boresights.copy_(
            bore.unsqueeze(0).expand(n_tx, 3).contiguous())
        ctx.rast.rx_boresights.copy_(
            bore.unsqueeze(0).expand(n_rx, 3).contiguous())


def render_and_evaluate_cuda(ctx, recompute_active_mask=True):
    """Render current pose, return cart_corr vs GT.

    ``recompute_active_mask=True`` re-runs FOV culling for the new pose
    (fast, <1 ms on GPU). Safe default. Set to False only when all grid
    candidates are within the init FOV (typically when search radius is
    tiny, e.g. local refinement steps).
    """
    if recompute_active_mask:
        active_mask = cull_gaussians(ctx.model, ctx.rast)
        vertex_areas = torch.zeros(ctx.model.N, device=ctx.device)
        vertex_areas[active_mask] = 1.0
    else:
        active_mask = None
        vertex_areas = ctx.vertex_areas

    with torch.no_grad():
        rp_real, rp_imag = render_gaussians(
            ctx.model, ctx.rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None)
        ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
        ra_cart = polar_to_cart_torch(ra_polar, ctx.sample_grid)
        cc = cart_corr_torch(ra_cart, ctx.gt_cart_norm).item()
    return cc




def destroy_alignment_context(ctx):
    """Release GPU tensors held by the context."""
    try:
        del ctx.rast
        del ctx.model
        del ctx.vertex_areas
        del ctx.sample_grid
        del ctx.gt_cart_norm
    except AttributeError:
        pass
    import gc
    gc.collect()
    torch.cuda.empty_cache()
