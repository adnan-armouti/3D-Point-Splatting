"""Refine the seed ego velocity of each frame against the measured radar cube.

Per frame, gradient-free and offline, using the untrained model, so that the
estimate comes from geometry and measured ADC alone and does not depend on
learned materials. The objective is the Pearson correlation of the
range-azimuth-Doppler (RAD) magnitude between render and measurement.

  1. Coarse: 3-D grid search in a box around the seed from ``v_ego.py``.
  2. Fine: bounded Nelder-Mead from the best grid point, to about 0.01 m/s.
  3. Sanity: flags adjacent frames more than 0.5 m/s apart.

Writes ``data/v_ego_cache/<scene>/frame_<F>_v_ego_refined.npy`` and a per-scene
``v_ego_refined_manifest.json``. Needs the mm3DPS environment and a GPU.

Usage::

    python -m baselines.dart.adapter.v_ego_refine --scene seq_0_frame_135
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import minimize

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.dart.adapter.v_ego import cache_path, load_seed, scene_frames  # noqa: E402
from mm3DPS import cuda  # noqa: E402
from mm3DPS.config import DEFAULT_DATA_ROOT, load_trained_config  # noqa: E402
from mm3DPS.frame_dataset import _build_frame_poses  # noqa: E402
from mm3DPS.pose_utils import apply_pose  # noqa: E402
from mm3DPS.rasterizer import Rasterizer, reparameterize_torch  # noqa: E402
from mm3DPS.rasterizer_factorized import render_factorized  # noqa: E402
from mm3DPS.train_gaussian import cull_gaussians, init_visible_weighted  # noqa: E402

# Chirp timing of the cascade: burst / chirps per frame, then per TX within a chirp.
T_BURST = 7.87e-3
N_CHIRPS = 16
T_C = T_BURST / N_CHIRPS
T_A = T_C / 12
WAVELENGTH = 3.0e8 / 77e9
# TDM firing order k(i) of each ADC TX channel.
TI_FIRING_INDEX = (11, 10, 9, 2, 1, 0, 8, 7, 6, 5, 4, 3)
N_DOP = 32  # Doppler FFT size, 2x zero-padding over the 16 chirps.
# Point selection of the search behind the shipped values (training uses 0.1761).
COS_BORE_MIN = 0.1

_RX_LOCS = [(0, 0), (1, 0), (2, 0), (3, 0), (11, 0), (12, 0), (13, 0),
            (14, 0), (46, 0), (47, 0), (48, 0), (49, 0), (50, 0),
            (51, 0), (52, 0), (53, 0)]
_TX_LOCS = [(0, 0), (4, 0), (8, 0), (9, 1), (10, 4), (11, 6),
            (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0)]


# ---------------------------------------------------------------------------
# Range-azimuth-Doppler cube
# ---------------------------------------------------------------------------

def _hann(n: int, device, dtype):
    return torch.hann_window(n, device=device, dtype=dtype)


def _vx_el0_weights(device, dtype) -> torch.Tensor:
    """(86, n_rx*n_tx) weights of the elevation-0 row of the virtual array. The
    first antenna at a bin has weight 1; each later one halves the existing
    weights and adds 0.5. The flat index is ``rx_id*n_tx + tx_id``."""
    n_rx, n_tx, n_az = 16, 12, 86
    contribs = [[] for _ in range(n_az)]
    for rx_id, rx_loc in enumerate(_RX_LOCS):
        for tx_id, tx_loc in enumerate(_TX_LOCS):
            if tx_loc[1] != 0:
                continue
            az = rx_loc[0] + tx_loc[0]
            if not contribs[az]:
                contribs[az].append((rx_id, tx_id, 1.0))
            else:
                contribs[az] = [(r, t, c * 0.5) for (r, t, c) in contribs[az]]
                contribs[az].append((rx_id, tx_id, 0.5))
    W = torch.zeros((n_az, n_rx * n_tx), dtype=dtype, device=device)
    for az in range(n_az):
        for rx_id, tx_id, coef in contribs[az]:
            W[az, rx_id * n_tx + tx_id] = coef
    return W


def _fft_axis(x: torch.Tensor, dim: int, n_fft: int) -> torch.Tensor:
    """Hann, ifftshift, FFT with zero-padding, drop bin 0, fftshift."""
    n = x.shape[dim]
    win = _hann(n, x.device, x.real.dtype).to(x.dtype)
    shape = [1] * x.ndim
    shape[dim] = n
    x = torch.fft.ifftshift(x * win.view(*shape), dim=dim)
    x = torch.fft.fft(x, n=n_fft, dim=dim).narrow(dim, 1, n_fft - 1)
    return torch.fft.fftshift(x, dim=dim)


def rp_stack_to_rad(rp_stack: torch.Tensor) -> torch.Tensor:
    """(CH, TX=12, RX=16, R) complex range profiles -> (N_DOP-1, 127, R) RAD cube."""
    n_ch, n_tx, n_rx, n_r = rp_stack.shape
    rp_flat = rp_stack.permute(0, 2, 1, 3).reshape(n_ch, n_rx * n_tx, n_r)
    vx86 = torch.matmul(_vx_el0_weights(rp_flat.device, rp_flat.dtype), rp_flat)  # (CH, 86, R)
    ra_stack = _fft_axis(vx86, dim=-2, n_fft=128)                                 # (CH, 127, R)
    return _fft_axis(ra_stack, dim=0, n_fft=N_DOP)


def measured_rad_mag(scene, frame, data_root, device="cuda"):
    """|RAD| of a measured frame: range Hann and range FFT, then the shared chain."""
    arr = np.load(os.path.join(data_root, scene, "radar", f"cascaded_frame_{frame}.npy"))
    adc = torch.from_numpy(arr.transpose(0, 2, 1, 3).astype(np.complex64)).to(device)
    win_r = _hann(adc.shape[-1], adc.device, adc.real.dtype).to(adc.dtype)
    rp = torch.fft.fft(adc * win_r[None, None, None, :], n=adc.shape[-1], dim=-1)
    return rp_stack_to_rad(rp).abs().float()


# ---------------------------------------------------------------------------
# Doppler render
# ---------------------------------------------------------------------------

class _PathCapture:
    """Stands in for ``cuda.step5_fused`` during one render_factorized call and
    keeps its inputs, so the expensive BSDF and geometry path runs only once per
    render and the 3DPS renderer needs no Doppler option of its own."""

    def __init__(self):
        self.args = None

    def __call__(self, *args):
        self.args = args
        return None, None

    def __enter__(self):
        self._original = cuda.step5_fused
        cuda.step5_fused = self
        return self

    def __exit__(self, *exc):
        cuda.step5_fused = self._original


def render_rad_mag(model, rast, v_ego, active_mask, vertex_areas):
    """|RAD| of a render at ``v_ego``. Each chirp m and TX i adds the phase
    -(4 pi / lambda) <u(p), v_ego> (m T_c + k(i) T_a) to every path."""
    positions = model.positions[active_mask]
    with torch.no_grad(), _PathCapture() as capture:
        render_factorized(
            positions, model.get_normals()[active_mask], vertex_areas[active_mask],
            model.raw_materials[active_mask], rast, reparameterize_torch)
    w_full, phi_base, n_peak, psf_real, psf_imag, K, eps = capture.args

    with torch.no_grad():
        device, dtype = positions.device, positions.dtype
        radar_center = 0.5 * (rast.tx_positions.mean(0) + rast.rx_positions.mean(0))
        diff = positions - radar_center.to(device=device, dtype=dtype).view(1, 3)
        u = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        A = -(4.0 * math.pi / WAVELENGTH) * (u * v_ego.view(1, 3)).sum(dim=-1)  # (M,)
        k = torch.as_tensor(TI_FIRING_INDEX, device=device, dtype=dtype)
        m = torch.arange(N_CHIRPS, device=device, dtype=dtype)
        t_off = m.unsqueeze(-1) * T_C + k.unsqueeze(0) * T_A                    # (CH, n_tx)

        chirps = []
        for chirp in range(N_CHIRPS):
            phi = phi_base + (A.unsqueeze(-1) * t_off[chirp].unsqueeze(0)).unsqueeze(-1)
            rp_r, rp_i = cuda.step5_fused(
                w_full, phi.contiguous(), n_peak, psf_real, psf_imag, K, eps)
            chirps.append(torch.complex(rp_r, rp_i))
        return rp_stack_to_rad(torch.stack(chirps, dim=0)).abs().float()


def _flat_pearson(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    am = a - a.mean()
    bm = b - b.mean()
    den = torch.sqrt((am * am).sum() * (bm * bm).sum()).clamp_min(1e-30)
    return float(((am * bm).sum() / den).item())


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def build_context(scene, frame, data_root, target_n=20000, device="cuda"):
    """Untrained model and rasterizer at this frame's pose, plus the seed velocity."""
    config = load_trained_config(scene, data_root=data_root)
    seed_cfg = os.path.join(data_root, scene, "configs", f"cascaded_frame_{frame}.json")
    assert os.path.exists(seed_cfg), f"no cascade config for {scene} F={frame}"
    rast = Rasterizer(
        config_file=seed_cfg, mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=device)
    poses, _ = _build_frame_poses(scene, frame, use_pass2=True, data_root=data_root, device=device)
    apply_pose(rast, poses[0])

    model = init_visible_weighted(scene, rast, target_n=target_n, cos_bore_min=COS_BORE_MIN)
    rast.free_mi_scene()
    gc.collect()
    torch.cuda.empty_cache()

    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=device)
    vertex_areas[active_mask] = 1.0
    v_ego0 = torch.as_tensor(load_seed(scene, frame, data_root), dtype=torch.float32, device=device)
    return model, rast, active_mask, vertex_areas, v_ego0


def coarse_grid(ctx, gt_mag, axis_steps: Tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)) -> dict:
    model, rast, mask, areas, v_ego0 = ctx
    best = {"cc": -1.0, "v_ego": v_ego0.clone()}
    seed_cc = None
    for a in axis_steps:
        for b in axis_steps:
            for c in axis_steps:
                v = v_ego0 + torch.tensor([a, b, c], device=v_ego0.device, dtype=torch.float32)
                cc = _flat_pearson(render_rad_mag(model, rast, v, mask, areas), gt_mag)
                if (a, b, c) == (0.0, 0.0, 0.0):
                    seed_cc = cc
                if cc > best["cc"]:
                    best = {"cc": cc, "v_ego": v.clone()}
    return {"seed_v_ego_mps": v_ego0.cpu().numpy().tolist(),
            "seed_cc": seed_cc,
            "stage1_best_v_ego_mps": best["v_ego"].cpu().numpy().tolist(),
            "stage1_best_cc": best["cc"]}


def fine_nelder_mead(ctx, gt_mag, coarse: dict, bound_mps: float = 0.3) -> dict:
    """Nelder-Mead from the best grid point, within +-bound_mps of the seed."""
    model, rast, mask, areas, v_ego0 = ctx
    seed = np.asarray(coarse["seed_v_ego_mps"], dtype=np.float32)

    def evaluate(x):
        v = torch.as_tensor(np.clip(x, seed - bound_mps, seed + bound_mps),
                            dtype=torch.float32, device=v_ego0.device)
        return _flat_pearson(render_rad_mag(model, rast, v, mask, areas), gt_mag)

    opt = minimize(
        lambda x: -evaluate(x),
        np.asarray(coarse["stage1_best_v_ego_mps"], dtype=np.float32),
        method="Nelder-Mead",
        options={"xatol": 1e-2, "fatol": 5e-4, "maxiter": 80, "adaptive": True})
    return {"final_v_ego_mps": np.clip(opt.x, seed - bound_mps, seed + bound_mps).tolist(),
            "final_cc": evaluate(opt.x)}


def sanity_flags(per_frame: dict, max_adjacent_delta_mps: float = 0.5) -> dict:
    """Largest velocity change to an adjacent frame, flagged if over the limit."""
    frames = sorted(per_frame)
    flags = {}
    for i, f in enumerate(frames):
        v = np.asarray(per_frame[f]["final_v_ego_mps"])
        deltas = [np.linalg.norm(v - np.asarray(per_frame[g]["final_v_ego_mps"]))
                  for g in frames[max(0, i - 1):i + 2] if g != f]
        max_delta = float(max(deltas, default=0.0))
        flags[f] = {"max_adjacent_delta_mps": max_delta,
                    "suspect": max_delta > max_adjacent_delta_mps}
    return flags


def refine_scene(scene, frames, data_root=DEFAULT_DATA_ROOT, target_n=20000, device="cuda") -> dict:
    per_frame = {}
    for frame in frames:
        ctx = build_context(scene, frame, data_root, target_n, device)
        gt_mag = measured_rad_mag(scene, frame, data_root, device)
        coarse = coarse_grid(ctx, gt_mag)
        fine = fine_nelder_mead(ctx, gt_mag, coarse)
        print(f"{scene} F={frame}: seed cc={coarse['seed_cc']:.4f}  "
              f"grid cc={coarse['stage1_best_cc']:.4f}  final cc={fine['final_cc']:.4f}  "
              f"v_ego={fine['final_v_ego_mps']}", flush=True)
        path = cache_path(scene, frame, "v_ego_refined", data_root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.asarray(fine["final_v_ego_mps"], dtype=np.float32))
        per_frame[int(frame)] = {**coarse, **fine}
        del ctx
        gc.collect()
        torch.cuda.empty_cache()

    for frame, flag in sanity_flags(per_frame).items():
        per_frame[frame]["sanity"] = flag
    manifest = {"scene": scene, "frames": [int(f) for f in frames],
                "target_n": target_n, "per_frame": per_frame}
    with open(os.path.join(data_root, "v_ego_cache", scene, "v_ego_refined_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", default=None,
                    help="comma-separated frames; default: the scene's nine frames")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--target_n", type=int, default=20000)
    args = ap.parse_args()
    frames = ([int(x) for x in args.frames.split(",") if x.strip()]
              if args.frames else scene_frames(args.scene))
    refine_scene(args.scene, frames, args.data_root, args.target_n)


if __name__ == "__main__":
    main()
