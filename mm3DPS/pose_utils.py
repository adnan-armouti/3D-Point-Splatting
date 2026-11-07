"""Pose helpers shared by the training entry points.

These three helpers (per-frame anchor loader, per-loop pose
interpolation, and applier) are imported by ``train_frame_nvs.py`` and
``rerender_test_rp.py``.
"""

from __future__ import annotations

import json

import torch


DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_pose_from_config(cfg_path: str, device: str = DEFAULT_DEVICE):
    """Read an aligned cascade JSON and return ``(tx_pos, rx_pos, tx_bore,
    rx_bore)`` tensors on ``device``. Positions in metres."""
    with open(cfg_path) as f:
        d = json.load(f)
    tx_pos = torch.tensor([e["pos_mm"]   for e in d["tx_array"]],
                          dtype=torch.float32, device=device) / 1000.0
    rx_pos = torch.tensor([e["pos_mm"]   for e in d["rx_array"]],
                          dtype=torch.float32, device=device) / 1000.0
    tx_bore = torch.tensor([e["boresight"] for e in d["tx_array"]],
                           dtype=torch.float32, device=device)
    rx_bore = torch.tensor([e["boresight"] for e in d["rx_array"]],
                           dtype=torch.float32, device=device)
    return tx_pos, rx_pos, tx_bore, rx_bore


def build_per_loop_poses(cfg_A_path: str, cfg_B_path: str,
                         n_loops: int = 16,
                         loop_dt_s: float = 7.87e-3 / 16.0,
                         frame_period_s: float = 0.1,
                         device: str = DEFAULT_DEVICE):
    """Build per-loop ``(tx_pos, rx_pos, tx_bore, rx_bore)`` by linearly
    interpolating between two aligned cascade configs.

    Time model: ``cfg_A`` corresponds to the physical middle of frame
    ``F-1``, ``cfg_B`` to the middle of frame ``F+1``. The 16 chirp loops
    live at times ``t_k = t_F + k * loop_dt``, where
    ``t_F = midpoint(t_A, t_B)``. So
    ``alpha_k = (t_k - t_A) / (t_B - t_A) = 0.5 + k * loop_dt / (2 * frame_period)``.
    """
    tx_A, rx_A, tb_A, rb_A = load_pose_from_config(cfg_A_path, device)
    tx_B, rx_B, tb_B, rb_B = load_pose_from_config(cfg_B_path, device)

    alphas = []
    poses = []
    for k in range(n_loops):
        alpha = 0.5 + k * loop_dt_s / (2.0 * frame_period_s)
        alphas.append(alpha)
        tx_p  = (1 - alpha) * tx_A + alpha * tx_B
        rx_p  = (1 - alpha) * rx_A + alpha * rx_B
        tb_k  = (1 - alpha) * tb_A + alpha * tb_B
        tb_k  = tb_k / tb_k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rb_k  = (1 - alpha) * rb_A + alpha * rb_B
        rb_k  = rb_k / rb_k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        poses.append({
            "alpha": alpha,
            "tx_positions":  tx_p.contiguous(),
            "rx_positions":  rx_p.contiguous(),
            "tx_boresights": tb_k.contiguous(),
            "rx_boresights": rb_k.contiguous(),
        })
    return poses, alphas


def apply_pose(rast, pose):
    """Copy a per-loop ``pose`` dict into a Rasterizer's per-element tensors."""
    with torch.no_grad():
        rast.tx_positions.copy_(pose["tx_positions"])
        rast.rx_positions.copy_(pose["rx_positions"])
        rast.tx_boresights.copy_(pose["tx_boresights"])
        rast.rx_boresights.copy_(pose["rx_boresights"])
