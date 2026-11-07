"""Adapter: 3DPS cascade data → RadarFields upstream format (one scene).

Writes directly into ``upstream/data/<scene>/`` and
``upstream/preprocess_results/``. Both directories are inside the
gitignored ``*/upstream/`` tree, so adapter outputs do not pollute git.

Layout produced (matches ``radarfields/dataset.py`` consumers):

    upstream/data/<scene>/
        radar/<NNNNN>.png          # uint8, (11 + W_range, H_az) — metadata top
    upstream/data/azimuth.csv      # shared antenna LUT (offset_deg, dBic)
    upstream/data/elevation.csv    # flat 0 dBic across ±40° (no measured elev)
    upstream/preprocess_results/<scene>.json
    upstream/preprocess_results/thresholded_fft/<scene>/<NNNNN>.npy
    upstream/preprocess_results/occupancy_component/<scene>/<NNNNN>.npy

Conventions (same as RadarSplat baseline):
  1. Polar magnitude via ``adc_to_ra_complex`` + abs (matches v6/v7 GT path).
  2. sin → uniform-angle resample to (200, 256) covering [-90°, +90°].
  3. Pose rotated +90° CCW about world-Z so sensor +X = left wedge edge,
     and sampler row 0 (CW from sensor +X) lands on the wedge's left edge.
  4. azim_span_deg=180 (passed via ``--intrinsics_radar`` JSON).

PNG byte layout: upstream's ``read_fft_image`` (utils/data.py:9-21) does
``Image.open → np.asarray → [11:, :] → .T``. So the file's first axis is
"rows = 11 metadata + range_bins" and second axis is "cols = azimuth_bins".
For our cascade after sin→angle resample (200, 256) sin-uniform-azim:
  on-disk PNG shape = (11 + 256, 200) = (267, 200).

Run:
    python -m baselines.radarfields.adapter.mm3dps_to_radarfields \
        --scene seq_0_frame_135
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.common import adapters as common  # noqa: E402
from baselines.common import nvs_split  # noqa: E402
from baselines.radarsplat.adapter.mm3dps_to_radarsplat import (  # noqa: E402
    H_FOV,
    AZIMUTH_RES_DEG,
    WEDGE_SPAN_DEG,
    resample_polar_sin_to_angle,
    rotate_pose_plus90_about_z,
)


# Upstream layout
UPSTREAM_ROOT = os.path.abspath(
    os.path.join(_REPO, "baselines", "radarfields", "upstream")
)
UPSTREAM_DATA = os.path.join(UPSTREAM_ROOT, "data")
UPSTREAM_PREPROC = os.path.join(UPSTREAM_ROOT, "preprocess_results")

W_METADATA = 11        # upstream stripped row count
W_RANGE = 256          # cascade ADC samples
H_AZ = H_FOV           # 200 (after sin→angle resample)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_to_float01(x: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalize to float in [0, 1]."""
    x = x.astype(np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _write_radarfields_png(ra_polar_angle: np.ndarray, path: str) -> None:
    """Save uniform-angle polar (H_az, W_range) as upstream-format uint8 PNG.

    Steps mirror the inverse of ``utils.data.read_fft_image``:
        1. Transpose to (W_range, H_az).
        2. Prepend 11 zero metadata rows → (11 + W_range, H_az).
        3. Save as 8-bit PNG.
    """
    assert ra_polar_angle.shape == (H_AZ, W_RANGE), ra_polar_angle.shape
    img = _normalize_to_float01(ra_polar_angle)              # (H_az, W_range)
    img = (img * 255.0).clip(0.0, 255.0).round().astype(np.uint8)
    img_t = img.T                                            # (W_range, H_az)
    strip = np.zeros((W_METADATA, H_AZ), dtype=np.uint8)
    full = np.concatenate([strip, img_t], axis=0)            # (11 + W_range, H_az)
    Image.fromarray(full, mode="L").save(path)


def _build_antenna_luts(out_dir: str) -> None:
    """Write azimuth.csv (combined tx*rx) and a flat elevation.csv.

    Sources: ``assets/antenna_pattern/MMWCAS/{tx1_76,rx1_76}.npy``.
    Each is (361, 2) — col 0 angle deg, col 1 dBic. Combined dBic = tx + rx.
    """
    tx = np.load(os.path.join(_REPO, "assets/antenna_pattern/MMWCAS/tx1_76.npy"))
    rx = np.load(os.path.join(_REPO, "assets/antenna_pattern/MMWCAS/rx1_76.npy"))
    assert tx.shape == (361, 2) and rx.shape == (361, 2)
    # Both are sampled at the same 1° grid; combined gain in dBic = tx + rx.
    angles = tx[:, 0]
    combined_dBic = tx[:, 1] + rx[:, 1]
    azim_lut = np.stack([angles, combined_dBic], axis=1)
    # Trim to ±90° (our wedge); upstream's LUT supports any range it wants.
    mask = (azim_lut[:, 0] >= -90.0) & (azim_lut[:, 0] <= 90.0)
    azim_lut = azim_lut[mask]
    np.savetxt(
        os.path.join(out_dir, "azimuth.csv"),
        azim_lut, fmt="%.6f", delimiter=",",
    )
    # Flat elevation LUT: 0 dBic across ±40° at 1°/bin.
    elev = np.stack(
        [np.arange(-40, 41, dtype=np.float64),
         np.zeros(81, dtype=np.float64)],
        axis=1,
    )
    np.savetxt(
        os.path.join(out_dir, "elevation.csv"),
        elev, fmt="%.6f", delimiter=",",
    )


def _compute_offsets_scalers(
    train_poses: np.ndarray, range_max_m: float,
) -> Tuple[List[float], List[float]]:
    """Compute XYZ normalization offsets/scalers so that the train scene
    plus the radar's max range fits inside the unit cube."""
    centroid = train_poses[:, :3, 3].mean(axis=0)            # (3,)
    extents = np.abs(train_poses[:, :3, 3] - centroid).max(axis=0)  # per-axis
    # Add the radar's max sensing range as margin; XY plane only (radar is BEV).
    extents[:2] += range_max_m
    extents[2] += 2.0                                        # 2 m vertical margin
    # Use a uniform scaler equal to max axis extent (so unit-cube bounds).
    scaler = float(extents.max())
    return list(map(float, -centroid)), [scaler, scaler, scaler]


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_scene(scene: str) -> dict:
    os.makedirs(UPSTREAM_DATA, exist_ok=True)
    os.makedirs(UPSTREAM_PREPROC, exist_ok=True)
    scene_data_dir = os.path.join(UPSTREAM_DATA, scene)
    radar_dir = os.path.join(scene_data_dir, "radar")
    thresh_dir = os.path.join(UPSTREAM_PREPROC, "thresholded_fft", scene)
    occ_dir = os.path.join(UPSTREAM_PREPROC, "occupancy_component", scene)
    for d in (scene_data_dir, radar_dir, thresh_dir, occ_dir):
        os.makedirs(d, exist_ok=True)

    # Antenna LUTs (shared across scenes; idempotent)
    _build_antenna_luts(UPSTREAM_DATA)

    split = nvs_split.cascaded_split(scene)
    frames = sorted(set(split["train_frames"] + [split["test_frame"]]))

    # Range resolution from center frame
    from mm3DPS.data.io_utils import compute_range_res_from_cfg
    range_res = float(compute_range_res_from_cfg(split["test_config"]))

    # Cache config & ADC paths by frame number
    frame_to_cfg = {f: c for f, c in zip(split["train_frames"], split["train_configs"])}
    frame_to_adc = {f: p for f, p in zip(split["train_frames"], split["train_files"])}
    frame_to_cfg[split["test_frame"]] = split["test_config"]
    frame_to_adc[split["test_frame"]] = split["test_file"]

    timestamps_radar: List[str] = []
    poses: List[np.ndarray] = []
    train_idx_local: List[int] = []
    test_idx_local: List[int] = []

    # Inline upstream helpers (radarfields/radar.py:87-107, 152-166) so we
    # don't need to import the radarfields env / upstream sys.path here.
    def compute_spherical_grid_noise_threshold(fft, min_range, max_range):
        fft = fft.clone()
        med_r, _ = torch.median(fft[:, min_range:max_range + 1], axis=1)
        med_a, _ = torch.median(fft, axis=0)
        grid = torch.maximum(med_a.unsqueeze(0), med_r.unsqueeze(1))
        return fft * (fft > 1.5 * grid)

    def compute_occupancy_component(fft, occupancy_threshold,
                                    highlights_threshold, min_range_bin,
                                    binarize=True):
        fft = fft.clone()
        fft[:, :min_range_bin] = 0.0
        occ = fft * (fft > occupancy_threshold)
        occ = torch.clip(occ, max=highlights_threshold)
        if binarize:
            occ[occ > 0.01] = 1.0
        return occ

    for k, frame in enumerate(frames):
        # 1. Load ADC, polar, resample to uniform-angle, normalize, write PNG.
        adc = common.load_cascaded_adc(frame_to_adc[frame])
        ra_sin = common.adc_to_polar_ra(adc)     # (127, 256)
        ra_ang = resample_polar_sin_to_angle(ra_sin, H_fov=H_AZ)    # (200, 256)
        png_name = f"{frame:05d}.png"
        _write_radarfields_png(ra_ang, os.path.join(radar_dir, png_name))
        timestamps_radar.append(png_name)

        # 2. Pose: torch-path pose + +90° CCW rotation about world-Z, written as
        # a 4x4 matrix list inline in the JSON.
        cfg = common.load_config(frame_to_cfg[frame])
        T, _, _ = common.pose_from_config(cfg)
        T_adj = rotate_pose_plus90_about_z(T)
        poses.append(np.asarray(T_adj, dtype=np.float32))

        if frame in set(split["train_frames"]):
            train_idx_local.append(k)
        else:
            test_idx_local.append(k)

        # 3. Pre-computed thresholded FFT and occupancy component (per frame).
        ra_norm = _normalize_to_float01(ra_ang)                     # float [0, 1]
        ra_t = torch.from_numpy(ra_norm)
        # min_range_bin is 1-indexed inclusive; we use bin 16 (= our 15-bin
        # near-range coupling drop, +1 for 1-indexing).
        thresholded = compute_spherical_grid_noise_threshold(
            ra_t, min_range=15, max_range=W_RANGE - 1
        ).cpu().numpy().astype(np.float32)
        np.save(os.path.join(thresh_dir, f"{frame:05d}.npy"), thresholded)

        occ = compute_occupancy_component(
            ra_t,
            occupancy_threshold=0.20,
            highlights_threshold=0.95,
            min_range_bin=15,
            binarize=True,
        ).cpu().numpy().astype(np.float32)
        np.save(os.path.join(occ_dir, f"{frame:05d}.npy"), occ)

    # Pose offsets & scalers (unit-cube normalization for the model's HashGrid).
    train_poses_arr = np.stack(
        [poses[i] for i in train_idx_local], axis=0
    )
    range_max_m = W_RANGE * range_res                     # ~15.18 m
    offsets, scalers = _compute_offsets_scalers(train_poses_arr, range_max_m)

    preprocess = {
        "timestamps_radar": timestamps_radar,
        "radar2worlds": [p.tolist() for p in poses],
        "offsets": offsets,
        "scalers": scalers,
        "train_indices": train_idx_local,
        "test_indices": test_idx_local,
    }
    preprocess_path = os.path.join(UPSTREAM_PREPROC, f"{scene}.json")
    with open(preprocess_path, "w") as f:
        json.dump(preprocess, f, indent=2)

    manifest = {
        "scene": scene,
        "frames": frames,
        "train_frames": split["train_frames"],
        "test_frame": split["test_frame"],
        "H_az": H_AZ,
        "W_range": W_RANGE,
        "range_resolution": range_res,
        "azim_span_deg": WEDGE_SPAN_DEG,
        "azimuth_resolution_deg": AZIMUTH_RES_DEG,
        "pose_rotation_about_world_z_deg": 90.0,
        "preprocess_path": os.path.relpath(preprocess_path, UPSTREAM_ROOT),
        "deviations_from_reference": [
            "Synthetic 9-frame sequence rather than upstream's longer driving sequences.",
            "sin-space → uniform-angle azimuth resample (matches RadarSplat baseline).",
            "Pose rotation +90° CCW about world-Z to align cascade wedge with sampler row 0.",
            "azim_span_deg=180 via 1-line patch to radarfields/sampler.py "
            "(patches/azim_span.patch); upstream default 360 unchanged.",
            "bin_size_radar=compute_range_res_from_cfg (~0.059 m), not upstream 0.044.",
            "num_azimuths_radar=200, num_range_bins=256 (vs upstream 400 / 7536).",
            "Combined tx*rx MMWCAS antenna LUT for azimuth; flat 0 dBic elevation LUT "
            "(no measured elevation pattern for cascade).",
            "bs=8 (vs upstream default 10).",
            "--refine_poses disabled (small-n training set; remove a confound).",
        ],
    }
    with open(os.path.join(scene_data_dir, "adapter_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    m = build_scene(args.scene)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
