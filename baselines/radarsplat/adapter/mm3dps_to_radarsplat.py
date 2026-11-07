"""Adapter: 3DPS cascade data → RadarSplat upstream format (one scene).

Writes, under ``baselines/radarsplat/data_radarsplat/<scene>/``:

    sensor.yaml
    radar_trajectory.tum                       # 9 lines, world-from-sensor
    images/<NNNN>.png                          # (H_fov, 11 + W_range) uint8
    synced_lidar/<NNNN>.pcd                    # from scene/pcl.npy XYZ
    synced_lidar_map_win5/<NNNN>.pcd           # duplicated (no 5-win fusion)
    radar_average_map_polar/<subdir>/<NNNN>.png # degenerate zeros
    multipath_model/dist:50/<NNNN>.npy         # degenerate empty dict

Key design decisions (all documented as deviations):

1. **Sin-space → uniform-angle resample** along azimuth. ``mm3DPS``'s polar RA
   image (``adc_to_ra_image_numpy`` output) is uniform in ``sin(angle)`` over
   ``[-90°, +90°]`` (FFT of the virtual array). RadarSplat's 3D
   rasterizer expects uniform angle. We linearly resample 127 sin-space bins
   to ``H_fov`` uniform-angle bins covering ``[-90°, +90°]``, zero-padding the
   ~10° at each extreme that lie beyond the FFT's ±79.4° max.

2. **Pose rotation by +90° CCW about world-Z.** RadarSplat's polar layout
   has row 0 = sensor +X (forward) and rows increase clockwise. The cascade's
   wedge is symmetric around boresight (±90°). If sensor +X = boresight, the
   wedge occupies the first AND last 100 rows (wraps). We apply a +90° CCW
   rotation to each stored pose so the new sensor +X points to the LEFT edge
   of the wedge; then rows 0..H_fov-1 (after upstream's sonar-style crop)
   cover the full 180° wedge contiguously, going left-edge → forward → right-edge.
   The rotation is applied uniformly to all 9 poses, so the scene's geometry
   is preserved up to a rigid rotation — Gaussian positions are learned in
   the rotated frame.

3. **sensor_type=scanning_radar, azimuth_coverage=180.** Triggers the
   partial-coverage branch added by ``patches/wedge_crop_guard.patch``.

4. **Degenerate radar_average_map and multipath.** Shipped as all-zeros PNG
   and empty-dict npy so the dataloader's existence checks pass. Both are
   multiplied by zero in the loss via ``--l1occloss_lambda 0``
   and ``--multipath_weight 0``.

5. **scene/pcl.npy** provides the synced LiDAR. We use the same cloud for
   both ``synced_lidar`` and ``synced_lidar_map_win5`` (we don't have the
   5-frame window fusion). Neither contributes to the training loss — they
   flow through to visualization and eval_geometry only.

Run:

    python -m baselines.radarsplat.adapter.mm3dps_to_radarsplat \
        --scene seq_0_frame_135 \
        --out-root baselines/radarsplat/data_radarsplat
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as SciRot

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.common import adapters as common  # noqa: E402
from baselines.common import nvs_split  # noqa: E402


# ---------------------------------------------------------------------------
# Constants (match upstream defaults + our sensor geometry)
# ---------------------------------------------------------------------------

W_METADATA = 11                         # upstream's leading metadata-strip width
AZIMUTH_RES_DEG = 0.9                   # RadarSplat upstream default
AZIMUTH_BEAMWIDTH_DEG = 1.8             # upstream default
AZIMUTH_COVERAGE_DEG = 180              # our wedge, triggers patch branch
WEDGE_SPAN_DEG = 180.0                  # physical span of the cascade wedge
WEDGE_HALFSPAN_DEG = WEDGE_SPAN_DEG / 2  # ±90° around boresight

H_FOV = int(round(WEDGE_SPAN_DEG / AZIMUTH_RES_DEG))  # 200 rows
RADAR_MAP_SUBDIR = (
    "radar_average_map_polar/"
    "res:0.0586_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
)
MULTIPATH_SUBDIR = "multipath_model/dist:50"
LIDAR_SUBDIR = "synced_lidar"
LIDAR_MAP_SUBDIR = "synced_lidar_map_win5"
IMAGES_SUBDIR = "images"


# ---------------------------------------------------------------------------
# Sin-space → uniform-angle resample
# ---------------------------------------------------------------------------

def resample_polar_angle_to_sin(ra_polar_angle: np.ndarray,
                                H_sin: int = 127) -> np.ndarray:
    """Inverse of ``resample_polar_sin_to_angle``: ``(H_fov, W)`` uniform-angle
    polar over ``[-90°, +90°]`` at 0.9°/bin → ``(H_sin, W)`` sin-space polar
    matching the 3DPS convention.

    For each output row j ∈ [0, H_sin-1]: target physical angle
    ``θ_out[j] = arcsin((j - (H_sin-1)/2) / ((H_sin+1)/2))``. Find the
    fractional bin position in the angle-uniform input (over ``[-90°, +90°]``
    with the same step ``WEDGE_SPAN_DEG / H_in``) and linearly interpolate.
    Used at metric time so both GT (native sin-space) and rendered (model
    output, angle-uniform) end up on the SAME 127-bin sin-space grid before
    ``ra_polar_to_cartesian``.
    """
    H_in, W = ra_polar_angle.shape
    half = (H_sin + 1) / 2.0
    sin_targets = (np.arange(H_sin) - (H_sin - 1) / 2.0) / half
    sin_targets = np.clip(sin_targets, -1.0, 1.0)
    target_deg = np.degrees(np.arcsin(sin_targets))

    step_deg = WEDGE_SPAN_DEG / H_in
    src_row_frac = (target_deg + WEDGE_HALFSPAN_DEG) / step_deg - 0.5
    src_row_low = np.floor(src_row_frac).astype(int)
    src_row_high = src_row_low + 1
    alpha = (src_row_frac - src_row_low).astype(np.float32)

    valid = (src_row_low >= 0) & (src_row_high < H_in)
    out = np.zeros((H_sin, W), dtype=np.float32)
    if valid.any():
        idx = np.where(valid)[0]
        lo = ra_polar_angle[src_row_low[valid]]
        hi = ra_polar_angle[src_row_high[valid]]
        a = alpha[valid].reshape(-1, 1)
        out[idx] = (1.0 - a) * lo + a * hi
    return out


def resample_polar_sin_to_angle(ra_polar_sin: np.ndarray,
                                H_fov: int = H_FOV) -> np.ndarray:
    """Resample ``(127, W)`` sin-space polar to ``(H_fov, W)`` uniform-angle polar.

    3DPS convention (from ``mm3DPS.data.ra_utils.virtual_array_to_ra_polar_numpy``):
        row i ∈ [0, 126] corresponds to ``sin(angle) = (i - 63) / 64``,
        i.e. angle ≈ arcsin((i-63)/64), ranging ~[-79.4°, +79.4°].

    Target: uniform angle in ``[-90°, +90°]`` at 0.9°/bin. Rows outside
    ±79.4° get zero-padded.
    """
    H_in, W = ra_polar_sin.shape
    assert H_in == 127, f"expected 127 sin-space azimuth bins, got {H_in}"

    target_deg = -90.0 + (np.arange(H_fov) + 0.5) * (WEDGE_SPAN_DEG / H_fov)
    target_sin = np.sin(np.deg2rad(target_deg))           # in [-1, +1)
    # Fractional row index in the sin-space polar.
    src_row_frac = target_sin * 64.0 + 63.0               # equivalent inverse of (i-63)/64
    src_row_low = np.floor(src_row_frac).astype(int)
    src_row_high = src_row_low + 1
    alpha = (src_row_frac - src_row_low).astype(np.float32)

    valid = (src_row_low >= 0) & (src_row_high < H_in)
    out = np.zeros((H_fov, W), dtype=np.float32)

    if valid.any():
        idx = np.where(valid)[0]
        lo = ra_polar_sin[src_row_low[valid]]              # (K, W)
        hi = ra_polar_sin[src_row_high[valid]]             # (K, W)
        a = alpha[valid].reshape(-1, 1)
        out[idx] = (1.0 - a) * lo + a * hi

    return out


# ---------------------------------------------------------------------------
# Pose transform: rotate sensor by +90° CCW about world-Z
# ---------------------------------------------------------------------------

def rotate_pose_plus90_about_z(T_world_from_sensor: np.ndarray) -> np.ndarray:
    """Return ``T' = T @ R_z(+90°)`` — applies a +90° CCW sensor-frame rotation
    about the world-Z axis (equivalently, rotates the sensor's local axes).

    If original sensor +X is boresight (roughly world +X), new sensor +X is
    boresight rotated +90° CCW about world-Z = "leftward of original boresight".
    Rendered row 0 = new sensor +X = left edge of the cascade wedge.
    """
    Rz = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0,  0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return T_world_from_sensor @ Rz


# ---------------------------------------------------------------------------
# Per-frame writers
# ---------------------------------------------------------------------------

def _normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalize to ``uint8`` in [0, 255]."""
    x = x.astype(np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    y = np.clip(y * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return y


def _write_png_with_metadata(arr: np.ndarray, path: str) -> None:
    """Write ``(H, W)`` uint8 with an 11-col zero metadata strip on the left."""
    assert arr.dtype == np.uint8 and arr.ndim == 2, arr.shape
    strip = np.zeros((arr.shape[0], W_METADATA), dtype=np.uint8)
    full = np.concatenate([strip, arr], axis=1)
    Image.fromarray(full, mode="L").save(path)


def _write_pcd(xyz: np.ndarray, path: str) -> None:
    """Write an ASCII PCD containing XYZ points. Open3D-readable."""
    xyz = np.asarray(xyz, dtype=np.float32)
    assert xyz.ndim == 2 and xyz.shape[1] == 3, xyz.shape
    with open(path, "w") as f:
        f.write(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {xyz.shape[0]}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {xyz.shape[0]}\n"
            "DATA ascii\n"
        )
        for p in xyz:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def _write_multipath_npy(path: str) -> None:
    """Write a degenerate multipath-source file. Schema matches upstream's
    ``dataloader.py:391`` consumer (``np.load(..).item()``: a dict)."""
    d = {
        "azi_id_list": np.zeros(0, dtype=np.int64),
        "range_id_list": np.zeros(0, dtype=np.int64),
        "reconstructed_signal": np.zeros((0, 1), dtype=np.float32),
    }
    np.save(path, d, allow_pickle=True)


# ---------------------------------------------------------------------------
# sensor.yaml writer
# ---------------------------------------------------------------------------

def _write_sensor_yaml(out_dir: str, range_resolution: float, W_range: int) -> None:
    """Emit a sensor.yaml compatible with upstream's dataloader."""
    yaml_text = (
        "sensor_type: \"scanning_radar\"\n"
        "use_polar: True\n"
        f"H: {H_FOV}\n"
        f"W: {W_range + W_METADATA}\n"
        f"W_metadata: {W_METADATA}\n"
        f"range_resolution: {range_resolution:.6f}\n"
        f"azimuth_resolution: {AZIMUTH_RES_DEG}\n"
        f"azimuth_beamwidth: {AZIMUTH_BEAMWIDTH_DEG}\n"
        f"azimuth_coverage: {AZIMUTH_COVERAGE_DEG}\n"
    )
    with open(os.path.join(out_dir, "sensor.yaml"), "w") as f:
        f.write(yaml_text)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_scene(scene: str, out_root: str) -> dict:
    out_dir = os.path.join(out_root, scene)
    for sub in (
        IMAGES_SUBDIR,
        LIDAR_SUBDIR,
        LIDAR_MAP_SUBDIR,
        RADAR_MAP_SUBDIR,
        MULTIPATH_SUBDIR,
    ):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    split = nvs_split.cascaded_split(scene)
    frames = split["train_frames"] + [split["test_frame"]]
    # sort so TUM order == on-disk PNG alphabetical order.
    # Pad filenames to 5 digits so lexical ordering matches numeric ordering
    # across all scenes (frame numbers range 101..882 across the benchmark).
    frames = sorted(set(frames))

    # Range resolution — take center frame's value; cascaded configs within a
    # scene use a fixed FMCW setup so this is constant.
    center_cfg_path = split["test_config"]
    from mm3DPS.data.io_utils import compute_range_res_from_cfg
    range_res = compute_range_res_from_cfg(center_cfg_path)

    # LiDAR point cloud (shared across frames — scene/pcl.npy).
    pcl_path = os.path.join(nvs_split.scene_dir(scene), "scene", "pcl.npy")
    pcl = np.load(pcl_path)
    xyz = np.ascontiguousarray(pcl[:, :3], dtype=np.float32)

    # Collect TUM lines while writing per-frame assets.
    tum_lines = []
    frame_to_config = {}
    for f, cfg_path in zip(split["train_frames"], split["train_configs"]):
        frame_to_config[f] = cfg_path
    frame_to_config[split["test_frame"]] = split["test_config"]

    frame_to_adcpath = {}
    for f, adc in zip(split["train_frames"], split["train_files"]):
        frame_to_adcpath[f] = adc
    frame_to_adcpath[split["test_frame"]] = split["test_file"]

    for frame in frames:
        cfg = common.load_config(frame_to_config[frame])
        adc = common.load_cascaded_adc(frame_to_adcpath[frame])

        # Polar RA → uniform-angle resample → uint8 → PNG
        ra_sin = common.adc_to_polar_ra(adc, sensor="cascaded")       # (127, 256)
        ra_ang = resample_polar_sin_to_angle(ra_sin, H_fov=H_FOV)     # (H_FOV, 256)
        img = _normalize_to_uint8(ra_ang)
        fname = f"{frame:05d}.png"
        _write_png_with_metadata(img, os.path.join(out_dir, IMAGES_SUBDIR, fname))

        # Degenerate radar_average_map (all zeros, post-strip shape). Upstream
        # dataloader does NOT strip metadata from this file, so write it
        # directly at the post-strip shape (H_fov, W_range) without the
        # 11-col leading strip.
        zero = np.zeros_like(img)
        Image.fromarray(zero, mode="L").save(
            os.path.join(out_dir, RADAR_MAP_SUBDIR, fname)
        )

        # Degenerate multipath source
        _write_multipath_npy(
            os.path.join(out_dir, MULTIPATH_SUBDIR, f"{frame:05d}.npy")
        )

        # Per-frame lidar pcds (same cloud; we have no 5-win fusion).
        _write_pcd(xyz, os.path.join(out_dir, LIDAR_SUBDIR, f"{frame:05d}.pcd"))
        _write_pcd(xyz, os.path.join(out_dir, LIDAR_MAP_SUBDIR, f"{frame:05d}.pcd"))

        # Pose → rotated → TUM line
        T, R, t = common.pose_from_config(cfg)
        T_adj = rotate_pose_plus90_about_z(T)
        R_adj = T_adj[:3, :3]
        t_adj = T_adj[:3, 3]
        q = SciRot.from_matrix(R_adj).as_quat()  # (qx, qy, qz, qw) — TUM order
        tum_lines.append(
            f"{float(frame):.6f} "
            f"{t_adj[0]:.6f} {t_adj[1]:.6f} {t_adj[2]:.6f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
        )

    with open(os.path.join(out_dir, "radar_trajectory.tum"), "w") as f:
        f.writelines(tum_lines)

    _write_sensor_yaml(out_dir, range_resolution=range_res, W_range=256)

    manifest = {
        "scene": scene,
        "frames": frames,
        "train_frames": split["train_frames"],
        "test_frame": split["test_frame"],
        "H_fov": H_FOV,
        "W_range": 256,
        "range_resolution": range_res,
        "azimuth_resolution_deg": AZIMUTH_RES_DEG,
        "azimuth_coverage_deg": AZIMUTH_COVERAGE_DEG,
        "pose_rotation_about_world_z_deg": 90.0,
        "normalization": "per_frame_minmax_uint8",
        "deviations_from_reference": [
            "Synthetic 9-frame sequence rather than Boreas 40-frame window.",
            "sin-space → uniform-angle azimuth resample (adapter only).",
            "Pose rotation +90° CCW about world-Z to align the 180° cascade "
            "wedge with the first 200 rows of the rendered 360° polar.",
            "Sensor_type 'scanning_radar' with azimuth_coverage=180 plus "
            "upstream patch (patches/wedge_crop_guard.patch) to crop L1+SSIM+"
            "occ loss to the populated wedge rows.",
            "Degenerate radar_average_map_polar (all zeros); "
            "--l1occloss_lambda 0 to zero out its contribution.",
            "Degenerate multipath_model (empty dict); --multipath_weight 0.",
            "Per-frame lidar uses scene/pcl.npy (no 5-frame-window fusion).",
        ],
    }
    with open(os.path.join(out_dir, "adapter_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument(
        "--out-root",
        default=os.path.join(
            _REPO, "baselines", "radarsplat", "data_radarsplat"
        ),
    )
    args = ap.parse_args()
    m = build_scene(args.scene, args.out_root)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
