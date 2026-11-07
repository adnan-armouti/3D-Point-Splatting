"""Shared helpers for LiDAR alignment: point cloud, radar config and GT RA loading."""

# Set Mitsuba variant before any mm3DPS imports
import mitsuba as mi
if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

import os
import json
import numpy as np
from typing import Dict, Any, Tuple

import open3d as o3d

# ============================================================================
# HELPER: Load point cloud from PLY or NPY
# ============================================================================

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Load point cloud from PLY or NPY file.

    For NPY files, assumes format:
    - (N, 3): x, y, z only
    - (N, 4): x, y, z, intensity
    - (N, 6): x, y, z, r, g, b OR x, y, z, n_x, n_y, n_z
    - (N, 7): x, y, z, n_x, n_y, n_z, intensity (our LiDAR format)

    Returns:
        Open3D PointCloud with points and optionally colors/intensities
    """
    path = str(path)

    if path.endswith('.ply'):
        return o3d.io.read_point_cloud(path)

    elif path.endswith('.npy'):
        data = np.load(path)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))

        ncols = data.shape[1]

        if ncols == 4:
            # x, y, z, intensity - store as grayscale colors
            intensity = data[:, 3]
            # Normalize intensity to [0, 1]
            if intensity.max() > intensity.min():
                intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            else:
                intensity_norm = np.ones_like(intensity)
            colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        elif ncols == 7:
            # x, y, z, n_x, n_y, n_z, intensity - use column 6 (intensity)
            # This is our LiDAR format: points + normals + intensity
            intensity = data[:, 6]  # Last column is intensity
            print(f"[load_point_cloud] 7-column format detected: using column 6 as intensity")
            print(f"  Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
            # Normalize intensity to [0, 1]
            if intensity.max() > intensity.min():
                intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            else:
                intensity_norm = np.ones_like(intensity)
            colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        elif ncols == 6:
            # x, y, z, r, g, b OR x, y, z, n_x, n_y, n_z
            # Check if values are in typical color range vs normal range
            vals = data[:, 3:6]
            if vals.min() >= 0 and vals.max() <= 255:
                # Likely RGB colors
                colors = vals
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            else:
                # Likely normals (can be negative) - use uniform intensity
                print(f"[load_point_cloud] 6-column format with negative values: assuming normals, using uniform intensity")
                intensity_norm = np.ones((data.shape[0],), dtype=np.float64)
                colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
                pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    else:
        raise ValueError(f"Unsupported file format: {path}. Expected .ply or .npy")


# ============================================================================
# CATEGORY 1: DIRECT IMPORTS (100% reused, debugged code)
# ============================================================================

# Rotation utilities (Rodrigues' formula)

# Config loading
from mm3DPS.evaluation.utils.single_view_viz import (
    load_config_positions_and_boresight,  # single_view_viz.py:189-240
)

# Point cloud utilities

# ADC/RA processing
from mm3DPS.data.ra_utils import adc_to_ra_image
from mm3DPS.data.io_utils import compute_range_res_from_cfg, load_target


# ============================================================================
# ANTENNA BEAM PATTERN WEIGHTING
# ============================================================================

# Default antenna pattern path — MMWCAS TX (shared with CUDA renderer).
# Resolved relative to the repo root so the vendored copy works the same
# regardless of cwd.
DEFAULT_ANTENNA_PATTERN_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'assets', 'antenna_pattern', 'MMWCAS', 'tx1_76.npy'))
if not os.path.isfile(DEFAULT_ANTENNA_PATTERN_PATH):
    DEFAULT_ANTENNA_PATTERN_PATH = None  # fall back; caller must pass explicit path


def load_antenna_pattern(pattern_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load antenna beam pattern from NPY file.

    The pattern file has shape (361, 2):
    - 361 rows: angles from -180° to +180° at 1° steps
    - Column 0: Elevation gain in dB (narrower beam, ~41° 0dB width)
    - Column 1: Azimuth gain in dB (wider beam, ~136° 0dB width)

    Args:
        pattern_path: Path to antenna pattern file. If None, uses default.

    Returns:
        angles_deg: Array of angles in degrees, shape (361,)
        az_gain_db: Azimuth gain in dB, shape (361,)
        el_gain_db: Elevation gain in dB, shape (361,)
    """
    if pattern_path is None:
        pattern_path = DEFAULT_ANTENNA_PATTERN_PATH

    pattern = np.load(pattern_path)

    # Pattern has 361 samples from -180 to +180 degrees
    angles_deg = np.linspace(-180, 180, pattern.shape[0])
    # Column 1 = azimuth (wider beam, ~136° 0dB width)
    # Column 0 = elevation (narrower beam, ~41° 0dB width)
    az_gain_db = pattern[:, 1]
    el_gain_db = pattern[:, 0]

    return angles_deg, az_gain_db, el_gain_db


def interpolate_antenna_gain(angles_deg: np.ndarray, gain_db: np.ndarray,
                              query_angles_deg: np.ndarray) -> np.ndarray:
    """
    Interpolate antenna gain for query angles.

    Args:
        angles_deg: Reference angles in degrees, shape (N,)
        gain_db: Gain values in dB at reference angles, shape (N,)
        query_angles_deg: Query angles in degrees, shape (...)

    Returns:
        Interpolated gain in dB, same shape as query_angles_deg
    """
    # Wrap query angles to [-180, 180]
    query_wrapped = np.mod(query_angles_deg + 180, 360) - 180

    # Interpolate (linear)
    return np.interp(query_wrapped, angles_deg, gain_db)






# ============================================================================
# STEP 1: Load Configuration and Data
# ============================================================================

def load_radar_config(config_path: str) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Load radar config - wraps existing load_config_positions_and_boresight.

    REUSE: Calls load_config_positions_and_boresight directly.

    Returns:
        params: Dict with num_az_bins, num_el_bins, num_adc, range_resolution
        cfg_center: Radar board center position (3,)
        cfg_bore: Boresight direction unit vector (3,)
    """
    tx_m, rx_m, cfg_center, cfg_bore = load_config_positions_and_boresight(config_path)

    # Build params dict (same pattern as single_view_viz.py:213-240)
    # Note: num_az_bins and num_el_bins are 127 (not 128) because radar FFT
    # drops the first bin (DC bin) before fft shift, resulting in 128-1=127 bins.
    # This ensures center azimuth bin (63) aligns with boresight direction.
    params = {
        'range_resolution': compute_range_res_from_cfg(config_path),
        'num_adc': 256,  # Default, override from config if available
        'num_az_bins': 127,  # 128-1: drop DC bin to match radar FFT output
        'num_el_bins': 127,  # 128-1: drop DC bin to match radar FFT output
    }

    # Load numAdcSamples from config if available
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if 'numAdcSamples' in cfg:
        params['num_adc'] = int(cfg['numAdcSamples'])

    return params, cfg_center, cfg_bore


# ============================================================================
# STEP 2: Transformation Functions for Boresight and Origin
# ============================================================================









# ============================================================================
# STEP 3: Voxelization with Transformed Grid
# ADAPTED from single_view_viz.py:327-419 (voxelize_lidar_to_radar_grid)
# Changes: Add origin/boresight params, return RAE tensor instead of points
# ============================================================================





# ============================================================================
# STEP 4: Process Radar Ground Truth to RA Map
# REUSE: 100% - just calls existing functions
# ============================================================================

def radar_gt_to_ra_map(gt_adc_path: str) -> np.ndarray:
    """
    Load radar ground truth ADC and convert to magnitude RA map.

    Calls load_target (mm3DPS.data.io_utils) and adc_to_ra_image (mm3DPS.data.ra_utils).

    Returns:
        ra_map: Shape (127, 256) - magnitude RA map (azimuth x range)
    """
    adc_data = load_target(gt_adc_path, device='cpu')
    ra_map = adc_to_ra_image(adc_data)
    return ra_map.numpy()  # Shape (127, 256)


# ============================================================================
# STEP 5: Alignment Metrics
# NEW: Standard metrics using numpy/scipy/skimage
# ============================================================================



# ============================================================================
# STEP 6: Single Evaluation Function
# NEW: Orchestration function
# ============================================================================



# ============================================================================
# STEP 7: Coarse-to-Fine Optimization
# NEW: Grid search + scipy.optimize.minimize
# ============================================================================



# ============================================================================
# STEP 8: Helper function to get RA map for given params (for visualization)
# ============================================================================



# ============================================================================
# STEP 9: Visualization Functions
# NEW: Matplotlib visualizations
# ============================================================================





# ============================================================================
# STEP 10: Output Functions
# ============================================================================





# ============================================================================
# CREATE ALIGNED CONFIG
# ============================================================================



# ============================================================================
# MAIN
# ============================================================================
