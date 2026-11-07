"""
Batched GPU evaluation for coarse grid search optimization.

Evaluates multiple parameter combinations simultaneously for massive speedup.
"""

import numpy as np
from typing import Tuple, Dict, Any, List

import cupy as cp

from .voxelize_gpu import (
    voxelize_lidar_gpu,
    rae_to_ra_map_gpu,
)
from .metrics_gpu import (
    minmax_normalize_gpu,
    compute_correlation_batched_gpu,
    compute_metrics_batched_gpu,
)


def generate_parameter_grid(
    range_search_m: Tuple[float, float],
    azimuth_search_deg: Tuple[float, float],
    rotation_elev_search_deg: Tuple[float, float],
    rotation_azim_search_deg: Tuple[float, float],
    coarse_steps: int = 7
) -> np.ndarray:
    """
    Generate all parameter combinations for grid search.

    Args:
        range_search_m: (min, max) range translation in meters
        azimuth_search_deg: (min, max) azimuth translation in degrees
        rotation_elev_search_deg: (min, max) rotation around elev axis in degrees
        rotation_azim_search_deg: (min, max) rotation around azim axis in degrees
        coarse_steps: Number of steps per dimension

    Returns:
        params_grid: Array of shape (N, 4) where N = coarse_steps^4
                     Columns: [delta_range_m, delta_azimuth_deg, rot_elev_deg, rot_azim_deg]
    """
    range_grid = np.linspace(range_search_m[0], range_search_m[1], coarse_steps)
    azimuth_grid = np.linspace(azimuth_search_deg[0], azimuth_search_deg[1], coarse_steps)
    rot_elev_grid = np.linspace(rotation_elev_search_deg[0], rotation_elev_search_deg[1], coarse_steps)
    rot_azim_grid = np.linspace(rotation_azim_search_deg[0], rotation_azim_search_deg[1], coarse_steps)

    # Create meshgrid and flatten
    dr, da, re, ra = np.meshgrid(range_grid, azimuth_grid, rot_elev_grid, rot_azim_grid, indexing='ij')

    params_grid = np.stack([
        dr.ravel(),
        da.ravel(),
        re.ravel(),
        ra.ravel()
    ], axis=1).astype(np.float32)

    return params_grid


def apply_transformations_to_boresight_origin(
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    delta_range_m: float,
    delta_azimuth_deg: float,
    rotation_elev_deg: float,
    rotation_azim_deg: float,
    reference_range_m: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply transformation parameters to compute new boresight and origin.

    This is done on CPU as it's fast and only called B times (not N points).

    Args:
        base_boresight: Original boresight direction (3,)
        base_origin: Original radar origin (3,)
        delta_range_m: Range translation
        delta_azimuth_deg: Azimuth translation
        rotation_elev_deg: Rotation around elevation axis (Z-axis)
        rotation_azim_deg: Rotation around azimuth axis (board X-axis)
        reference_range_m: Reference range for azimuth translation

    Returns:
        new_boresight: Transformed boresight (3,)
        new_origin: Transformed origin (3,)
    """
    # Helper function for axis-angle rotation (CPU)
    def axis_angle_to_matrix(rot_vec):
        angle = np.linalg.norm(rot_vec)
        if angle < 1e-8:
            return np.eye(3, dtype=np.float32)
        axis = rot_vec / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    boresight = base_boresight.copy().astype(np.float32)

    # 1. Rotation around elevation axis (Z-axis)
    rot_elev_rad = np.radians(rotation_elev_deg)
    rot_vec_elev = np.array([0.0, 0.0, rot_elev_rad], dtype=np.float32)
    R_elev = axis_angle_to_matrix(rot_vec_elev)
    boresight = normalize(R_elev @ boresight)

    # 2. Rotation around azimuth axis (board X-axis)
    z_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_board = np.cross(boresight, z_world)
    x_board_norm = np.linalg.norm(x_board)
    if x_board_norm < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_board = x_board / x_board_norm

    rot_azim_rad = np.radians(rotation_azim_deg)
    rot_vec_azim = x_board * rot_azim_rad
    R_azim = axis_angle_to_matrix(rot_vec_azim)
    boresight = normalize(R_azim @ boresight)

    # 3. Translation
    range_offset = normalize(boresight) * delta_range_m
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)

    # Recompute x_board after rotation
    x_board = np.cross(boresight, z_world)
    x_board_norm = np.linalg.norm(x_board)
    if x_board_norm < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_board = x_board / x_board_norm

    azimuth_offset = x_board * delta_azimuth_m
    new_origin = base_origin.astype(np.float32) + range_offset + azimuth_offset

    return boresight, new_origin


def evaluate_alignment_batched_gpu(
    pts_gpu: 'cp.ndarray',
    intensities_gpu: 'cp.ndarray',
    ra_radar_gpu: 'cp.ndarray',
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    params_batch: np.ndarray,
    az_edges: 'cp.ndarray',
    el_edges: 'cp.ndarray',
    r_edges: 'cp.ndarray',
    antenna_weights: 'cp.ndarray' = None,
    near_field_m: float = 1.5,
    metric: str = 'correlation',
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Evaluate alignment for a batch of parameter combinations on GPU.

    Args:
        pts_gpu: Point cloud (N, 3) on GPU
        intensities_gpu: Point intensities (N,) on GPU
        ra_radar_gpu: Radar RA map (H, W) on GPU, already normalized
        base_boresight: Base boresight direction (3,) on CPU
        base_origin: Base radar origin (3,) on CPU
        params_batch: Parameter combinations (B, 4) on CPU
                      Columns: [delta_range_m, delta_azimuth_deg, rot_elev_deg, rot_azim_deg]
        az_edges: Azimuth bin edges on GPU
        el_edges: Elevation bin edges on GPU
        r_edges: Range bin edges on GPU
        antenna_weights: Optional antenna pattern weights on GPU
        near_field_m: Near-field threshold
        metric: Metric to optimize ('correlation' or 'mse')
        verbose: Print progress

    Returns:
        metrics: Array of metric values (B,) on CPU
        best_idx: Index of best parameter combination
    """

    B = params_batch.shape[0]
    num_az = len(az_edges) - 1
    num_r = len(r_edges) - 1

    # Determine if we maximize or minimize
    maximize = metric in ['correlation', 'ssim', 'iou']

    # Allocate output arrays
    ra_lidar_batch = cp.zeros((B, num_az, num_r), dtype=cp.float32)

    # Process each parameter combination
    for b in range(B):
        dr, da, re, ra_param = params_batch[b]

        # Compute transformed boresight and origin (CPU - fast)
        boresight, origin = apply_transformations_to_boresight_origin(
            base_boresight, base_origin, dr, da, re, ra_param
        )

        # Transfer to GPU
        boresight_gpu = cp.asarray(boresight, dtype=cp.float32)
        origin_gpu = cp.asarray(origin, dtype=cp.float32)

        # Voxelize on GPU
        rae_tensor = voxelize_lidar_gpu(
            pts_gpu, intensities_gpu,
            origin_gpu, boresight_gpu,
            az_edges, el_edges, r_edges,
            antenna_weights, near_field_m
        )

        # Collapse to RA map
        ra_lidar = rae_to_ra_map_gpu(rae_tensor)

        # Handle shape mismatch (127 vs 128 azimuth bins)
        if ra_lidar.shape[0] == 128 and ra_radar_gpu.shape[0] == 127:
            ra_lidar = ra_lidar[1:, :]

        # Normalize
        ra_lidar = minmax_normalize_gpu(ra_lidar)

        ra_lidar_batch[b] = ra_lidar

        if verbose and (b + 1) % 100 == 0:
            print(f"    Batch progress: {b + 1}/{B}")

    # Compute metrics for all batch items at once
    metrics_gpu = compute_metrics_batched_gpu(ra_lidar_batch, ra_radar_gpu, metric)

    # Transfer to CPU
    metrics = metrics_gpu.get()

    # Find best
    if maximize:
        best_idx = int(np.argmax(metrics))
    else:
        best_idx = int(np.argmin(metrics))

    return metrics, best_idx


def coarse_grid_search_gpu(
    pts_gpu: 'cp.ndarray',
    intensities_gpu: 'cp.ndarray',
    ra_radar_gpu: 'cp.ndarray',
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    range_search_m: Tuple[float, float],
    azimuth_search_deg: Tuple[float, float],
    rotation_elev_search_deg: Tuple[float, float],
    rotation_azim_search_deg: Tuple[float, float],
    az_edges: 'cp.ndarray',
    el_edges: 'cp.ndarray',
    r_edges: 'cp.ndarray',
    antenna_weights: 'cp.ndarray' = None,
    near_field_m: float = 1.5,
    metric: str = 'correlation',
    coarse_steps: int = 7,
    batch_size: int = 100,
    verbose: bool = True
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    GPU-accelerated coarse grid search.

    Args:
        pts_gpu: Point cloud (N, 3) on GPU
        intensities_gpu: Point intensities (N,) on GPU
        ra_radar_gpu: Radar RA map on GPU, normalized to [0,1]
        base_boresight: Base boresight (3,) on CPU
        base_origin: Base origin (3,) on CPU
        range_search_m: Range search limits
        azimuth_search_deg: Azimuth search limits
        rotation_elev_search_deg: Rotation elev search limits
        rotation_azim_search_deg: Rotation azim search limits
        az_edges, el_edges, r_edges: Bin edges on GPU
        antenna_weights: Optional antenna weights on GPU
        near_field_m: Near-field threshold
        metric: Metric to optimize
        coarse_steps: Grid steps per dimension
        batch_size: Batch size for GPU evaluation
        verbose: Print progress

    Returns:
        best_params: Best parameter combination (4,)
        best_value: Best metric value
        all_results: All (params, metric) results
    """

    # Generate all parameter combinations
    params_grid = generate_parameter_grid(
        range_search_m, azimuth_search_deg,
        rotation_elev_search_deg, rotation_azim_search_deg,
        coarse_steps
    )

    total_evals = params_grid.shape[0]
    if verbose:
        print(f"Phase 1: GPU coarse grid search ({coarse_steps}^4 = {total_evals} evaluations)...")

    maximize = metric in ['correlation', 'ssim', 'iou']
    best_value = -np.inf if maximize else np.inf
    best_params = params_grid[0].copy()
    all_metrics = []

    # Process in batches
    num_batches = (total_evals + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_evals)
        params_batch = params_grid[start_idx:end_idx]

        metrics, local_best_idx = evaluate_alignment_batched_gpu(
            pts_gpu, intensities_gpu, ra_radar_gpu,
            base_boresight, base_origin, params_batch,
            az_edges, el_edges, r_edges,
            antenna_weights, near_field_m, metric,
            verbose=False
        )

        all_metrics.extend(metrics.tolist())

        # Update global best
        if maximize:
            batch_best_value = metrics[local_best_idx]
            if batch_best_value > best_value:
                best_value = batch_best_value
                best_params = params_batch[local_best_idx].copy()
        else:
            batch_best_value = metrics[local_best_idx]
            if batch_best_value < best_value:
                best_value = batch_best_value
                best_params = params_batch[local_best_idx].copy()

        if verbose and (batch_idx + 1) % 5 == 0:
            print(f"  Progress: {end_idx}/{total_evals} ({100*end_idx/total_evals:.1f}%)")

    if verbose:
        print(f"  Best grid point: range={best_params[0]:.2f}m, az={best_params[1]:.2f}deg, "
              f"rot_elev={best_params[2]:.2f}deg, rot_azim={best_params[3]:.2f}deg")
        print(f"  Best {metric}: {best_value:.4f}")

    # Combine params and metrics for output
    all_results = np.column_stack([params_grid, np.array(all_metrics)])

    return best_params, best_value, all_results
