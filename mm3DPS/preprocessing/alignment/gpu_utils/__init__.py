"""
GPU utilities for LiDAR-Radar alignment.

This module provides GPU-accelerated functions for:
- Voxelization of point clouds
- Metric computation (correlation, MSE, SSIM, IoU)
- Batched evaluation for grid search optimization
"""

from .voxelize_gpu import (
    voxelize_lidar_gpu,
    voxelize_lidar_batched_gpu,
    rae_to_ra_map_gpu,
)

from .metrics_gpu import (
    compute_alignment_metrics_gpu,
    compute_correlation_gpu,
    compute_mse_gpu,
)

from .batch_eval_gpu import (
    evaluate_alignment_batched_gpu,
    generate_parameter_grid,
)

__all__ = [
    'voxelize_lidar_gpu',
    'voxelize_lidar_batched_gpu',
    'rae_to_ra_map_gpu',
    'compute_alignment_metrics_gpu',
    'compute_correlation_gpu',
    'compute_mse_gpu',
    'evaluate_alignment_batched_gpu',
    'generate_parameter_grid',
]
