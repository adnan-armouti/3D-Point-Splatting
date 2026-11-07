"""
GPU-accelerated metric computation for LiDAR-Radar alignment.

Provides correlation, MSE, SSIM, and IoU metrics computed on GPU.
"""

import numpy as np

import cupy as cp



def minmax_normalize_gpu(arr: cp.ndarray) -> cp.ndarray:
    """
    Min-max normalize array to [0, 1] range on GPU.

    Args:
        arr: Input array on GPU

    Returns:
        Normalized array on GPU
    """
    arr_min = cp.min(arr)
    arr_max = cp.max(arr)
    if arr_max - arr_min < 1e-10:
        return cp.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def compute_correlation_gpu(a: cp.ndarray, b: cp.ndarray) -> float:
    """
    Compute Pearson correlation coefficient on GPU.

    Args:
        a: First array on GPU (will be flattened)
        b: Second array on GPU (will be flattened)

    Returns:
        Correlation coefficient (scalar, on CPU)
    """

    a_flat = a.ravel()
    b_flat = b.ravel()

    # Check for zero variance
    std_a = cp.std(a_flat)
    std_b = cp.std(b_flat)

    if std_a < 1e-10 or std_b < 1e-10:
        return 0.0

    # Compute correlation
    a_centered = a_flat - cp.mean(a_flat)
    b_centered = b_flat - cp.mean(b_flat)

    corr = cp.sum(a_centered * b_centered) / (cp.sqrt(cp.sum(a_centered**2)) * cp.sqrt(cp.sum(b_centered**2)))

    return float(corr.get())


def compute_mse_gpu(a: cp.ndarray, b: cp.ndarray) -> float:
    """
    Compute Mean Squared Error on GPU.

    Args:
        a: First array on GPU
        b: Second array on GPU

    Returns:
        MSE (scalar, on CPU)
    """
    mse = cp.mean((a - b) ** 2)
    return float(mse.get())


def compute_ssim_gpu(a: cp.ndarray, b: cp.ndarray, data_range: float = 1.0) -> float:
    """
    Compute Structural Similarity Index (SSIM) on GPU.

    Simplified SSIM computation without windowing for speed.
    For full SSIM, consider using skimage on CPU.

    Args:
        a: First 2D array on GPU
        b: Second 2D array on GPU
        data_range: Dynamic range of the images

    Returns:
        SSIM value (scalar, on CPU)
    """

    # Constants for numerical stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Global statistics (simplified SSIM without local windows)
    mu_a = cp.mean(a)
    mu_b = cp.mean(b)
    sigma_a_sq = cp.var(a)
    sigma_b_sq = cp.var(b)
    sigma_ab = cp.mean((a - mu_a) * (b - mu_b))

    # SSIM formula
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)

    ssim = numerator / denominator
    return float(ssim.get())


def compute_iou_gpu(a: cp.ndarray, b: cp.ndarray, threshold_percentile: float = 90.0) -> float:
    """
    Compute Intersection over Union on GPU.

    Binarizes both arrays at the given percentile threshold.

    Args:
        a: First array on GPU
        b: Second array on GPU
        threshold_percentile: Percentile for binarization

    Returns:
        IoU value (scalar, on CPU)
    """

    thresh_a = cp.percentile(a, threshold_percentile)
    thresh_b = cp.percentile(b, threshold_percentile)

    binary_a = a > thresh_a
    binary_b = b > thresh_b

    intersection = cp.sum(binary_a & binary_b)
    union = cp.sum(binary_a | binary_b)

    iou = intersection / (union + 1e-8)
    return float(iou.get())


def compute_alignment_metrics_gpu(
    ra_lidar: cp.ndarray,
    ra_radar: cp.ndarray,
    threshold_percentile: float = 90.0
) -> dict:
    """
    Compute all alignment metrics on GPU.

    Args:
        ra_lidar: LiDAR RA map on GPU (already normalized to [0,1])
        ra_radar: Radar RA map on GPU (already normalized to [0,1])
        threshold_percentile: Percentile for IoU binarization

    Returns:
        Dict with correlation, mse, ssim, iou
    """

    # Handle NaN/Inf
    ra_lidar = cp.nan_to_num(ra_lidar, nan=0.0, posinf=0.0, neginf=0.0)
    ra_radar = cp.nan_to_num(ra_radar, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'correlation': compute_correlation_gpu(ra_lidar, ra_radar),
        'mse': compute_mse_gpu(ra_lidar, ra_radar),
        'ssim': compute_ssim_gpu(ra_lidar, ra_radar),
        'iou': compute_iou_gpu(ra_lidar, ra_radar, threshold_percentile),
    }


def compute_correlation_batched_gpu(
    ra_lidar_batch: cp.ndarray,
    ra_radar: cp.ndarray
) -> cp.ndarray:
    """
    Compute correlation for a batch of LiDAR RA maps against a single radar RA map.

    Args:
        ra_lidar_batch: Batch of LiDAR RA maps (B, H, W) on GPU
        ra_radar: Single radar RA map (H, W) on GPU

    Returns:
        correlations: Array of correlation values (B,) on GPU
    """

    B = ra_lidar_batch.shape[0]
    H, W = ra_radar.shape

    # Flatten each RA map: (B, H, W) -> (B, H*W)
    lidar_flat = ra_lidar_batch.reshape(B, -1)
    radar_flat = ra_radar.ravel()

    # Center the data
    lidar_mean = cp.mean(lidar_flat, axis=1, keepdims=True)
    radar_mean = cp.mean(radar_flat)

    lidar_centered = lidar_flat - lidar_mean
    radar_centered = radar_flat - radar_mean

    # Compute correlation for each batch item
    # corr[b] = sum(lidar[b] * radar) / (||lidar[b]|| * ||radar||)
    numerator = cp.sum(lidar_centered * radar_centered, axis=1)
    lidar_norm = cp.sqrt(cp.sum(lidar_centered**2, axis=1))
    radar_norm = cp.sqrt(cp.sum(radar_centered**2))

    # Handle zero variance
    lidar_norm = cp.maximum(lidar_norm, 1e-10)

    correlations = numerator / (lidar_norm * radar_norm)

    return correlations


def compute_metrics_batched_gpu(
    ra_lidar_batch: cp.ndarray,
    ra_radar: cp.ndarray,
    metric: str = 'correlation'
) -> cp.ndarray:
    """
    Compute specified metric for a batch of LiDAR RA maps.

    Args:
        ra_lidar_batch: Batch of LiDAR RA maps (B, H, W) on GPU
        ra_radar: Single radar RA map (H, W) on GPU
        metric: Metric to compute ('correlation', 'mse')

    Returns:
        metrics: Array of metric values (B,) on GPU
    """

    if metric == 'correlation':
        return compute_correlation_batched_gpu(ra_lidar_batch, ra_radar)
    elif metric == 'mse':
        # MSE is simple: mean of squared differences per batch item
        B = ra_lidar_batch.shape[0]
        diff = ra_lidar_batch - ra_radar[cp.newaxis, :, :]
        mse = cp.mean(diff**2, axis=(1, 2))
        return mse
    else:
        raise ValueError(f"Batched metric '{metric}' not supported. Use 'correlation' or 'mse'.")
