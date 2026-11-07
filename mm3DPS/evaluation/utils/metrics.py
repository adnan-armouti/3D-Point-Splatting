"""Metric computation utilities."""


import numpy as np






def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-30:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_cart_ra_metrics(ra_gt_cart: np.ndarray, ra_rend_cart: np.ndarray) -> dict:
    """Compute all RA image metrics from Cartesian RA magnitude images.

    Both inputs are 2D float arrays.
    Metrics are computed on independently min-max normalized images.
    """
    gt_norm = _minmax_normalize(ra_gt_cart)
    rend_norm = _minmax_normalize(ra_rend_cart)

    gt_flat = gt_norm.ravel()
    rend_flat = rend_norm.ravel()

    cart_corr = float(np.corrcoef(gt_flat, rend_flat)[0, 1])

    mse = float(np.mean((rend_norm - gt_norm) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim_val = float(ssim_fn(gt_norm, rend_norm, data_range=1.0))
    except ImportError:
        ssim_val = None

    return {
        "cart_corr": cart_corr,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim_val,
    }


