"""Metric harness — wraps the authoritative 3DPS metrics.

**No new metric code.** The baselines all funnel their rendered Cartesian RA
and the ground-truth Cartesian RA through ``compute_cart_ra_metrics`` so the
numbers are bit-identical to the 3DPS pipeline. Range-profile correlation
is a thin helper over the same range-cropped polar images.

Public API:
    run_eval(baseline_name, scene,
             rendered_ra_cart, gt_ra_cart,
             *, rendered_ra_polar_cropped=None,
             gt_ra_polar_cropped=None,
             extra=None) -> dict

    write_metrics_json(path, result) -> None

The returned dict matches the ``Result schema`` in ``baselines/README.md``.
Caller must have already applied the range-bin crop 15..110 before Cartesian
conversion (use ``baselines.common.adapters.range_crop`` + ``adc_to_cart_ra``).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def _range_profile_corr(rend_polar: np.ndarray, gt_polar: np.ndarray) -> float:
    """Pearson correlation of per-range-bin magnitude summed over azimuth.

    Both inputs are 2D polar magnitude images, already range-cropped to the
    same bins (typically 15..110). Azimuth is axis 0, range is axis 1. The
    range axis must be identical; azimuth-bin count may differ (one side
    may be in uniform-angle, the other in uniform-sin) since we sum it out.
    """
    if rend_polar.shape[1] != gt_polar.shape[1]:
        raise ValueError(
            f"range-axis length differs: rend={rend_polar.shape[1]} "
            f"gt={gt_polar.shape[1]}"
        )
    rp_rend = rend_polar.sum(axis=0).astype(np.float64)
    rp_gt = gt_polar.sum(axis=0).astype(np.float64)
    if rp_rend.std() < 1e-30 or rp_gt.std() < 1e-30:
        return 0.0
    return float(np.corrcoef(rp_rend, rp_gt)[0, 1])


def run_eval(
    baseline_name: str,
    scene: str,
    rendered_ra_cart: np.ndarray,
    gt_ra_cart: np.ndarray,
    *,
    rendered_ra_polar_cropped: Optional[np.ndarray] = None,
    gt_ra_polar_cropped: Optional[np.ndarray] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Compute the shared metrics dict for one scene, one baseline.

    ``rendered_ra_cart`` and ``gt_ra_cart`` are the Cartesian RA magnitude
    images returned by ``baselines.common.adapters.adc_to_cart_ra`` (or the
    rendered-side equivalent). If both polar arrays are supplied we also
    compute ``range_profile_corr`` — otherwise it is ``None``.
    """
    from mm3DPS.evaluation.utils.metrics import compute_cart_ra_metrics

    if rendered_ra_cart.shape != gt_ra_cart.shape:
        raise ValueError(
            f"cart RA shape mismatch: rend={rendered_ra_cart.shape} "
            f"gt={gt_ra_cart.shape}"
        )

    cart = compute_cart_ra_metrics(gt_ra_cart, rendered_ra_cart)

    rp: Optional[float]
    if rendered_ra_polar_cropped is not None and gt_ra_polar_cropped is not None:
        rp = _range_profile_corr(rendered_ra_polar_cropped, gt_ra_polar_cropped)
    else:
        rp = None

    out = {
        "baseline": baseline_name,
        "scene": scene,
        "ra_corr": cart["cart_corr"],
        "range_profile_corr": rp,
        "cart_mse": cart["mse"],
        "cart_rmse": cart["rmse"],
        "cart_psnr": cart["psnr"],
        "cart_ssim": cart["ssim"],
    }
    if extra:
        # Result-schema fields the caller knows (test_frame, train_frames,
        # wall_time_seconds, peak_gpu_mem_mib, deviations_from_reference).
        out.update(extra)
    return out


def write_metrics_json(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=_default_json)


def _default_json(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON-serializable: {type(x)}")
