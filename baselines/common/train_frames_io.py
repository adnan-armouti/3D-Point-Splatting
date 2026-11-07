"""Per-train-frame artefact I/O for the supplement figure pipeline.

Each baseline already renders the held-out test frame and saves
``rendered_ra_cart.npy`` + dB/linear PNGs + ``metrics.json`` under its scene
output directory. For the supplement figure we need the same artefacts for
each of the 8 training frames so the figure script can show train + test
renders side by side.

All three baselines + our method share the same per-frame save layout:

    <scene_dir>/train_frames/frame_<F_train>/
        rendered_ra_cart.npy        # (399, 399) float32, after polar→cart
        rendered_ra_polar.npy       # native polar (optional — only when available)
        rendered_ra_dB.png          # matplotlib dB-scale render
        rendered_ra_linear.png      # matplotlib linear-scale render
        gt_ra_cart.npy              # GT Cartesian (399, 399) float32
        gt_ra_polar_full.npy        # GT polar (azimuth × range, uncropped) optional
        gt_ra_dB.png                # GT dB image
        gt_ra_linear.png            # GT linear image
        metrics.json                # per-frame compute_cart_ra_metrics output

Plus a parallel per-scene aggregate:

    <scene_dir>/metrics_train.json
        {"per_frame": [{frame, ra_corr, range_profile_corr, ...}, ...],
         "ra_corr_mean":    <float>, "ra_corr_std":    <float>,
         "range_profile_corr_mean":    <float>, "range_profile_corr_std": <float>}

This module is intended to be importable from the mm3DPS env (the env where
``compute_cart_ra_metrics`` lives) — i.e. it is callable from each
baseline's ``finalize_metrics.py`` and from ``mm3DPS.train_frame_nvs``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


def _save_ra_pngs(ra_cart: np.ndarray, out_dir: str, prefix: str,
                  range_res: float, title_prefix: Optional[str] = None) -> None:
    """Write ``<prefix>_dB.png`` and ``<prefix>_linear.png`` under ``out_dir``.

    Uses ``mm3DPS.evaluation.utils.visualization.save_ra_cartesian_png`` so the
    rendering style matches the test-frame figures saved by each baseline's
    ``finalize_metrics.py``.
    """
    from mm3DPS.evaluation.utils.visualization import save_ra_cartesian_png

    title_prefix = title_prefix or prefix
    for scale in ("linear", "dB"):
        save_ra_cartesian_png(
            ra_cart,
            os.path.join(out_dir, f"{prefix}_{scale}.png"),
            range_res=range_res, scale=scale,
            title=f"{title_prefix} ({scale})",
        )


def save_per_train_frame(
    *,
    out_root: str,
    frame: int,
    rendered_ra_cart: np.ndarray,
    gt_ra_cart: np.ndarray,
    range_res: float,
    baseline_name: str,
    scene: str,
    rendered_ra_polar: Optional[np.ndarray] = None,
    gt_ra_polar_full: Optional[np.ndarray] = None,
    rendered_ra_polar_cropped: Optional[np.ndarray] = None,
    gt_ra_polar_cropped: Optional[np.ndarray] = None,
    extra: Optional[dict] = None,
    image_title_suffix: Optional[str] = None,
) -> Dict:
    """Save per-frame artefacts under ``out_root/train_frames/frame_<F>/``.

    Returns the metrics dict (as written to ``metrics.json``) so callers can
    aggregate across frames. ``rendered_ra_polar*``/``gt_ra_polar*`` are
    optional — when both polar-cropped arrays are passed we also compute
    ``range_profile_corr`` (otherwise it is ``None``).
    """
    from baselines.common import eval as common_eval

    frame_dir = os.path.join(out_root, "train_frames", f"frame_{int(frame)}")
    os.makedirs(frame_dir, exist_ok=True)

    np.save(os.path.join(frame_dir, "rendered_ra_cart.npy"),
            rendered_ra_cart.astype(np.float32))
    np.save(os.path.join(frame_dir, "gt_ra_cart.npy"),
            gt_ra_cart.astype(np.float32))
    if rendered_ra_polar is not None:
        np.save(os.path.join(frame_dir, "rendered_ra_polar.npy"),
                rendered_ra_polar.astype(np.float32))
    if gt_ra_polar_full is not None:
        np.save(os.path.join(frame_dir, "gt_ra_polar_full.npy"),
                gt_ra_polar_full.astype(np.float32))

    title_pre = (
        f"{baseline_name} train frame {frame}{(' '+image_title_suffix) if image_title_suffix else ''}"
    )
    _save_ra_pngs(rendered_ra_cart, frame_dir, "rendered_ra",
                  range_res=range_res, title_prefix=title_pre)
    _save_ra_pngs(gt_ra_cart, frame_dir, "gt_ra",
                  range_res=range_res,
                  title_prefix=f"GT train frame {frame}")

    extra_full = {
        "frame": int(frame),
    }
    if extra:
        extra_full.update(extra)

    result = common_eval.run_eval(
        baseline_name, scene,
        rendered_ra_cart=rendered_ra_cart,
        gt_ra_cart=gt_ra_cart,
        rendered_ra_polar_cropped=rendered_ra_polar_cropped,
        gt_ra_polar_cropped=gt_ra_polar_cropped,
        extra=extra_full,
    )
    common_eval.write_metrics_json(
        os.path.join(frame_dir, "metrics.json"), result
    )
    return result


def write_train_aggregate(out_root: str, per_frame_results: List[Dict]) -> None:
    """Write ``<out_root>/metrics_train.json`` summarising per-train-frame
    correlations (mean + std across the 8 frames)."""
    if not per_frame_results:
        return

    ra_corrs = [float(r["ra_corr"]) for r in per_frame_results
                if r.get("ra_corr") is not None]
    rp_corrs = [float(r["range_profile_corr"]) for r in per_frame_results
                if r.get("range_profile_corr") is not None]

    agg = {
        "per_frame": per_frame_results,
        "ra_corr_mean": float(np.mean(ra_corrs)) if ra_corrs else None,
        "ra_corr_std":  float(np.std(ra_corrs))  if ra_corrs else None,
        "ra_corr_per_frame": ra_corrs,
        "range_profile_corr_mean": (
            float(np.mean(rp_corrs)) if rp_corrs else None
        ),
        "range_profile_corr_std": (
            float(np.std(rp_corrs))  if rp_corrs else None
        ),
        "n_train_frames": len(per_frame_results),
    }
    with open(os.path.join(out_root, "metrics_train.json"), "w") as f:
        json.dump(agg, f, indent=2, default=_default_json)


def _default_json(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON-serializable: {type(x)}")
