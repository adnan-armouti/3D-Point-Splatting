"""Phase-B finalization: read the radarsplat env's rendered polar + metadata,
compute GT-side, run metrics via baselines.common.eval. Runs in mm3DPS env.

Invoked by run_radarsplat_scene.py at the end of training as a subprocess —
separated because the radarsplat env is Python 3.9, while mm3DPS.data.ra_utils
uses PEP-604 union syntax (``float | None``) that requires Python 3.10+.

Inputs (all in --out-dir):
    rendered_ra_polar.npy   # (H_fov, W) float32
    train_meta.json         # {scene, test_frame, train_frames, wall_time_seconds, ...}

Outputs:
    metrics.json            # the Result schema dict
    gt_ra_polar_cropped.npy
    gt_ra_cart.npy
    rendered_ra_cart.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.common import adapters as common_adapt  # noqa: E402
from baselines.common import eval as common_eval  # noqa: E402
from baselines.common import nvs_split  # noqa: E402


def _gt_side(scene: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (gt_polar_full, gt_polar_cropped, gt_cart, range_res).

    ``gt_polar_full`` is the un-cropped (127, 256) sin-space polar — used
    for cart conversion (matches mm3DPS reference, which builds the
    GT cart from the full polar covering the full 15.18 m radar range).
    ``gt_polar_cropped`` is the bins 15..110 slice — kept for the
    range-profile correlation only.
    """
    split = nvs_split.cascaded_split(scene)
    adc = common_adapt.load_cascaded_adc(split["test_file"])
    # Polar magnitude via adc_to_ra_complex + abs .
    ra_polar_full = common_adapt.adc_to_polar_ra(adc, sensor="cascaded")
    ra_polar_cropped = common_adapt.range_crop(ra_polar_full)

    from mm3DPS.data.io_utils import compute_range_res_from_cfg
    from mm3DPS.data.ra_utils import ra_polar_to_cartesian

    range_res = compute_range_res_from_cfg(split["test_config"])
    # Cart from FULL polar — matches the reference neighbour_avg_baseline reference.
    ra_cart = ra_polar_to_cartesian(ra_polar_full, range_res).astype(np.float32)
    return (ra_polar_full.astype(np.float32),
            ra_polar_cropped.astype(np.float32),
            ra_cart,
            range_res)


def _gt_side_for_file(adc_path: str, cfg_path: str
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Same shape as ``_gt_side`` but for an arbitrary cascade ADC path."""
    adc = common_adapt.load_cascaded_adc(adc_path)
    ra_polar_full = common_adapt.adc_to_polar_ra(adc, sensor="cascaded")
    ra_polar_cropped = common_adapt.range_crop(ra_polar_full)

    from mm3DPS.data.io_utils import compute_range_res_from_cfg
    from mm3DPS.data.ra_utils import ra_polar_to_cartesian

    range_res = compute_range_res_from_cfg(cfg_path)
    ra_cart = ra_polar_to_cartesian(ra_polar_full, range_res).astype(np.float32)
    return (ra_polar_full.astype(np.float32),
            ra_polar_cropped.astype(np.float32),
            ra_cart,
            range_res)


def _match_shapes(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])
    def crop(x):
        dh, dw = (x.shape[0] - H) // 2, (x.shape[1] - W) // 2
        return x[dh:dh + H, dw:dw + W]
    return crop(a), crop(b)


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline-name", default="radarsplat")
    args = ap.parse_args()

    meta_path = os.path.join(args.out_dir, "train_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    rendered_polar = np.load(os.path.join(args.out_dir, "rendered_ra_polar.npy"))

    gt_polar_full, gt_polar_cropped, gt_cart, range_res = _gt_side(meta["scene"])

    # Resample the rendered polar from RadarSplat's uniform-angle space (200
    # bins over [-90°, +90°]) onto 3DPS's 127-bin sin-space grid, so both
    # GT and rendered are in the SAME azimuth space when fed to
    # ra_polar_to_cartesian (which assumes sin-space). Without this the
    # rendered wedge edges get compressed by ~6-9° and the cart_corr is
    # not a fair comparison vs the sin-space GT.
    from baselines.radarsplat.adapter.mm3dps_to_radarsplat import (
        resample_polar_angle_to_sin,
    )
    rendered_polar_sin = resample_polar_angle_to_sin(rendered_polar, H_sin=127)

    # Pred side: build cart from the FULL rendered polar (now sin-space,
    # matching the reference). The model was trained on the full 256-bin
    # range, so its rendered output covers the same physical extent.
    from mm3DPS.data.ra_utils import ra_polar_to_cartesian
    rend_cart = ra_polar_to_cartesian(rendered_polar_sin, range_res).astype(np.float32)
    # Range-profile correlation: cropped polar in each side's native
    # azimuth representation (axis 0 is summed out).
    rend_polar_cropped = common_adapt.range_crop(rendered_polar)

    gt_cart_m, rend_cart_m = _match_shapes(gt_cart, rend_cart)
    # Range-profile correlation: sum over azimuth of the bins-15..110 slice.
    # Range axes must match in length; azimuth count may differ
    # (gt is sin-space 127 bins, rend is uniform-angle 200 bins).
    W = min(gt_polar_cropped.shape[1], rend_polar_cropped.shape[1])
    gt_polar_m = gt_polar_cropped[:, :W]
    rend_polar_m = rend_polar_cropped[:, :W]

    extra = {
        "test_frame": int(meta["test_frame"]),
        "train_frames": [int(x) for x in meta["train_frames"]],
        "wall_time_seconds": float(meta["wall_time_seconds"]),
        "peak_gpu_mem_mib": float(meta["peak_gpu_mem_mib"]),
        "deviations_from_reference": meta.get("deviations_from_reference", []),
        "upstream_commit": meta.get("upstream_commit", "unknown"),
        "upstream_result_dir": meta.get("upstream_result_dir", None),
        "max_steps": int(meta.get("max_steps", 0)),
        "init_num_pts": int(meta.get("init_num_pts", 0)),
    }
    result = common_eval.run_eval(
        args.baseline_name,
        meta["scene"],
        rendered_ra_cart=rend_cart_m,
        gt_ra_cart=gt_cart_m,
        rendered_ra_polar_cropped=rend_polar_m,
        gt_ra_polar_cropped=gt_polar_m,
        extra=extra,
    )

    np.save(os.path.join(args.out_dir, "gt_ra_polar_full.npy"), gt_polar_full)
    np.save(os.path.join(args.out_dir, "gt_ra_polar_cropped.npy"), gt_polar_cropped)
    np.save(os.path.join(args.out_dir, "gt_ra_cart.npy"), gt_cart)
    np.save(os.path.join(args.out_dir, "rendered_ra_polar_sin.npy"), rendered_polar_sin)
    np.save(os.path.join(args.out_dir, "rendered_ra_cart.npy"), rend_cart)
    common_eval.write_metrics_json(
        os.path.join(args.out_dir, "metrics.json"), result
    )

    # Inspection PNGs — same util as mm3DPS (mm3DPS.evaluation.utils.visualization).
    # Generated for the held-out test frame at the final training iteration only.
    from mm3DPS.evaluation.utils.visualization import save_ra_cartesian_png
    for scale in ("linear", "dB"):
        save_ra_cartesian_png(
            gt_cart_m,
            os.path.join(args.out_dir, f"gt_ra_{scale}.png"),
            range_res=range_res,
            scale=scale,
            title=f"GT (test frame {meta['test_frame']}, {scale})",
        )
        save_ra_cartesian_png(
            rend_cart_m,
            os.path.join(args.out_dir, f"rasterized_ra_{scale}.png"),
            range_res=range_res,
            scale=scale,
            title=f"{args.baseline_name} rasterized (test frame {meta['test_frame']}, {scale})",
        )

    # ------------------------------------------------------------------
    # Per-train-frame artefacts (supplement figure pipeline).
    # ------------------------------------------------------------------
    _process_train_frames(args.out_dir, args.baseline_name,
                          meta, gt_cart.shape)

    print(json.dumps(result, indent=2))
    return 0


def _process_train_frames(out_dir: str, baseline_name: str,
                          meta: dict, gt_cart_shape) -> None:
    """For each training frame: build GT (cart + polar), load the runner's
    rendered polar dump, run the same sin-resample + cart conversion as the
    test path, and write per-frame artefacts under ``train_frames/frame_<F>``.

    Skips silently if the runner did not produce per-train-frame dumps
    (back-compat with older runs).
    """
    from baselines.common import train_frames_io
    from baselines.radarsplat.adapter.mm3dps_to_radarsplat import (
        resample_polar_angle_to_sin,
    )
    from mm3DPS.data.ra_utils import ra_polar_to_cartesian

    scene = meta["scene"]
    split = nvs_split.cascaded_split(scene)
    train_frames = list(map(int, split["train_frames"]))
    train_files = list(split["train_files"])
    train_configs = list(split["train_configs"])

    raw_dir = os.path.join(out_dir, "train_frames_polar_raw")
    if not os.path.isdir(raw_dir):
        print(f"[finalize] no train_frames_polar_raw/ in {out_dir}; skipping")
        return

    per_frame_results = []
    for f, adc_path, cfg_path in zip(train_frames, train_files, train_configs):
        rend_path = os.path.join(raw_dir, f"rendered_ra_polar_frame_{f}.npy")
        if not os.path.isfile(rend_path):
            print(f"[finalize] missing {rend_path}; skipping frame {f}")
            continue

        rendered_polar = np.load(rend_path).astype(np.float32)
        rendered_polar = np.clip(rendered_polar, 0.0, 1.0)

        gt_polar_full, gt_polar_cropped, gt_cart, range_res = _gt_side_for_file(
            adc_path, cfg_path,
        )

        rendered_polar_sin = resample_polar_angle_to_sin(rendered_polar, H_sin=127)
        rend_cart = ra_polar_to_cartesian(rendered_polar_sin, range_res).astype(np.float32)
        rend_polar_cropped = common_adapt.range_crop(rendered_polar)

        gt_cart_m, rend_cart_m = _match_shapes(gt_cart, rend_cart)
        W = min(gt_polar_cropped.shape[1], rend_polar_cropped.shape[1])
        gt_polar_m = gt_polar_cropped[:, :W]
        rend_polar_m = rend_polar_cropped[:, :W]

        result = train_frames_io.save_per_train_frame(
            out_root=out_dir,
            frame=f,
            rendered_ra_cart=rend_cart_m,
            gt_ra_cart=gt_cart_m,
            range_res=range_res,
            baseline_name=baseline_name,
            scene=scene,
            rendered_ra_polar=rendered_polar_sin,
            gt_ra_polar_full=gt_polar_full,
            rendered_ra_polar_cropped=rend_polar_m,
            gt_ra_polar_cropped=gt_polar_m,
            extra={"test_frame": int(meta["test_frame"])},
        )
        per_frame_results.append(result)

    train_frames_io.write_train_aggregate(out_dir, per_frame_results)


if __name__ == "__main__":
    raise SystemExit(main_cli())
