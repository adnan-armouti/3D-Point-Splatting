"""Phase-B finalization for DART — runs in mm3DPS env.

Reads ``rendered_ra_polar.npy`` + ``train_meta.json``, builds the
canonical cascade GT, runs the mm3DPS polar→cart pipeline +
``compute_cart_ra_metrics``, writes ``metrics.json`` + the four RA
inspection PNGs.

DART trains and evaluates on cascade data:
  * Rendered: cascade-shaped (Na, Nr) polar from DART, crop bins
    15..110, then ``ra_polar_to_cartesian``.
  * GT: cascade ADC → ``adc_to_ra_complex`` + abs → (127, 256) sin-space
    polar → ``ra_polar_to_cartesian`` on the full polar.
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
    """GT for the cascade test frame: ADC -> adc_to_ra_complex + abs ->
    (127, 256) sin-space polar -> ra_polar_to_cartesian on the FULL polar.
    """
    split = nvs_split.cascaded_split(scene)
    adc = common_adapt.load_cascaded_adc(split["test_file"])
    ra_polar_full = common_adapt.adc_to_polar_ra(adc)
    ra_polar_cropped = common_adapt.range_crop(ra_polar_full)

    from mm3DPS.data.io_utils import compute_range_res_from_cfg
    from mm3DPS.data.ra_utils import ra_polar_to_cartesian
    range_res = compute_range_res_from_cfg(split["test_config"])
    ra_cart = ra_polar_to_cartesian(ra_polar_full, range_res).astype(np.float32)
    return (ra_polar_full.astype(np.float32),
            ra_polar_cropped.astype(np.float32),
            ra_cart,
            range_res)


def _gt_side_for_file(adc_path: str, cfg_path: str
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Same as ``_gt_side`` but for an explicit ADC + config path."""
    adc = common_adapt.load_cascaded_adc(adc_path)
    ra_polar_full = common_adapt.adc_to_polar_ra(adc)
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
    ap.add_argument("--baseline-name", default="dart")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.out_dir, "train_meta.json")))
    rendered_polar = np.load(
        os.path.join(args.out_dir, "rendered_ra_polar.npy")
    ).astype(np.float32)
    rendered_polar = np.clip(rendered_polar, 0.0, None)        # non-negative

    gt_polar_full, gt_polar_cropped, gt_cart, range_res = _gt_side(meta["scene"])

    # Rendered cascade-shaped polar is (Na, Nr=256). Crop bins 15..110
    # (drop TX-RX coupling + DFT wrap-around) and run cart conversion.
    rend_polar_cropped = common_adapt.range_crop(rendered_polar)

    from mm3DPS.data.ra_utils import ra_polar_to_cartesian
    rend_cart = ra_polar_to_cartesian(rend_polar_cropped, range_res).astype(np.float32)

    gt_cart_m, rend_cart_m = _match_shapes(gt_cart, rend_cart)
    W = min(gt_polar_cropped.shape[1], rend_polar_cropped.shape[1])
    gt_polar_m = gt_polar_cropped[:, :W]
    rend_polar_m = rend_polar_cropped[:, :W]

    extra = {
        "test_frame": int(meta["test_frame"]),
        "train_frames": list(map(int, meta["train_frames"])),
        "wall_time_seconds": float(meta["wall_time_seconds"]),
        "peak_gpu_mem_mib": float(meta.get("peak_gpu_mem_mib", 0)),
        "deviations_from_reference": meta.get("deviations_from_reference", []),
        "upstream_commit": meta.get("upstream_commit", "unknown"),
        "out_dir": meta.get("out_dir", None),
        "epochs": int(meta.get("epochs", 0)),
    }
    result = common_eval.run_eval(
        args.baseline_name, meta["scene"],
        rendered_ra_cart=rend_cart_m, gt_ra_cart=gt_cart_m,
        rendered_ra_polar_cropped=rend_polar_m,
        gt_ra_polar_cropped=gt_polar_m,
        extra=extra,
    )

    np.save(os.path.join(args.out_dir, "gt_ra_polar_full.npy"), gt_polar_full)
    np.save(os.path.join(args.out_dir, "gt_ra_polar_cropped.npy"), gt_polar_cropped)
    np.save(os.path.join(args.out_dir, "gt_ra_cart.npy"), gt_cart)
    np.save(os.path.join(args.out_dir, "rendered_ra_cart.npy"), rend_cart)
    common_eval.write_metrics_json(os.path.join(args.out_dir, "metrics.json"), result)

    from mm3DPS.evaluation.utils.visualization import save_ra_cartesian_png
    for scale in ("linear", "dB"):
        save_ra_cartesian_png(
            gt_cart_m,
            os.path.join(args.out_dir, f"gt_ra_{scale}.png"),
            range_res=range_res, scale=scale,
            title=f"GT (test frame {meta['test_frame']}, {scale})",
        )
        save_ra_cartesian_png(
            rend_cart_m,
            os.path.join(args.out_dir, f"rasterized_ra_{scale}.png"),
            range_res=range_res, scale=scale,
            title=f"{args.baseline_name} rasterized "
                  f"(test frame {meta['test_frame']}, {scale})",
        )

    # Per-train-frame artefacts (supplement figure pipeline).
    _process_train_frames(args.out_dir, args.baseline_name, meta)

    print(json.dumps(result, indent=2))
    return 0


def _process_train_frames(out_dir: str, baseline_name: str, meta: dict) -> None:
    """For each training frame: build GT cart, load runner's rendered polar
    dump, run polar→cart through the SAME pipeline as the test path, and save
    artefacts under ``train_frames/frame_<F>``.
    """
    from baselines.common import train_frames_io
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
        rendered_polar = np.clip(rendered_polar, 0.0, None)

        gt_polar_full, gt_polar_cropped, gt_cart, range_res = _gt_side_for_file(
            adc_path, cfg_path,
        )

        # Mirror the test path: range-crop the rendered polar then cart.
        rend_polar_cropped = common_adapt.range_crop(rendered_polar)
        rend_cart = ra_polar_to_cartesian(rend_polar_cropped, range_res).astype(np.float32)

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
            rendered_ra_polar=rendered_polar,
            gt_ra_polar_full=gt_polar_full,
            rendered_ra_polar_cropped=rend_polar_m,
            gt_ra_polar_cropped=gt_polar_m,
            extra={"test_frame": int(meta["test_frame"])},
        )
        per_frame_results.append(result)

    train_frames_io.write_train_aggregate(out_dir, per_frame_results)


if __name__ == "__main__":
    raise SystemExit(main_cli())
