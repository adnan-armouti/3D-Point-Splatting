"""Metric step of a baseline run, under the mm3DPS environment.

Each baseline's training script dumps ``rendered_ra_polar.npy`` and
``train_meta.json`` under its own environment, then calls::

    python -m baselines.common.finalize --baseline dart --out-dir <run dir>

This builds the ground truth, sends both sides through the 3DPS polar-to-Cartesian
conversion, and writes ``metrics.json``, the array dumps, the four RA inspection
PNGs and the per-train-frame artefacts, so that all methods get the same metrics.

The ground-truth Cartesian image comes from the full (127, 256) sin-space polar
image, which covers the radar's full range. The cropped polar image (bins
15..110) is kept for the range-profile correlation only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.common import adapters as common_adapt  # noqa: E402
from baselines.common import eval as common_eval  # noqa: E402
from baselines.common import nvs_split, train_frames_io  # noqa: E402
from mm3DPS.data.io_utils import compute_range_res_from_cfg  # noqa: E402
from mm3DPS.data.ra_utils import ra_polar_to_cartesian  # noqa: E402
from mm3DPS.evaluation.utils.visualization import save_ra_cartesian_png  # noqa: E402


def gt_side(adc_path: str, cfg_path: str
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (gt_polar_full, gt_polar_cropped, gt_cart, range_res)."""
    adc = common_adapt.load_cascaded_adc(adc_path)
    ra_polar_full = common_adapt.adc_to_polar_ra(adc)
    ra_polar_cropped = common_adapt.range_crop(ra_polar_full)
    range_res = compute_range_res_from_cfg(cfg_path)
    ra_cart = ra_polar_to_cartesian(ra_polar_full, range_res).astype(np.float32)
    return (ra_polar_full.astype(np.float32),
            ra_polar_cropped.astype(np.float32),
            ra_cart,
            range_res)


def match_shapes(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop two images to their common shape."""
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])

    def crop(x):
        dh, dw = (x.shape[0] - H) // 2, (x.shape[1] - W) // 2
        return x[dh:dh + H, dw:dw + W]
    return crop(a), crop(b)


def match_range(gt_polar_cropped: np.ndarray, rend_polar_cropped: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Match range axes. Azimuth counts may differ, since azimuth is summed out."""
    W = min(gt_polar_cropped.shape[1], rend_polar_cropped.shape[1])
    return gt_polar_cropped[:, :W], rend_polar_cropped[:, :W]


class BaselineFinalizer:
    """Shared metric step. Subclasses define how a rendered polar image is prepared."""

    name = ""
    extra_meta_keys: dict = {}

    def __init__(self, out_dir: str, baseline_name: str = None):
        self.out_dir = out_dir
        self.baseline_name = baseline_name or self.name
        with open(os.path.join(out_dir, "train_meta.json")) as f:
            self.meta = json.load(f)

    # Per-baseline parts.

    def load_test_render(self) -> np.ndarray:
        return np.load(os.path.join(self.out_dir, "rendered_ra_polar.npy")).astype(np.float32)

    def clip_train_render(self, polar: np.ndarray) -> np.ndarray:
        return np.clip(polar, 0.0, 1.0)

    def cart_input(self, polar: np.ndarray) -> np.ndarray:
        """The polar image that goes into the Cartesian conversion."""
        raise NotImplementedError

    def train_frame_polar(self, polar: np.ndarray, cart_input: np.ndarray) -> np.ndarray:
        """The polar image saved for a training frame."""
        return cart_input

    def save_test_extras(self, cart_input: np.ndarray) -> None:
        """Additional arrays to save for the test frame."""

    # Shared steps.

    def _save(self, name: str, array: np.ndarray) -> None:
        np.save(os.path.join(self.out_dir, name), array)

    def _extra(self) -> dict:
        meta = self.meta
        extra = {
            "test_frame": int(meta["test_frame"]),
            "train_frames": [int(x) for x in meta["train_frames"]],
            "wall_time_seconds": float(meta["wall_time_seconds"]),
            "peak_gpu_mem_mib": float(meta.get("peak_gpu_mem_mib", 0)),
            "deviations_from_reference": meta.get("deviations_from_reference", []),
            "upstream_commit": meta.get("upstream_commit", "unknown"),
        }
        for key, default in self.extra_meta_keys.items():
            value = meta.get(key, default)
            extra[key] = int(value) if isinstance(default, int) else value
        return extra

    def finalize_test(self) -> dict:
        meta = self.meta
        split = nvs_split.cascaded_split(meta["scene"])
        gt_polar_full, gt_polar_cropped, gt_cart, range_res = gt_side(
            split["test_file"], split["test_config"])

        rendered_polar = self.load_test_render()
        cart_input = self.cart_input(rendered_polar)
        rend_cart = ra_polar_to_cartesian(cart_input, range_res).astype(np.float32)
        rend_polar_cropped = common_adapt.range_crop(rendered_polar)

        gt_cart_m, rend_cart_m = match_shapes(gt_cart, rend_cart)
        gt_polar_m, rend_polar_m = match_range(gt_polar_cropped, rend_polar_cropped)

        result = common_eval.run_eval(
            self.baseline_name, meta["scene"],
            rendered_ra_cart=rend_cart_m, gt_ra_cart=gt_cart_m,
            rendered_ra_polar_cropped=rend_polar_m,
            gt_ra_polar_cropped=gt_polar_m,
            extra=self._extra(),
        )

        self._save("gt_ra_polar_full.npy", gt_polar_full)
        self._save("gt_ra_polar_cropped.npy", gt_polar_cropped)
        self._save("gt_ra_cart.npy", gt_cart)
        self.save_test_extras(cart_input)
        self._save("rendered_ra_cart.npy", rend_cart)
        common_eval.write_metrics_json(os.path.join(self.out_dir, "metrics.json"), result)

        for scale in ("linear", "dB"):
            save_ra_cartesian_png(
                gt_cart_m,
                os.path.join(self.out_dir, f"gt_ra_{scale}.png"),
                range_res=range_res, scale=scale,
                title=f"GT (test frame {meta['test_frame']}, {scale})",
            )
            save_ra_cartesian_png(
                rend_cart_m,
                os.path.join(self.out_dir, f"rasterized_ra_{scale}.png"),
                range_res=range_res, scale=scale,
                title=f"{self.baseline_name} rasterized "
                      f"(test frame {meta['test_frame']}, {scale})",
            )
        return result

    def finalize_train_frames(self) -> None:
        """Same conversion as the test frame, for each training frame."""
        meta = self.meta
        scene = meta["scene"]
        split = nvs_split.cascaded_split(scene)

        raw_dir = os.path.join(self.out_dir, "train_frames_polar_raw")
        if not os.path.isdir(raw_dir):
            print(f"[finalize] no train_frames_polar_raw/ in {self.out_dir}; skipping")
            return

        per_frame_results = []
        for f, adc_path, cfg_path in zip(map(int, split["train_frames"]),
                                         split["train_files"], split["train_configs"]):
            rend_path = os.path.join(raw_dir, f"rendered_ra_polar_frame_{f}.npy")
            if not os.path.isfile(rend_path):
                print(f"[finalize] missing {rend_path}; skipping frame {f}")
                continue

            rendered_polar = self.clip_train_render(np.load(rend_path).astype(np.float32))
            gt_polar_full, gt_polar_cropped, gt_cart, range_res = gt_side(adc_path, cfg_path)

            cart_input = self.cart_input(rendered_polar)
            rend_cart = ra_polar_to_cartesian(cart_input, range_res).astype(np.float32)
            rend_polar_cropped = common_adapt.range_crop(rendered_polar)

            gt_cart_m, rend_cart_m = match_shapes(gt_cart, rend_cart)
            gt_polar_m, rend_polar_m = match_range(gt_polar_cropped, rend_polar_cropped)

            per_frame_results.append(train_frames_io.save_per_train_frame(
                out_root=self.out_dir,
                frame=f,
                rendered_ra_cart=rend_cart_m,
                gt_ra_cart=gt_cart_m,
                range_res=range_res,
                baseline_name=self.baseline_name,
                scene=scene,
                rendered_ra_polar=self.train_frame_polar(rendered_polar, cart_input),
                gt_ra_polar_full=gt_polar_full,
                rendered_ra_polar_cropped=rend_polar_m,
                gt_ra_polar_cropped=gt_polar_m,
                extra={"test_frame": int(meta["test_frame"])},
            ))

        train_frames_io.write_train_aggregate(self.out_dir, per_frame_results)

    def run(self) -> int:
        result = self.finalize_test()
        self.finalize_train_frames()
        print(json.dumps(result, indent=2))
        return 0


class SinResampledFinalizer(BaselineFinalizer):
    """Baselines that render on a uniform-angle grid (200 bins over +-90 deg).

    ``ra_polar_to_cartesian`` assumes sin-space, so the render is resampled onto
    the 127-bin sin-space grid, the same azimuth space as the ground truth.
    """

    def cart_input(self, polar):
        from baselines.radarsplat.adapter.mm3dps_to_radarsplat import (
            resample_polar_angle_to_sin,
        )
        return resample_polar_angle_to_sin(polar, H_sin=127)

    def save_test_extras(self, cart_input):
        self._save("rendered_ra_polar_sin.npy", cart_input)


class DartFinalizer(BaselineFinalizer):
    """DART renders in the cascade's own grid: crop bins 15..110, then convert."""

    name = "dart"
    extra_meta_keys = {"out_dir": None, "epochs": 0}

    def load_test_render(self):
        return np.clip(super().load_test_render(), 0.0, None)

    def clip_train_render(self, polar):
        return np.clip(polar, 0.0, None)

    def cart_input(self, polar):
        return common_adapt.range_crop(polar)

    def train_frame_polar(self, polar, cart_input):
        return polar


class RadarFieldsFinalizer(SinResampledFinalizer):
    name = "radarfields"
    extra_meta_keys = {"workspace": None, "max_iters": 0}

    def load_test_render(self):
        # [0, 1] is the training-target space.
        return np.clip(super().load_test_render(), 0.0, 1.0)


class RadarSplatFinalizer(SinResampledFinalizer):
    name = "radarsplat"
    extra_meta_keys = {"upstream_result_dir": None, "max_steps": 0, "init_num_pts": 0}

    def load_test_render(self):
        return np.load(os.path.join(self.out_dir, "rendered_ra_polar.npy"))


FINALIZERS = {cls.name: cls for cls in (DartFinalizer, RadarFieldsFinalizer, RadarSplatFinalizer)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True, choices=sorted(FINALIZERS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline-name", default=None)
    args = ap.parse_args()
    return FINALIZERS[args.baseline](args.out_dir, args.baseline_name).run()


if __name__ == "__main__":
    raise SystemExit(main())
