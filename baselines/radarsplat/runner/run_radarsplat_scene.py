"""Train + render + evaluate RadarSplat for one 3DPS scene.

Pipeline:

    1. Monkey-patch ``WaveSensorDataset.__init__`` to use our 8/1 NVS split
       (train=[0,1,2,3,5,6,7,8], val=[4]) regardless of ``test_every``.
    2. Instantiate upstream's ``Runner``, train for ``--max-steps``.
    3. Load the final checkpoint; render the held-out test frame via
       ``Runner.rasterize_splats`` + ``spectral_leakage`` +
       ``azimuth_antenna_gain_projection``. The wedge-crop branch patched
       into ``radar_simple_trainer.py`` also fires here. Save the rendered
       polar as ``rendered_ra_polar.npy`` (shape ``(H_fov, W_range)``).
    4. Compute ``ra_corr`` and ``range_profile_corr`` against the GT test
       frame via ``baselines.common.eval`` (re-uses ``compute_cart_ra_metrics``
       and the canonical polar→Cartesian path from ``mm3DPS.data.ra_utils``).

Output directory: ``baselines/radarsplat/results/<scene>/``

    metrics.json            # Result schema (ra_corr, range_profile_corr, ...)
    rendered_ra_polar.npy   # (H_fov, W_range) float32 — rendered polar, pre-cart
    gt_ra_polar_cropped.npy # (127, 95)   float32 — GT sin-space polar, range-cropped 15..110
    gt_ra_cart.npy          # (399, 399)  float32 — GT Cartesian
    rendered_ra_cart.npy    # (399, 399)  float32 — rendered Cartesian
    runtime.json            # wall time, peak gpu mem
    upstream_result_dir     # path to RadarSplat's own result_dir (ckpts, tb, renders)

Runs serially inside the ``radarsplat`` conda env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = os.path.abspath(os.path.join(_HERE, "..", "upstream"))
_UPSTREAM_EX = os.path.join(_UPSTREAM, "examples")
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_UPSTREAM_EX, _UPSTREAM, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Monkey-patch WaveSensorDataset to enforce our NVS split
# ---------------------------------------------------------------------------
import radar.dataset.dataloader as _dl  # noqa: E402

_orig_ds_init = _dl.WaveSensorDataset.__init__


def _patched_ds_init(self, parser, split="train"):
    _orig_ds_init(self, parser, split=split)
    if split == "train":
        self.indices = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    elif split == "val":
        self.indices = np.array([4])
    # split == "all" keeps upstream's default (all 9 indices).


_dl.WaveSensorDataset.__init__ = _patched_ds_init


# ---------------------------------------------------------------------------
# Upstream imports (must come AFTER the monkey-patch)
# ---------------------------------------------------------------------------
from radar_simple_trainer import Config, Runner  # noqa: E402
from gsplat.rendering import (  # noqa: E402
    azimuth_antenna_gain_projection,
    spectral_leakage,
)
from gsplat.strategy import DefaultStrategy  # noqa: E402

# Our shared modules — NOTE: cannot import baselines.common.adapters here
# because mm3DPS/data/ra_utils.py uses PEP-604 ``float | None`` at class level
# which Python 3.9 (radarsplat env) cannot evaluate. Metric finalization is
# run as a subprocess under the mm3DPS env via finalize_metrics.py.
from baselines.common import nvs_split  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cfg(
    scene: str,
    data_root: str,
    result_root: str,
    max_steps: int,
    init_num_pts: int,
) -> Config:
    cfg = Config(strategy=DefaultStrategy(verbose=True))
    cfg.data_dir = data_root
    cfg.seq_name = scene
    cfg.data_factor = 1                             # read from "images/", not "images_4/"
    cfg.result_dir = result_root
    cfg.max_steps = max_steps
    cfg.init_type = "random"
    cfg.init_num_pts = init_num_pts
    cfg.init_scale = 0.5
    cfg.test_every = 999                           # monkey-patched; irrelevant
    cfg.intermediate_azimuth_resolution = 0.1
    cfg.max_range = None                           # use all 256 range bins
    cfg.multipath_weight = 0.0                     # degenerate multipath file
    cfg.l1occloss_lambda = 0.0                     # degenerate occupancy target
    cfg.use_wandb = False
    cfg.disable_viewer = True
    cfg.save_fig = False
    cfg.save_steps = [max_steps]
    cfg.eval_steps = []                             # we render test frame ourselves
    cfg.save_fig = False
    cfg.frame_selection = [0, 9]                    # load all 9 frames (Runner.exp_name needs indexable)
    cfg.synced_lidar_map_name = "synced_lidar_map_win5"
    cfg.radar_average_map_name = (
        "radar_average_map_polar/"
        "res:0.0586_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
    )
    cfg.adjust_steps(1.0)
    return cfg




@torch.no_grad()
def _render_one(runner: Runner, data) -> np.ndarray:
    """Forward-render one batch (a single frame) and return the
    wedge-cropped polar RA as float32 ``(H_fov, W)``.

    Mirrors ``Runner.eval`` (polar path) up through the sonar-style crop,
    without any of the visualization side-effects.
    """
    device = runner.device
    radarposes = data["radarpose"].to(device)
    Ks = data["K"].to(device)
    pixels = data["image"].to(device) / 255.0
    height, width = pixels.shape[1:3]
    renders, *_ = runner.rasterize_splats(
        radarposes=radarposes,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=runner.cfg.sh_degree,
        near_plane=runner.cfg.near_plane,
        far_plane=runner.cfg.far_plane,
        use_polar=runner.parser.use_polar,
    )
    out_img = renders[0]                                         # (3600, W, 1)
    if runner.cfg.spectral_leakage:
        out_img = spectral_leakage(
            out_img,
            runner.parser.range_resolution,
            sinc_width=runner.cfg.sinc_width,
        )
    out_img = azimuth_antenna_gain_projection(
        out_img,
        new_resolution=runner.parser.azimuth_resolution,
        beamwidth=runner.parser.azimuth_beamwidth,
    )                                                             # (400, W)
    out_img = out_img.squeeze()                                   # (400, W)
    H_fov = pixels.shape[1]
    out_img = out_img[:H_fov, :]                                  # wedge crop
    # Mirror upstream: clamp to [0, 1] (eval:1175) and zero out the
    # near-range region the model never optimized (train:709).
    out_img = torch.clamp(out_img, 0.0, 1.0)
    min_bin_num = int(2.5 / runner.parser.range_resolution)
    out_img[:, :min_bin_num] = 0
    return out_img.detach().cpu().numpy().astype(np.float32)


def _render_test_frame(runner: Runner) -> np.ndarray:
    """Forward-render the held-out test frame (split='val')."""
    loader = torch.utils.data.DataLoader(
        runner.valset, batch_size=1, shuffle=False, num_workers=0
    )
    for data in loader:
        return _render_one(runner, data)
    raise RuntimeError("valset produced no batches")


def _render_train_frames(runner: Runner, train_frame_numbers) -> dict:
    """Forward-render each of the 8 training frames (split='train').

    Returns a ``{frame_number: rendered_polar}`` dict. The trainset's
    iteration order is fixed by the patched ``self.indices = [0,1,2,3,5,6,7,8]``
    so we pair each yielded sample with the corresponding entry in
    ``train_frame_numbers`` (monotonically sorted scene-internal frame numbers
    matching that index order via ``baselines.common.nvs_split.cascaded_split``).
    """
    loader = torch.utils.data.DataLoader(
        runner.trainset, batch_size=1, shuffle=False, num_workers=0
    )
    out = {}
    for i, data in enumerate(loader):
        rendered = _render_one(runner, data)
        f = int(train_frame_numbers[i])
        out[f] = rendered
    if len(out) != len(train_frame_numbers):
        raise RuntimeError(
            f"trainset yielded {len(out)} samples; expected "
            f"{len(train_frame_numbers)}"
        )
    return out


# (GT-side + metric computation live in finalize_metrics.py which runs in
# the mm3DPS env — see top-of-file comment.)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    scene: str,
    data_root: str,
    result_root: str,
    out_dir: str,
    max_steps: int,
    init_num_pts: int,
    baseline_name: str = "radarsplat",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    split = nvs_split.cascaded_split(scene)

    cfg = _build_cfg(
        scene=scene,
        data_root=data_root,
        result_root=result_root,
        max_steps=max_steps,
        init_num_pts=init_num_pts,
    )

    # Upstream's create_splats_with_optimizers references a module-level ``cfg``
    # which is only set when radar_simple_trainer.py runs as __main__. Inject it.
    import radar_simple_trainer as _rst
    _rst.cfg = cfg

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    runner.train()
    train_wall = time.time() - t_start

    # Render held-out frame
    rendered_polar = _render_test_frame(runner)
    peak_mem_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    np.save(os.path.join(out_dir, "rendered_ra_polar.npy"), rendered_polar)

    # Per-train-frame renders (supplement figure pipeline).
    # The trainset uses indices [0,1,2,3,5,6,7,8] (patched at top of file),
    # which correspond — in sorted scene order — to the 8 training frames
    # listed in ``split["train_frames"]``.
    train_frames_polar = _render_train_frames(runner, split["train_frames"])
    train_frames_dump_dir = os.path.join(out_dir, "train_frames_polar_raw")
    os.makedirs(train_frames_dump_dir, exist_ok=True)
    for f, polar in train_frames_polar.items():
        np.save(os.path.join(train_frames_dump_dir,
                             f"rendered_ra_polar_frame_{int(f)}.npy"),
                polar.astype(np.float32))

    manifest_path = os.path.join(data_root, scene, "adapter_manifest.json")
    deviations = []
    if os.path.isfile(manifest_path):
        try:
            deviations = json.load(open(manifest_path)).get(
                "deviations_from_reference", []
            )
        except Exception:
            pass
    deviations = list(deviations) + [
        f"Training iterations: {max_steps} (upstream default 2000).",
        f"init_num_pts: {init_num_pts} (upstream default 20000).",
        "multipath_weight=0, l1occloss_lambda=0 (components shipped degenerate; see adapter).",
    ]

    meta = {
        "scene": scene,
        "test_frame": int(split["test_frame"]),
        "train_frames": [int(x) for x in split["train_frames"]],
        "wall_time_seconds": train_wall,
        "peak_gpu_mem_mib": float(peak_mem_mib),
        "deviations_from_reference": deviations,
        "upstream_commit": "ea9c8f530c708622cc3b1b560436b5557ac6a49b",
        "upstream_result_dir": cfg.result_dir,
        "max_steps": max_steps,
        "init_num_pts": init_num_pts,
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Spawn metric finalization under mm3DPS env.
    import subprocess
    py = os.environ.get("MM3DPS_PYTHON", "python")
    finalize = os.path.join(_HERE, "finalize_metrics.py")
    proc = subprocess.run(
        [py, finalize, "--out-dir", out_dir, "--baseline-name", baseline_name],
        cwd=_REPO,
        check=True,
    )
    with open(os.path.join(out_dir, "metrics.json")) as f:
        result = json.load(f)
    return result


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data-root",
                    default=os.path.join(_REPO, "baselines/radarsplat/data_radarsplat"))
    ap.add_argument("--result-root",
                    default=os.path.join(_REPO, "baselines/radarsplat/upstream_results"))
    ap.add_argument("--out-dir",
                    default=None,
                    help="Where to write metrics.json + rendered .npy. "
                         "Default: baselines/radarsplat/results/<scene>/")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--init-num-pts", type=int, default=20000)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        _REPO, "baselines/radarsplat/results", args.scene
    )
    result = run(
        scene=args.scene,
        data_root=args.data_root,
        result_root=args.result_root,
        out_dir=out_dir,
        max_steps=args.max_steps,
        init_num_pts=args.init_num_pts,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
