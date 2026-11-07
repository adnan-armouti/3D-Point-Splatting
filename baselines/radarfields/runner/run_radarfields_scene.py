"""Train + render RadarFields for one 3DPS scene.

Drives upstream's ``main.py`` (train + auto-test) with our
``preprocess_file`` and intrinsics overrides, then captures the rendered
``pred_fft`` for the held-out test frame via a small ``save_figures``
monkey-patch. ``baselines.common.finalize`` (mm3DPS env) computes the 3DPS
metrics from the saved polar.
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
# Path setup — must come BEFORE upstream imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = os.path.abspath(os.path.join(_HERE, "..", "upstream"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_UPSTREAM, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Monkey-patch test_epoch BEFORE importing upstream main/train.
# We hook in the rendered pred_fft dump.
# ---------------------------------------------------------------------------
import radarfields.train as _rf_train  # noqa: E402
from radarfields.sampler import get_points  # noqa: E402

_orig_test_epoch = _rf_train.Trainer.test_epoch


def _patched_test_epoch(self, loader):
    """Custom test loop that ALSO saves pred_fft.npy for the test frame.

    Equivalent to upstream's logic for visualization but additionally
    dumps the raw rendered FFT polar (uniform-angle) so baselines.common.finalize
    can run the canonical mm3DPS polar→cart pipeline against it.

    When ``RF_TRAIN_DUMP_DIR`` is set (instead of ``RF_DUMP_DIR``), this
    loop ASSUMES the loader iterates over training frames at bs=1, and
    dumps each one as ``rendered_ra_polar_frame_<F>.npy`` keyed by the
    train-frame number list in ``RF_TRAIN_FRAMES`` (comma-separated, in
    the iteration order of the loader).
    """
    self.log(f"++> Test at epoch {self.epoch} (with pred_fft dump) ...")

    self.model.eval()
    if self.refine_poses:
        self.pose_model.eval()

    dump_dir = os.environ.get("RF_DUMP_DIR")
    train_dump_dir = os.environ.get("RF_TRAIN_DUMP_DIR")
    train_frame_list = os.environ.get("RF_TRAIN_FRAMES", "")
    train_frame_numbers = [int(x) for x in train_frame_list.split(",")
                           if x.strip()]

    saved = False
    train_idx = 0
    with torch.no_grad():
        self.local_step = 0
        for data in loader:
            self.local_step += 1
            if self.refine_poses:
                data["poses"] = self.pose_model.apply_to_poses(data["indices"])

            points = get_points(data, self.args, self.device)
            pred_fft, alpha_integrated, rd_integrated, alpha = self.predict_waveform(data, points)

            # pred_fft shape: [B, N_az, R] in upstream's convention.
            B = pred_fft.shape[0]

            if train_dump_dir is not None and train_frame_numbers:
                # Per-train-frame dump path.
                for b in range(B):
                    if train_idx >= len(train_frame_numbers):
                        break
                    f = train_frame_numbers[train_idx]
                    p = pred_fft[b].detach().cpu().numpy().astype(np.float32)
                    np.save(
                        os.path.join(train_dump_dir,
                                     f"rendered_ra_polar_frame_{int(f)}.npy"),
                        p,
                    )
                    train_idx += 1
                continue

            # Test (held-out) frame dump path.
            if not saved and dump_dir is not None and B >= 1:
                p = pred_fft[0].detach().cpu().numpy().astype(np.float32)  # (N_az, R)
                np.save(os.path.join(dump_dir, "rendered_ra_polar.npy"), p)
                saved = True
                self.log(f"  saved rendered_ra_polar.npy shape={p.shape} to {dump_dir}")

            # Skip upstream's heavy save_figures (PNG matplotlib renders we
            # don't need; baselines.common.finalize produces the canonical PNGs).
    self.log("++> Test done.")


_rf_train.Trainer.test_epoch = _patched_test_epoch


# ---------------------------------------------------------------------------
# Upstream entry points
# ---------------------------------------------------------------------------
import torch.nn as _nn  # noqa: E402
from radarfields.dataset import RadarDataset  # noqa: E402
from radarfields.nn.models import RadarField  # noqa: E402
from utils.data import filter_dict_for_dataclass  # noqa: E402
from utils.train import seed_everything  # noqa: E402
from radarfields.train import Trainer  # noqa: E402

from baselines.common import nvs_split  # noqa: E402


def _build_args(scene: str, workspace: str, max_iters: int) -> "argparse.Namespace":
    """Build the upstream Args namespace programmatically (no CLI)."""
    from parse import get_arg_parser
    parser = get_arg_parser()
    # Defaults from configs/radarfields.ini, plus our cascade-specific overrides.
    args = parser.parse_args([])
    args.name = f"radarfields_{scene}"
    args.seq = scene
    args.workspace = workspace
    args.preprocess_file = f"{scene}.json"
    args.iters = int(max_iters)
    args.bs = 8                              # 8 train frames
    args.lr = 1e-3
    args.tcnn = True
    args.train_thresholded = True
    args.reg_occ = True
    args.bimodal = True
    args.ground_occ = True
    args.penalize_above = True
    args.integrate_rays = True
    args.approximate_fft = True
    args.mask = True
    args.refine_poses = False                # disable per 
    args.render_fft = False                  # we save pred_fft via the monkey-patch
    args.save_loss_plot = False
    args.skip_test = False                   # auto-test at end of training
    # Cascade intrinsics (override 360°-spinning Boreas defaults).
    range_res = _range_res_for(scene)
    args.intrinsics_radar = {
        "opening_h": 1.8,
        "opening_v": 40.0,
        "num_azimuths_radar": 200,
        "num_range_bins": 256,
        "bin_size_radar": float(range_res),
        "azim_span_deg": 180.0,              # patched into sampler.py:32
    }
    # Range-bin sampling: 1-indexed inclusive in upstream. We sample bins 1..256
    # with an inner cropping; the model's HashGrid lives in normalized cube.
    args.min_range_bin = 1
    args.max_range_bin = 256
    args.num_range_samples = 256
    args.sample_all_ranges = True
    args.num_rays_radar = 100
    args.num_fov_samples = 10
    return args


def _range_res_for(scene: str) -> float:
    from mm3DPS.data.io_utils import compute_range_res_from_cfg
    split = nvs_split.cascaded_split(scene)
    return compute_range_res_from_cfg(split["test_config"])


def _train_and_test(args) -> None:
    """Mirror of upstream main.py's train()+test() but with our patched
    test_epoch hook (which dumps rendered_ra_polar.npy when training auto-tests
    via main.py:70)."""
    seed_everything(args.seed)
    args.data_path = os.path.join(_UPSTREAM, "data", args.seq)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RadarField(**args.model_settings, use_tcnn=args.tcnn)
    criterion = {
        "fft": _nn.L1Loss() if args.fft_loss == "l1" else _nn.MSELoss(),
        "occ": _nn.KLDivLoss(reduction="batchmean") if args.occ_loss == "kl" else _nn.MSELoss(),
    }

    optimizer = lambda m: torch.optim.Adam(m.get_params(args.lr), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda opt: torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.1 ** min(i / args.iters, 1)
    )

    # azim_span_deg is consumed by sampler.get_points via args; strip it from
    # the dict that gets splatted into RadarDataset's dataclass fields.
    ds_intrinsics = {k: v for k, v in args.intrinsics_radar.items()
                     if k != "azim_span_deg"}
    train_loader = RadarDataset(
        split="train",
        **filter_dict_for_dataclass(RadarDataset, vars(args)),
        **ds_intrinsics,
    ).dataloader(args.bs)
    args.all_poses = train_loader._data.poses_radar.to(args.device)
    args.test_indices = train_loader._data.preprocess["test_indices"]

    trainer = Trainer(args, model, split="train", criterion=criterion,
                      optimizer=optimizer, lr_scheduler=scheduler, device=args.device)
    max_epoch = int(np.ceil(args.iters / max(len(train_loader), 1)))
    print(f"max_epoch: {max_epoch}")
    trainer.train(train_loader, max_epoch)

    del train_loader, trainer
    torch.cuda.empty_cache()

    # Test (auto-dumps pred_fft via patched test_epoch).
    test_loader = RadarDataset(
        split="test",
        **filter_dict_for_dataclass(RadarDataset, vars(args)),
        **ds_intrinsics,
    ).dataloader(args.bs)
    args.all_poses = test_loader._data.poses_radar.to(args.device)
    args.test_indices = test_loader._data.preprocess["test_indices"]

    # Temporarily unset the train-dump env vars so the patched test_epoch
    # routes the held-out test frame through the test-dump path (writes
    # rendered_ra_polar.npy at <out_dir>). Otherwise the train-dump branch
    # in the patched test_epoch fires unconditionally and never reaches
    # the test-dump path. Restored after this test pass for the
    # per-train-frame loop below.
    _prev_train_dump = os.environ.pop("RF_TRAIN_DUMP_DIR", None)
    _prev_train_frames = os.environ.pop("RF_TRAIN_FRAMES", None)

    test_trainer = Trainer(args, model, split="test", criterion=criterion,
                           optimizer=None, lr_scheduler=None, device=args.device)
    test_trainer.test(test_loader)

    if _prev_train_dump is not None:
        os.environ["RF_TRAIN_DUMP_DIR"] = _prev_train_dump
    if _prev_train_frames is not None:
        os.environ["RF_TRAIN_FRAMES"] = _prev_train_frames

    # ------------------------------------------------------------------
    # Per-train-frame render dump (supplement figure pipeline).
    # We re-instantiate a TRAIN dataloader (no shuffle, bs=1) and run the
    # patched test_epoch over it with RF_TRAIN_DUMP_DIR set, so each yielded
    # sample writes a per-frame polar npy file.
    # ------------------------------------------------------------------
    train_dump_dir = os.environ.get("RF_TRAIN_DUMP_DIR")
    if train_dump_dir is not None:
        train_render_loader = RadarDataset(
            split="train",
            **filter_dict_for_dataclass(RadarDataset, vars(args)),
            **ds_intrinsics,
        ).dataloader(1)
        # Important: poses for the new loader must point to the TRAIN poses,
        # not the test ones still on args from the test pass.
        args.all_poses = train_render_loader._data.poses_radar.to(args.device)
        # Build a fresh trainer wrapping the same trained model. Use split=
        # "test" so internal flags select the eval path inside test_epoch.
        train_render_trainer = Trainer(
            args, model, split="test", criterion=criterion,
            optimizer=None, lr_scheduler=None, device=args.device,
        )
        train_render_trainer.test(train_render_loader)


def run(scene: str, out_dir: str, workspace: str, max_iters: int) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(workspace, exist_ok=True)
    os.environ["RF_DUMP_DIR"] = out_dir

    # Per-train-frame dump (supplement figure pipeline).
    train_dump_dir = os.path.join(out_dir, "train_frames_polar_raw")
    os.makedirs(train_dump_dir, exist_ok=True)
    os.environ["RF_TRAIN_DUMP_DIR"] = train_dump_dir
    split = nvs_split.cascaded_split(scene)
    os.environ["RF_TRAIN_FRAMES"] = ",".join(
        str(int(f)) for f in split["train_frames"]
    )

    args = _build_args(scene, workspace, max_iters)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    _train_and_test(args)
    train_wall = time.time() - t_start
    peak_mem_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    split = nvs_split.cascaded_split(scene)
    deviations = []
    manifest_path = os.path.join(_UPSTREAM, "data", scene, "adapter_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            deviations = json.load(open(manifest_path)).get(
                "deviations_from_reference", []
            )
        except Exception:
            pass
    deviations = list(deviations) + [
        f"Training iterations: {max_iters} (upstream default 800).",
        "--refine_poses disabled (small-n training set).",
    ]

    meta = {
        "scene": scene,
        "test_frame": int(split["test_frame"]),
        "train_frames": [int(x) for x in split["train_frames"]],
        "wall_time_seconds": train_wall,
        "peak_gpu_mem_mib": float(peak_mem_mib),
        "deviations_from_reference": deviations,
        "upstream_commit": "ee76d76570f58b3d8539eafd7df0c188b58af333",
        "workspace": workspace,
        "max_iters": max_iters,
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Spawn metric finalization under mm3DPS env.
    import subprocess
    py = os.environ.get("MM3DPS_PYTHON", "python")
    subprocess.run(
        [py, "-m", "baselines.common.finalize", "--baseline", "radarfields",
         "--out-dir", out_dir, "--baseline-name", "radarfields"],
        cwd=_REPO,
        check=True,
    )
    with open(os.path.join(out_dir, "metrics.json")) as f:
        return json.load(f)


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-dir",
                    default=None,
                    help="default: baselines/radarfields/results/<scene>/")
    ap.add_argument("--workspace",
                    default=os.path.join(_REPO, "baselines/radarfields/upstream_results"))
    ap.add_argument("--max-iters", type=int, default=400,
                    help="default 400 — default is 800, but on our "
                         "8-frame cascade adapter the HashGrid catastrophically NaN-explodes "
                         "between iter 700 and 800; 400 is the closest stable count below that.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        _REPO, "baselines/radarfields/results", args.scene
    )
    workspace = os.path.join(args.workspace, args.scene)
    result = run(args.scene, out_dir, workspace, args.max_iters)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
