"""Re-render rendered_test_rp_complex.npy from a saved best_model.pt.

Older training runs were saved before the trainer started exporting the
complex range profile (CRP) for the held-out test frame. This script
loads the saved model state, reconstructs the rasterizer + dataset for
that scene, renders the test pose, and writes
``rendered_test_rp_complex.npy`` (shape (12, 16, 256), complex64) into
the same run directory — without touching any other file.

Usage:
    python -m mm3DPS.rerender_test_rp --scene seq_0_frame_135 \\
        --test_frame 135 --train_frames 131,132,133,134,136,137,138,139
"""

import os
import sys
import argparse
import numpy as np
import torch

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm3DPS.rasterizer import Rasterizer
from mm3DPS.train_gaussian import (
    DEVICE, PointPrimitives, cull_gaussians, render_gaussians,
)
from mm3DPS.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, default_split
from mm3DPS.pose_utils import apply_pose
from mm3DPS.train_frame_nvs import _build_frame_poses
from mm3DPS.config import load_trained_config


def parse_train_frames(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def rerender_one(scene: str, test_frame: int, train_frames,
                  held_out_loop: int = 0, target_n: int = 20000,
                  use_pass2_alignment: bool = True,
                  data_root: str = DEFAULT_DATA_ROOT,
                  verbose: bool = True,
                  also_train_frames: bool = False) -> str:
    """Render the test frame's CRP from best_model.pt and save it.

    If ``also_train_frames`` is True, also renders each train frame at loop 0
    and writes ``train_frames/frame_<F>/rendered_rp_complex.npy``.

    Returns the path to the written test-frame .npy file.
    """
    suffix = "_aligned_pass2" if use_pass2_alignment else "_aligned"
    align_dir = os.path.join(data_root, "alignment_data", scene, "cascade")
    seed_frame = int(test_frame)
    seed_cfg = os.path.join(align_dir,
                             f"cascaded_frame_{seed_frame}{suffix}.json")
    assert os.path.isfile(seed_cfg), f"missing alignment config: {seed_cfg}"

    # Output dir name matches the trainer's convention
    run_tag = (f"{scene}_train{len(train_frames)}frames_1loops"
                f"_test{test_frame}_loop{held_out_loop}")
    if use_pass2_alignment:
        run_tag = f"{run_tag}_pass2"
    if int(target_n) != 90000:
        run_tag = f"{run_tag}_N{int(target_n)}"
    output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, run_tag)
    assert os.path.isdir(output_dir), f"run dir not found: {output_dir}"
    best_model_path = os.path.join(output_dir, "best_model.pt")
    assert os.path.isfile(best_model_path), (
        f"best_model.pt not found in {output_dir}")

    if verbose:
        print(f"[{scene}] re-rendering test frame {test_frame} from "
              f"{best_model_path}")

    # ── Rasterizer ──
    config = load_trained_config(scene, data_root=data_root)
    rast = Rasterizer(
        config_file=seed_cfg,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE)
    rast.free_mi_scene()

    # ── Model ──
    state = torch.load(best_model_path, map_location=DEVICE,
                        weights_only=False)
    N = int(state["positions"].shape[0])
    model = PointPrimitives(N=N, device=DEVICE)
    with torch.no_grad():
        model.positions.copy_(state["positions"].to(DEVICE))
        model.rotations.copy_(state["rotations"].to(DEVICE))
        model.raw_materials.copy_(state["raw_materials"].to(DEVICE))
    if verbose:
        print(f"  loaded {N} points from best_model.pt")

    # ── Active mask + vertex areas ──
    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0

    # ── Test pose ──
    test_poses, test_mode = _build_frame_poses(
        scene, test_frame, use_pass2=use_pass2_alignment, data_root=data_root,
        loop_dt_s=7.87e-3 / 16.0, frame_period_s=0.1,
        device=DEVICE)
    test_pose = test_poses[held_out_loop]

    # ── Render ──
    apply_pose(rast, test_pose)
    with torch.no_grad():
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None)
    rp_complex = (rp_real + 1j * rp_imag).detach().cpu().numpy().astype(
        np.complex64)
    if verbose:
        print(f"  rendered CRP: shape={rp_complex.shape}  "
              f"|max|={float(np.max(np.abs(rp_complex))):.2e}")

    # ── Save ──
    out_path = os.path.join(output_dir, "rendered_test_rp_complex.npy")
    np.save(out_path, rp_complex)
    if verbose:
        print(f"  saved → {out_path}")

    # ── Optional: also render each train frame at loop 0 ──
    if also_train_frames:
        train_root = os.path.join(output_dir, "train_frames")
        os.makedirs(train_root, exist_ok=True)
        for f in train_frames:
            frame_dir = os.path.join(train_root, f"frame_{f}")
            os.makedirs(frame_dir, exist_ok=True)
            poses, _ = _build_frame_poses(
                scene, f, use_pass2=use_pass2_alignment, data_root=data_root,
                loop_dt_s=7.87e-3 / 16.0, frame_period_s=0.1,
                device=DEVICE)
            apply_pose(rast, poses[0])  # train_loops=[0] for the frame-NVS recipe
            with torch.no_grad():
                rp_r, rp_i = render_gaussians(
                    model, rast, vertex_areas=vertex_areas,
                    active_mask=active_mask, shadow_mask=None)
            rp_c = (rp_r + 1j * rp_i).detach().cpu().numpy().astype(
                np.complex64)
            train_out = os.path.join(frame_dir, "rendered_rp_complex.npy")
            np.save(train_out, rp_c)
            if verbose:
                print(f"  train frame {f}: |max|="
                      f"{float(np.max(np.abs(rp_c))):.2e}  → {train_out}")
    return out_path


# Paper-scene defaults (used by --scenes-all)
PAPER_SCENES = [
    ("seq_0_frame_135", 135, [131, 132, 133, 134, 136, 137, 138, 139]),
    ("seq_1_frame_185", 185, [181, 182, 183, 184, 186, 187, 188, 189]),
    ("seq_1_frame_438", 438, [434, 435, 436, 437, 439, 440, 441, 442]),
    ("seq_2_frame_105", 105, [101, 102, 103, 104, 106, 107, 108, 109]),
    ("seq_2_frame_160", 160, [156, 157, 158, 159, 161, 162, 163, 164]),
    ("seq_2_frame_300", 300, [296, 297, 298, 299, 301, 302, 303, 304]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str)
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--test_frame", type=int)
    ap.add_argument("--train_frames", type=str)
    ap.add_argument("--held_out_loop", type=int, default=0)
    ap.add_argument("--target_n", type=int, default=20000)
    ap.add_argument("--scenes-all", action="store_true",
                    help="Re-render all 6 paper scenes that are missing the file.")
    ap.add_argument("--scenes-missing", action="store_true",
                    help="Same as --scenes-all but skip scenes that already have the .npy.")
    ap.add_argument("--also-train-frames", action="store_true",
                    help="Also render each train frame at loop 0 to "
                          "train_frames/frame_<F>/rendered_rp_complex.npy.")
    args = ap.parse_args()

    if args.scenes_all or args.scenes_missing:
        for scene, F, train_frames in PAPER_SCENES:
            run_dir = (f"{scene}_train{len(train_frames)}frames_1loops"
                        f"_test{F}_loop0_pass2_N{int(args.target_n)}")
            out_dir_full = os.path.join(DEFAULT_OUTPUT_ROOT, run_dir)
            test_out = os.path.join(out_dir_full,
                                      "rendered_test_rp_complex.npy")
            if args.scenes_missing:
                # Skip only if BOTH test and (if requested) all train CRPs
                # are already on disk.
                test_ok = os.path.isfile(test_out)
                train_ok = True
                if args.also_train_frames:
                    train_ok = all(
                        os.path.isfile(os.path.join(
                            out_dir_full, "train_frames", f"frame_{f}",
                            "rendered_rp_complex.npy"))
                        for f in train_frames)
                if test_ok and train_ok:
                    print(f"[{scene}] already has CRPs (test+train) → skip")
                    continue
            try:
                rerender_one(scene, F, train_frames,
                              held_out_loop=args.held_out_loop,
                              target_n=args.target_n,
                              data_root=args.data_root,
                              also_train_frames=args.also_train_frames)
            except Exception as e:
                print(f"[{scene}] FAILED: {e}")
                import traceback; traceback.print_exc()
        return

    assert args.scene, "supply --scene or use --scenes-all / --scenes-missing"
    _test, _train = default_split(args.scene)
    args.test_frame = _test if args.test_frame is None else args.test_frame
    train_frames = (_train if args.train_frames is None
                    else parse_train_frames(args.train_frames))
    rerender_one(args.scene, args.test_frame, train_frames,
                  held_out_loop=args.held_out_loop,
                  target_n=args.target_n,
                  data_root=args.data_root,
                  also_train_frames=args.also_train_frames)


if __name__ == "__main__":
    main()
