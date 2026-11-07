#!/usr/bin/env python3
"""
Cascaded Radar Alignment Orchestrator
======================================

End-to-end pipeline for aligning the 77 GHz cascaded FMCW radar to the
LiDAR-derived scene mesh.  Two complementary alignment methods are run
and the per-scene winner (highest RA-image correlation) is selected:

1. **LiDAR-based 4-DOF** (``cascaded_lidar`` / ``cascaded_lidar_gpu``):
   Voxelises the LiDAR point cloud into the radar's RAE grid and
   optimises range, azimuth, elevation-rotation, and azimuth-rotation
   to maximise correlation with the measured RA image.

2. **Renderer-based 2-DOF** (``cascaded_renderer``):
   Uses the differentiable Monte-Carlo renderer with 1-bounce to render
   RA images and optimises only range offset + azimuth offset (zero
   boresight rotation, preserving the IMU-derived tilt).

After both methods have been run (or loaded from prior results), the
orchestrator picks the winner per scene and writes the aligned config.

Usage (complete pipeline)::

    from mm3DPS.preprocessing.alignment.cascaded_alignment import (
        run_all, SCENES,
    )
    results = run_all(data_root="data", output_root="output/alignment")

Usage (selection from pre-computed results)::

    from mm3DPS.preprocessing.alignment.cascaded_alignment import (
        get_aligned_config_path, SCENES,
    )
    for scene_name, frame in SCENES:
        path = get_aligned_config_path(scene_name, frame, data_root="data")
        print(f"{scene_name} -> {path}")

The ``ALIGNMENT_RESULTS`` table encodes the validated correlation scores
from the original alignment runs and must match the configs referenced
by the training pipeline (train_v11).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ======================================================================
# Scene definitions
# ======================================================================

SCENES: List[Tuple[str, int]] = [
    ("seq_0_frame_135", 135),
    ("seq_0_frame_390", 390),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]

# ======================================================================
# Validated alignment results
# ======================================================================
# Correlation scores measured at 1500 renderer hits.
# ``winner`` is max(cc_lidar_4dof, cc_renderer_2dof).
# ``config_suffix`` is the file suffix of the winning aligned config.

ALIGNMENT_RESULTS: Dict[str, dict] = {
    "seq_0_frame_135": {
        "frame": 135,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.476,
        "cc_renderer_2dof": 0.466,
        "config_suffix": "_aligned_gpu",
    },
    "seq_0_frame_390": {
        "frame": 390,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.363,
        "cc_renderer_2dof": 0.172,
        "config_suffix": "_aligned_gpu",
    },
    "seq_1_frame_185": {
        "frame": 185,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.463,
        "cc_renderer_2dof": 0.283,
        "config_suffix": "_aligned_gpu",
    },
    "seq_1_frame_438": {
        "frame": 438,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.584,
        "cc_renderer_2dof": 0.613,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_105": {
        "frame": 105,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.113,
        "cc_renderer_2dof": 0.244,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_160": {
        "frame": 160,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.393,
        "cc_renderer_2dof": 0.577,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_300": {
        "frame": 300,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.432,
        "cc_renderer_2dof": 0.389,
        "config_suffix": "_aligned_gpu",
    },
}


# ======================================================================
# Config lookup helpers
# ======================================================================

def get_aligned_config_path(
    scene_name: str,
    frame: int,
    data_root: str = "data",
) -> str:
    """Return the path to the winning aligned config for a scene.

    Args:
        scene_name: e.g. ``"seq_0_frame_135"``
        frame: radar frame number (e.g. 135)
        data_root: root of the preprocessed data tree

    Returns:
        Absolute path to e.g.
        ``data/seq_0_frame_135/configs/cascaded_frame_135_aligned_gpu.json``
    """
    info = ALIGNMENT_RESULTS[scene_name]
    suffix = info["config_suffix"]
    return os.path.join(
        data_root, scene_name, "configs",
        f"cascaded_frame_{frame}{suffix}.json",
    )








# ======================================================================
# Summary / verification
# ======================================================================

def print_summary(data_root: str = "data") -> None:
    """Print a table of alignment results and verify config files exist."""
    print("=" * 78)
    print("Cascaded Radar Alignment — Per-Scene Results (1500-hit validation)")
    print("=" * 78)
    hdr = (
        f"{'Scene':<22} {'Winner':<16} {'CC_lidar':<10} "
        f"{'CC_renderer':<12} {'Config suffix'}"
    )
    print(hdr)
    print("-" * 78)

    all_ok = True
    winner_ccs = []
    for scene_name, frame in SCENES:
        info = ALIGNMENT_RESULTS[scene_name]
        cc_best = max(info["cc_lidar_4dof"], info["cc_renderer_2dof"])
        winner_ccs.append(cc_best)

        path = get_aligned_config_path(scene_name, frame, data_root)
        exists = os.path.isfile(path)
        status = "✓" if exists else "MISSING"
        if not exists:
            all_ok = False

        print(
            f"{scene_name:<22} {info['winner']:<16} "
            f"{info['cc_lidar_4dof']:<10.3f} "
            f"{info['cc_renderer_2dof']:<12.3f} "
            f"{info['config_suffix']}  [{status}]"
        )

    print("-" * 78)
    print(f"Mean winner CC: {sum(winner_ccs) / len(winner_ccs):.4f}")
    print(f"Config files: {'all present' if all_ok else 'SOME MISSING'}")
    print("=" * 78)


# ======================================================================
# Pass-1 single-frame alignment (CUDA backend)
# ======================================================================
#
# Mirrors the API of mm3DPS's pass-1 ``run_single_frame`` / ``run_all`` but
# replaces the Mitsuba MC renderer with the CUDA rendering backend
# (see ``cascaded_renderer_cuda``). Both alignment methods (renderer 2-DOF
# + LiDAR 4-DOF) are run per frame, scored with the SAME CUDA renderer cc
# (comparable metric), and the higher-cc winner is written. No trajectory
# prior — that's pass 2's job.
#
# Output contract (``{output_root}/{scene}/cascade/``):
#   cascaded_frame_<F>_aligned_2dof.json    — renderer-method candidate
#   cascaded_frame_<F>_aligned_gpu.json     — lidar-method candidate
#   cascaded_frame_<F>_aligned.json         — winner copy
#   cascaded_frame_<F>_alignment_log.json   — metadata + both scores

def run_single_frame(
    scene_name: str,
    frame: int,
    data_root: str = "data",
    output_root: str = "data/alignment_data",
    aligned_config_dir: Optional[str] = None,
    tx_pattern: Optional[str] = None,
    rx_pattern: Optional[str] = None,
    skip_if_exists: bool = True,
    target_n: int = 30000,
    verbose: bool = True,
) -> Optional[dict]:
    """Run both pass-1 alignment methods for one cascade frame and pick
    the higher-cc winner. All rendering goes through the CUDA backend.

    Args:
        scene_name: e.g. ``"seq_0_frame_135"``.
        frame: cascade frame index.
        data_root: root of preprocessed data tree.
        output_root: where aligned configs + logs land; tree layout is
            ``{output_root}/{scene}/cascade/...``.
        aligned_config_dir: override the default subdirectory (rarely needed).
        tx_pattern / rx_pattern: optional MMWCAS antenna pattern paths;
            defaults resolve to ``assets/antenna_pattern/MMWCAS/*_76.npy``.
        skip_if_exists: when True, return the existing log verbatim if all
            four output files are already on disk.
        target_n: FPS target point count for the CUDA alignment context.
            30 k is enough for alignment cc-ranking; training uses 90 k.
        verbose: print progress.

    Returns the log dict (same schema as pass-1 upstream) or None if the
    prerequisites are missing.
    """
    import time
    import json as _json
    import shutil

    # Resolve paths
    data_dir = os.path.join(data_root, scene_name)
    base_config_path = os.path.join(
        data_dir, "configs", f"cascaded_frame_{frame}.json")
    gt_adc_path = os.path.join(
        data_dir, "radar", f"cascaded_frame_{frame}.npy")
    mesh_path = os.path.join(data_dir, "scene", "mesh.ply")
    pcl_path = os.path.join(data_dir, "scene", "pcl.npy")

    missing = [p for p in (base_config_path, gt_adc_path, mesh_path)
               if not os.path.isfile(p)]
    if missing:
        if verbose:
            for p in missing:
                print(f"  [f={frame}] SKIP: not found: {p}")
        return None

    if aligned_config_dir is None:
        aligned_config_dir = os.path.join(output_root, scene_name, "cascade")
    os.makedirs(aligned_config_dir, exist_ok=True)

    aligned_2dof_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}_aligned_2dof.json")
    aligned_gpu_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}_aligned_gpu.json")
    aligned_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}_aligned.json")
    log_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}_alignment_log.json")

    if skip_if_exists and all(os.path.isfile(p) for p in (
            aligned_2dof_path, aligned_gpu_path, aligned_path, log_path)):
        if verbose:
            print(f"  [f={frame}] existing alignment, reusing log")
        try:
            return _json.load(open(log_path))
        except Exception:
            pass  # fall through to re-run

    # Lazy imports (Mitsuba / cupy / CUDA kernels pulled in on demand)
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb")

    from .cascaded_renderer_cuda import (
        build_alignment_context, apply_pose_to_ctx,
        render_and_evaluate_cuda, destroy_alignment_context,
    )
    from .cascaded_renderer import (
        apply_2dof_to_config, apply_4dof_to_config,
        grid_search_2dof, refine_2dof,
        save_aligned_config_from_pose,
    )

    with open(base_config_path) as f:
        base_config = _json.load(f)

    # Build the CUDA alignment context (FPS'd model + GT cart RA).
    # Shared across both alignment methods for a consistent cc objective.
    t_ctx = time.time()
    ctx = build_alignment_context(
        config_file=base_config_path,
        tx_pattern_file=tx_pattern, rx_pattern_file=rx_pattern,
        target_n=target_n, verbose=False,
    )
    ctx_build_s = time.time() - t_ctx

    # ── Method 1: renderer 2-DOF (CUDA) ──
    t_r = time.time()
    grid_params, _, _ = grid_search_2dof(
        ctx, base_config, verbose=False,
    )
    refined_params, cc_renderer, _ = refine_2dof(
        ctx, base_config, grid_params, max_evals=30, verbose=False,
    )
    tx_mm_r, rx_mm_r, bs_r = apply_2dof_to_config(
        base_config, refined_params[0], refined_params[1])
    save_aligned_config_from_pose(
        base_config, tx_mm_r, rx_mm_r, bs_r, aligned_2dof_path)
    t_renderer_s = time.time() - t_r
    if verbose:
        print(f"  [f={frame}] renderer_2dof: cc={cc_renderer:.4f}  "
              f"dr={refined_params[0]:+.3f}m da={refined_params[1]:+.2f}°  "
              f"[{t_renderer_s:.1f}s]")

    # ── Method 2: LiDAR 4-DOF (cupy) ──
    cc_lidar = None
    lidar_params_4dof = None
    t_l = time.time()
    try:
        import open3d as o3d
        from .cascaded_lidar_gpu import optimize_alignment_gpu
        from .cascaded_lidar import (
            load_point_cloud, load_radar_config, radar_gt_to_ra_map,
        )

        params, base_origin, base_boresight = load_radar_config(base_config_path)

        pcd = load_point_cloud(mesh_path)
        if not pcd.has_points() and os.path.isfile(pcl_path):
            pcl = np.load(pcl_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
            if pcl.shape[1] >= 7:
                inten = pcl[:, 6].astype(np.float32)
                inten = (inten - inten.min()) / max(
                    float(inten.max() - inten.min()), 1e-9)
                pcd.colors = o3d.utility.Vector3dVector(
                    np.stack([inten, inten, inten], axis=1))

        ra_radar = radar_gt_to_ra_map(gt_adc_path)
        max_range = params["num_adc"] * params["range_resolution"]
        results_4dof = optimize_alignment_gpu(
            pcd, ra_radar, params, base_boresight, base_origin,
            range_search_m=(-max_range * 0.1, max_range * 0.1),
            azimuth_search_deg=(-15.0, 15.0),
            rotation_elev_search_deg=(-10.0, 10.0),
            rotation_azim_search_deg=(-10.0, 10.0),
            metric="correlation", coarse_steps=5,
            near_field_m=1.5, batch_size=100, verbose=False,
        )
        opt = results_4dof["optimal_params"]
        lidar_params_4dof = [
            float(opt["delta_range_m"]),
            float(opt["delta_azimuth_deg"]),
            float(opt["rotation_elev_deg"]),
            float(opt["rotation_azim_deg"]),
        ]
        tx_mm_l, rx_mm_l, bs_l = apply_4dof_to_config(
            base_config, *lidar_params_4dof)

        # Score lidar-method pose through the CUDA renderer so its cc is
        # comparable to renderer_2dof's.
        apply_pose_to_ctx(ctx, tx_mm_l, rx_mm_l, bs_l)
        cc_lidar = float(render_and_evaluate_cuda(ctx))
        save_aligned_config_from_pose(
            base_config, tx_mm_l, rx_mm_l, bs_l, aligned_gpu_path)
    except ImportError:
        raise  # a broken install must stop the run, not skip LiDAR
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  [f={frame}] lidar_4dof: FAILED ({e}); renderer only")
    t_lidar_s = time.time() - t_l
    if verbose and cc_lidar is not None:
        print(f"  [f={frame}] lidar_4dof:    cc={cc_lidar:.4f}  "
              f"deltas={lidar_params_4dof}  [{t_lidar_s:.1f}s]")

    # ── Winner selection ──
    if cc_lidar is not None and cc_lidar > cc_renderer:
        winner = "lidar_4dof"
        winner_cc = cc_lidar
        winner_src = aligned_gpu_path
        suffix = "_aligned_gpu"
    else:
        winner = "renderer_2dof"
        winner_cc = float(cc_renderer)
        winner_src = aligned_2dof_path
        suffix = "_aligned_2dof"

    # Write winner copy (used by downstream code)
    shutil.copy2(winner_src, aligned_path)

    result = {
        "frame": int(frame),
        "winner": winner,
        "cc_lidar_4dof": cc_lidar,
        "cc_renderer_2dof": float(cc_renderer),
        "config_suffix": suffix,
        "winner_cc": float(winner_cc),
        "aligned_config_path": aligned_path,
        "params_renderer_2dof": {
            "delta_range_m": float(refined_params[0]),
            "delta_azimuth_deg": float(refined_params[1]),
        },
        "params_lidar_4dof": (
            None if lidar_params_4dof is None else {
                "delta_range_m": lidar_params_4dof[0],
                "delta_azimuth_deg": lidar_params_4dof[1],
                "rotation_elev_deg": lidar_params_4dof[2],
                "rotation_azim_deg": lidar_params_4dof[3],
            }
        ),
        "time_renderer_s": float(t_renderer_s),
        "time_lidar_s": float(t_lidar_s),
        "time_ctx_build_s": float(ctx_build_s),
        "backend": "cuda_v5",
    }
    with open(log_path, "w") as f:
        _json.dump(result, f, indent=2)

    if verbose:
        print(f"  [f={frame}] WINNER: {winner} (cc={winner_cc:.4f})")

    destroy_alignment_context(ctx)
    return result


def run_all(
    data_root: str = "data",
    output_root: str = "data/alignment_data",
    scenes: Optional[List[str]] = None,
    skip_if_exists: bool = True,
    target_n: int = 30000,
    verbose: bool = True,
) -> Dict[str, List[dict]]:
    """Run pass-1 alignment across every cascade frame in every scene.

    ``scenes`` defaults to every ``seq_*`` directory under ``data_root``.
    For each scene, every cascaded ADC file
    ``{data_root}/{scene}/radar/cascaded_frame_*.npy`` is aligned via
    ``run_single_frame``. Returns a dict mapping ``scene_name`` to a list
    of per-frame log dicts.
    """
    import glob

    if scenes is None:
        scenes = sorted(
            d for d in os.listdir(data_root)
            if d.startswith("seq_")
            and os.path.isdir(os.path.join(data_root, d, "radar"))
        )

    out = {}
    for sc in scenes:
        radar_dir = os.path.join(data_root, sc, "radar")
        files = sorted(glob.glob(os.path.join(radar_dir, "cascaded_frame_*.npy")))
        frames = []
        for p in files:
            base = os.path.basename(p).replace("cascaded_frame_", "").replace(".npy", "")
            try:
                frames.append(int(base))
            except ValueError:
                pass

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"pass-1 (CUDA backend): {sc}  ({len(frames)} frames)")
            print(f"{'=' * 70}")

        results = []
        for f in frames:
            r = run_single_frame(
                scene_name=sc, frame=f,
                data_root=data_root, output_root=output_root,
                skip_if_exists=skip_if_exists, target_n=target_n,
                verbose=verbose,
            )
            if r is not None:
                results.append(r)
        out[sc] = results

        if verbose and results:
            mean_cc = float(np.mean([r["winner_cc"] for r in results]))
            n_ren = sum(1 for r in results if r["winner"] == "renderer_2dof")
            n_lid = sum(1 for r in results if r["winner"] == "lidar_4dof")
            print(f"\n[{sc}] mean winner cc={mean_cc:.4f}  "
                  f"renderer_2dof={n_ren}  lidar_4dof={n_lid}")

    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pass-1 cascade alignment (CUDA backend)")
    parser.add_argument(
        "--scene", default=None,
        help="single scene name (e.g. seq_0_frame_135); omit to run all scenes")
    parser.add_argument(
        "--data-root", default="data",
        help="Root of preprocessed data tree")
    parser.add_argument(
        "--output-root", default="data/alignment_data",
        help="Root of alignment artefacts")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run alignment even if existing outputs are on disk")
    parser.add_argument(
        "--target-n", type=int, default=30000,
        help="FPS target point count for the CUDA alignment context")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print the historical alignment table (ALIGNMENT_RESULTS) only")
    args = parser.parse_args()

    if args.summary_only:
        print_summary(args.data_root)
    elif args.scene:
        run_single_frame.__call__  # touch to fail fast if something's unwired
        run_all(
            data_root=args.data_root, output_root=args.output_root,
            scenes=[args.scene],
            skip_if_exists=not args.force, target_n=args.target_n, verbose=True)
    else:
        run_all(
            data_root=args.data_root, output_root=args.output_root,
            skip_if_exists=not args.force, target_n=args.target_n, verbose=True)

