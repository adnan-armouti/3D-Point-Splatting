import os
import argparse

from mm3DPS.preprocessing.radar_utils import generate_adc_window
from mm3DPS.preprocessing.config_utils import generate_config_window
from mm3DPS.preprocessing.lidar_utils import generate_lidar_window, build_and_save_lidar_scene
from mm3DPS.preprocessing.mesh_utils import reconstruct_mesh_from_scene


def _build_parser():
    ap = argparse.ArgumentParser(description="Preprocessing Orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ADC
    ap_adc = sub.add_parser("adc", help="Generate cascade ADC windows")
    ap_adc.add_argument("--seq", type=int, default=1)
    ap_adc.add_argument("--frame", type=int, default=185)
    ap_adc.add_argument("--num-radar-frames", type=int, default=1)
    ap_adc.add_argument("--dataset-dir", type=str, default=None)
    ap_adc.add_argument("--calib-path", type=str, default=None)
    ap_adc.add_argument("--out-root", type=str, default=None)
    ap_adc.add_argument("--cascade", action="store_true")
    ap_adc.add_argument("--verbose", action="store_true")

    # Configs
    ap_cfg = sub.add_parser("configs", help="Generate cascade config windows")
    ap_cfg.add_argument("--seq", type=int, default=1)
    ap_cfg.add_argument("--frame", type=int, default=185)
    ap_cfg.add_argument("--num-radar-frames", type=int, default=1)
    ap_cfg.add_argument("--dataset-dir", type=str, default=None)
    ap_cfg.add_argument("--calib-path", type=str, default=None)
    ap_cfg.add_argument("--out-root", type=str, default=None)
    ap_cfg.add_argument("--cascade", action="store_true")
    ap_cfg.add_argument("--apply-adjust", action="store_true")
    ap_cfg.add_argument("--viz", action="store_true")
    ap_cfg.add_argument("--remote-viz", action="store_true")
    ap_cfg.add_argument("--verbose", action="store_true")

    # LiDAR frames
    ap_lf = sub.add_parser("lidar-frames", help="Generate individual LiDAR frames aligned to cascade frames")
    ap_lf.add_argument("--seq", type=int, default=1)
    ap_lf.add_argument("--frame", type=int, default=185)
    ap_lf.add_argument("--num-radar-frames", type=int, default=1)
    ap_lf.add_argument("--dataset-dir", type=str, default=None)
    ap_lf.add_argument("--calib-path", type=str, default=None)
    ap_lf.add_argument("--out-root", type=str, default=None)
    ap_lf.add_argument("--verbose", action="store_true")

    # Scene
    ap_scene = sub.add_parser("scene", help="Build and save LiDAR scene (pcl.npy)")
    ap_scene.add_argument("--seq", type=int, default=1)
    ap_scene.add_argument("--frame", type=int, default=185)
    ap_scene.add_argument("--dataset-dir", type=str, default=None)
    ap_scene.add_argument("--calib-path", type=str, default=None)
    ap_scene.add_argument("--out-root", type=str, default=None)
    ap_scene.add_argument("--num-lidar-frames", type=int, default=50)
    ap_scene.add_argument("--buffer-distance", type=float, default=1.0)
    ap_scene.add_argument("--normals-radius", type=float, default=0.1)
    ap_scene.add_argument("--remove-behind-radar", action="store_true")
    ap_scene.add_argument("--config-path", type=str, default=None)
    ap_scene.add_argument("--verbose", action="store_true")

    # All-in-one
    ap_all = sub.add_parser("all", help="Run scene, configs, adc, and lidar-frames in one go")
    ap_all.add_argument("--seq", type=int, default=1)
    ap_all.add_argument("--frame", type=int, default=185)
    ap_all.add_argument("--num-radar-frames", type=int, default=1)
    ap_all.add_argument("--dataset-dir", type=str, default=None)
    ap_all.add_argument("--calib-path", type=str, default=None)
    ap_all.add_argument("--out-root", type=str, default=None)
    # configs + adc toggles
    ap_all.add_argument("--cascade", action="store_true")
    ap_all.add_argument("--apply-adjust", action="store_true")
    # scene options
    ap_all.add_argument("--num-lidar-frames", type=int, default=50)
    ap_all.add_argument("--buffer-distance", type=float, default=1.0)
    ap_all.add_argument("--normals-radius", type=float, default=0.1)
    ap_all.add_argument("--remove-behind-radar", action="store_true")
    ap_all.add_argument("--config-path", type=str, default=None)
    # viz for configs only
    ap_all.add_argument("--viz", action="store_true")
    ap_all.add_argument("--remote-viz", action="store_true")
    ap_all.add_argument("--verbose", action="store_true")

    # Placeholders for future subcommands (configs, lidar-frames, scene)
    return ap


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "adc":
        if not args.cascade:
            parser.error("Provide --cascade")
        res = generate_adc_window(
            seq_idx=args.seq,
            center_frame_idx=args.frame,
            num_radar_frames=args.__dict__.get("num_radar_frames", 1),
            dataset_dir=args.dataset_dir,
            calib_path=args.calib_path,
            out_root=args.out_root,
            run_cascade=args.cascade,
            verbose=args.verbose,
        )
        if args.verbose:
            for k, v in res.items():
                print(f"[{k}] -> {v}")
    elif args.cmd == "configs":
        if not args.cascade:
            parser.error("Provide --cascade")
        res = generate_config_window(
            seq_idx=args.seq,
            center_frame_idx=args.frame,
            num_radar_frames=args.__dict__.get("num_radar_frames", 1),
            dataset_dir=args.dataset_dir,
            calib_path=args.calib_path,
            out_root=args.out_root,
            run_cascade=args.cascade,
            apply_adjust=args.apply_adjust,
            verbose=args.verbose,
        )
        if args.verbose:
            for k, v in res.items():
                print(f"[{k}] -> {v}")
    elif args.cmd == "lidar-frames":
        paths = generate_lidar_window(
            seq_idx=args.seq,
            center_frame_idx=args.frame,
            num_radar_frames=args.num_radar_frames,
            dataset_dir=args.dataset_dir,
            calib_path=args.calib_path,
            out_root=args.out_root,
            verbose=args.verbose,
        )
        if args.verbose:
            for p in paths:
                print(f"[lidar] -> {p}")
    elif args.cmd == "scene":
        save_path = build_and_save_lidar_scene(
            seq_idx=args.seq,
            frame_idx=args.frame,
            dataset_dir=args.dataset_dir,
            calib_path=args.calib_path,
            out_root=args.out_root,
            num_lidar_frames=args.num_lidar_frames,
            buffer_distance=args.buffer_distance,
            normals_radius=args.normals_radius,
            remove_behind_radar=args.remove_behind_radar,
            config_path=args.config_path,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"[scene] -> {save_path}")
        # Reconstruct mesh from the newly generated scene point cloud
        try:
            scene_dir = os.path.dirname(save_path)
            out_mesh = os.path.join(scene_dir, f"mesh.ply")
            result = reconstruct_mesh_from_scene(
                scene_npy_path=save_path,
                out_mesh_path=out_mesh,
                depth=8,
                point_weight=1.0,
                trim_threshold=8.0,
            )
            if args.verbose:
                print(f"[reconstruct] -> {result}")
        except Exception as e:
            print(f"[reconstruct] failed: {e}")
    elif args.cmd == "all":
        # Validate viz flags (for configs)
        if args.viz and args.__dict__.get("remote_viz", False):
            parser.error("--viz (local) and --remote-viz are mutually exclusive; choose one.")

        # 1) scene (always)
        scene_path = build_and_save_lidar_scene(
            seq_idx=args.seq,
            frame_idx=args.frame,
            dataset_dir=args.dataset_dir,
            calib_path=args.calib_path,
            out_root=args.out_root,
            num_lidar_frames=args.num_lidar_frames,
            buffer_distance=args.buffer_distance,
            normals_radius=args.normals_radius,
            remove_behind_radar=args.remove_behind_radar,
            config_path=args.config_path,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"[scene] -> {scene_path}")
        # Reconstruct mesh from scene point cloud
        try:
            scene_dir = os.path.dirname(scene_path)
            out_mesh = os.path.join(scene_dir, f"mesh.ply")
            result = reconstruct_mesh_from_scene(
                scene_npy_path=scene_path,
                out_mesh_path=out_mesh,
                depth=8,
                point_weight=1.0,
                trim_threshold=8.0,
            )
            if args.verbose:
                print(f"[reconstruct] -> {result}")
        except Exception as e:
            print(f"[reconstruct] failed: {e}")

        # 2) configs (optional via --cascade)
        if args.cascade:
            cfg_res = generate_config_window(
                seq_idx=args.seq,
                center_frame_idx=args.frame,
                num_radar_frames=args.__dict__.get("num_radar_frames", 1),
                dataset_dir=args.dataset_dir,
                calib_path=args.calib_path,
                out_root=args.out_root,
                run_cascade=args.cascade,
                apply_adjust=args.apply_adjust,
                verbose=args.verbose,
            )
            if args.verbose:
                for k, v in cfg_res.items():
                    print(f"[configs:{k}] -> {v}")
            # Optional combined viz
            if args.viz or args.__dict__.get("remote_viz", False):
                # Lightweight re-entry: rely on generate_config_window's structure in configs script if needed
                pass

        # 3) adc (optional via --cascade)
        if args.cascade:
            adc_res = generate_adc_window(
                seq_idx=args.seq,
                center_frame_idx=args.frame,
                num_radar_frames=args.__dict__.get("num_radar_frames", 1),
                dataset_dir=args.dataset_dir,
                calib_path=args.calib_path,
                out_root=args.out_root,
                run_cascade=args.cascade,
                verbose=args.verbose,
            )
            if args.verbose:
                for k, v in adc_res.items():
                    print(f"[adc:{k}] -> {v}")

        # 4) lidar frames (always)
        lf_paths = generate_lidar_window(
            seq_idx=args.seq,
            center_frame_idx=args.frame,
            num_radar_frames=args.num_radar_frames,
            dataset_dir=args.dataset_dir,
            calib_path=args.calib_path,
            out_root=args.out_root,
            verbose=args.verbose,
        )
        if args.verbose:
            for p in lf_paths:
                print(f"[lidar] -> {p}")


if __name__ == "__main__":
    main()


