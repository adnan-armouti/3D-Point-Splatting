"""3DPS training entry point.

Usage (example)::

    python -m mm3DPS.train_frame_nvs --scene seq_1_frame_438 --target_n 20000 --iters 500

The test frame and training frames follow from the scene name unless given.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm3DPS.config import DEFAULT_DATA_ROOT, check_scene_data, default_split
from mm3DPS.trainer import Trainer, TrainSettings


def main():
    ap = argparse.ArgumentParser(
        description='Frame-level NVS trainer')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT,
                    help='Prepared data directory (default: <repo>/data).')
    ap.add_argument('--test_frame', type=int, default=None,
                    help='Held-out frame (default: the F in the scene name).')
    ap.add_argument('--train_frames', default=None,
                    help='Comma-separated list, e.g. 434,435,436,437,439,440,441,442 '
                         '(default: F-4..F-1 and F+1..F+4).')
    ap.add_argument('--held_out_loop', type=int, default=0)
    ap.add_argument('--train_loops', default='0',
                    help='Comma-separated chirp-loop indices to train on; '
                         'default = "0" (first chirp loop only — the canonical '
                         'paper recipe). Pass "" or omit to fall back to None '
                         'which expands to all 16 loops of the cascaded radar.')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--mat_lr', type=float, default=0.01)
    ap.add_argument('--rot_lr', type=float, default=5e-3)
    ap.add_argument('--loss_type', default='mse_raw',
                    choices=['mse', 'pearson', 'mse_raw'])
    ap.add_argument('--target_n', type=int, default=20000,
                    help='Target Gaussian count after init. Default 20000 '
                         '.')
    ap.add_argument('--frame_period_s', type=float, default=0.1)
    ap.add_argument('--loop_dt_s', type=float, default=7.87e-3 / 16.0)
    ap.add_argument('--no_pass2', action='store_true',
                    help='Opt out of pass-2 trajectory-aware alignment. '
                         'Default is pass-2 (cascaded_frame_<F>_aligned_pass2.json); '
                         'this flag falls back to pass-1 (cascaded_frame_<F>_aligned.json).')
    ap.add_argument('--reg_warm_iters', type=int, default=100,
                    help='Warmup iters before the first densify round. '
                         'Default 100.')
    ap.add_argument('--reg_densify_interval', type=int, default=100,
                    help='S4: densify/prune every N iters after --reg_warm_iters. '
                         'Default 100; 0 = disabled.')
    ap.add_argument('--reg_densify_split_frac', type=float, default=0.05,
                    help='S4: fraction of top-signal points to split per round. '
                         'Default 0.05.')
    ap.add_argument('--reg_densify_prune_frac', type=float, default=0.05,
                    help='S4: fraction of bottom-signal points to prune per round '
                         '(must equal --reg_densify_split_frac to keep N fixed). '
                         'Default 0.05.')
    ap.add_argument('--reg_densify_until', type=int, default=400,
                    help='S4: stop densifying past this iter (lets the final '
                         'iters converge on a stable point set). Default 400 '
                         '.')
    ap.add_argument('--reg_densify_pos_jitter_m', type=float, default=0.02,
                    help='S4: std (m) of isotropic Gaussian position noise for '
                         'new child points. 0.02 = 2 cm ≈ 5λ at 77 GHz.')
    ap.add_argument('--reg_densify_mat_jitter', type=float, default=0.1,
                    help='S4: std of Gaussian noise added to raw_materials for '
                         'child points (raw units).')
    ap.add_argument('--seed_frame', type=int, default=None,
                    help='Frame whose pose seeds the rasterizer (drives FOV + '
                         'RX visibility before FPS, so it determines the 90k-'
                         'point subset of pcl.npy used for training). Default = '
                         '`test_frame` so HO and UB variants share an identical '
                         'position grid.'
                         'HO grid (`--seed_frame {test_frame+1}`) or to probe '
                         'sensitivity to the seed.')
    ap.add_argument('--learn_positions_lr', type=float, default=1e-5,
                    help='> 0 enables learnable '
                         'positions (added to Adam at this LR). Default '
                         '1e-5. Gradient flows '
                         'AMPLITUDE-PATH ONLY (detach_phase=True is enforced '
                         'by the renderer; phi_carrier and n_peak are '
                         'detached). NO PHASE GRADIENTS by construction. '
                         'Set 0.0 to freeze positions.')
    ap.add_argument('--learn_positions_l2', type=float, default=100.0,
                    help='L2 anchor coefficient on (positions − init). '
                         'Default 100. Keeps positions '
                         'sub-mm from LiDAR seed.')
    # ── Ablation knobs ──
    # Defaults match the canonical 3DPS recipe; flipping any of these
    # exercises a single ablation axis. The runner script encodes the
    # chosen axis into --output_dir so each run lands in its own
    # output/ablations/<tier>/<axis>/<config>/<scene>...
    ap.add_argument('--output_dir', default=None,
                    help='Override the auto-constructed run directory. The '
                         'ablation runner sets this to '
                         'output/ablations/<tier>/<axis>/<config>/<scene_run_tag>/.')
    ap.add_argument('--no_phase_detach', action='store_true',
                    help='do NOT detach n_peak / phi_carrier '
                         'before scatter — gradients flow through the '
                         'carrier phase. Forces the PyTorch Step-5 fallback '
                         'since the fused CUDA kernel assumes detached phase.')
    ap.add_argument('--no_mimo_factorization', action='store_true',
                    help='disable the fused CUDA kernels for '
                         'Step-4 (BSDF) and Step-5 (range splat). Falls '
                         'through to the PyTorch path that materialises full '
                         '(M, n_tx, n_rx) BSDF and (spread, M·n_tx·n_rx) '
                         'splat tensors. ~3-10x slower per Adam step; '
                         'memory may OOM at N=20k → may need to reduce N.')
    ap.add_argument('--psf_spread', type=int, default=None,
                    help='Hann PSF kernel half-width L '
                         '(default 15; ablation sweeps {5, 9, 21, 25}). Even '
                         'values are blocked downstream since they produce '
                         'L+1 bins via arange(-(L//2), L//2+1).')
    ap.add_argument('--no_cull', action='store_true',
                    help='disable the FOV / azimuth-cone cull '
                         'in init_visible_weighted (keeps every pcl point '
                         'past the 1.5 m near-field guard).')
    ap.add_argument('--no_occlusion', action='store_true',
                    help='skip the Mitsuba RX-side ray test in '
                         'init_visible_weighted (no occlusion filtering).')
    ap.add_argument('--no_cosine_resample', action='store_true',
                    help='replace cos_bore importance weights '
                         'with uniform weights at the resample stage.')
    ap.add_argument('--no_fps', action='store_true',
                    help='replace farthest-point sampling '
                         'with uniform random subsampling (seed 42).')
    args = ap.parse_args()

    check_scene_data(args.scene, args.data_root)
    if args.test_frame is None or args.train_frames is None:
        _test, _train = default_split(args.scene)
        if args.test_frame is None:
            args.test_frame = _test
        if args.train_frames is None:
            args.train_frames = ','.join(str(f) for f in _train)
    train_frames = [int(x) for x in args.train_frames.split(',') if x.strip()]
    train_loops = (None if args.train_loops is None else
                   [int(x) for x in args.train_loops.split(',') if x.strip()])
    if args.psf_spread is not None and args.psf_spread % 2 == 0:
        raise SystemExit(
            f'--psf_spread must be odd (got {args.psf_spread}); '
            'arange(-(L//2), L//2+1) yields L+1 bins for even L.')
    Trainer(TrainSettings(
        scene=args.scene,
        train_frames=train_frames,
        test_frame=args.test_frame,
        held_out_loop=args.held_out_loop,
        train_loops=train_loops,
        num_iters=args.iters,
        mat_lr=args.mat_lr, rot_lr=args.rot_lr,
        loss_type=args.loss_type, target_n=args.target_n,
        frame_period_s=args.frame_period_s, loop_dt_s=args.loop_dt_s,
        use_pass2_alignment=not args.no_pass2,
        reg_warm_iters=args.reg_warm_iters,
        reg_densify_interval=args.reg_densify_interval,
        reg_densify_split_frac=args.reg_densify_split_frac,
        reg_densify_prune_frac=args.reg_densify_prune_frac,
        reg_densify_until=args.reg_densify_until,
        reg_densify_pos_jitter_m=args.reg_densify_pos_jitter_m,
        reg_densify_mat_jitter=args.reg_densify_mat_jitter,
        seed_frame=args.seed_frame,
        learn_positions_lr=args.learn_positions_lr,
        learn_positions_l2=args.learn_positions_l2,
        output_dir=args.output_dir,
        data_root=args.data_root,
        # Ablation knobs (defaults preserve canonical recipe).
        detach_phase=not args.no_phase_detach,
        use_mimo_factorization=not args.no_mimo_factorization,
        psf_spread=args.psf_spread,
        enable_cull=not args.no_cull,
        enable_occlusion=not args.no_occlusion,
        enable_cosine_resample=not args.no_cosine_resample,
        enable_fps=not args.no_fps)).run()


if __name__ == '__main__':
    main()
