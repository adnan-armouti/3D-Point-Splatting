"""The 3DPS frame-level trainer: settings, the Trainer class and its exports."""

import gc
import os
import sys
import json
import time
from dataclasses import dataclass

import numpy as np
import torch

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm3DPS.data.io_utils import compute_range_res_from_cfg
from mm3DPS.evaluation.utils.metrics import compute_cart_ra_metrics
from mm3DPS.rasterizer import Rasterizer
from mm3DPS import cuda as v5cuda
from mm3DPS.train_gaussian import (
    DEVICE,
    init_visible_weighted,
    cull_gaussians,
    render_gaussians,
    range_profile_to_ra_mag,
    build_polar_to_cart_grid,
    polar_to_cart_torch,
    cart_corr_torch,
    compute_ra_loss_rp,
    get_lr_scale,
    rms_clip_grad,
)
from mm3DPS.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, load_trained_config
from mm3DPS.pose_utils import apply_pose
from mm3DPS.frame_dataset import build_frame_level_dataset


def _densify_step(model, optimizer, signal_per_pt, split_n, prune_n,
                  pos_jitter_m, mat_jitter, rot_jitter=0.05,
                  init_raw_ref=None, init_normals_ref=None,
                  verbose=False):
    """In-place split + prune to reallocate the fixed N-point budget.

    The bottom ``prune_n`` points (lowest signal) are *replaced* in-place
    by children of the top ``split_n`` points (highest signal). Child =
    parent params + Gaussian noise (position σ = ``pos_jitter_m``,
    rotation σ = ``rot_jitter``, material σ = ``mat_jitter``); Adam
    can repair this quickly.

    Tensor indices of non-modified points are preserved (prune slots are
    overwritten, not deleted). Adam state at replaced slots is zeroed so
    children start with no momentum. ``init_raw_ref`` / ``init_normals_ref``
    are patched so drift penalties see zero drift at birth.

    Requires ``split_n == prune_n`` and both ≤ 0.45·N.
    """
    assert split_n == prune_n, 'split_n and prune_n must match to preserve N'
    N = model.N
    assert split_n <= 0.45 * N, 'split/prune frac too large; may overlap'

    _, order = signal_per_pt.sort(descending=True)
    split_idx = order[:split_n].clone()
    prune_idx = order[-prune_n:].clone()

    overlap = set(split_idx.tolist()) & set(prune_idx.tolist())
    assert not overlap, f'split/prune index overlap: {overlap}'

    with torch.no_grad():
        # Gather parent params
        pos_parent = model.positions[split_idx].clone()
        rot_parent = model.rotations[split_idx].clone()
        raw_parent = model.raw_materials[split_idx].clone()

        # Generate child position + rotation via Gaussian jitter.
        pos_new = pos_parent + torch.randn_like(pos_parent) * pos_jitter_m
        rot_new = torch.nn.functional.normalize(
            rot_parent + torch.randn_like(rot_parent) * rot_jitter, dim=-1)

        # Material: parent + Gaussian noise
        raw_new = raw_parent + torch.randn_like(raw_parent) * mat_jitter

        # In-place overwrite of prune slots with children
        model.positions[prune_idx]     = pos_new
        model.rotations[prune_idx]     = rot_new
        model.raw_materials[prune_idx] = raw_new

        # Adam state surgery: zero momentum + velocity at replaced slots.
        for p in [model.positions, model.rotations, model.raw_materials]:
            st = optimizer.state.get(p)
            if st is None:
                continue
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st and st[k].shape == p.shape:
                    st[k][prune_idx] = 0.0

        # Patch init_* so drift penalties see zero drift at birth
        if init_raw_ref is not None:
            init_raw_ref[prune_idx] = raw_new
        if init_normals_ref is not None:
            q = torch.nn.functional.normalize(rot_new, dim=-1)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            nx = 2 * (x * z + w * y)
            ny = 2 * (y * z - w * x)
            nz = 1 - 2 * (x * x + y * y)
            init_normals_ref[prune_idx] = torch.stack([nx, ny, nz], dim=-1)

    if verbose:
        print(f'  [densify]: split {split_n} top-signal → replaced '
              f'{prune_n} bottom-signal  (pos_jitter={pos_jitter_m:.3f} m)')


@dataclass
class TrainSettings:
    """Settings of one training run. Defaults are the paper's recipe."""

    scene: str
    train_frames: list
    test_frame: int
    held_out_loop: int = 0
    train_loops: list = None
    num_iters: int = 500
    mat_lr: float = 0.01
    rot_lr: float = 5e-3
    loss_type: str = 'mse_raw'
    target_n: int = 90000
    frame_period_s: float = 0.1
    loop_dt_s: float = 7.87e-3 / 16.0
    use_pass2_alignment: bool = True
    data_root: str = DEFAULT_DATA_ROOT
    verbose: bool = True
    output_dir: str = None
    reg_warm_iters: int = 50
    reg_densify_interval: int = 0
    reg_densify_split_frac: float = 0.02
    reg_densify_prune_frac: float = 0.02
    reg_densify_until: int = 300
    reg_densify_pos_jitter_m: float = 0.02
    reg_densify_mat_jitter: float = 0.1
    # Pose that selects the point subset; defaults to the test frame.
    seed_frame: int = None
    # Positions learn from the amplitude gradient only (phase stays detached),
    # with sub-mm motion kept by an L2 anchor to the LiDAR init. 0 disables.
    learn_positions_lr: float = 1e-5
    learn_positions_l2: float = 100.0
    # Ablation switches.
    detach_phase: bool = True
    use_mimo_factorization: bool = True
    psf_spread: int = None  # None -> 15
    enable_cull: bool = True
    enable_occlusion: bool = True
    enable_cosine_resample: bool = True
    enable_fps: bool = True


class Trainer:
    """Fits one scene: ``setup``, ``train`` (one ``step`` per iteration),
    ``evaluate``, then ``save``. ``run`` does all four."""

    def __init__(self, settings: TrainSettings):
        self.cfg = settings

    def run(self):
        self.setup()
        self.train()
        self.evaluate()
        return self.save()

    # ── setup ──

    def setup(self):
        cfg = self.cfg
        assert v5cuda.is_available(), (
            'CUDA extension not built. '
            'cd mm3DPS/cuda && python setup.py build_ext --inplace')

        if cfg.train_loops is None:
            cfg.train_loops = list(range(16))
        cfg.train_loops = list(cfg.train_loops)
        self.test_in_train = int(cfg.test_frame) in [int(f) for f in cfg.train_frames]
        if cfg.seed_frame is None:
            cfg.seed_frame = int(cfg.test_frame)
        self._print_header()

        self._build_model()
        self._load_data()

        self.init_test_cc = self._render_and_cart_corr(self.test_sample)
        self.init_train_ccs = [self._render_and_cart_corr(s) for s in self.train_samples]
        if cfg.verbose:
            print(f'\n  [init] test  cc = {self.init_test_cc:.4f}')
            print(f'  [init] train mean cc = {np.mean(self.init_train_ccs):.4f} '
                  f'(std {np.std(self.init_train_ccs):.4f}) over {len(self.init_train_ccs)} samples')

        self._build_optimizer()

    def _print_header(self):
        cfg = self.cfg
        if not cfg.verbose:
            return
        print('=' * 72)
        print(f'frame-level NVS  scene={cfg.scene}')
        print(f'  train frames ({len(cfg.train_frames)}): {cfg.train_frames}')
        print(f'  train loops  ({len(cfg.train_loops)}): {cfg.train_loops}')
        print(f'  test  frame  = {cfg.test_frame}  (test loop={cfg.held_out_loop})'
              + ('   [test frame ALSO in train_frames → upper-bound run]'
                 if self.test_in_train else ''))
        print(f'  alignment    = {"pass-2" if cfg.use_pass2_alignment else "pass-1"}')
        print(f'  iters        = {cfg.num_iters}   loss_type={cfg.loss_type}')
        print('=' * 72)

    def _build_model(self):
        cfg = self.cfg
        config = load_trained_config(cfg.scene, data_root=cfg.data_root)

        # The seed frame's pose drives FOV culling + RX visibility before FPS
        # subsampling, so it determines which subset of pcl.npy this run trains on.
        suffix = '_aligned_pass2' if cfg.use_pass2_alignment else '_aligned'
        align_dir = os.path.join(cfg.data_root, 'alignment_data', cfg.scene, 'cascade')
        self.seed_cfg = os.path.join(align_dir,
                                     f'cascaded_frame_{cfg.seed_frame}{suffix}.json')
        if cfg.verbose:
            print(f'  seed frame   = {cfg.seed_frame}  '
                  f'(drives FOV+visibility before FPS → position grid)')
        assert os.path.exists(self.seed_cfg), f'missing: {self.seed_cfg}'

        self.rast = Rasterizer(
            config_file=self.seed_cfg,
            mesh_file=config.scene_file,
            tx_pattern_file=config.tx_pattern_file,
            rx_pattern_file=config.rx_pattern_file,
            device=DEVICE)

        # Init model at seed pose. We need a Mitsuba scene for the init-time
        # visibility test (skipped when enable_occlusion=False).
        self.model = init_visible_weighted(
            cfg.scene, self.rast, target_n=cfg.target_n,
            enable_cull=cfg.enable_cull,
            enable_occlusion=cfg.enable_occlusion,
            enable_cosine_resample=cfg.enable_cosine_resample,
            enable_fps=cfg.enable_fps)
        self.rast.free_mi_scene()

        # Every render in training, evaluation and export uses the same options.
        self.render_kwargs = dict(
            detach_phase=cfg.detach_phase,
            use_cuda_kernels=cfg.use_mimo_factorization,
            psf_spread=cfg.psf_spread,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # FOV mask for the seed pose, reused as a conservative mask.
        self.active_mask = cull_gaussians(self.model, self.rast)
        self.vertex_areas = torch.zeros(self.model.N, device=DEVICE)
        self.vertex_areas[self.active_mask] = 1.0

    def _load_data(self):
        cfg = self.cfg
        self.range_res = compute_range_res_from_cfg(self.seed_cfg)
        self.sample_grid = build_polar_to_cart_grid(127, 256, self.range_res, 400, DEVICE)

        self.train_samples, self.test_sample, self.diag = build_frame_level_dataset(
            scene=cfg.scene, train_frames=cfg.train_frames, test_frame=cfg.test_frame,
            held_out_loop=cfg.held_out_loop, use_pass2=cfg.use_pass2_alignment,
            train_loops=cfg.train_loops,
            loss_type=cfg.loss_type, loop_dt_s=cfg.loop_dt_s,
            frame_period_s=cfg.frame_period_s, data_root=cfg.data_root,
            device=DEVICE, verbose=cfg.verbose,
        )

        # GT cart norms for cart_corr evaluation
        for sample in [*self.train_samples, self.test_sample]:
            with torch.no_grad():
                ra_cart = polar_to_cart_torch(sample['gt_ra_polar'], self.sample_grid)
                mn, mx = ra_cart.min(), ra_cart.max()
                sample['gt_cart_norm'] = (
                    (ra_cart - mn) / (mx - mn).clamp(min=1e-30)).detach()

    def _build_optimizer(self):
        cfg = self.cfg
        model = self.model
        param_groups = [
            {'params': [model.raw_materials], 'lr': cfg.mat_lr, 'name': 'materials'},
            {'params': [model.rotations],     'lr': cfg.rot_lr, 'name': 'rotations'},
        ]
        self.clip_vals = {'materials': 1.0, 'rotations': 0.5}
        # detach_phase=True (default) ensures position gradient flows only
        # through alpha_tx, BSDF cos terms, antenna gain, not through
        # phi_carrier or n_peak.
        if cfg.learn_positions_lr > 0.0:
            model.positions.requires_grad_(True)
            param_groups.append({'params': [model.positions], 'lr': cfg.learn_positions_lr,
                                 'name': 'positions'})
            # Phase wraps every 0.97 mm at 77 GHz so we want movement << 1 mm per step.
            self.clip_vals['positions'] = 0.001
            if cfg.verbose:
                print(f'  [positions] learn_positions_lr={cfg.learn_positions_lr:g} '
                      f'(amplitude-path only, L2 anchor λ={cfg.learn_positions_l2:g})')
        self.base_lrs = {g['name']: g['lr'] for g in param_groups}
        self.optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

        self.loss_scale = 1.0 / len(self.train_samples)

        # Init references for densification and the position anchor.
        self.init_raw = model.raw_materials.detach().clone()
        self.init_normals = model.get_normals().detach().clone()  # (N, 3)
        self.init_positions = model.positions.detach().clone()  # (N, 3)
        self.n_densify_rounds = 0
        # Accumulated ‖∇p L‖ between densify steps selects which points to
        # split and prune. The phase path is detached, so this is amplitude only.
        self.pos_grad_accum = None
        if cfg.reg_densify_interval > 0:
            model.positions.requires_grad_(True)
            self.pos_grad_accum = torch.zeros(model.N, device=DEVICE)
            if cfg.verbose:
                print('  [densify] densify_signal=pos_grad_amp; positions '
                      'requires_grad=True (selection signal)')
                print(f'  [reg] densify_interval={cfg.reg_densify_interval}  '
                      f'warm_iters={cfg.reg_warm_iters}  until={cfg.reg_densify_until}')

    # ── rendering ──

    def _render(self):
        return render_gaussians(
            self.model, self.rast, vertex_areas=self.vertex_areas,
            active_mask=self.active_mask, shadow_mask=None,
            **self.render_kwargs)

    def _render_and_cart_corr(self, sample):
        apply_pose(self.rast, sample['pose'])
        with torch.no_grad():
            rp_real, rp_imag = self._render()
            ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart = polar_to_cart_torch(ra_polar, self.sample_grid)
            cc = cart_corr_torch(ra_cart, sample['gt_cart_norm']).item()
        return float(cc)

    # ── training ──

    def train(self):
        """Full-batch training: every sample every iteration."""
        cfg = self.cfg
        self.best_mean_train_cc = -1.0
        self.best_iter = 0
        self.best_state = None
        self.history = []
        t0 = time.time()

        for it in range(cfg.num_iters):
            loss_sum, mean_tc = self.step(it)
            self.history.append({
                'iter': it, 'loss': loss_sum,
                'mean_train_cc': mean_tc,
            })

            if mean_tc > self.best_mean_train_cc:
                self.best_mean_train_cc = mean_tc
                self.best_iter = it
                self.best_state = {k: v.data.clone()
                                   for k, v in self.model.state_dict().items()}

            if cfg.verbose and (it % 10 == 0 or it == cfg.num_iters - 1):
                elapsed = time.time() - t0
                print(f'  iter {it:4d}: loss={loss_sum:.5e}  '
                      f'mean_train_cc={mean_tc:.4f} '
                      f'(best={self.best_mean_train_cc:.4f}@{self.best_iter})  '
                      f'[{elapsed:.0f}s, n={len(self.train_samples)}]')

        self.train_elapsed = time.time() - t0

    def step(self, it):
        """One optimizer step. Returns (loss, mean train correlation)."""
        cfg = self.cfg
        model = self.model
        self.optimizer.zero_grad(set_to_none=True)
        per_sample_cc = []
        loss_sum = 0.0
        for s in self.train_samples:
            apply_pose(self.rast, s['pose'])
            rp_real, rp_imag = self._render()
            loss_k, _ = compute_ra_loss_rp(
                rp_real, rp_imag, s['gt_loss'], loss_type=cfg.loss_type)
            (loss_k * self.loss_scale).backward()
            with torch.no_grad():
                ra_polar = range_profile_to_ra_mag(rp_real.detach(), rp_imag.detach())
                ra_cart = polar_to_cart_torch(ra_polar, self.sample_grid)
                cc = cart_corr_torch(ra_cart, s['gt_cart_norm']).item()
                per_sample_cc.append(cc)
            loss_sum += loss_k.item() * self.loss_scale
            del rp_real, rp_imag, ra_polar, ra_cart, loss_k

        # Full-batch position-gradient magnitude, the densify selection signal.
        if self.pos_grad_accum is not None and model.positions.grad is not None:
            with torch.no_grad():
                step_grad = model.positions.grad.detach().norm(dim=-1)
                self.pos_grad_accum += step_grad

        # L2 anchor on learnable positions. Phase wraps every 0.97 mm at
        # 77 GHz, so positions should drift only sub-mm.
        if cfg.learn_positions_lr > 0.0 and cfg.learn_positions_l2 > 0.0:
            pos_drift = model.positions - self.init_positions
            reg_pos = cfg.learn_positions_l2 * pos_drift.pow(2).mean()
            reg_pos.backward()
            loss_sum += float(reg_pos.item())

        lr_scale = get_lr_scale(
            it, total_iters=cfg.num_iters,
            warmup_iters=0, warmup_factor=1.0, decay_start=200)
        for g in self.optimizer.param_groups:
            g['lr'] = self.base_lrs[g['name']] * lr_scale

        for g in self.optimizer.param_groups:
            clip = self.clip_vals.get(g['name'], 1.0)
            for p in g['params']:
                rms_clip_grad(p, clip)

        self.optimizer.step()

        self._maybe_densify(it)
        return loss_sum, float(np.mean(per_sample_cc))

    def _maybe_densify(self, it):
        """Adaptive density: at iter = reg_warm_iters and every
        reg_densify_interval iters up to reg_densify_until, split the points
        with the largest accumulated gradient and prune those with the smallest."""
        cfg = self.cfg
        model = self.model
        if not (cfg.reg_densify_interval > 0
                and it >= cfg.reg_warm_iters
                and it <= cfg.reg_densify_until
                and (it - cfg.reg_warm_iters) % cfg.reg_densify_interval == 0):
            return
        f_comb = None
        if self.pos_grad_accum is not None and self.pos_grad_accum.max() > 0:
            f_comb = self.pos_grad_accum.clone()  # (N,)
            self.pos_grad_accum.zero_()
        if f_comb is None:
            return
        split_n = int(cfg.reg_densify_split_frac * model.N)
        prune_n = int(cfg.reg_densify_prune_frac * model.N)
        if split_n > 0 and prune_n > 0 and split_n == prune_n:
            _densify_step(
                model, self.optimizer, f_comb,
                split_n=split_n, prune_n=prune_n,
                pos_jitter_m=cfg.reg_densify_pos_jitter_m,
                mat_jitter=cfg.reg_densify_mat_jitter,
                rot_jitter=0.05,
                init_raw_ref=self.init_raw,
                init_normals_ref=self.init_normals,
                verbose=cfg.verbose,
            )
            self.n_densify_rounds += 1
            # Positions moved, so recompute the FOV mask at the first train pose.
            apply_pose(self.rast, self.train_samples[0]['pose'])
            new_mask = cull_gaussians(model, self.rast)
            with torch.no_grad():
                self.active_mask = new_mask
                self.vertex_areas = torch.zeros(model.N, device=DEVICE)
                self.vertex_areas[self.active_mask] = 1.0

    # ── evaluation ──

    def evaluate(self):
        """Restore the best state and score the held-out test frame."""
        cfg = self.cfg
        if self.best_state is not None:
            with torch.no_grad():
                for k, v in self.best_state.items():
                    self.model.state_dict()[k].copy_(v)

        self.final_test_cc = self._render_and_cart_corr(self.test_sample)
        self.final_train_ccs = [self._render_and_cart_corr(s) for s in self.train_samples]
        self.final_train_mean = float(np.mean(self.final_train_ccs))

        apply_pose(self.rast, self.test_sample['pose'])
        with torch.no_grad():
            self.rp_real, self.rp_imag = self._render()
            self.test_ra_polar = range_profile_to_ra_mag(self.rp_real, self.rp_imag)
            self.test_ra_cart = polar_to_cart_torch(self.test_ra_polar, self.sample_grid)
            self.gt_ra_cart = polar_to_cart_torch(
                self.test_sample['gt_ra_polar'], self.sample_grid)

        # Full RA metric set on the test frame (matches baselines).
        self.test_metrics = compute_cart_ra_metrics(
            self.gt_ra_cart.detach().cpu().numpy().astype(np.float32),
            self.test_ra_cart.detach().cpu().numpy().astype(np.float32),
        )

        if cfg.verbose:
            print(f'\n  [final] test  cc = {self.final_test_cc:.4f} '
                  f'(init was {self.init_test_cc:.4f}, '
                  f'Δ {self.final_test_cc - self.init_test_cc:+.4f})')
            print(f'  [final] train mean cc = {self.final_train_mean:.4f}  '
                  f'(best iter {self.best_iter})')
            print(f'  [final] elapsed {self.train_elapsed:.0f}s '
                  f'({self.train_elapsed * 1000 / cfg.num_iters:.1f} ms/iter)')

    # ── saving ──

    def _default_output_dir(self):
        cfg = self.cfg
        tag = (f'train{len(cfg.train_frames)}frames_{len(cfg.train_loops)}loops'
               f'_test{cfg.test_frame}_loop{cfg.held_out_loop}')
        if self.test_in_train:
            tag = f'{tag}_ub'
        if cfg.use_pass2_alignment:
            tag = f'{tag}_pass2'
        if cfg.seed_frame != int(cfg.test_frame):
            tag = f'{tag}_seed{cfg.seed_frame}'
        if int(cfg.target_n) != 90000:
            tag = f'{tag}_N{int(cfg.target_n)}'
        return os.path.join(DEFAULT_OUTPUT_ROOT, f'{cfg.scene}_{tag}')

    def _results(self):
        cfg = self.cfg
        m = self.test_metrics
        return {
            'scene': cfg.scene,
            'train_frames': list(map(int, cfg.train_frames)),
            'train_loops': list(map(int, cfg.train_loops)),
            'test_frame': int(cfg.test_frame),
            'held_out_loop': int(cfg.held_out_loop),
            'test_in_train': bool(self.test_in_train),
            'num_iters': int(cfg.num_iters),
            'best_iter': int(self.best_iter),
            'best_mean_train_cc': float(self.best_mean_train_cc),
            'init_test_cc': float(self.init_test_cc),
            'final_test_cc': float(self.final_test_cc),
            # Full RA metric set on the test frame (matches baselines' metrics.json).
            'final_test_cart_corr': float(m['cart_corr']),
            'final_test_cart_mse':  float(m['mse']),
            'final_test_cart_rmse': float(m['rmse']),
            'final_test_cart_psnr': float(m['psnr']),
            'final_test_cart_ssim': float(m['ssim']) if m['ssim'] is not None else None,
            'init_train_mean_cc': float(np.mean(self.init_train_ccs)),
            'final_train_mean_cc': float(self.final_train_mean),
            'init_train_std_cc': float(np.std(self.init_train_ccs)),
            'final_train_std_cc': float(np.std(self.final_train_ccs)),
            'n_train_samples': int(len(self.train_samples)),
            'edge_frames': self.diag['edge_frames'],
            'test_pose_mode': self.diag['test_pose_mode'],
            'alignment_source': 'pass-2' if cfg.use_pass2_alignment else 'pass-1',
            'seed_frame':       int(cfg.seed_frame),
            'mat_lr': float(cfg.mat_lr), 'rot_lr': float(cfg.rot_lr),
            'loss_type': cfg.loss_type,
            'target_n': int(cfg.target_n),
            'elapsed_s': float(self.train_elapsed),
            'reg_warm_iters': int(cfg.reg_warm_iters),
            'reg_densify_interval': int(cfg.reg_densify_interval),
            'reg_densify_split_frac': float(cfg.reg_densify_split_frac),
            'reg_densify_prune_frac': float(cfg.reg_densify_prune_frac),
            'reg_densify_until': int(cfg.reg_densify_until),
            'reg_densify_pos_jitter_m': float(cfg.reg_densify_pos_jitter_m),
            'reg_densify_mat_jitter': float(cfg.reg_densify_mat_jitter),
            'reg_densify_rounds_completed': int(self.n_densify_rounds),
            'learn_positions_lr': float(cfg.learn_positions_lr),
            'learn_positions_l2': float(cfg.learn_positions_l2),
        }

    def save(self):
        """Write results, history, the best model, renders and per-frame exports."""
        cfg = self.cfg
        output_dir = cfg.output_dir or self._default_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        results = self._results()
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        np.savez(os.path.join(output_dir, 'history.npz'),
                 iters=np.array([h['iter'] for h in self.history]),
                 loss=np.array([h['loss'] for h in self.history]),
                 mean_train_cc=np.array([h['mean_train_cc'] for h in self.history]))
        if self.best_state is not None:
            torch.save(self.best_state, os.path.join(output_dir, 'best_model.pt'))

        test_ra_polar_np = self.test_ra_polar.detach().cpu().numpy().astype(np.float32)
        test_ra_cart_np = self.test_ra_cart.detach().cpu().numpy().astype(np.float32)
        test_gt_cart_np = self.gt_ra_cart.detach().cpu().numpy().astype(np.float32)
        # Complex per-(TX,RX) range profile, the renderer's native output.
        # Range FFT magnitude gives |RA|; inverse range FFT recovers raw ADC.
        # Shape: (n_tx, n_rx, K_range_bins), dtype complex64.
        test_rp_complex_np = (self.rp_real + 1j * self.rp_imag).detach().cpu().numpy().astype(np.complex64)
        np.save(os.path.join(output_dir, 'rendered_test_ra_polar.npy'), test_ra_polar_np)
        np.save(os.path.join(output_dir, 'rendered_test_ra_cart.npy'), test_ra_cart_np)
        np.save(os.path.join(output_dir, 'gt_test_ra_cart.npy'), test_gt_cart_np)
        np.save(os.path.join(output_dir, 'rendered_test_rp_complex.npy'), test_rp_complex_np)
        _save_ra_pngs_torch(test_ra_cart_np, output_dir, 'rendered_ra', self.range_res,
                            f'3DPS test frame {cfg.test_frame} loop {cfg.held_out_loop}')
        _save_ra_pngs_torch(test_gt_cart_np, output_dir, 'gt_ra', self.range_res,
                            f'GT test frame {cfg.test_frame} loop {cfg.held_out_loop}')

        # Per-train-frame renders, GT, PNGs and metrics_train.json.
        _save_train_frames_export(
            output_dir=output_dir,
            scene=cfg.scene,
            train_samples=self.train_samples,
            rast=self.rast,
            model=self.model,
            vertex_areas=self.vertex_areas,
            active_mask=self.active_mask,
            sample_grid=self.sample_grid,
            range_res=self.range_res,
            render_kwargs=self.render_kwargs,
        )

        if cfg.verbose:
            print(f'\n  results saved to: {output_dir}')
        return results


def _save_ra_pngs_torch(ra_cart_np, out_dir, prefix, range_res, title_prefix):
    """Save dB + linear PNGs of a Cartesian RA magnitude image.

    Mirrors the pattern used by ``mm3DPS.evaluation.utils.visualization.
    save_ra_cartesian_png`` (matplotlib, plasma cmap by default — kept as
    'hot' for visual consistency with the baseline finalize PNGs).
    """
    from mm3DPS.evaluation.utils.visualization import save_ra_cartesian_png
    for scale in ('linear', 'dB'):
        save_ra_cartesian_png(
            ra_cart_np,
            os.path.join(out_dir, f'{prefix}_{scale}.png'),
            range_res=range_res, scale=scale,
            title=f'{title_prefix} ({scale})',
        )


def _save_train_frames_export(*, output_dir, scene, train_samples, rast, model,
                              render_kwargs=None,
                              vertex_areas, active_mask, sample_grid,
                              range_res):
    """For each train_sample, render at its pose and save the supplement-figure
    artefacts (rendered/GT cart + polar + dB/linear PNGs + per-frame
    ra_corr metrics). Aggregates across train frames into ``metrics_train.json``.
    """
    train_dir_root = os.path.join(output_dir, 'train_frames')
    os.makedirs(train_dir_root, exist_ok=True)

    per_frame_records = []
    ra_corr_per_frame = []

    for s in train_samples:
        f = int(s['frame_idx'])
        loop_idx = int(s['loop_idx'])
        # Folder: when multiple loops per frame are saved, the first loop wins
        # (later loops overwrite). For the standard --train_loops 0 default
        # this is exactly one entry per train frame.
        frame_dir = os.path.join(train_dir_root, f'frame_{f}')
        os.makedirs(frame_dir, exist_ok=True)

        apply_pose(rast, s['pose'])
        with torch.no_grad():
            _rk = render_kwargs if render_kwargs is not None else dict()
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                **_rk)
            ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
            gt_polar = s['gt_ra_polar']
            gt_cart_full = polar_to_cart_torch(gt_polar, sample_grid)

        rendered_polar_np = ra_polar.detach().cpu().numpy().astype(np.float32)
        rendered_cart_np = ra_cart.detach().cpu().numpy().astype(np.float32)
        gt_polar_np = gt_polar.detach().cpu().numpy().astype(np.float32)
        gt_cart_np = gt_cart_full.detach().cpu().numpy().astype(np.float32)

        np.save(os.path.join(frame_dir, 'rendered_ra_polar.npy'),
                rendered_polar_np)
        np.save(os.path.join(frame_dir, 'rendered_ra_cart.npy'),
                rendered_cart_np)
        np.save(os.path.join(frame_dir, 'gt_ra_polar_full.npy'), gt_polar_np)
        np.save(os.path.join(frame_dir, 'gt_ra_cart.npy'), gt_cart_np)
        # Per-train-frame complex range profile (renderer's native output).
        rp_complex_np = (rp_real + 1j * rp_imag).detach().cpu().numpy().astype(np.complex64)
        np.save(os.path.join(frame_dir, 'rendered_rp_complex.npy'), rp_complex_np)

        _save_ra_pngs_torch(rendered_cart_np, frame_dir, 'rendered_ra',
                            range_res, f'3DPS train frame {f} loop {loop_idx}')
        _save_ra_pngs_torch(gt_cart_np, frame_dir, 'gt_ra',
                            range_res, f'GT train frame {f} loop {loop_idx}')

        # Compute the full RA metric set via the SAME harness the baselines
        # use (mm3DPS.evaluation.utils.metrics.compute_cart_ra_metrics) so
        # numbers are bit-identical to baseline metrics.json.
        m = compute_cart_ra_metrics(gt_cart_np, rendered_cart_np)
        per_frame_metric = {
            'baseline': '3dps',
            'scene': scene,
            'frame': f,
            'loop_idx': loop_idx,
            'ra_corr':         m['cart_corr'],
            'cart_mse':        m['mse'],
            'cart_rmse':       m['rmse'],
            'cart_psnr':       m['psnr'],
            'cart_ssim':       m['ssim'],
            'range_profile_corr': None,
        }
        with open(os.path.join(frame_dir, 'metrics.json'), 'w') as f_:
            json.dump(per_frame_metric, f_, indent=2)
        per_frame_records.append(per_frame_metric)
        ra_corr_per_frame.append(per_frame_metric['ra_corr'])

    # Aggregate every per-frame metric across the train set.
    def _agg(key):
        vals = [r.get(key) for r in per_frame_records
                if r.get(key) is not None]
        return ({'mean': float(np.mean(vals)),
                  'std':  float(np.std(vals)),
                  'per_frame': vals}
                 if vals else None)
    agg = {
        'per_frame': per_frame_records,
        'n_train_frames': len(per_frame_records),
        'ra_corr':   _agg('ra_corr'),
        'cart_mse':  _agg('cart_mse'),
        'cart_rmse': _agg('cart_rmse'),
        'cart_psnr': _agg('cart_psnr'),
        'cart_ssim': _agg('cart_ssim'),
    }
    with open(os.path.join(output_dir, 'metrics_train.json'), 'w') as f_:
        json.dump(agg, f_, indent=2)
