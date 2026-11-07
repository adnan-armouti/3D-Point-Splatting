"""Pass-2 cascade alignment driver (CUDA renderer + trajectory prior).

For each frame flagged by Stage A (``trajectory_fit.py``):

    1. Load the CUDA alignment context once (FPS'd model + GT RA).
    2. Build a synthetic "prior base config" from the Stage-A smoothed
       pose. This centres the 4-DOF grid search on the trajectory.
    3. Run ``grid_search_4dof`` with a tight search radius and a prior
       penalty weight, followed by ``refine_4dof`` for local polish.
    4. Rescore the pass-1 lidar winner config (if any) with the same
       CUDA backend — second candidate for the winner-selection step.
    5. Evaluate each candidate against Stage-A MAD residuals
       (trajectory-consistency gate). A candidate with |r|>2.5 in any
       component is rejected.
    6. Pick the winner (highest adjusted score among gate-passers; if
       both fail, fall back to the smoothed prior pose itself).
    7. Write ``cascaded_frame_<F>_aligned_pass2.json`` and a per-frame
       log ``cascaded_frame_<F>_alignment_log_pass2.json`` that records
       both candidates' scores and the gate decision.

Frames marked ``ok`` by Stage A are passed through: the pass-1 winner
config is copied verbatim to ``*_aligned_pass2.json`` for downstream
consistency. Frames marked ``soft`` get the same treatment as hard
(re-aligned) — softs aren't outright wrong, but re-aligning with the
CUDA+prior path can only hurt if the gate rejects it (and then we fall
back).

Usage::

    python -m mm3DPS.preprocessing.alignment.pass2.run_pass2 \
        --scene seq_0_frame_135
    python -m mm3DPS.preprocessing.alignment.pass2.run_pass2 --all
"""
import os
import sys
import json
import time
import copy
import argparse
import numpy as np

import mitsuba as mi
if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm3DPS.config import DEFAULT_DATA_ROOT
from mm3DPS.preprocessing.alignment.cascaded_renderer_cuda import (
    build_alignment_context, apply_pose_to_ctx, render_and_evaluate_cuda,
    destroy_alignment_context,
)
from mm3DPS.preprocessing.alignment.cascaded_renderer import (
    apply_4dof_to_config, grid_search_4dof, refine_4dof,
    save_aligned_config_from_pose,
)
from mm3DPS.preprocessing.alignment.pass2.trajectory_fit import (
    run_stage_a, discover_scenes,
)

# LiDAR alignment seeded from the pass-1 pose.
from mm3DPS.preprocessing.alignment.cascaded_lidar_gpu import (
    optimize_alignment_gpu_with_prior,
)
from mm3DPS.preprocessing.alignment.cascaded_lidar import (
    load_point_cloud, load_radar_config, radar_gt_to_ra_map,
)
import open3d as o3d


# ---------------------------------------------------------------------------
# Synthetic base-config builder
# ---------------------------------------------------------------------------

def _load_unaligned_base_config(scene, frame_idx, data_root):
    p = os.path.join(data_root, scene, 'configs', f'cascaded_frame_{frame_idx}.json')
    with open(p) as f:
        return json.load(f)


def _make_prior_base_config(unaligned_cfg, prior_pose):
    """Build a synthetic base config whose pose IS the prior (smoothed).

    Preserves every non-pose field (radar params) from ``unaligned_cfg``.
    Replaces tx/rx pos_mm and boresights with the prior pose.
    """
    cfg = copy.deepcopy(unaligned_cfg)
    tx_pos = prior_pose['tx_pos_mm']
    rx_pos = prior_pose['rx_pos_mm']
    bore   = list(map(float, prior_pose['boresight']))
    for i, tx in enumerate(cfg['tx_array']):
        tx['pos_mm']    = list(map(float, tx_pos[i]))
        tx['boresight'] = bore
    for i, rx in enumerate(cfg['rx_array']):
        rx['pos_mm']    = list(map(float, rx_pos[i]))
        rx['boresight'] = bore
    return cfg


def _pass1_lidar_config(scene, frame_idx, data_root):
    """Return the path to the pass-1 lidar winner config if it exists,
    else None. Preference: _aligned_gpu > _aligned_gpu_dual > None."""
    cfgs_dir = os.path.join(data_root, scene, 'configs')
    for suffix in ('_aligned_gpu', '_aligned_gpu_aligned', '_aligned_gpu_dual_v2',
                   '_aligned_gpu_dual'):
        p = os.path.join(cfgs_dir, f'cascaded_frame_{frame_idx}{suffix}.json')
        if os.path.isfile(p):
            return p
    return None


def _load_pose_from_config(cfg_path):
    with open(cfg_path) as f:
        d = json.load(f)
    tx_pos = np.array([e['pos_mm'] for e in d['tx_array']], dtype=float)
    rx_pos = np.array([e['pos_mm'] for e in d['rx_array']], dtype=float)
    bore = np.array(d['tx_array'][0]['boresight'], dtype=float)
    bore = bore / max(np.linalg.norm(bore), 1e-9)
    return tx_pos, rx_pos, bore


# ---------------------------------------------------------------------------
# Trajectory-consistency gate
# ---------------------------------------------------------------------------

def _compute_scene_sigmas(triage):
    """Recover the per-component MAD sigma used by Stage A to score residuals.

    Stage A writes per-frame residuals in MAD units, so we recover sigma as
    (raw_residual / residual_in_mad). Raw residuals = pass1 pose - smoothed.
    """
    frames = triage['frame_indices']
    pass1 = triage['pass1_pose']
    smoothed = triage['smoothed_pose']
    stats = triage['triage']

    raws = []
    mads = []
    for f in frames:
        r_cent = (np.array(pass1[str(f)]['center_mm'])
                  - np.array(smoothed[str(f)]['center_mm']))
        r_bore = (np.array(pass1[str(f)]['boresight'])
                  - np.array(smoothed[str(f)]['boresight']))
        raw6 = np.concatenate([r_cent, r_bore])
        mad6 = np.array(stats[str(f)]['residuals_mad'])
        raws.append(raw6)
        mads.append(mad6)
    raws = np.array(raws)   # (nf, 6)
    mads = np.array(mads)   # (nf, 6)

    # sigma = raw / mad where mad != 0; median over non-zero entries per column.
    sigma = np.zeros(6)
    for c in range(6):
        mask = np.abs(mads[:, c]) > 1e-6
        if mask.any():
            sigma[c] = float(np.median(np.abs(raws[mask, c]) / np.abs(mads[mask, c])))
        else:
            sigma[c] = 1.0
    # Floor to Stage-A's sigma_floor so numerical wiggles don't blow up
    sigma_floor = np.array([1.0, 1.0, 1.0, 0.005, 0.005, 0.005])
    sigma = np.maximum(sigma, sigma_floor)
    return sigma


def _residuals_against_prior(tx_pos_mm, rx_pos_mm, bore, prior_pose, sigma):
    """Return (residual_mad_6vec, max_abs_mad) for a candidate pose."""
    center = np.vstack([tx_pos_mm, rx_pos_mm]).mean(axis=0)
    prior_center = np.array(prior_pose['center_mm'])
    prior_bore = np.array(prior_pose['boresight'])
    r_cent = center - prior_center
    r_bore = np.asarray(bore) - prior_bore
    r6 = np.concatenate([r_cent, r_bore])
    r_mad = r6 / sigma
    return r_mad.tolist(), float(np.max(np.abs(r_mad)))


# ---------------------------------------------------------------------------
# Per-frame alignment
# ---------------------------------------------------------------------------

def _load_scene_lidar_and_params(scene, frame_idx, data_root):
    """Load a lidar o3d PointCloud and radar params dict for the B1-L candidate.

    Returns ``(lidar_pcd, params)`` or ``(None, None)`` if loading failed.
    Scene mesh (mesh.ply) is the primary source; falls back to pcl.npy if
    the mesh lacks points.
    """
    data_dir = os.path.join(data_root, scene)
    mesh_path = os.path.join(data_dir, 'scene', 'mesh.ply')
    pcl_path = os.path.join(data_dir, 'scene', 'pcl.npy')
    base_cfg_path = os.path.join(data_dir, 'configs', f'cascaded_frame_{frame_idx}.json')
    try:
        pcd = load_point_cloud(mesh_path)
        if not pcd.has_points() and os.path.isfile(pcl_path):
            pcl = np.load(pcl_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
            if pcl.shape[1] >= 7:
                inten = pcl[:, 6].astype(np.float32)
                inten = (inten - inten.min()) / max(inten.max() - inten.min(), 1e-9)
                colors = np.stack([inten, inten, inten], axis=1)
                pcd.colors = o3d.utility.Vector3dVector(colors)
        params, _, _ = load_radar_config(base_cfg_path)
        return pcd, params
    except Exception as e:
        print(f'  [lidar-prior] skip (load failed: {e})')
        return None, None


def _lidar_prior_candidate(lidar_pcd, radar_params, prior_pose, gt_adc_path,
                           prior_weight, gate_mad_threshold, sigma,
                           search_radius_lidar=None, verbose=False):
    """Run optimize_alignment_gpu_with_prior and return a pass-2 candidate dict.

    Returns None if lidar data unavailable or the call fails.
    """
    if lidar_pcd is None or radar_params is None:
        return None
    try:
        ra_radar = radar_gt_to_ra_map(gt_adc_path)
        prior_bore = np.array(prior_pose['boresight'], dtype=np.float64)
        prior_origin = np.array(prior_pose['center_mm'], dtype=np.float64) / 1000.0
        res = optimize_alignment_gpu_with_prior(
            lidar_pcd, ra_radar, radar_params,
            prior_boresight=prior_bore,
            prior_origin=prior_origin,
            search_radius=search_radius_lidar,
            prior_weight=prior_weight,
            metric='correlation',
            coarse_steps=7,
            near_field_m=1.5,
            batch_size=100,
            verbose=False,
        )
    except Exception as e:
        if verbose:
            print(f'  [lidar-prior] FAILED: {e}')
        return None

    opt = res['optimal_params']
    # Build a synthetic prior_cfg to apply the 4-DOF deltas (reuses the same
    # apply_4dof_to_config math the renderer path uses — antennae move as a
    # rigid board around the prior pose).
    # We need an unaligned_cfg to copy the per-element tx_array/rx_array geometry
    # from. Caller passes us a pose; the caller must apply 4DOF via its own path.
    # To keep this function pure, we return the raw opt + raw cc.
    return {
        'label':       'lidar_4dof_pass2',
        'opt_4dof':    opt,                       # 4-DOF deltas (from prior)
        'cc':          float(res['best_metrics']['correlation']),
        'adj':         float(res['adjusted_score']),
        'initial_cc':  float(res['initial_metrics']['correlation']),
        'grid_raw_best_cc': float(res['grid_raw_best_cc']),
    }


def align_one_frame(
    scene,
    frame_idx,
    triage,
    data_root=DEFAULT_DATA_ROOT,
    search_radius=None,
    prior_weight=0.01,
    gate_mad_threshold=2.5,
    target_n=30000,
    lidar_pcd=None,
    radar_params=None,
    enable_lidar_prior=True,
    verbose=True,
):
    """Align a single flagged frame using CUDA renderer + trajectory prior.

    Returns a result dict with winner info; writes ``_aligned_pass2.json``
    and ``_alignment_log_pass2.json`` to disk.
    """
    search_radius = search_radius or {
        'range_m': 0.4, 'range_step': 0.1,       # ±0.4 m in 0.1 m steps  → 9 ticks
        'azim_deg': 2.0, 'azim_step': 1.0,       # ±2 deg in 1 deg steps  → 5 ticks
        'elev_rot_deg': 3.0, 'elev_rot_step': 1.0,  # ±3 deg × 1 deg        → 7 ticks
        'azim_rot_deg': 3.0, 'azim_rot_step': 1.0,  # ±3 deg × 1 deg        → 7 ticks
    }
    # default total grid = 9×5×7×7 = 2205 cells

    prior_pose = triage['smoothed_pose'][str(frame_idx)]
    frame_status = triage['triage'][str(frame_idx)]['status']
    sigma = _compute_scene_sigmas(triage)

    unaligned_cfg = _load_unaligned_base_config(scene, frame_idx, data_root)
    prior_cfg = _make_prior_base_config(unaligned_cfg, prior_pose)

    # -------- CUDA context ------------------------------------------------
    base_cfg_path = os.path.join(data_root, scene, 'configs',
                                 f'cascaded_frame_{frame_idx}.json')
    t0 = time.time()
    ctx = build_alignment_context(
        config_file=base_cfg_path, target_n=target_n, verbose=False)
    ctx_build_s = time.time() - t0

    candidates = []  # list of dicts {label, tx, rx, bore, cc, adj, mad_max, gate_pass}

    # Reference 1: prior pose itself (no optimisation)
    apply_pose_to_ctx(
        ctx,
        np.array(prior_pose['tx_pos_mm']),
        np.array(prior_pose['rx_pos_mm']),
        np.array(prior_pose['boresight']),
    )
    cc_prior = render_and_evaluate_cuda(ctx)
    candidates.append({
        'label':    'prior_only',
        'tx_pos_mm': prior_pose['tx_pos_mm'],
        'rx_pos_mm': prior_pose['rx_pos_mm'],
        'boresight': prior_pose['boresight'],
        'cc':       cc_prior,
        'adj':      cc_prior,  # zero penalty at delta=0
        'mad_max':  0.0,
        'gate_pass': True,
    })

    # Candidate 1: renderer-4DOF with prior
    t0 = time.time()
    grid_params, _grid_cc, _ = grid_search_4dof(
        ctx, prior_cfg,
        range_min=-search_radius['range_m'], range_max=search_radius['range_m'],
        range_step=search_radius['range_step'],
        az_min=-search_radius['azim_deg'],  az_max=search_radius['azim_deg'],
        az_step=search_radius['azim_step'],
        elev_rot_min=-search_radius['elev_rot_deg'],
        elev_rot_max=search_radius['elev_rot_deg'],
        elev_rot_step=search_radius['elev_rot_step'],
        azim_rot_min=-search_radius['azim_rot_deg'],
        azim_rot_max=search_radius['azim_rot_deg'],
        azim_rot_step=search_radius['azim_rot_step'],
        prior_weight=prior_weight, verbose=False)
    refined_params, refined_cc, _ = refine_4dof(
        ctx, prior_cfg, grid_params, max_evals=40,
        prior_weight=prior_weight, verbose=False)
    renderer_s = time.time() - t0
    tx_mm_r, rx_mm_r, bs_r = apply_4dof_to_config(prior_cfg, *refined_params)
    r_mad, r_mad_max = _residuals_against_prior(
        tx_mm_r, rx_mm_r, bs_r, prior_pose, sigma)
    candidates.append({
        'label':    'renderer_4dof_pass2',
        'tx_pos_mm': tx_mm_r.tolist(),
        'rx_pos_mm': rx_mm_r.tolist(),
        'boresight': bs_r.tolist(),
        'cc':       refined_cc,
        'adj':      refined_cc - prior_weight * (
            refined_params[0] ** 2
            + (refined_params[1] / 10.0) ** 2
            + refined_params[2] ** 2
            + refined_params[3] ** 2
        ),
        'delta_4dof': [float(v) for v in refined_params],
        'residuals_mad_vs_prior': r_mad,
        'mad_max': r_mad_max,
        'gate_pass': r_mad_max <= gate_mad_threshold,
        'time_s':   renderer_s,
    })

    # Candidate 2: lidar-4DOF re-optimised with trajectory prior (B1-L).
    # The lidar objective (voxel-RAE ↔ radar-RAE correlation) is different
    # from the renderer's (analytic BSDF ↔ radar cart RA), so it can find
    # gate-passing local optima the renderer missed — important for the
    # frames where renderer_4dof_pass2 drifted too far off-trajectory.
    if enable_lidar_prior:
        gt_adc_path = os.path.join(
            data_root, scene, 'radar', f'cascaded_frame_{frame_idx}.npy')
        _lidar_pcd = lidar_pcd
        _radar_params = radar_params
        if _lidar_pcd is None or _radar_params is None:
            _lidar_pcd, _radar_params = _load_scene_lidar_and_params(
                scene, frame_idx, data_root)
        lc = _lidar_prior_candidate(
            _lidar_pcd, _radar_params, prior_pose, gt_adc_path,
            prior_weight=prior_weight,
            gate_mad_threshold=gate_mad_threshold, sigma=sigma,
            verbose=verbose)
        if lc is not None:
            opt = lc['opt_4dof']
            tx_mm_lp, rx_mm_lp, bs_lp = apply_4dof_to_config(
                prior_cfg,
                opt['delta_range_m'], opt['delta_azimuth_deg'],
                opt['rotation_elev_deg'], opt['rotation_azim_deg'])
            # Rescore the lidar-optimised pose through the CUDA renderer so
            # its cc is directly comparable to the renderer_4dof_pass2
            # candidate (which uses the same analytic BSDF ↔ radar objective).
            apply_pose_to_ctx(ctx, tx_mm_lp, rx_mm_lp, bs_lp)
            cc_cuda_at_lidar = render_and_evaluate_cuda(ctx)
            adj_cuda_at_lidar = cc_cuda_at_lidar - prior_weight * (
                opt['delta_range_m'] ** 2
                + (opt['delta_azimuth_deg'] / 10.0) ** 2
                + opt['rotation_elev_deg'] ** 2
                + opt['rotation_azim_deg'] ** 2
            )
            r_mad_lp, r_mad_max_lp = _residuals_against_prior(
                tx_mm_lp, rx_mm_lp, bs_lp, prior_pose, sigma)
            candidates.append({
                'label':    lc['label'],
                'tx_pos_mm': tx_mm_lp.tolist(),
                'rx_pos_mm': rx_mm_lp.tolist(),
                'boresight': bs_lp.tolist(),
                'cc':       float(cc_cuda_at_lidar),         # CUDA cc (comparable)
                'adj':      float(adj_cuda_at_lidar),
                'lidar_voxel_cc': float(lc['cc']),            # lidar-native objective (diag)
                'lidar_initial_cc': float(lc['initial_cc']),
                'delta_4dof': [
                    float(opt['delta_range_m']),
                    float(opt['delta_azimuth_deg']),
                    float(opt['rotation_elev_deg']),
                    float(opt['rotation_azim_deg']),
                ],
                'residuals_mad_vs_prior': r_mad_lp,
                'mad_max':  r_mad_max_lp,
                'gate_pass': r_mad_max_lp <= gate_mad_threshold,
            })

    # Candidate 3: pass-1 lidar winner (rescored via CUDA, no re-optim)
    lidar_cfg_path = _pass1_lidar_config(scene, frame_idx, data_root)
    if lidar_cfg_path is not None:
        tx_l, rx_l, bs_l = _load_pose_from_config(lidar_cfg_path)
        apply_pose_to_ctx(ctx, tx_l, rx_l, bs_l)
        cc_lidar = render_and_evaluate_cuda(ctx)
        r_mad_l, r_mad_max_l = _residuals_against_prior(
            tx_l, rx_l, bs_l, prior_pose, sigma)
        candidates.append({
            'label':    'lidar_4dof_pass1_rescored',
            'tx_pos_mm': tx_l.tolist(),
            'rx_pos_mm': rx_l.tolist(),
            'boresight': bs_l.tolist(),
            'cc':       cc_lidar,
            'adj':      cc_lidar,  # no prior penalty (this candidate wasn't optimised by pass 2)
            'residuals_mad_vs_prior': r_mad_l,
            'mad_max':  r_mad_max_l,
            'gate_pass': r_mad_max_l <= gate_mad_threshold,
            'config_path': lidar_cfg_path,
        })

    # -------- Winner selection -------------------------------------------
    gate_pass = [c for c in candidates if c['gate_pass']]
    if not gate_pass:
        # Every candidate failed the gate — fall back to prior_only
        winner = [c for c in candidates if c['label'] == 'prior_only'][0]
        reason = 'all_candidates_off_trajectory'
    else:
        winner = max(gate_pass, key=lambda c: c['adj'])
        reason = 'best_adjusted_score_among_gate_passers'

    # -------- Write outputs ----------------------------------------------
    out_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    os.makedirs(out_dir, exist_ok=True)
    aligned_pass2_path = os.path.join(
        out_dir, f'cascaded_frame_{frame_idx}_aligned_pass2.json')
    save_aligned_config_from_pose(
        unaligned_cfg,
        np.array(winner['tx_pos_mm']),
        np.array(winner['rx_pos_mm']),
        np.array(winner['boresight']),
        aligned_pass2_path,
    )
    log = {
        'scene': scene, 'frame_idx': int(frame_idx),
        'status_stage_a':     frame_status,
        'winner_label':       winner['label'],
        'winner_cc':          float(winner['cc']),
        'winner_adj':         float(winner['adj']),
        'winner_mad_max':     float(winner['mad_max']),
        'selection_reason':   reason,
        'candidates':         candidates,
        'prior_weight':       prior_weight,
        'gate_mad_threshold': gate_mad_threshold,
        'search_radius':      search_radius,
        'ctx_build_s':        float(ctx_build_s),
        'aligned_pass2_path': aligned_pass2_path,
    }
    with open(os.path.join(out_dir,
                           f'cascaded_frame_{frame_idx}_alignment_log_pass2.json'),
              'w') as f:
        json.dump(log, f, indent=2)

    if verbose:
        # Abbreviations chosen to disambiguate the three lidar labels.
        _lbl_abbrev = {
            'prior_only':                'prior',
            'renderer_4dof_pass2':       'rend_pass2',
            'lidar_4dof_pass2':          'lid_pass2',
            'lidar_4dof_pass1_rescored': 'lid_pass1',
        }
        cands = ' | '.join(
            f"{_lbl_abbrev.get(c['label'], c['label']):>10s}: cc={c['cc']:.3f} mad={c['mad_max']:5.2f}"
            f"{'' if c['gate_pass'] else '(FAIL)'}"
            for c in candidates
        )
        print(f'  f={frame_idx} [{frame_status}] → {_lbl_abbrev.get(winner["label"], winner["label"])} '
              f'cc={winner["cc"]:.4f} (adj={winner["adj"]:.4f})    {cands}')

    destroy_alignment_context(ctx)
    return log


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

def run_pass2_for_scene(
    scene,
    data_root=DEFAULT_DATA_ROOT,
    prior_weight=0.01,
    gate_mad_threshold=2.5,
    target_n=30000,
    refit_stage_a=True,
    also_realign_soft=True,
    verbose=True,
):
    """Run pass 2 for a single scene. Writes per-frame pass-2 configs +
    a scene-level ``pass2_summary.json``.
    """
    triage_path = os.path.join(data_root, 'alignment_data', scene, 'cascade',
                               'pass2_triage.json')
    if refit_stage_a or not os.path.isfile(triage_path):
        if verbose:
            print(f'\n[{scene}] running Stage A (trajectory fit)...')
        triage = run_stage_a(scene, data_root=data_root, write_triage=True, verbose=False)
    else:
        with open(triage_path) as f:
            triage = json.load(f)

    frames = triage['frame_indices']
    if verbose:
        n_ok   = len(triage['outlier_summary']['ok'])
        n_soft = len(triage['outlier_summary']['soft'])
        n_hard = len(triage['outlier_summary']['hard'])
        print(f'[{scene}] frames={len(frames)}  ok={n_ok} soft={n_soft} hard={n_hard}')
        print(f'[{scene}] prior_weight={prior_weight}  gate={gate_mad_threshold} MAD')

    # Pre-load scene-level lidar + radar params once (reused across flagged
    # frames in this scene). The lidar pcd is scene-constant; params differ
    # slightly per frame's config file but only in radar geometry — within a
    # scene they match, so the first flagged frame's config is representative.
    lidar_pcd, radar_params = None, None
    flagged = [f for f in frames
               if triage['triage'][str(f)]['status'] in ('soft', 'hard')]
    if flagged:
        lidar_pcd, radar_params = _load_scene_lidar_and_params(
            scene, flagged[0], data_root)
        if verbose and lidar_pcd is not None:
            n_pts = len(np.asarray(lidar_pcd.points))
            print(f'[{scene}] lidar loaded: {n_pts:,} points '
                  f'(shared across {len(flagged)} flagged frames)')

    per_frame_logs = []
    for f in frames:
        status = triage['triage'][str(f)]['status']
        out_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
        aligned_pass2 = os.path.join(
            out_dir, f'cascaded_frame_{f}_aligned_pass2.json')

        if status == 'ok' or (status == 'soft' and not also_realign_soft):
            # Pass-through: copy pass-1 winner verbatim (from pass1_pose dict)
            pose = triage['pass1_pose'][str(f)]
            unaligned_cfg = _load_unaligned_base_config(scene, f, data_root)
            save_aligned_config_from_pose(
                unaligned_cfg,
                np.array(pose['tx_pos_mm']),
                np.array(pose['rx_pos_mm']),
                np.array(pose['boresight']),
                aligned_pass2,
            )
            per_frame_logs.append({
                'scene': scene, 'frame_idx': int(f),
                'status_stage_a': status,
                'winner_label': 'pass1_passthrough',
                'aligned_pass2_path': aligned_pass2,
            })
            if verbose:
                print(f'  f={f} [{status}] → pass-through (copy pass-1 pose)')
        else:
            log = align_one_frame(
                scene, f, triage, data_root=data_root,
                prior_weight=prior_weight,
                gate_mad_threshold=gate_mad_threshold,
                target_n=target_n,
                lidar_pcd=lidar_pcd, radar_params=radar_params,
                verbose=verbose)
            per_frame_logs.append(log)

    # Scene-level summary + B3 verification (re-fit Stage A on pass-2 poses)
    # We build a fake "pass1_pose" dict from the pass-2 aligned configs and
    # recompute residuals against the same smoothed trajectory. We keep the
    # smoothed trajectory fixed (Stage A's output) — not a new fit — because
    # iteratively refitting after each stage can spiral.
    out_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    summary = {
        'scene': scene,
        'frames': list(map(int, frames)),
        'per_frame': per_frame_logs,
        'prior_weight': prior_weight,
        'gate_mad_threshold': gate_mad_threshold,
    }
    # Compute post-fit verification: per-frame max MAD vs smoothed
    sigma = _compute_scene_sigmas(triage)
    post = {}
    for f in frames:
        path = os.path.join(out_dir, f'cascaded_frame_{f}_aligned_pass2.json')
        tx_p, rx_p, bs_p = _load_pose_from_config(path)
        r_mad, r_max = _residuals_against_prior(
            tx_p, rx_p, bs_p, triage['smoothed_pose'][str(f)], sigma)
        post[str(f)] = {'residuals_mad': r_mad, 'max_abs_mad': r_max,
                        'status': ('hard' if r_max > 3.0 else
                                   'soft' if r_max > 2.0 else 'ok')}
    summary['post_pass2_triage'] = post
    post_hard = [f for f in frames if post[str(f)]['status'] == 'hard']
    post_soft = [f for f in frames if post[str(f)]['status'] == 'soft']
    summary['post_outlier_summary'] = {
        'hard': list(map(int, post_hard)),
        'soft': list(map(int, post_soft)),
    }

    with open(os.path.join(out_dir, 'pass2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        pre_hard = triage['outlier_summary']['hard']
        pre_soft = triage['outlier_summary']['soft']
        print(f'[{scene}] done.')
        print(f'  pre  hard={pre_hard} soft={pre_soft}')
        print(f'  post hard={post_hard} soft={post_soft}')

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Pass-2 cascade alignment driver')
    ap.add_argument('--scene', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--prior_weight', type=float, default=0.01)
    ap.add_argument('--gate', type=float, default=2.5,
                    help='MAD threshold for the trajectory-consistency gate')
    ap.add_argument('--target_n', type=int, default=30000,
                    help='FPS target point count for alignment context')
    ap.add_argument('--skip_stage_a_refit', action='store_true',
                    help='Use existing pass2_triage.json instead of re-running Stage A')
    ap.add_argument('--no_realign_soft', action='store_true',
                    help='Pass-through soft-flagged frames instead of re-aligning')
    args = ap.parse_args()

    if args.scene and not args.all:
        scenes = [args.scene]
    elif args.all:
        scenes = discover_scenes(args.data_root)
    else:
        ap.error('specify --scene <name> or --all')

    t_total = time.time()
    for sc in scenes:
        run_pass2_for_scene(
            sc, data_root=args.data_root,
            prior_weight=args.prior_weight,
            gate_mad_threshold=args.gate,
            target_n=args.target_n,
            refit_stage_a=not args.skip_stage_a_refit,
            also_realign_soft=not args.no_realign_soft,
            verbose=True)
    print(f'\n[pass2] total elapsed: {time.time()-t_total:.1f}s')
