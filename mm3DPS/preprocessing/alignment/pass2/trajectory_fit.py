"""Stage A of cascade alignment pass 2 — per-scene trajectory fit + outlier flagging.

For each training scene (9 frames), load the pass-1 aligned configs,
extract the 6 pose components (pos_x/y/z, bore_x/y/z), fit a robust
smoothed trajectory, and flag per-frame outliers by MAD residual.

Output: ``data/alignment_data/<scene>/cascade/pass2_triage.json``:

    {
      "scene": "seq_0_frame_135",
      "frame_indices": [131, 132, ..., 139],
      "pass1_pose": { <per-frame raw pose from pass-1 winner config> },
      "smoothed_pose": { <per-frame LOWESS/Theil-Sen-smoothed pose> },
      "residuals_mad": { <per-frame per-component residual in MAD units> },
      "status": { "131": "ok", "132": "ok", ..., "135": "hard", ... },
      "outlier_summary": { "hard": [135, 393, 394], "soft": [...], ... },
      "params": { "lowess_frac": 0.5, ... }
    }

Usage::

    python -m mm3DPS.preprocessing.alignment.pass2.trajectory_fit \
        --scene seq_0_frame_135  [--all]

No CUDA work in this stage — pure pose analysis from JSON configs.
"""
import os
import sys
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
from mm3DPS.config import DEFAULT_DATA_ROOT


# ---------------------------------------------------------------------------
# Pose extraction from pass-1 configs
# ---------------------------------------------------------------------------

def _pick_pass1_config(align_dir: str, frame_idx: int) -> Optional[str]:
    """Pick the pass-1 winner config for a frame.

    Preference order (matches how pass-1 wrote things):
        1. cascaded_frame_<F>_aligned.json         (universal "winner" copy)
        2. cascaded_frame_<F>_aligned_2dof.json    (renderer_2dof)
        3. cascaded_frame_<F>_aligned_gpu.json     (lidar_4dof)
    Falls through to None if none exist.
    """
    for suffix in ('_aligned', '_aligned_2dof', '_aligned_gpu'):
        p = os.path.join(align_dir, f'cascaded_frame_{frame_idx}{suffix}.json')
        if os.path.isfile(p):
            return p
    return None


def load_scene_poses(scene: str, data_root: str = DEFAULT_DATA_ROOT
                     ) -> Tuple[List[int], Dict[int, dict]]:
    """Return (sorted_frame_indices, {f: pose_dict, ...}) for a scene.

    pose_dict has keys:
        'tx_pos_mm'  : (n_tx, 3) list of positions
        'rx_pos_mm'  : (n_rx, 3) list of positions
        'boresight'  : (3,) unit boresight (mean across elements, renormalised)
        'center_mm'  : (3,) mean of tx+rx positions
        'config_path': path to the pass-1 winner config file
        'winner'     : pass-1 winner label, if available
    """
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    if not os.path.isdir(align_dir):
        raise FileNotFoundError(f'no alignment_data dir for scene {scene}')

    # Discover frames by looking at alignment log files (1 per frame)
    logs = sorted(glob.glob(os.path.join(align_dir, 'cascaded_frame_*_alignment_log.json')))
    frames = []
    for p in logs:
        base = os.path.basename(p)
        core = base.replace('cascaded_frame_', '').replace('_alignment_log.json', '')
        try:
            frames.append(int(core))
        except ValueError:
            pass
    frames = sorted(set(frames))

    poses = {}
    for f in frames:
        cfg_path = _pick_pass1_config(align_dir, f)
        if cfg_path is None:
            continue
        with open(cfg_path) as fh:
            d = json.load(fh)
        tx_pos = np.array([e['pos_mm']   for e in d['tx_array']], dtype=float)
        rx_pos = np.array([e['pos_mm']   for e in d['rx_array']], dtype=float)
        # All elements share a boresight in pass-1 — take tx_array[0].
        bore = np.array(d['tx_array'][0]['boresight'], dtype=float)
        bore = bore / max(np.linalg.norm(bore), 1e-9)
        center = np.vstack([tx_pos, rx_pos]).mean(axis=0)

        # Pull winner label if available
        winner = None
        log_path = os.path.join(align_dir, f'cascaded_frame_{f}_alignment_log.json')
        if os.path.isfile(log_path):
            try:
                winner = json.load(open(log_path)).get('winner')
            except Exception:
                pass

        poses[f] = {
            'tx_pos_mm':   tx_pos.tolist(),
            'rx_pos_mm':   rx_pos.tolist(),
            'boresight':   bore.tolist(),
            'center_mm':   center.tolist(),
            'config_path': cfg_path,
            'winner':      winner,
        }
    return frames, poses


# ---------------------------------------------------------------------------
# Robust trajectory fit
# ---------------------------------------------------------------------------

def _median_abs_dev(residuals: np.ndarray) -> float:
    """Robust std estimate: MAD × 1.4826."""
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return float(mad * 1.4826 + 1e-12)


def _huber_weight(r_over_sigma: np.ndarray, c: float = 1.345) -> np.ndarray:
    """Huber weights: 1 inside |r|<=c, c/|r| outside."""
    w = np.ones_like(r_over_sigma, dtype=float)
    big = np.abs(r_over_sigma) > c
    w[big] = c / np.abs(r_over_sigma[big]).clip(min=1e-12)
    return w


def _lowess_1d(x: np.ndarray, y: np.ndarray, frac: float = 0.5,
               n_huber_iters: int = 2) -> np.ndarray:
    """Local linear regression with Huber-weighted residuals.

    Bandwidth ``frac``: each target point uses the nearest frac*N neighbours
    with tricube distance weights. After an initial LS fit, run ``n_huber_iters``
    rounds of IRLS with Huber weights on the standardised residuals so
    single-frame outliers are downweighted but not ignored.

    Returns a smoothed y-value for every input x.
    """
    n = len(x)
    k = max(3, int(np.ceil(frac * n)))  # at least 3 neighbours
    fitted = np.zeros_like(y, dtype=float)

    # Initial fit (tricube distance weights only)
    for i in range(n):
        d = np.abs(x - x[i])
        idx = np.argsort(d)[:k]
        d_max = max(d[idx].max(), 1e-9)
        u = np.clip(d[idx] / d_max, 0.0, 1.0)
        w = (1.0 - u ** 3) ** 3
        X = np.column_stack([np.ones(k), x[idx] - x[i]])
        W = np.diag(w)
        try:
            beta = np.linalg.solve(X.T @ W @ X + 1e-9 * np.eye(2), X.T @ W @ y[idx])
            fitted[i] = beta[0]
        except np.linalg.LinAlgError:
            fitted[i] = np.average(y[idx], weights=w)

    # Huber IRLS passes
    for _ in range(n_huber_iters):
        resid = y - fitted
        sigma = _median_abs_dev(resid)
        for i in range(n):
            d = np.abs(x - x[i])
            idx = np.argsort(d)[:k]
            d_max = max(d[idx].max(), 1e-9)
            u = np.clip(d[idx] / d_max, 0.0, 1.0)
            w_tri = (1.0 - u ** 3) ** 3
            w_hub = _huber_weight(resid[idx] / sigma)
            w = w_tri * w_hub
            X = np.column_stack([np.ones(k), x[idx] - x[i]])
            W = np.diag(w)
            try:
                beta = np.linalg.solve(X.T @ W @ X + 1e-9 * np.eye(2), X.T @ W @ y[idx])
                fitted[i] = beta[0]
            except np.linalg.LinAlgError:
                fitted[i] = np.average(y[idx], weights=w)
    return fitted


def fit_trajectory(frames: List[int], poses: Dict[int, dict],
                   lowess_frac: float = 0.5, n_huber_iters: int = 2
                   ) -> Dict[int, dict]:
    """Fit a smoothed trajectory and return per-frame smoothed poses.

    Smoothing is applied to 6 scalar time-series:
        center_x, center_y, center_z, bore_x, bore_y, bore_z

    The per-element TX/RX positions are reconstructed from the smoothed
    center by applying the frame's board-frame offsets from the raw config
    (i.e. we smooth only the rigid-body component; intra-board geometry is
    preserved exactly).

    Boresight is renormalised to unit length after per-component smoothing.
    """
    x = np.array(frames, dtype=float)

    centers = np.array([poses[f]['center_mm']  for f in frames])
    bores   = np.array([poses[f]['boresight']  for f in frames])

    cx_s = _lowess_1d(x, centers[:, 0], frac=lowess_frac, n_huber_iters=n_huber_iters)
    cy_s = _lowess_1d(x, centers[:, 1], frac=lowess_frac, n_huber_iters=n_huber_iters)
    cz_s = _lowess_1d(x, centers[:, 2], frac=lowess_frac, n_huber_iters=n_huber_iters)
    bx_s = _lowess_1d(x, bores[:, 0], frac=lowess_frac, n_huber_iters=n_huber_iters)
    by_s = _lowess_1d(x, bores[:, 1], frac=lowess_frac, n_huber_iters=n_huber_iters)
    bz_s = _lowess_1d(x, bores[:, 2], frac=lowess_frac, n_huber_iters=n_huber_iters)

    smoothed = {}
    for i, f in enumerate(frames):
        tx_pos = np.array(poses[f]['tx_pos_mm'])
        rx_pos = np.array(poses[f]['rx_pos_mm'])
        raw_center = np.array(poses[f]['center_mm'])
        new_center = np.array([cx_s[i], cy_s[i], cz_s[i]])
        delta = new_center - raw_center
        tx_smoothed = tx_pos + delta
        rx_smoothed = rx_pos + delta

        bore_s = np.array([bx_s[i], by_s[i], bz_s[i]])
        bore_s = bore_s / max(np.linalg.norm(bore_s), 1e-9)

        smoothed[f] = {
            'tx_pos_mm':  tx_smoothed.tolist(),
            'rx_pos_mm':  rx_smoothed.tolist(),
            'boresight':  bore_s.tolist(),
            'center_mm':  new_center.tolist(),
        }
    return smoothed


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------

def compute_residuals_mad(frames: List[int], poses: Dict[int, dict],
                          smoothed: Dict[int, dict]) -> Dict[int, dict]:
    """Per-frame per-component residuals expressed in MAD units.

    Residual std estimated robustly from the full scene (6 scalar series),
    one sigma per component. A frame is:
        hard  : any component has |r|/sigma > 3
        soft  : max |r|/sigma in [2, 3]
        ok    : max |r|/sigma < 2
    Also boresight-norm sanity: raw ||b|| outside [0.98, 1.02] → hard.
    """
    centers = np.array([poses[f]['center_mm']  for f in frames])
    bores   = np.array([poses[f]['boresight']  for f in frames])
    centers_s = np.array([smoothed[f]['center_mm']  for f in frames])
    bores_s   = np.array([smoothed[f]['boresight']  for f in frames])

    resid_c = centers - centers_s
    resid_b = bores   - bores_s

    sigma = np.array([
        _median_abs_dev(resid_c[:, 0]),
        _median_abs_dev(resid_c[:, 1]),
        _median_abs_dev(resid_c[:, 2]),
        _median_abs_dev(resid_b[:, 0]),
        _median_abs_dev(resid_b[:, 1]),
        _median_abs_dev(resid_b[:, 2]),
    ])

    # Put a floor on sigma so a trajectory that is perfectly smooth in one
    # component (e.g. all frames sharing a boresight z-consistency) doesn't
    # flag tiny 1e-5 wiggles as 10-sigma outliers.
    sigma_floor = np.array([1.0, 1.0, 1.0, 0.005, 0.005, 0.005])  # mm, unitless
    sigma = np.maximum(sigma, sigma_floor)

    out = {}
    for i, f in enumerate(frames):
        r_mad = np.array([
            resid_c[i, 0] / sigma[0],
            resid_c[i, 1] / sigma[1],
            resid_c[i, 2] / sigma[2],
            resid_b[i, 0] / sigma[3],
            resid_b[i, 1] / sigma[4],
            resid_b[i, 2] / sigma[5],
        ])
        bore_norm_raw = float(np.linalg.norm(bores[i]))

        max_abs = float(np.max(np.abs(r_mad)))
        if max_abs > 3.0 or bore_norm_raw < 0.98 or bore_norm_raw > 1.02:
            status = 'hard'
        elif max_abs > 2.0:
            status = 'soft'
        else:
            status = 'ok'

        out[f] = {
            'residuals_mad':   r_mad.tolist(),  # [cx, cy, cz, bx, by, bz]
            'max_abs_mad':     max_abs,
            'bore_norm_raw':   bore_norm_raw,
            'status':          status,
        }
    return out


# ---------------------------------------------------------------------------
# Pipeline for one scene
# ---------------------------------------------------------------------------

def run_stage_a(scene: str,
                data_root: str = DEFAULT_DATA_ROOT,
                lowess_frac: float = 0.5,
                n_huber_iters: int = 2,
                write_triage: bool = True,
                verbose: bool = True) -> dict:
    """Run Stage A for a single scene. Returns the triage dict."""
    frames, poses = load_scene_poses(scene, data_root=data_root)
    if len(frames) < 5:
        raise RuntimeError(f'{scene}: only {len(frames)} frames found (need >=5)')

    smoothed = fit_trajectory(frames, poses,
                              lowess_frac=lowess_frac,
                              n_huber_iters=n_huber_iters)
    triage = compute_residuals_mad(frames, poses, smoothed)

    outlier_summary = {
        'hard': [f for f in frames if triage[f]['status'] == 'hard'],
        'soft': [f for f in frames if triage[f]['status'] == 'soft'],
        'ok':   [f for f in frames if triage[f]['status'] == 'ok'],
    }

    result = {
        'scene': scene,
        'frame_indices': frames,
        'pass1_pose':    {str(f): poses[f]    for f in frames},
        'smoothed_pose': {str(f): smoothed[f] for f in frames},
        'triage':        {str(f): triage[f]   for f in frames},
        'outlier_summary': {
            'hard': list(map(int, outlier_summary['hard'])),
            'soft': list(map(int, outlier_summary['soft'])),
            'ok':   list(map(int, outlier_summary['ok'])),
        },
        'params': {
            'lowess_frac':   lowess_frac,
            'n_huber_iters': n_huber_iters,
        },
    }

    if verbose:
        print(f'\n=== {scene} — Stage A ===')
        print(f'  frames ({len(frames)}): {frames}')
        print(f'  outliers:   hard={outlier_summary["hard"]}  soft={outlier_summary["soft"]}')
        print(f'  per-frame status + max |r|/MAD:')
        for f in frames:
            tr = triage[f]
            r = tr['residuals_mad']
            tag = {'ok': '  ', 'soft': 'SO', 'hard': 'XX'}.get(tr['status'], '??')
            print(f'    [{tag}] f={f}: max={tr["max_abs_mad"]:+.2f}  '
                  f'r=[cx={r[0]:+.2f} cy={r[1]:+.2f} cz={r[2]:+.2f} '
                  f'bx={r[3]:+.2f} by={r[4]:+.2f} bz={r[5]:+.2f}]')

    if write_triage:
        out_path = os.path.join(data_root, 'alignment_data', scene, 'cascade',
                                'pass2_triage.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as fh:
            json.dump(result, fh, indent=2)
        if verbose:
            print(f'  triage written to: {out_path}')

    return result


def discover_scenes(data_root: str = DEFAULT_DATA_ROOT) -> List[str]:
    base = os.path.join(data_root, 'alignment_data')
    return sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d, 'cascade'))
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Pass-2 Stage A: trajectory fit + outlier flagging')
    ap.add_argument('--scene', default=None,
                    help='Specific scene to process; omit + pass --all for every scene')
    ap.add_argument('--all', action='store_true',
                    help='Process every scene under data/alignment_data/')
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--lowess_frac', type=float, default=0.5)
    ap.add_argument('--huber_iters', type=int, default=2)
    ap.add_argument('--dry_run', action='store_true',
                    help='Do not write pass2_triage.json')
    args = ap.parse_args()

    if args.scene and not args.all:
        scenes = [args.scene]
    elif args.all:
        scenes = discover_scenes(args.data_root)
    else:
        ap.error('specify --scene <name> or --all')

    all_results = {}
    for sc in scenes:
        try:
            res = run_stage_a(
                sc, data_root=args.data_root,
                lowess_frac=args.lowess_frac, n_huber_iters=args.huber_iters,
                write_triage=not args.dry_run, verbose=True)
            all_results[sc] = res['outlier_summary']
        except Exception as e:
            print(f'  {sc}: FAILED ({e})')

    print('\n======================================================================')
    print('SUMMARY — Stage A outlier counts per scene')
    print('======================================================================')
    print(f'{"scene":<22} {"ok":>3} {"soft":>5} {"hard":>5}  hard_frames')
    for sc, s in all_results.items():
        print(f'{sc:<22} {len(s["ok"]):>3} {len(s["soft"]):>5} '
              f'{len(s["hard"]):>5}  {s["hard"]}')
