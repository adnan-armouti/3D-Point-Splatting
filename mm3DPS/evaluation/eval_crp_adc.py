"""CRP / ADC product-agnostic evaluation: per-scene driver and command line.

The phase correction and the metrics are in ``crp_adc_metrics``.
"""


import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from mm3DPS.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT
from mm3DPS.evaluation.crp_adc_metrics import (
    NEAR_FIELD_BINS,
    _compute_alt_phase_metrics,
    _compute_complex_only_metrics,
    _compute_diffphase_metrics,
    _compute_magnitude_metrics,
    _phase_correct_va_only,
    adc_txrx_to_vx_crp,
    apply_hann_az,
    crp_to_adc,
    near_field_mask_crp,
    phase_correct_per_range,
    phase_correct_va_range,
    round_trip_sanity_test,
    rp_txrx_to_vx_crp,
)


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Eval result for a single (scene, frame) pair."""
    frame: int
    crp: dict
    adc: dict
    sanity: dict
    # Arrays kept only when the caller needs them for figures (test frame).
    crp_pred_corr: Optional[np.ndarray] = None
    crp_gt: Optional[np.ndarray] = None
    adc_pred_corr: Optional[np.ndarray] = None
    adc_gt: Optional[np.ndarray] = None
    phi_star: Optional[np.ndarray] = None


@dataclass
class SceneResult:
    name: str
    test_frame: int
    test: FrameResult                    # held-out test view
    train: List[FrameResult]             # 8 train views


def evaluate_frame(
    frame: int,
    gt_adc_path: str,
    rendered_rp_path: str,
    held_out_loop: int = 0,
    start_bin: int = NEAR_FIELD_BINS,
    keep_arrays: bool = False,
) -> FrameResult:
    """Evaluate a single (scene, frame): load GT + rendered, phase-correct,
    mask near-field, compute metrics.

    `start_bin`: range bins [0:start_bin] are zeroed in BOTH GT and pred CRPs
    after phase correction (TX-RX coupling rejection). Set to 0 to disable.
    The masked CRPs are then IFFT_range'd to give near-field-suppressed ADCs.
    CRP metrics are computed only over bins [start_bin:] for both signals.
    `keep_arrays`: when True, retain the masked CRPs/ADCs on the result
    so the figure pipeline can read them. Set False for train frames to
    keep memory bounded.
    """
    # GT raw ADC: (n_loops, n_rx, n_tx, K) complex128
    gt_raw = np.load(gt_adc_path)
    assert gt_raw.ndim == 4, f"unexpected GT shape: {gt_raw.shape}"
    gt_adc_txrx = gt_raw[held_out_loop].transpose(1, 0, 2).astype(np.complex128)

    # Rendered CRP: (12, 16, 256) complex64
    rp_pred_txrx = np.load(rendered_rp_path).astype(np.complex128)
    assert rp_pred_txrx.shape == (12, 16, 256), (
        f"unexpected rendered shape: {rp_pred_txrx.shape}")

    sanity = round_trip_sanity_test(gt_adc_txrx)

    crp_gt_full = adc_txrx_to_vx_crp(gt_adc_txrx)
    crp_pred_raw = rp_txrx_to_vx_crp(rp_pred_txrx)

    # ── Two phase-correction modes (both reported side-by-side) ──
    # Mode R: per-range β only (azimuth-DC bin scheme; user's earlier choice)
    crp_pred_R_full, phi_star = phase_correct_per_range(crp_pred_raw,
                                                          crp_gt_full)
    # Mode VR: per-VA α + per-range β joint LS (calibration removal)
    crp_pred_VR_full, alpha_VR, beta_VR = phase_correct_va_range(
        crp_pred_raw, crp_gt_full, start_bin=start_bin)

    # ── Near-field mask (TX-RX coupling) applied to GT and both pred CRPs ──
    crp_gt_m = near_field_mask_crp(crp_gt_full, start_bin=start_bin)
    crp_pred_R_m = near_field_mask_crp(crp_pred_R_full, start_bin=start_bin)
    crp_pred_VR_m = near_field_mask_crp(crp_pred_VR_full, start_bin=start_bin)

    # ── Apply Hann_az on the VA axis to put metric in trainer's loss domain ──
    # The trainer's loss is on |FFT_az(Hann_az × CRP)|, so edge VAs (Hann≈0)
    # are unconstrained by training and predict noise. Without this windowing
    # those noisy edge cells drag |CRP| Pearson down even though |RA| is
    # unaffected. Apply identically to GT and pred for fair comparison.
    crp_gt_e = apply_hann_az(crp_gt_m)
    crp_pred_R_e = apply_hann_az(crp_pred_R_m)
    crp_pred_VR_e = apply_hann_az(crp_pred_VR_m)

    # ── ADCs from the (Hann_az × masked) CRPs ──
    adc_gt_vx = crp_to_adc(crp_gt_e)
    adc_pred_R_vx = crp_to_adc(crp_pred_R_e)
    adc_pred_VR_vx = crp_to_adc(crp_pred_VR_e)

    # ── Magnitude metrics ──
    # |CRP| is invariant to phase correction (rotation preserves magnitude
    # pointwise) — pick either pred. |ADC| IS NOT invariant: IFFT_range
    # mixes phase and magnitude so a different CRP phase pattern produces
    # a different |ADC|. We compute |ADC| on the VR-corrected ADC since
    # VR is the recommended mode (the magnitude after calibration removal).
    crp_mag = _compute_magnitude_metrics(
        crp_pred_VR_e[:, start_bin:], crp_gt_e[:, start_bin:])
    adc_mag = _compute_magnitude_metrics(adc_pred_VR_vx, adc_gt_vx)

    # ── Differential-phase metrics (phase-invariant relative phase) ──
    # Computed on the *uncorrected* (raw) CRP/ADC: differential phase is
    # invariant to per-VA and per-range rotations by construction, so the
    # R / VR calibration removal would have no effect. Using the raw signal
    # is the cleanest framing for the paper: "no calibration, no penalty
    # on absolute per-bin phase, only relative-phase fidelity".
    crp_pred_raw_e = apply_hann_az(near_field_mask_crp(crp_pred_raw,
                                                        start_bin=start_bin))
    adc_pred_raw_vx = crp_to_adc(crp_pred_raw_e)
    crp_dphi = _compute_diffphase_metrics(
        crp_pred_raw_e[:, start_bin:], crp_gt_e[:, start_bin:])
    adc_dphi = _compute_diffphase_metrics(adc_pred_raw_vx, adc_gt_vx)

    # ── Alternative phase-fidelity metrics (per-VA, per-range, coherence) ──
    crp_alt = _compute_alt_phase_metrics(
        crp_pred_raw_e[:, start_bin:], crp_gt_e[:, start_bin:])
    adc_alt = _compute_alt_phase_metrics(adc_pred_raw_vx, adc_gt_vx)

    # ── Per-VA-only ("V" mode) phase correction for headline-comparison ──
    # 86 DOFs removed (vs 256 for R, 342 for VR). Symmetric to R.
    crp_pred_V_full = _phase_correct_va_only(
        crp_pred_raw, crp_gt_full, start_bin=start_bin)
    crp_pred_V_e = apply_hann_az(near_field_mask_crp(
        crp_pred_V_full, start_bin=start_bin))
    adc_pred_V_vx = crp_to_adc(crp_pred_V_e)

    # ── Complex metrics under each correction mode (kept for back-compat) ──
    crp_raw = _compute_complex_only_metrics(
        crp_pred_raw_e[:, start_bin:], crp_gt_e[:, start_bin:])
    crp_V = _compute_complex_only_metrics(
        crp_pred_V_e[:, start_bin:], crp_gt_e[:, start_bin:])
    crp_R = _compute_complex_only_metrics(
        crp_pred_R_e[:, start_bin:], crp_gt_e[:, start_bin:])
    crp_VR = _compute_complex_only_metrics(
        crp_pred_VR_e[:, start_bin:], crp_gt_e[:, start_bin:])
    adc_raw = _compute_complex_only_metrics(adc_pred_raw_vx, adc_gt_vx)
    adc_V = _compute_complex_only_metrics(adc_pred_V_vx, adc_gt_vx)
    adc_R = _compute_complex_only_metrics(adc_pred_R_vx, adc_gt_vx)
    adc_VR = _compute_complex_only_metrics(adc_pred_VR_vx, adc_gt_vx)

    crp_metrics = {
        **crp_mag,
        **crp_dphi,
        **crp_alt,
        **{f"{k}_raw": v for k, v in crp_raw.items()},
        **{f"{k}_V":   v for k, v in crp_V.items()},
        **{f"{k}_R":   v for k, v in crp_R.items()},
        **{f"{k}_VR":  v for k, v in crp_VR.items()},
    }
    adc_metrics = {
        **adc_mag,
        **adc_dphi,
        **adc_alt,
        **{f"{k}_raw": v for k, v in adc_raw.items()},
        **{f"{k}_V":   v for k, v in adc_V.items()},
        **{f"{k}_R":   v for k, v in adc_R.items()},
        **{f"{k}_VR":  v for k, v in adc_VR.items()},
    }

    out = FrameResult(frame=frame, crp=crp_metrics, adc=adc_metrics,
                       sanity=sanity)
    if keep_arrays:
        # Save the VR-corrected, Hann_az-weighted, near-field-masked arrays
        # as the canonical "complex output" — these are the exact arrays
        # the metrics were computed on, so figures show what's evaluated.
        out.crp_pred_corr = crp_pred_VR_e.astype(np.complex64)
        out.crp_gt = crp_gt_e.astype(np.complex64)
        out.adc_pred_corr = adc_pred_VR_vx.astype(np.complex64)
        out.adc_gt = adc_gt_vx.astype(np.complex64)
        out.phi_star = phi_star.astype(np.float32)
    return out


def evaluate_scene(
    name: str,
    test_frame: int,
    train_frames: List[int],
    held_out_loop: int = 0,
    start_bin: int = NEAR_FIELD_BINS,
    ours_dir: Optional[str] = None,
    run_tag_template: Optional[str] = None,
) -> SceneResult:
    """Evaluate one scene: test view + 8 train views."""
    F = test_frame
    test_gt = _scene_gt_adc_path(name, F)
    test_rp = _scene_rendered_rp_test(name, F, ours_dir, run_tag_template)
    test_res = evaluate_frame(F, test_gt, test_rp,
                                held_out_loop=held_out_loop,
                                start_bin=start_bin, keep_arrays=True)

    train_results = []
    for f in train_frames:
        gt_p = _scene_gt_adc_path(name, f)
        rp_p = _scene_rendered_rp_train(name, F, f, ours_dir, run_tag_template)
        if not (os.path.isfile(gt_p) and os.path.isfile(rp_p)):
            print(f"    [skip train frame {f}] missing GT or pred")
            continue
        r = evaluate_frame(f, gt_p, rp_p, held_out_loop=held_out_loop,
                            start_bin=start_bin, keep_arrays=False)
        train_results.append(r)

    return SceneResult(name=name, test_frame=F, test=test_res,
                        train=train_results)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    # Magnitude — RA-style on min-max-normalized |·| (invariant to phase corr)
    "mag_corr", "mag_psnr", "mag_ssim", "mag_mse", "mag_rmse",
    # Differential-phase (absolute-phase-invariant relative phase fidelity).
    # Both forms reported: RMSE in degrees and magnitude-weighted ⟨cos(Δφ)⟩
    # in [-1, 1] (random reference 0). The coherence form is what the paper
    # tables and figure overlays use for headline numbers.
    "dphi_r_rmse_deg", "dphi_v_rmse_deg",
    "dphi_r_coherence", "dphi_v_coherence",
    # Alternative phase-fidelity metrics (per-VA |ρ| mean, per-range |ρ|
    # mean, magnitude-weighted phase coherence) — for reporting comparison.
    "rho_per_va_mean", "rho_per_range_mean", "phase_coherence",
    # Complex with NO phase correction (raw)
    "complex_corr_raw", "complex_psnr_raw", "complex_mse_raw", "phase_rmse_deg_raw",
    # Complex under per-VA α only ("V") — 86 DOFs
    "complex_corr_V", "complex_psnr_V", "complex_mse_V", "phase_rmse_deg_V",
    # Complex under per-range β only ("R") — azimuth-DC bin correction (256 DOFs)
    "complex_corr_R", "complex_psnr_R", "complex_mse_R", "phase_rmse_deg_R",
    # Complex under per-VA α + per-range β ("VR") — joint LS calibration removal (342 DOFs)
    "complex_corr_VR", "complex_psnr_VR", "complex_mse_VR", "phase_rmse_deg_VR",
)


def _mean_metrics(frames: List[FrameResult]) -> dict:
    """Mean of each metric across a list of FrameResult."""
    out = {"crp": {}, "adc": {}}
    for k in _METRIC_KEYS:
        cv = [f.crp[k] for f in frames if np.isfinite(f.crp[k])]
        av = [f.adc[k] for f in frames if np.isfinite(f.adc[k])]
        out["crp"][k] = {"mean": float(np.mean(cv)) if cv else float("nan"),
                          "std":  float(np.std(cv))  if cv else float("nan")}
        out["adc"][k] = {"mean": float(np.mean(av)) if av else float("nan"),
                          "std":  float(np.std(av))  if av else float("nan")}
    return out


def aggregate(results: List[SceneResult]) -> dict:
    """Aggregate train and test metrics across scenes.

    For TEST: mean / std taken over the 6 scene-level test values.
    For TRAIN: mean / std taken over the 6 scene-level train means
      (so each scene contributes equally regardless of train-frame count).
    """
    test_frames = [r.test for r in results]
    # Per-scene train mean (single FrameResult-like dict per scene)
    train_scene_means = []
    for r in results:
        m = _mean_metrics(r.train)
        # Repackage into FrameResult so _mean_metrics can be reused
        train_scene_means.append(FrameResult(
            frame=-1,
            crp={k: m["crp"][k]["mean"] for k in _METRIC_KEYS},
            adc={k: m["adc"][k]["mean"] for k in _METRIC_KEYS},
            sanity={"crp_max_rel_err": 0.0, "adc_max_rel_err": 0.0},
        ))

    out = {
        "test": _mean_metrics(test_frames),
        "train": _mean_metrics(train_scene_means),
        "sanity": {},
    }
    for k in ("crp_max_rel_err", "adc_max_rel_err"):
        vals = [r.test.sanity[k] for r in results]
        out["sanity"][k] = {"mean": float(np.mean(vals)),
                             "max":  float(np.max(vals))}
    return out


def save_results(results: List[SceneResult], output_dir: str) -> None:
    """Write per-scene metrics + arrays + aggregate JSON.

    Per-scene structure:
      {
        "test_frame": int,
        "sanity": dict,
        "test": {"crp": {...}, "adc": {...}},
        "train_frames": [{"frame": F, "crp": {...}, "adc": {...}}, ...],
        "train_mean": {"crp": {...}, "adc": {...}}
      }
    """
    os.makedirs(output_dir, exist_ok=True)
    per_scene = {}
    for r in results:
        scene_dir = os.path.join(output_dir, r.name)
        os.makedirs(scene_dir, exist_ok=True)
        # Test-frame arrays for the figure pipeline
        if r.test.crp_pred_corr is not None:
            np.save(os.path.join(scene_dir, "crp_pred_corr.npy"),
                     r.test.crp_pred_corr)
            np.save(os.path.join(scene_dir, "crp_gt.npy"), r.test.crp_gt)
            np.save(os.path.join(scene_dir, "adc_pred_corr.npy"),
                     r.test.adc_pred_corr)
            np.save(os.path.join(scene_dir, "adc_gt.npy"), r.test.adc_gt)
            np.save(os.path.join(scene_dir, "phi_star.npy"), r.test.phi_star)

        train_mean = _mean_metrics(r.train)
        # Strip the std for the simple "mean" view
        train_mean_simple = {
            "crp": {k: train_mean["crp"][k]["mean"] for k in _METRIC_KEYS},
            "adc": {k: train_mean["adc"][k]["mean"] for k in _METRIC_KEYS},
        }
        per_scene[r.name] = {
            "test_frame": r.test_frame,
            "sanity": r.test.sanity,
            "test": {"crp": r.test.crp, "adc": r.test.adc},
            "train_frames": [
                {"frame": f.frame, "crp": f.crp, "adc": f.adc}
                for f in r.train
            ],
            "train_mean": train_mean_simple,
            "train_mean_std": train_mean,  # full mean+std view
        }

    agg = aggregate(results)
    out = {"per_scene": per_scene, "aggregate": agg}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Convenience: scene path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "..", ".."))


_DEFAULT_OURS_DIR = DEFAULT_OUTPUT_ROOT
_DATA_ROOT = DEFAULT_DATA_ROOT
_DEFAULT_RUN_TAG_TEMPLATE = (
    "{scene}_train8frames_1loops_test{test_frame}_loop0_pass2_N20000"
)


def _scene_run_dir(scene: str, test_frame: int,
                   ours_dir: Optional[str] = None,
                   run_tag_template: Optional[str] = None) -> str:
    """Resolve the per-scene 3DPS run directory.

    The defaults read ``output/training/<scene>_train8frames_..._N20000``.
    The CLI exposes the overrides as ``--ours_dir`` and ``--run_tag``.
    """
    od = ours_dir if ours_dir is not None else _DEFAULT_OURS_DIR
    tmpl = (run_tag_template if run_tag_template is not None
            else _DEFAULT_RUN_TAG_TEMPLATE)
    return os.path.join(od, tmpl.format(scene=scene, test_frame=test_frame))


def _scene_gt_adc_path(scene: str, frame: int) -> str:
    """data/<scene>/radar/cascaded_frame_<F>.npy"""
    return os.path.join(_DATA_ROOT, scene, "radar",
                         f"cascaded_frame_{frame}.npy")


def _scene_rendered_rp_test(scene: str, test_frame: int,
                             ours_dir: Optional[str] = None,
                             run_tag_template: Optional[str] = None) -> str:
    return os.path.join(
        _scene_run_dir(scene, test_frame, ours_dir, run_tag_template),
        "rendered_test_rp_complex.npy")


def _scene_rendered_rp_train(scene: str, test_frame: int,
                               train_frame: int,
                               ours_dir: Optional[str] = None,
                               run_tag_template: Optional[str] = None) -> str:
    return os.path.join(
        _scene_run_dir(scene, test_frame, ours_dir, run_tag_template),
        "train_frames", f"frame_{train_frame}", "rendered_rp_complex.npy")


def test_frame_of(scene: str) -> int:
    return int(scene.rsplit("_", 1)[-1])


# Paper scenes (name, test_frame, train_frames)
PAPER_SCENES_FULL = [
    ("seq_0_frame_135", 135, [131, 132, 133, 134, 136, 137, 138, 139]),
    ("seq_1_frame_185", 185, [181, 182, 183, 184, 186, 187, 188, 189]),
    ("seq_1_frame_438", 438, [434, 435, 436, 437, 439, 440, 441, 442]),
    ("seq_2_frame_105", 105, [101, 102, 103, 104, 106, 107, 108, 109]),
    ("seq_2_frame_160", 160, [156, 157, 158, 159, 161, 162, 163, 164]),
    ("seq_2_frame_300", 300, [296, 297, 298, 299, 301, 302, 303, 304]),
]
PAPER_SCENES = [s[0] for s in PAPER_SCENES_FULL]
PAPER_SCENES_INFO = {s[0]: (s[1], s[2]) for s in PAPER_SCENES_FULL}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=PAPER_SCENES)
    ap.add_argument("--output_dir", default=os.path.join(
        PROJECT_ROOT, "output", "crp_adc_eval"))
    ap.add_argument("--sanity_only", action="store_true",
                    help="Only run round-trip sanity test on GT (no rendered).")
    ap.add_argument("--start_bin", type=int, default=NEAR_FIELD_BINS,
                    help=f"Near-field mask: zero CRP bins [0:start_bin] in both"
                          f" GT and pred (default {NEAR_FIELD_BINS}; pass 0 to"
                          f" disable masking).")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                    help="Prepared data directory (default: <repo>/data).")
    ap.add_argument("--ours_dir", default=None,
                    help="Directory holding the per-scene 3DPS run directories "
                         "(default: <repo>/output/training).")
    ap.add_argument("--run_tag", default=None,
                    help="Run-tag template inside --ours_dir; supports "
                         "{scene} and {test_frame} placeholders. Default "
                         f"'{_DEFAULT_RUN_TAG_TEMPLATE}'.")
    args = ap.parse_args()
    global _DATA_ROOT
    _DATA_ROOT = args.data_root

    if args.sanity_only:
        print("=== Round-trip sanity test (GT only) ===")
        for scene in args.scenes:
            F = test_frame_of(scene)
            gt_path = _scene_gt_adc_path(scene, F)
            if not os.path.isfile(gt_path):
                print(f"  [skip] {scene}: GT not found at {gt_path}")
                continue
            gt_raw = np.load(gt_path)
            adc_txrx = gt_raw[0].transpose(1, 0, 2).astype(np.complex128)
            s = round_trip_sanity_test(adc_txrx)
            print(f"  {scene}: "
                  f"CRP rel_err={s['crp_max_rel_err']:.2e}  "
                  f"ADC rel_err={s['adc_max_rel_err']:.2e}  "
                  f"(CRP mag={s['crp_max_mag']:.2e}  "
                  f"ADC mag={s['adc_max_mag']:.2e})")
        return

    print(f"=== CRP/ADC eval (train+test, {len(args.scenes)} scenes,  "
          f"start_bin={args.start_bin}) ===")
    results = []
    for scene in args.scenes:
        if scene not in PAPER_SCENES_INFO:
            print(f"  [skip] {scene}: not in PAPER_SCENES_INFO "
                  f"(unknown train_frames)")
            continue
        F, train_frames = PAPER_SCENES_INFO[scene]
        # quick sanity check that test rendered CRP exists
        if not os.path.isfile(_scene_rendered_rp_test(scene, F,
                                                       args.ours_dir,
                                                       args.run_tag)):
            print(f"  [skip] {scene}: rendered test CRP not found")
            continue
        print(f"  evaluating {scene} (test={F}, train={train_frames})...")
        r = evaluate_scene(scene, F, train_frames,
                            start_bin=args.start_bin,
                            ours_dir=args.ours_dir,
                            run_tag_template=args.run_tag)
        tcrp, tadc = r.test.crp, r.test.adc
        print(f"    test  CRP: corr={tcrp['mag_corr']:.3f}  "
              f"PSNR={tcrp['mag_psnr']:.2f}  SSIM={tcrp['mag_ssim']:.3f}  "
              f"||  VR: |ρ|={tcrp['complex_corr_VR']:.3f}  "
              f"σ_φ={tcrp['phase_rmse_deg_VR']:.1f}°")
        print(f"    test  ADC: env corr={tadc['mag_corr']:.3f}  "
              f"||  VR: |ρ|={tadc['complex_corr_VR']:.3f}  "
              f"σ_φ={tadc['phase_rmse_deg_VR']:.1f}°")
        if r.train:
            tm = _mean_metrics(r.train)
            print(f"    train CRP (mean): corr="
                  f"{tm['crp']['mag_corr']['mean']:.3f}  "
                  f"PSNR={tm['crp']['mag_psnr']['mean']:.2f}  "
                  f"|| VR: |ρ|={tm['crp']['complex_corr_VR']['mean']:.3f}  "
                  f"σ_φ={tm['crp']['phase_rmse_deg_VR']['mean']:.1f}°")
            print(f"    train ADC (mean): env corr="
                  f"{tm['adc']['mag_corr']['mean']:.3f}  "
                  f"|| VR: |ρ|={tm['adc']['complex_corr_VR']['mean']:.3f}  "
                  f"σ_φ={tm['adc']['phase_rmse_deg_VR']['mean']:.1f}°")
        results.append(r)

    if results:
        save_results(results, args.output_dir)
        agg = aggregate(results)
        print("\n=== Aggregate (across {} scenes) ===".format(len(results)))
        for split in ("test", "train"):
            print(f"  --- {split} ---")
            for dom in ("crp", "adc"):
                m = agg[split][dom]
                if dom == "crp":
                    # Image-style report for CRP (PSNR/SSIM make sense here)
                    print(f"    CRP  |.|  corr={m['mag_corr']['mean']:.3f}  "
                          f"PSNR={m['mag_psnr']['mean']:.2f}  "
                          f"SSIM={m['mag_ssim']['mean']:.3f}  "
                          f"|| VR  |ρ|={m['complex_corr_VR']['mean']:.3f}  "
                          f"phRMSE={m['phase_rmse_deg_VR']['mean']:.1f}°")
                else:
                    # Time-series report for ADC (no PSNR/SSIM)
                    print(f"    ADC  envelope corr={m['mag_corr']['mean']:.3f}  "
                          f"|| VR  |ρ|={m['complex_corr_VR']['mean']:.3f}  "
                          f"phRMSE={m['phase_rmse_deg_VR']['mean']:.1f}°")
        print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
