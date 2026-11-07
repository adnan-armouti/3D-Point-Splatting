"""LaTeX tables for CRP / ADC product-agnostic evaluation.

Reads `output/crp_adc_eval/results.json` (produced by
`mm3DPS.evaluation.eval_crp_adc`) and emits:

  crp_adc_main_table.tex          — per-scene train + test means with the
                                     RA-style |·| metric set on one side and
                                     two complex-valued metric sets on the
                                     other (per-range $\\beta$ only vs.
                                     per-VA $\\alpha$ + per-range $\\beta$
                                     joint LS calibration removal).

  crp_adc_supplement_train.tex    — per-scene CRP/ADC training-view fidelity
                                     (each row = one scene, 8-train-frame mean).
                                     Mean row matches main paper
                                     Table~\\ref{tab:results} 3DPS Train Mean row.
  crp_adc_supplement_test.tex     — same structure for held-out test frames;
                                     Mean row matches Table~\\ref{tab:results}
                                     3DPS Test Mean row.

The thick vertical bars in every (split, domain) cell separate, in order:
    [ |·| metrics ] || [ complex per-range β only ] || [ complex VA + range ]
"""

import argparse
import json
import os
import statistics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scene_short(name: str) -> str:
    return name.replace("seq_", "S").replace("_frame_", "\\,F")


def _fmt(x, fmt=".3f"):
    if x is None:
        return "--"
    if isinstance(x, float) and (x != x):  # NaN
        return "--"
    return format(x, fmt)


def _u(d, key, scene_or_agg=False):
    """Unwrap mean from agg dict or read value directly from per-scene dict."""
    v = d[key]
    if isinstance(v, dict) and "mean" in v:
        return v["mean"]
    return v


# ---------------------------------------------------------------------------
# Block formatters
# ---------------------------------------------------------------------------

def _block_mag(m: dict, bold: bool = False) -> str:
    """|·| group: corr, PSNR, SSIM, MSE (4 cells)."""
    def f(x, fmt=".3f"):
        s = _fmt(x, fmt)
        return f"\\textbf{{{s}}}" if bold else s
    return (f"{f(_u(m, 'mag_corr'))} & "
            f"{f(_u(m, 'mag_psnr'), '.2f')} & "
            f"{f(_u(m, 'mag_ssim'))} & "
            f"{f(_u(m, 'mag_mse'), '.4f')}")


def _block_complex_crp(m: dict, suffix: str, bold: bool = False) -> str:
    """CRP complex group (image-style): |ρ|, PSNR_ℂ (2 cells in main)."""
    def f(x, fmt=".3f"):
        s = _fmt(x, fmt)
        return f"\\textbf{{{s}}}" if bold else s
    return (f"{f(_u(m, f'complex_corr{suffix}'))} & "
            f"{f(_u(m, f'complex_psnr{suffix}'), '.2f')}")




def _block_complex_adc(m: dict, suffix: str, bold: bool = False) -> str:
    """ADC complex group (radar time-series): |ρ|, phase RMSE (°)."""
    def f(x, fmt=".3f"):
        s = _fmt(x, fmt)
        return f"\\textbf{{{s}}}" if bold else s
    return (f"{f(_u(m, f'complex_corr{suffix}'))} & "
            f"{f(_u(m, f'phase_rmse_deg{suffix}'), '.1f')}")


# ---------------------------------------------------------------------------
# Main per-domain sub-table
# ---------------------------------------------------------------------------

def make_crp_subtable(per: dict, agg: dict) -> str:
    """CRP sub-table — image-style metrics (PSNR / SSIM make sense here).

    Layout per split: [ ρ, PSNR, SSIM, MSE ] || [ ρ_ℂ, PSNR_ℂ (R) ] || [ ρ_ℂ, PSNR_ℂ (VR) ]
    = 4 + 2 + 2 = 8 cells per split, 17 cols total.
    """
    scenes = list(per.keys())
    lines = []
    lines.append("% --- CRP sub-table (image-style metrics) ---")
    lines.append("\\setlength{\\tabcolsep}{2.5pt}")
    lines.append("\\begin{tabular}{l "
                  "cccc!{\\vrule width 1pt}cc!{\\vrule width 1pt}cc | "
                  "cccc!{\\vrule width 1pt}cc!{\\vrule width 1pt}cc}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{8}{c|}{\\textbf{Train (8 views)}} "
                  "& \\multicolumn{8}{c}{\\textbf{Test (held out)}} \\\\")
    lines.append("\\cmidrule(lr){2-9} \\cmidrule(lr){10-17}")
    lines.append(
        "& \\multicolumn{4}{c!{\\vrule width 1pt}}"
        "{$|\\mathrm{CRP}|$ (magnitude)} "
        "& \\multicolumn{2}{c!{\\vrule width 1pt}}"
        "{CRP complex (R)} "
        "& \\multicolumn{2}{c|}{CRP complex (VR)} "
        "& \\multicolumn{4}{c!{\\vrule width 1pt}}"
        "{$|\\mathrm{CRP}|$ (magnitude)} "
        "& \\multicolumn{2}{c!{\\vrule width 1pt}}"
        "{CRP complex (R)} "
        "& \\multicolumn{2}{c}{CRP complex (VR)} \\\\")
    lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} "
                  "\\cmidrule(lr){10-13} \\cmidrule(lr){14-15} \\cmidrule(lr){16-17}")
    mag_cols = "$\\rho$ & PSNR & SSIM & MSE"
    cmp_cols = "$|\\rho|$ & PSNR$_{\\mathbb{C}}$"
    lines.append(f"Scene & {mag_cols} & {cmp_cols} & {cmp_cols} "
                  f"& {mag_cols} & {cmp_cols} & {cmp_cols} \\\\")
    lines.append("\\midrule")
    for sc in scenes:
        tm = per[sc]["train_mean"]; ts = per[sc]["test"]
        lines.append(
            f"{_scene_short(sc)} & "
            f"{_block_mag(tm['crp'])} & "
            f"{_block_complex_crp(tm['crp'], '_R')} & "
            f"{_block_complex_crp(tm['crp'], '_VR')} & "
            f"{_block_mag(ts['crp'])} & "
            f"{_block_complex_crp(ts['crp'], '_R')} & "
            f"{_block_complex_crp(ts['crp'], '_VR')} \\\\"
        )
    lines.append("\\midrule")
    tr = agg["train"]["crp"]; te = agg["test"]["crp"]
    lines.append(
        "\\textbf{Mean} & "
        f"{_block_mag(tr, bold=True)} & "
        f"{_block_complex_crp(tr, '_R', bold=True)} & "
        f"{_block_complex_crp(tr, '_VR', bold=True)} & "
        f"{_block_mag(te, bold=True)} & "
        f"{_block_complex_crp(te, '_R', bold=True)} & "
        f"{_block_complex_crp(te, '_VR', bold=True)} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def make_adc_subtable(per: dict, agg: dict) -> str:
    """ADC sub-table — radar time-series metrics (no PSNR/SSIM).

    Layout per split: [ envelope ρ ] || [ |ρ|, phaseRMSE (R) ] || [ |ρ|, phaseRMSE (VR) ]
    = 1 + 2 + 2 = 5 cells per split, 11 cols total.
    """
    scenes = list(per.keys())
    lines = []
    lines.append("% --- ADC sub-table (radar time-series metrics) ---")
    lines.append("\\setlength{\\tabcolsep}{2.5pt}")
    lines.append("\\begin{tabular}{l "
                  "c!{\\vrule width 1pt}cc!{\\vrule width 1pt}cc | "
                  "c!{\\vrule width 1pt}cc!{\\vrule width 1pt}cc}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{5}{c|}{\\textbf{Train (8 views)}} "
                  "& \\multicolumn{5}{c}{\\textbf{Test (held out)}} \\\\")
    lines.append("\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}")
    lines.append(
        "& \\multicolumn{1}{c!{\\vrule width 1pt}}{envelope} "
        "& \\multicolumn{2}{c!{\\vrule width 1pt}}{ADC complex (R)} "
        "& \\multicolumn{2}{c|}{ADC complex (VR)} "
        "& \\multicolumn{1}{c!{\\vrule width 1pt}}{envelope} "
        "& \\multicolumn{2}{c!{\\vrule width 1pt}}{ADC complex (R)} "
        "& \\multicolumn{2}{c}{ADC complex (VR)} \\\\")
    lines.append("\\cmidrule(lr){2-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} "
                  "\\cmidrule(lr){7-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}")
    env = "$\\rho_{|\\cdot|}$"
    cmp_cols = "$|\\rho|$ & $\\sigma_\\phi$"
    lines.append(f"Scene & {env} & {cmp_cols} & {cmp_cols} "
                  f"& {env} & {cmp_cols} & {cmp_cols} \\\\")
    lines.append("\\midrule")
    for sc in scenes:
        tm = per[sc]["train_mean"]; ts = per[sc]["test"]
        env_tm = _fmt(_u(tm["adc"], "mag_corr"))
        env_ts = _fmt(_u(ts["adc"], "mag_corr"))
        lines.append(
            f"{_scene_short(sc)} & "
            f"{env_tm} & "
            f"{_block_complex_adc(tm['adc'], '_R')} & "
            f"{_block_complex_adc(tm['adc'], '_VR')} & "
            f"{env_ts} & "
            f"{_block_complex_adc(ts['adc'], '_R')} & "
            f"{_block_complex_adc(ts['adc'], '_VR')} \\\\"
        )
    lines.append("\\midrule")
    tr = agg["train"]["adc"]; te = agg["test"]["adc"]
    env_tr = f"\\textbf{{{_fmt(_u(tr, 'mag_corr'))}}}"
    env_te = f"\\textbf{{{_fmt(_u(te, 'mag_corr'))}}}"
    lines.append(
        "\\textbf{Mean} & "
        f"{env_tr} & "
        f"{_block_complex_adc(tr, '_R', bold=True)} & "
        f"{_block_complex_adc(tr, '_VR', bold=True)} & "
        f"{env_te} & "
        f"{_block_complex_adc(te, '_R', bold=True)} & "
        f"{_block_complex_adc(te, '_VR', bold=True)} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def make_caption() -> str:
    """LaTeX caption — explains the metric families and correction modes."""
    return (
        "\\caption{\\textbf{Product-agnostic CRP/ADC evaluation, train (mean of 8 "
        "views) and held-out test.} GT range FFT uses a Hann window (matching "
        "the trainer's loss-domain GT in \\texttt{adc\\_to\\_ra\\_complex}); near-"
        "field range bins $[0{:}15]$ are zeroed in both GT and predicted CRPs. "
        "\\textbf{(a) CRP} uses image-style metrics (CRP is a 2D azimuth$\\times$range tensor "
        "of weakly-correlated cells): $\\rho$ is Pearson on $|\\cdot|_{\\mathrm{norm}}$, "
        "PSNR/SSIM/MSE on independent min-max normalized $|\\cdot|$ "
        "(matches \\texttt{compute\\_cartesian\\_ra\\_metrics}); "
        "$|\\rho|$ and PSNR$_{\\mathbb{C}}$ are the complex analogs on unit-peak-"
        "normalized signals. "
        "\\textbf{(b) ADC} uses the radar-standard time-series metric set "
        "(no PSNR/SSIM, which are image-domain): "
        "envelope $\\rho_{|\\cdot|}$, complex correlation $|\\rho|$ after optimal "
        "scalar gain matching, and magnitude-weighted phase RMSE $\\sigma_\\phi$ "
        "in degrees. "
        "Each domain reports two phase-correction modes, separated by thick vertical "
        "bars within each split: \\textbf{R} -- per-range $\\beta[r]$ correction at "
        "the GT azimuth-DC bin (256 DOFs); \\textbf{VR} -- joint per-VA $\\alpha[v]$ + "
        "per-range $\\beta[r]$ LS calibration removal (342 DOFs $\\approx$ 1.5\\% "
        "of phase content, modelling per-channel RF and per-range timing).}\n")


def make_main_table(results: dict) -> str:
    """Two stacked sub-tables (CRP then ADC), one shared caption."""
    per = results["per_scene"]
    agg = results["aggregate"]

    lines = []
    lines.append("% Auto-generated by tables/generate_crp_adc_table.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(make_caption())
    lines.append("\\label{tab:crp_adc_main}")
    lines.append("")
    lines.append("\\textbf{(a) CRP results} (image-style metrics).\\\\[2pt]")
    lines.append(make_crp_subtable(per, agg))
    lines.append("")
    lines.append("\\vspace{4pt}")
    lines.append("")
    lines.append("\\textbf{(b) ADC results} ($=\\mathrm{IFFT}_r$ of VR-corrected "
                  "near-field-masked CRP; radar time-series metrics).\\\\[2pt]")
    lines.append(make_adc_subtable(per, agg))
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Supplement table — all metrics including MSE_C
# ---------------------------------------------------------------------------





def make_per_scene_split_table(results: dict, split: str,
                                method_label: str = "3DPS (Ours)",
                                label_suffix: str = "") -> str:
    """Per-scene CRP/ADC fidelity for one split (train or test).

    Same row structure as supplement Tables 5/6 (RA per-scene): one row per
    scene with the scene's mean (8-train-view mean for `train`; held-out
    test value for `test`), plus a final Mean row that matches the
    corresponding row of the main paper Table~\\ref{tab:results}.

    Single method per table — by default 3DPS, but can be re-pointed at any
    method whose `results.json` follows the eval_crp_adc schema (e.g. the differentiable renderer
    via `--results_json output/crp_adc_eval/baselines/_legacy/results.json`).
    Same column layout as main Table~\\ref{tab:results}: CRP magnitude
    (Corr/PSNR/SSIM/MSE) || CRP Complex (|ρ|, σ_φ) || ADC envelope || ADC
    Complex (σ_φ).
    """
    assert split in ("train", "test")
    pretty_split = "training-view" if split == "train" else "held-out test"
    label = f"tab:crp_adc_{split}{label_suffix}"
    per = results["per_scene"]
    agg = results["aggregate"][split]
    block_key = "train_mean" if split == "train" else "test"

    L = []
    L.append(f"% Auto-generated supplement table for {split} CRP/ADC.")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\small")
    L.append(
        f"\\caption{{\\textbf{{Per-scene CRP and ADC fidelity ({pretty_split}), "
        f"{method_label} only.}} Mean row anchors the {method_label} {split} "
        f"row of main paper Table~\\ref{{tab:results}}. Std is across the six scenes. "
        f"\\colorbox{{magbg}}{{\\textbf{{Magnitude}}}} columns: Pearson correlation, "
        f"PSNR, SSIM, RMSE on min-max-normalized $|\\cdot|$. "
        f"\\colorbox{{vrbg}}{{\\textbf{{Rel.\\ phase}}}} columns: magnitude-weighted RMSE "
        f"of the differential phase across range / virtual antennas / fast-time, in degrees "
        f"(random reference: $103.9^\\circ$).}}"
    )
    L.append(f"\\label{{{label}}}")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    # 1 (Scene) + 4 (CRP mag) + 2 (CRP rel.phase) + 1 (ADC env) + 1 (ADC rel.phase) = 9 cols
    L.append("\\begin{tabular}{l | cccc | cc | c | c}")
    L.append("\\toprule")
    L.append("& \\multicolumn{6}{c|}{\\textbf{CRP}} "
              "& \\multicolumn{2}{c}{\\textbf{ADC}} \\\\")
    L.append("\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}")
    L.append("& \\multicolumn{4}{c|}{\\colorbox{magbg}{\\textbf{Magnitude}}} "
              "& \\multicolumn{2}{c|}{\\colorbox{vrbg}{\\textbf{Rel.\\ phase}}} "
              "& \\colorbox{magbg}{\\textbf{Env.}} "
              "& \\colorbox{vrbg}{\\textbf{Rel.\\ phase}} \\\\")
    L.append("Scene & "
              "Corr & PSNR & SSIM & RMSE & "
              "$\\Delta\\varphi_r$ & $\\Delta\\varphi_v$ & "
              "$\\rho_{|\\cdot|}$ & "
              "$\\Delta\\varphi_t$ \\\\")
    L.append("\\midrule")

    def _row(label_cell, c, a, bold=False):
        def f(x, fmt=".3f"):
            s = _fmt(x, fmt)
            return f"\\textbf{{{s}}}" if bold else s
        return (
            f"{label_cell} & "
            f"{f(_u(c, 'mag_corr'))} & {f(_u(c, 'mag_psnr'), '.1f')} & "
            f"{f(_u(c, 'mag_ssim'))} & {f(_u(c, 'mag_rmse'), '.4f')} & "
            f"{f(_u(c, 'dphi_r_rmse_deg'), '.1f')} & {f(_u(c, 'dphi_v_rmse_deg'), '.1f')} & "
            f"{f(_u(a, 'mag_corr'))} & "
            f"{f(_u(a, 'dphi_r_rmse_deg'), '.1f')} \\\\"
        )

    for sc, d in per.items():
        crp = d[block_key]["crp"]
        adc = d[block_key]["adc"]
        L.append(_row(_scene_short(sc), crp, adc))
    L.append("\\midrule")
    L.append(_row("\\textbf{Mean}", agg["crp"], agg["adc"], bold=True))

    # Std row across the six scenes.
    def _std(sub, key):
        vals = []
        for sc, d in per.items():
            v = _u(d[block_key][sub], key)
            if v is not None and not (isinstance(v, float) and v != v):
                vals.append(float(v))
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    L.append(
        f"\\textbf{{Std}}  & "
        f"{_std('crp', 'mag_corr'):.3f} & "
        f"{_std('crp', 'mag_psnr'):.2f} & "
        f"{_std('crp', 'mag_ssim'):.3f} & "
        f"{_std('crp', 'mag_rmse'):.4f} & "
        f"{_std('crp', 'dphi_r_rmse_deg'):.2f} & "
        f"{_std('crp', 'dphi_v_rmse_deg'):.2f} & "
        f"{_std('adc', 'mag_corr'):.3f} & "
        f"{_std('adc', 'dphi_r_rmse_deg'):.2f} \\\\"
    )
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"




# ---------------------------------------------------------------------------
# Round-trip note
# ---------------------------------------------------------------------------

def make_round_trip_note(results: dict) -> str:
    san = results["aggregate"]["sanity"]
    crp = san["crp_max_rel_err"]["max"]
    adc = san["adc_max_rel_err"]["max"]
    return (f"% FFT round-trip across scenes: max rel error "
            f"{crp:.1e} (CRP), {adc:.1e} (ADC) — float64 machine precision.\n"
            "% R  = per-range β phase correction only (azimuth-DC bin scheme;"
            " 256 DOFs removed).\n"
            "% VR = joint per-VA α + per-range β LS correction "
            "(342 DOFs total ≈ 1.5%% of phase content).\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json",
                     default="output/crp_adc_eval/results.json")
    ap.add_argument("--output_dir",
                     default="output/crp_adc_eval/tables")
    ap.add_argument("--method_label", default="3DPS (Ours)",
                     help="Method label for caption + LaTeX label suffix. "
                          "Pass 'the differentiable renderer' when generating the differentiable renderer's parallel tables.")
    ap.add_argument("--label_suffix", default="",
                     help="Suffix appended to the LaTeX label (e.g. '_baseline' "
                          "to avoid collisions with the 3DPS table).")
    ap.add_argument("--filename_prefix", default="crp_adc",
                     help="Prefix for output filenames (default 'crp_adc'; "
                          "use 'crp_adc_baseline' for the parallel the differentiable renderer run).")
    args = ap.parse_args()

    with open(args.results_json) as f:
        results = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    note = make_round_trip_note(results)

    main_path = os.path.join(args.output_dir,
                              f"{args.filename_prefix}_main_table.tex")
    with open(main_path, "w") as f:
        f.write(note + make_main_table(results))
    print(f"Wrote {main_path}")

    # Two supplement tables: one per split (train, test). Each row is a
    # scene; each scene's value is its mean over all train frames (for the
    # train table) or its held-out test value (for the test table). Mean
    # row matches the corresponding row in main paper Table~\ref{tab:results}.
    for split in ("train", "test"):
        out = os.path.join(args.output_dir,
                            f"{args.filename_prefix}_supplement_{split}.tex")
        with open(out, "w") as f:
            f.write(note + make_per_scene_split_table(
                results, split,
                method_label=args.method_label,
                label_suffix=args.label_suffix,
            ))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
