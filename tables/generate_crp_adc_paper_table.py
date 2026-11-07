"""NeurIPS-style cross-method, cross-product results table for the main paper.

Single combined results table that replaces the per-scene RA tables (now in
the supplement). Headline message: only 3DPS produces correct CRP and ADC;
the three optical-NVS-derived baselines have nothing to report on those
columns and on Complex metrics — signaled by `--`.

Layout (12 columns, single row section per split with 4 method rows each):

  Method | RA cart (Corr PSNR SSIM RMSE)
         | CRP (Corr PSNR SSIM)
         | CRP Complex (|ρ|, σ_φ)
         | ADC (env ρ_|·|)
         | ADC Complex (σ_φ)            -- |ρ| is shown only under CRP since
                                           it's identical between CRP and ADC
                                           by Parseval's theorem (unitary FFT)

Method order: prior work first (RadarSplat, Radar Fields, DART), 3DPS at the
bottom in bold (NeurIPS convention).

Header-text highlighters: yellow on Magnitude / Env., blue on Complex.

Wrapped in \\resizebox{\\textwidth}{!}{...} to guarantee page-width fit.

Reads:
  - output/training/...                    (3DPS RA metrics)
  - baselines/{dart,radarsplat,radarfields}/...           (baseline RA)
  - output/crp_adc_eval/results.json                       (3DPS CRP/ADC)

Writes:
  output/tables/crp_adc_results.tex
"""

import argparse
import json
import math
import os
import sys

# Reuse the existing loaders from generate_tables.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_tables import (
    SCENES, METHODS,
    load_ours_test_metrics, load_ours_train_metrics,
    load_baseline_metrics, load_baseline_train_metrics,
)


METHOD_LABEL = {
    "ours":        "\\textbf{3DPS (Ours)}",
    "radarsplat":  "RadarSplat~\\cite{kung2025radarsplat}",
    "radarfields": "Radar Fields~\\cite{10.1145/3641519.3657510}",
    "dart":        "DART~\\cite{huang2024dart}",
}
# NeurIPS convention: prior work first, ours last (and bolded).
METHOD_ORDER = ("dart", "radarfields", "radarsplat", "ours")


CRP_ADC_RESULTS = "output/crp_adc_eval/results.json"
DEFAULT_OURS_DIR = "output/training"
DEFAULT_BASELINES_DIR = "baselines"
DEFAULT_OUT = "output/tables/crp_adc_results.tex"


# ---------------------------------------------------------------------------
# Aggregation across scenes
# ---------------------------------------------------------------------------

def _scene_mean(values):
    vs = [v for v in values
          if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (sum(vs) / len(vs)) if vs else None


def aggregate_ra_test(ours_dir: str, baselines_dir: str) -> dict:
    out = {m: {} for m in METHODS}
    for m in METHODS:
        per_scene = {}
        for sc in SCENES:
            if m == "ours":
                d = load_ours_test_metrics(sc, ours_dir)
            else:
                d = load_baseline_metrics(m, sc, baselines_dir)
            if d is not None:
                per_scene[sc] = d
        for k in ("cart_corr", "cart_psnr", "cart_ssim", "cart_rmse"):
            out[m][k] = _scene_mean([per_scene.get(sc, {}).get(k)
                                       for sc in SCENES])
    return out


def aggregate_ra_train(ours_dir: str, baselines_dir: str) -> dict:
    out = {m: {} for m in METHODS}
    for m in METHODS:
        per_scene = {}
        for sc in SCENES:
            if m == "ours":
                d = load_ours_train_metrics(sc, ours_dir)
            else:
                d = load_baseline_train_metrics(m, sc, baselines_dir)
            if d is None:
                continue
            per_scene[sc] = {k: v["mean"] for k, v in d.items()
                              if isinstance(v, dict)}
        for k in ("cart_corr", "cart_psnr", "cart_ssim", "cart_rmse"):
            out[m][k] = _scene_mean([per_scene.get(sc, {}).get(k)
                                       for sc in SCENES])
    return out


def aggregate_crp_adc(crp_adc_path: str) -> dict:
    """Pull train/test aggregate CRP and ADC from a single results.json.

    Used for both 3DPS (`output/crp_adc_eval/results.json`) and any future
    complex-output baselines — they share the schema produced by
    `mm3DPS.evaluation.eval_crp_adc.save_results`.

    Reported columns:
      Magnitude (RA-style):
        - mag_corr, mag_psnr, mag_ssim on independently min-max-normalized
          $|\\cdot|$, identical convention to the RA cart columns.
      Differential phase (phase-invariant):
        - dphi_r_rmse_deg : magnitude-weighted RMSE of phase difference
          between adjacent range / fast-time bins.
        - dphi_v_rmse_deg : same across the virtual-antenna axis.
        Random-phase reference: $\\pi/\\sqrt{3} \\approx 103.9^\\circ$.
        These are absolute-phase-invariant: invariant to per-VA rotations
        (\\Delta\\varphi_r) or per-range rotations (\\Delta\\varphi_v) of
        the predicted CRP, so they isolate the relative-phase content the
        magnitude-only training loss does not constrain.
    """
    with open(crp_adc_path) as f:
        r = json.load(f)
    agg = r["aggregate"]
    out = {"train": {}, "test": {}}
    for split in ("train", "test"):
        b = agg[split]
        out[split] = {
            "crp_corr":     b["crp"]["mag_corr"]["mean"],
            "crp_psnr":     b["crp"]["mag_psnr"]["mean"],
            "crp_ssim":     b["crp"]["mag_ssim"]["mean"],
            "crp_dphi_r":   b["crp"]["dphi_r_rmse_deg"]["mean"],
            "crp_dphi_v":   b["crp"]["dphi_v_rmse_deg"]["mean"],
            "adc_env":      b["adc"]["mag_corr"]["mean"],
            "adc_dphi_r":   b["adc"]["dphi_r_rmse_deg"]["mean"],
        }
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(v, spec):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return spec.format(v)


# 11 numeric columns of Tab 2 in left-to-right order. Each entry is
# (metric_key, source_dict, format_spec, higher_is_better).
# `source_dict` is 'ra' for the first four columns and 'crp_adc' for the rest.
TAB2_COLS = [
    ('cart_corr',   'ra',      '{:.3f}', True),
    ('cart_psnr',   'ra',      '{:.1f}', True),
    ('cart_ssim',   'ra',      '{:.3f}', True),
    ('cart_rmse',   'ra',      '{:.4f}', False),
    ('crp_corr',    'crp_adc', '{:.3f}', True),
    ('crp_psnr',    'crp_adc', '{:.1f}', True),
    ('crp_ssim',    'crp_adc', '{:.3f}', True),
    ('crp_dphi_r',  'crp_adc', '{:.1f}', False),
    ('crp_dphi_v',  'crp_adc', '{:.1f}', False),
    ('adc_env',     'crp_adc', '{:.3f}', True),
    ('adc_dphi_r',  'crp_adc', '{:.1f}', False),
]


def _column_best_second(per_method_value: dict, hib: bool, fmt_spec: str
                         ) -> tuple[set, set]:
    """Return (best_methods, second_methods) using display-rounded tie detection.

    Methods with value None are excluded from the competition.
    """
    cands = [(m, float(v)) for m, v in per_method_value.items() if v is not None]
    if not cands:
        return set(), set()
    keyed = [(m, fmt_spec.format(v)) for m, v in cands]

    def _val(s: str) -> float:
        return float(s)

    keyed.sort(key=lambda t: -_val(t[1]) if hib else _val(t[1]))
    best_key = keyed[0][1]
    best_set = {m for m, k in keyed if k == best_key}
    rest = [(m, k) for m, k in keyed if m not in best_set]
    if not rest:
        return best_set, set()
    second_key = rest[0][1]
    second_set = {m for m, k in rest if k == second_key}
    return best_set, second_set


def _make_section_rows(
    ra_section: dict,           # method -> ra_dict
    crp_adc_section: dict,      # method -> crp_adc_dict (or absent)
    methods: list,              # ordered list of method keys
    method_label_fn,            # callable: method -> displayed label
) -> list:
    """Build LaTeX rows for one section (Train Mean or Test Mean) with
    column-aware bold/underline decoration."""
    # 1. Resolve raw values per (method, column).
    raw: list[dict] = []
    for col_key, src, _, _ in TAB2_COLS:
        per_method = {}
        for m in methods:
            d = ra_section.get(m) if src == 'ra' else crp_adc_section.get(m)
            per_method[m] = d.get(col_key) if d is not None else None
        raw.append(per_method)
    # 2. Compute (best, second) per column.
    decoration = []
    for (col_key, src, fmt_spec, hib), per_method in zip(TAB2_COLS, raw):
        decoration.append(_column_best_second(per_method, hib, fmt_spec))
    # 3. Emit rows.
    rows = []
    for m in methods:
        cells = [method_label_fn(m)]
        for col_idx, (col_key, src, fmt_spec, _) in enumerate(TAB2_COLS):
            v = raw[col_idx][m]
            s = fmt(v, fmt_spec)
            best_set, second_set = decoration[col_idx]
            if s != '--':
                if m in best_set:
                    s = f'\\textbf{{{s}}}'
                elif m in second_set:
                    s = f'\\underline{{{s}}}'
            cells.append(s)
        rows.append(' & '.join(cells) + ' \\\\')
    return rows


def make_table(ra_train, ra_test, crp_adc) -> str:
    L = []
    L.append("% Auto-generated by tables/generate_crp_adc_paper_table.py")
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{\\textbf{Cross-method, cross-product results on six "
              "ColoRadar scenes (mean across scenes).} \\colorbox{magbg}{Magnitude}: "
              "Pearson correlation, PSNR, SSIM, RMSE on min-max-normalized $|\\cdot|$. "
              "\\colorbox{vrbg}{Rel.\\ phase}: magnitude-weighted RMSE of the differential "
              "phase across range ($\\Delta\\varphi_r$) and across virtual antennas "
              "($\\Delta\\varphi_v$); on ADC, $\\Delta\\varphi_t$ is the differential "
              "across fast-time samples. Random-phase reference: $103.9^\\circ$. "
              "These metrics are absolute-phase-invariant by construction so are unaffected "
              "by per-VA / per-range calibration drift the magnitude-only training loss does "
              "not constrain. \\textbf{Bold}: best per column (ties jointly). "
              "\\underline{Underline}: second-best. Per-scene breakdowns and standard "
              "deviations in supplement Tables~\\ref{tab:test_ra}--\\ref{tab:crp_adc_test}.}")
    L.append("\\label{tab:results}")
    L.append("\\setlength{\\tabcolsep}{2.5pt}")
    L.append("\\resizebox{\\textwidth}{!}{")
    # 1 (Method) + 4 (RA) + 3 (CRP mag) + 2 (CRP rel.phase) + 1 (ADC env) + 1 (ADC rel.phase) = 12 cols
    L.append("\\begin{tabular}{l | cccc | ccc | cc | c | c}")
    L.append("\\toprule")
    # Top header — domain group
    L.append("& \\multicolumn{4}{c|}{\\textbf{RA cart}} "
              "& \\multicolumn{5}{c|}{\\textbf{CRP}} "
              "& \\multicolumn{2}{c}{\\textbf{ADC}} \\\\")
    L.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-10}\\cmidrule(lr){11-12}")
    # Sub-header — metric family with highlighter on label text
    L.append("& \\multicolumn{4}{c|}{\\colorbox{magbg}{\\textbf{Magnitude}}} "
              "& \\multicolumn{3}{c|}{\\colorbox{magbg}{\\textbf{Magnitude}}} "
              "& \\multicolumn{2}{c|}{\\colorbox{vrbg}{\\textbf{Rel.\\ phase}}} "
              "& \\colorbox{magbg}{\\textbf{Env.}} "
              "& \\colorbox{vrbg}{\\textbf{Rel.\\ phase}} \\\\")
    # Metric names
    L.append("Method & "
              "Corr & PSNR & SSIM & RMSE & "
              "Corr & PSNR & SSIM & "
              "$\\Delta\\varphi_r$ & $\\Delta\\varphi_v$ & "
              "$\\rho_{|\\cdot|}$ & "
              "$\\Delta\\varphi_t$ \\\\")
    L.append("\\midrule")

    def _label_fn(m):
        # METHOD_LABEL already wraps "ours" in \textbf{...}; reuse as-is.
        return METHOD_LABEL[m]

    # ── Train Mean ──
    L.append("\\multicolumn{12}{l}{\\textbf{Train Mean}} \\\\")
    crp_train = {m: (crp_adc.get(m, {}).get("train") if isinstance(crp_adc, dict) else None)
                 for m in METHOD_ORDER}
    for line in _make_section_rows(ra_train, crp_train, METHOD_ORDER, _label_fn):
        L.append(line)
    L.append("\\midrule")

    # ── Test Mean ──
    L.append("\\multicolumn{12}{l}{\\textbf{Test Mean}} \\\\")
    crp_test = {m: (crp_adc.get(m, {}).get("test") if isinstance(crp_adc, dict) else None)
                for m in METHOD_ORDER}
    for line in _make_section_rows(ra_test, crp_test, METHOD_ORDER, _label_fn):
        L.append(line)

    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("}")
    L.append("\\end{table*}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir",        default=DEFAULT_OURS_DIR)
    ap.add_argument("--baselines_dir",   default=DEFAULT_BASELINES_DIR)
    ap.add_argument("--crp_adc_results", default=CRP_ADC_RESULTS)
    ap.add_argument("--output",          default=DEFAULT_OUT)
    args = ap.parse_args()

    ra_train = aggregate_ra_train(args.ours_dir, args.baselines_dir)
    ra_test = aggregate_ra_test(args.ours_dir, args.baselines_dir)
    # Per-method CRP/ADC blocks. Optical-NVS baselines (RadarSplat / Radar
    # Fields / DART) emit magnitude-only RA — no CRP/ADC entry. Only 3DPS
    # contributes complex output.
    crp_adc = {"ours": aggregate_crp_adc(args.crp_adc_results)}

    tex = make_table(ra_train, ra_test, crp_adc)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(tex)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
