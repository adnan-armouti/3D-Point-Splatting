# Paper table scripts

Each script regenerates a table that appears in the paper. Scripts assume
they are invoked from the repo root with the mm3DPS package importable, your
training runs under `output/training/`, and the per-baseline
`results/<scene>/metrics.json` files produced by
`baselines/<name>/runner/run_all_scenes.py`.

## Tables (.tex)

| Output .tex                                             | Script                                |
|---------------------------------------------------------|---------------------------------------|
| `output/tables/test_ra_results.tex`                            | `generate_tables.py`                  |
| `output/tables/train_ra_results.tex`                           | `generate_tables.py`                  |
| `output/tables/crp_adc_results.tex`                            | `generate_crp_adc_paper_table.py`     |
| `output/crp_adc_eval/tables/*.tex`                     | `generate_crp_adc_table.py`           |
| `output/tables/ablations.tex`                                  | `generate_ablation_table.py`          |

## Inputs each script expects

- `output/training/<scene>_<tag>/best_model.pt`: your trained scenes.
- `baselines/<name>/results/<scene>/metrics.json` — produced by each baseline's
  `runner/run_all_scenes.py`. Generate these before running the comparison
  tables.
- `output/crp_adc_eval/<scene>/{crp,adc}_{pred_corr,gt}.npy` — produced by
  `mm3DPS.evaluation.eval_crp_adc` for the CRP/ADC product-agnostic
  evaluation. Generate this before running the CRP/ADC table scripts.

## Notes

- All scripts read paths via CLI flags; defaults assume the layout described
  above. Pass `--help` for the full set of overrides.
