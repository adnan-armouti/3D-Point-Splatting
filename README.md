# 3DPS: 3D Point Splatting for mmWave Radar Novel View Synthesis

> Code release for the NeurIPS 2026 paper.
> Project page: <https://3d-point-splatting.github.io>

## Abstract

3DPS introduces a differentiable, point-splatting renderer for the
77 GHz cascaded-radar Range–Azimuth (RA) imaging pipeline. Each scene
is represented as a per-point set of physically-grounded ITU material
parameters splatted onto a hemisphere; closed-form path integration
through a per-frame chirp-window FFT cascade yields range-profile
phasors at ~3 minutes per scene on a single RTX 4090. We evaluate on a
six-scene ColoRadar benchmark with a strict middle-frame-bracketed
8-train / 1-test held-out NVS protocol, and compare against three
recent radar-NVS baselines (DART, RadarFields, RadarSplat) re-trained
under an identical protocol. 3DPS attains the highest mean test-frame
RA Cartesian correlation (`cart_corr`) on the benchmark, training in about
3 minutes per scene.

---

## Requirements

- Python 3.12 (tested), CUDA 11.8 (tested, RTX 4090).
- PyTorch 2.7.1 (cu118), Mitsuba 3.7.1, DrJit 1.2.0, NumPy 1.26, SciPy
  1.16, scikit-image 0.25, matplotlib 3.10, trimesh 4.7, Open3D 0.19,
  plyfile 1.1, h5py 3.15, tqdm 4.67. See [requirements.txt](requirements.txt) /
  [environment.yml](environment.yml) for pinned versions.

```bash
conda env create -f environment.yml -n mm3DPS
conda activate mm3DPS
# Build the CUDA extension (required for training):
cd mm3DPS/cuda && python setup.py build_ext --inplace && cd ../..
```

LiDAR alignment uses CuPy, which needs a CUDA 12 runtime on the machine even
though PyTorch uses CUDA 11.8. Alignment stops with an error if CuPy cannot load.

The three baselines (DART, RadarFields, RadarSplat) each require their
own conda env with the upstream package installed. The path to that
upstream commit is pinned in `baselines/<name>/UPSTREAM_COMMIT.txt`,
and any local modifications are recorded as `.patch` files under
`baselines/<name>/patches/`.

---

## Data

The release ships **no raw data**. See [data/README.md](data/README.md)
for the full preprocessing pipeline. In short:

1. Download ColoRadar (sequences `2_28_2021_outdoors_run0`, `run1`,
   `run2`) under `data/coloradar/raw/kitti/`, or set `COLORADAR_ROOT`
   to its location.
2. Run `python -m mm3DPS.preprocessing.alignment.run_alignment --scene <scene>`
   followed by `python -m mm3DPS.preprocessing.alignment.pass2.run_pass2 --scene <scene>`.
3. The result is
   `data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json`,
   the canonical pass-2 alignment used by every training and evaluation
   entry point in this release.

> Running alignment yourself may give poses that differ from those behind the
> paper's results. On the one scene we re-checked, training quality was similar.

> All training and evaluation use
> `cascaded_frame_<F>_aligned_pass2.json` by default. The opt-out flag
> `--no_pass2` falls back to pass-1 (`cascaded_frame_<F>_aligned.json`).

### The six paper scenes

```
seq_0_frame_135   seq_1_frame_185   seq_1_frame_438
seq_2_frame_105   seq_2_frame_160   seq_2_frame_300
```

Each scene uses a 9-frame cascaded sequence under
`data/<scene>/radar/cascaded_frame_*.npy` plus matching pass-2 aligned
configs under `data/alignment_data/<scene>/cascade/`. The held-out test
frame is the middle frame of each 9-frame sequence (e.g. 135 for
`seq_0_frame_135`); the eight training frames are
`{F-4, F-3, F-2, F-1, F+1, F+2, F+3, F+4}`.

---

## Training

Single scene (about 3 min on an RTX 4090, 0.35 GB peak GPU memory):

```bash
python -m mm3DPS.train_frame_nvs \
       --scene seq_0_frame_135 \
       --target_n 20000 \
       --iters 500
```

(The thin wrapper `python train.py --scene seq_0_frame_135 ...` is
equivalent.)

All six scenes:

```bash
for s in seq_0_frame_135 seq_1_frame_185 seq_1_frame_438 \
         seq_2_frame_105 seq_2_frame_160 seq_2_frame_300; do
  python -m mm3DPS.train_frame_nvs --scene "$s" --target_n 20000 --iters 500
done
```

The test frame and eight training frames follow from the scene name;
override them with `--test_frame` and `--train_frames`. Data is read from
`data/` by default; use `--data_root` to point elsewhere.

Each run writes a per-scene output dir (default
`output/training/<scene>_<tag>/`) containing `best_model.pt`, `results.json`,
and intermediate artefacts. The CLI exposes the full set of
hyperparameters (LR sweeps, ablation flags, etc.); pass `--help` for
the complete list.

---

## Evaluation

### RA novel-view metrics

Per scene:

```bash
python -m mm3DPS.frame_nvs_report --output-root output/training
```

This walks every run directory, parses each `results.json`, and emits
a markdown table whose `cart_corr` column is the paper's primary
test-frame RA correlation.

### CRP/ADC product-agnostic evaluation

Render the complex range profiles of a trained scene, then evaluate them:

```bash
python -m mm3DPS.rerender_test_rp --scene seq_0_frame_135 --also-train-frames
python -m mm3DPS.evaluation.eval_crp_adc --scenes seq_0_frame_135
```

The first command saves the rendered profiles in the scene's run directory.
The second writes `results.json` and per-scene `.npy` arrays under
`output/crp_adc_eval/`, which the table scripts in `tables/` read. The evaluation
also takes `--data_root`, `--ours_dir` and `--output_dir`.

---

## Results

Held-out test-frame RA Cartesian correlation (`cart_corr`), mean over the
six paper scenes, on a single RTX 4090. Higher is better.

| Method        | Mean correlation | Wall time |
|---------------|------|-----------|
| DART          | 0.112 | 0.43 min/scene |
| RadarFields   | 0.133 | 1.53 min/scene |
| RadarSplat    | 0.339 | 19.74 min/scene |
| **3DPS**      | **0.587** | **3.02 min/scene** |

The mmIR and Sionna RT baselines from the paper will be added in a later
update.

Refer to the paper for the full per-scene breakdown plus the CRP/ADC
product-agnostic evaluation (Table 2).

---

## Baselines

See [baselines/README.md](baselines/README.md) for the per-baseline
operational guide. Briefly:

- **DART** — `python -m baselines.dart.runner.run_all_scenes --gpus 0 1`
- **RadarFields** — `python -m baselines.radarfields.runner.run_all_scenes --gpus 0 1`
- **RadarSplat** — `python -m baselines.radarsplat.runner.run_all_scenes --gpus 0 1`

Each driver starts the actual training under each
baseline's own conda env, then runs `finalize_metrics.py` (under the
mm3DPS env) to apply the 3DPS metrics so all four
methods produce comparable numbers. Set `DART_PYTHON`, `RADARFIELDS_PYTHON`
and `RADARSPLAT_PYTHON` to each baseline environment's Python.

---

## Tables

See [tables/README.md](tables/README.md) for the script-by-script table
mapping. Inputs:

- Your training runs under `output/training/`.
- Per-baseline `metrics.json` produced by each baseline's `run_all_scenes.py`.
- The CRP/ADC evaluation output under `output/crp_adc_eval/`.

---

## Repository structure

```
README.md
LICENSE                            # Cornell University terms
requirements.txt
environment.yml
train.py                           # thin wrapper around mm3DPS.train_frame_nvs

mm3DPS/                            # canonical 3DPS package
  __init__.py
  config.py                        # default paths and per-scene config loading
  rasterizer.py                    # core rasterizer
  rasterizer_factorized.py         # MIMO-factored rasterizer (paper default)
  bsdf_torch.py                    # ITU-material BSDF + Fresnel + reparam
  psf.py                           # Hann PSF table
  train_frame_nvs.py               # canonical paper training entry point
  train_gaussian.py                # point initialization and rendering helpers
  pose_utils.py                    # per-loop pose interpolation helpers
  rerender_test_rp.py              # CRP/ADC complex re-render evaluator
  frame_nvs_report.py              # NVS metrics aggregation
  cuda/                            # CUDA extension, required for training
  data/                            # ADC↔RA, range resolution and ADC loading
  evaluation/eval_crp_adc.py       # CRP/ADC evaluation
  evaluation/utils/                # metrics and visualization helpers
  preprocessing/                   # ColoRadar -> data/ pipeline + alignment
    preproc.py                     #   top-level driver (adc / configs / lidar / mesh)
    ColoRadar_tools/               #   raw ColoRadar parsers
    calib/                         #   sensor calibration files (cascade + transforms)
    alignment/                     #   cascade alignment (pass-1 + pass-2)

baselines/
  common/                          # shared scene list, NVS split, eval harness
  dart/                            # WiseLabCMU/dart adapter + runner + patches
  radarfields/                     # princeton-computational-imaging/RadarFields
  radarsplat/                      # umautobots/radarsplat
  README.md

configs/
  <scene>/config.json              # per-scene data and antenna pattern paths

assets/
  antenna_pattern/MMWCAS/          # tx1_76.npy + rx1_76.npy

data/
  README.md                        # ColoRadar fetch + preprocessing pipeline

tables/                            # paper table generators
  README.md
```

---

## License

See [LICENSE](LICENSE).

```
Copyright © 2026 by Cornell University.

All Rights Reserved.

Permission to use, copy, modify and distribute any part of “3DPS”
(“WORK”) and its associated copyrights for educational, research and
non-profit purposes, without fee, and without a written agreement is hereby
granted, provided that the above copyright notice, this paragraph and the
following three paragraphs shall be displayed in all copies or substantial
portions of the Work.

Those desiring to incorporate WORK into commercial products or use
WORK and its associated copyrights for commercial purposes should
contact the Center for Technology Licensing (“CTL”) at Cornell University
at 395 Pine Tree Road, Suite 310, Ithaca, NY 14850; email:
ctl-connect@cornell.edu; Tel: 607-254-4698; FAX: 607-254-5454 for a
commercial license. Inquiries should reference CTL docket number 12012.

IN NO EVENT SHALL CORNELL UNIVERSITY BE LIABLE TO ANY
PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF WORK AND ITS ASSOCIATED
COPYRIGHTS, EVEN IF CORNELL UNIVERSITY MAY HAVE BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE WORK PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND
CORNELL UNIVERSITY HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS. CORNELL UNIVERSITY MAKES NO
REPRESENTATIONS AND EXTENDS NO WARRANTIES OF ANY
KIND, EITHER IMPLIED OR EXPRESS, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF
WORK AND ITS ASSOCIATED COPYRIGHTS WILL NOT INFRINGE
ANY PATENT, TRADEMARK OR OTHER RIGHTS.
```

## Citation

```bibtex
@inproceedings{armouti2026threedps,
  title     = {{3D} Point Splatting for {mmWave} Radar Novel View Synthesis},
  author    = {Armouti, Adnan and Gao, Yixuan and Nandakumar, Rajalakshmi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
