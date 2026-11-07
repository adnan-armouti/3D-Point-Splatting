# DART baseline (cascade)

[WiseLabCMU/dart](https://github.com/WiseLabCMU/dart) — Doppler-Aware
Radar Tomography. We adapt DART to the cascaded TI MMWCAS dataset and
report novel-view RA correlation against the held-out test frame.

## Layout

```
baselines/dart/
  adapter/              # mm3DPS -> DART adapter (data.h5 / sensor.json builder)
  patches/              # cascade_az127.patch — adds an 86-element cascade
                        # azimuth gain (cascade_az127) to upstream antenna.py
  runner/               # script_train wrapper + per-scene driver + all-scenes driver
  UPSTREAM_COMMIT.txt   # pinned upstream commit
```

## Upstream

`UPSTREAM_COMMIT.txt` pins the WiseLabCMU/dart commit. Apply
`patches/cascade_az127.patch` against that commit to add the
`cascade_az127` gain function used by the cascade adapter.

## Adapter (cascade only)

`adapter/mm3dps_to_dart.py` converts one scene to DART format:

- 86-element row-0 cascade virtual array (rather than the upstream 8-element
  awr1843boost array) and azimuth FFT-128 to `Na=127` bins. The custom
  `cascade_az127` gain matches that bin count.
- `sensor.json`: `(Nr=256, Nd=16, Na=127)` plus `r=[0, r_max, 256]` and
  `d=[-D_MAX_MPS, D_MAX_MPS, 16]`. `r_max` is derived from the per-scene
  cascade pass-2 alignment (`compute_range_res_from_cfg`).
- `data.h5`, `data_test.h5`: per-Doppler column records (one row per `(frame,
  doppler_bin)` pair) with the upstream's `(x, A, v, s, p, q, i, doppler,
  rad)` schema. Columns whose `psi_min/pi/s` weight is zero are dropped.

## Running

Build the H5 once per scene:

```bash
python -m baselines.dart.adapter.mm3dps_to_dart --scene seq_0_frame_135
```

Train + render one scene (DART env):

```bash
DART_PYTHON=$(which python)  # the env that has dart + jax installed
python -m baselines.dart.runner.run_dart_scene --scene seq_0_frame_135 \
       --epochs 300 --batch 8
```

The all-scenes driver runs all six paper scenes across two GPUs:

```bash
python -m baselines.dart.runner.run_all_scenes --gpus 0 1
```

The runner writes `results/<scene>/` containing `train_meta.json`,
`rendered_ra_polar.npy`, then spawns `finalize_metrics.py` (mm3DPS env)
which produces `metrics.json` + the four RA inspection PNGs (linear / dB
on the GT and rasterized images).

## Aggregate

`run_all_scenes.py` writes `results/aggregate.json`:

```json
{
  "baseline": "dart",
  "n_scenes": 6,
  "ra_corr": {"mean": ..., "stdev": ..., "n": 6},
  "range_profile_corr": {"mean": ..., "stdev": ..., "n": 6}
}
```

## Deviations from upstream

- Synthetic 9-frame sequence (8 train + 1 held-out test), versus DART's
  hundreds-of-frame Boreas collections. The model is data-starved at 16
  cascade chirps and converges to its ceiling within ~300 epochs.
- Custom `cascade_az127` antenna gain (86-element half-lambda array)
  added via `patches/cascade_az127.patch`.
- `--pval=0.15` (versus 0 in upstream defaults) ensures a non-empty
  validation set under the 8-frame regime.
- `--adj=Identity` — no learned pose refinement.

These are recorded in each scene's `adapter_manifest.json` and surfaced
on the `metrics.json` output as `deviations_from_reference`.
