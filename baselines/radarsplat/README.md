# RadarSplat baseline (cascade)

[umautobots/radarsplat](https://github.com/umautobots/radarsplat) (ICCV 2025)
— Gaussian-splatting radar field. We adapt RadarSplat to the cascaded
TI MMWCAS dataset and report novel-view RA correlation against the
held-out test frame.

## Layout

```
baselines/radarsplat/
  adapter/              # mm3DPS -> RadarSplat per-scene dataset (TUM poses + GT)
  patches/              # local patches against the upstream commit
  runner/               # per-scene training driver
  UPSTREAM_COMMIT.txt   # pinned upstream commit
```

## Adapter (cascade only)

`adapter/mm3dps_to_radarsplat.py` converts one scene to RadarSplat's
expected layout:

- Sin-space → uniform-angle resample along azimuth (RadarSplat's polar
  RA is uniform in angle, not in `sin(angle)`).
- 9-frame sequence (8 train + 1 test) with TUM-format poses.
- `bin_size_radar` derived from
  `mm3DPS.data.io_utils.compute_range_res_from_cfg`.
- Range crop 15..110 matches the metric harness (drop TX-RX leakage and
  DFT wrap-around).

## Running

```bash
# RadarSplat env (gsplat + pytorch + cu12)
python -m baselines.radarsplat.runner.run_radarsplat_scene \
       --scene seq_0_frame_135 --epochs 7000

# All six scenes:
python -m baselines.common.runner --baseline radarsplat --gpus 0 1
```

The runner writes `results/<scene>/` and spawns `baselines.common.finalize`
(mm3DPS env) for the metric step.

## Deviations from upstream

- Synthetic 9-frame sequence vs upstream's longer collections.
- Sin-space → uniform-angle resample at adapter time so that the network's
  uniform-angle assumption holds against our raw RA polar.
- Per-scene normalisation: 99th percentile of train RA cubes.
- `--adj=Identity` — no learned pose refinement.

These are recorded in each scene's `metrics.json` as
`deviations_from_reference`.
