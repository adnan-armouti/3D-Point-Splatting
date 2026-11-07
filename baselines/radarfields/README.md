# RadarFields baseline (cascade)

[princeton-computational-imaging/RadarFields](https://github.com/princeton-computational-imaging/RadarFields)
(SIGGRAPH 2024) — neural radar field. We adapt RadarFields to the cascaded
TI MMWCAS dataset and report novel-view RA correlation against the
held-out test frame.

## Layout

```
baselines/radarfields/
  adapter/              # mm3DPS -> RadarFields per-scene config
  patches/              # local patches against the upstream commit
  runner/               # per-scene training driver
  UPSTREAM_COMMIT.txt   # pinned upstream commit
```

## Adapter (cascade only)

`adapter/mm3dps_to_radarfields.py` converts one scene to RadarFields'
expected layout:

- ADC -> RA polar magnitude via the canonical `mm3DPS.data.ra_utils`
  pipeline (FFT-based, periodic Hann window, 127 azimuth × 256 range).
- Range crop bins 15..110 to drop TX-RX leakage and DFT wrap-around.
- Per-frame configs (8 train + 1 test, middle-frame-bracketed) supply the
  poses; `bin_size_radar` is derived from
  `mm3DPS.data.io_utils.compute_range_res_from_cfg`.
- Boresight convention matches RadarFields' `+X = forward` (RH world).

## Running

```bash
# RadarFields env (pytorch + cu12 wheels)
python -m baselines.radarfields.runner.run_radarfields_scene \
       --scene seq_0_frame_135 --epochs 5000

# All six scenes:
python -m baselines.common.runner --baseline radarfields --gpus 0 1
```

The runner writes `results/<scene>/` and spawns `baselines.common.finalize`
(mm3DPS env) for the metric step.

## Deviations from upstream

- Synthetic 9-frame sequence (8 train + 1 held-out test), vs upstream's
  long sweeps from a moving rig.
- Pose-only training (no IMU), `+X-forward` cascade convention preserved.
- Per-scene normalisation: 99th percentile of train RA cubes (consistent
  with the other two baselines).

These are recorded in each scene's `metrics.json` as
`deviations_from_reference`.
