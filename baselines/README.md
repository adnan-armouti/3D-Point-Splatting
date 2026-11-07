# Radar-NVS baselines for 3DPS

This directory hosts three radar-NVS baselines benchmarked against 3DPS
under an identical NVS protocol and identical metrics. Per-baseline
guides live in:

- [dart/README.md](dart/README.md) — Doppler-Aware Radar Tomography
  ([WiseLabCMU/dart](https://github.com/WiseLabCMU/dart), CVPR 2024)
- [radarfields/README.md](radarfields/README.md) — RadarFields
  ([princeton-computational-imaging/RadarFields](https://github.com/princeton-computational-imaging/RadarFields), SIGGRAPH 2024)
- [radarsplat/README.md](radarsplat/README.md) — RadarSplat
  ([umautobots/radarsplat](https://github.com/umautobots/radarsplat), ICCV 2025)

All three operate on the cascaded TI MMWCAS dataset with the same
8-train / 1-held-out-test split per scene.

---

## Pose convention

All three baselines and 3DPS read per-frame radar poses from
`data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json`
— the pass-2 trajectory-aware refinement. This is hardcoded in
[common/nvs_split.py](common/nvs_split.py) and (for DART specifically)
[dart/adapter/mm3dps_to_dart.py](dart/adapter/mm3dps_to_dart.py).

---

## Shared infrastructure

The shared infrastructure that every baseline imports lives under
[common/](common/):

- `common/nvs_split.py` — six paper scenes (`BENCHMARK_SCENES`), data root,
  alignment dirs, and the 8-train / 1-test cascaded split per
  scene (middle-frame-bracketed; `data/<scene>/radar/cascaded_frame_*.npy`
  with `data/alignment_data/<scene>/cascade/*_aligned_pass2.json`).
- `common/adapters.py` — ADC loading, `pose_from_config`, and
  `adc_to_polar_ra` / `adc_to_cart_ra` thin wrappers around
  `mm3DPS.data.ra_utils`.
- `common/eval.py` — `run_eval(...)` produces the canonical
  `metrics.json` schema.
- `common/train_frames_io.py` — shared per-train-frame artefact writer
  (used by all three baselines' `finalize_metrics.py`).

---

## Result schema

Every `metrics.json` follows the same schema produced by
`common/eval.py:run_eval(...)`:

```json
{
  "baseline":              "dart" | "radarfields" | "radarsplat",
  "scene":                 "seq_0_frame_135",
  "ra_corr":               0.xxx,
  "range_profile_corr":    0.xxx,
  "test_frame":            135,
  "train_frames":          [131, 132, 133, 134, 136, 137, 138, 139],
  "wall_time_seconds":     0.0,
  "peak_gpu_mem_mib":      0.0,
  "deviations_from_reference": ["...", "..."]
}
```

all-scenes drivers (`runner/run_all_scenes.py` in each baseline) write a per-
baseline `aggregate.json` summarising mean/stdev across the six paper
scenes.

---

## Six paper scenes

```
seq_0_frame_135   seq_1_frame_185   seq_1_frame_438
seq_2_frame_105   seq_2_frame_160   seq_2_frame_300
```

Each ships with the 9-frame cascaded sequence under
`data/<scene>/radar/cascaded_frame_*.npy` plus the matching pass-2
aligned configs under `data/alignment_data/<scene>/cascade/`.

---

## Running

Per-baseline:

```bash
# DART (cascade)
python -m baselines.dart.runner.run_all_scenes --gpus 0 1

# RadarFields
python -m baselines.radarfields.runner.run_all_scenes --gpus 0 1

# RadarSplat
python -m baselines.radarsplat.runner.run_all_scenes --gpus 0 1
```

`MM3DPS_PYTHON` should point at the env that has `mm3DPS` importable
(used by each runner to spawn `finalize_metrics.py` for the metric step).
The training step itself runs in each baseline's own env.

---

## Upstream pinning

Every baseline's `UPSTREAM_COMMIT.txt` pins the upstream git commit it
was tested against. Local modifications applied via `patches/*.patch` are
documented per-baseline.
