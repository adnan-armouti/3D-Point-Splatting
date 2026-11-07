# Data preparation

The release ships with **no** raw radar / LiDAR data. Reproducing the
paper requires fetching ColoRadar and running the provided alignment +
preprocessing scripts to populate this directory.

## Expected layout

After preprocessing, this directory must look like:

```
data/
  seq_0_frame_135/
    radar/
      cascaded_frame_131.npy  ...  cascaded_frame_139.npy   # 9 frames, complex (16,16,12,256)
    scene/
      mesh.ply                                              # LiDAR-derived mesh
      pcl.npy                                               # raw LiDAR point cloud (with intensity)
    configs/
      cascaded_frame_131.json  ...  cascaded_frame_139.json # raw per-frame sensor configs
  seq_1_frame_185/
    ...
  seq_2_frame_300/
    ...
  alignment_data/
    seq_0_frame_135/cascade/cascaded_frame_131_aligned_pass2.json  ... 139_aligned_pass2.json
    seq_1_frame_185/cascade/...
    ...
  v_ego_cache/                                              # required by the DART baseline
    seq_0_frame_135/frame_131_v_ego.npy                     # (3,) world-frame m/s
    ...
```

## Six paper scenes

```
seq_0_frame_135   seq_1_frame_185   seq_1_frame_438
seq_2_frame_105   seq_2_frame_160   seq_2_frame_300
```

## ColoRadar

We use the [ColoRadar](https://arpg.github.io/coloradar/) dataset from
the Autonomous Robotics and Perception Group (CU Boulder) — sequences
``2_28_2021_outdoors_run0``, ``run1``, ``run2`` (the three outdoor
runs). Cite the dataset per their publication when using this release.

## Pipeline

End-to-end raw-ColoRadar → 3DPS-ready data is driven by
``mm3DPS.preprocessing.preproc``, which has subcommands for each stage.
The cascade-only flow per scene:

```bash
# 1) Cascade ADC windows  -> data/<scene>/radar/cascaded_frame_<F>.npy
python -m mm3DPS.preprocessing.preproc adc \
    --seq <0|1|2> --frame <F> --num-radar-frames 9 \
    --dataset-dir <coloradar_root> --cascade

# 2) Per-frame raw cascade configs  -> data/<scene>/configs/cascaded_frame_<F>.json
python -m mm3DPS.preprocessing.preproc configs \
    --seq <0|1|2> --frame <F> --num-radar-frames 9 \
    --dataset-dir <coloradar_root> --cascade

# 3) LiDAR fusion + mesh reconstruction
#    Outputs:  data/<scene>/scene/{pcl.npy, mesh.ply}
python -m mm3DPS.preprocessing.preproc lidar  --seq <0|1|2> --frame <F> ...
python -m mm3DPS.preprocessing.preproc mesh   --seq <0|1|2> --frame <F> ...
```

`preproc.py --help` documents the full argument set per subcommand.

The mesh reconstruction step uses the
[PoissonRecon](https://github.com/mkazhdan/PoissonRecon) C++ binary
(Kazhdan et al.). Place a clone (or your own build) at
``third_party/PoissonRecon/`` so the recon step can resolve the
``PoissonRecon`` and ``SurfaceTrimmer`` binaries; they are built via
``make`` on first call.

## Cascade pass-2 alignment

The canonical alignment used throughout the release is the project's
**pass-2** trajectory-aware refinement. After step (3) above produces
the per-frame raw cascade configs and a per-scene mesh, build the
pass-2 configs with:

```bash
python -m mm3DPS.preprocessing.alignment.run_alignment --scene <scene>
python -m mm3DPS.preprocessing.alignment.pass2.run_pass2 --scene <scene>
```

This populates
``data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json``,
which is what every training and evaluation entry point reads.

## Scene-config mapping

Each scene's paths live at
``configs/<scene>/config.json`` and points at
``data/<scene>/...`` and ``assets/...`` with relative paths resolved
against the repository root (or ``$MM3DPS_ROOT`` if set).
