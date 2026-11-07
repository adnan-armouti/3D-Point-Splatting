"""Train + render DART for one 3DPS scene (cascade only).

Builds DART's cfg dict programmatically (mirror of upstream train.py) for
the ``ngpsh`` field, runs ``script_train``, then loads the checkpoint and
renders the held-out test frame's pose via ``dart.render``. The rendered
RDA cube is reduced to RA (sum over Doppler, matching upstream radar
conventions) and dumped to ``rendered_ra_polar.npy``. ``baselines.common.finalize``
(mm3DPS env) computes the metrics from there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = os.path.abspath(os.path.join(_HERE, "..", "upstream"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_UPSTREAM, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_cfg(scene_dir_name: str, data_root: str, out_dir: str, epochs: int,
               batch: int, lr: float = 0.01, key: int = 42,
               pval: float = 0.15) -> dict:
    """Mirror of train.py's cfg construction (ngpsh field, Identity adjust)."""
    sensor_path = os.path.join(data_root, scene_dir_name, "sensor.json")
    train_h5 = os.path.join(data_root, scene_dir_name, "data.h5")
    with open(sensor_path) as f:
        sensor_cfg = json.load(f)
    sensor_cfg.update(k=128)

    cfg = {
        "sensor": sensor_cfg,
        "shuffle_buffer": 500_000,
        "lr": lr,
        "batch": batch,
        "epochs": epochs,
        "key": key,
        "out": out_dir,
        "loss": {"weight": None, "loss": "l1", "eps": 1e-6, "delta": 1.0},
        "dataset": {
            "pval": pval,
            "iid_val": False,
            "path": train_h5,
            "doppler_decimation": 0,
        },
        "schedules": {
            "alpha_clip": {
                "func": "linear_piecewise",
                "args": {"values": [-1.0, 0.0, 0.05], "steps": [100, 500]},
            }
        },
        # NGPSH (paper default) field
        "field_name": "NGPSH",
        "field": {
            "levels": 12, "exponent": 0.43, "base": 4.0,
            "size": 16, "features": 2, "units": [64, 32],
            "alpha_scale": 0.1, "harmonics": 25,
        },
        # Identity adjustment (no pose refinement).
        "adjustment_name": "Identity",
        "adjustment": {},
    }
    return cfg


def _render_with_pose(dart, params, pose_dict):
    """Build a single-element RadarPose batch from a numpy/list pose dict
    and render. Returns ``(rda, ra_polar)`` where ``ra_polar`` is
    ``(Na, Nr)`` float32.
    """
    from jax import numpy as jnp
    from dart import types as dart_types

    pose = dart_types.RadarPose(
        v=jnp.array([pose_dict["v"]], dtype=jnp.float32),
        s=jnp.array([pose_dict["s"]], dtype=jnp.float32),
        p=jnp.array([pose_dict["p"]], dtype=jnp.float32),
        q=jnp.array([pose_dict["q"]], dtype=jnp.float32),
        x=jnp.array([pose_dict["x"]], dtype=jnp.float32),
        A=jnp.array([pose_dict["A"]], dtype=jnp.float32),
        i=jnp.array([pose_dict["i"]], dtype=jnp.int32),
    )
    rendered = dart.render(params, pose, key=42)             # (1, Nr, Nd, Na)
    rda = np.asarray(rendered[0], dtype=np.float32)          # (Nr, Nd, Na)
    ra = rda.sum(axis=1)                                     # (Nr, Na)
    ra_polar = ra.T.astype(np.float32)                       # (Na, Nr)
    return rda, ra_polar


def _build_train_pose_dict(scene: str, train_frame: int) -> dict:
    """Mirror of the DART adapter's per-frame pose construction."""
    from baselines.common import adapters as common
    from baselines.dart.adapter.mm3dps_to_dart import (
        build_radar_pose, _config_for_frame, _v_ego_for_cascade_frame,
    )

    cfg_path = _config_for_frame(scene, train_frame)
    cfg = common.load_config(cfg_path)
    T, R, t = common.pose_from_config(cfg)
    v_world = _v_ego_for_cascade_frame(scene, train_frame)
    pose_fields = build_radar_pose(t, R, v_world, frame_idx=0)
    return {
        "v": pose_fields["v"].tolist(),
        "s": float(pose_fields["s"]),
        "p": pose_fields["p"].tolist(),
        "q": pose_fields["q"].tolist(),
        "x": pose_fields["x"].tolist(),
        "A": pose_fields["A"].tolist(),
        "i": int(pose_fields["i"]),
    }


def _render_test(out_dir: str, test_meta_path: str, scene: str) -> None:
    """Load trained DART, build the test RadarPose, render, dump RA polar.

    Also dumps per-train-frame renders for the supplement figure pipeline,
    one ``rendered_ra_polar_frame_<F>.npy`` per training frame, under
    ``<out_dir>/train_frames_polar_raw/``.
    """
    from dart import DART

    with open(test_meta_path) as f:
        meta = json.load(f)

    dart = DART.from_config(**json.load(open(os.path.join(out_dir, "metadata.json"))))
    params = dart.load(os.path.join(out_dir, "model"))

    # Test frame.
    rda, ra_polar = _render_with_pose(dart, params, meta["test_pose"])
    np.save(os.path.join(out_dir, "rendered_ra_polar.npy"), ra_polar)
    np.save(os.path.join(out_dir, "rendered_rda.npy"), rda)

    # Train frames.
    train_frames = list(map(int, meta.get("train_frames", [])))
    if not train_frames:
        return

    train_dump_dir = os.path.join(out_dir, "train_frames_polar_raw")
    os.makedirs(train_dump_dir, exist_ok=True)
    for f in train_frames:
        try:
            pose_dict = _build_train_pose_dict(scene, f)
        except Exception as e:
            print(f"[run_dart_scene] failed to build train pose for frame {f}: {e}")
            continue
        _, ra_polar_f = _render_with_pose(dart, params, pose_dict)
        np.save(
            os.path.join(train_dump_dir,
                         f"rendered_ra_polar_frame_{int(f)}.npy"),
            ra_polar_f.astype(np.float32),
        )


def run(scene: str, data_root: str, out_dir: str,
        epochs: int, batch: int) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    scene_dir_name = scene
    cfg = _build_cfg(scene_dir_name, data_root, out_dir, epochs=epochs, batch=batch)

    t0 = time.time()
    from dart.script import script_train
    script_train(cfg)
    train_wall = time.time() - t0

    test_meta_path = os.path.join(data_root, scene_dir_name, "test_meta.json")
    _render_test(out_dir, test_meta_path, scene=scene)

    test_meta = json.load(open(test_meta_path))
    manifest_path = os.path.join(data_root, scene_dir_name, "adapter_manifest.json")
    deviations = []
    if os.path.isfile(manifest_path):
        try:
            deviations = json.load(open(manifest_path)).get("deviations_from_reference", [])
        except Exception:
            pass
    deviations = list(deviations) + [
        f"Training epochs: {epochs} (upstream default 3); with 8 train frames × "
        f"Nd valid columns / batch is small, more epochs are needed to give the "
        f"HashGrid + plenoctree NN time to settle.",
    ]
    meta = {
        "scene": scene,
        "test_frame": int(test_meta["test_frame"]),
        "train_frames": list(map(int, test_meta["train_frames"])),
        "wall_time_seconds": float(train_wall),
        "peak_gpu_mem_mib": 0.0,                      # JAX doesn't expose this easily
        "deviations_from_reference": deviations,
        "upstream_commit": "01e7057eba7ffe6e176c18e8f8e63c6576291c4d",
        "out_dir": out_dir,
        "epochs": int(epochs),
        "batch": int(batch),
        "Nr": int(test_meta["Nr"]),
        "Nd": int(test_meta["Nd"]),
        "Na": int(test_meta["Na"]),
        "range_res": float(test_meta["range_res"]),
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Finalize in mm3DPS env.
    import subprocess
    py = os.environ.get("MM3DPS_PYTHON", "python")
    subprocess.run(
        [py, "-m", "baselines.common.finalize", "--baseline", "dart",
         "--out-dir", out_dir, "--baseline-name", "dart"],
        cwd=_REPO,
        check=True,
    )
    with open(os.path.join(out_dir, "metrics.json")) as f:
        return json.load(f)


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument(
        "--data-root",
        default=os.path.join(_REPO, "baselines/dart/data_dart"),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default: baselines/dart/results/<scene>/",
    )
    ap.add_argument("--epochs", type=int, default=300,
                    help="DART epochs (upstream default 3 — but with our 8-frame "
                         "setup that's < 1 batch/epoch); more epochs are needed "
                         "to give the HashGrid + plenoctree NN time to settle.")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        _REPO, "baselines/dart/results", args.scene
    )
    result = run(args.scene, args.data_root, out_dir,
                 args.epochs, args.batch)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
