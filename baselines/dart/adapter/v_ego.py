"""Ego velocity per cascade frame, which DART's pose input needs.

``v_ego_values.json`` ships the values behind the paper's DART results, so DART
runs without the raw dataset. Two scripts produced them:

* the seed (this file): the raw ColoRadar ground-truth trajectory, differenced
  across frames F-1 and F+1 at the cascade timestamps;
* the refinement (``v_ego_refine.py``): a search around the seed against the
  measured range-azimuth-Doppler cube.

Values regenerated with those scripts go to ``data/v_ego_cache/<scene>/`` and
take priority over the shipped ones. Computing a seed needs raw ColoRadar under
``COLORADAR_ROOT`` (see ``mm3DPS.config``).

Usage::

    python -m baselines.dart.adapter.v_ego --scene seq_0_frame_135
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mm3DPS.config import COLORADAR_RAW_ROOT, DEFAULT_DATA_ROOT, default_split  # noqa: E402

SHIPPED_VALUES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v_ego_values.json")

# Scene index -> raw ColoRadar sequence.
SEQ_NAME_FOR_IDX = {
    0: "2_28_2021_outdoors_run0",
    1: "2_28_2021_outdoors_run1",
    2: "2_28_2021_outdoors_run2",
}


def scene_to_seq_name(scene: str) -> str:
    match = re.match(r"seq_(\d+)_frame_\d+$", scene)
    if match is None:
        raise ValueError(f"no raw sequence mapping for scene {scene!r}")
    return SEQ_NAME_FOR_IDX[int(match.group(1))]


def scene_frames(scene: str) -> list:
    """The nine frames of a paper scene, in order."""
    test_frame, train_frames = default_split(scene)
    return sorted([test_frame, *train_frames])


def compute_seed(scene: str, frame: int, raw_root: str = COLORADAR_RAW_ROOT) -> np.ndarray:
    """(GT_pos(t[F+1]) - GT_pos(t[F-1])) / (t[F+1] - t[F-1]) as a (3,) float32
    array in meters/second, world frame. Uses a one-sided difference at the ends
    of a sequence."""
    seq = os.path.join(raw_root, scene_to_seq_name(scene))
    gt = np.loadtxt(os.path.join(seq, "groundtruth", "groundtruth_poses.txt"))  # (N, 7): xyz + quat
    gt_xyz = gt[:, :3].astype(np.float64)
    gt_ts = np.loadtxt(os.path.join(seq, "groundtruth", "timestamps.txt")).astype(np.float64)
    cas_ts = np.loadtxt(os.path.join(seq, "cascade", "adc_samples", "timestamps.txt")).astype(np.float64)

    f_m = max(0, frame - 1)
    f_p = min(len(cas_ts) - 1, frame + 1)
    pos_m = np.array([np.interp(cas_ts[f_m], gt_ts, gt_xyz[:, i]) for i in range(3)])
    pos_p = np.array([np.interp(cas_ts[f_p], gt_ts, gt_xyz[:, i]) for i in range(3)])
    return ((pos_p - pos_m) / (cas_ts[f_p] - cas_ts[f_m])).astype(np.float32)


def cache_path(scene: str, frame: int, kind: str, data_root: str = DEFAULT_DATA_ROOT) -> str:
    """Regenerated value; ``kind`` is ``v_ego`` (seed) or ``v_ego_refined``."""
    return os.path.join(data_root, "v_ego_cache", scene, f"frame_{frame}_{kind}.npy")


def shipped_value(scene: str, frame: int, key: str):
    """Value from ``v_ego_values.json``; ``key`` is ``seed`` or ``refined``."""
    with open(SHIPPED_VALUES) as f:
        entry = json.load(f).get(scene, {}).get(str(frame))
    return None if entry is None else np.array(entry[key], dtype=np.float32)


def load_seed(scene: str, frame: int, data_root: str = DEFAULT_DATA_ROOT) -> np.ndarray:
    """Seed velocity: regenerated, else shipped, else computed from raw data."""
    path = cache_path(scene, frame, "v_ego", data_root)
    if os.path.isfile(path):
        return np.load(path).astype(np.float32)
    value = shipped_value(scene, frame, "seed")
    return value if value is not None else compute_seed(scene, frame)


def load_v_ego(scene: str, frame: int, data_root: str = DEFAULT_DATA_ROOT) -> np.ndarray:
    """Velocity for DART: refined if available, else the seed."""
    path = cache_path(scene, frame, "v_ego_refined", data_root)
    if os.path.isfile(path):
        return np.load(path).astype(np.float32)
    value = shipped_value(scene, frame, "refined")
    return value if value is not None else load_seed(scene, frame, data_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute seed ego velocities from raw ColoRadar.")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--raw_root", default=COLORADAR_RAW_ROOT)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    args = ap.parse_args()
    for frame in scene_frames(args.scene):
        path = cache_path(args.scene, frame, "v_ego", args.data_root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        value = compute_seed(args.scene, frame, args.raw_root)
        np.save(path, value)
        print(f"{args.scene} F={frame}  v_ego={value.tolist()}  -> {path}")


if __name__ == "__main__":
    main()
