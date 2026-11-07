"""Benchmark scene list and the 8-train / 1-test middle-frame-bracketed split.

Enumerates the 9 on-disk cascaded frames per scene, sorts by frame number,
picks index 4 as test and indices ``{0..3, 5..8}`` as train. One module,
not three copies.

For each scene the returned dict holds::

    {
      "train_files":    [<8 paths to radar/cascaded_frame_*.npy>],
      "train_configs":  [<8 paths to aligned per-frame config JSON>],
      "train_frames":   [<8 ints — frame numbers parsed from filenames>],
      "test_file":      "<path to middle cascaded_frame_*.npy>",
      "test_config":    "<path to middle aligned config JSON>",
      "test_frame":     <int>,
    }

Aligned per-frame configs live under
``data/alignment_data/<scene>/cascade/`` (pass-2 alignment).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List

# 6 scenes, alphabetical. Every baseline imports from here; do not hard-code
# lists in per-baseline modules.
BENCHMARK_SCENES: tuple[str, ...] = (
    "seq_0_frame_135",
    "seq_1_frame_185",
    "seq_1_frame_438",
    "seq_2_frame_105",
    "seq_2_frame_160",
    "seq_2_frame_300",
)

# Repository root, two levels up from this file.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
ALIGNMENT_ROOT = os.path.join(DATA_ROOT, "alignment_data")


def scene_dir(scene: str) -> str:
    return os.path.join(DATA_ROOT, scene)


def cascade_alignment_dir(scene: str) -> str:
    """Where per-frame aligned cascaded configs live (one per 9 frames)."""
    return os.path.join(ALIGNMENT_ROOT, scene, "cascade")


_CASC_RE = re.compile(r"cascaded_frame_(\d+)\.npy$")


def _frame_number(path: str, pattern: re.Pattern) -> int:
    m = pattern.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"could not parse frame number from {path}")
    return int(m.group(1))


def _pick_split(sorted_files: List[str], frame_pattern: re.Pattern,
                configs_for_frame) -> Dict[str, object]:
    if len(sorted_files) != 9:
        raise ValueError(
            f"expected exactly 9 frames per sensor, got {len(sorted_files)}: "
            f"{sorted_files}"
        )
    frames = [_frame_number(p, frame_pattern) for p in sorted_files]
    train_idx = [0, 1, 2, 3, 5, 6, 7, 8]
    test_idx = 4

    train_files = [sorted_files[i] for i in train_idx]
    train_frames = [frames[i] for i in train_idx]
    train_configs = [configs_for_frame(frames[i]) for i in train_idx]

    test_file = sorted_files[test_idx]
    test_frame = frames[test_idx]
    test_config = configs_for_frame(test_frame)

    missing = [p for p in train_configs + [test_config] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"aligned config missing for frames — cannot enumerate split: {missing}"
        )

    return {
        "train_files": train_files,
        "train_configs": train_configs,
        "train_frames": train_frames,
        "test_file": test_file,
        "test_config": test_config,
        "test_frame": test_frame,
    }


def cascaded_split(scene: str) -> Dict[str, object]:
    radar_glob = os.path.join(scene_dir(scene), "radar", "cascaded_frame_*.npy")
    files = sorted(glob.glob(radar_glob), key=lambda p: _frame_number(p, _CASC_RE))
    align_dir = cascade_alignment_dir(scene)

    def _cfg(frame: int) -> str:
        return os.path.join(align_dir, f"cascaded_frame_{frame}_aligned_pass2.json")

    return _pick_split(files, _CASC_RE, _cfg)


