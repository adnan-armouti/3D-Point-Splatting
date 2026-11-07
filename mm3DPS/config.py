"""Paths and per-scene configuration for mm3DPS.

Every entry point takes the default locations below and can override them with
its command-line flags, so no script depends on a machine-specific path.

JSON configs (``configs/<scene>/config.json``) hold the file paths of one scene
and load into a :class:`SceneConfig`. Training settings come from the command
line of the training entry points. Relative paths are resolved against the
repository root, which ``$MM3DPS_ROOT`` overrides.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields

REPO_ROOT = os.path.abspath(os.environ.get("MM3DPS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data")
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "output", "training")
# Raw ColoRadar is only needed to re-run preprocessing.
COLORADAR_RAW_ROOT = os.environ.get(
    "COLORADAR_ROOT", os.path.join(DEFAULT_DATA_ROOT, "coloradar", "raw", "kitti"))


def default_split(scene):
    """Paper split for a scene named ``..._frame_<F>``: test F, train F+-1..4."""
    match = re.search(r"_frame_(\d+)", scene)
    if match is None:
        raise SystemExit(
            f"Cannot infer frames from scene name '{scene}'. "
            "Pass --test_frame and --train_frames.")
    frame = int(match.group(1))
    return frame, [frame + d for d in (-4, -3, -2, -1, 1, 2, 3, 4)]


def check_scene_data(scene, data_root):
    """Stop with a clear message if a scene's data is not where it should be."""
    needed = [os.path.join(data_root, scene, "radar"),
              os.path.join(data_root, "alignment_data", scene, "cascade")]
    missing = [p for p in needed if not os.path.isdir(p)]
    if missing:
        raise SystemExit(
            f"Data for scene '{scene}' not found. Missing: {', '.join(missing)}. "
            "Set --data_root, or see data/README.md to prepare the data.")


@dataclass
class SceneConfig:
    """File paths of one scene."""

    config_file: str
    scene_file: str
    gt_adc_file: str
    tx_pattern_file: str
    rx_pattern_file: str


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def load_config_from_json(json_path: str) -> SceneConfig:
    """Load a per-scene config from JSON, with its paths resolved."""
    with open(json_path, "r") as f:
        config = SceneConfig(**json.load(f))
    for field in fields(config):
        setattr(config, field.name, _resolve(getattr(config, field.name)))
    return config


def load_trained_config(scene: str,
                        data_root: str = DEFAULT_DATA_ROOT) -> SceneConfig:
    """Load ``configs/<scene>/config.json``, with data files under ``data_root``."""
    config_path = os.path.join(CONFIGS_DIR, scene, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found for scene {scene!r}: {config_path}. "
            "Set MM3DPS_ROOT or run from the repository root.")
    config = load_config_from_json(config_path)
    repo_data = os.path.join(REPO_ROOT, "data")
    for name in ("config_file", "scene_file", "gt_adc_file"):
        setattr(config, name, os.path.join(
            data_root, os.path.relpath(getattr(config, name), repo_data)))
    return config
