"""Smoke tests for the shared baseline infrastructure.

Runs CPU-only (no renderer, no GPU). Verifies:
  * ``nvs_split.BENCHMARK_SCENES`` matches the README table.
  * ``nvs_split`` returns 8-train / 1-test middle-frame-bracketed indices
    for the cascaded sensor, and every returned config path exists on disk.
  * ``adapters.load_cascaded_adc`` produces ``(16, 16, 12, 256)`` complex.
  * ``adapters.pose_from_config`` returns an SE(3) FLU world-from-sensor pose.
  * ``adapters.adc_to_polar_ra`` returns ``(127, 256)`` float32.
  * ``adapters.range_crop`` produces the 15..110 slice over the range axis.
  * ``adapters.fov_wedge_mask`` shape + sanity (sensor origin is inside wedge).
  * ``eval._range_profile_corr`` returns 1.0 on identical inputs.
  * ``eval.run_eval`` produces the README "Result schema" keys.

Runs on scene ``seq_0_frame_135`` only to keep the smoke test fast.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# Make the repo importable for ``mm3DPS`` and ``baselines``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.common import adapters, eval as common_eval, nvs_split


SCENE = "seq_0_frame_135"


# ---------------------------------------------------------------------------
# Scene list
# ---------------------------------------------------------------------------

def test_benchmark_scenes_are_the_paper_six():
    assert nvs_split.BENCHMARK_SCENES == (
        "seq_0_frame_135",
        "seq_1_frame_185",
        "seq_1_frame_438",
        "seq_2_frame_105",
        "seq_2_frame_160",
        "seq_2_frame_300",
    )


def test_scene_paths_exist():
    for s in nvs_split.BENCHMARK_SCENES:
        assert os.path.isdir(nvs_split.scene_dir(s)), f"missing scene dir: {s}"
        assert os.path.isdir(nvs_split.cascade_alignment_dir(s)), s


# ---------------------------------------------------------------------------
# nvs_split.py
# ---------------------------------------------------------------------------

def test_cascaded_split_matches_readme_for_seq_0_frame_135():
    split = nvs_split.cascaded_split(SCENE)
    assert split["test_frame"] == 135
    assert split["train_frames"] == [131, 132, 133, 134, 136, 137, 138, 139]
    assert len(split["train_files"]) == 8
    assert len(split["train_configs"]) == 8
    for p in split["train_files"] + [split["test_file"]]:
        assert os.path.isfile(p), p
    for p in split["train_configs"] + [split["test_config"]]:
        assert os.path.isfile(p), p


def test_split_is_always_middle_frame_bracketed():
    # Test frame must be strictly between train frame range (interpolation,
    # never extrapolation).
    for s in nvs_split.BENCHMARK_SCENES:
        sp = nvs_split.cascaded_split(s)
        tf = sp["test_frame"]
        trf = sp["train_frames"]
        assert min(trf) < tf < max(trf), f"{s}: not bracketed"


# ---------------------------------------------------------------------------
# adapters.py
# ---------------------------------------------------------------------------

def test_load_cascaded_adc_shape():
    sp = nvs_split.cascaded_split(SCENE)
    adc = adapters.load_cascaded_adc(sp["test_file"])
    assert adc.shape == (16, 16, 12, 256)
    assert np.issubdtype(adc.dtype, np.complexfloating)


def test_pose_from_config_is_SE3_flu():
    sp = nvs_split.cascaded_split(SCENE)
    cfg = adapters.load_config(sp["test_config"])
    T, R, t = adapters.pose_from_config(cfg)
    assert T.shape == (4, 4)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) > 0.99
    np.testing.assert_allclose(T[3, :], np.array([0, 0, 0, 1]))
    assert np.linalg.norm(t) < 1e4


def test_cascaded_adc_to_polar_ra_shape():
    sp = nvs_split.cascaded_split(SCENE)
    adc = adapters.load_cascaded_adc(sp["test_file"])
    ra = adapters.adc_to_polar_ra(adc)
    assert ra.shape == (127, 256)
    assert ra.dtype == np.float32
    assert np.all(np.isfinite(ra))


def test_range_crop_default_bins():
    x = np.arange(127 * 256, dtype=np.float32).reshape(127, 256)
    cropped = adapters.range_crop(x)
    assert cropped.shape == (127, 110 - 15)
    np.testing.assert_array_equal(cropped, x[:, 15:110])


def test_adc_to_cart_ra_smoke():
    sp = nvs_split.cascaded_split(SCENE)
    adc = adapters.load_cascaded_adc(sp["test_file"])
    cart = adapters.adc_to_cart_ra(adc, sp["test_config"])
    assert cart.ndim == 2
    assert cart.shape == (399, 399)
    assert np.all(np.isfinite(cart))


def test_fov_wedge_mask_shape_and_origin_inside():
    mask = adapters.fov_wedge_mask((399, 399))
    assert mask.shape == (399, 399)
    assert mask.dtype == bool
    H, W = mask.shape
    assert bool(mask[H // 2, W // 2])
    assert not bool(mask[H // 2, 0])


# ---------------------------------------------------------------------------
# eval.py
# ---------------------------------------------------------------------------

def test_range_profile_corr_identical_is_one():
    rp = common_eval._range_profile_corr(
        np.arange(127 * 95, dtype=np.float32).reshape(127, 95),
        np.arange(127 * 95, dtype=np.float32).reshape(127, 95),
    )
    assert rp == pytest.approx(1.0, abs=1e-6)


def test_run_eval_result_has_readme_schema():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(32, 32)).astype(np.float32)
    rend = gt + 0.1 * rng.normal(size=gt.shape).astype(np.float32)
    gp = rng.uniform(size=(127, 95)).astype(np.float32)
    rp = gp + 0.01 * rng.normal(size=gp.shape).astype(np.float32)
    extra = {
        "test_frame": 135,
        "train_frames": [131, 132, 133, 134, 136, 137, 138, 139],
        "wall_time_seconds": 1.23,
        "peak_gpu_mem_mib": 42.0,
        "deviations_from_reference": ["synthetic test"],
    }
    res = common_eval.run_eval(
        "radarsplat",
        SCENE,
        rend,
        gt,
        rendered_ra_polar_cropped=rp,
        gt_ra_polar_cropped=gp,
        extra=extra,
    )
    for k in (
        "baseline",
        "scene",
        "ra_corr",
        "range_profile_corr",
        "test_frame",
        "train_frames",
        "wall_time_seconds",
        "peak_gpu_mem_mib",
        "deviations_from_reference",
    ):
        assert k in res, f"missing key: {k}"
    assert res["baseline"] == "radarsplat"
    assert res["scene"] == SCENE
    assert isinstance(res["ra_corr"], float)
    assert res["range_profile_corr"] is not None
    assert res["ra_corr"] > 0.5

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "sub", "metrics.json")
        common_eval.write_metrics_json(out_path, res)
        assert os.path.isfile(out_path)
        import json
        loaded = json.load(open(out_path))
        assert loaded["baseline"] == "radarsplat"
