"""Data adapter primitives shared by all three baselines.

These are stateless wrappers around ``mm3DPS`` utilities. They exist so the
baselines do not re-implement ADC->RA conversion or pose extraction — any
drift there would make the metrics incomparable against 3DPS.

Inputs and outputs:

* ``load_cascaded_adc(path)`` -> complex128 ``(16, 16, 12, 256)``
    = ``(chirps, RX, TX, ADC)`` as stored on disk.
* ``load_config(path)`` -> dict (raw aligned sensor config).
* ``pose_from_config(config)`` -> ``(T, R, t)`` — world-from-sensor, meters,
    FLU (+X forward, +Y left, +Z up). ``T`` is a 4x4 homogeneous matrix.
* ``adc_to_polar_ra(adc)`` -> float32 polar magnitude RA image, shape
    ``(127, 256)`` (azimuth × range).
* ``adc_to_cart_ra(adc, config, *, range_crop=(15, 110))`` -> float32 Cartesian
    RA magnitude (output of ``ra_polar_to_cartesian`` on the range-cropped polar).
* ``fov_wedge_mask(cart_shape, az_range_deg=(-21, 69))`` -> bool mask over a
    Cartesian RA image (True inside the wedge).
* ``range_crop(x, bins=(15, 110))`` -> slice the range axis of a polar RA image.

Pose semantics:
    The aligned cascaded JSON stores ``tx_array[0].pos_mm`` (mm) and
    ``tx_array[0].boresight`` (unit 3-vector, world frame). The sensor-local
    +X axis is the boresight; +Z is world-up; +Y = Z × X. That yields a
    right-handed world-from-sensor (meters, FLU) rotation, matching the
    conventions of all three baseline upstreams.
"""

from __future__ import annotations

import json
import os
from typing import Tuple, Union

import numpy as np

# Expected on-disk ADC shape — assert guards against silent layout drift.
_CASC_SHAPE: Tuple[int, ...] = (16, 16, 12, 256)   # (chirps, RX, TX, ADC)

# Range-bin crop: drop bins 0..14 (TX-RX leakage) and bins 110..end (DFT wrap).
DEFAULT_RANGE_CROP: Tuple[int, int] = (15, 110)

# Cascaded radar forward wedge. The on-disk RA azimuth axis spans the ±90°
# forward hemisphere; the cascaded radar's useful signal sits within about
# ``(-21°, +69°)`` of sensor +X. Used by ``fov_wedge_mask``.
DEFAULT_FOV_DEG: Tuple[float, float] = (-21.0, 69.0)


# ---------------------------------------------------------------------------
# ADC loaders
# ---------------------------------------------------------------------------

def load_cascaded_adc(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.shape != _CASC_SHAPE:
        raise ValueError(
            f"cascaded ADC shape mismatch for {path}: got {arr.shape}, "
            f"expected {_CASC_SHAPE}"
        )
    if not np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"cascaded ADC dtype {arr.dtype} is not complex")
    return arr


# ---------------------------------------------------------------------------
# Config + pose
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _orthonormal_flu_basis(boresight: np.ndarray) -> np.ndarray:
    """Build a 3x3 world-from-sensor rotation from a boresight vector.

    Sensor axes: +X = boresight (forward), +Z = world-up, +Y = Z × X (left).
    Columns of the returned matrix are the sensor axes expressed in world
    coordinates (world-from-sensor).
    """
    b = np.asarray(boresight, dtype=np.float64)
    n = np.linalg.norm(b)
    if n < 1e-9:
        raise ValueError("boresight magnitude ~ 0")
    x_s = b / n
    z_w = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(x_s, z_w))) > 0.999:
        raise ValueError(
            "boresight nearly parallel to world-up; sensor orientation is degenerate"
        )
    y_s = np.cross(z_w, x_s)
    y_s = y_s / np.linalg.norm(y_s)
    z_s = np.cross(x_s, y_s)
    z_s = z_s / np.linalg.norm(z_s)
    R = np.stack([x_s, y_s, z_s], axis=1)  # columns are sensor axes in world
    return R.astype(np.float64)


def pose_from_config(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(T, R, t)`` world-from-sensor, meters, FLU.

    ``T`` is 4x4; ``R`` is 3x3 (columns = sensor axes in world); ``t`` is
    the sensor origin in world coordinates (meters).
    """
    tx0 = config["tx_array"][0]
    t_mm = np.asarray(tx0["pos_mm"], dtype=np.float64)
    t_m = t_mm / 1000.0
    R = _orthonormal_flu_basis(np.asarray(tx0["boresight"], dtype=np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t_m
    return T, R, t_m


# ---------------------------------------------------------------------------
# ADC → RA image (polar / Cartesian)
# ---------------------------------------------------------------------------

def _casc_adc_to_ri(adc_complex: np.ndarray) -> np.ndarray:
    """Reduce cascaded ``(16, 16, 12, 256)`` complex to ``(12, 16, 256, 2)`` float32.

    Picks chirp 0, transposes RX↔TX, stacks real/imag.
    """
    if adc_complex.shape != _CASC_SHAPE:
        raise ValueError(
            f"expected cascaded ADC shape {_CASC_SHAPE}, got {adc_complex.shape}"
        )
    casc = adc_complex[0].transpose(1, 0, 2)  # (TX=12, RX=16, ADC=256)
    return np.stack([casc.real, casc.imag], axis=-1).astype(np.float32)


def adc_to_polar_ra(adc: np.ndarray) -> np.ndarray:
    """Magnitude polar RA image (azimuth × range) for the cascaded radar.

    Returns float32 of shape ``(127, 256)``. Uses ``adc_to_ra_complex`` then
    ``abs`` (torch FFT, periodic Hann window).
    """
    import torch
    from mm3DPS.data.ra_utils import adc_to_ra_complex
    ri = _casc_adc_to_ri(adc)
    ri_t = torch.from_numpy(ri)
    if torch.cuda.is_available():
        ri_t = ri_t.cuda()
    with torch.no_grad():
        polar = torch.abs(adc_to_ra_complex(ri_t)).float().cpu().numpy()
    return polar


def range_crop(x: np.ndarray, bins: Tuple[int, int] = DEFAULT_RANGE_CROP) -> np.ndarray:
    """Slice the range axis of a polar RA image to ``[bins[0]:bins[1]]``.

    Assumes the range axis is axis ``-1`` (standard for polar RA in 3DPS).
    """
    lo, hi = bins
    if x.ndim < 2:
        raise ValueError(f"range_crop expects ≥2D input; got shape {x.shape}")
    return x[..., lo:hi]


def adc_to_cart_ra(
    adc: np.ndarray,
    config: Union[dict, str],
    *,
    range_crop_bins: Tuple[int, int] = DEFAULT_RANGE_CROP,
) -> np.ndarray:
    """ADC → cropped-polar → Cartesian RA (magnitude, float32).

    Uses ``mm3DPS.data.io_utils.compute_range_res_from_cfg`` for the range
    resolution and ``mm3DPS.data.ra_utils.ra_polar_to_cartesian`` for the
    polar→Cartesian resample. ``config`` may be a dict or a path — a dict is
    serialized to a tempfile only because ``compute_range_res_from_cfg``
    takes a path. If ``config`` is already a path we skip the round-trip.
    """
    from mm3DPS.data.ra_utils import ra_polar_to_cartesian
    from mm3DPS.data.io_utils import compute_range_res_from_cfg

    if isinstance(config, str):
        cfg_path = config
    else:
        import tempfile
        fd, cfg_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(config, f)
            range_res = compute_range_res_from_cfg(cfg_path)
        finally:
            os.unlink(cfg_path)
        ra_polar = adc_to_polar_ra(adc)
        ra_polar = range_crop(ra_polar, range_crop_bins)
        return ra_polar_to_cartesian(ra_polar, range_res)

    range_res = compute_range_res_from_cfg(cfg_path)
    ra_polar = adc_to_polar_ra(adc)
    ra_polar = range_crop(ra_polar, range_crop_bins)
    return ra_polar_to_cartesian(ra_polar, range_res)


# ---------------------------------------------------------------------------
# FoV wedge mask (Cartesian)
# ---------------------------------------------------------------------------

def fov_wedge_mask(
    cart_shape: Tuple[int, int],
    az_range_deg: Tuple[float, float] = DEFAULT_FOV_DEG,
) -> np.ndarray:
    """Boolean mask over a Cartesian RA image; True inside the sensor FoV wedge.

    ``cart_shape`` is ``(H, W)`` — the output shape of
    ``mm3DPS.data.ra_utils.ra_polar_to_cartesian`` (default ``(399, 399)``).
    The azimuth angle of a pixel is ``atan2(x, y)`` measured from +y (sensor
    forward), positive to the right (increasing x). A pixel is inside the
    wedge when ``az_range_deg[0] <= angle_deg <= az_range_deg[1]``.
    """
    H, W = cart_shape
    xi = np.linspace(-1.0, 1.0, W)
    yi = np.linspace(0.0, 1.0, H)
    X, Y = np.meshgrid(xi, yi)
    angle_deg = np.degrees(np.arctan2(X, np.maximum(Y, 1e-12)))
    lo, hi = az_range_deg
    return (angle_deg >= lo) & (angle_deg <= hi)
