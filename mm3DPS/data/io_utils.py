"""Radar config and target ADC loading."""

import json

import numpy as np


def compute_range_res_from_cfg(cfg_path: str) -> float:
    """Compute range resolution R = c / (2  S  (rampEndTime - adcStartTime))."""
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    c = 299_792_458.0
    slope_hz_per_s = float(cfg["freqSlope"])  # Hz/s
    t_adc = float(cfg["adcStartTime"])        # s
    t_end = float(cfg["rampEndTime"])         # s
    T_chirp = t_end - t_adc
    if slope_hz_per_s <= 0 or T_chirp <= 0:
        return 0.117
    bandwidth = slope_hz_per_s * T_chirp
    return float(c / (2.0 * bandwidth))


def load_target(path: str, device: str):
    """Target ADC frame (12,16,256,2) as float32 real-imag tensor."""
    # Imported here because some baseline environments use only the helper above.
    import torch

    arr  = np.load(path)                       # (1,256,12,16) complex64
    cmpl = np.squeeze(arr[0]).transpose(1, 0, 2)
    return torch.view_as_real(torch.from_numpy(cmpl)).float().to(device)
