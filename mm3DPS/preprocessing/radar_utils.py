import os
import numpy as np
import math

from mm3DPS.preprocessing.ColoRadar_tools.dataset_loaders import (
    get_cascade_params,
    get_adc_frame,
)

from mm3DPS.preprocessing.io_paths import get_base_dir, get_subdir


# returns sample i for a blackman window of size n
# param[in] i: sample index
# param[in] n: size of the window
# return: value of sample i of a blackman window of size n


# applies antenna coupling calibration
# param[in] adc_samples: raw complex adc samples for a frame of radar data
# param[in] coupling_calib: antenna coupling calibration returned 
#                           from dataset_loaders.get_cascade_params()\
# return: post-range-fft radar data with antenna coupling calibration applied


# applies phase calibration 
# param[in] adc_samples: raw complex adc samples for a frame of radar data
# param[in] phase_calib: phase calibration returned from dataset_loaders.get_cascade_params()
# return: adc samples with phase calibration applied


# applies frequency calibration
# param[in] adc_samples: raw complex adc samples for a frame of radar data
# param[in] freq_calib: frequency calibration returned from dataset_loaders.get_cascade_params()
# param[in] wave_config: waveform config returned from dataset_loaders.get_cascade_params()
# return: adc samples with frequency calibration applied
def apply_frequency_calibration(adc_samples, freq_calib, wave_config):
  # print("frequency adc_samples[0,0,0,0]: ", adc_samples[0,0,0,0])

  num_tx = freq_calib['num_tx']
  num_rx = freq_calib['num_rx']
  cal_frequency_slope = freq_calib['frequency_slope']
  cal_sampling_rate = freq_calib['sampling_rate']
  frequency_slope = wave_config['frequency_slope']
  sampling_rate = wave_config['adc_sample_frequency']
  num_adc_samples_per_chirp = wave_config['num_adc_samples_per_chirp']
  freq_cal_mat = freq_calib['cal_matrix']

  # create frequency calibration matrix
  delta_p = freq_cal_mat - freq_cal_mat[0,0]
  freq_calib = (2.0 * math.pi * delta_p 
                * (frequency_slope / cal_frequency_slope)
                * (cal_sampling_rate / sampling_rate))
  freq_calib /= num_adc_samples_per_chirp
  freq_calib_vec = np.expand_dims(freq_calib, -1) * np.arange(num_adc_samples_per_chirp)
  freq_calib_vec = np.exp(-1j * freq_calib_vec)

  # apply frequency calibration
  freq_calibrated_adc_samples = freq_calib_vec.reshape(num_tx,num_rx,1,num_adc_samples_per_chirp) * adc_samples
  # print(freq_calibrated_adc_samples[0,0,0,0])

  return freq_calibrated_adc_samples


def _save_adc(base_dir: str, kind: str, frame_idx: int, adc: np.ndarray) -> str:
    radar_dir = get_subdir(base_dir, "radar")
    fname = f"{kind}_frame_{frame_idx}.npy"
    path = os.path.join(radar_dir, fname)
    np.save(path, adc)
    return path


def generate_adc_window(*,
                        seq_idx: int,
                        center_frame_idx: int,
                        num_radar_frames: int,
                        dataset_dir: str,
                        calib_path: str,
                        out_root: str,
                        run_cascade: bool,
                        verbose: bool = False) -> dict:
    if num_radar_frames < 1 or (num_radar_frames % 2) != 1:
        raise ValueError("num_radar_frames must be odd and >= 1")

    seq_path = dataset_dir + str(seq_idx)
    base_dir = get_base_dir(seq_idx, center_frame_idx, out_root)

    half = num_radar_frames // 2
    cascade_indices = list(range(center_frame_idx - half, center_frame_idx + half + 1))

    results = {"cascade_adc": []}

    if run_cascade:
        all_params = get_cascade_params(calib_path)
        wf = all_params["waveform"]
        for c_idx in cascade_indices:
            adc = get_adc_frame(c_idx, seq_path, wf)
            # adc = apply_phase_calibration(adc, all_params["phase"])  # phase first - DISABLED to match CoIR reference
            adc = apply_frequency_calibration(adc, all_params["frequency"], wf)  # freq only (matches CoIR)
            adc = adc.transpose(2, 1, 0, 3)  # (N_C, N_Rx, N_Tx, N_ADC)
            if verbose:
                print(f"[adc] cascade idx={c_idx} shape={adc.shape}")
            results["cascade_adc"].append(_save_adc(base_dir, "cascaded", c_idx, adc))

    return results


