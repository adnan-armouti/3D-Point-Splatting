"""Range-Azimuth (RA) map utilities for training."""

import os
import torch
import numpy as np
from scipy import interpolate


def txrx_to_vx_chirps_torch(adc_data: torch.Tensor) -> torch.Tensor:
    """Torch port of txrx_to_vx_chirps.
    adc_data: complex tensor with shape (NUM_CHIRPS=1, NUM_RX=16, NUM_TX=12, NUM_ADC_SAMPLES)
    returns: complex tensor of shape (1, 7, 86, NUM_ADC_SAMPLES)
    """
    assert adc_data.ndim == 4 and adc_data.size(0) == 1, "expected (1, RX, TX, ADC)"
    NUM_RX, NUM_TX = adc_data.size(1), adc_data.size(2)
    rx_locations = [(0, 0), (1, 0), (2, 0), (3, 0), (11, 0), (12, 0), (13, 0), (14, 0), (46, 0), (47, 0), (48, 0), (49, 0), (50, 0), (51, 0), (52, 0), (53, 0)]
    tx_locations = [(0, 0), (4, 0), (8, 0), (9, 1), (10, 4), (11, 6), (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0)]
    device = adc_data.device
    dtype = adc_data.dtype
    out = torch.zeros((1, 7, 86, adc_data.size(-1)), dtype=dtype, device=device)
    filled = torch.zeros((1, 7, 86), dtype=torch.bool, device=device)
    for rx_id in range(NUM_RX):
        rx_loc = rx_locations[rx_id]
        for tx_id in range(NUM_TX):
            tx_loc = tx_locations[tx_id]
            vx_x = rx_loc[0] + tx_loc[0]
            vx_y = rx_loc[1] + tx_loc[1]
            if not filled[0, vx_y, vx_x]:
                out[:, vx_y, vx_x, :] = adc_data[:, rx_id, tx_id, :]
                filled[0, vx_y, vx_x] = True
            else:
                out[:, vx_y, vx_x, :] = (out[:, vx_y, vx_x, :] + adc_data[:, rx_id, tx_id, :]) / 2
    return out


def adc_to_ra_image(adc_ri: torch.Tensor) -> torch.Tensor:
    """Build RA magnitude image from ADC real-imag tensor using the notebook pipeline.
    adc_ri shape: 4D real-imag with last dim == 2; contains TX(12), RX(16), ADC(256) in some order.
    returns: 2D magnitude tensor [A=127, R=256] matching the notebook (after azimuth FFT and trimming first bin).
    """
    assert adc_ri.ndim == 4 and adc_ri.size(-1) == 2, "expected (..., 2) real-imag"
    dims = list(adc_ri.shape[:-1])
    idx_adc = int(dims.index(256))
    idx_tx  = int(dims.index(12))
    idx_rx  = int(dims.index(16))
    order = [idx_rx, idx_tx, idx_adc, len(dims)]
    x = adc_ri.permute(order)  # (RX, TX, ADC, 2)
    x_c = torch.complex(x[..., 0].contiguous(), x[..., 1].contiguous())  # (RX, TX, ADC)
    x_c = x_c.unsqueeze(0)  # (1, RX, TX, ADC)
    vx = txrx_to_vx_chirps_torch(x_c)  # (1, 7, 86, ADC)
    ra = vx[0, 0, :, :]  # pick elevation row 0 -> (86, ADC)
    # range window and FFT
    num_adc = ra.size(-1)
    ra = ra * torch.hann_window(num_adc, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[None, :]
    ra = torch.fft.fft(ra, n=num_adc, dim=-1)
    # azimuth window and FFT to 128 bins
    num_vx = ra.size(0)
    ra = ra * torch.hann_window(num_vx, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[:, None]
    ra = torch.fft.ifftshift(ra, dim=0)
    ra = torch.fft.fft(ra, n=128, dim=0)
    ra = ra[1:, :]  # drop first azimuth bin
    ra = torch.fft.fftshift(ra, dim=0)
    return ra.abs().float()


def adc_to_ra_complex(adc_ri: torch.Tensor) -> torch.Tensor:
    """Build complex RA image from ADC real-imag tensor (for gradient-based training).

    Same as adc_to_ra_image but returns COMPLEX values (not magnitude) to enable
    loss computation on both real and imaginary components.

    adc_ri shape: 4D real-imag with last dim == 2; contains TX(12), RX(16), ADC(256) in some order.
    returns: 2D complex tensor [A=127, R=256] (complex64)
    """
    assert adc_ri.ndim == 4 and adc_ri.size(-1) == 2, "expected (..., 2) real-imag"
    dims = list(adc_ri.shape[:-1])
    idx_adc = int(dims.index(256))
    idx_tx  = int(dims.index(12))
    idx_rx  = int(dims.index(16))
    order = [idx_rx, idx_tx, idx_adc, len(dims)]
    x = adc_ri.permute(order)  # (RX, TX, ADC, 2)
    x_c = torch.complex(x[..., 0].contiguous(), x[..., 1].contiguous())  # (RX, TX, ADC)
    x_c = x_c.unsqueeze(0)  # (1, RX, TX, ADC)
    vx = txrx_to_vx_chirps_torch(x_c)  # (1, 7, 86, ADC)
    ra = vx[0, 0, :, :]  # pick elevation row 0 -> (86, ADC)
    # range window and FFT
    num_adc = ra.size(-1)
    ra = ra * torch.hann_window(num_adc, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[None, :]
    ra = torch.fft.fft(ra, n=num_adc, dim=-1)
    # azimuth window and FFT to 128 bins
    num_vx = ra.size(0)
    ra = ra * torch.hann_window(num_vx, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[:, None]
    ra = torch.fft.ifftshift(ra, dim=0)
    ra = torch.fft.fft(ra, n=128, dim=0)
    ra = ra[1:, :]  # drop first azimuth bin
    ra = torch.fft.fftshift(ra, dim=0)
    # Return COMPLEX values (not magnitude) for loss on real + imag
    return ra  # Complex tensor (127, 256)
# TX and RX locations for virtual array mapping (matches inspect_ra_coir.py)
RX_LOCATIONS = [(0, 0), (1, 0), (2, 0), (3, 0), (11, 0), (12, 0), (13, 0), (14, 0),
                (46, 0), (47, 0), (48, 0), (49, 0), (50, 0), (51, 0), (52, 0), (53, 0)]
TX_LOCATIONS = [(0, 0), (4, 0), (8, 0), (9, 1), (10, 4), (11, 6),
                (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0)]


def txrx_to_virtual_array_numpy(adc_txrx: np.ndarray) -> np.ndarray:
    """Convert TX/RX format ADC data to virtual array format (NumPy version).

    Matches the implementation in inspect_ra_coir.py exactly.

    Args:
        adc_txrx: Complex array of shape (TX=12, RX=16, ADC=256)

    Returns:
        np.ndarray: Virtual array of shape (86, 256) - elevation 0 only
    """
    NUM_TX, NUM_RX = 12, 16
    NUM_ADC = adc_txrx.shape[-1]

    # Create output arrays for full virtual array (7, 86, ADC)
    vx_full = np.zeros((7, 86, NUM_ADC), dtype=np.complex128)
    filled = np.zeros((7, 86), dtype=bool)

    for rx_id in range(NUM_RX):
        rx_loc = RX_LOCATIONS[rx_id]
        for tx_id in range(NUM_TX):
            tx_loc = TX_LOCATIONS[tx_id]
            vx_x = rx_loc[0] + tx_loc[0]  # azimuth (0-85)
            vx_y = rx_loc[1] + tx_loc[1]  # elevation (0-6)

            if not filled[vx_y, vx_x]:
                vx_full[vx_y, vx_x, :] = adc_txrx[tx_id, rx_id, :]
                filled[vx_y, vx_x] = True
            else:
                # Average with existing value (same as ra_utils.py)
                vx_full[vx_y, vx_x, :] = (vx_full[vx_y, vx_x, :] + adc_txrx[tx_id, rx_id, :]) / 2

    # Return elevation 0 only: (86, 256)
    return vx_full[0, :, :]








def save_ra_cartesian_png(
    ra_cart: np.ndarray,
    output_path: str,
    range_res: float = 0.0557,
    scale: str = "dB",
    title: str = "",
    cmap: str = "hot",
    db_floor: float = -40.0,
    dpi: int = 100,
):
    """Save a single RA Cartesian image as .png (linear or dB scale).

    Args:
        ra_cart: 2D Cartesian RA magnitude array (rows=range, cols=azimuth).
        output_path: Path to save PNG.
        range_res: Range resolution in meters (for axis extents).
        scale: 'dB' or 'linear'.
        title: Figure title.
        cmap: Colormap name.
        db_floor: dB floor for dB scale.
        dpi: Output resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ra = np.asarray(ra_cart, dtype=np.float64)
    n_range, n_az = ra.shape
    range_depth = n_range * range_res
    range_width = range_depth / 2.0
    extent = [-range_width, range_width, 0.0, range_depth]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

    if scale == "dB":
        ra_max = np.max(ra) if np.max(ra) > 0 else 1.0
        ra_ratio = np.clip(ra / ra_max, 1e-30, None)
        ra_disp = np.clip(20.0 * np.log10(ra_ratio), db_floor, 0.0)
        im = ax.imshow(ra_disp, cmap=cmap, aspect="auto", origin="lower",
                        vmin=db_floor, vmax=0.0, extent=extent)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("dB")
    else:
        mn, mx = ra.min(), ra.max()
        ra_disp = (ra - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(ra)
        im = ax.imshow(ra_disp, cmap=cmap, aspect="auto", origin="lower",
                        vmin=0, vmax=1, extent=extent)
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xlabel("Azimuth (m)")
    ax.set_ylabel("Range (m)")
    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path





def ra_polar_to_cartesian(ra_az_by_range: np.ndarray, range_res: float,
                          grid_res: int = 400, range_bias: float = 0.0) -> np.ndarray:
    """Convert RA map from polar (Az x Range) to cartesian grid (Y,X) like plot_range_azimuth_heatmap_img."""
    # Expect shape (azimuth_bins, range_bins)
    data = np.asarray(ra_az_by_range, dtype=np.float32)
    num_angle_bins = int(data.shape[0]) + 1
    num_adc        = int(data.shape[1])
    data = data.T  # now (range, azimuth)

    # angle/range axes
    t = np.arange(-num_angle_bins//2 + 1, num_angle_bins//2) * (2.0 / float(num_angle_bins))
    t = np.arcsin(t)
    r = np.arange(num_adc, dtype=np.float32) * float(range_res)

    range_depth = float(num_adc) * float(range_res)
    range_width = range_depth / 2.0
    xi = np.linspace(-range_width,  range_width, grid_res)
    yi = np.linspace(0.0,          range_depth, grid_res)
    xi, yi = np.meshgrid(xi, yi)

    x = r[:, None] * np.sin(t)
    y = r[:, None] * np.cos(t) - float(range_bias)

    zi = interpolate.griddata((x.ravel(), y.ravel()), data.ravel(), (xi, yi), method='linear')
    # Drop last row/col to mirror reference and remove potential extrapolated boundary
    zi = zi[:-1, :-1]
    # Replace NaNs with zeros to avoid blank renders
    zi = np.nan_to_num(zi, nan=0.0, posinf=0.0, neginf=0.0)
    # Flip to match plotting in reference (top becomes bottom, bottom becomes top)
    return zi[:, ::-1]












