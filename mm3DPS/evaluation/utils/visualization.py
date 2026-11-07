"""Figure generation utilities for paper publication."""

import os
from typing import Optional

import numpy as np








def save_ra_cartesian_png(
    ra_cart: np.ndarray,
    output_path: str,
    range_res: float = 0.0557,
    scale: str = "dB",
    title: Optional[str] = None,
    cmap: str = "hot",
    db_floor: float = -40.0,
    dpi: int = 150,
):
    """Save a single RA Cartesian image with proper axis labels.

    X-axis = azimuth (cross-range, meters), Y-axis = range (meters).

    Args:
        ra_cart: 2D Cartesian RA magnitude array (rows=range, cols=azimuth).
        output_path: Path to save PNG.
        range_res: Range resolution in meters (for axis extents).
        scale: 'dB' or 'linear'.
        title: Optional figure title.
        cmap: Colormap name.
        db_floor: dB floor for dB scale.
        dpi: Output resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ra = np.asarray(ra_cart, dtype=np.float64)

    # Compute physical extent: rows=range, cols=azimuth
    n_range, n_az = ra.shape
    num_adc = n_range  # Cartesian grid matches range extent
    range_depth = num_adc * range_res
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




