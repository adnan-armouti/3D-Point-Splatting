import json
import numpy as np
from typing import Optional, Tuple, Any, List

try:
    import open3d as o3d
except Exception:
    o3d = None





# Import RA utils for cartesian conversion and image saving


def load_config_positions_and_boresight(config_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if config_path is None:
        return None, None, None, None
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        tx_m = np.array([t['pos_mm'] for t in cfg['tx_array']], dtype=float) / 1000.0
        rx_m = np.array([r['pos_mm'] for r in cfg['rx_array']], dtype=float) / 1000.0
        center = np.mean(np.vstack([tx_m, rx_m]), axis=0)
        bore = None
        if 'boresight' in cfg['tx_array'][0]:
            bore = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
            n = float(np.linalg.norm(bore))
            if n > 1e-9:
                bore = bore / n
            else:
                bore = None
        return tx_m, rx_m, center, bore
    except Exception:
        return None, None, None, None




















def build_scene(geoms: List[Any], add_axes_at: Optional[np.ndarray]) -> None:
    if o3d is None:
        return

    # Add coordinate frame if requested
    if add_axes_at is not None:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # Interactive remote visualization removed for the public release.
    # Save geometries to disk if needed; downstream code can load them with Open3D.
    print("  [INFO] interactive visualization is disabled in the release build")



    


