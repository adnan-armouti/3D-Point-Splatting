import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from typing import Dict, Tuple
import open3d as o3d
# Cached Hann windows by (length, device)
_HANN_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}












# Verbatim helper from 05_visualize_voxels.py


# LiDAR point cloud helper (from 09_viz_lidar_mesh_pcl.py)
def load_point_cloud(pcl_path: str) -> o3d.geometry.PointCloud | None:
    if o3d is None:
        print("Error: open3d is required to visualize geometries. pip install open3d")
        return None
    try:
        import os as _os
        ext = _os.path.splitext(pcl_path)[1].lower()

        # Handle .npy files separately
        if ext == '.npy':
            data = np.load(pcl_path)
            if data.shape[1] < 3:
                print(f"[warn] .npy file has insufficient columns: {data.shape}")
                return None
            # Extract XYZ (first 3 columns)
            points = data[:, :3].astype(np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # If there's a 4th column, treat it as intensity
            if data.shape[1] >= 4:
                intensity = data[:, 3].astype(np.float32)
                try:
                    setattr(pcd, 'intensities', intensity)
                    print(f"[loader] Loaded .npy with intensity: n={len(points)}, intensity range=[{float(intensity.min()):.6f}, {float(intensity.max()):.6f}]")
                except Exception:
                    print(f"[loader] Loaded .npy: n={len(points)} (intensity cannot be attached as attribute, skipping)")
            else:
                print(f"[loader] Loaded .npy: n={len(points)} (no intensity)")
            # Assign default color if no colors
            if not pcd.has_colors():
                pcd.paint_uniform_color([0.1, 0.8, 1.0])
            return pcd

        # For other formats, use Open3D's loader
        pcd = o3d.io.read_point_cloud(pcl_path)
        if pcd is None or (not pcd.has_points()):
            print(f"[warn] Point cloud is empty: {pcl_path}")
            return None
        # Attempt to read scalar intensity from Open3D Tensor API and attach
        try:
            import os as _os
            ext = _os.path.splitext(pcl_path)[1].lower()
            if hasattr(o3d, 't') and ext in ('.ply', '.pcd'):
                try:
                    tpcd = o3d.t.io.read_point_cloud(pcl_path)
                    # Common field name is 'intensity'; allow case-insensitive
                    inten = None
                    if 'intensity' in tpcd.point:
                        inten = tpcd.point['intensity'].numpy().reshape(-1)
                        try:
                            if inten.size > 0:
                                print(f"[loader] Tensor 'intensity' detected: min/max/mean={float(np.min(inten)):.6f}/{float(np.max(inten)):.6f}/{float(np.mean(inten)):.6f}")
                        except Exception:
                            pass
                    elif 'Intensity' in tpcd.point:
                        inten = tpcd.point['Intensity'].numpy().reshape(-1)
                        try:
                            if inten.size > 0:
                                print(f"[loader] Tensor 'Intensity' detected: min/max/mean={float(np.min(inten)):.6f}/{float(np.max(inten)):.6f}/{float(np.mean(inten)):.6f}")
                        except Exception:
                            pass
                    if inten is not None:
                        # Prefer converting the tensor PCD to legacy to guarantee alignment of attributes
                        try:
                            pcd_legacy = tpcd.to_legacy()
                            n_t = int(len(pcd_legacy.points))
                            if inten.shape[0] == n_t:
                                pcd = pcd_legacy
                                setattr(pcd, 'intensities', inten.astype(np.float32))
                                print(f"[loader] Attached scalar intensity from tensor API (n={n_t})")
                            else:
                                # Fallback: attach only if it coincidentally matches the initially loaded legacy count
                                if inten.shape[0] == len(pcd.points):
                                    setattr(pcd, 'intensities', inten.astype(np.float32))
                                    print(f"[loader] Attached scalar intensity matching legacy count (n={len(pcd.points)})")
                                else:
                                    print(f"[loader] WARNING: intensity length {inten.shape[0]} mismatches legacy({len(pcd.points)}) and tensor({n_t}); skipping attach")
                        except Exception as _e_to_legacy:
                            # As a last resort, try to attach to the already-loaded legacy if counts match
                            if inten.shape[0] == len(pcd.points):
                                setattr(pcd, 'intensities', inten.astype(np.float32))
                                print(f"[loader] Attached scalar intensity to legacy (to_legacy failed): n={len(pcd.points)}")
                except Exception:
                    pass
        except Exception:
            pass
        # If no colors, assign a pleasant cyan-like color
        if not pcd.has_colors():
            pts = np.asarray(pcd.points)
            if pts.size > 0:
                pcd.paint_uniform_color([0.1, 0.8, 1.0])
        # Debug summary about loaded PCD
        try:
            n_pts = int(len(pcd.points))
            has_cols = bool(pcd.has_colors())
            has_scalar = bool(hasattr(pcd, 'intensities') and len(pcd.intensities) == n_pts)
            inten_stats = None
            if has_scalar:
                vals = np.asarray(pcd.intensities, dtype=np.float64).reshape(-1)
                if vals.size > 0 and np.all(np.isfinite(vals)):
                    inten_stats = (float(vals.min()), float(vals.max()), float(vals.mean()))
            gray_stats = None
            if has_cols:
                cols = np.asarray(pcd.colors, dtype=np.float64)
                if cols.ndim == 2 and cols.shape[0] == n_pts and cols.shape[1] >= 3:
                    gray = (0.2126 * cols[:, 0] + 0.7152 * cols[:, 1] + 0.0722 * cols[:, 2]).astype(np.float64)
                    if gray.size > 0 and np.all(np.isfinite(gray)):
                        gray_stats = (float(gray.min()), float(gray.max()), float(gray.mean()))
            print(f"[loader] Loaded PCD: n={n_pts}, has_colors={has_cols}, has_scalar_intensity={has_scalar}")
            if inten_stats is not None:
                print(f"[loader] Scalar intensity stats: min/max/mean={inten_stats[0]:.6f}/{inten_stats[1]:.6f}/{inten_stats[2]:.6f}")
            if gray_stats is not None:
                print(f"[loader] Color->gray stats: min/max/mean={gray_stats[0]:.6f}/{gray_stats[1]:.6f}/{gray_stats[2]:.6f}")
        except Exception:
            pass
        return pcd
    except Exception as e:
        print(f"[warn] Failed to load point cloud '{pcl_path}': {e}")
        return None


# Extract per-point intensity from an Open3D point cloud


# Helpers analogous to 11_lidar_visualization_new.py, adapted for Open3D







# Consistent board rotation from boresight (matches other viewers)

# RA-style angle grids (DC-removed centers) as used for wireframes/radar
def make_angle_grids_np(az_bins_full: int, el_bins_full: int):
    eps = 1e-6
    t_az = np.arange(-az_bins_full // 2 + 1, az_bins_full // 2, dtype=np.float64) * (2.0 / float(az_bins_full))
    t_el = np.arange(-el_bins_full // 2 + 1, el_bins_full // 2, dtype=np.float64) * (2.0 / float(el_bins_full))
    t_az = np.clip(t_az, -1.0 + eps, 1.0 - eps)
    t_el = np.clip(t_el, -1.0 + eps, 1.0 - eps)
    return np.arcsin(t_az), np.arcsin(t_el)

def _centers_to_edges(centers: np.ndarray, low_clip: float, high_clip: float) -> np.ndarray:
    centers = centers.astype(np.float64)
    if centers.size == 1:
        width = 1e-3
        return np.array([centers[0] - width, centers[0] + width], dtype=np.float64)
    diffs = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + 0.5 * diffs
    edges[0] = centers[0] - 0.5 * diffs[0]
    edges[-1] = centers[-1] + 0.5 * diffs[-1]
    edges[0] = max(edges[0], low_clip)
    edges[-1] = min(edges[-1], high_clip)
    return edges


