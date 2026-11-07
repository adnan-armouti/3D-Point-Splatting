import os
import numpy as np
from typing import List

from mm3DPS.preprocessing.ColoRadar_tools.dataset_loaders import (
    get_lidar_params,
    get_cascade_params,
    get_timestamps,
    get_groundtruth_params,
    get_groundtruth,
    get_pointcloud,
    get_heatmap,
)
from mm3DPS.preprocessing.ColoRadar_tools.plot_pointclouds import interpolate_poses, transform_pcl, polar_to_cartesian
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from mm3DPS.preprocessing.io_paths import get_base_dir, get_subdir, find_lowest_cascaded_config
from mm3DPS.preprocessing.config_utils import get_radar_config_info
import math
import torch
import faiss
from tqdm import tqdm


def map_cascade_to_lidar(seq_path: str,
                         calib_path: str,
                         cascade_frame_idx: int,
                         max_time_diff: float | None = 0.5) -> int:
    lidar_params = get_lidar_params(calib_path)
    radar_params = get_cascade_params(calib_path)["heatmap"]
    cas_ts = np.asarray(get_timestamps(seq_path, radar_params), dtype=float)
    lid_ts = np.asarray(get_timestamps(seq_path, lidar_params), dtype=float)
    if cascade_frame_idx < 0 or cascade_frame_idx >= len(cas_ts):
        raise IndexError("cascade_frame_idx out of range")
    ref = cas_ts[cascade_frame_idx]
    diffs = np.abs(lid_ts - ref)
    li = int(np.argmin(diffs))
    if (max_time_diff is not None) and (diffs[li] > max_time_diff):
        raise ValueError(
            f"No LiDAR frame within {max_time_diff:.3f}s (closest Deltat {float(diffs[li]):.3f}s)"
        )
    return li


def get_lidar_world_transforms(seq_path: str,
                               calib_path: str) -> List[np.ndarray]:
    lidar_params = get_lidar_params(calib_path)
    gt_params = get_groundtruth_params()
    gt_ts = np.asarray(get_timestamps(seq_path, gt_params), dtype=float)
    lid_ts = np.asarray(get_timestamps(seq_path, lidar_params), dtype=float)
    gt_poses = get_groundtruth(seq_path)

    lid_gt, _ = interpolate_poses(gt_poses, gt_ts, lid_ts)
    T_bs = np.eye(4)
    T_bs[:3, 3] = lidar_params["translation"]
    T_bs[:3, :3] = Rotation.from_quat(lidar_params["rotation"]).as_matrix()
    return [T_wb @ T_bs for T_wb in lid_gt]


def load_lidar_frame_world(seq_path: str,
                           calib_path: str,
                           lid_idx: int) -> np.ndarray:
    cloud_local = get_pointcloud(lid_idx, seq_path, get_lidar_params(calib_path))
    T_ws_list = get_lidar_world_transforms(seq_path, calib_path)
    T_ws = T_ws_list[lid_idx]
    cloud_world = (T_ws @ np.hstack([cloud_local[:, :3], np.ones((cloud_local.shape[0], 1))]).T).T[:, :3]
    intensity = cloud_local[:, 3:4] if cloud_local.shape[1] >= 4 else np.zeros((cloud_local.shape[0], 1), dtype=cloud_local.dtype)
    return np.hstack([cloud_world, intensity]).astype(np.float32)


def save_lidar_frame(base_dir: str, lid_idx: int, cloud_world_xyzi: np.ndarray) -> str:
    lidar_dir = get_subdir(base_dir, "lidar")
    path = os.path.join(lidar_dir, f"lidar_frame_{lid_idx}.npy")
    np.save(path, cloud_world_xyzi)
    return path




def generate_lidar_window(*,
                          seq_idx: int,
                          center_frame_idx: int,
                          num_radar_frames: int,
                          dataset_dir: str,
                          calib_path: str,
                          out_root: str,
                          verbose: bool = False) -> list[str]:
    if num_radar_frames < 1 or (num_radar_frames % 2) != 1:
        raise ValueError("num_radar_frames must be odd and >= 1")

    seq_path = dataset_dir + str(seq_idx)
    base_dir = get_base_dir(seq_idx, center_frame_idx, out_root)
    half = num_radar_frames // 2
    cascade_indices = list(range(center_frame_idx - half, center_frame_idx + half + 1))

    out_paths: list[str] = []
    for cas_idx in cascade_indices:
        lid_idx = map_cascade_to_lidar(seq_path, calib_path, cas_idx, max_time_diff=0.5)
        cloud_world_xyzi = load_lidar_frame_world(seq_path, calib_path, lid_idx)
        out_paths.append(save_lidar_frame(base_dir, lid_idx, cloud_world_xyzi))
        if verbose:
            print(f"[lidar] cascade {cas_idx} -> lidar {lid_idx} saved")
    return out_paths


def build_and_save_lidar_scene(*,
                               seq_idx: int,
                               frame_idx: int,
                               dataset_dir: str,
                               calib_path: str,
                               out_root: str,
                               num_lidar_frames: int = 50,
                               buffer_distance: float = 1.0,
                               normals_radius: float = 0.1,
                               remove_behind_radar: bool = False,
                               config_path: str | None = None,
                               verbose: bool = False) -> str:
    # Fuse LiDAR frames
    fused_world_unique, sensor_positions = generate_fused_lidar_pointcloud(
        seq_idx=seq_idx,
        dataset_dir=dataset_dir,
        calib_path=calib_path,
        radar_mid_idx=frame_idx,
        num_lidar_frames=num_lidar_frames,
        verbose=verbose,
    )

    # Optionally remove points behind radar using lowest cascaded config under base dir
    base_dir = get_base_dir(seq_idx, frame_idx, out_root)
    if remove_behind_radar:
        cfg_to_use = config_path or find_lowest_cascaded_config(base_dir)
        if cfg_to_use is None:
            if verbose:
                print("[scene] No cascaded config found under configs/. Skipping behind-radar removal.")
            filtered_points = fused_world_unique
        else:
            radar_center, boresight, _, _ = get_radar_config_info(cfg_to_use)
            filtered_points, _ = remove_points_behind_radar(
                points=fused_world_unique,
                radar_position=radar_center,
                radar_direction=boresight,
                buffer_distance=buffer_distance,
            )
    else:
        filtered_points = fused_world_unique

    # Normals estimation and orientation
    normals = estimate_normals_gpu(
        points=filtered_points[:, :3],
        mode="radius",
        radius=normals_radius,
        use_octree_like=True,
        orient_toward_origin=False,
        origin=None,
        faiss_devices=[0],
        verbose=verbose,
    )
    normals_oriented = orient_normals_toward_sensors_gpu(
        filtered_points[:, :3], normals, sensor_positions, device="cuda:0"
    )

    # Assemble and save scene
    xyz = filtered_points[:, :3].astype(np.float32)
    inten = filtered_points[:, 3:4].astype(np.float32)
    scene = np.concatenate([xyz, normals_oriented.astype(np.float32), inten], axis=1)

    scene_dir = get_subdir(base_dir, "scene")
    save_path = os.path.join(scene_dir, "pcl.npy")
    np.save(save_path, scene)
    if verbose:
        print(f"[scene] saved: {save_path} N={scene.shape[0]}")
    return save_path


# -------------------- Moved from pcl_utils.py (LiDAR helpers) --------------------

def get_heatmap_points(params, min_range):
    pcl = np.zeros([params['num_elevation_bins'], params['num_azimuth_bins'], params['num_range_bins'] - min_range, 5])
    for range_idx in range(params['num_range_bins'] - min_range):
        for az_idx in range(params['num_azimuth_bins']):
            for el_idx in range(params['num_elevation_bins']):
                pcl[el_idx, az_idx, range_idx, :3] = polar_to_cartesian(range_idx + min_range, az_idx, el_idx, params)
    pcl = pcl.reshape(-1, 5)
    return pcl


def get_radar_fov_bbox(seq_path, calib_path, radar_frame_idx, min_range_bin=10, intensity_thr=0.0):
    radar_params = get_cascade_params(calib_path)['heatmap']
    T_bs = np.eye(4)
    T_bs[:3, 3] = radar_params['translation']
    T_bs[:3, :3] = Rotation.from_quat(radar_params['rotation']).as_matrix()
    radar_params['T_bs'] = T_bs

    gt_params = get_groundtruth_params()
    gt_ts = np.asarray(get_timestamps(seq_path, gt_params), dtype=float)
    radar_ts = np.asarray(get_timestamps(seq_path, radar_params), dtype=float)
    gt_poses = get_groundtruth(seq_path)

    radar_gt, _ = interpolate_poses(gt_poses, gt_ts, radar_ts)
    if radar_frame_idx >= len(radar_gt):
        raise IndexError("radar_frame_idx out of range")

    T_wb = radar_gt[radar_frame_idx]

    vox = get_heatmap_points(radar_params, min_range_bin)
    hm = get_heatmap(radar_frame_idx, seq_path, radar_params)
    vox[:, 3:] = hm[:, :, min_range_bin:, :].reshape(-1, 2)

    vox[:, 3] -= vox[:, 3].min()
    vox[:, 3] /= (vox[:, 3].ptp() + 1e-9)
    vox = vox[vox[:, 3] > intensity_thr]

    T_ws = T_wb @ T_bs
    vox_world = transform_pcl(vox, T_ws)

    x_min, x_max = vox_world[:, 0].min(), vox_world[:, 0].max()
    y_min, y_max = vox_world[:, 1].min(), vox_world[:, 1].max()
    z_min, z_max = vox_world[:, 2].min(), vox_world[:, 2].max()

    return dict(
        x_min=float(x_min), x_max=float(x_max),
        y_min=float(y_min), y_max=float(y_max),
        z_min=float(z_min), z_max=float(z_max)
    )


def find_matching_lidar_frame(seq_path: str, radar_params: dict, lidar_params: dict, radar_frame_idx: int, max_time_diff=None):
    radar_ts = np.asarray(get_timestamps(seq_path, radar_params), dtype=float)
    lidar_ts = np.asarray(get_timestamps(seq_path, lidar_params), dtype=float)
    if radar_frame_idx < 0 or radar_frame_idx >= len(radar_ts):
        raise IndexError("radar_frame_idx out of range")
    ref = radar_ts[radar_frame_idx]
    diffs = np.abs(lidar_ts - ref)
    lidar_idx = int(np.argmin(diffs))
    if (max_time_diff is not None) and (diffs[lidar_idx] > max_time_diff):
        raise ValueError("No LiDAR frame within {:.3f}s (closest Deltat {:.3f}s)".format(max_time_diff, diffs[lidar_idx]))
    return lidar_idx


def generate_fused_lidar_pointcloud(seq_idx=1,
    dataset_dir=None,
    calib_path=None,
    radar_mid_idx=182,
    num_lidar_frames=50,
    vox_size=0.0005,
    min_range_bin=1,
    intensity_thr=0.0,
    verbose: bool = True,):
    seq_path = dataset_dir + str(seq_idx)
    lidar_params = get_lidar_params(calib_path)
    radar_params = get_cascade_params(calib_path)['heatmap']

    radar_start_idx = radar_mid_idx - (num_lidar_frames // 2)
    lidar_start_idx = find_matching_lidar_frame(seq_path, radar_params, lidar_params, radar_start_idx, max_time_diff=0.5)
    lidar_mid_idx = find_matching_lidar_frame(seq_path, radar_params, lidar_params, radar_mid_idx, max_time_diff=0.5)
    radar_stop_idx = radar_mid_idx + (num_lidar_frames // 2)
    lidar_stop_idx = find_matching_lidar_frame(seq_path, radar_params, lidar_params, radar_stop_idx, max_time_diff=0.5)

    print(
        f"Closest start LiDAR frame: {lidar_start_idx} | "
        f"Closest mid LiDAR frame: {lidar_mid_idx} | "
        f"Closest stop LiDAR frame: {lidar_stop_idx}")

    radar_frames = np.arange(radar_start_idx, radar_stop_idx, 1).tolist()
    fov_boxes = [
        get_radar_fov_bbox(seq_path=seq_path, calib_path=calib_path, radar_frame_idx=rf_idx, min_range_bin=min_range_bin, intensity_thr=intensity_thr)
        for rf_idx in radar_frames
    ]
    master_bbox = {k: (min if k.endswith('min') else max)(box[k] for box in fov_boxes) for k in ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']}
    if verbose:
        print("Master (union) radar-FOV bbox:")
        for k, v in master_bbox.items():
            print(f"  {k}: {v:.3f}")
        print(f"master_bbox.keys(): {master_bbox.keys()} ")

    def _crop_world(cloud_world, bbox):
        m = (
            (cloud_world[:, 0] > bbox['x_min']) & (cloud_world[:, 0] < bbox['x_max']) &
            (cloud_world[:, 1] > bbox['y_min']) & (cloud_world[:, 1] < bbox['y_max']) &
            (cloud_world[:, 2] > bbox['z_min']) & (cloud_world[:, 2] < bbox['z_max'])
        )
        return cloud_world[m]

    _lidar_params_cached = get_lidar_params(calib_path)
    _T_bs_cached = np.eye(4)
    _T_bs_cached[:3, 3] = _lidar_params_cached['translation']
    _T_bs_cached[:3, :3] = Rotation.from_quat(_lidar_params_cached['rotation']).as_matrix()
    _gt_params_cached = get_groundtruth_params()
    _gt_ts_cached = np.asarray(get_timestamps(seq_path, _gt_params_cached), dtype=float)
    _lidar_ts_cached = np.asarray(get_timestamps(seq_path, _lidar_params_cached), dtype=float)
    _gt_poses_cached = get_groundtruth(seq_path)
    _lidar_gt_cached, _ = interpolate_poses(_gt_poses_cached, _gt_ts_cached, _lidar_ts_cached)

    accum = []
    lidar_positions = []
    indices = range(lidar_start_idx, lidar_stop_idx + 1)
    iterator = indices
    pbar = None
    if not verbose and 'tqdm' in globals() and tqdm is not None:
        pbar = tqdm(indices, desc="Fusing LiDAR frames", unit="frame")
        iterator = pbar

    for idx in iterator:
        T_wb = _lidar_gt_cached[idx]
        T_ws = T_wb @ _T_bs_cached
        sensor_position = T_ws[:3, 3]
        lidar_positions.append(sensor_position)

        cloud_local = get_pointcloud(idx, seq_path, _lidar_params_cached)
        cloud_w = transform_pcl(cloud_local, T_ws)
        cropped = _crop_world(cloud_w, master_bbox)
        accum.append(cropped)
        if verbose:
            print(f"frame {idx:4d}  pts loaded {cloud_w.shape[0]:5d}  cropped {cropped.shape[0]:5d}  sensor pos=({sensor_position[0]:.2f}, {sensor_position[1]:.2f}, {sensor_position[2]:.2f})")
        elif pbar is not None:
            pbar.set_postfix({
                "loaded": int(cloud_w.shape[0]),
                "kept": int(cropped.shape[0]),
                "x": f"{sensor_position[0]:.2f}",
                "y": f"{sensor_position[1]:.2f}",
                "z": f"{sensor_position[2]:.2f}",
            })
    if pbar is not None:
        pbar.close()

    fused_world = np.vstack(accum)
    print("Total fused points:", fused_world.shape[0])
    uniq_rows, uniq_idx = np.unique(fused_world, axis=0, return_index=True)
    fused_world_unique = fused_world[np.sort(uniq_idx)]
    print(f"Removed {fused_world.shape[0] - fused_world_unique.shape[0]} exact duplicate points; new total: {fused_world_unique.shape[0]}")

    # Remove any sensor-position points accidentally present in the fused array
    try:
        pts = fused_world_unique[:, :3]
        sensor_pts = np.asarray(lidar_positions, dtype=np.float64)
        if pts.size == 0 or sensor_pts.size == 0:
            if verbose:
                print("[check] Skipped sensor-in-array check: empty input.")
        else:
            # Check if sensor positions appear exactly (within tol) in fused points
            tree_pts = cKDTree(pts)
            dists, _ = tree_pts.query(sensor_pts, k=1)
            tol = 1e-9
            in_mask = dists <= tol
            num_present = int(np.count_nonzero(in_mask))
            num_total = int(sensor_pts.shape[0])
            if verbose:
                print(f"[check] Sensor positions present in fused_world_unique within tol {tol:g}: {num_present}/{num_total}")

            # Remove all fused points that coincide with any sensor position (within tol)
            tree_sens = cKDTree(sensor_pts)
            d_ps, _ = tree_sens.query(pts, k=1)
            rm_mask = d_ps <= tol
            removed = int(np.count_nonzero(rm_mask))
            if removed > 0:
                fused_world_unique = fused_world_unique[~rm_mask]
            if verbose:
                print(f"[remove] tol={tol:g} -> removed {removed} rows from fused_world_unique matching sensor positions. New count: {fused_world_unique.shape[0]}")
    except Exception as e:
        if verbose:
            print(f"[warn] Sensor-in-array check failed: {e}")

    return fused_world_unique, np.array(lidar_positions)


def remove_points_behind_radar(points, radar_position, radar_direction=None, buffer_distance=0.0):
    points = np.asarray(points, dtype=np.float64)
    radar_pos = np.asarray(radar_position, dtype=np.float64)
    if points.shape[1] not in [3, 4]:
        raise ValueError("points must be (N, 3) or (N, 4)")
    if radar_pos.shape != (3,):
        raise ValueError("radar_position must be (3,)")
    coords = points[:, :3]
    if radar_direction is None:
        radar_dir = radar_pos - np.array([0.0, 0.0, 0.0])
        radar_dir = radar_dir / (np.linalg.norm(radar_dir) + 1e-12)
    else:
        radar_dir = np.asarray(radar_direction, dtype=np.float64)
        radar_dir = radar_dir / (np.linalg.norm(radar_dir) + 1e-12)
    point_vectors = coords - radar_pos
    distances_from_radar = np.linalg.norm(point_vectors, axis=1)
    projections = np.dot(point_vectors, radar_dir)
    keep_mask = (projections >= -buffer_distance) | (distances_from_radar <= buffer_distance)
    filtered_points = points[keep_mask]
    print(f"Removed {np.sum(~keep_mask)} points behind radar")
    print(f"Kept {np.sum(keep_mask)} points")
    return filtered_points, keep_mask


@torch.no_grad()
def orient_normals_toward_sensors_gpu(points: np.ndarray, normals: np.ndarray, sensor_positions: np.ndarray, device: str = "cuda", chunk: int = 1_000_000) -> np.ndarray:
    if points.size == 0 or normals.size == 0 or sensor_positions.size == 0:
        return normals.astype(np.float32, copy=True)
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    nrm = torch.as_tensor(normals, dtype=torch.float32, device=device)
    sens = torch.as_tensor(sensor_positions, dtype=torch.float32, device=device)
    N = pts.shape[0]
    out = torch.empty_like(nrm)
    nrm = nrm / (torch.linalg.norm(nrm, dim=1, keepdim=True).clamp_min(1e-12))
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        P = pts[s:e]
        Nn = nrm[s:e]
        d = torch.cdist(P, sens)
        j = torch.argmin(d, dim=1)
        S = sens[j]
        v = S - P
        flip = ((Nn * v).sum(dim=1) < 0)
        Nn[flip] = -Nn[flip]
        out[s:e] = Nn
    return out.detach().cpu().numpy().astype(np.float32)


def estimate_normals_gpu(points: np.ndarray, mode: str = "k", k: int = 30, radius: float | str = "auto", auto_radius_frac: float = 0.01, use_octree_like: bool = True, bucket_size: int = 32, orient_toward_origin: bool = True, origin: np.ndarray | None = None, require_min_neighbors: int = 3, max_batch: int = 262144, faiss_nprobe: int = 64, faiss_devices: list[int] | None = None, verbose: bool = True,) -> np.ndarray:
    P = np.asarray(points, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be (N,3) array")
    N = P.shape[0]
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if mode == "radius":
        if isinstance(radius, str) and radius.lower() == "auto":
            mn = P.min(axis=0); mx = P.max(axis=0)
            diag = float(np.linalg.norm(mx - mn))
            radius = float(diag * auto_radius_frac) if diag > 0 else 0.0
        radius = float(radius)
        if radius <= 0:
            raise ValueError("radius must be > 0 (or 'auto') in radius mode")
    if faiss_devices is None:
        faiss_devices = [0]
    d = 3
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.add(P)
    co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
    try:
        res_list = [faiss.StandardGpuResources() for _ in faiss_devices]
        devs = np.asarray(faiss_devices, dtype=np.int32)
        gpu_index = faiss.index_cpu_to_gpu_multiple(res_list, devs, cpu_index, co)
    except Exception:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            res_list = [res]
        except Exception as e:
            raise RuntimeError("Failed to initialize FAISS GPU index") from e

    if orient_toward_origin:
        O = np.zeros(3, dtype=np.float32) if origin is None else np.asarray(origin, np.float32)
    else:
        O = None

    def normals_from_neighbors(idx_batch: np.ndarray, mask_batch: np.ndarray | None = None) -> np.ndarray:
        device = torch.device("cuda")
        idx_t = torch.from_numpy(idx_batch.astype(np.int64)).to(device)
        pts_t = torch.from_numpy(P).to(device)
        neigh = pts_t[idx_t.clip(min=0)]
        if mask_batch is not None:
            m = torch.from_numpy(mask_batch).to(device)
            neigh = torch.where(m[..., None], neigh, torch.zeros_like(neigh))
            denom = torch.clamp(m.sum(dim=1, keepdim=True).to(torch.float32), min=1.0)
        else:
            denom = torch.tensor(idx_batch.shape[1], device=device, dtype=torch.float32).view(-1, 1)
        mean = neigh.sum(dim=1, keepdim=True) / denom[:, None, :]
        X = neigh - mean
        cov = torch.bmm(X.transpose(1, 2), X)
        evals, evecs = torch.linalg.eigh(cov)
        n = evecs[:, :, 0]
        if O is not None:
            Q = pts_t[:idx_batch.shape[0]]
            v = (torch.from_numpy(O).to(device) - Q)
            flip = (n * v).sum(dim=1) < 0
            n[flip] = -n[flip]
        norm = torch.linalg.norm(n, dim=1, keepdim=True).clamp_min_(1e-12)
        n = n / norm
        return n.detach().cpu().numpy().astype(np.float32)

    def make_radius_neighbors():
        vol = (4.0/3.0) * math.pi * (radius ** 3)
        density = N / max(1e-9, float((P.max(0) - P.min(0)).prod()))
        k_cap = int(min(max(32, math.ceil(2.5 * density * vol)), 4096))
        if verbose:
            print(f"[radius] using k_cap={k_cap} (density={density:.3f}, vol={vol:.6f})")
        all_idx = np.empty((N, k_cap), dtype=np.int32)
        all_dist2 = np.empty((N, k_cap), dtype=np.float32)
        for s in tqdm(range(0, N, max_batch), desc="faiss search (radius)", unit="pts"):
            e = min(N, s + max_batch)
            D, I = gpu_index.search(P[s:e], k_cap)
            all_idx[s:e] = I.astype(np.int32)
            all_dist2[s:e] = D.astype(np.float32)
        rad2 = radius * radius
        mask = all_dist2 <= rad2
        mask &= (all_idx != np.arange(N, dtype=np.int32)[:, None])
        return all_idx, mask

    def make_k_neighbors():
        kk = int(max(1, k + 1))
        all_idx = np.empty((N, kk), dtype=np.int32)
        for s in tqdm(range(0, N, max_batch), desc="faiss search (k)", unit="pts"):
            e = min(N, s + max_batch)
            D, I = gpu_index.search(P[s:e], kk)
            all_idx[s:e] = I.astype(np.int32)
        row = np.arange(N, dtype=np.int32)
        self_mask = (all_idx == row[:, None])
        compact = np.empty((N, k), dtype=np.int32)
        for i in tqdm(range(N), desc="postprocess (drop self)", disable=(N > 5_000_000)):
            m = ~self_mask[i]
            compact[i] = all_idx[i][m][:k]
        return compact

    if mode.lower() == "k":
        nbr_idx = make_k_neighbors()
        if require_min_neighbors > 1 and k < require_min_neighbors:
            raise ValueError(f"require_min_neighbors={require_min_neighbors} but k={k}")
        normals = np.zeros((N, 3), dtype=np.float32)
        for s in tqdm(range(0, N, max_batch), desc="compute normals (k)", unit="pts"):
            e = min(N, s + max_batch)
            n = normals_from_neighbors(nbr_idx[s:e])
            normals[s:e] = n
        return normals
    elif mode.lower() == "radius":
        idx, mask = make_radius_neighbors()
        valid_counts = mask.sum(axis=1)
        normals = np.zeros((N, 3), dtype=np.float32)
        for s in tqdm(range(0, N, max_batch), desc="compute normals (radius)", unit="pts"):
            e = min(N, s + max_batch)
            n = normals_from_neighbors(idx[s:e], mask[s:e])
            normals[s:e] = n
        normals[valid_counts < max(require_min_neighbors, 1)] = 0.0
        return normals
    else:
        raise ValueError("mode must be 'k' or 'radius'")


