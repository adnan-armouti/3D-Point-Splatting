"""Radar sensor alignment subpackage.

Provides two alignment methods for the cascaded 77 GHz FMCW radar
and an orchestrator that runs both and selects the per-scene winner:

1. **LiDAR-based 4-DOF** (``cascaded_lidar`` / ``cascaded_lidar_gpu``):
   Voxelises a LiDAR point cloud into the radar's range-azimuth-elevation
   grid and optimises range, azimuth, elevation-rotation, and
   azimuth-rotation to maximise correlation with the measured RA image.

2. **Renderer-based 2-DOF** (``cascaded_renderer``):
   Uses the differentiable renderer to render RA images and optimises only
   range offset and azimuth offset (preserving the IMU-derived boresight).

The recommended entry point is ``cascaded_alignment``, which runs both
methods (or loads existing results) and picks the winner per scene.

Modules
-------
cascaded_alignment  Orchestrator: runs both methods, selects winner per scene
cascaded_lidar      Shared helpers for LiDAR alignment
cascaded_lidar_gpu  LiDAR 4-DOF alignment (requires CuPy)
cascaded_renderer   Renderer-based 2-DOF alignment
gpu_utils/          CuPy-accelerated voxelization, metrics, and grid search
"""
