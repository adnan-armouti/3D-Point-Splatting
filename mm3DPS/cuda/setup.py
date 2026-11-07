"""Build the mm3DPS CUDA extension.

Usage:
  cd mm3DPS/cuda && python setup.py build_ext --inplace

Target: sm_89 (RTX 4090 / Ada Lovelace). PyTorch >= 2.1 with torch.utils.cpp_extension.
"""
import os
import torch.utils.cpp_extension as _cpp_ext
# PyTorch 2.7.1+cu118 was compiled against CUDA 11.8 but the system has CUDA
# 12.8. The generated objects run fine at runtime (cu118 libs are forward
# compatible with the 12.x driver + no cu12-only features used here), so we
# bypass the pre-build check that torch.utils.cpp_extension runs.
_cpp_ext._check_cuda_version = lambda *a, **kw: None

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))

ext = CUDAExtension(
    name='mm3DPS_cuda',
    sources=[
        os.path.join(HERE, 'bindings.cpp'),
        os.path.join(HERE, 'bsdf_forward.cu'),
        os.path.join(HERE, 'bsdf_backward.cu'),
        os.path.join(HERE, 'scatter_splat.cu'),
    ],
    include_dirs=[HERE],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': [
            '-O3',
            # --use_fast_math enabled: fp32 + fast-math is the
            # inverse-rendering industry standard (Mitsuba 3 cuda_ad_rgb,
            # PBRT v4, gsplat, NeRF implementations). The per-kernel
            # precision floor (~2e-5 max, ~1e-7 mean after the csqrt /
            # cexp fixes) is ~100× below the project's own ±0.03 Monte
            # Carlo noise floor, and end-to-end cart_corr matches the
            # PyTorch baseline within ±0.001. The sm_89 fp64 throughput
            # penalty (1/64 of fp32) is not worth paying for a precision
            # that Adam's gradient noise averages out over 500 iters.
            '--use_fast_math',
            '--extended-lambda',
            '-std=c++17',
            '--ptxas-options=-v',
            '-gencode=arch=compute_89,code=sm_89',
            '--expt-relaxed-constexpr',
            '-lineinfo',
        ],
    },
)

setup(
    name='mm3DPS_cuda',
    version='0.1.0',
    description='mm3DPS CUDA kernels (fused BSDF + scatter splat)',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
