"""Full Jones BSDF for mmWave radar — pure PyTorch, no DrJit.

Port of mm3DPS/preprocessing/mmwave_jones.py + mmwave_scalar.py.
Uses all 6 ITU parameters.  Full autograd through every path.

Components:
  - Complex Fresnel reflection (amplitude + phase)
  - ITU-R P.2040-4 slab model (multi-layer interference)
  - Kirchhoff Approximation lobe (GGX NDF)
  - Small Perturbation Method lobe (vMF)
  - Directive + Lambertian incoherent lobes
  - Coherent/incoherent split (Rayleigh factor)
  - Jones polarization (s/p decomposition)
  - Coherent backscatter enhancement (CBS)

Returns POWER.  Caller must sqrt() for field amplitude.
"""

import math
import torch
from torch import Tensor
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_LIGHT = 299_792_458.0
WAVELENGTH = 3.9e-3                   # metres (matches the differentiable renderer's WAVELENGTH_77GHZ exactly)
K_WAVE = 2.0 * math.pi / WAVELENGTH  # ~1611 rad/m
INV_PI = 1.0 / math.pi
TWO_PI = 2.0 * math.pi

# Default TX/RX polarization: vertical = (0, 0, 1)
DEFAULT_POL = None  # set lazily on first use


# ===================================================================
#  Complex number helpers (using PyTorch native complex where useful)
# ===================================================================

def _cpx(r: Tensor, i: Tensor) -> Tensor:
    """Build complex tensor from real and imaginary parts."""
    return torch.complex(r.float(), i.float())


def _cpx_abs_sq(z: Tensor) -> Tensor:
    """Power = |z|² = real² + imag²."""
    return z.real ** 2 + z.imag ** 2


# ===================================================================
#  Permittivity → IOR
# ===================================================================



# ===================================================================
#  Fresnel reflection
# ===================================================================





# ===================================================================
#  ITU Slab Fresnel (P.2040-4 Eqs 43-44)
# ===================================================================

def itu_slab_fresnel(
    eps_real: Tensor, eps_imag: Tensor,
    cos_theta_i: Tensor, thickness: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """ITU-R P.2040-4 single-layer slab reflection + transmission.

    Returns (R_TE, R_TM, T_TE, T_TM) as complex tensors.
    """
    cos_i = cos_theta_i.clamp(1e-6, 1.0)
    sin_sq = 1.0 - cos_i ** 2

    # Complex permittivity
    eta = _cpx(eps_real, -eps_imag)

    # √(η - sin²θ)
    a = torch.sqrt(eta - sin_sq.to(torch.complex64))

    cos_c = cos_i.to(torch.complex64)

    # Single-interface Fresnel coefficients
    r_te = (cos_c - a) / (cos_c + a + 1e-10)
    r_tm = (eta * cos_c - a) / (eta * cos_c + a + 1e-10)

    # Phase delay through slab
    q = (TWO_PI / WAVELENGTH) * thickness.to(torch.complex64) * a

    # exp(-jq) and exp(-j2q)
    ejq = torch.exp(-1j * q)
    ej2q = torch.exp(-2j * q)

    # Slab reflection: R = r(1 - exp(-j2q)) / (1 - r²exp(-j2q))
    R_TE = r_te * (1.0 - ej2q) / (1.0 - r_te ** 2 * ej2q + 1e-10)
    R_TM = r_tm * (1.0 - ej2q) / (1.0 - r_tm ** 2 * ej2q + 1e-10)

    # Slab transmission: T = (1 - r²)exp(-jq) / (1 - r²exp(-j2q))
    T_TE = (1.0 - r_te ** 2) * ejq / (1.0 - r_te ** 2 * ej2q + 1e-10)
    T_TM = (1.0 - r_tm ** 2) * ejq / (1.0 - r_tm ** 2 * ej2q + 1e-10)

    return R_TE, R_TM, T_TE, T_TM




# ===================================================================
#  SPM Validity + Parameter Enforcement
# ===================================================================



# ===================================================================
#  Scattering Lobes
# ===================================================================









# ===================================================================
#  Coherent / Incoherent Blend Factors
# ===================================================================

def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x.clamp(-50.0, 50.0))






# ===================================================================
#  Coherent Backscatter Enhancement
# ===================================================================





# ===================================================================
#  Jones Polarization
# ===================================================================

def _get_default_pol(device):
    """Lazy-init default vertical polarization vector."""
    global DEFAULT_POL
    if DEFAULT_POL is None or DEFAULT_POL.device != device:
        DEFAULT_POL = torch.tensor([0.0, 0.0, 1.0], device=device)
    return DEFAULT_POL


def compute_sp_basis(wi: Tensor, n: Tensor) -> Tuple[Tensor, Tensor]:
    """s/p polarization basis for plane of incidence.

    s = (wi × n) / |wi × n|  (perpendicular to incidence plane)
    p = (s × wi) / |s × wi|  (in incidence plane)
    """
    s = torch.cross(wi, n, dim=-1)
    s_norm = s.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    s = s / s_norm

    # Fallback for grazing incidence
    grazing = s_norm.squeeze(-1) < 1e-4
    if grazing.any():
        fallback_ref = torch.tensor([1.0, 0.0, 0.0], device=wi.device).expand_as(wi)
        s_fb = torch.cross(wi, fallback_ref, dim=-1)
        s_fb = s_fb / s_fb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        s = torch.where(grazing.unsqueeze(-1), s_fb, s)

    p = torch.cross(s, wi, dim=-1)
    p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return s, p




# ===================================================================
#  Main BSDF Entry Points
# ===================================================================





# ===================================================================
#  Backward-compatible aliases (used by adc_synthesis.py)
# ===================================================================

# Keep Tier 1 for testing/fallback


KAPPA_77GHZ = K_WAVE  # alias for adc_synthesis.py
