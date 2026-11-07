"""mm3DPS CUDA extension loader + autograd.Function wrappers.

Tries to import the compiled extension. If it isn't built yet, sets the
module-level ``ext`` to None so callers can fall back to the PyTorch path.

Build:
    cd mm3DPS/cuda
    python setup.py build_ext --inplace
"""

import importlib
import importlib.util
import os
import sys

# torch must be imported before the extension .so so that libc10/libtorch are
# preloaded into the process (otherwise ImportError on libc10.so).
import torch  # noqa: F401

__all__ = [
    "ext",
    "is_available",
    "scatter_splat",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
ext = None
_LOAD_ERROR = None


def _load_extension():
    """Load the compiled extension from this directory (build_ext --inplace
    drops the .so next to setup.py). Falls back to sys.path if not found.
    """
    # First, try to find an .so next to this __init__.py
    for fname in os.listdir(_HERE):
        if fname.startswith("mm3DPS_cuda") and fname.endswith(".so"):
            so_path = os.path.join(_HERE, fname)
            spec = importlib.util.spec_from_file_location(
                "mm3DPS_cuda", so_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["mm3DPS_cuda"] = mod
            return mod
    # Fallback: sys.path import
    return importlib.import_module("mm3DPS_cuda")


try:
    ext = _load_extension()
except Exception as e:  # ImportError, OSError, FileNotFoundError, etc.
    ext = None
    _LOAD_ERROR = repr(e)


def is_available() -> bool:
    """True if the compiled CUDA extension is importable."""
    return ext is not None


def load_error() -> str:
    """Return the last import error (useful for debugging) or '' if loaded."""
    return _LOAD_ERROR or ""


def scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag):
    """In-place atomic scatter splat. Equivalent to:

        rp_real.view(-1).scatter_add_(0, flat_idx, contrib_real)
        rp_imag.view(-1).scatter_add_(0, flat_idx, contrib_imag)

    but issues a single fused CUDA kernel launch.
    """
    if ext is None:
        raise RuntimeError(
            "mm3DPS_cuda extension is not built. "
            "Run: cd mm3DPS/cuda && python setup.py build_ext --inplace"
        )
    ext.scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag)


class ScatterSplatFn(torch.autograd.Function):
    """Autograd wrapper for the scatter_splat CUDA kernel.

    Forward: writes contrib_real / contrib_imag into rp_real / rp_imag
    via atomicAdd, returning the accumulated rp tensors.

    Backward: grad flows to contrib_real / contrib_imag via a gather
    (`grad_contrib[i] = grad_rp.view(-1)[flat_idx[i]]`) — this is
    PyTorch's standard scatter_add backward, just faster because we do
    a single fused gather per output instead of two.
    """

    @staticmethod
    def forward(ctx, contrib_real, contrib_imag, flat_idx,
                out_shape_tx, out_shape_rx, out_shape_k):
        contrib_real = contrib_real.contiguous()
        contrib_imag = contrib_imag.contiguous()
        flat_idx     = flat_idx.contiguous()
        rp_real = torch.zeros(out_shape_tx, out_shape_rx, out_shape_k,
                              device=contrib_real.device,
                              dtype=contrib_real.dtype)
        rp_imag = torch.zeros_like(rp_real)
        ext.scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag)
        ctx.save_for_backward(flat_idx)
        ctx.rp_numel = rp_real.numel()
        return rp_real, rp_imag

    @staticmethod
    def backward(ctx, grad_rp_real, grad_rp_imag):
        (flat_idx,) = ctx.saved_tensors
        # Gather: grad_contrib[i] = grad_rp.view(-1)[flat_idx[i]]
        grad_real_flat = grad_rp_real.reshape(-1)
        grad_imag_flat = grad_rp_imag.reshape(-1)
        grad_contrib_real = grad_real_flat[flat_idx]
        grad_contrib_imag = grad_imag_flat[flat_idx]
        return grad_contrib_real, grad_contrib_imag, None, None, None, None


def splat_scatter(contrib_real, contrib_imag, flat_idx, out_shape):
    """Autograd-friendly entry point. Returns (rp_real, rp_imag)."""
    if ext is None:
        raise RuntimeError("mm3DPS_cuda extension is not built.")
    return ScatterSplatFn.apply(
        contrib_real, contrib_imag, flat_idx,
        out_shape[0], out_shape[1], out_shape[2],
    )


class Step5FusedFn(torch.autograd.Function):
    """Fused Step-5 autograd.Function.

    Forward: takes (w_full, phi_carrier, n_peak, psf_real, psf_imag, K,
    w_threshold), returns (rp_real, rp_imag). The kernel fuses:
      - carrier phasor computation (sin/cos of phi)
      - PSF table lookup with linear interpolation
      - bin index computation (floor + offset mod K)
      - contrib = carrier × psf complex multiply
      - atomic scatter into rp_real, rp_imag
    All intermediates stay in registers. No (SPREAD, P) contrib
    materialization.

    Backward: only computes grad_w_full. phi_carrier, n_peak, and the
    PSF tables are treated as non-differentiable inputs (which matches
    the production path where detach_phase=True and n_peak is
    explicitly detached).
    """

    @staticmethod
    def forward(ctx, w_full, phi_carrier, n_peak, psf_real, psf_imag,
                K, w_threshold):
        w_full      = w_full.contiguous()
        phi_carrier = phi_carrier.contiguous()
        n_peak      = n_peak.contiguous()
        psf_real    = psf_real.contiguous()
        psf_imag    = psf_imag.contiguous()
        rp_real, rp_imag = ext.step5_fused_forward(
            w_full, phi_carrier, n_peak, psf_real, psf_imag,
            K, float(w_threshold),
        )
        ctx.save_for_backward(w_full, phi_carrier, n_peak, psf_real, psf_imag)
        ctx.w_threshold = float(w_threshold)
        return rp_real, rp_imag

    @staticmethod
    def backward(ctx, grad_rp_real, grad_rp_imag):
        w_full, phi_carrier, n_peak, psf_real, psf_imag = ctx.saved_tensors
        grad_w = ext.step5_fused_backward(
            grad_rp_real.contiguous(), grad_rp_imag.contiguous(),
            w_full, phi_carrier, n_peak, psf_real, psf_imag,
            ctx.w_threshold,
        )
        # w_full, phi_carrier, n_peak, psf_real, psf_imag, K, w_threshold
        return grad_w, None, None, None, None, None, None


def step5_fused(w_full, phi_carrier, n_peak, psf_real, psf_imag,
                K, w_threshold=1e-20):
    """Autograd-friendly entry point for the fused Step-5 kernel."""
    if ext is None:
        raise RuntimeError("mm3DPS_cuda extension is not built.")
    return Step5FusedFn.apply(
        w_full, phi_carrier, n_peak, psf_real, psf_imag, K, w_threshold,
    )


class BSDFStep4ForwardFn(torch.autograd.Function):
    """fused Step-4 BSDF forward wrapped as an autograd.Function.

    Forward: calls the CUDA kernel. Backward: falls back to the PyTorch
    reference implementation. A separate kernel replaces the backward with an
    analytical CUDA kernel.

    This is used only when `use_cuda_kernels=True` is passed through
    render_factorized; the pure-PyTorch path remains fully supported.
    """

    @staticmethod
    def forward(ctx,
                wi, wi_r, wo, n_eff, s_in,
                cos_i, cos_o, lambda_i, lambda_o,
                alpha_sq, kappa_SPM, norm_SPM, eps_factor,
                eps_real_m, eps_imag_m, thickness_m,
                E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
                tau_eff):
        assert ext is not None, "mm3DPS_cuda extension not loaded"
        # Ensure contiguous float32 — cheap if already.
        args = [t.contiguous() for t in [
            wi, wi_r, wo, n_eff, s_in,
            cos_i, cos_o, lambda_i, lambda_o,
            alpha_sq, kappa_SPM, norm_SPM, eps_factor,
            eps_real_m, eps_imag_m, thickness_m,
            E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
            tau_eff,
        ]]
        f_cos = ext.bsdf_step4_forward(*args)
        # Save for backward. The PyTorch reference is used for backward
        # (so we need the inputs to rerun Step 4 in autograd mode).
        ctx.save_for_backward(*args)
        return f_cos

    @staticmethod
    def backward(ctx, grad_f_cos):
        # analytical CUDA backward via bsdf_step4_backward.
        saved = ctx.saved_tensors
        grad_f_cos = grad_f_cos.contiguous()
        grads = ext.bsdf_step4_backward(
            grad_f_cos,
            saved[0],  saved[1],  saved[2],  saved[3],  saved[4],
            saved[5],  saved[6],  saved[7],  saved[8],
            saved[9],  saved[10], saved[11], saved[12],
            saved[13], saved[14], saved[15],
            saved[16], saved[17], saved[18], saved[19],
            saved[20],
        )
        return tuple(grads)


def bsdf_step4_forward(
    wi, wi_r, wo, n_eff, s_in,
    cos_i, cos_o, lambda_i, lambda_o,
    alpha_sq, kappa_SPM, norm_SPM, eps_factor,
    eps_real_m, eps_imag_m, thickness_m,
    E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
    tau_eff,
):
    """Autograd-friendly entry point for the CUDA Step-4 BSDF kernel."""
    if ext is None:
        raise RuntimeError("mm3DPS_cuda extension is not built.")
    return BSDFStep4ForwardFn.apply(
        wi, wi_r, wo, n_eff, s_in,
        cos_i, cos_o, lambda_i, lambda_o,
        alpha_sq, kappa_SPM, norm_SPM, eps_factor,
        eps_real_m, eps_imag_m, thickness_m,
        E_s_out_re, E_s_out_im, E_p_out_re, E_p_out_im,
        tau_eff,
    )
