"""Numerical equivalence tests for the scatter_splat CUDA kernel."""
import pytest
import torch

from mm3DPS import cuda as v5cuda


pytestmark = pytest.mark.skipif(
    not v5cuda.is_available(),
    reason=f"mm3DPS_cuda extension not built: {v5cuda.load_error()}",
)


@pytest.mark.parametrize("n_items,out_size", [
    (100, 64),
    (10_000, 4096),
    (1_000_000, 49_152),  # realistic: 90K pts × 12 × 16 / 5 spread
])
def test_scatter_splat_matches_scatter_add(n_items, out_size):
    torch.manual_seed(0)
    dev = "cuda"
    contrib_real = torch.randn(n_items, device=dev)
    contrib_imag = torch.randn(n_items, device=dev)
    flat_idx = torch.randint(0, out_size, (n_items,), device=dev, dtype=torch.int64)

    rp_real_ref = torch.zeros(out_size, device=dev)
    rp_imag_ref = torch.zeros(out_size, device=dev)
    rp_real_ref.scatter_add_(0, flat_idx, contrib_real)
    rp_imag_ref.scatter_add_(0, flat_idx, contrib_imag)

    rp_real = torch.zeros(out_size, device=dev)
    rp_imag = torch.zeros(out_size, device=dev)
    v5cuda.scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag)
    torch.cuda.synchronize()

    # Atomic ordering differs between our kernel and PyTorch's, so identical
    # floats are not guaranteed. Loose tolerance covers accumulation order.
    assert torch.allclose(rp_real, rp_real_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(rp_imag, rp_imag_ref, rtol=1e-4, atol=1e-4)


def test_ext_loads():
    assert v5cuda.is_available()
    assert hasattr(v5cuda.ext, "scatter_splat")
    assert hasattr(v5cuda.ext, "bsdf_forward_stub")
    assert hasattr(v5cuda.ext, "bsdf_backward_stub")


def test_bsdf_forward_stub_zero_fills():
    x = torch.ones(1024, device="cuda")
    v5cuda.ext.bsdf_forward_stub(x)
    torch.cuda.synchronize()
    assert (x == 0).all()
