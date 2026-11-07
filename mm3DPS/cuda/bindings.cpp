// pybind11 bindings for the mm3DPS CUDA extension.
//
// Exposes thin wrappers that: (1) check tensors are CUDA + contiguous +
// the right dtype, (2) call into the .cu launcher, (3) return outputs.
//
// Phase A only exposes a build-verification entry point and a working
// scatter_splat. The BSDF forward/backward are Phase B / C stubs.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace mm25v5 {

void launch_bsdf_forward_stub(
    float* f_cos, int64_t n_elem, cudaStream_t stream);

void launch_bsdf_backward_stub(
    float* grad_raw_mat, int64_t n_elem, cudaStream_t stream);

void launch_scatter_splat(
    const float* contrib_real, const float* contrib_imag,
    const int64_t* flat_idx,
    float* rp_real, float* rp_imag,
    int64_t n_items, int64_t out_size, cudaStream_t stream);

void launch_step5_fused_forward(
    const float* w_full, const float* phi_carrier, const float* n_peak,
    const float* psf_real, const float* psf_imag,
    float* rp_real, float* rp_imag,
    int M, int n_tx, int n_rx, int K,
    int spread, int n_grid, float w_threshold,
    cudaStream_t stream);

void launch_step5_fused_backward(
    const float* grad_rp_real, const float* grad_rp_imag,
    const float* w_full, const float* phi_carrier, const float* n_peak,
    const float* psf_real, const float* psf_imag,
    float* grad_w,
    int M, int n_tx, int n_rx, int K,
    int spread, int n_grid, float w_threshold,
    cudaStream_t stream);

void launch_bsdf_step4_intermediates(
    const float* wi, const float* wi_r, const float* wo,
    const float* n_eff, const float* s_in,
    const float* cos_i, const float* cos_o,
    const float* lambda_i, const float* lambda_o,
    const float* alpha_sq, const float* kappa_SPM,
    const float* norm_SPM, const float* eps_factor,
    const float* eps_real_m, const float* eps_imag_m, const float* thickness_m,
    const float* E_s_out_re, const float* E_s_out_im,
    const float* E_p_out_re, const float* E_p_out_im,
    float* f_KA, float* f_SPM, float* R_jones_h, float* R_jones_macro,
    float* cos_h, float* cos_dev,
    int M, int n_tx, int n_rx, cudaStream_t stream);

void launch_bsdf_microfacet_basis(
    const float* wi, const float* wo,
    float* sh_xyz, float* pin_xyz, float* pout_xyz,
    int M, int n_tx, int n_rx, cudaStream_t stream);

void launch_itu_slab_fresnel_debug(
    const float* eps_real, const float* eps_imag,
    const float* cos_i, const float* thickness,
    float* R_TE_re, float* R_TE_im,
    float* R_TM_re, float* R_TM_im,
    int n, cudaStream_t stream);

void launch_zero(float* ptr, int64_t n, cudaStream_t stream);

void launch_bsdf_step4_backward(
    const float* grad_f_cos,
    const float* wi, const float* wi_r, const float* wo,
    const float* n_eff, const float* s_in,
    const float* cos_i, const float* cos_o,
    const float* lambda_i, const float* lambda_o,
    const float* alpha_sq, const float* kappa_SPM,
    const float* norm_SPM, const float* eps_factor,
    const float* eps_real_m, const float* eps_imag_m, const float* thickness_m,
    const float* E_s_out_re, const float* E_s_out_im,
    const float* E_p_out_re, const float* E_p_out_im,
    const float* tau_eff,
    float* grad_wi, float* grad_wi_r, float* grad_wo,
    float* grad_n_eff, float* grad_s_in,
    float* grad_cos_i, float* grad_cos_o,
    float* grad_lambda_i, float* grad_lambda_o,
    float* grad_alpha_sq, float* grad_kappa_SPM,
    float* grad_norm_SPM, float* grad_eps_factor,
    float* grad_eps_real_m, float* grad_eps_imag_m, float* grad_thickness_m,
    float* grad_E_s_out_re, float* grad_E_s_out_im,
    float* grad_E_p_out_re, float* grad_E_p_out_im,
    float* grad_tau_eff,
    int M, int n_tx, int n_rx,
    cudaStream_t stream);

void launch_bsdf_step4_forward(
    const float* wi, const float* wi_r, const float* wo,
    const float* n_eff, const float* s_in,
    const float* cos_i, const float* cos_o,
    const float* lambda_i, const float* lambda_o,
    const float* alpha_sq, const float* kappa_SPM,
    const float* norm_SPM, const float* eps_factor,
    const float* eps_real_m, const float* eps_imag_m, const float* thickness_m,
    const float* E_s_out_re, const float* E_s_out_im,
    const float* E_p_out_re, const float* E_p_out_im,
    const float* tau_eff,
    float* f_cos,
    int M, int n_tx, int n_rx,
    cudaStream_t stream);

} // namespace mm25v5

#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(),         #x " must be a CUDA tensor")
#define CHECK_CONTIG(x)     TORCH_CHECK(x.is_contiguous(),   #x " must be contiguous")
#define CHECK_FLOAT32(x)    TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT64(x)      TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be int64")

// Phase A: smoke test — zero-fill a float tensor on the current stream.
// Used by tests to confirm the extension loads and launches a kernel.
torch::Tensor bsdf_forward_stub(torch::Tensor f_cos) {
    CHECK_CUDA(f_cos);
    CHECK_CONTIG(f_cos);
    CHECK_FLOAT32(f_cos);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_bsdf_forward_stub(
        f_cos.data_ptr<float>(), f_cos.numel(), stream.stream());
    return f_cos;
}

torch::Tensor bsdf_backward_stub(torch::Tensor grad_raw_mat) {
    CHECK_CUDA(grad_raw_mat);
    CHECK_CONTIG(grad_raw_mat);
    CHECK_FLOAT32(grad_raw_mat);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_bsdf_backward_stub(
        grad_raw_mat.data_ptr<float>(), grad_raw_mat.numel(), stream.stream());
    return grad_raw_mat;
}

// Phase A: working atomic scatter-add. Takes flattened contributions and
// scatters them into rp_real / rp_imag. Equivalent to
//   rp_real.view(-1).scatter_add_(0, flat_idx, contrib_real)
//   rp_imag.view(-1).scatter_add_(0, flat_idx, contrib_imag)
// but issues only one kernel launch for both and reads flat_idx once.
void scatter_splat(
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_idx,
    torch::Tensor rp_real, torch::Tensor rp_imag)
{
    CHECK_CUDA(contrib_real); CHECK_CONTIG(contrib_real); CHECK_FLOAT32(contrib_real);
    CHECK_CUDA(contrib_imag); CHECK_CONTIG(contrib_imag); CHECK_FLOAT32(contrib_imag);
    CHECK_CUDA(flat_idx);     CHECK_CONTIG(flat_idx);     CHECK_INT64(flat_idx);
    CHECK_CUDA(rp_real);      CHECK_CONTIG(rp_real);      CHECK_FLOAT32(rp_real);
    CHECK_CUDA(rp_imag);      CHECK_CONTIG(rp_imag);      CHECK_FLOAT32(rp_imag);

    TORCH_CHECK(contrib_real.numel() == contrib_imag.numel(),
                "contrib_real and contrib_imag must have same numel");
    TORCH_CHECK(flat_idx.numel() == contrib_real.numel(),
                "flat_idx must match contrib numel");
    TORCH_CHECK(rp_real.numel() == rp_imag.numel(),
                "rp_real and rp_imag must have same numel");

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_scatter_splat(
        contrib_real.data_ptr<float>(),
        contrib_imag.data_ptr<float>(),
        flat_idx.data_ptr<int64_t>(),
        rp_real.data_ptr<float>(),
        rp_imag.data_ptr<float>(),
        contrib_real.numel(),
        rp_real.numel(),
        stream.stream());
}

// Phase B: fused BSDF Step-4 forward.
// Takes the Python-precomputed per-(m,t) and per-(m,r) tensors plus per-m
// material scalars and returns f_cos (M, n_tx, n_rx).
torch::Tensor bsdf_step4_forward(
    torch::Tensor wi, torch::Tensor wi_r, torch::Tensor wo,
    torch::Tensor n_eff, torch::Tensor s_in,
    torch::Tensor cos_i, torch::Tensor cos_o,
    torch::Tensor lambda_i, torch::Tensor lambda_o,
    torch::Tensor alpha_sq, torch::Tensor kappa_SPM,
    torch::Tensor norm_SPM, torch::Tensor eps_factor,
    torch::Tensor eps_real_m, torch::Tensor eps_imag_m, torch::Tensor thickness_m,
    torch::Tensor E_s_out_re, torch::Tensor E_s_out_im,
    torch::Tensor E_p_out_re, torch::Tensor E_p_out_im,
    torch::Tensor tau_eff)
{
    // Dtype + CUDA + contiguous checks on the key tensors. We assume the
    // caller has already materialized them as float32 contiguous CUDA.
    CHECK_CUDA(wi); CHECK_CONTIG(wi); CHECK_FLOAT32(wi);
    CHECK_CUDA(wi_r); CHECK_CONTIG(wi_r); CHECK_FLOAT32(wi_r);
    CHECK_CUDA(wo); CHECK_CONTIG(wo); CHECK_FLOAT32(wo);
    CHECK_CUDA(n_eff); CHECK_CONTIG(n_eff); CHECK_FLOAT32(n_eff);
    CHECK_CUDA(s_in); CHECK_CONTIG(s_in); CHECK_FLOAT32(s_in);
    CHECK_CUDA(cos_i); CHECK_CONTIG(cos_i); CHECK_FLOAT32(cos_i);
    CHECK_CUDA(cos_o); CHECK_CONTIG(cos_o); CHECK_FLOAT32(cos_o);
    CHECK_CUDA(lambda_i); CHECK_CONTIG(lambda_i); CHECK_FLOAT32(lambda_i);
    CHECK_CUDA(lambda_o); CHECK_CONTIG(lambda_o); CHECK_FLOAT32(lambda_o);
    CHECK_CUDA(alpha_sq); CHECK_CONTIG(alpha_sq); CHECK_FLOAT32(alpha_sq);
    CHECK_CUDA(kappa_SPM); CHECK_CONTIG(kappa_SPM); CHECK_FLOAT32(kappa_SPM);
    CHECK_CUDA(norm_SPM); CHECK_CONTIG(norm_SPM); CHECK_FLOAT32(norm_SPM);
    CHECK_CUDA(eps_factor); CHECK_CONTIG(eps_factor); CHECK_FLOAT32(eps_factor);
    CHECK_CUDA(eps_real_m); CHECK_CONTIG(eps_real_m); CHECK_FLOAT32(eps_real_m);
    CHECK_CUDA(eps_imag_m); CHECK_CONTIG(eps_imag_m); CHECK_FLOAT32(eps_imag_m);
    CHECK_CUDA(thickness_m); CHECK_CONTIG(thickness_m); CHECK_FLOAT32(thickness_m);
    CHECK_CUDA(E_s_out_re); CHECK_CONTIG(E_s_out_re); CHECK_FLOAT32(E_s_out_re);
    CHECK_CUDA(E_s_out_im); CHECK_CONTIG(E_s_out_im); CHECK_FLOAT32(E_s_out_im);
    CHECK_CUDA(E_p_out_re); CHECK_CONTIG(E_p_out_re); CHECK_FLOAT32(E_p_out_re);
    CHECK_CUDA(E_p_out_im); CHECK_CONTIG(E_p_out_im); CHECK_FLOAT32(E_p_out_im);
    CHECK_CUDA(tau_eff); CHECK_CONTIG(tau_eff); CHECK_FLOAT32(tau_eff);

    const int M    = (int)wi.size(0);
    const int n_tx = (int)wi.size(1);
    const int n_rx = (int)wo.size(1);
    TORCH_CHECK(wi.size(2) == 3 && wo.size(2) == 3,
                "wi / wo last dim must be 3");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(wi.device());
    auto f_cos = torch::empty({M, n_tx, n_rx}, options);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_bsdf_step4_forward(
        wi.data_ptr<float>(), wi_r.data_ptr<float>(), wo.data_ptr<float>(),
        n_eff.data_ptr<float>(), s_in.data_ptr<float>(),
        cos_i.data_ptr<float>(), cos_o.data_ptr<float>(),
        lambda_i.data_ptr<float>(), lambda_o.data_ptr<float>(),
        alpha_sq.data_ptr<float>(), kappa_SPM.data_ptr<float>(),
        norm_SPM.data_ptr<float>(), eps_factor.data_ptr<float>(),
        eps_real_m.data_ptr<float>(), eps_imag_m.data_ptr<float>(),
        thickness_m.data_ptr<float>(),
        E_s_out_re.data_ptr<float>(), E_s_out_im.data_ptr<float>(),
        E_p_out_re.data_ptr<float>(), E_p_out_im.data_ptr<float>(),
        tau_eff.data_ptr<float>(),
        f_cos.data_ptr<float>(),
        M, n_tx, n_rx,
        stream.stream());
    return f_cos;
}

// Phase C: fused BSDF Step-4 backward. Takes grad_f_cos and all 21
// forward inputs, returns a list of 21 grad tensors in the same order
// as the forward inputs.
std::vector<torch::Tensor> bsdf_step4_backward(
    torch::Tensor grad_f_cos,
    torch::Tensor wi, torch::Tensor wi_r, torch::Tensor wo,
    torch::Tensor n_eff, torch::Tensor s_in,
    torch::Tensor cos_i, torch::Tensor cos_o,
    torch::Tensor lambda_i, torch::Tensor lambda_o,
    torch::Tensor alpha_sq, torch::Tensor kappa_SPM,
    torch::Tensor norm_SPM, torch::Tensor eps_factor,
    torch::Tensor eps_real_m, torch::Tensor eps_imag_m, torch::Tensor thickness_m,
    torch::Tensor E_s_out_re, torch::Tensor E_s_out_im,
    torch::Tensor E_p_out_re, torch::Tensor E_p_out_im,
    torch::Tensor tau_eff)
{
    CHECK_CUDA(grad_f_cos); CHECK_CONTIG(grad_f_cos); CHECK_FLOAT32(grad_f_cos);
    CHECK_CUDA(wi); CHECK_CONTIG(wi); CHECK_FLOAT32(wi);
    CHECK_CUDA(wi_r); CHECK_CONTIG(wi_r); CHECK_FLOAT32(wi_r);
    CHECK_CUDA(wo); CHECK_CONTIG(wo); CHECK_FLOAT32(wo);
    CHECK_CUDA(n_eff); CHECK_CONTIG(n_eff); CHECK_FLOAT32(n_eff);
    CHECK_CUDA(s_in); CHECK_CONTIG(s_in); CHECK_FLOAT32(s_in);
    CHECK_CUDA(cos_i); CHECK_CONTIG(cos_i); CHECK_FLOAT32(cos_i);
    CHECK_CUDA(cos_o); CHECK_CONTIG(cos_o); CHECK_FLOAT32(cos_o);
    CHECK_CUDA(lambda_i); CHECK_CONTIG(lambda_i); CHECK_FLOAT32(lambda_i);
    CHECK_CUDA(lambda_o); CHECK_CONTIG(lambda_o); CHECK_FLOAT32(lambda_o);
    CHECK_CUDA(alpha_sq); CHECK_CONTIG(alpha_sq); CHECK_FLOAT32(alpha_sq);
    CHECK_CUDA(kappa_SPM); CHECK_CONTIG(kappa_SPM); CHECK_FLOAT32(kappa_SPM);
    CHECK_CUDA(norm_SPM); CHECK_CONTIG(norm_SPM); CHECK_FLOAT32(norm_SPM);
    CHECK_CUDA(eps_factor); CHECK_CONTIG(eps_factor); CHECK_FLOAT32(eps_factor);
    CHECK_CUDA(eps_real_m); CHECK_CONTIG(eps_real_m); CHECK_FLOAT32(eps_real_m);
    CHECK_CUDA(eps_imag_m); CHECK_CONTIG(eps_imag_m); CHECK_FLOAT32(eps_imag_m);
    CHECK_CUDA(thickness_m); CHECK_CONTIG(thickness_m); CHECK_FLOAT32(thickness_m);
    CHECK_CUDA(E_s_out_re); CHECK_CONTIG(E_s_out_re); CHECK_FLOAT32(E_s_out_re);
    CHECK_CUDA(E_s_out_im); CHECK_CONTIG(E_s_out_im); CHECK_FLOAT32(E_s_out_im);
    CHECK_CUDA(E_p_out_re); CHECK_CONTIG(E_p_out_re); CHECK_FLOAT32(E_p_out_re);
    CHECK_CUDA(E_p_out_im); CHECK_CONTIG(E_p_out_im); CHECK_FLOAT32(E_p_out_im);
    CHECK_CUDA(tau_eff); CHECK_CONTIG(tau_eff); CHECK_FLOAT32(tau_eff);

    const int M    = (int)wi.size(0);
    const int n_tx = (int)wi.size(1);
    const int n_rx = (int)wo.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(wi.device());

    auto make_grad = [&](const std::vector<int64_t>& shape) {
        return torch::zeros(shape, options);
    };
    auto grad_wi          = make_grad({M, n_tx, 3});
    auto grad_wi_r        = make_grad({M, n_tx, 3});
    auto grad_wo          = make_grad({M, n_rx, 3});
    auto grad_n_eff       = make_grad({M, n_tx, 3});
    auto grad_s_in        = make_grad({M, n_tx, 3});
    auto grad_cos_i       = make_grad({M, n_tx});
    auto grad_cos_o       = make_grad({M, n_rx});
    auto grad_lambda_i    = make_grad({M, n_tx});
    auto grad_lambda_o    = make_grad({M, n_rx});
    auto grad_alpha_sq    = make_grad({M});
    auto grad_kappa_SPM   = make_grad({M});
    auto grad_norm_SPM    = make_grad({M});
    auto grad_eps_factor  = make_grad({M});
    auto grad_eps_real_m  = make_grad({M});
    auto grad_eps_imag_m  = make_grad({M});
    auto grad_thickness_m = make_grad({M});
    auto grad_E_s_out_re  = make_grad({M, n_tx});
    auto grad_E_s_out_im  = make_grad({M, n_tx});
    auto grad_E_p_out_re  = make_grad({M, n_tx});
    auto grad_E_p_out_im  = make_grad({M, n_tx});
    auto grad_tau_eff     = make_grad({M, n_tx});

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_bsdf_step4_backward(
        grad_f_cos.data_ptr<float>(),
        wi.data_ptr<float>(), wi_r.data_ptr<float>(), wo.data_ptr<float>(),
        n_eff.data_ptr<float>(), s_in.data_ptr<float>(),
        cos_i.data_ptr<float>(), cos_o.data_ptr<float>(),
        lambda_i.data_ptr<float>(), lambda_o.data_ptr<float>(),
        alpha_sq.data_ptr<float>(), kappa_SPM.data_ptr<float>(),
        norm_SPM.data_ptr<float>(), eps_factor.data_ptr<float>(),
        eps_real_m.data_ptr<float>(), eps_imag_m.data_ptr<float>(),
        thickness_m.data_ptr<float>(),
        E_s_out_re.data_ptr<float>(), E_s_out_im.data_ptr<float>(),
        E_p_out_re.data_ptr<float>(), E_p_out_im.data_ptr<float>(),
        tau_eff.data_ptr<float>(),
        grad_wi.data_ptr<float>(), grad_wi_r.data_ptr<float>(), grad_wo.data_ptr<float>(),
        grad_n_eff.data_ptr<float>(), grad_s_in.data_ptr<float>(),
        grad_cos_i.data_ptr<float>(), grad_cos_o.data_ptr<float>(),
        grad_lambda_i.data_ptr<float>(), grad_lambda_o.data_ptr<float>(),
        grad_alpha_sq.data_ptr<float>(), grad_kappa_SPM.data_ptr<float>(),
        grad_norm_SPM.data_ptr<float>(), grad_eps_factor.data_ptr<float>(),
        grad_eps_real_m.data_ptr<float>(), grad_eps_imag_m.data_ptr<float>(),
        grad_thickness_m.data_ptr<float>(),
        grad_E_s_out_re.data_ptr<float>(), grad_E_s_out_im.data_ptr<float>(),
        grad_E_p_out_re.data_ptr<float>(), grad_E_p_out_im.data_ptr<float>(),
        grad_tau_eff.data_ptr<float>(),
        M, n_tx, n_rx, stream.stream());

    return {
        grad_wi, grad_wi_r, grad_wo, grad_n_eff, grad_s_in,
        grad_cos_i, grad_cos_o, grad_lambda_i, grad_lambda_o,
        grad_alpha_sq, grad_kappa_SPM, grad_norm_SPM, grad_eps_factor,
        grad_eps_real_m, grad_eps_imag_m, grad_thickness_m,
        grad_E_s_out_re, grad_E_s_out_im,
        grad_E_p_out_re, grad_E_p_out_im,
        grad_tau_eff,
    };
}

// Debug entry: run the full Step-4 kernel but emit per-(m, t, r)
// intermediates (f_KA, f_SPM, R_jones_h, R_jones_macro, cos_h, cos_dev)
// so tests can localize numerical drift.
std::vector<torch::Tensor> bsdf_step4_intermediates(
    torch::Tensor wi, torch::Tensor wi_r, torch::Tensor wo,
    torch::Tensor n_eff, torch::Tensor s_in,
    torch::Tensor cos_i, torch::Tensor cos_o,
    torch::Tensor lambda_i, torch::Tensor lambda_o,
    torch::Tensor alpha_sq, torch::Tensor kappa_SPM,
    torch::Tensor norm_SPM, torch::Tensor eps_factor,
    torch::Tensor eps_real_m, torch::Tensor eps_imag_m, torch::Tensor thickness_m,
    torch::Tensor E_s_out_re, torch::Tensor E_s_out_im,
    torch::Tensor E_p_out_re, torch::Tensor E_p_out_im)
{
    const int M = (int)wi.size(0);
    const int n_tx = (int)wi.size(1);
    const int n_rx = (int)wo.size(1);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(wi.device());
    auto f_KA = torch::empty({M, n_tx, n_rx}, opt);
    auto f_SPM = torch::empty({M, n_tx, n_rx}, opt);
    auto R_jones_h = torch::empty({M, n_tx, n_rx}, opt);
    auto R_jones_macro = torch::empty({M, n_tx, n_rx}, opt);
    auto cos_h = torch::empty({M, n_tx, n_rx}, opt);
    auto cos_dev = torch::empty({M, n_tx, n_rx}, opt);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_bsdf_step4_intermediates(
        wi.data_ptr<float>(), wi_r.data_ptr<float>(), wo.data_ptr<float>(),
        n_eff.data_ptr<float>(), s_in.data_ptr<float>(),
        cos_i.data_ptr<float>(), cos_o.data_ptr<float>(),
        lambda_i.data_ptr<float>(), lambda_o.data_ptr<float>(),
        alpha_sq.data_ptr<float>(), kappa_SPM.data_ptr<float>(),
        norm_SPM.data_ptr<float>(), eps_factor.data_ptr<float>(),
        eps_real_m.data_ptr<float>(), eps_imag_m.data_ptr<float>(),
        thickness_m.data_ptr<float>(),
        E_s_out_re.data_ptr<float>(), E_s_out_im.data_ptr<float>(),
        E_p_out_re.data_ptr<float>(), E_p_out_im.data_ptr<float>(),
        f_KA.data_ptr<float>(), f_SPM.data_ptr<float>(),
        R_jones_h.data_ptr<float>(), R_jones_macro.data_ptr<float>(),
        cos_h.data_ptr<float>(), cos_dev.data_ptr<float>(),
        M, n_tx, n_rx, stream.stream());
    return {f_KA, f_SPM, R_jones_h, R_jones_macro, cos_h, cos_dev};
}

std::vector<torch::Tensor> bsdf_microfacet_basis(torch::Tensor wi, torch::Tensor wo) {
    int M = (int)wi.size(0);
    int n_tx = (int)wi.size(1);
    int n_rx = (int)wo.size(1);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(wi.device());
    auto sh_xyz = torch::empty({M, n_tx, n_rx, 3}, opt);
    auto pin_xyz = torch::empty({M, n_tx, n_rx, 3}, opt);
    auto pout_xyz = torch::empty({M, n_tx, n_rx, 3}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_bsdf_microfacet_basis(
        wi.data_ptr<float>(), wo.data_ptr<float>(),
        sh_xyz.data_ptr<float>(), pin_xyz.data_ptr<float>(), pout_xyz.data_ptr<float>(),
        M, n_tx, n_rx, stream.stream());
    return {sh_xyz, pin_xyz, pout_xyz};
}

// Debug entry: run CUDA itu_slab_fresnel on a batch of inputs and return
// R_TE_re, R_TE_im, R_TM_re, R_TM_im as 4 float32 tensors.
std::vector<torch::Tensor> itu_slab_fresnel_debug(
    torch::Tensor eps_real, torch::Tensor eps_imag,
    torch::Tensor cos_i, torch::Tensor thickness)
{
    CHECK_CUDA(eps_real); CHECK_CONTIG(eps_real); CHECK_FLOAT32(eps_real);
    CHECK_CUDA(eps_imag); CHECK_CONTIG(eps_imag); CHECK_FLOAT32(eps_imag);
    CHECK_CUDA(cos_i); CHECK_CONTIG(cos_i); CHECK_FLOAT32(cos_i);
    CHECK_CUDA(thickness); CHECK_CONTIG(thickness); CHECK_FLOAT32(thickness);
    int n = (int)eps_real.numel();
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(eps_real.device());
    auto R_TE_re = torch::empty({n}, opt);
    auto R_TE_im = torch::empty({n}, opt);
    auto R_TM_re = torch::empty({n}, opt);
    auto R_TM_im = torch::empty({n}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_itu_slab_fresnel_debug(
        eps_real.data_ptr<float>(), eps_imag.data_ptr<float>(),
        cos_i.data_ptr<float>(), thickness.data_ptr<float>(),
        R_TE_re.data_ptr<float>(), R_TE_im.data_ptr<float>(),
        R_TM_re.data_ptr<float>(), R_TM_im.data_ptr<float>(),
        n, stream.stream());
    return {R_TE_re, R_TE_im, R_TM_re, R_TM_im};
}

// Phase D: fused Step-5 forward. Returns (rp_real, rp_imag) from
// (w_full, phi_carrier, n_peak, psf_real, psf_imag). Shapes:
//   w_full, phi_carrier, n_peak: (M, n_tx, n_rx)
//   psf_real, psf_imag:          (spread, n_grid)
//   rp_real, rp_imag (out):      (n_tx, n_rx, K)
std::vector<torch::Tensor> step5_fused_forward(
    torch::Tensor w_full, torch::Tensor phi_carrier, torch::Tensor n_peak,
    torch::Tensor psf_real, torch::Tensor psf_imag,
    int64_t K, double w_threshold)
{
    CHECK_CUDA(w_full); CHECK_CONTIG(w_full); CHECK_FLOAT32(w_full);
    CHECK_CUDA(phi_carrier); CHECK_CONTIG(phi_carrier); CHECK_FLOAT32(phi_carrier);
    CHECK_CUDA(n_peak); CHECK_CONTIG(n_peak); CHECK_FLOAT32(n_peak);
    CHECK_CUDA(psf_real); CHECK_CONTIG(psf_real); CHECK_FLOAT32(psf_real);
    CHECK_CUDA(psf_imag); CHECK_CONTIG(psf_imag); CHECK_FLOAT32(psf_imag);

    const int M    = (int)w_full.size(0);
    const int n_tx = (int)w_full.size(1);
    const int n_rx = (int)w_full.size(2);
    const int spread = (int)psf_real.size(0);
    const int n_grid = (int)psf_real.size(1);

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32).device(w_full.device());
    auto rp_real = torch::zeros({n_tx, n_rx, (int64_t)K}, opts);
    auto rp_imag = torch::zeros({n_tx, n_rx, (int64_t)K}, opts);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_step5_fused_forward(
        w_full.data_ptr<float>(), phi_carrier.data_ptr<float>(),
        n_peak.data_ptr<float>(),
        psf_real.data_ptr<float>(), psf_imag.data_ptr<float>(),
        rp_real.data_ptr<float>(), rp_imag.data_ptr<float>(),
        M, n_tx, n_rx, (int)K, spread, n_grid, (float)w_threshold,
        stream.stream());
    return {rp_real, rp_imag};
}

torch::Tensor step5_fused_backward(
    torch::Tensor grad_rp_real, torch::Tensor grad_rp_imag,
    torch::Tensor w_full, torch::Tensor phi_carrier, torch::Tensor n_peak,
    torch::Tensor psf_real, torch::Tensor psf_imag,
    double w_threshold)
{
    CHECK_CUDA(grad_rp_real); CHECK_CONTIG(grad_rp_real); CHECK_FLOAT32(grad_rp_real);
    CHECK_CUDA(grad_rp_imag); CHECK_CONTIG(grad_rp_imag); CHECK_FLOAT32(grad_rp_imag);
    CHECK_CUDA(w_full); CHECK_CONTIG(w_full); CHECK_FLOAT32(w_full);
    CHECK_CUDA(phi_carrier); CHECK_CONTIG(phi_carrier); CHECK_FLOAT32(phi_carrier);
    CHECK_CUDA(n_peak); CHECK_CONTIG(n_peak); CHECK_FLOAT32(n_peak);

    const int M    = (int)w_full.size(0);
    const int n_tx = (int)w_full.size(1);
    const int n_rx = (int)w_full.size(2);
    const int K    = (int)grad_rp_real.size(2);
    const int spread = (int)psf_real.size(0);
    const int n_grid = (int)psf_real.size(1);

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32).device(w_full.device());
    auto grad_w = torch::empty({M, n_tx, n_rx}, opts);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_step5_fused_backward(
        grad_rp_real.data_ptr<float>(), grad_rp_imag.data_ptr<float>(),
        w_full.data_ptr<float>(), phi_carrier.data_ptr<float>(),
        n_peak.data_ptr<float>(),
        psf_real.data_ptr<float>(), psf_imag.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        M, n_tx, n_rx, K, spread, n_grid, (float)w_threshold,
        stream.stream());
    return grad_w;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "mm3DPS CUDA kernels";
    m.def("bsdf_forward_stub", &bsdf_forward_stub,
          "Phase A stub for BSDF forward (zero-fills the output).");
    m.def("bsdf_backward_stub", &bsdf_backward_stub,
          "Phase A stub for BSDF backward (zero-fills the output).");
    m.def("scatter_splat", &scatter_splat,
          "Atomic scatter splat into (rp_real, rp_imag). Equivalent to "
          "two scatter_add_ calls, but with a single kernel launch.");
    m.def("bsdf_step4_forward", &bsdf_step4_forward,
          "Phase B fused BSDF Step-4 forward kernel. Replaces lines "
          "228-420 of rasterizer_factorized.py.");
    m.def("bsdf_step4_backward", &bsdf_step4_backward,
          "Phase C fused BSDF Step-4 backward kernel. Takes grad_f_cos "
          "and the 21 forward inputs, returns 21 grad tensors.");
    m.def("step5_fused_forward", &step5_fused_forward,
          "Phase D fused Step-5 range-profile splat forward. Returns "
          "(rp_real, rp_imag) given (w_full, phi_carrier, n_peak, "
          "psf_real, psf_imag).");
    m.def("step5_fused_backward", &step5_fused_backward,
          "Phase D fused Step-5 backward. Returns grad_w given "
          "grad_rp_real/imag and the same forward inputs.");
    m.def("bsdf_microfacet_basis", &bsdf_microfacet_basis,
          "Debug-only: compute s_h, p_h_in, p_h_out from wi, wo.");
    m.def("bsdf_step4_intermediates", &bsdf_step4_intermediates,
          "Debug-only: return per-(m, t, r) f_KA, f_SPM, R_jones_h, "
          "R_jones_macro, cos_h, cos_dev from the Step-4 kernel.");
    m.def("itu_slab_fresnel_debug", &itu_slab_fresnel_debug,
          "Test-only: evaluate the CUDA itu_slab_fresnel helper on a "
          "batch of (eps_real, eps_imag, cos_i, thickness) scalars. "
          "Returns [R_TE_re, R_TE_im, R_TM_re, R_TM_im].");
}
