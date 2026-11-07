// Device-side helpers shared by the mm3DPS CUDA kernels.
//
// Scope:
//  - Complex-number ops on float2 (the plan keeps everything in float/float2
//    registers; no std::complex, no thrust::complex).
//  - Vector-3 helpers (dot, cross, normalize).
//  - Device-side reparam for the 6 raw material parameters (matches the
//    reparameterize_torch function in mm3DPS/bsdf_torch.py).
//  - itu_slab_fresnel port (scalar version, single path).
//
// All functions are __device__ __forceinline__ to encourage the compiler
// to keep intermediates in registers across the fused kernel.

#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace mm25v5 {

// --- scalar constants -------------------------------------------------------

constexpr float C_LIGHT = 299792458.0f;
constexpr float PI      = 3.14159265358979323846f;
constexpr float TWO_PI  = 6.28318530717958647692f;
constexpr float INV_PI  = 0.31830988618379067154f;
// WAVELENGTH matches mm3DPS/bsdf_torch.py:WAVELENGTH = 3.9e-3 (77 GHz carrier).
constexpr float WAVELENGTH = 3.9e-3f;
constexpr float K_WAVE     = TWO_PI / WAVELENGTH;     // ~1611 rad/m

// Double-precision versions of the math constants, used by the double
// path in the forward kernel. Critical: the float constants above would
// silently cap the kernel's accuracy at ~7 decimals even when all
// arithmetic runs in double.
constexpr double PI_D       = 3.141592653589793238462643383279502884;
constexpr double TWO_PI_D   = 6.283185307179586476925286766559005768;
constexpr double INV_PI_D   = 0.318309886183790671537767526745028724;
constexpr double WAVELENGTH_D = 3.9e-3;
constexpr double K_WAVE_D   = TWO_PI_D / WAVELENGTH_D;

// --- float2 complex ops -----------------------------------------------------

__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ float2 cscale(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}
__device__ __forceinline__ float cabs_sq(float2 a) {
    return a.x * a.x + a.y * a.y;
}
// complex division a / b
__device__ __forceinline__ float2 cdiv(float2 a, float2 b) {
    float denom = b.x * b.x + b.y * b.y;
    float inv = 1.0f / fmaxf(denom, 1e-30f);
    return make_float2((a.x * b.x + a.y * b.y) * inv,
                       (a.y * b.x - a.x * b.y) * inv);
}
// Numerically stable complex sqrt (principal branch).
//
// The naive form `re = sqrt((|z|+x)/2), im = sign(y)*sqrt((|z|-x)/2)`
// suffers catastrophic cancellation in one of the two sqrt arguments when
// |y| << |x|: the subtraction `|z|-x` loses all the small-y information.
// We instead compute the large component directly and derive the small
// one from the identity 2*re*im = y (which follows from (re+i*im)^2 = z).
__device__ __forceinline__ float2 csqrt(float2 a) {
    if (a.x == 0.0f && a.y == 0.0f) return make_float2(0.0f, 0.0f);
    float r = sqrtf(a.x * a.x + a.y * a.y);
    float re, im;
    if (a.x >= 0.0f) {
        re = sqrtf(0.5f * (r + a.x));
        // Guard the divide: if re is zero then a = 0 (handled above).
        im = (re > 0.0f) ? (0.5f * a.y / re) : 0.0f;
    } else {
        im = sqrtf(0.5f * (r - a.x));
        if (a.y < 0.0f) im = -im;
        re = (im != 0.0f) ? (0.5f * a.y / im) : 0.0f;
    }
    return make_float2(re, im);
}
// complex exp. Uses float64 range reduction for the imaginary part so it
// remains accurate when the argument is much larger than 2π (the slab
// Fresnel phase q = (2π/λ)·d·a reaches ~9000 rad at d=1.7 m, and the
// sincosf intrinsic enabled by --use_fast_math only gives good
// precision for |x| ≲ π. Double-precision fmod costs a few ops but
// restores correctness.
__device__ __forceinline__ float2 cexp(float2 a) {
    float er = expf(a.x);
    double y_reduced = fmod((double)a.y, 6.283185307179586);
    float s, c;
    sincosf((float)y_reduced, &s, &c);
    return make_float2(er * c, er * s);
}

// --- vec3 helpers -----------------------------------------------------------

__device__ __forceinline__ float vdot(const float ax, const float ay, const float az,
                                       const float bx, const float by, const float bz) {
    return ax * bx + ay * by + az * bz;
}
__device__ __forceinline__ void vcross(const float ax, const float ay, const float az,
                                        const float bx, const float by, const float bz,
                                        float& ox, float& oy, float& oz) {
    ox = ay * bz - az * by;
    oy = az * bx - ax * bz;
    oz = ax * by - ay * bx;
}

// --- material reparam ------------------------------------------------------
//
// Matches mm3DPS/rasterizer.py:reparameterize_torch exactly:
//   raw[0] -> eps_real  = 1 + softplus(raw)
//   raw[1] -> eps_imag  = exp(clamp(raw, -7, 16))
//   raw[2] -> sigma_h   = exp(clamp(raw, -16, -7))
//   raw[3] -> l_c       = exp(clamp(raw, -10, 2))
//   raw[4] -> tau_base  = 0.05 + 0.9 * sigmoid(raw)
//   raw[5] -> thickness = exp(clamp(raw, -7, 2))

__device__ __forceinline__ float softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ void reparam6(
    const float r0, const float r1, const float r2,
    const float r3, const float r4, const float r5,
    float& eps_real, float& eps_imag,
    float& sigma_h, float& l_c,
    float& tau_base, float& thickness)
{
    eps_real  = 1.0f + softplus(r0);
    eps_imag  = expf(clampf(r1, -7.0f, 16.0f));
    sigma_h   = expf(clampf(r2, -16.0f, -7.0f));
    l_c       = expf(clampf(r3, -10.0f, 2.0f));
    tau_base  = 0.05f + 0.9f * sigmoid(r4);
    thickness = expf(clampf(r5, -7.0f, 2.0f));
}

// --- itu_slab_fresnel port -------------------------------------------------
//
// Ported from mm3DPS/bsdf_torch.py:itu_slab_fresnel. Computes the reflection
// coefficients R_TE, R_TM of a dielectric slab of thickness d with complex
// permittivity eps_r + i*eps_i, at incidence cosine cos_i. Assumes lossy
// medium (the Python path adds a small imaginary component for stability).
//
// Returns (R_TE, R_TM) as two float2 values via output pointers.

__device__ __forceinline__ void itu_slab_fresnel(
    float eps_real, float eps_imag, float cos_i, float thickness,
    float2& R_TE, float2& R_TM)
{
    // Match mm3DPS/bsdf_torch.py:itu_slab_fresnel exactly.
    // Python: eta = complex(eps_real, -eps_imag)  (negative imag convention)
    //         cos_i clamped to [1e-6, 1]
    //         sin2 = 1 - cos_i^2
    //         a = csqrt(eta - sin2)
    //         r_te = (cos_i - a) / (cos_i + a + 1e-10)
    //         r_tm = (eta*cos_i - a) / (eta*cos_i + a + 1e-10)
    //         q = (2π/λ) * d * a
    //         ej2q = exp(-2j*q)
    //         R = r*(1 - ej2q) / (1 - r^2 * ej2q + 1e-10)
    cos_i = clampf(cos_i, 1e-6f, 1.0f);
    float sin2 = 1.0f - cos_i * cos_i;

    float2 eta = make_float2(eps_real, -eps_imag);
    float2 inside = csub(eta, make_float2(sin2, 0.0f));
    float2 a = csqrt(inside);

    float2 cos_i_c = make_float2(cos_i, 0.0f);
    float2 eps = make_float2(1e-10f, 0.0f);

    float2 num_te = csub(cos_i_c, a);
    float2 den_te = cadd(cadd(cos_i_c, a), eps);
    float2 r_te = cdiv(num_te, den_te);

    float2 eta_cos_i = cscale(eta, cos_i);
    float2 num_tm = csub(eta_cos_i, a);
    float2 den_tm = cadd(cadd(eta_cos_i, a), eps);
    float2 r_tm = cdiv(num_tm, den_tm);

    // q = (2π/λ) * d * a   (complex)
    float2 q = cscale(a, K_WAVE * thickness);
    // ej2q = exp(-2j * q) = exp(2*q.y + j*(-2*q.x))
    float2 ej2q = cexp(make_float2(2.0f * q.y, -2.0f * q.x));

    float2 one = make_float2(1.0f, 0.0f);
    float2 one_minus_ej2q = csub(one, ej2q);

    // R_TE = r_te * (1 - ej2q) / (1 - r_te^2 * ej2q + 1e-10)
    float2 r_te_sq = cmul(r_te, r_te);
    float2 num_TE = cmul(r_te, one_minus_ej2q);
    float2 den_TE = cadd(csub(one, cmul(r_te_sq, ej2q)), eps);
    R_TE = cdiv(num_TE, den_TE);

    float2 r_tm_sq = cmul(r_tm, r_tm);
    float2 num_TM = cmul(r_tm, one_minus_ej2q);
    float2 den_TM = cadd(csub(one, cmul(r_tm_sq, ej2q)), eps);
    R_TM = cdiv(num_TM, den_TM);
}

// --- double-precision (float64) versions of the helpers above -------------
//
// Used by the `double` variant of the BSDF forward kernel to push max
// per-path error from ~1e-4 (float32 accumulated rounding) down to
// float32 representability (~1e-7), independent of fma ordering. Inputs
// stay float32 contiguous; the kernel promotes inside and demotes on
// output.

__device__ __forceinline__ double2 d_cadd(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ double2 d_csub(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ double2 d_cmul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y,
                        a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ double2 d_cscale(double2 a, double s) {
    return make_double2(a.x * s, a.y * s);
}
__device__ __forceinline__ double d_cabs_sq(double2 a) {
    return a.x * a.x + a.y * a.y;
}
__device__ __forceinline__ double2 d_cdiv(double2 a, double2 b) {
    double denom = b.x * b.x + b.y * b.y;
    double inv = 1.0 / fmax(denom, 1e-300);
    return make_double2((a.x * b.x + a.y * b.y) * inv,
                        (a.y * b.x - a.x * b.y) * inv);
}
__device__ __forceinline__ double2 d_csqrt(double2 a) {
    if (a.x == 0.0 && a.y == 0.0) return make_double2(0.0, 0.0);
    double r = sqrt(a.x * a.x + a.y * a.y);
    double re, im;
    if (a.x >= 0.0) {
        re = sqrt(0.5 * (r + a.x));
        im = (re > 0.0) ? (0.5 * a.y / re) : 0.0;
    } else {
        im = sqrt(0.5 * (r - a.x));
        if (a.y < 0.0) im = -im;
        re = (im != 0.0) ? (0.5 * a.y / im) : 0.0;
    }
    return make_double2(re, im);
}
__device__ __forceinline__ double2 d_cexp(double2 a) {
    double er = exp(a.x);
    double y_reduced = fmod(a.y, 6.283185307179586);
    double s, c;
    sincos(y_reduced, &s, &c);
    return make_double2(er * c, er * s);
}

__device__ __forceinline__ double d_clamp(double x, double lo, double hi) {
    return fmin(fmax(x, lo), hi);
}

__device__ __forceinline__ double d_vdot(const double ax, const double ay, const double az,
                                          const double bx, const double by, const double bz) {
    return ax * bx + ay * by + az * bz;
}
__device__ __forceinline__ void d_vcross(const double ax, const double ay, const double az,
                                          const double bx, const double by, const double bz,
                                          double& ox, double& oy, double& oz) {
    ox = ay * bz - az * by;
    oy = az * bx - ax * bz;
    oz = ax * by - ay * bx;
}

__device__ __forceinline__ void itu_slab_fresnel_d(
    double eps_real, double eps_imag, double cos_i, double thickness,
    double2& R_TE, double2& R_TM)
{
    cos_i = d_clamp(cos_i, 1e-12, 1.0);
    double sin2 = 1.0 - cos_i * cos_i;
    double2 eta = make_double2(eps_real, -eps_imag);
    double2 inside = d_csub(eta, make_double2(sin2, 0.0));
    double2 a = d_csqrt(inside);
    double2 cos_i_c = make_double2(cos_i, 0.0);
    double2 eps = make_double2(1e-10, 0.0);
    double2 num_te = d_csub(cos_i_c, a);
    double2 den_te = d_cadd(d_cadd(cos_i_c, a), eps);
    double2 r_te = d_cdiv(num_te, den_te);
    double2 eta_cos_i = d_cscale(eta, cos_i);
    double2 num_tm = d_csub(eta_cos_i, a);
    double2 den_tm = d_cadd(d_cadd(eta_cos_i, a), eps);
    double2 r_tm = d_cdiv(num_tm, den_tm);
    double2 q = d_cscale(a, K_WAVE_D * thickness);
    double2 ej2q = d_cexp(make_double2(2.0 * q.y, -2.0 * q.x));
    double2 one = make_double2(1.0, 0.0);
    double2 one_minus_ej2q = d_csub(one, ej2q);
    double2 r_te_sq = d_cmul(r_te, r_te);
    double2 num_TE = d_cmul(r_te, one_minus_ej2q);
    double2 den_TE = d_cadd(d_csub(one, d_cmul(r_te_sq, ej2q)), eps);
    R_TE = d_cdiv(num_TE, den_TE);
    double2 r_tm_sq = d_cmul(r_tm, r_tm);
    double2 num_TM = d_cmul(r_tm, one_minus_ej2q);
    double2 den_TM = d_cadd(d_csub(one, d_cmul(r_tm_sq, ej2q)), eps);
    R_TM = d_cdiv(num_TM, den_TM);
}

} // namespace mm25v5
