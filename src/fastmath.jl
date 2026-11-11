# `exp2`, `log2` and `log10` for device code, with Float32 versions written as plain
# arithmetic for the KernelAbstractions CPU backend (PoCL).
#
# On PoCL, the `exp2`, `log2` and `log10` builtins are calls that LLVM cannot vectorize, so a
# kernel using them runs one work-item at a time (`exp` and `log` do vectorize). The
# polynomial versions below are only arithmetic, bit operations and `ifelse`, so a kernel
# using them vectorizes across work-items: about 7x faster for `exp2` and 10x for `log2` and
# `log10`. GPUs have fast native instructions for these functions, which beat the
# polynomials, so `fastexp2`, `fastlog2` and `fastlog10` call Base, and only PoCL's method
# table overrides them with the polynomial versions.

# p * 2^n, for n in [-162, 128]: two factors keep each exponent valid, so the result
# overflows to Inf and underflows gradually to subnormals and zero
@inline function _scale_pow2(p::Float32, n::Int32)
    n1 = n >> Int32(1)
    n2 = n - n1
    s1 = reinterpret(Float32, (n1 + Int32(127)) << 23)
    s2 = reinterpret(Float32, (n2 + Int32(127)) << 23)
    return (p * s1) * s2
end

"""
    polynomial_exp2(x::Float32)

`2^x`, within 1 ulp of the exact result. Vectorizes on PoCL, unlike `exp2`. Special values
match `exp2`.
"""
@inline function polynomial_exp2(x::Float32)
    # beyond these, the result is Inf or 0 (NaN passes through)
    xc = clamp(x, -160.0f0, 128.0f0)
    n = round(xc)
    r = xc - n  # exact, |r| <= 0.5
    # 2^r = 1 + r P(r) on [-0.5, 0.5] (Cephes exp2f)
    p = muladd(
        r,
        evalpoly(
            r,
            (
                6.931472f-1, 2.4022648f-1, 5.5503324f-2,
                9.618437f-3, 1.3398874f-3, 1.5353362f-4,
            ),
        ),
        1.0f0,
    )
    return _scale_pow2(p, unsafe_trunc(Int32, n))
end

"""
    polynomial_log2(x::Float32)

`log2(x)`, within 1.8 ulp of the exact result (including subnormal `x`). Vectorizes on PoCL,
unlike `log2`. Gives `-Inf` for zero, `Inf` for `Inf`, and `NaN` for negative `x` or `NaN`.
"""
@inline function polynomial_log2(x::Float32)
    # x = 2^e * m, with m in [sqrt(1/2), sqrt(2)); subnormals are scaled up by 2^23 first
    subnormal = x < floatmin(Float32)
    xs = ifelse(subnormal, x * 8.388608f6, x)
    b = reinterpret(Int32, xs)
    e = (b >> 23) - ifelse(subnormal, Int32(150), Int32(127))
    m = reinterpret(Float32, (b & Int32(0x007fffff)) | Int32(0x3f800000))  # [1, 2)
    large = m > 1.4142135f0
    m = ifelse(large, 0.5f0 * m, m)
    e = ifelse(large, e + Int32(1), e)

    # log(1 + f) = f - f^2/2 + f^3 P(f) (Cephes logf)
    f = m - 1.0f0
    z = f * f
    y = f * z * evalpoly(
        f,
        (
            3.3333331f-1, -2.4999994f-1, 2.0000714f-1, -1.6668057f-1, 1.4249323f-1,
            -1.2420141f-1, 1.1676999f-1, -1.1514610f-1, 7.0376836f-2,
        ),
    )
    ln_m = muladd(-0.5f0, z, y) + f

    # log2(m) = ln(m) * log2(e), with log2(e) = 1 + 0.44269504 to keep the rounding error low
    r = muladd(ln_m, 0.44269504f0, ln_m) + Float32(e)

    r = ifelse(x == Inf32, Inf32, r)
    r = ifelse(x == 0.0f0, -Inf32, r)
    return ifelse((x < 0.0f0) | isnan(x), NaN32, r)
end

"""
    polynomial_log10(x::Float32)

`log10(x)`, computed as `polynomial_log2(x) * log10(2)`: within 3 ulp. Vectorizes on PoCL,
unlike `log10`. Special values match [`polynomial_log2`](@ref).
"""
@inline polynomial_log10(x::Float32) = polynomial_log2(x) * 0.30103f0

"""
    fastexp2(x)
    fastlog2(x)
    fastlog10(x)

`exp2`, `log2` and `log10`, except in Float32 kernels compiled for PoCL, where they are
[`polynomial_exp2`](@ref), [`polynomial_log2`](@ref) and [`polynomial_log10`](@ref).
"""
@inline fastexp2(x) = exp2(x)
@inline fastlog2(x) = log2(x)
@inline fastlog10(x) = log10(x)

# Not `KernelAbstractions.POCL.@device_override`: it only works inside the POCL module. And a
# plain (not "consistent") overlay, since the results can differ from Base in the last bits.
const POCL_METHOD_TABLE = KernelAbstractions.POCL.method_table
Base.Experimental.@overlay POCL_METHOD_TABLE @inline fastexp2(x::Float32) = polynomial_exp2(x)
Base.Experimental.@overlay POCL_METHOD_TABLE @inline fastlog2(x::Float32) = polynomial_log2(x)
Base.Experimental.@overlay POCL_METHOD_TABLE @inline fastlog10(x::Float32) = polynomial_log10(x)

"""
    fastpow(x, y)

`x^y` for `x ≥ 0`, computed as `fastexp2(y * fastlog2(x))`. On the KernelAbstractions CPU
backend (PoCL) this is cheaper than `pow` and `cbrt`, and vectorizes. The result can differ
from `x^y` in the last few bits; more when `y * log2(x)` is large, as its rounding error is
amplified. Negative `x` gives `NaN` in kernels, and a `DomainError` in host code.
"""
@inline fastpow(x::T, y::T) where {T <: AbstractFloat} =
    ifelse(iszero(y), one(T), fastexp2(y * fastlog2(x)))
