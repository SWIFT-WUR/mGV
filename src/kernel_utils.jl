"""
    @kernel_inbounds [config...] function name(args...) ... end

Define a KernelAbstractions kernel, like `@kernel`, with bounds checking disabled in its body.

A bounds check adds a branch that reports an exception to every array access. On the CPU
backend (PoCL) these branches prevent vectorizing the kernel across work-items, which makes
small kernels 2-5x slower. Kernels must therefore only index within bounds: guard the
work-item index (`if i <= n`) where the `ndrange` can exceed the array size.

To check the bounds anyway, start Julia with `--check-bounds=yes`, which overrides
`@inbounds`. An out-of-bounds access then throws a `KernelException`. `Pkg.test` does this
by default.
"""
macro kernel_inbounds(args...)
    fn = last(args)
    Meta.isexpr(fn, :function, 2) ||
        throw(ArgumentError("@kernel_inbounds expects a `function name(args...) ... end` definition"))
    signature, body = fn.args
    # What `@inbounds` expands to. Wrapping the body in `@inbounds` itself does not work:
    # `@kernel` rewrites the body, and drops the macro call.
    fn_inbounds = Expr(
        :function, signature,
        Expr(:block, Expr(:inbounds, true), body, Expr(:inbounds, :pop)),
    )
    return esc(Expr(:macrocall, Symbol("@kernel"), __source__, args[1:(end - 1)]..., fn_inbounds))
end

# Method table of the KernelAbstractions CPU backend (PoCL), to override functions only in
# kernels compiled for it. Not `KernelAbstractions.POCL.@device_override`: it only works
# inside the POCL module.
const POCL_METHOD_TABLE = KernelAbstractions.POCL.method_table

"""
    foldsteps(f, init, Val(N))

Run `N` steps of a loop with state: `state = f(step, state)` for `step` in `1:N`, starting
from `init`, and return the final state. Use with a `do` block, which should start with
`@inline`:

    state = foldsteps(init, Val(24)) do step, (a, b)
        @inline
        return (a + b, b)
    end

On the CPU backend (PoCL) the loop is unrolled: PoCL vectorizes a kernel across work-items,
which an inner loop prevents. And without vectorization, unrolling still lets independent
work-items run interleaved, instead of waiting for each step of a dependent chain. On GPUs
this is a normal loop, as unrolling a long loop body there uses more registers, which makes
the kernel slower.

Write the state as destructured arguments of the `do` block, rather than assigning variables
of the enclosing function: a closure that assigns those boxes them, which kernels can't use.
"""
@inline function foldsteps(f::F, init, ::Val{N}) where {F, N}
    state = init
    for step in 1:N
        state = f(step, state)
    end
    return state
end

@generated function unrolled_foldsteps(f::F, init, ::Val{N}) where {F, N}
    return quote
        $(Expr(:meta, :inline))
        state = init
        Base.Cartesian.@nexprs $N step -> (state = f(step, state))
        return state
    end
end

# Not a "consistent" overlay: for floating-point state, the results of the two versions can
# differ in the last bits, as the compiler may contract the unrolled operations differently.
Base.Experimental.@overlay POCL_METHOD_TABLE @inline foldsteps(f::F, init, n::Val{N}) where {F, N} =
    unrolled_foldsteps(f, init, n)
