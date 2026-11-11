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
