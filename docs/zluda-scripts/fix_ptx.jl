# Applies the PTX fix-ups of the CUDACore patch to a PTX file, outside CUDA.jl, e.g. to test with
# loadptx.jl whether ZLUDA accepts a kernel dumped with ZLUDA_PTX_DUMP_DIR.
#   source env.sh; julia --startup-file=no fix_ptx.jl in.ptx out.ptx
include(joinpath(ENV["ZLUDA_CUDACORE"], "src", "compiler", "zluda_ptx.jl"))
write(ARGS[2], zluda_fix_ptx(read(ARGS[1], String)))
