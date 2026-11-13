# Environment for running CUDA.jl and mGV on an AMD GPU through ZLUDA.
# See docs/zluda-cuda-run.md, section 2. Source it, do not execute it:
#
#   export ZLUDA_WORK=/path/to/work/dir      # everything the experiment creates goes here
#   source docs/zluda-scripts/env.sh
#
# Optional, before sourcing: MGV_SRC (the mGV checkout to copy from; default: the checkout this
# file is in), MGV_COPY (the copy of mGV to run; default $ZLUDA_WORK/mGV), ZLUDA_JULIA_ENV (the
# Julia environment; default $ZLUDA_WORK/env).

if [ -z "${ZLUDA_WORK:-}" ]; then
    echo "env.sh: set ZLUDA_WORK to a work directory first" >&2
    return 1 2>/dev/null || exit 1
fi

export ZLUDA_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MGV_SRC="${MGV_SRC:-$(cd "$ZLUDA_SCRIPTS/../.." && pwd)}"

# Where things live (setup.sh creates them)
export ZLUDA_DIR="$ZLUDA_WORK/zluda/zluda"            # the unpacked ZLUDA release (libnvcuda.so, ...)
export ZLUDA_LIBCUDA_DIR="$ZLUDA_WORK/libcuda"        # holds only libcuda.so(.1) -> libnvcuda.so
export ZLUDA_CUDACORE="$ZLUDA_WORK/CUDACore"          # CUDACore 6.5.1 with zluda-cudacore-6.5.1.patch
export ZLUDA_JULIA_ENV="${ZLUDA_JULIA_ENV:-$ZLUDA_WORK/env}"
export MGV_COPY="${MGV_COPY:-$ZLUDA_WORK/mGV}"        # the copy of mGV that is run
export MGV_OUT="$ZLUDA_WORK/out"                      # outputs of the mgv_*.jl scripts

# CUDA.jl opens the driver by its name, "libcuda.so.1", so the dynamic linker has to find ZLUDA's.
# Only this directory, not $ZLUDA_DIR: that one also has ZLUDA's libcublas/libcusparse/... and
# CUDA.jl would pick those up instead of the libraries of its own runtime artifact.
case ":${LD_LIBRARY_PATH:-}:" in
    *":$ZLUDA_LIBCUDA_DIR:"*) ;;
    *) export LD_LIBRARY_PATH="$ZLUDA_LIBCUDA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
# Do not let CUDA.jl replace the system driver (here: ZLUDA) by the NVIDIA forward-compatibility
# driver it bundles (it does so if that one reports a newer version).
export JULIA_CUDA_USE_COMPAT=false
# HIP/ROCr device 0 is the discrete GPU on the machine this was developed on. ZLUDA exposed only
# that GPU with or without these, but it has been reported to pick an integrated GPU.
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
# ZLUDA (and comgr below it) cache compiled kernels under $XDG_CACHE_HOME/zluda and .../comgr.
export XDG_CACHE_HOME="$ZLUDA_WORK/cache"
# CUDA.jl dumps kernel images that the driver rejects with `tempname()`, i.e. into $TMPDIR.
export TMPDIR="$ZLUDA_WORK/tmp"
# Hide the default (global) Julia environment. test/gpu_test.jl loads every GPU package it can
# find; AMDGPU from the global environment then fails to load its own ROCm next to the one ZLUDA
# loaded ("undefined symbol: hsa_amd_vmem_export_fabric_handle, version ROCR_1").
export JULIA_LOAD_PATH="@:@stdlib"
