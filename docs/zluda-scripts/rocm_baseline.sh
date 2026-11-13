#!/usr/bin/env bash
# Native AMDGPU (ROCm) baseline for the CPU-versus-GPU comparison of mGV, on the same copy of mGV
# as the ZLUDA runs (section 4.4 of docs/zluda-cuda-run.md). Run it in a shell in which env.sh has
# NOT been sourced: ZLUDA must not be on LD_LIBRARY_PATH, and AMDGPU loads its own ROCm.
#
#   ZLUDA_WORK=/path/to/work/dir bash docs/zluda-scripts/rocm_baseline.sh [ntimesteps]
#
# It makes a second Julia environment, $ZLUDA_WORK/env-rocm (a copy of $ZLUDA_WORK/env plus AMDGPU).
set -euo pipefail
: "${ZLUDA_WORK:?set ZLUDA_WORK to the work directory of setup.sh}"
ZLUDA_WORK="$(cd "$ZLUDA_WORK" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case ":${LD_LIBRARY_PATH:-}:" in *":$ZLUDA_WORK/libcuda:"*) echo "ZLUDA is on LD_LIBRARY_PATH: use a fresh shell" >&2; exit 1;; esac
unset JULIA_LOAD_PATH JULIA_CUDA_USE_COMPAT
ENVR="$ZLUDA_WORK/env-rocm"
if [ ! -d "$ENVR" ]; then
    mkdir -p "$ENVR"
    cp "$ZLUDA_WORK/env/Project.toml" "$ZLUDA_WORK/env/Manifest.toml" "$ZLUDA_WORK/env/LocalPreferences.toml" "$ENVR/"
    julia --startup-file=no --project="$ENVR" -e 'import Pkg; Pkg.add(Pkg.PackageSpec(name = "AMDGPU", version = "2.11"))'
fi
export MGV_COPY="${MGV_COPY:-$ZLUDA_WORK/mGV}" MGV_OUT="$ZLUDA_WORK/out" TMPDIR="$ZLUDA_WORK/tmp"
MGV_GPU=rocm MGV_SAVE="$MGV_OUT/rocm.jls" julia --startup-file=no -t 4,2 --project="$ENVR" "$HERE/mgv_compare.jl" "${1:-10}"
