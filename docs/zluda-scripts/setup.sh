#!/usr/bin/env bash
# Sets up everything for docs/zluda-cuda-run.md under $ZLUDA_WORK: downloads and unpacks ZLUDA,
# makes the libcuda-only directory, the Julia environment with the patched CUDACore, and a copy of
# mGV. Safe to run again (it skips what exists). Needs: gh, git, patch, julia, network access.
#
#   export ZLUDA_WORK=/path/to/work/dir
#   bash docs/zluda-scripts/setup.sh
#
# Optional: MGV_REV (commit of mGV to copy, default HEAD). For a second copy of mGV at another
# commit, run it again with MGV_REV, MGV_COPY and ZLUDA_JULIA_ENV set (see the document).
set -euo pipefail

ZLUDA_TAG=v7-preview.21
ZLUDA_ASSET=zluda-linux-eb89ae3b.tar.gz
ZLUDA_SHA256=b02fb8fc0a97b8a0c77aa93cba89fe2ffefc32ba59021bcb07529128955a8a54
CUDA_VERSION=6.5.1
KA_VERSION=0.10.7
CUDA_RUNTIME=13.0     # ZLUDA v7-preview.21 reports itself as a driver for CUDA 13.0

: "${ZLUDA_WORK:?set ZLUDA_WORK to a work directory}"
mkdir -p "$ZLUDA_WORK"
ZLUDA_WORK="$(cd "$ZLUDA_WORK" && pwd)"; export ZLUDA_WORK
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
PATCH="$HERE/../zluda-cudacore-$CUDA_VERSION.patch"
mkdir -p "$ZLUDA_WORK"/{dl,zluda,out,tmp,cache} "$ZLUDA_LIBCUDA_DIR" "$ZLUDA_JULIA_ENV"

echo "== 1. ZLUDA $ZLUDA_TAG"
if [ ! -f "$ZLUDA_WORK/dl/$ZLUDA_ASSET" ]; then
    gh release download "$ZLUDA_TAG" -R vosen/ZLUDA -p "$ZLUDA_ASSET" -D "$ZLUDA_WORK/dl"
fi
echo "$ZLUDA_SHA256  $ZLUDA_WORK/dl/$ZLUDA_ASSET" | sha256sum -c -
[ -f "$ZLUDA_DIR/libnvcuda.so" ] || tar -xzf "$ZLUDA_WORK/dl/$ZLUDA_ASSET" -C "$ZLUDA_WORK/zluda"

echo "== 2. libcuda-only directory"
ln -sfn "$ZLUDA_DIR/libnvcuda.so" "$ZLUDA_LIBCUDA_DIR/libcuda.so.1"
ln -sfn "$ZLUDA_DIR/libnvcuda.so" "$ZLUDA_LIBCUDA_DIR/libcuda.so"

echo "== 3. Julia environment $ZLUDA_JULIA_ENV"
# No precompilation until everything is in place (stock CUDACore fails to precompile on ZLUDA)
export JULIA_PKG_PRECOMPILE_AUTO=0
if [ ! -f "$ZLUDA_JULIA_ENV/Project.toml" ]; then
    # The JLLs have to be listed in the project (here under [extras]), otherwise their
    # preferences in LocalPreferences.toml are ignored.
    cat > "$ZLUDA_JULIA_ENV/Project.toml" <<TOML
[extras]
CUDA_Driver_jll = "4ee394cb-3365-5eb0-8335-949819d2adfc"
CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
TOML
    cat > "$ZLUDA_JULIA_ENV/LocalPreferences.toml" <<TOML
[CUDA_Runtime_jll]
version = "$CUDA_RUNTIME"

[CUDA_Driver_jll]
compat = "false"
TOML
    julia --startup-file=no --project="$ZLUDA_JULIA_ENV" -e '
        import Pkg
        Pkg.add([Pkg.PackageSpec(name = "CUDA", version = ARGS[1]),
                 Pkg.PackageSpec(name = "KernelAbstractions", version = ARGS[2])])' "$CUDA_VERSION" "$KA_VERSION"
fi

echo "== 4. patched CUDACore in $ZLUDA_CUDACORE"
if [ ! -d "$ZLUDA_CUDACORE" ]; then
    src="$(julia --startup-file=no --project="$ZLUDA_JULIA_ENV" -e '
        import Pkg
        for (_, info) in Pkg.dependencies()
            info.name == "CUDACore" && print(info.source)
        end')"
    grep -q "^version = \"$CUDA_VERSION\"" "$src/Project.toml" || { echo "CUDACore at $src is not $CUDA_VERSION" >&2; exit 1; }
    cp -r "$src" "$ZLUDA_CUDACORE"
    chmod -R u+w "$ZLUDA_CUDACORE"
    (cd "$ZLUDA_CUDACORE" && patch -p1 < "$PATCH")
fi
julia --startup-file=no --project="$ZLUDA_JULIA_ENV" -e 'import Pkg; Pkg.develop(path = ARGS[1])' "$ZLUDA_CUDACORE"

echo "== 5. copy of mGV in $MGV_COPY (from $MGV_SRC)"
if [ ! -d "$MGV_COPY" ]; then
    mkdir -p "$MGV_COPY"
    # input_data is large: left out of the copy and linked instead (it is only read)
    git -C "$MGV_SRC" archive "${MGV_REV:-HEAD}" -- . ':!input_data' | tar -x -C "$MGV_COPY"
    ln -s "$MGV_SRC/input_data" "$MGV_COPY/input_data"
    # configs/mekong_config.toml writes to ../output_data/mekong/, which must exist
    mkdir -p "$MGV_COPY/output_data/mekong"
fi
julia --startup-file=no --project="$ZLUDA_JULIA_ENV" -e '
    import Pkg
    Pkg.develop(path = ARGS[1])
    Pkg.add(["BasicModelInterface", "Zarr", "NCDatasets", "TOML", "Test", "Statistics", "Serialization"])' "$MGV_COPY"

echo "== 6. artifacts and precompilation (with ZLUDA as the driver)"
unset JULIA_PKG_PRECOMPILE_AUTO
julia --startup-file=no --project="$ZLUDA_JULIA_ENV" -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "== done. Next:  source $HERE/env.sh  &&  cd $HERE  &&  julia --startup-file=no driver_query.jl"
