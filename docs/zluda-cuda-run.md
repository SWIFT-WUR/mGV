# Running mGV's CUDA backend on an AMD GPU through ZLUDA

Experiment of 2026-11-13 on the development machine (Radeon RX 7800 XT, no NVIDIA GPU). This
document describes how to repeat it from nothing, what had to be worked around, and what came
out. Companion files, all untracked like this one:

- [`zluda-cudacore-6.5.1.patch`](zluda-cudacore-6.5.1.patch): the changes to CUDACore 6.5.1
  (the core package of CUDA.jl 6.5.1) without which nothing runs.
- [`zluda-scripts/`](zluda-scripts/): `env.sh`, `setup.sh` and the test scripts. They take all
  paths from the variable `ZLUDA_WORK` (a work directory of your choice) and from their own
  location, so they do not depend on where the experiment was first run.

## 1. Summary

[ZLUDA](https://github.com/vosen/ZLUDA) is a replacement for NVIDIA's driver library
`libcuda.so` that implements the CUDA driver API on top of AMD's HIP runtime (part of ROCm), so
that unmodified CUDA programs run on AMD GPUs. The question was whether mGV's CUDA backend
(`ext/CUDAExt.jl`, through CUDA.jl and KernelAbstractions), and in particular the CUDA host
copier for asynchronous output copies, can be tested this way on a machine without an NVIDIA GPU.

Outcome:

- **Stock CUDA.jl 6.5.1 does not work on ZLUDA v7-preview.21.** `using CUDA` already fails
  during precompilation. With eight workarounds, kept as a patch to a private copy of CUDACore,
  CUDA.jl is functional, mGV runs on the RX 7800 XT with `backend = "cuda"`, and
  `test/gpu_test.jl` passes both without the CUDA host copier (commit `1e2849f`) and with it
  (commit `b665972`).
- The differences between mGV's output on CUDA/ZLUDA and on the CPU are of the same size as
  those between native AMDGPU and the CPU on the same machine.

Caveats, each detailed below:

1. **Pinning is unsupported.** ZLUDA cannot page-lock existing host memory
   (`cuMemHostRegister_v2` is not implemented). mGV's output blocks therefore stay pageable, and
   every copy the host copier queues blocks the model until it is done. The copier's
   asynchronous path was exercised only in a stand-alone script, with memory allocated by the
   driver, not inside mGV.
2. **Intermittent crashes.** 7 of 35 mGV runs on ZLUDA ended in a segmentation fault and 2 more
   exited silently with status 0 and no output. They are tied to ZLUDA, not to the copier.
3. **One silently wrong result:** Float64 `rem`/`mod` on the device. mGV's kernels are Float32
   only and are not affected.
4. **Weak evidence for NVIDIA hardware.** The patch bypasses several parts of CUDA.jl that an
   NVIDIA driver uses, and timing, numerics and failure modes are those of ZLUDA and HIP
   (section 6).

Throughout, "measured" means observed in a run described here; "read" means taken from source
code or documentation; "conjecture" is marked as such.

## 2. Setup

### 2.1 Versions and identifiers

| Item | Version |
|---|---|
| Machine | Ryzen 7 7700 (16 threads), 61 GiB RAM, Radeon RX 7800 XT (gfx1101, 16 GiB) and the Raphael integrated GPU (gfx1036); Linux Mint 22.3, kernel 6.8.0-149 |
| ZLUDA | release tag `v7-preview.21` (pre-release, published 2026-11-11), asset `zluda-linux-eb89ae3b.tar.gz`, 41 849 409 bytes, sha256 `b02fb8fc0a97b8a0c77aa93cba89fe2ffefc32ba59021bcb07529128955a8a54` |
| ROCm used by ZLUDA | `/opt/rocm-7.2.2` (ROCm 7.2.2; `libamdhip64.so.7.2.70202`, `hipconfig --version` 7.2.53211-671d39a71e) |
| Second ROCm on the machine | `/opt/rocm/core-10.1` (version file `10.1.0`; `libamdhip64.so.7.16.26385`); `/opt/rocm/{bin,lib,...}` and `/usr/bin/hipconfig` point to it, and AMDGPU.jl loads it |
| Julia | 1.12.6 (LLVM 18.1.7) |
| CUDA.jl | CUDA 6.5.1, CUDACore 6.5.1 (patched copy), CUDATools 6.5.1 |
| CUDA JLLs | CUDA_Driver_jll 13.5.0+0, CUDA_Runtime_jll 0.25.0+3, CUDA_Compiler_jll 0.6.5+0, NVPTX_LLVM_Backend_jll 23.1.2+2 |
| CUDA toolkit artifacts | runtime 13.0.0 (`libcudart.so.13.0.96`), compiler 13.0.88 |
| Other packages | KernelAbstractions 0.10.7, KernelInterface 0.4.6, GPUCompiler 2.22.2, GPUArrays 12.2.0, LLVM.jl 10.4.3, AcceleratedKernels 0.5.12, UnsafeAtomics 0.4.0, Zarr 0.10.2, NCDatasets 0.14.15, BasicModelInterface 0.1.1, ZarrScaleCast 0.1.0 (`main`); AMDGPU 2.11.0 for the native baseline |
| mGV | without CUDA host copier: commit `1e2849f` (`1e2849fd6827d8836e16bd84ce1d5bd3836792ba`); with it: commit `b665972` (`b6659720acfb1c5374af098010cc7f515fa464c7`, "Enable async copying for CUDA") |

The copier was first tested as uncommitted working-tree files copied over `1e2849f` at 20:45
CET. Commit `b665972`, made at 21:09 the same evening, contains exactly those files; the tests
were repeated on it. Their sha256:

```
1f3681f17d86fc12a44e38ad4fcffb63ecf2f666977191e8f55e84b981884f18  ext/CUDAExt.jl
0e63cc33c21ea125ea2aa7773228f195fef34fa7b1ea3930c9137a2702a29deb  src/async_writer.jl
1832c15b4a2994fa2b56f1c01e7ed4d10c4cfa7d02320699824dd3b810cf619e  test/gpu_test.jl
c480f539436b4d3df172f254a8665109d463fc63ab42ed1978aa05890a7bf3da  README.md
956753f1772e4db9ce709ff34b5f8a72f44f46a2bfbc3b6f3da558ebe5c2f213  docs/src/performance.md
```

**Which ROCm ZLUDA uses, and why.** ZLUDA's `libnvcuda.so` has `libamdhip64.so` as a plain
dependency, so the dynamic linker picks it. On this machine `/etc/ld.so.conf.d/10-rocm-opencl.conf`
contains `/opt/rocm-7.2.2/lib`, so that is the one found (`ldd libnvcuda.so`, and the list of
loaded libraries printed by `driver_query.jl`). To make ZLUDA use the other install, put its
`lib` directory in front on `LD_LIBRARY_PATH`; this was tried for the timing test only
(section 4.3) and gave the same results. ZLUDA's documentation (`docs/src/quick_start.md` in its
repository; the release archive itself contains no README) states no ROCm version requirement
for Linux.

### 2.2 Quick route

Needs `gh` (GitHub CLI), `git`, `patch`, `julia` and network access. `ZLUDA_WORK` is a
directory for everything the experiment creates (about 190 MB), for instance under `/tmp`.

```bash
export ZLUDA_WORK=/path/to/work/dir
MGV_REV=1e2849f bash docs/zluda-scripts/setup.sh    # about 1.5 minutes if the artifacts are already in ~/.julia
source docs/zluda-scripts/env.sh
cd "$ZLUDA_WORK"                                    # any directory outside the repository
julia --startup-file=no "$ZLUDA_SCRIPTS/driver_query.jl"
julia --project="$ZLUDA_JULIA_ENV" "$ZLUDA_SCRIPTS/step1.jl"
```

`setup.sh` was run from an empty work directory on 2026-11-13 and the checks of section 4 were
repeated in the result. The download of the CUDA artifacts was not part of that repeat (they
were already in `~/.julia`), and `MGV_REV` was left at its default, `HEAD`, which was `1e2849f`
when the first copy was made and `b665972` when the second was.

For a second copy of mGV at another commit (the one with the copier), with its own Julia
environment, in the same work directory:

```bash
export ZLUDA_WORK=/path/to/work/dir
export MGV_REV=b665972 MGV_COPY=$ZLUDA_WORK/mGV-wt ZLUDA_JULIA_ENV=$ZLUDA_WORK/env-wt
bash docs/zluda-scripts/setup.sh
source docs/zluda-scripts/env.sh        # with the same three variables still set
```

To test uncommitted changes, copy the changed files over `$MGV_COPY` by hand after `setup.sh`.

### 2.3 The steps `setup.sh` takes, and why

**1. Download and unpack ZLUDA** from the official releases only, and check the hash.

```bash
gh release download v7-preview.21 -R vosen/ZLUDA -p zluda-linux-eb89ae3b.tar.gz -D "$ZLUDA_WORK/dl"
echo "b02fb8fc0a97b8a0c77aa93cba89fe2ffefc32ba59021bcb07529128955a8a54  $ZLUDA_WORK/dl/zluda-linux-eb89ae3b.tar.gz" | sha256sum -c -
tar -xzf "$ZLUDA_WORK/dl/zluda-linux-eb89ae3b.tar.gz" -C "$ZLUDA_WORK/zluda"    # gives $ZLUDA_WORK/zluda/zluda/
```

The archive (118 MB unpacked) contains `libnvcuda.so` (the driver; `libcuda.so` and
`libcuda.so.1` are symlinks to it), replacements for `libcublas`, `libcusparse`, `libcufft`,
`libcudnn` and `libnvidia-ml`, a `ptxas` executable that only copies its input to its output
(read from `ptxas/src/main.rs`), and a `trace/` directory with a logging layer (section 3.3).

**2. A directory with only `libcuda.so.1`.** ZLUDA's documented way of loading is
`LD_LIBRARY_PATH=<ZLUDA directory>` (an alternative with `LD_AUDIT=<ZLUDA directory>/zluda_ld`
exists and was not tried). CUDA.jl opens the driver by the name `libcuda.so.1`, so the dynamic
linker must find ZLUDA's. The ZLUDA directory itself must not be on `LD_LIBRARY_PATH`: CUDA.jl
then loads ZLUDA's `libcublasLt.so.13` and `libcusparse.so.12` instead of those of its own
runtime artifact and warns "CUDA runtime library `libcublasLt.so.13` was loaded from a system
path ... This may cause errors" (measured). Hence:

```bash
ln -sfn "$ZLUDA_WORK/zluda/zluda/libnvcuda.so" "$ZLUDA_WORK/libcuda/libcuda.so.1"
ln -sfn "$ZLUDA_WORK/zluda/zluda/libnvcuda.so" "$ZLUDA_WORK/libcuda/libcuda.so"
```

**3. Environment variables** ([`zluda-scripts/env.sh`](zluda-scripts/env.sh)):

| Variable | Value | Why |
|---|---|---|
| `LD_LIBRARY_PATH` | `$ZLUDA_WORK/libcuda` in front | so that `libcuda.so.1` resolves to ZLUDA |
| `JULIA_CUDA_USE_COMPAT` | `false` | CUDA_Driver_jll bundles NVIDIA's forward-compatibility driver (13.5) and switches to it when it is newer than the system driver; ZLUDA reports 13.0. Read from the JLL's `__init__`; the preference `compat = "false"` below does the same |
| `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` | `0` | restrict HIP to the discrete GPU. ZLUDA has been reported to pick an integrated GPU. Measured here: ZLUDA lists exactly one device, the RX 7800 XT, with and without these variables |
| `XDG_CACHE_HOME` | `$ZLUDA_WORK/cache` | ZLUDA and comgr (ROCm's compiler library) cache compiled kernels in `$XDG_CACHE_HOME/zluda` and `.../comgr`; this keeps them out of `~/.cache` and makes a cold cache easy to get |
| `TMPDIR` | `$ZLUDA_WORK/tmp` | CUDA.jl writes every kernel image the driver rejects to `tempname()`; without this they pile up as `/tmp/jl_*.cubin` |
| `JULIA_LOAD_PATH` | `@:@stdlib` | hides the default Julia environment. `test/gpu_test.jl` loads every GPU package it can find; with AMDGPU in the default environment it fails with `could not load library "/opt/rocm/core-10.1/lib/libamdhip64.so"` / `undefined symbol: hsa_amd_vmem_export_fabric_handle, version ROCR_1`, because ZLUDA already loaded the other ROCm's `libhsa-runtime64` |
| `MGV_COPY`, `MGV_OUT`, `ZLUDA_JULIA_ENV`, `ZLUDA_CUDACORE`, `ZLUDA_DIR`, `ZLUDA_SCRIPTS` | paths under `$ZLUDA_WORK` | used by the scripts |

**4. Julia environment** `$ZLUDA_WORK/env`. `Project.toml` lists the two JLLs under `[extras]`,
because preferences in `LocalPreferences.toml` are only applied to packages that the project
names; without this the two preferences below are silently ignored (measured: same precompile
cache, same error).

```toml
# Project.toml (the [deps] are added by Pkg)
[extras]
CUDA_Driver_jll = "4ee394cb-3365-5eb0-8335-949819d2adfc"
CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
```

```toml
# LocalPreferences.toml
[CUDA_Runtime_jll]
version = "13.0"

[CUDA_Driver_jll]
compat = "false"
```

`version = "13.0"` selects the CUDA 13.0 toolkit artifacts. Without it CUDA.jl reports "CUDA.jl
could not find an appropriate CUDA runtime to use. CUDA.jl's JLLs were precompiled without an
NVIDIA driver present." whenever the JLL was first precompiled without ZLUDA on the library
path. The packages: `CUDA` 6.5.1 and `KernelAbstractions` 0.10.7 (what mGV's `[compat]` asks
for), then `mGV` and `CUDACore` by path (`Pkg.develop`), and for the tests `BasicModelInterface`,
`Zarr`, `NCDatasets`, `TOML`, `Test`, `Statistics`, `Serialization`. `setup.sh` runs these Pkg
operations with `JULIA_PKG_PRECOMPILE_AUTO=0` and precompiles once at the end, because CUDA.jl
with the stock CUDACore fails to precompile on ZLUDA.

**5. Patched CUDACore.** The pristine source is the installed package (on this machine
`~/.julia/packages/CUDACore/gfTex`; `setup.sh` asks Pkg for the path and checks the version).

```bash
cp -r ~/.julia/packages/CUDACore/gfTex "$ZLUDA_WORK/CUDACore" && chmod -R u+w "$ZLUDA_WORK/CUDACore"
(cd "$ZLUDA_WORK/CUDACore" && patch -p1 < /path/to/mGV/docs/zluda-cudacore-6.5.1.patch)
julia --project="$ZLUDA_WORK/env" -e 'import Pkg; Pkg.develop(path = ARGS[1])' "$ZLUDA_WORK/CUDACore"
```

The patch was made with `diff -ruN` between the pristine package and the copy that was run, and
checked with `patch -p1 --dry-run` on a fresh copy of the package (applies cleanly; the result
is identical to the copy that was run). It touches seven files and adds
`src/compiler/zluda_ptx.jl`. Nothing under `~/.julia/packages` is modified.

**6. A copy of mGV**, so that outputs and Pkg do not touch the checkout:

```bash
git -C /path/to/mGV archive 1e2849f -- . ':!input_data' | tar -x -C "$ZLUDA_WORK/mGV"
ln -s /path/to/mGV/input_data "$ZLUDA_WORK/mGV/input_data"      # read only; 85 MB for the Mekong
mkdir -p "$ZLUDA_WORK/mGV/output_data/mekong"
```

`configs/mekong_config.toml` writes to `../output_data/mekong/` relative to `configs/`; mGV
stops with "Output directory '...' does not exist" if it is missing, and `test/gpu_test.jl` uses
that config as it is for its first test set. `output_data` is not tracked, so the copy does not
have it. The other scripts write to `$MGV_OUT`.

## 3. The workarounds

### 3.1 Overview

"Needed by": *CUDA.jl* means nothing in CUDA.jl works without it; *mGV* means plain CUDA.jl use
(arrays, broadcasts, simple kernels) works without it but mGV does not.

| # | Symptom on stock CUDA.jl 6.5.1 | Cause | What the patch does | Needed by |
|---|---|---|---|---|
| 1 | "CUDA.jl could not find an appropriate CUDA runtime to use" | toolkit selection was baked into the JLL's precompile cache without a driver | not in the patch: the preferences of section 2.3, step 4 | CUDA.jl |
| 2 | `CuError` with code 801 from `unsafe_wrap(CuContext, ...)` (`lib/cudadrv/context.jl:35`) while precompiling | `cuCtxGetId` returns `CUDA_ERROR_NOT_SUPPORTED` (801); CUDA.jl calls it for every driver ≥ 12 | `ctx_id_supported() = false`: contexts get no id, as with a driver older than 12 | CUDA.jl |
| 3 | "Failed to load compiled kernel image. If you think this is a bug, please file an issue and attach /tmp/jl_….cubin", code 801 | CUDA.jl assembles PTX to an NVIDIA binary (cubin) with NVIDIA's `ptxas`; ZLUDA only accepts PTX text | `assemble` returns the PTX text, NUL-terminated, and never runs `ptxas`/`nvlink` | CUDA.jl |
| 4 | the same message and code 801, now with a PTX image | `cuModuleLoadDataEx` with JIT options (CUDA.jl always passes an error-log buffer) returns 801; with zero options it succeeds | `CuModule` loads with zero options | CUDA.jl |
| 5 | code 801 right after the first kernel launch or `synchronize` | `cuCtxGetFlags` returns 801; used by the non-blocking `device_synchronize` path for drivers ≥ 12.5 | that path is disabled; the older `synchronize_object(ctx)` path is used | CUDA.jl |
| 6 | showing any `CuError` throws a second `CuError` (801); with the tracing layer the process segfaults | `cuGetErrorName` returns 801 | `name(err)` returns the numeric code | CUDA.jl |
| 7 | process aborts: `panicked at ptx/src/pass/instruction_mode_to_global_mode/mod.rs:917:59: called Option::unwrap() on a None value`, or `CUDA error: unknown error (code 999, CUDA_ERROR_UNKNOWN)` at module load, for every kernel that can throw an exception | ZLUDA's PTX compiler does not handle what Julia's exception machinery produces (section 3.2) | reduced exception runtime in `src/device/runtime.jl` and textual PTX fix-ups in `src/compiler/zluda_ptx.jl` | mGV (and any kernel with `sqrt`, a bounds check, an integer conversion, ...) |
| 8 | `CUDA error: operation not supported (code 801, CUDA_ERROR_NOT_SUPPORTED)` from `cuMemHostRegister_v2` (`register` at `memory.jl:180`, `try_register` at `pins.jl:55`, `pin` at `pins.jl:119`, `pagelock!` at `CUDAKernels.jl`) | ZLUDA does not implement registering (page-locking) existing host memory | `KernelAbstractions.pagelock!(::CUDABackend, ::Array)` returns `missing` on this error; `ZLUDA_PAGELOCK_THROW=1` restores the throw | mGV with output |

Also returning 801 but harmless: `cuLogsDumpToMemory` (CUDA.jl only uses it to decorate errors).
NVML is reported as "missing" by `CUDA.versioninfo()`; nothing here needs it. CUDA.jl warns once
per process that "Your AMD Radeon RX 7800 XT [ZLUDA] GPU (compute capability 8.5) is not fully
supported by CUDA 13.0"; it compiles for `sm_80` with PTX 9.0.

With the stock package the first failure (2) already happens while precompiling the extension
`AcceleratedKernels → CUDACoreExt`, whose precompile workload launches a kernel, so `using CUDA`
itself fails with "Failed to precompile CUDACoreExt".

### 3.2 Workaround 7 in detail

Julia kernels that can throw contain calls to CUDA.jl's device-side exception runtime
(`report_exception`, `signal_exception`, ...) and non-inlined helper functions marked
`.noreturn`, such as `julia_throw_complex_domainerror` for `sqrt`. In the no-output run of mGV,
3 of the 32 compiled kernels have such paths: `net_radiation_kernel` (`throw_boundserror`),
`aerodynamic_resistance_kernel` (`throw_complex_domainerror`) and one broadcast
(`throw_inexacterror`). The problems, all measured by loading PTX files with `loadptx.jl`
through ZLUDA's tracing layer:

| Construct in the PTX | ZLUDA's reaction | Fix |
|---|---|---|
| a non-kernel `.func` without a `ret` (the `.noreturn` helpers end in `trap;`) | panic and abort at `mod.rs:917` (read from ZLUDA's source: `cfg.functions_rets.get(name).unwrap()`). An extra `ret;` after the `trap;` does not help | the last `trap;` of such a function becomes `ret;` |
| `fence.sc.gpu;`, `fence.sc.sys;` (from the atomics in the stock exception runtime) | `Unrecognized statement "fence.sc.gpu;"`, code 999 | lines dropped |
| `isspacep.local %p, %rd;` (same origin) | `Unrecognized statement "isspacep.local \t%p1, %rd5;"`, code 999 | replaced by `setp.ne.b64 %p, %rd, %rd;` (always false) |
| `call.uni f, ();` (a call without arguments) | `Unrecognized statement "call.uni gpu_report_exception, ();"`, code 999 | rewritten to `call.uni f;` |
| a function or kernel whose last instruction is a call to a noreturn helper (it "falls off the end") | code 999 without any message | `bra.uni` to a label inserted before the function's `ret;`. A second `ret;` is not an option: a comment in ZLUDA's source says the pass relies on a single `ret` per function |
| a `.func` whose body is only `ret;` between labels (an emptied `report_exception`) | code 999 without any message. A hand-written `.func helper() { ret; }` without labels loads, so the exact trigger was not isolated | the reduced runtime functions store the status flag instead of doing nothing |
| `exit;` followed by `ret;` in `signal_exception`, and the `vprintf` calls of the stock runtime | with the three "Unrecognized statement" fixes applied, the stock runtime still failed (panic at `mod.rs:917` with `exit;`, code 999 without it). Which of the remaining constructs is responsible was not isolated | the reduced runtime has neither |

The reduced exception runtime only sets the status flag that the host checks after a kernel. An
exception is therefore still reported: `sqrt.(CuArray([1f0, -1f0]))` raises "KernelException:
exception thrown during kernel execution on device AMD Radeon RX 7800 XT [ZLUDA]" (measured,
`modrem.jl`). What is lost: the message with the exception type, and the stop of the offending
thread. A thread that throws returns from the helper and continues with whatever follows, so
after a `KernelException` the device arrays must be considered garbage.

`fence.*` and `isspacep.local` do not occur in mGV's kernels once the runtime is reduced; the
fix-ups for them remain for user code with atomics and were only tested on the PTX of the stock
runtime.

### 3.3 Re-deriving a workaround for another ZLUDA or CUDA.jl version

- **Ask the driver directly.** [`driver_query.jl`](zluda-scripts/driver_query.jl) calls
  `libcuda.so.1` without CUDA.jl and prints the driver version, the devices, and the return
  codes of `cuCtxGetId` and `cuCtxGetFlags`. If a newer ZLUDA returns 0 for them, workarounds 2
  and 5 can go. ZLUDA's library exports 728 `cu*` symbols, among them functions it does
  not implement (such as `cuCtxGetId`), so the presence of a symbol says nothing; the return
  code 801 does.
- **Trace the calls.** ZLUDA's `trace/libcuda.so.1` logs every driver call and its result and
  forwards to the library named in `ZLUDA_CUDA_LIB`. Put a directory with only a symlink to it on
  `LD_LIBRARY_PATH` (for the same reason as in section 2.3, step 2):

  ```bash
  mkdir -p "$ZLUDA_WORK/libcuda-trace" "$ZLUDA_WORK/trace"
  ln -sfn "$ZLUDA_DIR/trace/libcuda.so.1" "$ZLUDA_WORK/libcuda-trace/libcuda.so.1"
  ZLUDA_CUDA_LIB="$ZLUDA_DIR/libcuda.so" LD_LIBRARY_PATH="$ZLUDA_WORK/libcuda-trace" \
      ZLUDA_LOG_DIR="$ZLUDA_WORK/trace" julia --project="$ZLUDA_JULIA_ENV" script.jl 2> trace.txt
  grep ZLUDA_TRACE trace.txt | grep -v CUDA_SUCCESS
  ```

  Lines with `CUDA_ERROR_NOT_SUPPORTED` name the missing function. `CUDA_ERROR_NOT_READY` from
  `cuStreamQuery`/`cuEventQuery` is normal polling. Workarounds 2, 4, 5 and 6 were found this way.
- **Find out why a kernel is rejected.** Set `ZLUDA_PTX_DUMP_DIR=<dir>` (added by the patch) to
  keep the PTX of every kernel as generated, before the fix-ups; `ZLUDA_PTX_FIX=0` switches the
  fix-ups off. [`fix_ptx.jl`](zluda-scripts/fix_ptx.jl) applies the fix-ups to a file and
  [`loadptx.jl`](zluda-scripts/loadptx.jl) loads a PTX file with ZLUDA. Under the tracing layer,
  parser complaints ("Unrecognized statement ...") are written to
  `$ZLUDA_LOG_DIR/julia*/module_NNNN.log`. Failures after parsing give code 999 and no message
  at all; those were found by deleting parts of the PTX file until it loaded.
- **In CUDA.jl**, the places that depend on the driver version are found with
  `grep -rn 'driver_version() >=' lib src` in CUDACore. ZLUDA claims 13.0, so every such branch
  is taken, whether or not ZLUDA implements the calls in it. Branches that were not reached in
  this experiment (graphs/stream capture, memory pools with limits, `cuEventElapsedTime_v2`,
  unified memory) are untested.

## 4. Results

All commands are run after `source docs/zluda-scripts/env.sh`, from a directory outside the
repository, with the scripts addressed as `$ZLUDA_SCRIPTS/<name>`. "First setup" is the
directory in which the workarounds were developed; "second setup" is a new work directory made
by `setup.sh` afterwards, in which everything in this section was run again (same machine,
same evening). Unless a count is given, a result was obtained once in each setup and was
identical in both.

### 4.1 Baseline: CUDA.jl without mGV

`julia --project=$ZLUDA_JULIA_ENV $ZLUDA_SCRIPTS/step1.jl` (single-threaded):

| Check | Result |
|---|---|
| `CUDA.functional(true)` | `true` |
| `CUDA.versioninfo()` | runs: "runtime 13.0.0, artifact installation", "driver for CUDA 13.0", "compiler 13.0.88", "NVML: missing", "1 device: 0: AMD Radeon RX 7800 XT [ZLUDA] (sm_85, 15.643 GiB / 15.984 GiB available), compiles to sm_80 / PTX 9.0" |
| device | `AMD Radeon RX 7800 XT [ZLUDA]`, capability 8.5.0, 16.0 GiB |
| loaded libraries | `$ZLUDA_WORK/libcuda/libcuda.so.1`, `/opt/rocm-7.2.2/lib/libamdhip64.so`, `libcudart.so` from the runtime artifact |
| `CuArray` allocation and round trip, `CUDA.fill` | equal |
| broadcast `a .* 2f0 .+ 1f0` | equal to CPU |
| broadcast `sin.(a) .+ exp.(a ./ 1000f0) .+ sqrt.(a)`, Float32 | max abs difference from CPU 3.8146973e-6 (values up to about 34) |
| broadcast `x .^ 0.37 .+ log.(x)`, Float64 | max relative difference 2.57e-16 |
| `sum` of 1:1000 | 500500.0 |
| hand-written `@cuda` kernel (4 blocks of 256 threads) | equal |
| KernelAbstractions `@kernel` on `CUDABackend()`, 1D, and 2D with workgroup size 64 | equal |
| `CUDA.synchronize()`, `KernelAbstractions.synchronize`, `CUDA.device_synchronize()` | return |
| `CUDA.@allowscalar`, `any` | correct |

`julia --project=$ZLUDA_JULIA_ENV $ZLUDA_SCRIPTS/mathfns.jl` compares 28 functions in
broadcasts with the CPU on 4096 values in [0.1, 20], for Float32 and Float64 (two runs, identical
output). Exact: `x^3`, `abs`/`clamp`, `floor`/`round`, `muladd`, `unsafe_trunc`, `ifelse`/`isnan`,
`sign`/`copysign`, integer `fld`, `sqrt`, and Float32 `mod`/`rem`. Within 0.9 to 1.8 eps
(relative): `sin`, `cos`, `tan`, `exp`, `exp2`, `exp10`, `expm1`, `log`, `log10`, `log1p`, `cbrt`,
`tanh`, `atan`, `asin`, `x^2.5`, `hypot`, `atan(y, x)`, `sinh`/`cosh`. Wrong: Float64 `mod`/`rem`
(section 5.1).

### 4.2 Driver features the host copier needs

`julia -t 4,2 --project=$ZLUDA_JULIA_ENV $ZLUDA_SCRIPTS/step2.jl`. The host copier of
`ext/CUDAExt.jl` uses: a second stream (`CuStream()`), events (`CuEvent(EVENT_DISABLE_TIMING)`),
`CUDA.record(event, stream)`, `CUDA.wait(event, stream)` (the driver's `cuStreamWaitEvent`),
`CUDA.stream!(stream) do ... end` around `KernelAbstractions.copyto!` (the driver's
`cuMemcpyDtoHAsync`), and `CUDA.synchronize(event)` from another task under `CUDA.context!`.

| Feature | Result |
|---|---|
| `CUDA.pin(::Array)` | **fails**: `CUDA error: operation not supported (code 801, CUDA_ERROR_NOT_SUPPORTED)`, stack top `throw_api_error` (`libcuda.jl:32`), `cuMemHostRegister_v2`, `register` (`memory.jl:180`), `try_register` (`pins.jl:55`) |
| `KernelAbstractions.pagelock!(CUDABackend(), ::Array)` | stock: the same error. Patched: returns `missing`, the array is not pinned (`is_pinned` false) |
| memory allocated pinned by the driver (`CUDACore.alloc(CUDACore.HostMemory, n)`, i.e. `cuMemHostAlloc`) | works, `is_pinned` true (`step2b.jl`) |
| `KernelAbstractions.copyto!(backend, host, dev)` then `synchronize`; 4 000 000 Float32 | data equal; also into an offset view (1000 elements, rest of the array untouched). Host memory pageable here; pinned memory is covered by `step2b.jl` |
| `CuStream()` and a copy on it inside `CUDA.stream!`, then `CUDA.synchronize(stream)` | data equal |
| `CuEvent(CUDA.EVENT_DISABLE_TIMING)` | works |
| `CUDA.record(event, CUDA.stream())` | works |
| `CUDA.wait(event, other_stream)` | works |
| `CUDA.synchronize(event)` on the host; `CUDA.isdone(event)` | returns; `true` |
| record on a new, empty stream, then wait for it on the task's stream (what the copier's constructor and `begin_staging!` do) | works |
| `CUDA.synchronize(event)` from a spawned task inside `CUDA.context!(...)` (`wait_host_copy`) | works |

None of these was silently wrong in the data checks made.

### 4.3 Does `copyto!` return before the copy is done? Timing

`julia -t 4,2 --project=$ZLUDA_JULIA_ENV $ZLUDA_SCRIPTS/step2b.jl`. A kernel that takes about
0.75 s (1 000 000 elements, 20 000 `sin` evaluations each) is launched on the task's stream
without synchronizing (the launch itself returns in 0.001 to 0.002 ms). Then a 4 MB array that
the kernel does not touch is copied to the host with `KernelAbstractions.copyto!`, and the time
until `copyto!` returns and the time of the `synchronize` after it are taken. Host memory is
either an ordinary `Array` (pageable) or an `Array` wrapped around `cuMemHostAlloc` memory
(pinned). Each cell is the median of 5 repetitions within one run, with minimum and maximum in
parentheses; times in ms. Three runs: A on ROCm 7.2.2 in the first setup, B on ROCm 7.2.2 in the
second setup (re-measured), C on ROCm 10.1 (`LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:$LD_LIBRARY_PATH`
and a separate `XDG_CACHE_HOME`, first setup).

Time until `copyto!` returns:

| Case | Run A | Run B | Run C (ROCm 10.1) |
|---|---|---|---|
| long kernel, launch + synchronize (minimum of 3) | 753.5 | 743.2 | 749.2 |
| idle device, pageable | 0.359 (0.164–23.853) | 0.314 (0.258–24.12) | 0.395 (0.164–25.002) |
| idle device, pinned | 0.003 (0.002–0.018) | 0.004 (0.003–0.021) | 0.002 (0.002–0.017) |
| kernel queued, copy on the same stream, pageable | **734.516** (731.367–736.697) | **731.863** (724.265–739.545) | **736.012** (726.534–743.752) |
| kernel queued, copy on the same stream, pinned | **0.019** (0.018–0.021) | **0.019** (0.011–0.02) | **0.02** (0.008–0.022) |
| kernel queued, copy on a second stream, pageable | 0.295 (0.291–0.301) | 0.296 (0.288–0.544) | 0.292 (0.291–0.545) |
| kernel queued, copy on a second stream, pinned | 0.016 (0.015–0.022) | 0.018 (0.016–0.023) | 0.019 (0.016–0.024) |

Time of the `synchronize` that follows (median of the same 5 repetitions):

| Case | Run A | Run B | Run C |
|---|---|---|---|
| idle, pageable / pinned | 0.001 / 0.372 | 0.001 / 0.291 | 0.001 / 0.168 |
| same stream, pageable / pinned | 0.015 / 729.717 | 0.015 / 733.577 | 0.013 / 725.665 |
| second stream, pageable / pinned | 0.001 / 0.29 | 0.001 / 0.294 | 0.001 / 0.291 |

The copied data was correct in every repetition. So, on ZLUDA: a copy to **pinned** memory
returns at once and completes in stream order; a copy to **pageable** memory blocks until
everything queued before it on the same stream has run, and is complete when it returns. A copy
on another stream does not wait for the kernel. This is the behaviour NVIDIA documents for
`cuMemcpyDtoHAsync`, but here it is produced by ZLUDA on top of HIP.

**The full copier sequence** (same script, second part): fill the kernel's output array with
-1 and synchronize; launch the long kernel on the task's ("model") stream; `record(staged,
model stream)`; `wait(staged, copier stream)`; `copyto!` of the kernel's *output* to the host on
the copier stream; `record(done, copier stream)`; `synchronize(done)`. Three repetitions per
memory kind per run; times in ms.

| Run, memory | launch | record + wait | `copyto!` | `record(done)` | `synchronize(done)` |
|---|---|---|---|---|---|
| A, pageable | 0.036 / 0.024 / 0.026 | 99.573 / 0.02 / 0.022 | 644.344 / 748.545 / 734.445 | 0.028 / 0.026 / 0.015 | 2.56 / 0.026 / 0.018 |
| A, pinned | 0.026 / 0.028 / 0.028 | 0.018 / 0.022 / 0.023 | 0.021 / 0.031 / 0.023 | 0.002 / 0.003 / 0.003 | 734.532 / 731.062 / 733.46 |
| B, pageable | 0.037 / 0.024 / 0.028 | 100.073 / 0.026 / 0.018 | 637.615 / 746.915 / 731.123 | 0.028 / 0.03 / 0.032 | 2.448 / 0.025 / 0.015 |
| B, pinned | 0.032 / 0.035 / 0.031 | 0.018 / 0.027 / 0.024 | 0.022 / 0.037 / 0.025 | 0.002 / 0.003 / 0.002 | 735.179 / 732.379 / 735.88 |
| C, pageable | 0.038 / 0.032 / 0.029 | 99.573 / 0.026 / 0.02 | 638.627 / 748.288 / 733.797 | 0.025 / 0.028 / 0.014 | 2.521 / 0.027 / 0.022 |
| C, pinned | 0.028 / 0.03 / 0.028 | 0.02 / 0.023 / 0.024 | 0.02 / 0.023 / 0.022 | 0.003 / 0.003 / 0.003 | 735.401 / 733.076 / 726.304 |

In all 18 repetitions the host array contained no stale value (-1), no untouched element, and
was equal to the device array afterwards: the copier stream really waits for the event recorded
on the model stream. With pinned memory the host is free for the whole duration of the kernel
(all calls return within 0.04 ms) and only `synchronize(done)` waits. With pageable memory the
`copyto!` call itself waits.

In the very first repetition of each run, "record + wait" took about 100 ms and `copyto!`
correspondingly less (their sum is again the kernel's duration). Conjecture: a one-time cost on
the host, such as compiling these methods; it was not investigated.

### 4.4 mGV on the CUDA backend, without the host copier (commit `1e2849f`)

Mekong test domain (`configs/mekong_config.toml`: 894 active cells of 36 × 36, 3580 tiles,
routing and snow on), `julia -t 4,2`.

**Without output** (`$ZLUDA_SCRIPTS/mgv_noout.jl 3`; `variables = []`, 3 timesteps; once per
setup): works. `mGV.Model` constructed in 18.7 s / 19.1 s, first `update!` 4.75 s / 4.79 s
(32 kernels are compiled in it), later ones 0.01 s and less, `finalize!` returns.

**With output and the stock `pagelock!`**
(`ZLUDA_PAGELOCK_THROW=1 julia ... $ZLUDA_SCRIPTS/mgv_out.jl`; 4 runs): mGV stops while starting
the output writer, after "Allocating 3 output blocks of 8 timesteps (1.801 MiB)", with the error
of workaround 8. Stack: `pagelock!` (`CUDAKernels.jl`), `pin_memory!` (`src/backend_setup.jl:94`),
`OutputWriter` (`src/async_writer.jl:169`), `start_output_writer` (`src/io.jl:427`), `mGV.Model`
(`src/mGV.jl:175`).

**With output and workaround 8, against the CPU**
(`MGV_SAVE=$MGV_OUT/cuda.jls julia ... $ZLUDA_SCRIPTS/mgv_compare.jl 10`): 10 timesteps,
land-only Zarr output, unpacked Float32, `write_interval = 8`, GPU and CPU in the same process;
20 output variables. The numbers were identical in all 4 completed runs with 10 timesteps (3 in the first setup,
1 in the second). The native baseline
is `ZLUDA_WORK=... bash docs/zluda-scripts/rocm_baseline.sh 10` in a shell without `env.sh`
(AMDGPU 2.11.0 on ROCm 10.1; 2 runs, identical).

Seven variables are bit-identical to the CPU on both GPU backends: `precipitation`, `tair`,
`snow_coverage`, `snow_melt`, `swe` (all zero or forcing), `snow_albedo` and `snow_surf_temp`
(all NaN). For the other 13, "max abs" is the largest absolute difference from the CPU over all
cells and timesteps, and "rel" is that divided by the largest absolute CPU value:

| Variable | max abs CPU value | CUDA/ZLUDA: max abs | rel | AMDGPU: max abs | rel |
|---|---|---|---|---|---|
| `canopy_evaporation` | 3.561567 | 0.0003350065 | 9.41e-5 | 0.00030865893 | 8.67e-5 |
| `discharge` | 642.8963 | 0.06391907 | 9.94e-5 | 0.06616211 | 1.03e-4 |
| `net_radiation` | 154.95045 | 0.014038086 | 9.06e-5 | 0.0132751465 | 8.57e-5 |
| `potential_evaporation` | 7.8962855 | 0.0009598732 | 1.22e-4 | 0.00096559525 | 1.22e-4 |
| `soil_evaporation` | 3.0035615 | 0.00033962727 | 1.13e-4 | 0.000341177 | 1.14e-4 |
| `soil_moisture` (3 layers) | 854.96545 | 0.018676758 | 2.18e-5 | 0.018623352 | 2.18e-5 |
| `surface_runoff` | 0.7047119 | 0.009765625 | 1.39e-2 | 0.009765625 | 1.39e-2 |
| `total_et` | 6.013404 | 0.0030918121 | 5.14e-4 | 0.0030817986 | 5.12e-4 |
| `total_runoff` | 49.96806 | 0.009766102 | 1.95e-4 | 0.009765387 | 1.95e-4 |
| `transpiration` | 5.3764772 | 0.0030589104 | 5.69e-4 | 0.0030491352 | 5.67e-4 |
| `travel_time` | 2.5290768e7 | 1.9903446e7 | 0.787 | 1.5950858e7 | 0.631 |
| `tsurf` | 33.948975 | 0.0022411346 | 6.60e-5 | 0.0022201538 | 6.54e-5 |
| `water_storage` | 0.5712403 | 0.00030642748 | 5.36e-4 | 0.00022673607 | 3.97e-4 |

NaNs are in the same places as on the CPU, except for 10 values of `travel_time`, on both GPU
backends. The fraction of values that differ at all from the CPU is between 7% and 99.5% per
variable on ZLUDA and between 5% and 93% on AMDGPU. The two GPU backends thus differ from the
CPU in the same variables and by the same amounts, including the two large ones
(`surface_runoff`, `travel_time`); these are properties of the model on a GPU, not of ZLUDA.

CUDA/ZLUDA against native AMDGPU directly (`julia $ZLUDA_SCRIPTS/cmp_jls.jl $MGV_OUT/rocm.jls
$MGV_OUT/cuda.jls`; 2 comparisons, identical): the same 7 variables identical and no NaN
mismatch. Largest absolute differences: `canopy_evaporation` 0.0003350065, `discharge`
0.015113831, `net_radiation` 0.0017471313, `potential_evaporation` 0.00015687943,
`soil_evaporation` 0.0001732111, `soil_moisture` 0.009399414, `surface_runoff` 0.009277344
(1.3% of the maximum), `total_et` 0.00020194054, `total_runoff` 0.009282112, `transpiration`
0.0003437996, `travel_time` 8.695263e6 (34% of the maximum), `tsurf` 0.00017738342,
`water_storage` 0.00030642748.

The wall times these scripts print (about 4.6 to 5.6 s for the GPU part, including kernel
compilation) say nothing about speed.

**`test/gpu_test.jl`**
(`MGV_TEST_BACKEND=cuda julia -t 4,2 --project=$ZLUDA_JULIA_ENV $MGV_COPY/test/gpu_test.jl`;
once per setup, both passed; times of the first / second run):

```
Test Summary:               | Pass  Total   Time
CUDABackend BMI layer views |   43     43  36.8s        (36.5s)
Test Summary:                                       | Pass  Total  Time
CUDABackend output with and without the host copier |   24     24  0.3s
Test Summary:                                                | Pass  Total  Time
CUDABackend unpacked output with and without the host copier |   76     76  0.8s
```

At this commit CUDA has no host copier, so "with the host copier" is also the copy-task path.

### 4.5 mGV with the CUDA host copier (commit `b665972`)

Second copy and environment as in section 2.2. `test/gpu_test.jl`, same command (once on the
working-tree files in the first setup, once on the commit in the second; both passed):

```
Test Summary:               | Pass  Total   Time
CUDABackend BMI layer views |   43     43  38.3s        (36.8s)
Test Summary:                                       | Pass  Total  Time
CUDABackend output with and without the host copier |   25     25  0.3s
Test Summary:                                                | Pass  Total  Time
CUDABackend unpacked output with and without the host copier |   76     76  0.8s
```

The extra test is `copier_type !== Nothing`. The test compares packed and unpacked output of 20
timesteps with the copier and with `host_copier` redefined to return `nothing`, and requires
them to be equal.

`julia -t 4,2 --project=$ZLUDA_JULIA_ENV $ZLUDA_SCRIPTS/mgv_copier_check.jl` (6 completed runs:
4 in the first setup, 2 in the second):

- `typeof(mGV.host_copier(CUDABackend()))` is `CUDAExt.HostCopier`.
- `mGV.pin_memory!(backend, ::Array)` returns `false`: the output blocks are pageable.
- 60 timesteps with `write_interval = 8`: with the copier 0.102, 0.104, 0.102, 0.102, 0.097,
  0.097 s; without it (copy task) 0.208, 0.224, 0.223, 0.216, 0.205, 0.217 s. Outputs identical
  (20 variables) in all 6 runs. The domain is tiny and the copies block (below), so these times
  are not a performance measurement of the copier.

**What this does and does not show.** No error in the copier was found: its calls are accepted,
and the output is identical with and without it. But because the blocks are pageable, each
`copyto!` in `copy_to_host!` blocks until the model stream has reached the `staged` event
(section 4.3), so inside mGV the copies were in effect synchronous. The situation the copier is
written for (pinned blocks, copies still running while the model continues, `begin_staging!`
holding the next block back with the `done` event) occurred only in `step2b.jl`.

## 5. Known problems

### 5.1 Float64 `rem` and `mod` are silently wrong on the device

`julia --project=$ZLUDA_JULIA_ENV $ZLUDA_SCRIPTS/modrem.jl` (two runs, identical), 4096 values
of `x` in [0.1, 20]:

| Expression (broadcast over a `CuArray{Float64}`) | Values that differ from the CPU |
|---|---|
| `rem(x, 7.0)` | 1646 of 4096, e.g. x = 7.0006103515625: GPU 5.551115123125783e-17, CPU 0.0006103515625 |
| `mod(x, 3.0)` and `rem(x, 3.0)` | 2468 of 4096, e.g. x = 3.0011720657348633: GPU 2.7755575615628914e-17, CPU 0.0011720657348632812 |
| `x - 3.0 * floor(x / 3.0)` | 0 |
| `rem(Float32(x), 3f0)` | 0 |

A further check on 30 001 values in [0, 30] (the last part of `modrem.jl`; measured twice,
once with an ad-hoc version of it):

- Values with `x < y` are always right.
- `rem(x, 3.0)`: 17 991 values wrong. In each of the nine full intervals between multiples of
  3 above the first, 1999 of 3000 values are wrong.
- `rem(x, 7.0)`: 13 997 values wrong: 3999 of 7000 in each of the three full intervals above
  the first, and 2000 of 2001 in the last, partial one.
- The wrong values are not only tiny: they range from 0.0 to 1.4044477616111843e306 for
  `y = 3.0` and from 2.2250738585072014e-308 to 2.8088955232223686e306 for `y = 7.0`.
- Which values within an interval are wrong was not characterised further: remainders as small
  as 0.001 are wrong and remainders as large as `y - 0.001` are right.

No error is raised. The cause was not investigated (the instruction or library routine that
ZLUDA mistranslates is unknown). None of
the 32 mGV kernels dumped in the no-output run contains a Float64 operation, so mGV is not
affected, but any Float64 `mod`/`rem`/`%` in kernel code would be.

### 5.2 Intermittent crashes of mGV runs

Frequency. Of 35 mGV processes run on ZLUDA with `-t 4,2` (23 in the first setup, 12 in the
second; `mgv_noout`, `mgv_out`, `mgv_compare`, `mgv_copier_check`, `test/gpu_test.jl`):

- **7 ended with a segmentation fault** (exit status 139; `coredumpctl list` shows them as
  SIGSEGV), without any Julia error message or stack trace. They happened at commit `1e2849f`
  (4 times, `mgv_compare.jl`) and with the copier (3 times, `mgv_copier_check.jl`).
- **2 ended with exit status 0 and no output at all**, not even the first line the script
  prints (both `mgv_out.jl`, each time the first run of that script in a setup). Julia buffers
  standard output when it is a file, so this is what an abrupt `_exit(0)` would look like;
  whether that is what happened is not known.
- The scripts that do not load mGV (`step1`, `step2`, `step2b`, `mathfns`, `modrem`) never
  crashed this way in 11 runs with workarounds 1 to 7 in place, 5 of them with `-t 4,2`.

Kernel log (two of the crashes; the instruction offset within `libjulia-internal` is the same,
`0x88271`, and the address is page-aligned):

```
julia[66624]: segfault at 7451a95e3000 ip 00007451a86bd271 sp 00007ffd74fd7420 error 4 in libjulia-internal.so.1.12.6[7451a8635000+20e000]
julia[87664]: segfault at 7915e6ba3000 ip 00007915e5cbd271 sp 00007ffe087bf9f0 error 4 in libjulia-internal.so.1.12.6[7915e5c35000+20e000]
```

Cold versus warm kernel cache. `mgv_compare.jl 3` was run 4 times each with a new, empty
`XDG_CACHE_HOME` and 4 times with the filled one: 2 of the 4 cold runs crashed (the 1st and
4th), none of the 4 warm runs. The first `mgv_compare.jl` run in the second setup (empty cache)
crashed as well. But the association is not strict: the first runs of `mgv_copier_check.jl`
crashed in both setups (twice in a row in the first) although `test/gpu_test.jl` had run just
before with the same cache, and both silent exits were first runs of `mgv_out.jl` right after a
successful `mgv_noout.jl`.

One core dump was examined with gdb (`coredumpctl debug 87664`, measured): the faulting thread
is in `_jl_mutex_wait(..., safepoint=1)` under `jl_type_infer`, the main thread is in
`ijl_gc_safepoint`, and another Julia thread is inside
`SignalHandler(int, siginfo_t*, void*)` **from ZLUDA's `libcuda.so.1`**, entered as a signal
handler from the same `_jl_mutex_wait`. Julia stops its threads for garbage collection by making
a "safepoint" page unreadable and catching the resulting SIGSEGV in its own handler.

Hypothesis (not proven): ZLUDA's library contains LLVM (it compiles PTX to AMD code), and LLVM
installs its own crash handlers for SIGSEGV when it is first used for a compilation. From then
on a safepoint fault of a Julia thread reaches that handler instead of Julia's and is treated as
a crash. This fits the gdb observation, the absence of a Julia stack trace, the higher rate
with a cold cache (more compilation by ZLUDA), and the fact that single-threaded scripts were
not hit. It does not explain why warm-cache first runs of particular scripts crash. Not tried:
running mGV with one thread, or pre-filling the cache with ZLUDA's `zluda_precompile` tool.

Practical consequence: rerun a crashed run; a second attempt succeeded every time. Make scripts
`flush(stdout)` (or write to a terminal), otherwise a crash also loses the output before it.

### 5.3 Things that cost time and need not be rediscovered

- **Rejected kernel images land in `/tmp`** as `jl_*.cubin` (CUDA.jl keeps them "to attach to an
  issue") unless `TMPDIR` is set. 20 of them appeared before `TMPDIR` was set (half real NVIDIA cubins from before
  workaround 3, half PTX text); they were removed.
- **Preferences are ignored unless the package is named in the project** (section 2.3, step 4).
  The symptom is only that nothing changes.
- **The ZLUDA directory on `LD_LIBRARY_PATH`** replaces CUDA.jl's cuBLAS and cuSPARSE by ZLUDA's.
- **The default Julia environment leaks into the test** through `Base.find_package` in
  `test/gpu_test.jl` (section 2.3, step 3).
- **Dead ends for workaround 7:** appending an unreachable `ret;` after `trap;` (still panics);
  replacing `trap;` by `ret;` while keeping the stock exception runtime (parser errors, then
  code 999); also removing `exit;` (code 999 without message). Reducing the runtime in Julia was
  what worked, together with the textual fixes.
- **Code 999 carries no information.** ZLUDA returns `CUDA_ERROR_UNKNOWN` for any module it
  cannot compile after parsing, and the tracing layer has nothing to add. Bisecting the PTX file
  with `loadptx.jl` is the only way found.
- **With the tracing layer and the stock `name(::CuError)`**, the process segfaults inside the
  tracing library when CUDA.jl calls `cuGetErrorName` (seen twice, before workaround 6).
- **Process aborts instead of errors.** A panic in ZLUDA's PTX compiler kills the Julia process
  (SIGABRT), including precompilation workers; the message is on standard error, before a long
  Rust and Julia stack dump.
- **CUDA.jl 6.x is split into packages.** The code to patch is in `CUDACore`, not `CUDA`; the
  driver selection logic is in `CUDA_Driver_jll`'s generated wrapper
  (`src/wrappers/x86_64-linux-gnu-tegra+none.jl`), and the toolkit selection in
  `CUDA_Runtime_jll/.pkg/platform_augmentation.jl`.
- **`KernelAbstractions.copyto!` is asynchronous regardless of pinning** in CUDA.jl: it calls
  `cuMemcpyDtoHAsync` on the task's stream for any contiguous host array. Whether the call blocks
  is decided by the driver (section 4.3). Plain `copyto!(::Array, ::CuArray)` synchronizes first.
- **The scratch script that this experiment was meant to adapt** (an AMDGPU version of the
  timing test, `ka_async_copy.jl`) no longer existed; `step2b.jl` was written new and is not
  guaranteed to measure in exactly the same way as the AMDGPU numbers recorded elsewhere.

## 6. What a result under ZLUDA says about NVIDIA hardware

It says something about:

- **mGV's own CUDA code path**: backend registration, `CUDAExt`, the adaptation of the model to
  `CuArray`s, all kernels compiling for NVPTX and giving results in line with another GPU backend.
- **The copier's logic**: the sequence of stream and event calls is accepted, and in the
  stand-alone test it orders the copy after the kernel correctly, with pinned and with pageable
  memory. A logic error such as a missing wait or a copy on the wrong stream would probably show
  under ZLUDA as well (stale data in section 4.3 or unequal output in section 4.5).

It says little or nothing about:

- **Anything the patch bypasses.** On NVIDIA hardware CUDA.jl uses NVIDIA's `ptxas`, JIT
  options, the full exception runtime, context ids and the non-blocking device synchronize; none
  of that ran here.
- **Pinning.** `KernelAbstractions.pagelock!`/`CUDA.pin` on the output blocks is exactly what
  makes the copier asynchronous, and it could not be executed. That mGV's blocks pin correctly,
  are unpinned when freed (the copier relies on `CUDA.pin` doing that itself), and that the
  model really continues during the copies is untested.
- **Timing.** Whether and how long calls block is decided by ZLUDA's mapping onto HIP. The
  notes on mGV's AMDGPU copier state that in HIP `hipMemcpyWithStream` blocks and only
  `hipMemcpyAsync` does not; that was not verified here. The agreement with NVIDIA's documented behaviour in section 4.3 is
  encouraging but is not a measurement of NVIDIA's driver.
- **Numerics**, which come from ZLUDA's translation of PTX to AMD code (section 5.1 is ZLUDA's
  error), and **stability** (section 5.2 is specific to ZLUDA with Julia).
- **Performance**, in either direction.

A pass under ZLUDA is therefore weak evidence that the same code works on NVIDIA hardware, and a
failure under ZLUDA has to be traced to its cause before it counts against mGV: every failure
met in this experiment was a gap in ZLUDA.

**Open suggestion for mGV** (not implemented): `pin_memory!` in `src/backend_setup.jl` treats
`missing` from `KernelAbstractions.pagelock!` as "this backend cannot pin", but CUDA.jl's
`pagelock!` throws when registering fails. `pin_memory!` could catch that for CUDA, warn once
that output copies will block, and return `false`, so that a driver or a memory limit that
refuses to pin degrades the run instead of stopping it at start-up. On ZLUDA this would replace
workaround 8.

## 7. What the experiment left on the machine, and cleaning up

- **Work directories.** Everything else was under a session scratch directory in `/tmp`
  (`$ZLUDA_WORK` and equivalents, 175 to 190 MB per setup) and can simply be deleted:
  `rm -rf "$ZLUDA_WORK"`. The repository gained only this document, the patch and
  `docs/zluda-scripts/`, all untracked.
- **`~/.julia/artifacts`** (installed by Julia's package manager):
  - `07c2acc31b02e6cef67307574f4496b099692141`: CUDA runtime 13.0 (1.0 GiB download, 1.4 GB on
    disk), created 20:08.
  - `d089f86d72edd00351251e0d663c78989a317c46`: CUDA compiler tools (249.4 MiB download, 436 MB
    on disk), created 20:07.
  - Two more CUDA artifacts were created at 20:06 the same evening,
    `b7bddf2042616bb5a8ee0a05b7eb971f5ddeedfe` (runtime with `libcudart.so.13.4.92`, 1.4 GB) and
    `4e86ab4ae153ad3b9efeb9da49490f64da758c82` (compiler tools, 657 MB). Whether the first
    `Pkg.add` of this experiment or another Julia process installed them was not established.
  - Removal: delete the Julia environments that used them and let `Pkg.gc()` collect them
    (it only removes artifacts that have been unused for a while), or delete the directories by
    hand. Not done here, and not tested.
- **`~/.julia/compiled/v1.12`**: additional precompile caches for CUDACore, CUDA and its
  sub-packages, the mGV copies and their extensions (up to 464 MB of files were written there
  that evening, by this experiment and by other work). Julia prunes old cache files itself.
- **`~/.julia/packages`**: nothing was modified. Package versions that the environments needed
  and that were not yet installed may have been added by Pkg (CUDA 6.5.1 and AMDGPU 2.11.0 were
  already there).
- **Core dumps.** The machine's `core_pattern` hands crashes to `systemd-coredump`. This
  experiment produced 17 entries (`coredumpctl list --since "2026-11-13 20:00"`): 7 SIGSEGV from
  section 5.2, 2 SIGSEGV from the tracing library, and 8 SIGABRT from ZLUDA panics (7 while the
  workarounds were developed, 1 from a later test of `loadptx.jl` on an unfixed PTX file). At
  the end 7 core files were stored, 3.7 GB in `/var/lib/systemd/coredump/core.julia.1000.*`;
  systemd-coredump had already dropped older ones by itself (11 files and 4.5 GB half an hour
  earlier). No age-based clean-up rule was found on this machine (`/etc/systemd/coredump.conf`
  is all defaults). Removing the files needs root:
  `sudo rm /var/lib/systemd/coredump/core.julia.1000.*`.
- **`/tmp/jl_*.cubin`**: removed (section 5.3).
- **No system-wide installation, no change to shell profiles, no background processes.**
