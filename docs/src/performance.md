# Getting optimal performance from mGV

There are a few things to take into account when you want to run
mGV as fast as possible. Some of these steps can have a major
impact while others are more incremental.

## Run the model on GPU

[The model has GPU acceleration](gpu.md). Using this (if you have a
GPU available locally or on HPC) will be the largest speed up.
The model currently only supports a single GPU at a time,
however, the model does not need a lot of VRAM (2 - 8 GB for a
global 5′ grid, depending on the configuration and parameter file).

Do note that this leads to bottlenecks elsewhere in the code.
Use the other performance tips from this page to make optimal use of
the GPU resources.

## Use Zarr as input & output files

Zarr is a lot faster than netCDF as input and output files. This
mostly matters for dynamic input (forcing) which needs to be loaded
for every time step of the model.

To configure the model to output in Zarr, modify your config by
settting `format = "zarr"` under `[output]`.

It's important to make sure that the forcing variables all have the
same time chunks, otherwise chunks might get decompressed more than
once. If you use a single Zarr store for all forcing input this is
usually the case.

## Launch julia with multiple threads

Some parts of the model are capable of running in parallel.
This requires multiple threads. Therefore make sure you launch
Julia (or the model directly) with >1 thread, e.g. `julia -t auto`

More specifically:
- The Zarr input data is prefetched, so the data loads while the
  model is running the next time steps.
- Forcing variables are read in parallel.
- Output writing is on its own thread, and variables are written
  in parallel.
- If you don't have GPU available, the model's computation is
  multithreaded on CPU.

## Have input & output on a fast disk

Depending on your setup (e.g. HPC), make sure the data is available
on a node-local scratch or SSD instead of a shared network filesystem.

## Write fewer variables to file

Reducing [the variables written to output file](output.md) will help
speeding up output writing, as well as reduce data transfers on the
GPU. Only write the variables to file you really need.

## Optimize the forcing input

The mGV github repository contains a script (`scripts/convert_forcing_landonly.jl`)
which converts the input data to a format more streamlined for
use in the model. Before running the model on a GPU (node),
convert the Zarr forcing data to the optimized format for faster
reading.
