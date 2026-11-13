# Model output

mGV writes the output of each simulated year to the directory specified by `dir` in the `[output]`
section of the config file: one `.nc` file for NetCDF, or one `.zarr` folder for Zarr.
Output variables are written for each simulated time step (daily by default) and grid cell. Soil
moisture is also output separately for each soil layer.

## Variables

### Choosing output variables

By default all variables listed below are written. To write only some of them, list them in the config, for example:

```toml
[output]
variables = ["discharge", "total_runoff"]
```

- Leave out the `variables` line to write all of them.
- `variables = []` writes nothing and creates no output file.
- An unknown name stops the model at startup, with an error listing the valid names.

Writing fewer variables generally makes the model faster, because less data is copied from the GPU and
written to disk. The output files also take up less disk space.

Snow variables are not written when `enable_snow = false`, and discharge and travel time are not written
when `enable_routing = false`. If you list such a variable anyway, the model warns that it is skipped.

### Available output variables

```@eval
using mGV, Markdown
rows = [
    "| `$(meta.output_name)` | $(meta.description) | `$(mGV.to_string(meta.unit))` | `$standard_name` |"
        for (standard_name, meta) in mGV.output_variables()
]
header = ["| Name | Description | Unit | Standard name |", "|------|-------------|------|---------------|"]
Markdown.parse(join([header; rows], "\n"))
```

The weighted sums and the snow variables combine the vegetation types and snow
bands in each grid cell into one value: each value is weighted by the fraction of the cell it covers, which
gives the cell average. Transpiration, canopy evaporation and canopy water storage are also weighted by
canopy coverage, so only the canopy-covered part of each vegetation type contributes. Snow albedo and snow
surface temperature are averaged over the snow-covered parts of the cell only.

## Formats

Set the format with `format` in the `[output]` section:

| `format` | Layout | Notes |
|----------|--------|-------|
| `"zarr"` (default) | Gridded: `(time, lat, lon)`, or `(layer, time, lat, lon)` | Cells outside the model domain are NaN |
| `"zarr_landonly"` | Only the model's (land) cells: `(time, cell)`, or `(layer, time, cell)` | Fastest to write, and smallest on disk |
| `"netcdf"` | Gridded, like `"zarr"` | Slowest: written by a single thread |

Zarr output is written as Zarr v3. The land-only format is the same as the
[land-only forcing](performance.md#Optimize-the-forcing-input). Besides the `lat` and `lon`
coordinates of the grid, the store holds the 0-based grid indices of every cell, `lat_index` and
`lon_index`, so the output can be put back on the grid. For example, in Python:

```python
import numpy as np
import xarray as xr

# Packed output (the default) needs zarr-python >= 3.4 and the cast-value-rs package, see Packing
ds = xr.open_zarr("outputfile_1979.zarr", zarr_format=3, consolidated=False)
grid = np.full((ds.sizes["lat"], ds.sizes["lon"]), np.nan, dtype="float32")
grid[ds.lat_index.values, ds.lon_index.values] = ds.total_runoff.isel(time=0).values
```

### Writing in blocks

The model keeps the output of several time steps in memory, and writes them to disk together while it
computes the next time steps. `write_interval` (default 8) sets the number of time steps in a block:

```toml
[output]
write_interval = 8
```

This is also the size of the time chunks of the output. Larger blocks are written more efficiently,
but take more memory. The model keeps three blocks in (host) memory: while the writer writes one,
the model fills the next. Each block takes
`write_interval × number of land cells × bytes per cell and time step`, where every output field
(a variable, or one soil layer of a layered variable) takes 2 bytes when it is packed and 4 bytes
when it is not (see [Packing](@ref)). The model prints the size of the blocks at startup. For example, all variables of
a global 5′ run (about 2.2 million land cells, 22 fields including the soil layers) take about
1.5 GB per block of 8 time steps without packing (4.6 GB for the three blocks), and about 0.9 GB
with packing (2.6 GB), as three of the fields are never packed.

On a GPU, the model also keeps the output of one time step on the device (2 or 4 bytes per field and
land cell, as above), which it copies to the block in host memory while it computes the next time step.
The blocks in host memory are pinned (page-locked) when the GPU backend supports it.

Output appears on disk one block at a time; the last, partial, block of a run is written when the
model is finalized. When the model finishes, it prints the time the writer spent writing.

### Packing

By default, most output variables are packed: they are stored as 16-bit integers instead of 32-bit
floats, which halves the size of the output in memory and makes it smaller on disk. Set `packing`
in the `[output]` section to store every variable as Float32 instead:

```toml
[output]
packing = false  # default: true
```

Packing is lossy. A packed variable is stored with a fixed precision (the step below), and values
outside its range are clipped to the edge of the range. The model warns when it clips values, once
per variable and output file, with the first time step where it happened. Readers decode the
packed values when they read them, so packed output has the same variables as unpacked output, and
NaN (no data) stays NaN. The variables that are not packed (step "-") are always stored as Float32.

```@eval
using mGV, Markdown
fmt(x) = string(round(x; sigdigits = 6))
rows = map(collect(values(mGV.output_variables()))) do meta
    p = meta.packing
    isnothing(p) && return "| `$(meta.output_name)` | - | - | `$(mGV.to_string(meta.unit))` |"
    lo, hi = p.offset - 32767 / p.scale, p.offset + 32767 / p.scale
    return "| `$(meta.output_name)` | $(fmt(1 / p.scale)) | $(fmt(lo)) to $(fmt(hi)) | `$(mGV.to_string(meta.unit))` |"
end
header = ["| Name | Step | Range | Unit |", "|------|------|-------|------|"]
Markdown.parse(join([header; rows], "\n"))
```

For example, precipitation and runoff are clipped above about 1295 mm/day, and the evaporation
variables (such as `total_et`) below about -2.77 mm/day. Use `packing = false` if your run can
exceed a range, or if you need full Float32 precision.

The packing follows the Zarr v3 `scale_offset` and `cast_value` codecs
([zarr-extensions](https://github.com/zarr-developers/zarr-extensions)): the stored value is
`round((value - offset) × scale)` as Int16, and -32768 stands for NaN. To read packed Zarr output, you
need a Zarr v3 reader that supports both codecs: in Python, zarr-python 3.4 or later with the
`cast-value-rs` package installed; in Julia, Zarr.jl with
[ZarrScaleCast.jl](https://github.com/SWIFT-WUR/ZarrScaleCast.jl) loaded. Without them, the packed
variables cannot be decoded. NetCDF output is packed with the
standard CF attributes `scale_factor` and `add_offset` (and `_FillValue = -32768`), which NetCDF
readers such as xarray and NCDatasets.jl apply when they read the data.
