# Model output

mGV writes the output of each simulated year to the directory specified by `dir` in the `[output]`
section of the config file: one `.nc` file for NetCDF, or one `.zarr` folder for Zarr.
Output variables are written for each simulated time step (daily by default) and grid cell. Soil
moisture is also output separately for each soil layer.

## Output formats

Set the format with `format` in the `[output]` section:

| `format` | Layout | Notes |
|----------|--------|-------|
| `"zarr"` (default) | Gridded: `(time, lat, lon)`, or `(layer, time, lat, lon)` | Cells outside the model domain are NaN |
| `"zarr_landonly"` | Only the model's (land) cells: `(time, cell)`, or `(layer, time, cell)` | Fastest to write, and smallest on disk |
| `"netcdf"` | Gridded, like `"zarr"` | Slowest: written by a single thread |

The land-only format is the same as the [land-only forcing](performance.md#Optimize-the-forcing-input).
Besides the `lat` and `lon` coordinates of the grid, the store holds the 0-based grid indices of
every cell, `lat_index` and `lon_index`, so the output can be put back on the grid. For example, in Python:

```python
import numpy as np
import xarray as xr

ds = xr.open_zarr("outputfile_1979.zarr")
grid = np.full((ds.sizes["lat"], ds.sizes["lon"]), np.nan, dtype="float32")
grid[ds.lat_index.values, ds.lon_index.values] = ds.total_runoff.isel(time=0).values
```

## Writing in blocks of time steps

The model keeps the output of several time steps in memory, and writes them to disk together while it
computes the next time steps. `write_interval` (default 8) sets the number of time steps in a block:

```toml
[output]
write_interval = 8
```

This is also the size of the time chunks of the output. Larger blocks are written more efficiently,
but take more memory: the model keeps two blocks, of
`write_interval × number of cells × number of output fields × 4 bytes` each. For example, all
variables of a global 5′ run (about 2.2 million land cells, 22 fields including the soil layers) take
about 1.5 GB per block of 8 time steps.

Output appears on disk one block at a time; the last, partial, block of a run is written when the
model is finalized. When the model finishes, it prints the time the writer spent writing.

## Choosing output variables

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

## Available output variables

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
