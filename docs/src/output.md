# Model output

mGV writes the output of each simulated year to the directory specified by `dir` in the `[output]`
section of the config file: one `.nc` file for NetCDF, or one `.zarr` folder for Zarr.
Output variables are written for each simulated time step (daily by default) and grid cell. Soil
moisture is also output separately for each soil layer.

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
