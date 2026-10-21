# Model output

mGV writes the output of each simulated year to the directory specified by `dir` in the `[output]`
section of the config file: one `.nc` file for NetCDF, or one `.zarr` folder for Zarr (`format`).
Output variables are written for each simulated time step (daily by default) and grid cell. Soil
moisture is also reported separately for each soil layer.

## Choosing output variables

By default all variables listed below are written. To write only some of them, list them in the config:

```toml
[output]
variables = ["discharge_output", "total_runoff_output"]
```

- Leave out the `variables` line to write all of them.
- `variables = []` writes nothing and creates no output file.
- An unknown name stops the model at startup, with an error listing the valid names.

Writing fewer variables generally makes the model faster, because less data is copied from the GPU and
written to disk. The output files also take up less disk space.

Snow variables are not written when `enable_snow = false`, and discharge and travel time are not written
when `enable_routing = false`. If you list such a variable anyway, the model warns that it is skipped.

## Available output variables

| Name | Description | Unit |
|------|-------------|------|
| `tsurf_output` | Surface temperature, blended with the snow surface temperature according to snow coverage | °C |
| `tair_output` | Air temperature (from the forcing) | °C |
| `precipitation_output` | Precipitation (from the forcing) | mm/day |
| `total_et_output` | Total evapotranspiration | mm/day |
| `surface_runoff_output` | Surface runoff | mm/day |
| `total_runoff_output` | Total runoff (surface and subsurface) | mm/day |
| `discharge_output` | River discharge (routing) | m³/s |
| `travel_time_output` | Time for water to flow through the grid cell (routing) | s |
| `potential_evaporation_summed_output` | Potential evaporation | mm/day |
| `net_radiation_summed_output` | Net radiation | W/m² |
| `transpiration_summed_output` | Transpiration | mm/day |
| `canopy_evaporation_summed_output` | Evaporation of water held on the vegetation canopy | mm/day |
| `water_storage_summed_output` | Water held on the vegetation canopy | mm |
| `swe_summed_output` | Snow water equivalent | mm |
| `snow_albedo_output` | Snow albedo, only where there is snow | fraction (0–1) |
| `snow_surf_temp_output` | Snow surface temperature, only where there is snow | °C |
| `snow_coverage_output` | Fraction of the grid cell covered by snow | fraction (0–1) |
| `snow_melt_output` | Snowmelt | mm/day |
| `soil_evaporation_output` | Evaporation from bare soil | mm/day |
| `soil_moisture_output` | Soil moisture, per soil layer | mm |

Variables ending in `_summed_output`, and the other snow variables, combine the vegetation types and snow
bands in each grid cell into one value: each value is weighted by the fraction of the cell it covers, which
gives the cell average. Transpiration, canopy evaporation and canopy water storage are also weighted by
canopy coverage, so only the canopy-covered part of each vegetation type contributes. Snow albedo and snow
surface temperature are averaged over the snow-covered parts of the cell only.
