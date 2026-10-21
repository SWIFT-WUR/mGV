# Model output

mGV writes one output file per simulated year to the `dir` set in the `[output]` section of the
config file, as NetCDF or Zarr (`format`). Every variable holds one value per grid cell per day.

## Choosing output variables

By default all variables listed below are written. To write only some of them, list them in the config:

```toml
[output]
variables = ["discharge_output", "total_runoff_output"]
```

- Leave `variables` out to write all of them.
- `variables = []` writes nothing and creates no output file, e.g. when the model is used through BMI.
- An unknown name stops the model at startup, with an error listing the valid names.

Writing fewer variables also makes the model faster, because less data is copied from the GPU and written to disk.

Snow variables are not written when `enable_snow = false`, and discharge and travel time are not written
when `enable_routing = false`. If you list such a variable anyway, the model warns that it is skipped.

## Available output variables

| Name | Description | Unit |
|------|-------------|------|
| `tsurf_output` | Surface temperature, blended with the snow surface temperature where there is snow | °C |
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
| `snow_albedo_output` | Snow albedo, only where there is snow | - |
| `snow_surf_temp_output` | Snow surface temperature, only where there is snow | °C |
| `snow_coverage_output` | Fraction of the grid cell covered by snow | - |
| `snow_melt_output` | Snowmelt | mm/day |
| `soil_evaporation_output` | Evaporation from bare soil | mm/day |
| `soil_moisture_output` | Soil moisture, per soil layer | mm |

Variables ending in `_summed_output`, and the snow variables, are averages over the vegetation types and
snow bands in each grid cell, weighted by their area.
