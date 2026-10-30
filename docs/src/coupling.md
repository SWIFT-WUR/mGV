# Model coupling

mGV implements the [Basic Model Interface](https://bmi.csdms.org/) (BMI), allowing it to
be coupled to other models. The variables exposed through the BMI are set in the `[API]`
section of the configuration file:

```toml
[API]
variables = [
    "land_subsurface__runoff",
    "channel_water__volume_flow_rate",
]
```

### Grid

The model only holds the active (land) cells of the input grid (see
[Model input data](input_data.md)), so the BMI grids are `"unstructured"`: a variable's
values are a vector with one value per active cell. The grid nodes are the cell centers,
and all values are located on the nodes (`get_var_location` returns `"node"`); the grids
have no edges or faces. `get_grid_x` and `get_grid_y` give the longitude and latitude of
each cell. The cells are in the order of the input grid, with
longitude varying fastest.

Per-layer and per-band variables (names ending in `_layer_N` or `_band_N`) have one value
per cell as well. Variables per vegetation type hold the values of all cells for the first
vegetation type, then for the second, and so on. Their grid has a node for each cell and
vegetation type: `get_grid_z` gives the vegetation type (1 to `nveg`) of each node.

### Units

Values are exchanged in mGV's native units, as reported by `get_var_units`. These are not
necessarily SI units: for example, runoff is in mm day⁻¹ and temperatures are in °C. The
coupling framework is responsible for converting them to the units of the other model.

## Coupling with a groundwater model (MODFLOW)

The VIC-WUR coupling with MODFLOW exchanges three variables. Their mGV counterparts are:

| VIC-WUR                              | mGV (BMI name)                                       |
| ------------------------------------ | -----------------------------------------------------|
| `OUT_DISCHARGE`                      | `channel_water__volume_flow_rate`                    |
| `OUT_GWRECHARGE`                     | `land_subsurface__runoff`                            |

### River discharge

River discharge (m³ s⁻¹) is available as `channel_water__volume_flow_rate`, which requires
`enable_routing = true`. MODFLOW only needs the river stage: its RIV package derives the
river head from the discharge using its own channel shape parameterization, and uses
that head to compute the exchange between river and aquifer.

### Groundwater recharge

mGV computes subsurface runoff (baseflow from the bottom soil layer) in the land module
and by default adds it to the total runoff (`land__runoff~total`), which feeds the river
routing. When a groundwater model handles the subsurface flow, this baseflow would be
counted twice. Disable it with:

```toml
[API]
# set the following to `true` to manually solve subsurface routing via BMI
#  with a groundwater model (e.g. MODFLOW).
external_groundwater_routing = true
```

With this setting the routing only receives the surface runoff, and the subsurface
runoff (`land_subsurface__runoff`) should be passed to the groundwater model as
recharge.

!!! note
    With `external_groundwater_routing = true` the `total_runoff` output also excludes
    the subsurface runoff.

### Groundwater abstraction

Groundwater abstraction is not implemented in mGV. Water demand and supply belong in a
human water use model, which can be added as an additional model in the coupling framework.
