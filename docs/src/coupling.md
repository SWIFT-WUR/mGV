# Model coupling

## BMI implementation

BMI is a set of standard controls and query functions that can be added to a numerical model
to make it easier to learn and to couple it with other softwares. 
For more information and the spec list see: [BMI specifications](https://bmi.csdms.io/en/stable/).
mGV, based on the Julia programming language, uses the following [BMI for Julia](https://github.com/Deltares/BasicModelInterface.jl) specifications based on BMI 2.0 version. 

The BMI implementation in mGV uses rectilinear grids, and is based on three different coordinates:
longitude, latitude and vegetation types. 
Variables depending on soil layers as well as on snow bands are exposed as two-dimensional grids.
To set of get these variables the index of the layer/band is required by adding `_layer_N` / `_band_N` at the end of the variable name.

Currently the variables exposed through the BMI are model state and output variables.
Their names is based on the [CSDMS Standard Names](https://csdms.colorado.edu/wiki/CSDMS_Standard_Names), and the variables exposed through the BMI are set in the `[API]`
section of the configuration file:

```toml
[API]
variables = [
    "land_subsurface__runoff",
    "channel_water__volume_flow_rate",
]
```

Variables having a dependence on soild layers or snow band are expanded adding the corresponding layer or band at the end of the name. For example, adding  `soil__thermal_conductivity` to the ``` [API] ``` section of the config file, will automatically add `soil__thermal_conductivity_layer_1`, ..., `soil__thermal_conductivity_layer_N` (dependingly from the actual number of soil layers) to the list of the output variables. 

### Exposed variables
The full list of the variables and parameters exposed or accessible throught BMI is contained in standard_name.jl file.
Currently there are three `OrderedDict`: `standard_name_map` contains the list of variables beside those related to routing; `standard_routing_map` contains the routing variables; `standard_param_map` contains some parameters but is not actually used (for now).
Each entry of the dictionaries corresponds to a variable/parameter with some metadata like the corresponding units of measurement, a brief description and a tag. 

### Units

Values are exchanged in mGV's native units, as reported by `get_var_units`. These are not
necessarily SI units: for example, runoff is in mm day⁻¹ and temperatures are in °C. The
coupling framework is responsible for converting them to the units of the other model.

### Remote BMI controller

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