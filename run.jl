# Load package definitions
include("src/packages.jl")

# Utilities
include("src/utils/runtime_utils.jl")
include("src/utils/diagnostics.jl") 

# Setup backend; device (GPU/CPU) type and float type
include("src/backend_setup.jl")

# Load user-modifiable configuration
include("config/user_config.jl")

# Constants
include("src/constants/sim_constants.jl")
include("src/constants/physical_constants.jl")

# I/O helpers
include("src/io/io_helpers.jl")
include("src/io/io_writer.jl")
include("src/io/async_writer.jl")
include("src/io/parameter_reader.jl")

# Initialization routines
include("src/init/init_calc.jl")

# Physics modules
include("src/snow/snow.jl")
include("src/physics/physics.jl")
include("src/physics/evapotranspiration.jl")
include("src/physics/groundwater.jl")
include("src/physics/runoff.jl")
include("src/physics/temperature.jl")

# Routing module
include("src/routing/routing.jl")
include("src/routing/init_routing.jl")


# Main program entry point
include("src/main.jl")