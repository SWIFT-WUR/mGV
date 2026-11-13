set -e

mkdir -p output_data/mekong output_data/indus

# Extra Julia options, e.g. JULIA_FLAGS=--check-bounds=yes (as in CI) to check the kernels'
# array accesses, which skip bounds checking by default (see `@kernel inbounds=true`)
JULIA_FLAGS=${JULIA_FLAGS:-}

julia --project=. $JULIA_FLAGS -e 'using mGV; mGV.run()' -- configs/mekong_config.toml
# Indus input data is not yet in the repo
# julia --project=. $JULIA_FLAGS -e 'using mGV; mGV.run()' -- configs/indus_config.toml

python3 validations/plot_dashboard.py mekong
