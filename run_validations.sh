set -e

mkdir -p output_data/mekong output_data/indus

julia --project=. -e 'using mGV; mGV.run()' -- configs/mekong_config.toml
# Indus input data is not yet in the repo
# julia --project=. -e 'using mGV; mGV.run()' -- configs/indus_config.toml

python3 validations/plot_dashboard.py mekong
