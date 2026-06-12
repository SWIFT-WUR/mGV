julia --project=. run.jl configs/mekong_config.toml
julia --project=. run.jl configs/indus_config.toml

cd ./validations
python plot_dashboards_mekong_indus.py
