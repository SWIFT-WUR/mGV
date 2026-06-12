julia --project=. run.jl configs/mekong_config.toml 1979 1980
#julia --project=. run.jl configs/indus_config.toml

cd ./validations
#python plot_dashboards_mekong_indus.py
python plot_dashboards_mekong.py
