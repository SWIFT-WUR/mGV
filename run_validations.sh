julia --project=. run.jl mekong 1979 1980
julia --project=. run.jl indus 1979 1980

cd ./validations
python plot_dashboards_mekong_indus.py
