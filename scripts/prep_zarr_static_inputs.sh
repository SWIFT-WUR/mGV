#!/bin/bash
#SBATCH --job-name=mGV_prep_zarr
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/nobackup/WUR/ESG/datad002/mGV/benchmark_logs/prep_zarr.%j.out
#SBATCH --error=/lustre/nobackup/WUR/ESG/datad002/mGV/benchmark_logs/prep_zarr.%j.err

set -euo pipefail

export JULIA_DEPOT_PATH="/lustre/nobackup/WUR/ESG/datad002/mGV/.julia_depot_1.12:"
JULIA=/lustre/nobackup/WUR/ESG/datad002/.juliaup/bin/julia

cd /lustre/nobackup/WUR/ESG/datad002/mGV

SRC_PARAMS=/lustre/nobackup/WUR/ESG/liu297/vic_global/00pre_analysis/downscaling5min/07TestRun/param/vic_global_5min_params_combined.nc
PARAMS_F32=input_data/global/vic_global_5min_params_combined_f32.nc
PARAMS_ZARR=input_data/global/vic_global_5min_params_combined_f32.zarr

ROUTING_F32=input_data/global/routing/vic_global_5min_routing_param_wbt_f32.nc
ROUTING_ZARR=input_data/global/routing/vic_global_5min_routing_param_wbt_f32.zarr

DOMAIN_F32=input_data/global/vic_global_5min_domain_f32.nc
DOMAIN_ZARR=input_data/global/vic_global_5min_domain_f32.zarr

echo "=== 1/4: params NetCDF to Float32 NetCDF ==="
if [ ! -f "$PARAMS_F32" ]; then
    $JULIA --project=. scripts/convert_to_f32.jl "$SRC_PARAMS" "$PARAMS_F32"
else
    echo "Skipping existing: $PARAMS_F32"
fi

echo "=== 2/4: params Float32 NetCDF to Zarr ==="
$JULIA --project=. scripts/convert_params_to_zarr.jl "$PARAMS_F32" "$PARAMS_ZARR"

echo "=== 3/4: routing Float32 NetCDF to Zarr ==="
$JULIA --project=. scripts/convert_routing_to_zarr.jl "$ROUTING_F32" "$ROUTING_ZARR"

echo "=== 4/4: domain Float32 NetCDF to Zarr ==="
$JULIA --project=. scripts/convert_domain_to_zarr.jl "$DOMAIN_F32" "$DOMAIN_ZARR"

echo "All conversions finished."
