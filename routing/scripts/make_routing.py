"""Build a 5 arcmin D8 routing parameter file from the HydroSHEDS 30s DEM with pyflwdir.

Output follows the layout of the VIC routing files used by mGV (downstream_id,
downstream, cell_dist, cell_area, accumulation), plus elevation and basin ids.

usage: make_routing.py dem.tif out.nc [west south east north]
"""
import sys
import time

import netCDF4
import numpy as np
import pyflwdir
import rasterio
from rasterio.windows import from_bounds

import os
PRE = int(os.environ.get("PRE", "1"))  # aggregate the 30s DEM by PRE x PRE first (memory)
SCALE = 10 // PRE  # -> 5 arcmin

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

dem_path, out_path = sys.argv[1], sys.argv[2]
with rasterio.open(dem_path) as src:
    if len(sys.argv) > 3:
        w, s, e, n = map(float, sys.argv[3:7])
        win = from_bounds(w, s, e, n, src.transform).round_offsets().round_lengths()
    else:
        win = None
    dem = src.read(1, window=win)
    transform = src.window_transform(win) if win is not None else src.transform
    nodata = src.nodata
log("dem", dem.shape)
# crop to a multiple of 10 (5 arcmin)
ny, nx = (dem.shape[0] // 10) * 10, (dem.shape[1] // 10) * 10
dem = dem[:ny, :nx]
if PRE > 1:
    # mean of the valid 30s cells in each PRE x PRE block
    v = (dem != nodata).reshape(ny // PRE, PRE, nx // PRE, PRE)
    s_ = np.where(v, dem.reshape(v.shape), 0).astype(np.float32).sum(axis=(1, 3))
    c_ = v.sum(axis=(1, 3))
    del v
    dem = np.where(c_ > 0, s_ / np.maximum(c_, 1), -9999.0).astype(np.float32)
    del s_, c_
    transform = transform * transform.scale(PRE, PRE)
    log("aggregated dem", dem.shape)
else:
    dem = dem.astype(np.float32)
    dem[dem == nodata] = -9999.0
flw = pyflwdir.from_dem(dem, nodata=-9999.0, transform=transform, latlon=True, outlets="edge")
log("flwdir 30s done")
uparea = flw.upstream_area(unit="km2")
log("uparea 30s done")
flw_lr, idxs_out = flw.upscale(SCALE, method="ihu", uparea=uparea)
log("upscale done", flw_lr.shape)
del uparea

# Elevation at the sub-grid outlet pixel (gives consistent downstream slopes)
valid = flw_lr.mask if flw_lr.mask is not None else (flw_lr.idxs_ds >= 0).reshape(flw_lr.shape)
valid = np.asarray(valid).reshape(flw_lr.shape)
io = idxs_out.ravel()
elev = np.full(io.size, np.nan, dtype=np.float32)
ok = io >= 0
elev[ok] = dem.ravel()[io[ok]]
elev = elev.reshape(flw_lr.shape)
del dem, flw

shape = flw_lr.shape
idxs_ds = flw_lr.idxs_ds.reshape(shape)
lin = np.arange(shape[0] * shape[1]).reshape(shape)
valid = valid & (idxs_ds >= 0)
cell_id = np.where(valid, lin + 1, -1).astype(np.int32)
is_pit = idxs_ds == lin
down = np.where(valid & ~is_pit, idxs_ds + 1, -1).astype(np.int32)
tr = flw_lr.transform
lon = tr.c + tr.a * (np.arange(shape[1]) + 0.5)
lat = tr.f + tr.e * (np.arange(shape[0]) + 0.5)
r_, c_ = np.divmod(np.where(idxs_ds >= 0, idxs_ds, lin), shape[1])
la1, lo1 = np.radians(lat)[:, None], np.radians(lon)[None, :]
la2, lo2 = np.radians(lat[r_]), np.radians(lon[c_])
hav = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
dist = 2 * 6371007.2 * np.arcsin(np.sqrt(hav))
dist = np.where(valid, np.where(is_pit, np.sqrt(flw_lr.area.reshape(shape)), dist), np.nan).astype(np.float32)
area = np.where(valid, flw_lr.area.reshape(shape), np.nan).astype(np.float32)
acc = np.where(valid, flw_lr.upstream_area(unit="m2").reshape(shape), np.nan).astype(np.float32)
basins = np.where(valid, flw_lr.basins(), 0).astype(np.int32)
log("valid cells", int(valid.sum()), "basins", int(len(np.unique(basins)) - 1))

# north-up rows -> south-to-north latitude, like the existing input files
flip = lambda a: a[::-1, :]
# ids encode north-up linear indices; flipping rows keeps them consistent as labels
with netCDF4.Dataset(out_path, "w") as ds:
    ds.createDimension("lat", shape[0])
    ds.createDimension("lon", shape[1])
    ds.createVariable("lat", "f8", ("lat",))[:] = lat[::-1]
    ds.createVariable("lon", "f8", ("lon",))[:] = lon
    for name, arr, fill in [
        ("downstream_id", cell_id, -1), ("downstream", down, -1), ("basins", basins, 0),
        ("cell_dist", dist, np.nan), ("cell_area", area, np.nan),
        ("accumulation", acc, np.nan), ("elevation", elev, np.nan),
    ]:
        v = ds.createVariable(name, arr.dtype, ("lat", "lon"), fill_value=fill, zlib=True, complevel=1)
        v[:] = flip(arr)
log("written", out_path)
