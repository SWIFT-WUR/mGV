#!/usr/bin/env python3
"""
4 clean comparison plots: VIC vs VIC-WUR-Julia, 1979
White background, no precipitation, no air temp, no snow physics.
Surface temperature uses proper per-dataset land masks.
"""
import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import warnings
import os
import sys
import csv as csv_mod


warnings.filterwarnings("ignore")
matplotlib.use('Agg')

VIC_C = "#C0392B"; MGV_C = "#2471A3"; FA = 0.10
SM_COLORS = ["#1A8A5A", "#E67E22", "#7D3C98"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#BBBBBB", "axes.labelcolor": "#222222",
    "xtick.color": "#444444", "ytick.color": "#444444", "text.color": "#222222",
    "grid.color": "#E5E5E5", "grid.linewidth": 0.7, "axes.grid": True,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.framealpha": 0.95, "legend.edgecolor": "#CCCCCC",
})

LEGEND_ELEMS = [
    Line2D([0],[0], color=VIC_C, lw=2.0, label="VIC (reference)"),
    Line2D([0],[0], color=MGV_C, lw=2.0, ls='--', label="VIC-WUR-Julia"),
]
DOY = np.arange(1, 366)

# ET variable spec (reused for both basins)
ET_VARS = [
    ("OUT_EVAP",       "total_et_output",                    "Total ET",           "mm d$^{-1}$"),
    ("OUT_TRANSP_VEG", "transpiration_summed_output",        "Transpiration",       "mm d$^{-1}$"),
    ("OUT_EVAP_CANOP", "canopy_evaporation_summed_output",   "Canopy Evaporation",  "mm d$^{-1}$"),
    ("OUT_EVAP_BARE",  "soil_evaporation_output",            "Soil Evaporation",    "mm d$^{-1}$"),
    ("OUT_PET",        "potential_evaporation_summed_output","Potential ET",         "mm d$^{-1}$"),
]

# ── CSV reference helpers ─────────────────────────────────────────────────────
VIC_CSV_VARS = [
    "OUT_SURF_TEMP", "OUT_R_NET", "OUT_EVAP", "OUT_TRANSP_VEG",
    "OUT_EVAP_CANOP", "OUT_EVAP_BARE", "OUT_PET",
    "OUT_RUNOFF", "OUT_BASEFLOW",
    "OUT_SOIL_MOIST_L1", "OUT_SOIL_MOIST_L2", "OUT_SOIL_MOIST_L3",
]

def preprocess_vic_to_csv(vic_nc_path, out_csv_path):
    """Extract spatially-averaged VIC timeseries from NetCDF and save as CSV."""
    print(f"  Pre-processing VIC reference -> {out_csv_path}")
    ds = nc.Dataset(vic_nc_path)
    mask = get_land_mask(ds, "OUT_SURF_TEMP")

    def _spatial_mean(raw):
        if raw.ndim == 3:
            T, nlat, nlon = raw.shape
            if mask is not None:
                m3 = np.broadcast_to(mask[np.newaxis], raw.shape).copy()
                raw = np.where(m3, raw, np.nan)
            return np.nanmean(raw.reshape(T, -1), axis=1)[:365]
        return raw.ravel()[:365]

    def _load(varname, layer=None):
        if varname not in ds.variables:
            return np.full(365, np.nan)
        raw = np.ma.filled(ds.variables[varname][:], np.nan).astype(float)
        raw[np.abs(raw) > 1e15] = np.nan
        sh = raw.shape
        if raw.ndim == 4 and layer is not None:
            if sh[1] <= 5 and sh[0] > 50:
                raw = raw[:, layer]
            elif sh[0] <= 5 and sh[1] > 50:
                raw = raw[layer]
        return _spatial_mean(raw)

    rows = []
    scalar_vars = [v for v in VIC_CSV_VARS if not v.startswith("OUT_SOIL_MOIST_L")]
    for i in range(365):
        row = {"doy": i + 1}
        for v in scalar_vars:
            ts = _load(v)
            row[v] = float(ts[i]) if i < len(ts) else float("nan")
        for l in range(3):
            ts = _load("OUT_SOIL_MOIST", layer=l)
            row[f"OUT_SOIL_MOIST_L{l+1}"] = float(ts[i]) if i < len(ts) else float("nan")
        rows.append(row)

    ds.close()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["doy"] + scalar_vars + ["OUT_SOIL_MOIST_L1", "OUT_SOIL_MOIST_L2", "OUT_SOIL_MOIST_L3"]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved reference CSV ({out_csv_path.stat().st_size / 1024:.1f} KB)")


class VicCsvDataset:
    """Thin wrapper around a VIC CSV reference file, mimicking the nc.Dataset interface
    just enough for the load_ts / load_sm helpers to work."""
    def __init__(self, path):
        self._data = {}  # varname -> np.array of shape (365, 1, 1)
        with open(path, newline="") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        for col in rows[0].keys():
            if col == "doy":
                continue
            arr = np.array([float(r[col]) for r in rows])  # (365,)
            self._data[col] = arr[:, np.newaxis, np.newaxis]  # (365, 1, 1)
        # Soil moisture: merge L1/L2/L3 -> (time, layer, lat, lon)
        layers = []
        for l in range(1, 4):
            key = f"OUT_SOIL_MOIST_L{l}"
            if key in self._data:
                layers.append(self._data.pop(key))
        if layers:
            self._data["OUT_SOIL_MOIST"] = np.stack(layers, axis=1)  # (365, 3, 1, 1)
        self.variables = self._data

    def close(self):
        pass


def open_vic(nc_path, csv_path, label):
    """Open VIC data: prefer the NetCDF, fall back to the pre-processed CSV.

    The CSV holds VIC's basin average over VIC's own land mask. mGV is averaged over
    mGV's mask, which is not the same set of cells, so CSV-based scores are NOT
    comparable -- on Indus the CSV path reported surface runoff NSE -1.04 where the
    shared-mask value is +1.00. Only the NetCDF path can build a shared mask.
    """
    if nc_path.exists():
        if not csv_path.exists():
            print(f"  Found NetCDF for {label}, building CSV cache...")
            preprocess_vic_to_csv(nc_path, csv_path)
        else:
            print(f"  Found NetCDF for {label} (CSV cache already present)")
        return nc.Dataset(nc_path)
    if csv_path.exists():
        print(
            f"WARNING: {label}: no VIC NetCDF, falling back to the pre-processed CSV.\n"
            f"         The CSV is averaged over VIC's land mask while mGV is averaged\n"
            f"         over its own, so NSE/PBIAS below are NOT a like-for-like\n"
            f"         comparison and may be badly misleading. Restore {nc_path.name}\n"
            f"         for a valid shared-mask comparison.",
            file=sys.stderr,
        )
        return VicCsvDataset(csv_path)
    print(f"WARNING: No VIC data found for {label} (neither NetCDF nor CSV)", file=sys.stderr)
    return None


# ── Data helpers ──────────────────────────────────────────────────────────────
def open_dataset(path, label):
    """Open a NetCDF file, returning None with a warning if not found."""
    if not os.path.isfile(path):
        print(f"WARNING: {label} file not found, plotting without it: {path}", file=sys.stderr)
        return None
    return nc.Dataset(path)

def get_land_mask(ds, ref_var):
    """2-D bool mask: pixels with at least some valid, non-zero data."""
    if ds is None or ref_var not in ds.variables:
        return None
    raw = np.ma.filled(ds.variables[ref_var][:], np.nan).astype(float)
    raw[np.abs(raw) > 1e15] = np.nan
    if raw.ndim == 4:
        raw = raw[0]
    any_finite  = np.any(np.isfinite(raw), axis=0)
    not_all_zero = np.nanmax(np.abs(raw), axis=0) > 1e-6
    return any_finite & not_all_zero

def load_ts(ds, varname, mask=None, vmax=None):
    if ds is None or varname not in ds.variables:
        return None
    raw = np.ma.filled(ds.variables[varname][:], np.nan).astype(float)
    raw[np.abs(raw) > 1e15] = np.nan
    # mGV soil-like 4-D: (layer, time, lat, lon) -- average layers
    if raw.ndim == 4 and raw.shape[0] <= 5 and raw.shape[1] > 50:
        raw = np.nanmean(raw, axis=0)
    if raw.ndim != 3:
        return raw.ravel()[:365]
    T, nlat, nlon = raw.shape
    if mask is not None:
        assert mask.shape == (nlat, nlon), f"mask shape {mask.shape} vs data {(nlat,nlon)}"
        m3 = np.broadcast_to(mask[np.newaxis], raw.shape).copy()
        raw = np.where(m3, raw, np.nan)
    ts = np.nanmean(raw.reshape(T, -1), axis=1)
    if vmax is not None:
        ts[ts > vmax * 5] = np.nan
    return ts[:365]

def load_sm(ds, layer, is_mgv=False, mask=None):
    vn = "soil_moisture_output" if is_mgv else "OUT_SOIL_MOIST"
    if ds is None or vn not in ds.variables:
        return None
    raw = np.ma.filled(ds.variables[vn][:], np.nan).astype(float)
    raw[np.abs(raw) > 1e15] = np.nan
    sh = raw.shape
    if raw.ndim == 4:
        if sh[0] <= 5 and sh[1] > 50:    # (layer, time, lat, lon)
            raw = raw[layer]
        elif sh[1] <= 5 and sh[0] > 50:  # (time, layer, lat, lon)
            raw = raw[:, layer]
    if raw.ndim != 3:
        return raw.ravel()[:365]
    T, nlat, nlon = raw.shape
    if mask is not None:
        m3 = np.broadcast_to(mask[np.newaxis], raw.shape).copy()
        raw = np.where(m3, raw, np.nan)
    return np.nanmean(raw.reshape(T, -1), axis=1)[:365]

def load_baseflow(mds, mask=None):
    t = load_ts(mds, "total_runoff_output",   mask=mask)
    s = load_ts(mds, "surface_runoff_output", mask=mask)
    if t is None or s is None: return None
    n  = min(len(t), len(s))
    bf = t[:n] - s[:n]; bf[bf < 0] = 0.0
    return bf

def get_masks(vds, mds, vic_ref, mgv_ref):
    vm = get_land_mask(vds, vic_ref)
    mm = get_land_mask(mds, mgv_ref)
    if vm is not None and mm is not None and vm.shape == mm.shape:
        shared = vm & mm
        print(f"  shared land cells: {shared.sum()}")
        return shared, shared
    print(f"  VIC cells: {vm.sum() if vm is not None else '?'}  mGV cells: {mm.sum() if mm is not None else '?'}")
    return vm, mm

# ── Plot helpers ──────────────────────────────────────────────────────────────
def style_ax(ax, ylabel=None, xlabels=True):
    ax.set_xlim(1, 365)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(91))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(30))
    ax.tick_params(labelsize=9)
    if not xlabels:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Day of Year", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)

def annotate(ax, v, m):
    ok = np.isfinite(v) & np.isfinite(m)
    if ok.sum() < 5: return
    
    vo, mo = v[ok], m[ok]
    sum_abs_v = np.sum(np.abs(vo))
    mean_v = np.mean(vo)
    
    # PBIAS & NMAE
    pbias = 100.0 * np.sum(vo - mo) / sum_abs_v if sum_abs_v > 0 else np.nan
    nmae = 100.0 * np.sum(np.abs(vo - mo)) / sum_abs_v if sum_abs_v > 0 else np.nan
    
    # NSE
    ss_res = np.sum((vo - mo)**2)
    ss_tot = np.sum((vo - mean_v)**2)
    nse = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Proper R^2
    r2 = np.nan
    if len(vo) > 1:
        corr_matrix = np.corrcoef(vo, mo)
        if corr_matrix.shape == (2, 2):
            r2 = corr_matrix[0, 1]**2
            
    textstr = (
        f"PBIAS: {pbias:+.1f}%\n"
        f"NMAE:  {nmae:.1f}%\n"
        f"NSE:   {nse:.3f}\n"
        f"R²:    {r2:.3f}"
    )
    
    ax.text(0.98, 0.97, textstr,
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            color="#444444", linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC', lw=0.6, alpha=0.9))

def plot_var(ax, v_ts, m_ts, title, unit, xlabels=True):
    style_ax(ax, ylabel=unit, xlabels=xlabels)
    ax.set_title(title)
    if v_ts is None and m_ts is None:
        ax.text(0.5, 0.5, "variable not found", transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color="#888888")
        return
    if v_ts is not None and m_ts is not None:
        # Full comparison: both datasets available
        n = min(len(v_ts), len(m_ts), 365)
        x, v, m = DOY[:n], v_ts[:n], m_ts[:n]
        ax.fill_between(x, v, m, color=MGV_C, alpha=FA)
        ax.plot(x, v, color=VIC_C, lw=0.9, alpha=0.9)
        ax.plot(x, m, color=MGV_C, lw=0.9, ls='--', alpha=0.85)
        annotate(ax, v, m)
    elif v_ts is not None:
        # VIC only
        n = min(len(v_ts), 365)
        ax.plot(DOY[:n], v_ts[:n], color=VIC_C, lw=0.9, alpha=0.9)
    else:
        # mGV only
        n = min(len(m_ts), 365)
        ax.plot(DOY[:n], m_ts[:n], color=MGV_C, lw=0.9, ls='--', alpha=0.85)

def plot_sm_combined(ax, vds, mds, mask_v, mask_m, xlabels=True):
    style_ax(ax, ylabel="mm", xlabels=xlabels)
    ax.set_title("Soil Moisture (L1 / L2 / L3)")
    leg = []
    v_sum = np.zeros(365)
    m_sum = np.zeros(365)
    valid_count = 0
    
    for l in range(3):
        v = load_sm(vds, l, is_mgv=False, mask=mask_v)
        m = load_sm(mds, l, is_mgv=True,  mask=mask_m)
        c = SM_COLORS[l]
        n = 365
        if v is not None and m is not None:
            n = min(len(v), len(m), 365)
            ax.fill_between(DOY[:n], v[:n], m[:n], color=c, alpha=0.08)
            v_sum[:n] += v[:n]
            m_sum[:n] += m[:n]
            valid_count += 1
        if v is not None:
            n = min(len(v), 365)
            ax.plot(DOY[:n], v[:n], color=c, lw=0.9, ls='-',  alpha=0.9)
        if m is not None:
            n = min(len(m), 365)
            ax.plot(DOY[:n], m[:n], color=c, lw=0.9, ls='--', alpha=0.85)
        leg.append(Line2D([0],[0], color=c, lw=2, label=f"L{l+1}"))
    leg += [Line2D([0],[0], color='grey', lw=1.4, ls='-',  label='VIC'),
            Line2D([0],[0], color='grey', lw=1.4, ls='--', label='VIC-WUR-Julia')]
    ax.legend(handles=leg, fontsize=7.5, loc='upper left', ncol=2, framealpha=0.95)
    
    if valid_count == 3:
        ok = (v_sum > 0) & (m_sum > 0)
        if np.any(ok):
            annotate(ax, v_sum[ok], m_sum[ok])

def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close(fig)

def make_water_fig(title, vds, mds, mask_v, mask_m):
    """Create a 2-row water balance figure with 5 ET panels + 3 hydro/SM panels."""
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                           left=0.06, right=0.98, top=0.93, bottom=0.09)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], wspace=0.38)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.38)

    ax_et = [fig.add_subplot(gs_top[0, c]) for c in range(5)]
    ax_hy = [fig.add_subplot(gs_bot[0, c]) for c in range(3)]

    for c, (vv, mv, ttl, unit) in enumerate(ET_VARS):
        plot_var(ax_et[c],
                 load_ts(vds, vv, mask=mask_v),
                 load_ts(mds, mv, mask=mask_m),
                 ttl, unit, xlabels=True)
    ax_et[0].legend(handles=LEGEND_ELEMS, fontsize=8.5, loc='lower right')

    plot_var(ax_hy[0],
             load_ts(vds, "OUT_RUNOFF",   mask=mask_v),
             load_ts(mds, "surface_runoff_output", mask=mask_m),
             "Surface Runoff", "mm d$^{-1}$")
    plot_var(ax_hy[1],
             load_ts(vds, "OUT_BASEFLOW", mask=mask_v),
             load_baseflow(mds, mask=mask_m),
             "Baseflow", "mm d$^{-1}$")
    plot_sm_combined(ax_hy[2], vds, mds, mask_v, mask_m)
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'case',
        help="Generate validation plots for example case. Must be either 'Mekong' or 'Indus'",
        type=str
    )
    args = parser.parse_args()
    case = str(args.case).capitalize()

    if case not in ["Indus", "Mekong"]:
        msg = "Invalid validation case."
        raise ValueError(msg)

    outdir = Path(__file__).parent.resolve()

    if case == "Mekong":
        vic_nc   = outdir / "mekong_VICrun" / "results" / "mekong_test.1979-01-01.nc"
        vic_csv  = outdir / "mekong_VICrun" / "vic_reference_mekong.csv"
        mgv_file = outdir / ".." / "output_data" / "mekong" / "outputfile_mekong_1979.nc"
    elif case == "Indus":
        vic_nc   = outdir / "indus_VICrun" / "results" / "indus_test.1979-01-01.nc"
        vic_csv  = outdir / "indus_VICrun" / "vic_reference_indus.csv"
        mgv_file = outdir / ".." / "output_data" / "indus" / "outputfile_indus_1979.nc"

    # Exit only if BOTH VIC source and mGV output are missing
    if not vic_nc.exists() and not vic_csv.exists() and not mgv_file.exists():
        print("WARNING: Both input files missing, skipping dashboard.", file=sys.stderr)
        sys.exit(0)

    print(f"=== {case} ===")
    vic_mek = open_vic(vic_nc, vic_csv, case)
    mgv_mek = open_dataset(mgv_file, f"mGV {case}")
    mask_v_mek, mask_m_mek = get_masks(vic_mek, mgv_mek, "OUT_SURF_TEMP", "tsurf_output")

    # Energy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{case} Basin — Energy Balance |  VIC vs VIC-WUR-Julia  |  1979",
                fontsize=14, fontweight='bold')
    plt.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.88, bottom=0.13)
    plot_var(axes[0],
            load_ts(vic_mek, "OUT_R_NET",    mask=mask_v_mek),
            load_ts(mgv_mek, "net_radiation_summed_output", mask=mask_m_mek),
            "Net Radiation", "W m$^{-2}$")
    plot_var(axes[1],
            load_ts(vic_mek, "OUT_SURF_TEMP", mask=mask_v_mek),
            load_ts(mgv_mek, "tsurf_output",  mask=mask_m_mek),
            "Surface Temperature", "°C")
    axes[0].legend(handles=LEGEND_ELEMS, fontsize=9)
    save(fig, f"{outdir}/{case.lower()}_energy.png")

    # Water
    fig = make_water_fig(
        f"{case} Basin — Water Balance & ET  |  VIC vs VIC-WUR-Julia  |  1979",
        vic_mek, mgv_mek, mask_v_mek, mask_m_mek)
    save(fig, f"{outdir}/{case.lower()}_water.png")

    if vic_mek is not None: vic_mek.close()
    if mgv_mek is not None: mgv_mek.close()
    print("All done.")
