#!/usr/bin/env python3
"""
4 clean comparison plots: VIC vs VIC-WUR-Julia, 1979
White background, no precipitation, no air temp, no snow physics.
Surface temperature uses proper per-dataset land masks.
"""
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

VIC_MEK = "/home/karun/workspace/mGV/validations/mekong_VICrun/results/mekong_test.1979-01-01.nc"
MGV_MEK = "/home/karun/workspace/mGV/output_data/mekong/outputfile_mekong_1979.nc"
VIC_IND = "/home/karun/workspace/mGV/validations/indus_VICrun/results/indus_test.1979-01-01.nc"
MGV_IND = "/home/karun/workspace/mGV/output_data/indus/outputfile_indus_1979.nc"
OUTDIR  = "/home/karun/workspace/mGV/validations"

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

# ── Data helpers ──────────────────────────────────────────────────────────────
def get_land_mask(ds, ref_var):
    """2-D bool mask: pixels with at least some valid, non-zero data."""
    if ref_var not in ds.variables:
        return None
    raw = np.ma.filled(ds.variables[ref_var][:], np.nan).astype(float)
    raw[np.abs(raw) > 1e15] = np.nan
    if raw.ndim == 4:
        raw = raw[0]
    any_finite  = np.any(np.isfinite(raw), axis=0)
    not_all_zero = np.nanmax(np.abs(raw), axis=0) > 1e-6
    return any_finite & not_all_zero

def load_ts(ds, varname, mask=None, vmax=None):
    if varname not in ds.variables:
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
    if vn not in ds.variables:
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
    denom = np.sum(np.abs(v[ok]))
    pbias = 100.0 * np.sum(v[ok] - m[ok]) / denom if denom > 0 else np.nan
    ss_res = np.sum((v[ok] - m[ok])**2)
    ss_tot = np.sum((v[ok] - np.mean(v[ok]))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ax.text(0.98, 0.97,
            f"PBIAS = {pbias:+.1f}%   R² = {r2:.3f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color="#444444",
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC', lw=0.6))

def plot_var(ax, v_ts, m_ts, title, unit, xlabels=True):
    style_ax(ax, ylabel=unit, xlabels=xlabels)
    ax.set_title(title)
    if v_ts is None or m_ts is None:
        ax.text(0.5, 0.5, "variable not found", transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color="#888888")
        return
    n = min(len(v_ts), len(m_ts), 365)
    x, v, m = DOY[:n], v_ts[:n], m_ts[:n]
    ax.fill_between(x, v, m, color=MGV_C, alpha=FA)
    ax.plot(x, v, color=VIC_C, lw=0.9, alpha=0.9)
    ax.plot(x, m, color=MGV_C, lw=0.9, ls='--', alpha=0.85)
    annotate(ax, v, m)

def plot_sm_combined(ax, vds, mds, mask_v, mask_m, xlabels=True):
    style_ax(ax, ylabel="mm", xlabels=xlabels)
    ax.set_title("Soil Moisture (L1 / L2 / L3)")
    leg = []
    for l in range(3):
        v = load_sm(vds, l, is_mgv=False, mask=mask_v)
        m = load_sm(mds, l, is_mgv=True,  mask=mask_m)
        c = SM_COLORS[l]
        if v is not None and m is not None:
            n = min(len(v), len(m), 365)
            ax.fill_between(DOY[:n], v[:n], m[:n], color=c, alpha=0.08)
            ax.plot(DOY[:n], v[:n], color=c, lw=0.9, ls='-',  alpha=0.9)
            ax.plot(DOY[:n], m[:n], color=c, lw=0.9, ls='--', alpha=0.85)
        leg.append(Line2D([0],[0], color=c, lw=2, label=f"L{l+1}"))
    leg += [Line2D([0],[0], color='grey', lw=1.4, ls='-',  label='VIC'),
            Line2D([0],[0], color='grey', lw=1.4, ls='--', label='VIC-WUR-Julia')]
    ax.legend(handles=leg, fontsize=7.5, loc='upper right', ncol=2, framealpha=0.95)

def save(fig, name):
    path = f"{OUTDIR}/{name}"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close(fig)

# ET variable spec (reused for both basins)
ET_VARS = [
    ("OUT_EVAP",       "total_et_output",                    "Total ET",           "mm d$^{-1}$"),
    ("OUT_TRANSP_VEG", "transpiration_summed_output",        "Transpiration",       "mm d$^{-1}$"),
    ("OUT_EVAP_CANOP", "canopy_evaporation_summed_output",   "Canopy Evaporation",  "mm d$^{-1}$"),
    ("OUT_EVAP_BARE",  "soil_evaporation_output",            "Soil Evaporation",    "mm d$^{-1}$"),
    ("OUT_PET",        "potential_evaporation_summed_output","Potential ET",         "mm d$^{-1}$"),
]

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
    ax_et[0].legend(handles=LEGEND_ELEMS, fontsize=8.5, loc='upper right')

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

# =============================================================================
# INDUS
# =============================================================================
print("=== Indus ===")
vic_ind = nc.Dataset(VIC_IND)
mgv_ind = nc.Dataset(MGV_IND)
mask_v_ind, mask_m_ind = get_masks(vic_ind, mgv_ind, "OUT_SURF_TEMP", "tsurf_output")

# Energy
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Indus Basin — Energy Balance  |  VIC vs VIC-WUR-Julia  |  1979",
             fontsize=14, fontweight='bold')
plt.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.88, bottom=0.13)
plot_var(axes[0],
         load_ts(vic_ind, "OUT_R_NET",    mask=mask_v_ind),
         load_ts(mgv_ind, "net_radiation_summed_output", mask=mask_m_ind),
         "Net Radiation", "W m$^{-2}$")
plot_var(axes[1],
         load_ts(vic_ind, "OUT_SURF_TEMP", mask=mask_v_ind),
         load_ts(mgv_ind, "tsurf_output",  mask=mask_m_ind),
         "Surface Temperature", "°C")
axes[0].legend(handles=LEGEND_ELEMS, fontsize=9)
save(fig, "indus_energy.png")

# Water
fig = make_water_fig(
    "Indus Basin — Water Balance & ET  |  VIC vs VIC-WUR-Julia  |  1979",
    vic_ind, mgv_ind, mask_v_ind, mask_m_ind)
save(fig, "indus_water.png")

vic_ind.close(); mgv_ind.close()

# =============================================================================
# MEKONG
# =============================================================================
print("=== Mekong ===")
vic_mek = nc.Dataset(VIC_MEK)
mgv_mek = nc.Dataset(MGV_MEK)
mask_v_mek, mask_m_mek = get_masks(vic_mek, mgv_mek, "OUT_SURF_TEMP", "tsurf_output")

# Energy
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Mekong Basin — Energy Balance  |  VIC vs VIC-WUR-Julia  |  1979",
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
save(fig, "mekong_energy.png")

# Water
fig = make_water_fig(
    "Mekong Basin — Water Balance & ET  |  VIC vs VIC-WUR-Julia  |  1979",
    vic_mek, mgv_mek, mask_v_mek, mask_m_mek)
save(fig, "mekong_water.png")

vic_mek.close(); mgv_mek.close()
print("All done.")
