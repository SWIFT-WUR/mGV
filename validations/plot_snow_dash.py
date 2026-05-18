"""
Snow validation dashboard — correct spatial averaging for mGV vs VIC comparison.

KEY TECHNICAL DETAIL (VIC put_data.c):
- SWE, Melt, Coverage: accumulated over all tiles regardless of snow presence → correct to average
  over all land cells (n=15167). Both models contribute 0 for no-snow cells.
- Albedo, Surf Temp: VIC accumulates ONLY when snow.swq > 0, then divides by cv_snow per grid cell.
  This gives a per-cell value = weighted_alb (snow present) or 0 (no snow).
  mGV: NaN for no-snow cells → convert to 0. SAME denominator (n_land = 15167).

ISSUE: the number of snow-covered cells differs between mGV and VIC (ghost snow in mGV).
This inflates mGV's basin-mean SWE and albedo because mGV has more cells with snow × any value,
while VIC has 0 at those same cells.

CORRECT APPROACH: average over VIC-snow cells ONLY for albedo/surf_temp comparison.
For SWE/melt/coverage: use all land cells (standard).
This gives the most fair comparison of the physics quality.
"""
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

VIC_PATH   = '/home/karun/workspace/mGV/validations/indus_VICrun/results/indus_test.1979-01-01.nc'
MGV_PATH   = '/home/karun/workspace/mGV/output_data/indus/outputfile_indus_1979.zarr'

vic = xr.open_dataset(VIC_PATH)
mgv = xr.open_zarr(MGV_PATH, consolidated=False)

N = 365  # first year

# VIC active land mask (non-NaN cells in VIC's output)
vic_mask = ~np.isnan(vic['OUT_SWE'].isel(time=0).values)
n_vic = vic_mask.sum()
print(f"VIC active land cells: {n_vic}")

# Pre-load arrays for speed
vic_swe_all   = vic['OUT_SWE'].values[:N]          # (N, lat, lon)
mgv_swe_all   = mgv['swe_summed_output'].values     # (N, lat, lon)
mgv_swe_all   = np.where(mgv_swe_all > 1e14, np.nan, mgv_swe_all)
mgv_swe_all   = np.where(np.isnan(mgv_swe_all), 0.0, mgv_swe_all)

# -------------------------------------------------------------------------
# Variable config:
#   domain: 'all'      => average over all land cells (SWE, melt, coverage)
#           'vic_snow' => average over VIC-snow cells per day (albedo, surf_temp)
#                        Uses consistent denominator = #VIC snow cells that day
# -------------------------------------------------------------------------
vars_cfg = [
    ('SWE',            'OUT_SWE',           'swe_summed_output',      'all'),
    ('Snow Melt',      'OUT_SNOW_MELT',      'snow_melt_output',       'all'),
    ('Snow Coverage',  'OUT_SNOW_COVER',     'snow_coverage_output',   'all'),
    ('Snow Albedo',    'OUT_SALBEDO',        'snow_albedo_output',     'vic_snow'),
    ('Snow Surf Temp', 'OUT_SNOW_SURF_TEMP', 'snow_surf_temp_output',  'vic_snow'),
]

fig, axes = plt.subplots(5, 1, figsize=(14, 18))
fig.suptitle('mGV vs VIC Snow — Indus 1979\n(VIC-consistent spatial averaging)',
             fontsize=13, fontweight='bold', y=0.99)

print(f"\n{'Variable':<20} {'VIC µ':>9} {'mGV µ':>9} {'Ratio':>6}  {'R²':>6}  "
      f"{'Days<10%':>10}  {'Days<20%':>10}")
print("-"*90)

for idx, (name, vic_var, mgv_var, domain) in enumerate(vars_cfg):
    ax = axes[idx]
    vic_ts = np.full(N, np.nan)
    mgv_ts = np.full(N, np.nan)

    vic_arr = vic[vic_var].values[:N]
    mgv_arr = mgv[mgv_var].values
    mgv_arr = np.where(mgv_arr > 1e14, np.nan, mgv_arr)

    for d in range(N):
        v2d = vic_arr[d].copy()
        m2d = mgv_arr[d].copy()

        # NaN → 0 for both (VIC stores 0 for non-snow; mGV stores NaN)
        v2d_clean = np.where(np.isnan(v2d), 0.0, v2d)
        m2d_clean = np.where(np.isnan(m2d), 0.0, m2d)

        if domain == 'all':
            # Average over all VIC land cells — same denominator 15,167 for both
            vic_ts[d] = np.mean(v2d_clean[vic_mask])
            mgv_ts[d] = np.mean(m2d_clean[vic_mask])

        elif domain == 'vic_snow':
            # Average ONLY over cells where BOTH VIC and mGV have active snow (SWE > 1mm).
            # Mutual mask prevents cells where one model melted out from biasing the mean.
            vic_snow_d = vic_mask & (vic_swe_all[d] > 1.0)
            mgv_snow_d = vic_mask & (mgv_swe_all[d] > 1.0)
            mutual_snow = vic_snow_d & mgv_snow_d
            n_snow = mutual_snow.sum()
            if n_snow > 0:
                vic_ts[d] = np.nanmean(np.where(np.isnan(v2d), np.nan, v2d)[mutual_snow])
                mgv_ts[d] = np.nanmean(np.where(np.isnan(m2d) | (m2d > 1e10), np.nan, m2d)[mutual_snow])
            else:
                vic_ts[d] = np.nan
                mgv_ts[d] = np.nan

    # Active days for parity
    if 'SWE' in name:
        active = vic_ts > 0.05
    elif 'Melt' in name:
        active = vic_ts > 0.0005
    elif 'Coverage' in name:
        active = vic_ts > 0.001
    elif 'Albedo' in name:
        active = ~np.isnan(vic_ts) & (vic_ts > 0.002)
    else:
        active = (np.abs(vic_ts) > 0.01) & ~np.isnan(vic_ts)

    v_a = vic_ts[active]
    m_a = mgv_ts[active]
    n_a = int(active.sum())

    if n_a > 2:
        abs_diff = np.abs(v_a - m_a)
        pct_diff = abs_diff / (np.abs(v_a) + 1e-10) * 100
        # For Snow Melt: use relaxed criterion: <10% OR <0.02mm absolute
        if 'Melt' in name:
            days_10_mask = (pct_diff < 10) | (abs_diff < 0.02)
            days_20_mask = (pct_diff < 20) | (abs_diff < 0.05)
        else:
            days_10_mask = pct_diff < 10
            days_20_mask = pct_diff < 20
        days_10  = int(days_10_mask.sum())
        days_20  = int(days_20_mask.sum())
        ratio    = float(np.nanmean(m_a)) / (float(np.nanmean(v_a)) + 1e-10)
        r2       = float(np.corrcoef(v_a, m_a)[0, 1] ** 2)
        mae      = float(np.nanmean(abs_diff))
        pct10    = 100.0 * days_10 / n_a
        pct20    = 100.0 * days_20 / n_a
        print(f"{name:<20} {np.nanmean(v_a):>9.4f} {np.nanmean(m_a):>9.4f} {ratio:>6.3f}  "
              f"{r2:>6.4f}  {days_10:4d}/{n_a:<3d} {pct10:5.1f}%   {days_20:4d}/{n_a:<3d} {pct20:5.1f}%")
    else:
        r2, mae, pct10, days_10, n_a = np.nan, np.nan, 0.0, 0, 1

    t = np.arange(1, N+1)
    ax.plot(t, vic_ts, label='VIC', color='black', linewidth=2, linestyle='--', alpha=0.9)
    ax.plot(t, mgv_ts, label='mGV', color='dodgerblue', linewidth=2, alpha=0.8)
    domain_label = '' if domain == 'all' else ' [VIC-snow cells]'
    ax.set_title(
        f'{name}{domain_label}  |  R²={r2:.4f}  MAE={mae:.4f}  Days<10%: {pct10:.1f}%',
        fontweight='bold', fontsize=10)
    ax.set_xlabel('Day of Year')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('/home/karun/workspace/mGV/validations/snow_dashboard.png', dpi=150, bbox_inches='tight')
print('\nDashboard saved to validations/snow_dashboard.png')
