# -*- coding: utf-8 -*-
"""
Figure 4: Hydroclimatic Contrasts and Low-Frequency Climate Coupling
SVG Optimized Export:
- Replaced inset_axes with explicit absolute coord axes (`fig.add_axes`) for colorbars.
- This creates standalone <g> groups in SVG, making them robust in Adobe Illustrator.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from analysis_inputs import annual_discharge, ipo_phase_years
from config import TABLES_DIR
from scipy import signal, stats
import pywt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings

warnings.filterwarnings('ignore')

# Use Agg backend for headless consistency
plt.switch_backend('Agg')

# Set aesthetic parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
RAINFALL_DIR = BASE_DIR / "climate" / "rainfall"
RUNOFF_DIR = BASE_DIR / "hydrology" / "runoff"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
RAW_DATA_DIR = BASE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IPO_FILE = BASE_DIR / "climate" / "ipo_index.csv"
CHL_FILE = BASE_DIR / "climate" / "GBR_chlorophyll_monthly_1998_2025.nc"
RESIDUALS_FILE = Path(__file__).resolve().parent.parent / "results" / "tables" / "all_residuals.csv"

COLORS = {
    'primary': '#1f77b4',
    'secondary': '#d62728',
    'accent': '#ff7f0e',
    'success': '#2ca02c',
    'gray': '#7f7f7f',
    'pos_ipo': '#fde0dd',
    'neg_ipo': '#7a0177',
}

# =============================================================================
# 1. Data Loading functions
# =============================================================================

def process_rainfall(neg_years, pos_years):
    maps = {}
    for path in sorted(RAINFALL_DIR.glob('*.monthly_rain.nc')):
        year = int(path.name.split('.')[0])
        if year not in neg_years and year not in pos_years:
            continue
        with xr.open_dataset(path) as data:
            crop = data['monthly_rain'].sel(lat=slice(-25, -10), lon=slice(142, 154))
            if set(crop.time.dt.month.values) != set(range(1, 13)) or crop.sizes['time'] != 12:
                raise ValueError(f'Incomplete rainfall year: {path}')
            maps[year] = crop.sum(dim='time', skipna=False).load()
    negative = [grid for year, grid in maps.items() if year in neg_years]
    positive = [grid for year, grid in maps.items() if year in pos_years]
    return xr.concat(negative, 'year').mean('year') - xr.concat(positive, 'year').mean('year')


def process_chlorophyll(neg_years, pos_years):
    with xr.open_dataset(CHL_FILE) as data:
        monthly = data['CHL'].sel(latitude=slice(-25, -10), longitude=slice(142, 154))
        grouped = monthly.resample(time='YE')
        annual = grouped.mean().where(grouped.count() >= 12).load()
    neg = annual.sel(time=annual.time.dt.year.isin(neg_years)).mean('time')
    pos = annual.sel(time=annual.time.dt.year.isin(pos_years)).mean('time')
    area_weights = np.cos(np.deg2rad(annual['latitude']))
    series = annual.weighted(area_weights).mean(['latitude', 'longitude']).to_series()
    series.index = series.index.year
    return neg - pos, series


def create_ultimate_figure():
    from publication_style import save_publication
    neg_yrs, pos_yrs, ipo = ipo_phase_years()
    rain_diff = process_rainfall(neg_yrs, pos_yrs)
    chl_diff, chl_ts = process_chlorophyll(neg_yrs, pos_yrs)
    rivers = ['Pascoe', 'Burdekin', 'Fitzroy']
    discharge = {river: annual_discharge(river) for river in rivers}

    fig = plt.figure(figsize=(12.5, 12))
    gs = GridSpec(2, 2, height_ratios=[1.3, 1], hspace=0.33, wspace=0.24)

    # --- A: Rainfall Anomaly ---
    ax_rain = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_rain.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    ax_rain.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='dimgray', linewidth=0.6)
    ax_rain.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    im_a = None
    if rain_diff is not None:
        v = np.nanmax(np.abs(rain_diff.values)) * 0.9
        im_a = rain_diff.plot(ax=ax_rain, cmap='RdBu_r', vmin=-v, vmax=v, add_colorbar=False, transform=ccrs.PlateCarree())
        
    ax_rain.set_title('Rainfall anomaly (Neg − Pos IPO)', loc='center', fontweight='bold', fontsize=10.5, pad=6)
    ax_rain.text(-0.14, 1.09, 'A', transform=ax_rain.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    ax_rain.set_xticks(np.arange(142.5, 155, 4), crs=ccrs.PlateCarree())
    ax_rain.set_yticks(np.arange(-25, -9.5, 5), crs=ccrs.PlateCarree())
    ax_rain.xaxis.set_major_formatter(LongitudeFormatter())
    ax_rain.yaxis.set_major_formatter(LatitudeFormatter())
    ax_rain.set_xlabel(''); ax_rain.set_ylabel('')
    ax_rain.text(0.03, 0.97, 'Red = Wetter in Neg IPO', transform=ax_rain.transAxes, va='top', ha='left', fontsize=8.8, 
               fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))
    
    r_coords = {'Pascoe': [143.05015, -12.65779], 'Burdekin': [147.24362, -19.75856], 'Fitzroy': [150.10714, -23.08897]}
    r_offsets = {'Pascoe': (-0.45, -0.15, 'right'), 'Burdekin': (-0.45, -0.20, 'right'), 'Fitzroy': (-0.45, -0.15, 'right')}
    for n, c in r_coords.items():
        dx, dy, ha_lbl = r_offsets[n]
        ax_rain.plot(c[0], c[1], 'ko', markersize=4.8, markeredgecolor='white', markeredgewidth=0.6, transform=ccrs.PlateCarree(), zorder=6)
        ax_rain.text(c[0] + dx, c[1] + dy, n, transform=ccrs.PlateCarree(), fontsize=9.2, fontweight='bold',
                     va='center', ha=ha_lbl, zorder=7,
                     bbox=dict(facecolor='white', alpha=0.78, edgecolor='none', boxstyle='round,pad=0.15'))

    # --- B: Chl-a Anomaly ---
    ax_chl = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_chl.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    ax_chl.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='dimgray', linewidth=0.6)
    
    im_b = None
    if chl_diff is not None:
        v = np.nanpercentile(np.abs(chl_diff.values), 98)
        im_b = chl_diff.plot(ax=ax_chl, cmap='RdBu_r', vmin=-v, vmax=v, add_colorbar=False, transform=ccrs.PlateCarree())
        
    ax_chl.set_title('Chl-a anomaly (Neg − Pos IPO)', loc='center', fontweight='bold', fontsize=10.5, pad=6)
    ax_chl.text(-0.14, 1.09, 'B', transform=ax_chl.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    ax_chl.set_xticks(np.arange(142.5, 155, 4), crs=ccrs.PlateCarree())
    ax_chl.set_yticks(np.arange(-25, -9.5, 5), crs=ccrs.PlateCarree())
    ax_chl.xaxis.set_major_formatter(LongitudeFormatter())
    ax_chl.yaxis.set_major_formatter(LatitudeFormatter())
    ax_chl.set_xlabel(''); ax_chl.set_ylabel('')
    ax_chl.text(0.03, 0.97, 'Red = Higher Chl-a', transform=ax_chl.transAxes, va='top', ha='left', fontsize=8.8, 
               fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))

    # ---- ADD ABSOLUTE VERTICAL COLORBARS (MATCHING OLD DRAFT) ----
    if im_a is not None:
        cax_a = fig.add_axes([0.085, 0.595, 0.014, 0.222])
        cb_a = plt.colorbar(im_a, cax=cax_a, orientation='vertical')
        cb_a.set_label('Rain anomaly (mm yr$^{-1}$)', fontweight='bold', fontsize=9.5, labelpad=6)
        cax_a.yaxis.set_ticks_position('left')
        cax_a.yaxis.set_label_position('left')
        
    if im_b is not None:
        cax_b = fig.add_axes([0.892, 0.595, 0.014, 0.222])
        cb_b = plt.colorbar(im_b, cax=cax_b, orientation='vertical')
        cb_b.set_label('Chl-a anomaly (mg m$^{-3}$)', fontweight='bold', fontsize=9.5, rotation=270, labelpad=18)

    ax_box = fig.add_subplot(gs[1, 0])
    ax_ts = fig.add_subplot(gs[1, 1])
    results = []
    values, positions_box = [], []
    p_vals = []
    for idx_r, (river, series) in enumerate(discharge.items()):
        positive = series.loc[series.index.isin(pos_yrs)]
        negative = series.loc[series.index.isin(neg_yrs)]
        values.extend([positive, negative])
        base_x = idx_r * 2.5 + 1.0
        positions_box.extend([base_x, base_x + 0.8])
        pval = float(stats.mannwhitneyu(negative, positive, alternative='two-sided').pvalue)
        p_vals.append((base_x, base_x + 0.8, max(positive.max(), negative.max()), pval))
        results.append({
            'river': river, 'positive_years': len(positive), 'negative_years': len(negative),
            'positive_mean_m3s': positive.mean(), 'negative_mean_m3s': negative.mean(),
            'neg_minus_pos_m3s': negative.mean() - positive.mean(),
            'unadjusted_rank_test_p_two_sided': pval,
        })
    pd.DataFrame(results).to_csv(TABLES_DIR / 'river_phase_contrasts.csv', index=False)
    boxes = ax_box.boxplot(values, positions=positions_box, patch_artist=True, widths=0.62,
                           flierprops=dict(marker='o', markersize=4.5, markeredgecolor='#333333'))
    for index, box in enumerate(boxes['boxes']):
        box.set_facecolor(COLORS['pos_ipo'] if index % 2 == 0 else COLORS['neg_ipo'])
        box.set_edgecolor('#333333')
    for median in boxes['medians']:
        median.set_color('#ff7f0e')
        median.set_linewidth(1.5)
    from matplotlib.patches import Patch
    ax_box.legend(
        handles=[Patch(facecolor=COLORS['pos_ipo'], edgecolor='#333333', label='Pos IPO'),
                 Patch(facecolor=COLORS['neg_ipo'], edgecolor='#333333', label='Neg IPO')],
        loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#cccccc', fontsize=8.8
    )
    y_top_max = max(p[2] for p in p_vals)
    for x1, x2, ymax_pair, pv in p_vals:
        yb = ymax_pair + 0.055 * y_top_max
        ax_box.plot([x1, x2], [yb, yb], color='#333333', lw=0.9)
        ax_box.text((x1 + x2) / 2.0, yb + 0.018 * y_top_max, f'$p$ = {pv:.3f}',
                    ha='center', va='bottom', fontsize=8.2)
    ax_box.set_ylim(-40, y_top_max * 1.18)
    ax_box.set_xticks([1.4, 3.9, 6.4])
    ax_box.set_xticklabels(rivers, fontsize=9.5, fontweight='bold')
    ax_box.set(ylabel='Annual mean discharge (m$^3$ s$^{-1}$)', title='River discharge by IPO phase')
    ax_box.text(-0.14, 1.08, 'C', transform=ax_box.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')

    # --- D: Area-Weighted Chl-a Anomaly vs Inverted Annual IPO (1998-2025) ---
    joined_chl = pd.concat([chl_ts.rename('chl'), ipo.rename('ipo')], axis=1).dropna().sort_index()
    cv = stats.zscore(joined_chl['chl'].to_numpy(dtype=float))
    # Preserve NOAA TPI's physical 1971-2000 climatological zero (mean = 0.000) rather than re-centering on 1998-2025
    ipo_full = ipo.sort_index().dropna()
    clim_ref = ipo_full.loc[1971:2000] if 1971 in ipo_full.index else ipo_full
    ipo_clim_mean = float(clim_ref.mean())
    ipo_clim_sd = float(clim_ref.std(ddof=1))
    iv = (joined_chl['ipo'].to_numpy(dtype=float) - ipo_clim_mean) / ipo_clim_sd
    b_lp, a_lp = signal.butter(3, 1.0 / 6.5, btype='low')
    ipo_lp_full = pd.Series(signal.filtfilt(b_lp, a_lp, ipo_full.to_numpy(dtype=float)), index=ipo_full.index)
    iv_lp = (ipo_lp_full.reindex(joined_chl.index).to_numpy(dtype=float) - ipo_clim_mean) / ipo_clim_sd
    r_chl, p_chl = stats.pearsonr(cv, iv)
    # Bretherton et al. (1999) effective sample size adjustment for lag-1 autocorrelation
    r1_c = float(np.corrcoef(cv[:-1], cv[1:])[0, 1])
    r1_i = float(np.corrcoef(iv[:-1], iv[1:])[0, 1])
    n_eff = len(cv) * (1.0 - r1_c * r1_i) / (1.0 + r1_c * r1_i)
    n_eff = max(3.0, min(float(len(cv)), n_eff))
    t_eff = r_chl * np.sqrt((n_eff - 2.0) / max(1.0 - r_chl ** 2, 1e-12))
    p_eff = float(2.0 * stats.t.sf(abs(t_eff), df=n_eff - 2.0))

    pd.DataFrame([{
        'start_year': int(joined_chl.index.min()),
        'end_year': int(joined_chl.index.max()),
        'n_years': int(len(joined_chl)),
        'pearson_r_raw_ipo': float(r_chl),
        'p_two_sided': float(p_chl),
        'n_eff_bretherton': float(n_eff),
        'p_two_sided_neff': p_eff,
    }]).to_csv(TABLES_DIR / 'chl_ipo_correlation.csv', index=False)

    ax_ts.axhline(0, color='gray', linestyle=':', lw=0.9)
    ax_ts.plot(joined_chl.index, -iv_lp, color='#5b7c99', linestyle='-', lw=2.0, alpha=0.75, label='13-yr low-pass −TPI')
    ax_ts.plot(joined_chl.index, -iv, color='black', linestyle='--', lw=1.5, label='Inverted annual TPI (−TPI)')
    ax_ts.plot(joined_chl.index, cv, color=COLORS['success'], marker='o', markersize=3.8, lw=1.6, label='Area-weighted Chl-a')
    r_disp = f"{r_chl:+.3f}".replace('-', '−')
    ax_ts.text(
        0.04, 0.96,
        f'Annual TPI vs Chl-a: $r$ = {r_disp}\n$p$ = {p_chl:.3f} ($N_{{\\mathrm{{eff}}}}$ $p$ = {p_eff:.3f})',
        transform=ax_ts.transAxes, va='top', ha='left', fontsize=8.8,
        bbox=dict(facecolor='white', alpha=0.88, edgecolor='none', pad=2),
    )
    ax_ts.set_ylim(-2.15, 2.60)
    ax_ts.set(xlabel='Year', ylabel='Standardized index / anomaly (SD)', title='Chlorophyll-a and inverted TPI (1971–2000 ref.)')
    ax_ts.text(-0.14, 1.08, 'D', transform=ax_ts.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    ax_ts.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cccccc', fontsize=8.3)
    for axis in (ax_box, ax_ts):
        axis.spines[['top', 'right']].set_visible(False)

    # --- Exploratory Grinsted et al. (2004) log-scale smoothed WTC diagnostic (saved for SI) ---
    res_df = pd.read_csv(RESIDUALS_FILE)
    res_annual = res_df.groupby('year')['residual'].mean().dropna()
    joined_wtc = pd.concat([res_annual.rename('res'), ipo.rename('ipo')], axis=1).dropna().sort_index()
    full_years = np.arange(int(joined_wtc.index.min()), int(joined_wtc.index.max()) + 1)
    joined_wtc = joined_wtc.reindex(full_years).interpolate(method='linear')
    x_sig = stats.zscore(joined_wtc['ipo'].to_numpy(dtype=float))
    y_sig = stats.zscore(joined_wtc['res'].to_numpy(dtype=float))
    n_wtc = len(x_sig)

    periods_wtc = 2.0 ** np.linspace(np.log2(2.0), np.log2(28.0), 60)
    dj = float(np.log2(periods_wtc[1] / periods_wtc[0]))
    scales = periods_wtc / 1.0
    wavelet = 'cmor1.5-1.0'
    box_n = max(3, int(round(0.6 / dj)))
    if box_n % 2 == 0:
        box_n += 1

    from scipy.ndimage import gaussian_filter1d, uniform_filter1d

    def _grinsted_wtc(xs, ys):
        Wx, _ = pywt.cwt(xs, scales, wavelet, sampling_period=1.0)
        Wy, _ = pywt.cwt(ys, scales, wavelet, sampling_period=1.0)

        def sm(arr):
            out = np.empty_like(arr, dtype=complex if np.iscomplexobj(arr) else float)
            for i, sc in enumerate(scales):
                if np.iscomplexobj(arr):
                    out[i] = gaussian_filter1d(arr[i].real, sigma=sc, mode='nearest') + 1j * gaussian_filter1d(arr[i].imag, sigma=sc, mode='nearest')
                else:
                    out[i] = gaussian_filter1d(arr[i], sigma=sc, mode='nearest')
            if np.iscomplexobj(out):
                return uniform_filter1d(out.real, size=box_n, axis=0, mode='nearest') + 1j * uniform_filter1d(out.imag, size=box_n, axis=0, mode='nearest')
            return uniform_filter1d(out, size=box_n, axis=0, mode='nearest')

        s_norm = 1.0 / scales[:, None]
        S_Wxy = sm(s_norm * Wx * np.conj(Wy))
        S_Wx = sm(s_norm * (np.abs(Wx) ** 2))
        S_Wy = sm(s_norm * (np.abs(Wy) ** 2))
        return np.clip((np.abs(S_Wxy) ** 2) / np.maximum(S_Wx * S_Wy, 1e-12), 0.0, 1.0)

    R2_wtc = _grinsted_wtc(x_sig, y_sig)
    band_mask = (periods_wtc >= 15.0) & (periods_wtc <= 30.0)
    obs_band_mean = float(R2_wtc[band_mask].mean())
    band_idx = np.where(band_mask)[0]
    peak_i = band_idx[int(np.argmax(R2_wtc[band_idx].mean(axis=1)))]
    peak_period = float(periods_wtc[peak_i])
    peak_mean_coh = float(R2_wtc[peak_i].mean())

    # AR(1) Monte Carlo significance test (500 draws)
    a1_x = float(np.corrcoef(x_sig[:-1], x_sig[1:])[0, 1])
    a1_y = float(np.corrcoef(y_sig[:-1], y_sig[1:])[0, 1])
    rng_w = np.random.default_rng(42)
    null_band = np.empty(500, dtype=float)
    for m in range(500):
        ex, ey = rng_w.normal(size=n_wtc), rng_w.normal(size=n_wtc)
        sx, sy = np.empty(n_wtc), np.empty(n_wtc)
        sx[0], sy[0] = ex[0], ey[0]
        for k_t in range(1, n_wtc):
            sx[k_t] = a1_x * sx[k_t - 1] + np.sqrt(max(1.0 - a1_x ** 2, 0.0)) * ex[k_t]
            sy[k_t] = a1_y * sy[k_t - 1] + np.sqrt(max(1.0 - a1_y ** 2, 0.0)) * ey[k_t]
        null_band[m] = float(_grinsted_wtc(sx, sy)[band_mask].mean())
    p_wtc_ar1 = float((1 + np.sum(null_band >= obs_band_mean)) / 501.0)

    pd.DataFrame([{
        'start_year': int(full_years[0]),
        'end_year': int(full_years[-1]),
        'n_years': int(n_wtc),
        'peak_period_years': peak_period,
        'peak_mean_coherence': peak_mean_coh,
        'band_15_30_mean_coherence': obs_band_mean,
        'ar1_null_mean': float(null_band.mean()),
        'ar1_null_q95': float(np.quantile(null_band, 0.95)),
        'ar1_p_value': p_wtc_ar1,
        'near_coi': True,
    }]).to_csv(TABLES_DIR / 'wtc_summary.csv', index=False)

    save_publication(fig, OUTPUT_DIR / 'Figure4_Hydroclimatic_Coupling')
    plt.close()
    print("SVG Colorbars Optimized: Removed inset_axes, using absolute bounding boxes.")

if __name__ == "__main__":
    create_ultimate_figure()
