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

def get_ipo_phases():
    try:
        df = pd.read_csv(IPO_FILE, encoding='utf-8')
    except:
        df = pd.read_csv(IPO_FILE, encoding='gbk')
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    df = df.dropna(subset=[df.columns[0]])
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    for col in df.columns[1:13]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ipo_annual'] = df.iloc[:, 1:13].mean(axis=1)
    df = df.dropna(subset=['ipo_annual'])
    mean_val, std_val = df['ipo_annual'].mean(), df['ipo_annual'].std()
    year_col = df.columns[0]
    neg_years = df[df['ipo_annual'] < (mean_val - 0.5 * std_val)][year_col].tolist()
    pos_years = df[df['ipo_annual'] > (mean_val + 0.5 * std_val)][year_col].tolist()
    return neg_years, pos_years, df[[year_col, 'ipo_annual']]

def process_rainfall(neg_years, pos_years):
    all_years = sorted([int(f.name.split('.')[0]) for f in RAINFALL_DIR.glob('*.monthly_rain.nc')])
    neg_maps, pos_maps = [], []
    for year in all_years:
        with xr.open_dataset(RAINFALL_DIR / f"{year}.monthly_rain.nc") as ds:
            gbr_ds = ds['monthly_rain'].sel(lat=slice(-25, -10), lon=slice(142, 154))
            annual_rain = gbr_ds.sum(dim='time')
            if year in neg_years: neg_maps.append(annual_rain)
            elif year in pos_years: pos_maps.append(annual_rain)
    if not neg_maps or not pos_maps: return None
    return xr.concat(neg_maps, 'year').mean('year') - xr.concat(pos_maps, 'year').mean('year')

def process_chlorophyll(neg_years, pos_years):
    try:
        ds = xr.open_dataset(CHL_FILE)
        chl_var = next(v for v in ds.data_vars if 'chl' in v.lower())
        ds_annual = ds[chl_var].resample(time='YE').mean(dim='time')
        data_years = ds_annual.time.dt.year.values
        valid_neg = [y for y in neg_years if y in data_years]
        valid_pos = [y for y in pos_years if y in data_years]
        m_neg = ds_annual.sel(time=ds_annual.time.dt.year.isin(valid_neg)).mean('time')
        m_pos = ds_annual.sel(time=ds_annual.time.dt.year.isin(valid_pos)).mean('time')
        reg_avg = ds_annual.mean(dim=['latitude', 'longitude']).to_series()
        reg_avg.index = reg_avg.index.year
        return m_neg - m_pos, reg_avg
    except: return None, None

def load_river_data(river_name, neg_years, pos_years):
    try:
        river_path = RUNOFF_DIR / river_name
        csv_file = list(river_path.glob("*.csv"))[0]
        df = pd.read_csv(csv_file, comment='#', names=['timestamp', 'val', 'q', 'i'])
        df['year'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True).dt.year
        ann = df.groupby('year')['val'].mean().reset_index()
        return (ann[ann['year'].isin(pos_years)]['val'].values, 
                ann[ann['year'].isin(neg_years)]['val'].values)
    except: return None, None

# =============================================================================
# 2. Wavelet Utils
# =============================================================================

def sliding_smooth(wave, window_size):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(wave, size=window_size, axis=1, mode='nearest')

def compute_wtc(s1, s2, scales, smooth_win=9):
    n = len(s1)
    c1, _ = pywt.cwt(s1, scales, 'cmor1.5-1.0')
    c2, _ = pywt.cwt(s2, scales, 'cmor1.5-1.0')
    sinv = (1.0 / scales)[:, np.newaxis]
    Wxy = sinv * (c1 * np.conj(c2))
    Wxx = sinv * (np.abs(c1) ** 2)
    Wyy = sinv * (np.abs(c2) ** 2)
    sWxy_t, sWxx_t, sWyy_t = sliding_smooth(Wxy, smooth_win), sliding_smooth(Wxx, smooth_win), sliding_smooth(Wyy, smooth_win)
    def smooth_scale(wave):
        ns = len(scales)
        sm = np.zeros_like(wave)
        dj, dj0 = 0.3, 0.6
        steps = dj0 / (dj * 2)
        for si in range(ns):
            sm[si, :] = np.mean(wave[max(0, int(si-steps)):min(ns, int(si+steps+1)), :], axis=0)
        return sm
    sWxy, sWxx, sWyy = smooth_scale(sWxy_t), smooth_scale(sWxx_t), smooth_scale(sWyy_t)
    Rsq = np.abs(sWxy)**2 / (sWxx * sWyy + 1e-12)
    e_fold = 2 ** 0.5
    coi_v = np.minimum(np.arange(n), np.arange(n)[::-1]) * e_fold
    period = scales
    coi = np.interp(coi_v, scales, period)
    return Rsq, period, coi, np.angle(sWxy)

def wtc_monte_carlo(s1, s2, scales, win=9, n_mc=300):
    n, ns = len(s1), len(scales)
    dist = np.zeros((ns, n_mc * n))
    for i in range(n_mc):
        r, _, _, _ = compute_wtc(s1, np.random.permutation(s2), scales, win)
        dist[:, i*n : (i+1)*n] = r
    return np.percentile(dist, 95, axis=1)

# =============================================================================
# 3. Figure Generation
# =============================================================================

def create_ultimate_figure():
    neg_yrs, pos_yrs, ipo_df = get_ipo_phases()
    rain_diff = process_rainfall(neg_yrs, pos_yrs)
    chl_diff, chl_ts = process_chlorophyll(neg_yrs, pos_yrs)
    rivers = ['Pascoe', 'Burdekin', 'Fitzroy']
    r_data = {r: load_river_data(r, neg_yrs, pos_yrs) for r in rivers}
    
    res_df = pd.read_csv(RESIDUALS_FILE).groupby('year')['residual'].mean().reset_index()
    merged = ipo_df.merge(res_df, left_on=ipo_df.columns[0], right_on='year', how='inner')
    yrs_w, ipo_z, cots_z = merged['year'].values, signal.detrend(stats.zscore(merged['ipo_annual'].values)), signal.detrend(stats.zscore(merged['residual'].values))
    scales = np.arange(2.0, 30.1, 0.3)
    Rsq, period, coi, phase = compute_wtc(ipo_z, cots_z, scales, 9)
    sig_95 = wtc_monte_carlo(ipo_z, cots_z, scales, 9, n_mc=500)

    fig = plt.figure(figsize=(12.5, 20))
    gs = GridSpec(3, 2, height_ratios=[1.3, 1, 1], hspace=0.32, wspace=0.10)

    # --- (a) Rainfall Anomaly ---
    ax_rain = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_rain.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    ax_rain.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='dimgray', linewidth=0.6)
    ax_rain.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    im_a = None
    if rain_diff is not None:
        v = np.nanmax(np.abs(rain_diff.values)) * 0.9
        im_a = rain_diff.plot(ax=ax_rain, cmap='RdBu_r', vmin=-v, vmax=v, add_colorbar=False, transform=ccrs.PlateCarree())
        
    ax_rain.set_title('(a) Rainfall Anomaly (Neg - Pos IPO)', loc='left', fontweight='bold', fontsize=15)
    ax_rain.set_xticks(np.arange(142.5, 155, 4), crs=ccrs.PlateCarree())
    ax_rain.set_yticks(np.arange(-25, -9.5, 5), crs=ccrs.PlateCarree())
    ax_rain.xaxis.set_major_formatter(LongitudeFormatter())
    ax_rain.yaxis.set_major_formatter(LatitudeFormatter())
    ax_rain.set_xlabel(''); ax_rain.set_ylabel('')
    ax_rain.text(0.02, 0.98, 'Red = Wetter in Neg IPO', transform=ax_rain.transAxes, va='top', fontsize=10, 
               fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    r_coords = {'Pascoe': [143.2, -12.5], 'Burdekin': [147.2, -19.7], 'Fitzroy': [150.5, -23.4]}
    for n, c in r_coords.items():
        ax_rain.plot(c[0], c[1], 'ko', markersize=4, transform=ccrs.PlateCarree())
        ax_rain.text(c[0]+0.4, c[1], n, transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold')

    # --- (b) Chl-a Anomaly ---
    ax_chl = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_chl.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    ax_chl.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='dimgray', linewidth=0.6)
    
    im_b = None
    if chl_diff is not None:
        v = np.nanpercentile(np.abs(chl_diff.values), 98)
        im_b = chl_diff.plot(ax=ax_chl, cmap='RdBu_r', vmin=-v, vmax=v, add_colorbar=False, transform=ccrs.PlateCarree())
        
    ax_chl.set_title('(b) Chl-a Anomaly (Neg - Pos IPO)', loc='left', fontweight='bold', fontsize=15)
    ax_chl.set_xticks(np.arange(142.5, 155, 4), crs=ccrs.PlateCarree())
    ax_chl.set_yticks(np.arange(-25, -9.5, 5), crs=ccrs.PlateCarree())
    ax_chl.xaxis.set_major_formatter(LongitudeFormatter())
    ax_chl.yaxis.set_major_formatter(LatitudeFormatter())
    ax_chl.set_xlabel(''); ax_chl.set_ylabel('')
    ax_chl.text(0.02, 0.98, 'Red = Higher Productivity', transform=ax_chl.transAxes, va='top', fontsize=10, 
               fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # CAPTURE POSITIONS TO PROCEED WITH ALIGNMENT
    fig.canvas.draw()
    pos_a = ax_rain.get_position()
    pos_b = ax_chl.get_position()

    # ---- ADD ABSOLUTE COLORBARS FOR ILLUSTRATOR SVG RELIABILITY ----
    cb_w = 0.012
    if im_a is not None:
        # absolute figure coords: [left, bottom, width, height]
        cax_a = fig.add_axes([pos_a.x0 - 0.05, pos_a.y0 + 0.2*pos_a.height, cb_w, 0.6*pos_a.height])
        cb_a = plt.colorbar(im_a, cax=cax_a, orientation='vertical')
        cb_a.set_label('Rain Anomaly (mm/yr)', fontweight='bold', fontsize=9)
        cax_a.yaxis.set_ticks_position('left'); cax_a.yaxis.set_label_position('left')
        
    if im_b is not None:
        cax_b = fig.add_axes([pos_b.x1 + 0.015, pos_b.y0 + 0.2*pos_b.height, cb_w, 0.6*pos_b.height])
        cb_b = plt.colorbar(im_b, cax=cax_b, orientation='vertical')
        cb_b.set_label('Chl-a Anomaly ($mg\\ m^{-3}$)', fontweight='bold', fontsize=9)

    # --- (c, d, e, f) PIXEL PERFECT ALIGNED ---
    ax_box = fig.add_subplot(gs[1, 0]); ax_ts = fig.add_subplot(gs[1, 1])
    ax_wtc = fig.add_subplot(gs[2, 0]); ax_glob = fig.add_subplot(gs[2, 1])

    def align_final(target_ax, anchor_p):
        curr_p = target_ax.get_position()
        target_ax.set_position([anchor_p.x0, curr_p.y0, anchor_p.width, curr_p.height])
        target_ax.set_box_aspect(1) # SQUARE

    align_final(ax_box, pos_a); align_final(ax_ts, pos_b)
    align_final(ax_wtc, pos_a); align_final(ax_glob, pos_b)

    # --- Plot (c) ---
    p_d, labs = [], []
    for r in rivers:
        pos, neg = r_data[r]
        if pos is not None: p_d.extend([pos, neg]); labs.extend([f'{r}\nPos', f'{r}\nNeg'])
    if p_d:
        bp = ax_box.boxplot(p_d, patch_artist=True, labels=labs, showfliers=False, widths=0.6)
        for i, p in enumerate(bp['boxes']): p.set_facecolor(COLORS['pos_ipo'] if i%2==0 else COLORS['neg_ipo']); p.set_alpha(0.7)
        for i in range(len(rivers)):
            stat, p = stats.mannwhitneyu(p_d[i*2+1], p_d[i*2], alternative='greater')
            mv = max(np.max(p_d[i*2]), np.max(p_d[i*2+1]))
            if p < 0.05:
                ax_box.text(i*2 + 1.5, mv * 1.15, '**' if p < 0.01 else '*', ha='center', fontweight='bold', fontsize=12)
                ax_box.plot([i*2+1.1, i*2+1.9], [mv*1.1, mv*1.1], 'k-', lw=1)
    ax_box.set_ylabel('Discharge ($m^3/s$)', fontweight='bold'); ax_box.set_title('(c) River Runoff Sensitivity', loc='left', fontweight='bold')
    ax_box.spines[['top', 'right']].set_visible(False)

    # --- Plot (d) ---
    if chl_ts is not None:
        cy = chl_ts.index.intersection(ipo_df[ipo_df.columns[0]])
        cv, iv = stats.zscore(chl_ts.loc[cy]), stats.zscore(ipo_df[ipo_df[ipo_df.columns[0]].isin(cy)]['ipo_annual'].values)
        ax_ts.plot(cy, -1*iv, 'k--', alpha=0.7, label='Inverted IPO'); ax_ts.plot(cy, cv, color=COLORS['success'], lw=2.5, label='GBR Chl-a')
        r, p = stats.pearsonr(cv, iv)
        sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax_ts.text(0.05, 0.95, f'r = {-r:.3f}{sig_str}', transform=ax_ts.transAxes, va='top', fontweight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    ax_ts.set_title('(d) IPO-Chl Temporal Coupling', loc='left', fontweight='bold'); ax_ts.legend(loc='lower left', frameon=False, fontsize=9)
    ax_ts.spines[['top', 'right']].set_visible(False); ax_ts.set_ylabel('Standardized Anomaly', fontweight='bold')

    # --- Plot (e) ---
    im_e = ax_wtc.pcolormesh(yrs_w, period, Rsq, cmap='RdYlBu_r', vmin=0, vmax=1, shading='auto')
    ax_wtc.contour(yrs_w, period, Rsq/np.tile(sig_95[:, None], (1, len(yrs_w))), levels=[1], colors='white', linewidths=2.0, linestyles='--')
    ax_wtc.plot(yrs_w, coi, 'k--', alpha=0.7); ax_wtc.fill_between(yrs_w, coi, period.max(), alpha=0.3, color='gray')
    ax_wtc.set_yscale('log', base=2); ax_wtc.set_yticks([2, 4, 8, 16, 30]); ax_wtc.set_yticklabels(['2', '4', '8', '16', '30'])
    ax_wtc.invert_yaxis(); ax_wtc.set_title('(e) IPO-COTS Coherence', loc='left', fontweight='bold')
    ax_wtc.set_ylabel('Period (yr)', fontweight='bold'); ax_wtc.set_xlabel('Year', fontweight='bold')
    
    # Absolute Colorbar for (e)
    fig.canvas.draw()
    pos_e = ax_wtc.get_position()
    cax_e = fig.add_axes([pos_e.x0 - 0.05, pos_e.y0 + 0.2*pos_e.height, cb_w, 0.6*pos_e.height])
    cb_e = plt.colorbar(im_e, cax=cax_e, orientation='vertical'); cb_e.set_label('Coherence $R^2$', fontweight='bold', fontsize=9)
    cax_e.yaxis.set_ticks_position('left'); cax_e.yaxis.set_label_position('left')
    
    nw = len(yrs_w); st, ss = max(1, nw//12), max(1, len(period)//10)
    for si in range(0, len(period), ss):
        for ti in range(st//2, nw, st):
            if Rsq[si, ti] > 0.5:
                ax_wtc.annotate('', xy=(yrs_w[ti]+0.5*np.cos(phase[si, ti]), period[si]*(1+0.1*np.sin(phase[si, ti]))), 
                                xytext=(yrs_w[ti], period[si]), arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

    # --- Plot (f) ---
    m_wc = np.array([np.mean(Rsq[si, period[si]<coi]) if np.any(period[si]<coi) else np.nan for si in range(len(period))])
    ax_glob.plot(period, m_wc, color=COLORS['primary'], lw=2.5, label='WTC Mean')
    ax_glob.plot(period, sig_95, color=COLORS['secondary'], lw=1.5, ls='--', label='95% Signif.')
    ax_glob.axvspan(3, 7, color=COLORS['accent'], alpha=0.1, label='ENSO-scale (3-7y)'); ax_glob.axvspan(15, 30, color=COLORS['success'], alpha=0.15, label='IPO (15-30y)')
    pk = np.nanargmax(m_wc)
    ax_glob.scatter(period[pk], m_wc[pk], color='black', s=60, marker='*', zorder=5)
    ax_glob.annotate(f'Peak: {period[pk]:.1f}y', xy=(period[pk], m_wc[pk]), xytext=(period[pk]+3, m_wc[pk]+0.05), arrowprops=dict(arrowstyle='->', color='black', lw=1), fontsize=10, fontweight='bold')
    wtc_summary = pd.DataFrame(
        [
            {"metric": "wtc_global_mean_peak_period_years", "value": float(period[pk])},
            {"metric": "wtc_global_mean_coherence_at_peak", "value": float(m_wc[pk])},
            {"metric": "wtc_series_years", "value": f"{int(yrs_w.min())}-{int(yrs_w.max())}"},
        ]
    )
    wtc_summary.to_csv(
        Path(__file__).resolve().parent.parent / "results" / "tables" / "wtc_summary.csv",
        index=False,
    )
    ax_glob.set_title('(f) Global Coherence', loc='left', fontweight='bold'); ax_glob.set_ylim(0, 1.1)
    ax_glob.set_xlabel('Period (yr)', fontweight='bold'); ax_glob.set_ylabel('Mean Coherence', fontweight='bold')
    ax_glob.spines[['top', 'right']].set_visible(False)

    plt.savefig(OUTPUT_DIR / 'Figure4_Hydroclimatic_Coupling.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(OUTPUT_DIR / 'Figure4_Hydroclimatic_Coupling.svg', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("SVG Colorbars Optimized: Removed inset_axes, using absolute bounding boxes.")

if __name__ == "__main__":
    create_ultimate_figure()
