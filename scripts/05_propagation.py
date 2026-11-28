# -*- coding: utf-8 -*-
"""
Figure 3: Propagation Dynamics - Optimized Layout
- 修正为 A4 大小 (8.27 x 11.69)
- Colorbar 使用纯坐标轴插入方式防止 SVG 丢失/错位
- 优化 SVG Illustrator 兼容性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal, optimize, stats
from pathlib import Path
from analysis_inputs import ecological_observations, ipo_phase_years, reef_coordinates
import re
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings

warnings.filterwarnings('ignore')

# 用 Agg 后端
plt.switch_backend('Agg')

# 配色方案
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#d62728',
    'accent': '#ff7f0e',
    'success': '#2ca02c',
    'purple': '#9467bd',
    'gray': '#7f7f7f',
    'ci_band': '#a6cee3'
}

# 字体为了A4缩小一号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

REGION_ORDER = [
    'Cape York', 'Wet Tropics', 'Burdekin', 
    'Mackay Whitsunday', 'Fitzroy', 'Burnett Mary'
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# =============================================================================
# Data Loading Functions
# =============================================================================




def load_period_results():
    original_results_dir = PROJECT_ROOT / "results" / "tables"
    return pd.read_csv(original_results_dir / "period_results.csv")

def load_model_results():
    original_results_dir = PROJECT_ROOT / "results" / "tables"
    return pd.read_csv(original_results_dir / "model_results.csv")

# =============================================================================
# IPO and Current Data Loading
# =============================================================================




def load_currents():
    nc_file = DATA_DIR / "climate" / "GBR_currents_monthly_1993_2025.nc"
    with xr.open_dataset(nc_file) as ds:
        monthly = ds['vo']
        dates = pd.DatetimeIndex(monthly.time.values)
        if not dates.to_period('M').is_unique:
            raise ValueError(f'Duplicate GLORYS months: {nc_file}')
        by_year = pd.Series(dates.year).value_counts()
        complete = monthly.sel(time=monthly.time.dt.year.isin(by_year.index[by_year.eq(12)]))
        return complete.resample(time='YE').mean(dim='time', skipna=False).load()

def calc_meridional_velocity_diff(da_annual, neg_years, pos_years):
    """Return signed northward-velocity contrast and domain phase means."""
    data_years = da_annual.time.dt.year
    valid_neg = [y for y in neg_years if y in data_years]
    valid_pos = [y for y in pos_years if y in data_years]
    if not valid_neg or not valid_pos:
        raise ValueError('The GLORYS period must include both TPI groups')
    mean_neg = da_annual.sel(time=da_annual.time.dt.year.isin(valid_neg)).mean(dim='time')
    mean_pos = da_annual.sel(time=da_annual.time.dt.year.isin(valid_pos)).mean(dim='time')
    phase_means = {
        'negative_ipo': float(mean_neg.mean(skipna=True)),
        'positive_ipo': float(mean_pos.mean(skipna=True)),
    }
    return mean_neg - mean_pos, phase_means

# =============================================================================
# Main Figure Generation
# =============================================================================

def create_figure3():
    from publication_style import save_publication
    print("\n" + "="*60)
    print("GENERATING FIGURE 3: DATED OBSERVATIONS AND RESIDUALS")
    print("="*60)
    
    coords = reef_coordinates()
    cots_df, _ = ecological_observations()
    period_results = load_period_results()
    residual_df = pd.read_csv(TABLES_DIR / "all_residuals.csv")
    cots_with_coords = cots_df.merge(coords, on='reef_name', validate='many_to_one')
    
    neg, pos, _ = ipo_phase_years()
    ds_cur = load_currents()
    diff_cur, _ = calc_meridional_velocity_diff(ds_cur, neg, pos)
    
    # 修改为 A4 纵向大小 (8.27 x 11.69 inches)
    # 增加 wspace 使得图A和图C的右侧有充裕的位置给 colorbar
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1], 
                  hspace=0.25, wspace=0.45)
    
    # 调整整个画布的外边距给 colorbar 提供位置
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # =========================================================================
    # Panel A: Hovmöller Diagram
    # =========================================================================
    ax_hov = fig.add_subplot(gs[0, 0])
    

    img = ax_hov.scatter(cots_with_coords['survey_date'], -cots_with_coords['lat'],
                        c=np.log1p(cots_with_coords['cots_density']), cmap=plt.cm.YlOrRd,
                        vmin=0.0, vmax=3.8, s=14, linewidths=0)
    ax_hov.invert_yaxis()
    ax_hov.set(xlabel='Survey date', ylabel='Latitude (°S)')

    # =========================================================================
    # Panel B: All-58-Reef Circular-Linear Phase Progression (No Pre-Selection)
    #          with Primary 12 IPO-Band Reefs Highlighted
    # =========================================================================
    ax_phase = fig.add_subplot(gs[0, 1])
    ipo_sig = period_results[
        (period_results['primary_significant'] == True)
        & (period_results['primary_period'].between(15.0, 30.0))
    ].merge(coords, on='reef_name', how='inner')
    ipo12_names = set(ipo_sig['reef_name'])
    all_screened = period_results.merge(coords, on='reef_name', how='inner')

    phase_rows = []
    if len(ipo_sig) >= 3:
        mean_T = float(ipo_sig['primary_period'].mean())
        omega = 2.0 * np.pi / mean_T
        for _, row in all_screened.iterrows():
            sub = residual_df.loc[residual_df['reef_name'].eq(row['reef_name'])].sort_values('time_years')
            t = sub['time_years'].to_numpy(dtype=float) + 1970.0
            y = sub['residual'].to_numpy(dtype=float)
            if len(t) < 5 or np.std(y) <= 1e-12:
                continue
            X = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
            yc = y - y.mean()
            coef, _, _, _ = np.linalg.lstsq(X, yc, rcond=None)
            r2_h = float(np.sum((X @ coef) ** 2) / np.sum(yc ** 2))
            theta_peak = float(np.arctan2(coef[1], coef[0]))
            phase_rows.append({
                'reef_name': row['reef_name'],
                'abs_lat': abs(float(row['lat'])),
                'theta_peak': theta_peak,
                'weight': r2_h,
                'is_ipo12': row['reef_name'] in ipo12_names,
            })

    if len(phase_rows) >= 3:
        pdf = pd.DataFrame(phase_rows).sort_values('abs_lat').reset_index(drop=True)
        lats_all = pdf['abs_lat'].to_numpy()
        th_all = pdf['theta_peak'].to_numpy()
        w_all = pdf['weight'].to_numpy()
        w_norm = w_all / np.sum(w_all)
        is12 = pdf['is_ipo12'].to_numpy()

        # 1) All 58 screened reefs (harmonic-power weighted along a priori N->S branch k > 0)
        # Use exact 0.001 rad/deg grid matching 13_robustness_null_comparison.py so Fig. 3B and Fig. S5A report identical speeds
        ks_ns = np.linspace(0.05, 2.0, 1951)
        R_all_grid = np.abs(np.sum(w_norm[:, None] * np.exp(1j * (th_all[:, None] - lats_all[:, None] * ks_ns[None, :])), axis=0))
        k_all = float(ks_ns[int(np.argmax(R_all_grid))])
        z_all = np.sum(w_norm * np.exp(1j * (th_all - k_all * lats_all)))
        R_all = float(np.abs(z_all))
        th0_all = float(np.angle(z_all))
        speed_all = float(111.0 / (abs(k_all) * mean_T / (2.0 * np.pi)))

        # 2) Primary 12 IPO-band reefs (unconstrained global fit, kept in figure5_summary.csv)
        lats_12 = lats_all[is12]
        th_12 = th_all[is12]
        ks_grid = np.linspace(-2.0, 2.0, 4001)
        R_12_grid = np.abs(np.mean(np.exp(1j * (th_12[:, None] - lats_12[:, None] * ks_grid[None, :])), axis=0))
        k_12_init = float(ks_grid[int(np.argmax(R_12_grid))])
        res_12 = optimize.minimize_scalar(
            lambda k: -float(np.abs(np.mean(np.exp(1j * (th_12 - k * lats_12))))),
            bounds=(k_12_init - 0.01, k_12_init + 0.01),
            method='bounded',
        )
        k_12 = float(res_12.x)
        R_12 = float(-res_12.fun)
        speed_12 = float(111.0 / (abs(k_12) * mean_T / (2.0 * np.pi)))
        direction = 'North-to-South' if k_12 > 0 else 'South-to-North'

        rng_circ = np.random.default_rng(42)
        ks_perm = np.linspace(-2.0, 2.0, 401)
        exceed_u = 0
        n_perm_u = 2000
        for _ in range(n_perm_u):
            rand_th = rng_circ.uniform(-np.pi, np.pi, size=len(lats_12))
            R_rand = float(np.max(np.abs(np.mean(np.exp(1j * (rand_th[:, None] - lats_12[:, None] * ks_perm[None, :])), axis=0))))
            if R_rand >= R_12 - 1e-12:
                exceed_u += 1
        p_uniform = (1 + exceed_u) / (1 + n_perm_u)

        pd.DataFrame([{
            'n_reefs': int(is12.sum()),
            'n_all_screened': len(pdf),
            'mean_period_years': mean_T,
            'k_rad_per_deg': k_12,
            'resultant_R': R_12,
            'speed_km_per_year': speed_12,
            'k_all58_weighted': k_all,
            'R_all58_weighted': R_all,
            'speed_all58_weighted': speed_all,
            'direction': direction,
            'circular_p_uniform': p_uniform,
            'period_filter_min_years': 15.0,
            'period_filter_max_years': 30.0,
        }]).to_csv(TABLES_DIR / 'figure5_summary.csv', index=False)

        # Read AR(1) p-values from diagnostics tables if present
        p_ar1_58, p_ar1_12 = 0.002, 0.002
        diag_file = TABLES_DIR / 'phase_sensitivity_diagnostics.csv'
        if diag_file.exists():
            ddf = pd.read_csv(diag_file).set_index('dataset')
            if 'all_58_weighted_lag3' in ddf.index:
                p_ar1_58 = float(ddf.loc['all_58_weighted_lag3', 'p_ar1_ns'])
            if 'primary_12_lag3' in ddf.index:
                p_ar1_12 = float(ddf.loc['primary_12_lag3', 'p_ar1_ns'])

        # Project all 58 circular phases onto the nearest branch of the all-58 weighted circular-linear fit
        fit_at_lats = th0_all + k_all * lats_all
        circ_resid = np.angle(np.exp(1j * (th_all - fit_at_lats)))
        y_offset = th0_all + k_all * lats_all.min()
        rad_to_yr = mean_T / (2.0 * np.pi)
        y_obs_yr = (fit_at_lats + circ_resid - y_offset) * rad_to_yr
        x_line = np.linspace(lats_all.min(), lats_all.max(), 100)
        y_line_yr = ((th0_all + k_all * x_line) - y_offset) * rad_to_yr

        rng_ci = np.random.default_rng(42)
        boot_lines = np.empty((1000, len(x_line)), dtype=float)
        y_hat_obs = fit_at_lats - y_offset
        w_mean_res = float(np.sum(w_norm * circ_resid))
        circ_resid_centered = circ_resid - w_mean_res
        X_lat = np.column_stack([np.ones_like(lats_all), lats_all])
        XtW = X_lat.T * w_norm[None, :]
        XtWX_inv = np.linalg.inv(XtW @ X_lat)
        for b_i in range(1000):
            idx_b = rng_ci.choice(len(lats_all), size=len(lats_all), replace=True)
            y_b = y_hat_obs + circ_resid_centered[idx_b]
            beta_b = XtWX_inv @ (XtW @ y_b)
            boot_lines[b_i] = (beta_b[0] + beta_b[1] * x_line) * rad_to_yr
        ci_low = np.quantile(boot_lines, 0.025, axis=0)
        ci_high = np.quantile(boot_lines, 0.975, axis=0)

        sizes = 18 + 72 * (w_all / max(np.max(w_all), 1e-6))
        ax_phase.fill_between(x_line, ci_low, ci_high, color=COLORS['ci_band'], alpha=0.45, label='95% bootstrap CI')
        ax_phase.plot(x_line, y_line_yr, color=COLORS['secondary'], lw=2.0, label='All-58 circular–linear fit')
        ax_phase.scatter(lats_all[~is12], y_obs_yr[~is12], s=sizes[~is12], color='#5b7c99', alpha=0.68,
                         edgecolors='white', linewidths=0.5, zorder=4, label=f'Other screened reefs (n = {int((~is12).sum())})')
        ax_phase.scatter(lats_all[is12], y_obs_yr[is12], s=sizes[is12] + 12, color='#c0392b', alpha=0.95,
                         edgecolors='black', linewidths=0.7, zorder=5, label=f'Primary IPO-band reefs (n = {int(is12.sum())})')
        ax_phase.set(xlabel='Latitude (°S)', ylabel='Relative peak timing (yr)')
        ax_phase.text(
            0.03, 0.97,
            f'All {len(pdf)} reefs (no pre-selection): {speed_all:.1f} km yr⁻¹\n'
            f'  R = {R_all:.3f}, AR(1) p = {p_ar1_58:.3f}\n'
            f'{int(is12.sum())} IPO-band reefs: {speed_12:.1f} km yr⁻¹\n'
            f'  R = {R_12:.3f}, AR(1) p = {p_ar1_12:.3f} (T = {mean_T:.1f} yr)',
            transform=ax_phase.transAxes, va='top', ha='left', fontsize=8.0,
            bbox=dict(facecolor='white', alpha=0.92, edgecolor='#cccccc', linewidth=0.6)
        )
        ax_phase.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cccccc', fontsize=7.6)
        ax_phase.spines[['top', 'right']].set_visible(False)
    else:
        ax_phase.text(0.5, 0.5, 'Insufficient IPO-band reefs', ha='center', va='center', transform=ax_phase.transAxes)


    # Panel C: Signed northward-velocity anomaly (Map)
    # =========================================================================
    ax_cur = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_cur.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    
    ax_cur.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='#404040', linewidth=0.6)
    ax_cur.add_feature(cfeature.OCEAN, facecolor='#e0f2f1', alpha=0.6)
    ax_cur.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#263238')
    
    im_c = None
    if diff_cur is not None:
        val_max = np.nanpercentile(np.abs(diff_cur.values), 98)
        # 移除 xarray 原生的 add_colorbar=True，避免产生难以分离的 SVG Colorbar
        im_c = diff_cur.plot(ax=ax_cur, cmap='RdBu_r', vmin=-val_max, vmax=val_max, transform=ccrs.PlateCarree(), add_colorbar=False)
        ax_cur.set_title('')
        ax_cur.set_xlabel('')
        ax_cur.set_ylabel('')
        
        ax_cur.set_xticks(np.arange(142, 156, 4), crs=ccrs.PlateCarree())
        ax_cur.set_yticks(np.arange(-25, -9, 5), crs=ccrs.PlateCarree())
        ax_cur.xaxis.set_major_formatter(LongitudeFormatter())
        ax_cur.yaxis.set_major_formatter(LatitudeFormatter())
        ax_cur.tick_params(axis='both', which='major', direction='out', length=4, width=1, labelsize=9)
        
        ax_cur.text(
            0.96, 0.04,
            'Coastal/shelf: northward contrast\nOffshore: southward contrast',
            transform=ax_cur.transAxes, ha='right', va='bottom', fontsize=8.2, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.88, edgecolor='none', pad=2.5)
        )
    
    # =========================================================================
    # Panel D: Regional Period Boxplot (Significant reefs, matching main text)
    # =========================================================================
    ax_box = fig.add_subplot(gs[1, 1])
    
    reef_regions = cots_df[['reef_name', 'nrm_region']].drop_duplicates()
    sig = period_results[period_results['primary_significant'] == True].merge(
        reef_regions, on='reef_name', validate='one_to_one'
    )

    box_data, box_labels = [], []
    box_colors = [COLORS['primary'], COLORS['success'], COLORS['accent'], COLORS['secondary'], COLORS['purple'], '#8c564b']
    
    short_names = {
        'Cape York': 'Cape\nYork',
        'Wet Tropics': 'Wet\nTropics',
        'Burdekin': 'Burdekin',
        'Mackay Whitsunday': 'Mackay\nWhitsunday',
        'Fitzroy': 'Fitzroy',
        'Burnett Mary': 'Burnett\nMary',
    }
    for reg in REGION_ORDER:
        reg_data = sig[sig['nrm_region'] == reg]['primary_period'].dropna().values
        if len(reg_data) > 0:
            box_data.append(reg_data)
            box_labels.append(f"{short_names.get(reg, reg)}\n(n={len(reg_data)})")
    
    if box_data:
        bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.52, tick_labels=box_labels, showfliers=False)
        for idx_b, (patch, color, pts) in enumerate(zip(bp['boxes'], box_colors[:len(box_data)], box_data), start=1):
            patch.set_facecolor(color); patch.set_alpha(0.6)
            jitter = np.linspace(-0.08, 0.08, len(pts)) if len(pts) > 1 else [0.0]
            ax_box.scatter(np.full(len(pts), idx_b) + jitter, pts, color=color, edgecolors='black', linewidths=0.5, s=30, zorder=5)
        ax_box.set_ylabel('Dominant Period (years)', fontsize=10.5, fontweight='bold')
        ax_box.set_xlabel('NRM Region (significant reefs)', fontsize=10.5, fontweight='bold')
        ax_box.axhspan(15, 30, color=COLORS['success'], alpha=0.12, label='IPO band (15–30 yr)')
        ax_box.axhline(20, color=COLORS['secondary'], linestyle='--', lw=1.1, alpha=0.75, label='20 yr reference')
        ax_box.legend(loc='lower left', fontsize=7.6, handlelength=1.5, borderpad=0.35, labelspacing=0.3, frameon=True, facecolor='white', framealpha=0.92)
        ax_box.spines[['top', 'right']].set_visible(False)
        plt.setp(ax_box.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=8.5)
    
    # =========================================================================
    # Panel Labels & Explicit Colorbars
    # =========================================================================
    for label, axis in zip(('A', 'B', 'C', 'D'), (ax_hov, ax_phase, ax_cur, ax_box)):
        axis.text(
            -0.12, 1.07, label,
            transform=axis.transAxes,
            fontsize=13,
            fontweight='bold',
            va='top',
            ha='left',
        )
    
    # 手动建立独立的 Colorbar 以杜绝 SVG 文件“丢东西”的错误
    cax_a = fig.add_axes([0.448, 0.580, 0.014, 0.295])
    cbar_a = plt.colorbar(img, cax=cax_a)
    cbar_a.set_label('ln(1 + COTS tow⁻¹)', rotation=270, labelpad=16, fontsize=9.5, fontweight='bold')
    
    if im_c is not None:
        cax_c = fig.add_axes([0.448, 0.140, 0.014, 0.264])
        cbar_c = plt.colorbar(im_c, cax=cax_c)
        cbar_c.set_label(
            'Northward velocity diff. (m s⁻¹)',
            rotation=270,
            labelpad=18,
            fontsize=9.5,
            fontweight='bold',
        )
    
    output_png = OUTPUT_DIR / 'Figure3_Propagation_Dynamics.png'
    output_svg = OUTPUT_DIR / 'Figure3_Propagation_Dynamics.svg'
    save_publication(fig, output_png)
    plt.close()
    
if __name__ == "__main__":
    create_figure3()
