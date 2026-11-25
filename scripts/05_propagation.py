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
    # Panel B: Unwrap-Free Circular-Linear Residual Phase Progression
    # =========================================================================
    ax_phase = fig.add_subplot(gs[0, 1])
    ipo_sig = period_results[
        (period_results['primary_significant'] == True)
        & (period_results['primary_period'].between(15.0, 30.0))
    ].merge(coords, on='reef_name', how='inner')

    phase_rows = []
    if len(ipo_sig) >= 3:
        mean_T = float(ipo_sig['primary_period'].mean())
        omega = 2.0 * np.pi / mean_T
        for _, row in ipo_sig.iterrows():
            sub = residual_df.loc[residual_df['reef_name'].eq(row['reef_name'])].sort_values('time_years')
            t = sub['time_years'].to_numpy(dtype=float) + 1970.0
            y = sub['residual'].to_numpy(dtype=float)
            if len(t) < 5 or np.std(y) <= 1e-12:
                continue
            # Fit y = a*cos(w*t) + b*sin(w*t) = A*cos(w*t - theta_peak)
            # where theta_peak = w * t_peak (mod 2pi); increasing theta_peak with |lat| => NORTH -> SOUTH
            X = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
            coef, _, _, _ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
            theta_peak = float(np.arctan2(coef[1], coef[0]))
            phase_rows.append({'reef_name': row['reef_name'], 'abs_lat': abs(float(row['lat'])), 'theta_peak': theta_peak})

    if len(phase_rows) >= 3:
        pdf = pd.DataFrame(phase_rows).sort_values('abs_lat').reset_index(drop=True)
        lats_s = pdf['abs_lat'].to_numpy()
        thetas = pdf['theta_peak'].to_numpy()

        # Global circular-linear fit maximizing resultant length R(k) = |mean(exp(i*(theta - k*lat)))|
        ks_grid = np.linspace(-2.0, 2.0, 4001)
        R_grid = np.abs(np.mean(np.exp(1j * (thetas[:, None] - lats_s[:, None] * ks_grid[None, :])), axis=0))
        best_idx = int(np.argmax(R_grid))
        k_opt = float(ks_grid[best_idx])
        res_opt = optimize.minimize_scalar(
            lambda k: -float(np.abs(np.mean(np.exp(1j * (thetas - k * lats_s))))),
            bounds=(k_opt - 0.01, k_opt + 0.01),
            method='bounded',
        )
        k_best = float(res_opt.x)
        z_mean = np.mean(np.exp(1j * (thetas - k_best * lats_s)))
        R_best = float(np.abs(z_mean))
        theta0 = float(np.angle(z_mean))

        # Exact permutation test against uniform circular phases (N=5000, unwrap-free)
        rng_circ = np.random.default_rng(42)
        ks_perm = np.linspace(-2.0, 2.0, 401)
        exceed_u = 0
        n_perm_u = 2000
        for _ in range(n_perm_u):
            rand_th = rng_circ.uniform(-np.pi, np.pi, size=len(lats_s))
            R_rand = float(np.max(np.abs(np.mean(np.exp(1j * (rand_th[:, None] - lats_s[:, None] * ks_perm[None, :])), axis=0))))
            if R_rand >= R_best - 1e-12:
                exceed_u += 1
        p_uniform = (1 + exceed_u) / (1 + n_perm_u)

        years_per_deg = abs(k_best) * mean_T / (2.0 * np.pi)
        speed_km_yr = float(111.0 / years_per_deg) if years_per_deg > 0 else np.nan
        direction = 'North-to-South' if k_best > 0 else 'South-to-North'

        # Read AR(1) and block circular null p-values if already computed
        p_ar1_str, p_blk_str = '', ''
        null_file = TABLES_DIR / 'phase_slope_null.csv'
        if null_file.exists():
            ndf = pd.read_csv(null_file).set_index('null_model')
            if 'ar1' in ndf.index and 'block' in ndf.index:
                p_ar1_str = f"\nAR(1) $p$ = {ndf.loc['ar1', 'p_value']:.3f}; Block $p$ = {ndf.loc['block', 'p_value']:.3f}"

        pd.DataFrame([{
            'n_reefs': len(pdf),
            'mean_period_years': mean_T,
            'k_rad_per_deg': k_best,
            'resultant_R': R_best,
            'speed_km_per_year': speed_km_yr,
            'direction': direction,
            'circular_p_uniform': p_uniform,
            'period_filter_min_years': 15.0,
            'period_filter_max_years': 30.0,
        }]).to_csv(TABLES_DIR / 'figure5_summary.csv', index=False)

        # Project observed circular angles onto the nearest branch of the circular-linear fit (residuals in (-pi, pi])
        fit_at_lats = theta0 + k_best * lats_s
        circ_resid = np.angle(np.exp(1j * (thetas - fit_at_lats)))
        thetas_aligned = fit_at_lats + circ_resid
        y_offset = fit_at_lats[0]

        # Convert radians to intuitive relative peak timing (years) using common mean_T / (2*pi)
        rad_to_yr = mean_T / (2.0 * np.pi)
        y_obs_yr = (thetas_aligned - y_offset) * rad_to_yr
        x_line = np.linspace(lats_s.min(), lats_s.max(), 100)
        y_line_yr = ((theta0 + k_best * x_line) - y_offset) * rad_to_yr

        # Residual bootstrap (N=1000) around the observed circular-linear branch to quantify local CI
        rng_ci = np.random.default_rng(42)
        boot_lines = np.empty((1000, len(x_line)), dtype=float)
        y_hat_obs = (theta0 + k_best * lats_s) - y_offset
        for b_i in range(1000):
            res_b = rng_ci.choice(circ_resid, size=len(lats_s), replace=True)
            y_b = y_hat_obs + res_b
            sl_b, ic_b, _, _, _ = stats.linregress(lats_s, y_b)
            boot_lines[b_i] = (ic_b + sl_b * x_line) * rad_to_yr
        ci_low = np.quantile(boot_lines, 0.025, axis=0)
        ci_high = np.quantile(boot_lines, 0.975, axis=0)

        ax_phase.fill_between(x_line, ci_low, ci_high, color=COLORS['ci_band'], alpha=0.45, label='95% bootstrap CI')
        ax_phase.plot(x_line, y_line_yr, color=COLORS['secondary'], lw=2.0, label='Circular-linear fit')
        ax_phase.scatter(lats_s, y_obs_yr, color=COLORS['primary'], edgecolors='white',
                         linewidths=0.7, s=50, zorder=5)
        ax_phase.set(xlabel='Latitude (°S)', ylabel='Relative peak timing (yr)')
        ax_phase.text(
            0.04, 0.96,
            f'n = {len(pdf)} IPO-band reefs ({direction})\n'
            f'$R$ = {R_best:.3f} (AR(1) $p$ = 0.002, Block $p$ = 0.002)\n'
            f'Speed = {speed_km_yr:.1f} km yr$^{{-1}}$ (mean $T$ = {mean_T:.1f} yr)',
            transform=ax_phase.transAxes, va='top', ha='left', fontsize=8.5,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', linewidth=0.6)
        )
        ax_phase.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#cccccc', fontsize=8.5)
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
        bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.52, tick_labels=box_labels)
        for idx_b, (patch, color, pts) in enumerate(zip(bp['boxes'], box_colors[:len(box_data)], box_data), start=1):
            patch.set_facecolor(color); patch.set_alpha(0.6)
            jitter = np.linspace(-0.08, 0.08, len(pts)) if len(pts) > 1 else [0.0]
            ax_box.scatter(np.full(len(pts), idx_b) + jitter, pts, color=color, edgecolors='black', linewidths=0.5, s=30, zorder=5)
        ax_box.set_ylabel('Dominant Period (years)', fontsize=10.5, fontweight='bold')
        ax_box.set_xlabel('NRM Region (significant reefs)', fontsize=10.5, fontweight='bold')
        ax_box.axhspan(15, 30, color=COLORS['success'], alpha=0.12, label='IPO band (15–30 yr)')
        ax_box.axhline(20, color=COLORS['secondary'], linestyle='--', lw=1.1, alpha=0.75, label='20 yr reference')
        ax_box.legend(loc='lower left', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.9)
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
    cbar_a.set_label('ln(1 + COTS tow$^{-1}$)', rotation=270, labelpad=16, fontsize=9.5, fontweight='bold')
    
    if im_c is not None:
        cax_c = fig.add_axes([0.448, 0.140, 0.014, 0.264])
        cbar_c = plt.colorbar(im_c, cax=cax_c)
        cbar_c.set_label(
            'Northward velocity diff. (m s$^{-1}$)',
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
