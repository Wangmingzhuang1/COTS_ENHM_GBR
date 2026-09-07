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

def load_coordinates():
    coord_file = DATA_DIR / "spatial" / "site_lon_lat.csv"
    df = pd.read_csv(coord_file, encoding='gbk')
    df.columns = ['site', 'coords'] + list(df.columns[2:])
    
    def parse_dms(dms_str):
        if pd.isna(dms_str) or not dms_str: return None, None
        def parse_part(part_str):
            nums = re.findall(r"(\d+(?:\.\d+)?)", part_str)
            direction = re.search(r"([NSEW])", part_str)
            if not direction: return None
            direction = direction.group(1)
            vals = []
            if len(nums) == 3: vals = [float(n) for n in nums]
            elif len(nums) == 2:
                n1, n2 = nums[0], nums[1]
                if len(n1) == 5 and float(n1) > 10000: vals = [float(n1[:3]), float(n1[3:]), float(n2)]
                elif len(n2) == 4 and float(n2) > 60: vals = [float(n1), float(n2[:2]), float(n2[2:])]
                else: vals = [float(n1), float(n2), 0.0]
            elif len(nums) == 1: vals = [float(nums[0]), 0.0, 0.0]
            else: return None
            deg = vals[0] + vals[1]/60 + vals[2]/3600
            if direction in ['S', 'W']: deg = -deg
            return deg
        dms_str = dms_str.strip().upper()
        match = re.match(r"(.*[SN])\s*(.*[EW])", dms_str)
        if match:
            lat_str, lon_str = match.groups()
            lat = parse_part(lat_str)
            lon = parse_part(lon_str)
            return lat, lon
        return None, None
    
    coords_parsed = []
    df['site'] = df['site'].replace({'Kangaroo Reef': 'Kangaroo Reef West'})
    for _, row in df.iterrows():
        lat, lon = parse_dms(row['coords'])
        coords_parsed.append({'reef_name': row['site'], 'lat': lat, 'lon': lon})
    return pd.DataFrame(coords_parsed).dropna()

def load_cots_data():
    csv_files = list((DATA_DIR / "reef_surveys").glob("*_cots_modelled_*.csv"))
    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            reef_name = f.name.split('_Manta Tow_cots_modelled_')[0]
            if 'year' in df.columns: year_col = 'year'
            elif 'Year' in df.columns: year_col = 'Year'
            else: continue
            if 'cotsptow' in df.columns: val_col = 'cotsptow'
            elif 'cots' in df.columns: val_col = 'cots'
            else: continue
            region_col = 'nrm_region' if 'nrm_region' in df.columns else None
            for _, row in df.iterrows():
                try:
                    data_row = {'reef_name': reef_name, 'year': int(row[year_col]), 'cots_density': float(row[val_col])}
                    if region_col: data_row['nrm_region'] = row[region_col]
                    all_data.append(data_row)
                except: continue
        except: continue
    return pd.DataFrame(all_data)

def load_period_results():
    original_results_dir = PROJECT_ROOT / "results" / "tables"
    return pd.read_csv(original_results_dir / "period_results.csv")

def load_model_results():
    original_results_dir = PROJECT_ROOT / "results" / "tables"
    return pd.read_csv(original_results_dir / "model_results.csv")

# =============================================================================
# IPO and Current Data Loading
# =============================================================================

def load_ipo_annual():
    ipo_file = DATA_DIR / "climate" / "ipo_index.csv"
    try: df = pd.read_csv(ipo_file, encoding='utf-8')
    except: df = pd.read_csv(ipo_file, encoding='gbk')
    year_col = df.columns[0]
    monthly_cols = df.columns[1:13]
    monthly_data = []
    for _, row in df.iterrows():
        try:
            year = int(row[year_col])
            for m, col in enumerate(monthly_cols):
                val = float(row[col])
                if not np.isnan(val): monthly_data.append({'year': year, 'month': m+1, 'ipo': val})
        except: pass
    df_mon = pd.DataFrame(monthly_data)
    df_mon['date'] = pd.to_datetime(df_mon[['year', 'month']].assign(day=1))
    ipo_ts = df_mon.set_index('date')['ipo']
    ipo_annual = ipo_ts.resample('YE').mean()
    std = ipo_annual.std()
    mean = ipo_annual.mean()
    neg_years = ipo_annual[ipo_annual < (mean - 0.5*std)].index.year.tolist()
    pos_years = ipo_annual[ipo_annual > (mean + 0.5*std)].index.year.tolist()
    return ipo_annual, neg_years, pos_years

def load_currents():
    nc_file = DATA_DIR / "climate" / "GBR_currents_monthly_1993_2025.nc"
    try:
        ds = xr.open_dataset(nc_file)
        v_var = None
        for v in ['vo', 'v', 'northward_velocity']:
            if v in ds.variables:
                v_var = v; break
        if not v_var: return None
        ds_annual = ds[v_var].resample(time='YE').mean(dim='time')
        return ds_annual
    except Exception as e: return None

def calc_speed_diff(da_annual, neg_years, pos_years):
    da_speed = np.abs(da_annual)
    data_years = da_speed.time.dt.year
    valid_neg = [y for y in neg_years if y in data_years]
    valid_pos = [y for y in pos_years if y in data_years]
    if not valid_neg or not valid_pos: return None, None
    mean_neg = da_speed.sel(time=da_speed.time.dt.year.isin(valid_neg)).mean(dim='time')
    mean_pos = da_speed.sel(time=da_speed.time.dt.year.isin(valid_pos)).mean(dim='time')
    return mean_neg - mean_pos, (valid_neg, valid_pos)

# =============================================================================
# Main Figure Generation
# =============================================================================

def create_figure3():
    print("\n" + "="*60)
    print("GENERATING FIGURE 3: PROPAGATION DYNAMICS (A4 OPTIMIZED)")
    print("="*60)
    
    coords = load_coordinates()
    cots_df = load_cots_data()
    period_results = load_period_results()
    residual_df = pd.read_csv(TABLES_DIR / "all_residuals.csv")
    cots_with_coords = cots_df.merge(coords, on='reef_name', how='inner')
    
    if len(cots_with_coords) == 0: return
    
    ipo_annual, neg, pos = load_ipo_annual()
    ds_cur = load_currents()
    diff_cur = None
    if ds_cur is not None:
        diff_cur, _ = calc_speed_diff(ds_cur, neg, pos)
    
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
    
    ptable = cots_with_coords.pivot_table(index=['lat', 'nrm_region', 'reef_name'], columns='year', values='cots_density', aggfunc='mean')
    ptable = ptable.reset_index().sort_values('lat', ascending=False)
    matrix_data = ptable.drop(columns=['lat', 'nrm_region', 'reef_name'])
    matrix_values = np.log1p(matrix_data.values)
    
    latitudes = np.abs(ptable['lat'].values)
    years = matrix_data.columns.values
    
    cmap = plt.cm.YlOrRd
    img = ax_hov.imshow(matrix_values, aspect='auto', cmap=cmap, 
                       interpolation='nearest', origin='upper',
                       extent=[years.min(), years.max(), latitudes.max(), latitudes.min()])
    
    ax_hov.set_ylabel('Latitude (°S)', fontsize=10, fontweight='bold')
    ax_hov.set_xlabel('Year', fontsize=10, fontweight='bold')
    # 移除 matplotlib 原生的 colorbar 以避免在改变轴尺寸时出错，我们在最后用绝对位置绘制
    
    if len(latitudes) > 20:
        start_year, start_lat = 1995, latitudes[5]
        end_year, end_lat = 2015, latitudes[-10]
        ax_hov.annotate("", xy=(end_year, end_lat), xytext=(start_year, start_lat), arrowprops=dict(arrowstyle="->", color="blue", lw=2, alpha=0.8))
        ax_hov.text((start_year+end_year)/2, (start_lat+end_lat)/2 - 0.5, "Wave", rotation=40, color="blue", fontweight='bold', fontsize=9, ha='center')
    
    # =========================================================================
    # Panel B: Phase-Latitude Regression
    # =========================================================================
    ax_phase = fig.add_subplot(gs[0, 1])
    
    target_reefs = period_results[(period_results['primary_significant'] == True) & (period_results['primary_period'].between(15, 30))].copy()
    target_reefs = target_reefs.merge(coords, on='reef_name', how='inner')
    
    phases, latitudes_phase = [], []
    for _, row in target_reefs.iterrows():
        reef = row['reef_name']
        period = row['primary_period']
        lat = np.abs(row['lat'])
        rs = residual_df[residual_df['reef_name'] == reef].sort_values('year')
        if len(rs) < 5: continue
        t = rs['year'].to_numpy(dtype=float)
        y = rs['residual'].to_numpy(dtype=float)
        if np.std(y) <= 1e-12: continue
        omega = 2 * np.pi / period
        def sine_model(t, A, phi): return A * np.sin(omega * t + phi)
        try:
            popt, _ = optimize.curve_fit(sine_model, t, y, p0=[np.std(y), 0], maxfev=2000)
            phases.append(np.angle(np.exp(1j * popt[1])))
            latitudes_phase.append(lat)
        except: continue
    
    if len(phases) >= 3:
        lats = np.array(latitudes_phase)
        phis = np.array(phases)
        sort_idx = np.argsort(lats)
        lats_sorted, phis_sorted = lats[sort_idx], np.unwrap(phis[sort_idx])
        res = stats.linregress(lats_sorted, phis_sorted)
        mean_period = target_reefs['primary_period'].mean()
        
        # p-value
        sig_str = "***" if res.pvalue < 0.001 else "**" if res.pvalue < 0.01 else "*" if res.pvalue < 0.05 else ""
        
        speed_km_per_year = np.nan
        speed_ci_min, speed_ci_max = np.nan, np.nan
        if res.slope != 0:
            years_per_degree = abs(res.slope * mean_period / (2 * np.pi))
            speed_km_per_year = 111.0 / years_per_degree if years_per_degree > 0 else np.nan
            
            n_obs = len(phases)
            t_crit = stats.t.ppf(0.975, df=n_obs-2)
            slope_err = res.stderr
            slope_min = abs(res.slope) - t_crit * slope_err
            slope_max = abs(res.slope) + t_crit * slope_err
            
            speed_ci_min = 111.0 / (slope_max * mean_period / (2 * np.pi)) if slope_max > 0 else np.nan
            speed_ci_max = 111.0 / (slope_min * mean_period / (2 * np.pi)) if slope_min > 0 else np.inf
        
        sc = ax_phase.scatter(lats_sorted, phis_sorted, c=lats_sorted, cmap='coolwarm', s=60, edgecolors='k', zorder=5)
        fit_x = np.linspace(lats_sorted.min(), lats_sorted.max(), 100)
        fit_y = res.intercept + res.slope * fit_x
        ax_phase.plot(fit_x, fit_y, 'k--', lw=1.5, label=f'$R^2$={res.rvalue**2:.2f}{sig_str}')
        
        # CI Band
        x_mean = np.mean(lats_sorted)
        S_xx = np.sum((lats_sorted - x_mean)**2)
        s_err = np.sqrt(np.sum((phis_sorted - (res.intercept + res.slope * lats_sorted))**2) / (n_obs - 2))
        ci_band = t_crit * s_err * np.sqrt(1/n_obs + (fit_x - x_mean)**2 / S_xx)
        ax_phase.fill_between(fit_x, fit_y - ci_band, fit_y + ci_band, color="#a6cee3", alpha=0.3, zorder=3)
        
        ax_phase.set_xlabel('Latitude (°S)', fontsize=10, fontweight='bold')
        ax_phase.set_ylabel('Phase (radians)', fontsize=10, fontweight='bold')
        
        ci_text = f"[{speed_ci_min:.0f}, {speed_ci_max:.0f}]" if not np.isinf(speed_ci_max) else "> 0"
        textstr = f'Speed: {speed_km_per_year:.2f} km/yr\n95% CI: {ci_text}\nn = {len(phases)}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax_phase.text(0.05, 0.95, textstr, transform=ax_phase.transAxes, fontsize=8, verticalalignment='top', bbox=props)
        ax_phase.legend(loc='lower right', fontsize=8)

        pd.DataFrame([
            {
                'n_reefs': len(phases),
                'mean_period_years': float(mean_period),
                'speed_km_per_year': float(speed_km_per_year),
                'r_squared': float(res.rvalue**2),
                'p_value': float(res.pvalue),
                'period_filter_min_years': 15.0,
                'period_filter_max_years': 30.0,
            }
        ]).to_csv(TABLES_DIR / 'figure5_summary.csv', index=False)
    
    # =========================================================================
    # Panel C: Current Speed Anomaly (Map)
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
        ax_cur.set_xlabel('')
        ax_cur.set_ylabel('')
        
        ax_cur.set_xticks(np.arange(142, 156, 4), crs=ccrs.PlateCarree())
        ax_cur.set_yticks(np.arange(-25, -9, 5), crs=ccrs.PlateCarree())
        ax_cur.xaxis.set_major_formatter(LongitudeFormatter())
        ax_cur.yaxis.set_major_formatter(LatitudeFormatter())
        ax_cur.tick_params(axis='both', which='major', direction='out', length=4, width=1, labelsize=9)
        
        ax_cur.text(0.95, 0.05, 'Blue = Retention\nRed = Transport', transform=ax_cur.transAxes, ha='right', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax_cur.text(0.05, 0.95, "Mean Flow:\nNeg IPO: 0.112 m/s\nPos IPO: 0.126 m/s\n(11% slower)", transform=ax_cur.transAxes, ha='left', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
    
    # =========================================================================
    # Panel D: Regional Period Boxplot
    # =========================================================================
    ax_box = fig.add_subplot(gs[1, 1])
    
    try:
        model_results = load_model_results()
        if 'region' not in period_results.columns:
            reg_col = 'region' if 'region' in model_results.columns else 'nrm_region'
            period_with_region = period_results.merge(model_results[['reef_name', reg_col]], on='reef_name', how='left')
            period_with_region = period_with_region.rename(columns={reg_col: 'nrm_region'})
        else:
            period_with_region = period_results.rename(columns={'region': 'nrm_region'})
    except:
        reef_regions = cots_with_coords[['reef_name', 'nrm_region']].drop_duplicates()
        period_with_region = period_results.merge(reef_regions, on='reef_name', how='inner')
    
    sig = period_with_region[period_with_region['primary_significant'] == True]
    
    box_data, box_labels = [], []
    box_colors = [COLORS['primary'], COLORS['success'], COLORS['accent'], COLORS['secondary'], COLORS['purple'], '#8c564b']
    
    for reg in REGION_ORDER:
        reg_data = sig[sig['nrm_region'] == reg]['primary_period'].dropna().values
        if len(reg_data) > 0:
            box_data.append(reg_data)
            box_labels.append(reg)
    
    if box_data:
        bp = ax_box.boxplot(box_data, patch_artist=True, tick_labels=[l.replace(' ', '\n') for l in box_labels])
        for patch, color in zip(bp['boxes'], box_colors[:len(box_data)]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax_box.set_ylabel('Dominant Period (years)', fontsize=10, fontweight='bold')
        ax_box.set_xlabel('NRM Region', fontsize=10, fontweight='bold')
        ax_box.axhspan(15, 30, color=COLORS['secondary'], alpha=0.1, label='IPO (15-30y)')
        ax_box.axhline(20, color=COLORS['secondary'], linestyle='--', lw=1, alpha=0.5)
        ax_box.legend(loc='upper right', fontsize=8)
        # 用回车代替斜体更适配A4拥挤的宽度
        plt.setp(ax_box.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # =========================================================================
    # Axis alignment - A4 safe calculation
    # =========================================================================
    fig.canvas.draw()
    pos_c = ax_cur.get_position()
    pos_d = ax_box.get_position()
    pos_a = ax_hov.get_position()
    pos_b = ax_phase.get_position()
    
    # A4下 y 轴更长，我们直接把 A, B 放在 C, D 正上方 0.08 处
    y_new = pos_c.y0 + pos_c.height + 0.08
    
    ax_box.set_position([pos_d.x0, pos_c.y0, pos_c.width, pos_c.height])
    ax_hov.set_position([pos_c.x0, y_new, pos_c.width, pos_c.height])
    ax_phase.set_position([pos_d.x0, y_new, pos_c.width, pos_c.height])
    
    # 手动建立独立的 Colorbar 以杜绝 SVG 文件“丢东西”的错误
    cb_w = 0.015
    
    # 为 Hovmöller 增加图例
    cax_a = fig.add_axes([pos_c.x1 + 0.015, y_new + 0.1*pos_c.height, cb_w, 0.8*pos_c.height])
    cbar_a = plt.colorbar(img, cax=cax_a)
    cbar_a.set_label('log(COTS + 1)', rotation=270, labelpad=12, fontsize=8)
    
    # 为海流速度图增加图例
    if im_c is not None:
        cax_c = fig.add_axes([pos_c.x1 + 0.015, pos_c.y0 + 0.1*pos_c.height, cb_w, 0.8*pos_c.height])
        cbar_c = plt.colorbar(im_c, cax=cax_c)
        cbar_c.set_label('Speed Anomaly ($m\\ s^{-1}$)', fontsize=8)
    
    output_png = OUTPUT_DIR / 'Figure3_Propagation_Dynamics.png'
    output_svg = OUTPUT_DIR / 'Figure3_Propagation_Dynamics.svg'
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.savefig(output_svg, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    create_figure3()
