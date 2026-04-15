# -*- coding: utf-8 -*-
"""
Supplementary Figure: Hydrological Basis of IPO-COTS Cascade
(a) Rainfall Anomaly Map (Neg IPO - Pos IPO)
(b) River Runoff Boxplots for key GBR rivers
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings

warnings.filterwarnings('ignore')

# Set aesthetic parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data"
RAINFALL_DIR = BASE_DIR / "climate" / "rainfall"
RUNOFF_DIR = BASE_DIR / "hydrology" / "runoff"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

IPO_FILE = BASE_DIR / "climate" / "ipo_index.csv"

# =============================================================================
# 1. IPO Phase Classification
# =============================================================================

def get_ipo_phases():
    try:
        # Try to read with header=0 to get column names automatically
        df = pd.read_csv(IPO_FILE, encoding='utf-8')
    except:
        df = pd.read_csv(IPO_FILE, encoding='gbk')
    
    # Ensure year column is numeric
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    df = df.dropna(subset=[df.columns[0]])
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    
    # Filter for years >= 1980
    df = df[df.iloc[:, 0] >= 1980]
    
    # Calculate annual mean for the 12 month columns (columns 1 to 13)
    # Ensure they are numeric
    for col in df.columns[1:13]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['ipo_annual'] = df.iloc[:, 1:13].mean(axis=1)
    df = df.dropna(subset=['ipo_annual'])
    
    mean_val = df['ipo_annual'].mean()
    std_val = df['ipo_annual'].std()
    
    year_col = df.columns[0]
    neg_years = df[df['ipo_annual'] < (mean_val - 0.5 * std_val)][year_col].tolist()
    pos_years = df[df['ipo_annual'] > (mean_val + 0.5 * std_val)][year_col].tolist()
    
    print(f"Negative IPO years: {neg_years}")
    print(f"Positive IPO years: {pos_years}")
    
    return neg_years, pos_years, df[[year_col, 'ipo_annual']]

# =============================================================================
# 2. Rainfall Anomaly Analysis
# =============================================================================

def process_rainfall(neg_years, pos_years):
    print("Processing rainfall NetCDF files...")
    all_years = sorted([int(f.name.split('.')[0]) for f in RAINFALL_DIR.glob('*.monthly_rain.nc')])
    
    neg_maps = []
    pos_maps = []
    
    # GBR Extent for cropping
    lat_range = slice(-10, -25) # North to South for SILO indexing
    lon_range = slice(142, 154)
    
    for year in all_years:
        fpath = RAINFALL_DIR / f"{year}.monthly_rain.nc"
        ds = xr.open_dataset(fpath)
        
        # Crop and aggregate to annual total
        # Variable name is 'monthly_rain'
        # Note: SILO lats are usually -44 to -10 (South to North or vice versa)
        # Check order: ds.lat[0] is -44, ds.lat[-1] is -10. So it is strictly increasing.
        gbr_ds = ds['monthly_rain'].sel(lat=slice(-25, -10), lon=slice(142, 154))
        annual_rain = gbr_ds.sum(dim='time')
        
        if year in neg_years:
            neg_maps.append(annual_rain)
        elif year in pos_years:
            pos_maps.append(annual_rain)
        
        ds.close()

    if not neg_maps or not pos_maps:
        print("Error: Could not find valid years for both phases.")
        return None
    
    mean_neg = xr.concat(neg_maps, dim='year').mean(dim='year')
    mean_pos = xr.concat(pos_maps, dim='year').mean(dim='year')
    
    anomaly = mean_neg - mean_pos
    return anomaly

# =============================================================================
# 3. River Runoff Analysis
# =============================================================================

def load_river_data(river_name, neg_years, pos_years):
    river_path = RUNOFF_DIR / river_name
    csv_file = list(river_path.glob("*.csv"))[0]
    
    # Read ignoring comments (starting with #)
    df = pd.read_csv(csv_file, comment='#', names=['timestamp', 'value', 'quality', 'interpolation'])
    
    # Parse timestamp (e.g., 1980-01-01T00:00:00.000+10:00)
    df['year'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True).dt.year
    
    # Calculate annual mean discharge (cumecs)
    annual_mean = df.groupby('year')['value'].mean().reset_index()
    
    neg_data = annual_mean[annual_mean['year'].isin(neg_years)]['value'].values
    pos_data = annual_mean[annual_mean['year'].isin(pos_years)]['value'].values
    
    return neg_data, pos_data

# =============================================================================
# 4. Main Plotting
# =============================================================================

def generate_figure():
    neg_years, pos_years, ipo_df = get_ipo_phases()
    rain_anomaly = process_rainfall(neg_years, pos_years)
    
    # River data
    rivers = ['Pascoe', 'Burdekin', 'Fitzroy']
    river_results = {}
    for r in rivers:
        river_results[r] = load_river_data(r, neg_years, pos_years)
    
    # Plotting
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
    
    # --- (a) Rainfall Anomaly Map ---
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_map.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    
    ax_map.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='#404040', linewidth=0.6)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    if rain_anomaly is not None:
        vmax = np.nanmax(np.abs(rain_anomaly.values)) * 0.9 # Slightly more range
        # Use RdBu_r so red = wetter in Neg IPO (positive anomaly)
        im = rain_anomaly.plot(ax=ax_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax, 
                               add_colorbar=False, transform=ccrs.PlateCarree())
        
        cbar = plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar.set_label('Rainfall Anomaly (mm/year)', fontweight='bold')
        
        ax_map.set_title('(a) Rainfall Anomaly (Negative - Positive IPO)', loc='left', fontweight='bold', fontsize=14, pad=15)
        
        # Ticks
        ax_map.set_xticks(np.arange(142, 155, 4), crs=ccrs.PlateCarree())
        ax_map.set_yticks(np.arange(-25, -9, 5), crs=ccrs.PlateCarree())
        ax_map.xaxis.set_major_formatter(LongitudeFormatter())
        ax_map.yaxis.set_major_formatter(LatitudeFormatter())
        
        # Add labels for rivers
        river_coords = {'Pascoe': [143.3, -12.5], 'Burdekin': [147.2, -19.7], 'Fitzroy': [150.5, -23.4]}
        for name, coord in river_coords.items():
            ax_map.plot(coord[0], coord[1], 'ko', markersize=4, transform=ccrs.PlateCarree())
            ax_map.text(coord[0]+0.3, coord[1]-0.2, name, transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold')

    # --- (b) River Runoff Boxplots ---
    ax_box = fig.add_subplot(gs[0, 1])
    
    plot_data = []
    labels = []
    for r in rivers:
        neg, pos = river_results[r]
        plot_data.append(pos) # Order: Pos IPO first
        plot_data.append(neg) # Then Neg IPO
        labels.extend([f'{r}\nPos IPO', f'{r}\nNeg IPO'])
    
    bp = ax_box.boxplot(plot_data, patch_artist=True, labels=labels, 
                        showfliers=False, widths=0.6)
    
    colors = ['#fde0dd', '#7a0177'] * 3 # Pink for Pos, Purple for Neg (matches your theme)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
        
    ax_box.set_ylabel('Annual Mean Discharge ($m^3/s$)', fontsize=12, fontweight='bold')
    ax_box.set_title('(b) River Runoff Sensitivity to IPO Phase', loc='left', fontweight='bold', fontsize=14, pad=15)
    ax_box.spines[['top', 'right']].set_visible(False)
    
    # Significance markers
    for i, r in enumerate(rivers):
        pos = plot_data[i*2]
        neg = plot_data[i*2 + 1]
        stat, p_val = stats.mannwhitneyu(neg, pos, alternative='greater')
        print(f"River: {r}, Pos median: {np.median(pos):.2f}, Neg median: {np.median(neg):.2f}, p-value (MWU): {p_val:.4f}")
        
        max_y = max(np.max(neg), np.max(pos)) if len(neg)>0 and len(pos)>0 else 100
        y_pos = max_y * 1.05
        
        if p_val < 0.05:
            sig_str = '*'
            if p_val < 0.01: sig_str = '**'
            if p_val < 0.001: sig_str = '***'
            
            x1, x2 = i*2 + 1, i*2 + 2
            ax_box.plot([x1, x1, x2, x2], [y_pos, y_pos*1.05, y_pos*1.05, y_pos], lw=1, color='black')
            ax_box.text((x1+x2)*0.5, y_pos*1.05, sig_str, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'FigureS2_Hydrology_Mechanism.png', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / 'FigureS2_Hydrology_Mechanism.svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: FigureS2_Hydrology_Mechanism.png and .svg in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_figure()
