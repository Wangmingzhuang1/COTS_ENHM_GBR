# -*- coding: utf-8 -*-
"""
Combined COTS Spatial and Temporal Overview (Top Journal Quality)
Panel (a): Study Area Map with NRM Regions and Reference Cities
Panel (b): Temporal Distribution (Aligned & Boxed)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path
import re
import warnings
import geopandas as gpd

warnings.filterwarnings('ignore')

# Global Styles
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "reef_surveys"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared Colors and Levels
COTS_UNIT = r'($COTS\ tow^{-1}$)'
COTS_LEVEL_INFO = [
    ('Severe Outbreak', '#b30000', 'Severe (鈮?.0)'),
    ('Established Outbreak', '#e34a33', 'Established (0.2-1.0)'),
    ('Potential/Incident', '#fdbb84', 'Potential (0.1-0.2)'),
    ('Non-outbreak', '#fee8c8', 'Non-outbreak (0.01-0.1)'),
    ('No COTS', '#238b45', 'No COTS (<0.01)')
]

# 鍙傝€冨煄甯傚潗鏍?REFERENCE_CITIES = [
    {'name': 'Cairns', 'lat': -16.9186, 'lon': 145.7781},
    {'name': 'Townsville', 'lat': -19.2588, 'lon': 146.8169},
    {'name': 'Mackay', 'lat': -21.1412, 'lon': 149.1861},
    {'name': 'Rockhampton', 'lat': -23.3750, 'lon': 150.5100}
]

def get_status(val):
    try:
        fval = float(val)
        if fval >= 1.0: return 'Severe Outbreak'
        if fval >= 0.2: return 'Established Outbreak'
        if fval >= 0.1: return 'Potential/Incident'
        if fval >= 0.01: return 'Non-outbreak'
        return 'No COTS'
    except: return 'No COTS'

def parse_dms(dms_str):
    if pd.isna(dms_str) or not dms_str: return None, None
    def parse_part(part_str):
        nums = re.findall(r"(\d+(?:\.\d+)?)", part_str)
        direction = re.search(r"([NSEW])", part_str)
        if not direction: return None
        direction = direction.group(1)
        vals = [0.0, 0.0, 0.0]
        if len(nums) >= 1: vals[0] = float(nums[0])
        if len(nums) >= 2: vals[1] = float(nums[1])
        if len(nums) >= 3: vals[2] = float(nums[2])
        deg = vals[0] + vals[1]/60 + vals[2]/3600
        if direction in ['S', 'W']: deg = -deg
        return deg
    match = re.match(r"(.*[SN])\s*(.*[EW])", dms_str.upper())
    if match: return parse_part(match.group(1)), parse_part(match.group(2))
    return None, None

def load_spatial_data():
    coord_file = PROJECT_ROOT / "data" / "spatial" / "site_lon_lat.csv"
    coord_df = pd.read_csv(coord_file, encoding='gbk')
    coord_df.columns = ['site', 'coords'] + list(coord_df.columns[2:])
    coords_dict = {}
    for _, row in coord_df.iterrows():
        lat, lon = parse_dms(row['coords'])
        if lat is not None: coords_dict[row['site']] = (lat, lon)
    csv_files = list(DATA_DIR.glob('*cots*.csv'))  # DATA_DIR is reef_surveys
    aggregated = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            cots_col = 'cotsptow' if 'cotsptow' in df.columns else 'cots_density'
            if cots_col not in df.columns: continue
            reef_name = f.name.split('_Manta Tow_')[0]
            if reef_name not in coords_dict: continue
            max_cots = df[cots_col].max()
            aggregated.append({'lat': coords_dict[reef_name][0], 'lon': coords_dict[reef_name][1], 'max_cots': max_cots})
        except: continue
    return pd.DataFrame(aggregated)

def load_temporal_data():
    csv_files = list(DATA_DIR.glob('*cots*.csv'))  # DATA_DIR is reef_surveys
    all_data = []
    for f in csv_files:
        try:
            try: df = pd.read_csv(f, encoding='utf-8')
            except: df = pd.read_csv(f, encoding='gbk')
            cols = {c.lower(): c for c in df.columns}
            year_col = next((cols[c] for c in ['year', 'survey_year', 'decimalyear'] if c in cols), None)
            cots_col = next((cols[c] for c in ['cotsptow', 'cots_density', 'density'] if c in cols), None)
            if year_col is None or cots_col is None: continue
            df_slice = df[[year_col, cots_col]].dropna()
            if 'decimalyear' in year_col.lower():
                df_slice['yr'] = df_slice[year_col].astype(float).apply(np.floor).astype(int)
            else:
                df_slice['yr'] = df_slice[year_col].astype(float).astype(int)
            yearly_max = df_slice.groupby('yr')[cots_col].max().reset_index()
            yearly_max['status'] = yearly_max[cots_col].apply(get_status)
            all_data.append(yearly_max[['yr', 'status']])
        except: continue
    if not all_data: return None
    master_df = pd.concat(all_data)
    stats = master_df.groupby(['yr', 'status']).size().unstack(fill_value=0)
    order = [info[0] for info in reversed(COTS_LEVEL_INFO)]
    for col in order:
        if col not in stats.columns: stats[col] = 0
    return stats[order]

def plot_combined():
    fig = plt.figure(figsize=(10, 14))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.15)

    # --- Panel (a): Spatial Map ---
    ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax1.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    
    ax1.add_feature(cfeature.OCEAN, facecolor='#a9d0e4', alpha=0.9, zorder=0)
    ax1.add_feature(cfeature.LAND, facecolor='#F9D493', edgecolor='#404040', linewidth=0.5, zorder=5)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#263238', zorder=6)

    spatial_base = PROJECT_ROOT / "data" / "spatial"
    reef_shp = spatial_base / "GBR_Features" / "Great_Barrier_Reef_Features.shp"
    nrm_shp = spatial_base / "NRM_Regions" / "NRM_Terrestrial_and_Marine_Regions_GBR_GDA20.shp"
    
    if reef_shp.exists():
        reefs_gdf = gpd.read_file(reef_shp)
        if reefs_gdf.crs != "EPSG:4326": reefs_gdf = reefs_gdf.to_crs("EPSG:4326")
        reefs_gdf.plot(ax=ax1, facecolor='#757575', edgecolor='#616161', linewidth=0.05, alpha=0.8, zorder=2)
    
    if nrm_shp.exists():
        nrm_gdf = gpd.read_file(nrm_shp)
        if nrm_gdf.crs != "EPSG:4326": nrm_gdf = nrm_gdf.to_crs("EPSG:4326")
        nrm_gdf.boundary.plot(ax=ax1, color='#555555', linestyle=(0, (5, 5)), linewidth=0.8, zorder=3, alpha=0.6)
        
        # 鏍囨敞鍖哄煙鍚嶇О (鎻愬崌鑷虫渶椤跺眰 zorder=25)
        from matplotlib.patheffects import withStroke
        for _, row in nrm_gdf.iterrows():
            centroid = row.geometry.centroid
            name = row.get('REGION', '')
            y_off = 0
            if name == 'Cape York': y_off = 1.0
            if name == 'Burnett Mary': y_off = -0.5
            
            ax1.text(centroid.x - 0.5, centroid.y + y_off, name, fontsize=9.5, fontweight='bold', 
                    ha='right', va='center', zorder=25, transform=ccrs.PlateCarree(), color='black',
                    path_effects=[withStroke(linewidth=2, foreground='white', alpha=0.6)])

    # 缁樺埗鍙傝€冨煄甯?(鎻愬崌鑷?zorder=26)
    for city in REFERENCE_CITIES:
        ax1.plot(city['lon'], city['lat'], 'ks', markersize=4, zorder=26, transform=ccrs.PlateCarree())
        ax1.text(city['lon'] - 0.2, city['lat'], city['name'], fontsize=8.5, fontweight='bold',
                ha='right', va='center', zorder=26, transform=ccrs.PlateCarree(), color='black',
                path_effects=[withStroke(linewidth=1.5, foreground='white', alpha=0.7)])

    df_spa = load_spatial_data()
    status_map = {info[0]: info[1] for info in COTS_LEVEL_INFO}
    for _, row in df_spa.iterrows():
        st = get_status(row['max_cots'])
        ax1.scatter(row['lon'], row['lat'], c=status_map[st], s=35, edgecolors='k', linewidths=0.4, zorder=10)

    ax1.set_xticks(np.arange(142, 156, 4), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(-25, -9, 5), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(LongitudeFormatter())
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    
    # 鏍囩 a
    ax1.text(0.02, 0.98, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    
    # 鎸囧寳閽?    ax1.text(143.2, -11, 'N', fontsize=12, fontweight='bold', ha='center', transform=ccrs.PlateCarree())
    ax1.annotate('', xy=(143.2, -11), xytext=(143.2, -12), arrowprops=dict(arrowstyle='->', lw=1.5), transform=ccrs.PlateCarree())

    # 姣斾緥灏?    scale_lon_start, scale_lon_end = 142.8, 144.6
    scale_lat = -24.0
    ax1.plot([scale_lon_start, scale_lon_end], [scale_lat, scale_lat], color='black', linewidth=2, transform=ccrs.PlateCarree(), zorder=15)
    ax1.text((scale_lon_start + scale_lon_end)/2, scale_lat + 0.3, '200 km', transform=ccrs.PlateCarree(), 
             ha='center', va='bottom', fontsize=9, fontweight='bold', zorder=16)

    # 鍥句緥
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=info[2], markerfacecolor=info[1], markersize=8, markeredgecolor='k') for info in COTS_LEVEL_INFO]
    ax1.legend(handles=legend_elements, loc='upper right', title=f'Max COTS Density\n{COTS_UNIT}', fontsize=9, title_fontsize=10, frameon=True, framealpha=0.9)

    # --- Panel (b): Temporal Plot ---
    fig.canvas.draw()
    bbox1 = ax1.get_position()
    ax2 = fig.add_axes([bbox1.x0, 0.1, bbox1.width, 0.15]) 
    
    stats = load_temporal_data()
    if stats is not None:
        all_years = np.arange(1983, 2026)
        stats = stats.reindex(all_years, fill_value=0)
        cum_stats = stats.cumsum(axis=1)
        colors = [info[1] for info in reversed(COTS_LEVEL_INFO)]
        last_y = np.zeros(len(stats))
        for i, col in enumerate(stats.columns):
            new_y = cum_stats[col].values
            ax2.fill_between(stats.index, last_y, new_y, step='mid', color=colors[i], alpha=0.9, edgecolor='white', linewidth=0.2)
            last_y = new_y
        total_surveyed = stats.sum(axis=1)
        ax2.step(stats.index, total_surveyed, where='mid', color='black', alpha=0.9, linewidth=1.2)
        
        ax2.set_xlabel('Year', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Surveyed Reefs', fontsize=10, fontweight='bold')
        ax2.set_ylim(0, total_surveyed.max() * 1.05)
        ax2.set_xlim(1982.5, 2025.5)
        ax2.set_xticks(np.arange(1985, 2026, 5))
        
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            
        ax2.text(0.02, 0.92, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

    # 淇濆瓨
    primary_png = OUTPUT_DIR / 'Figure1_Combined_COTS_Overview.png'
    primary_svg = OUTPUT_DIR / 'Figure1_Combined_COTS_Overview.svg'
    compat_png = OUTPUT_DIR / 'combined_cots_overview_v2.png'
    compat_svg = OUTPUT_DIR / 'combined_cots_overview_v2.svg'

    plt.savefig(primary_png, dpi=400, bbox_inches='tight')
    plt.savefig(primary_svg, bbox_inches='tight')
    plt.savefig(compat_png, dpi=400, bbox_inches='tight')
    plt.savefig(compat_svg, bbox_inches='tight')
    plt.close()
    print(f"Final maps with labels saved to {primary_png}")

if __name__ == "__main__":
    plot_combined()
