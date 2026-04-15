# -*- coding: utf-8 -*-
"""
Optimized Study Area Map Generator
1. 缁熶竴绀佸尯鐐瑰ぇ灏?
2. 鏍规嵁鍘嗗彶鏈€澶?COTS 瀵嗗害鐫€鑹?(Established=0.2, Severe=1.0)
3. 娣诲姞鐝婄憵绀佸湴鐞嗕俊鎭潰 (Polygon)
4. NRM 鍖哄煙鍒嗙晫绾块噰鐢ㄨ櫄绾?
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

warnings.filterwarnings('ignore')

# 璁剧疆缁樺浘椋庢牸
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "reef_surveys"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 瀹氫箟 COTS 鍒嗙骇棰滆壊
# Severe (>=1.0): 娣辩孩; Established ([0.2, 1.0)): 姗欑孩; Potential ([0.1, 0.2)): 姗? Non-outbreak ([0.01, 0.1)): 榛? No COTS (<0.01): 娣辩豢
COTS_UNIT = r'($COTS\ tow^{-1}$)'
COTS_LEVELS = [
    (1.0, '#b30000', 'Severe Outbreak (鈮?.0)'),
    (0.2, '#e34a33', 'Established Outbreak (0.2-1.0)'),
    (0.1, '#fdbb84', 'Potential/Incident (0.1-0.2)'),
    (0.01, '#fee8c8', 'Non-outbreak (0.01-0.1)'),
    (0.0, '#238b45', 'No COTS (<0.01)')
]

def get_cots_color(val):
    for thresh, color, label in COTS_LEVELS:
        if val >= thresh:
            return color
    return '#238b45'

REGION_ORDER = ['Cape York', 'Wet Tropics', 'Burdekin', 'Mackay Whitsunday', 'Fitzroy', 'Burnett Mary']

def parse_dms(dms_str):
    """瑙ｆ瀽搴﹀垎绉掑潗鏍?""
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
    dms_str = dms_str.strip().upper()
    match = re.match(r"(.*[SN])\s*(.*[EW])", dms_str)
    if match:
        return parse_part(match.group(1)), parse_part(match.group(2))
    return None, None

def load_reef_data():
    """鑱氬悎姣忎釜绀佸尯鐨勫潗鏍囧拰鏈€澶?COTS 瀵嗗害"""
    print("Aggregating reef data...")
    # 鍔犺浇鍧愭爣
    coord_file = PROJECT_ROOT / "data" / "spatial" / "site_lon_lat.csv"
    coord_df = pd.read_csv(coord_file, encoding='gbk')
    coord_df.columns = ['site', 'coords'] + list(coord_df.columns[2:])
    
    # 瑙ｆ瀽鍧愭爣
    coords_dict = {}
    for _, row in coord_df.iterrows():
        lat, lon = parse_dms(row['coords'])
        if lat is not None:
            coords_dict[row['site']] = (lat, lon)

    # 鑱氬悎 COTS
    csv_files = list(DATA_DIR.glob('*_cots_modelled_*.csv'))
    aggregated = []
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # 澶勭悊鍒楀悕宸紓
            cots_col = 'cotsptow' if 'cotsptow' in df.columns else 'cots_density'
            if cots_col not in df.columns: continue
            
            reef_name = f.name.split('_Manta Tow_cots_modelled_')[0]
            if reef_name not in coords_dict: continue
            
            max_cots = df[cots_col].max()
            region = df.get('nrm_region', pd.Series(['Unknown'])).iloc[0]
            
            lat, lon = coords_dict[reef_name]
            aggregated.append({
                'reef_name': reef_name,
                'lat': lat,
                'lon': lon,
                'region': region,
                'max_cots': max_cots
            })
        except: continue
        
    return pd.DataFrame(aggregated)

import geopandas as gpd

def plot_optimized_map():
    df = load_reef_data()
    print(f"Plotting {len(df)} reefs with official GBRMPA spatial data...")
    
    # 瀹氫箟瀹樻柟鏁版嵁璺緞
    spatial_base = PROJECT_ROOT / "data" / "spatial"
    reef_shp = spatial_base / "GBR_Features" / "Great_Barrier_Reef_Features.shp"
    nrm_shp = spatial_base / "NRM_Regions" / "NRM_Terrestrial_and_Marine_Regions_GBR_GDA20.shp"
    
    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    
    # 1. 鍩虹瑕佺礌 (鍔犳繁鑳屾櫙棰滆壊浠ュ鍔犲姣斿害)
    ax.add_feature(cfeature.OCEAN, facecolor='#a9d0e4', alpha=0.9, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#F9E0B2', edgecolor='#404040', linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#263238', zorder=6)
    
    # 2. 鍔犺浇骞剁粯鍒跺畼鏂圭強鐟氱 (Polygon) - 鎹㈢敤鐏拌壊锛寊order 璁句负杈冧綆
    if reef_shp.exists():
        print("Loading official Reef Features...")
        reefs_gdf = gpd.read_file(reef_shp)
        if reefs_gdf.crs and reefs_gdf.crs != "EPSG:4326":
            reefs_gdf = reefs_gdf.to_crs("EPSG:4326")
        
        # 缁樺埗娴撶伆鑹茬強鐟氱
        reefs_gdf.plot(ax=ax, facecolor='#757575', edgecolor='#616161', linewidth=0.1, alpha=0.8, zorder=2, transform=ccrs.PlateCarree())
    else:
        print(f"Warning: Reef SHP not found at {reef_shp}")

    # 3. 鍔犺浇骞剁粯鍒剁姸鎬?NRM 鍖哄煙鍒嗙晫绾?(Official Boundaries)
    if nrm_shp.exists():
        print("Loading official NRM boundaries...")
        nrm_gdf = gpd.read_file(nrm_shp)
        if nrm_gdf.crs and nrm_gdf.crs != "EPSG:4326":
            nrm_gdf = nrm_gdf.to_crs("EPSG:4326")
        
        # 缁樺埗铏氱嚎杈圭晫
        nrm_gdf.boundary.plot(ax=ax, color='#555555', linestyle=(0, (5, 5)), linewidth=1.0, zorder=3, alpha=0.7, transform=ccrs.PlateCarree())
        
        # 娣诲姞鍖哄煙鏍囨敞 (鍙€?
        for _, row in nrm_gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(centroid.x, centroid.y, row.get('REGION', ''), fontsize=9, fontstyle='italic', ha='center', zorder=4, transform=ccrs.PlateCarree(), alpha=0.6)
    else:
        print(f"Warning: NRM SHP not found at {nrm_shp}")

    # 4. 缁樺埗绀佸尯鐐?(淇濇寔缁熶竴澶у皬 s=50)
    for _, row in df.iterrows():
        color = get_cots_color(row['max_cots'])
        ax.scatter(row['lon'], row['lat'], c=color, s=50, edgecolors='k', linewidths=0.5, 
                   zorder=10, transform=ccrs.PlateCarree())
        
    # 5. 鎸囧寳閽堜笌鍒诲害
    ax.set_xticks(np.arange(142, 156, 4), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-25, -9, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    # 鎸囧寳閽?(浼樺寲浣嶇疆)
    ax.text(143, -10.5, 'N', fontsize=14, fontweight='bold', ha='center', transform=ccrs.PlateCarree())
    ax.annotate('', xy=(143, -10.5), xytext=(143, -11.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                transform=ccrs.PlateCarree())
    
    # 6. 鍥句緥
    legend_elements = []
    for thresh, color, label in COTS_LEVELS:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                          markerfacecolor=color, markersize=10, markeredgecolor='k'))
    
    ax.legend(handles=legend_elements, loc='upper right', title=f'Max COTS Density {COTS_UNIT}',
              fontsize=10, title_fontsize=11, framealpha=0.9)
    
    # 姣斾緥灏?(浼樺寲)
    # 浣跨敤缁樺浘鍖哄煙搴曢儴
    ax_scale = fig.add_axes([0.65, 0.12, 0.15, 0.02])
    ax_scale.plot([0, 100], [0.5, 0.5], 'k-', linewidth=2)
    ax_scale.text(50, 1.2, '200 km', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_scale.axis('off')

    # 绉婚櫎鏍囬 (鎸夌収鐢ㄦ埛瑕佹眰)
    # plt.title(...) 
    
    primary_png = OUTPUT_DIR / 'Figure1_Study_Area.png'
    primary_svg = OUTPUT_DIR / 'Figure1_Study_Area.svg'
    compat_png = OUTPUT_DIR / 'optimized_Figure1_Study_Area_v2.png'
    compat_svg = OUTPUT_DIR / 'optimized_Figure1_Study_Area_v2.svg'

    plt.savefig(primary_png, bbox_inches='tight', dpi=300)
    plt.savefig(primary_svg, bbox_inches='tight')
    plt.savefig(compat_png, bbox_inches='tight', dpi=300)
    plt.savefig(compat_svg, bbox_inches='tight')
    plt.close()
    print(f"Done! Saved to {primary_png}")

if __name__ == "__main__":
    plot_optimized_map()
