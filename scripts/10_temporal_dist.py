# -*- coding: utf-8 -*-
"""
COTS Temporal Distribution Plot
灞曠ず澶у牎绀佽寖鍥村唴涓嶅悓鐖嗗彂绛夌骇鐨勭鍖哄勾搴︾垎鍙戠姸鎬佺粺璁★紝鍖呭惈骞村害璋冩煡鎬绘暟鍙宠酱銆?"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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

# 瀹氫箟 COTS 鍒嗙骇棰滆壊 (涓?Figure 1 淇濇寔涓€鑷?
# 椤哄簭锛氫粠搴曢儴寮€濮嬪爢鍙?(No COTS -> Severe)
COTS_LEVEL_INFO = [
    ('Severe Outbreak', '#b30000', 'Severe (鈮?.0)'),
    ('Established Outbreak', '#e34a33', 'Established (0.2-1.0)'),
    ('Potential/Incident', '#fdbb84', 'Potential (0.1-0.2)'),
    ('Non-outbreak', '#fee8c8', 'Non-outbreak (0.01-0.1)'),
    ('No COTS', '#238b45', 'No COTS (<0.01)')
]

def get_status(val):
    try:
        fval = float(val)
        if fval >= 1.0: return 'Severe Outbreak'
        if fval >= 0.2: return 'Established Outbreak'
        if fval >= 0.1: return 'Potential/Incident'
        if fval >= 0.01: return 'Non-outbreak'
        return 'No COTS'
    except:
        return 'No COTS'

def process_temporal_data():
    print("Collecting temporal data from CSV files...")
    csv_files = list(DATA_DIR.glob('*cots*.csv'))
    print(f"Found {len(csv_files)} potential COTS files.")
    
    all_data = []
    
    for f in csv_files:
        try:
            # 灏濊瘯涓嶅悓缂栫爜
            try:
                df = pd.read_csv(f, encoding='utf-8')
            except:
                df = pd.read_csv(f, encoding='gbk')
            
            # 鏌ユ壘鍒?            cols = {c.lower(): c for c in df.columns}
            
            year_col = None
            for c in ['year', 'survey_year', 'decimalyear']:
                if c in cols:
                    year_col = cols[c]
                    break
            
            cots_col = None
            for c in ['cotsptow', 'cots_density', 'density']:
                if c in cols:
                    cots_col = cols[c]
                    break
            
            if year_col is None or cots_col is None:
                continue
            
            # 娓呯悊鏁版嵁
            df_slice = df[[year_col, cots_col]].dropna()
            if df_slice.empty: continue
            
            # 澶勭悊骞翠唤锛屾湁浜涙槸灏忔暟
            if 'decimalyear' in year_col.lower():
                df_slice['yr'] = df_slice[year_col].astype(float).apply(np.floor).astype(int)
            else:
                df_slice['yr'] = df_slice[year_col].astype(float).astype(int)
            
            # 姣忕鍖烘瘡骞磋仛鍚堜竴娆★紙鍙栨渶澶у€硷級
            yearly_max = df_slice.groupby('yr')[cots_col].max().reset_index()
            yearly_max['status'] = yearly_max[cots_col].apply(get_status)
            
            all_data.append(yearly_max[['yr', 'status']])
            if len(all_data) < 5:
                print(f"Processed {f.name}: found {len(yearly_max)} years of data.")
            
        except Exception as e:
            # print(f"Error skipping {f.name}: {e}")
            continue

    if not all_data:
        print("No valid data found.")
        return None
        
    master_df = pd.concat(all_data)
    # 璁＄畻閫忚鍥?    stats = master_df.groupby(['yr', 'status']).size().unstack(fill_value=0)
    
    # 鎸夌収 No COTS 鍒?Severe 鎺掑簭
    order = [info[0] for info in reversed(COTS_LEVEL_INFO)]
    for col in order:
        if col not in stats.columns:
            stats[col] = 0
    
    return stats[order]

def plot_temporal_distribution():
    stats = process_temporal_data()
    if stats is None: return
    
    # 杩囨护鑼冨洿锛屽苟骞虫粦鏃堕棿鐐癸紙纭繚骞翠唤杩炵画锛?    all_years = np.arange(stats.index.min(), 2026) # 鍖呭惈 2025
    stats = stats.reindex(all_years, fill_value=0)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 鍫嗗彔鍥剧粯鍒?    colors = [info[1] for info in reversed(COTS_LEVEL_INFO)]
    labels = [info[2] for info in reversed(COTS_LEVEL_INFO)]
    
    ax1.stackplot(stats.index, stats.T, labels=labels, colors=colors, alpha=0.85, edgecolor='none')
    
    # 璁剧疆宸︿晶鍧愭爣杞?    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Surveyed Reefs', fontsize=12, fontweight='bold')
    
    # 缁樺埗鍙宠酱锛氬勾搴﹁皟鏌ユ€绘暟
    ax2 = ax1.twinx()
    total_surveyed = stats.sum(axis=1)
    ax2.step(stats.index, total_surveyed, where='mid', color='black', alpha=0.6, linewidth=1.5, label='Total Surveyed')
    ax2.set_ylabel('Total Number of Reefs Surveyed', fontsize=12, color='black', alpha=0.8)
    
    # 灏?total surveyed 鏀惧叆鍥句緥
    # 鑾峰彇 handles
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # 椤哄簭锛歋evere 鍦ㄦ渶涓婏紝Total Surveyed 鍦ㄦ渶鍚?    ax1.legend(h1[::-1] + h2, l1[::-1] + l2, loc='upper left', framealpha=0.9, fontsize=10)
    
    # 缇庡寲鏃堕棿杞?    ax1.set_xlim(stats.index.min(), 2025)
    ax1.set_xticks(np.arange(np.floor(stats.index.min()/5)*5, 2026, 5))
    
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    save_path = OUTPUT_DIR / 'cots_temporal_distribution.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.with_suffix('.svg'))
    plt.close()
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    plot_temporal_distribution()
