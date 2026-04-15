# -*- coding: utf-8 -*-
"""
Generate enhm Diagnostic Figures (Optimized for A4 Illustrator)
鐩存帴浣跨敤棰勫厛璁＄畻濂界殑 CSV 鏁版嵁蹇€熺粯鍥撅紝浼樺寲鐢诲竷澶у皬涓?A4 閫傜敤锛?璁惧畾瀛椾綋涓烘柊缃楅┈锛岄噰鐢ㄧ函 SVG 鐭㈤噺瑕佺礌渚夸簬鍚庢湡 Illustrator 杞荤粯銆?"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.timeseries import LombScargle

# 寮哄埗浣跨敤鍚庣
plt.switch_backend('Agg')

# 璁剧疆閰嶈壊鍜?A4 鐭㈤噺鍥惧睘鎬?COLORS = {
    'primary': '#1f77b4',
    'secondary': '#d62728',
    'accent': '#ff7f0e',
    'success': '#2ca02c',
    'purple': '#9467bd',
    'gray': '#7f7f7f'
}

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics"

# 鍒涘缓涓撻棬鏀炬柊璇婃柇鍥剧殑鏂囦欢澶?os.makedirs(OUTPUT_DIR, exist_ok=True)

# 鍖哄煙椤哄簭锛屽拰涓诲浘琛ㄤ繚鎸佷竴鑷?REGION_ORDER = [
    'Cape York', 'Wet Tropics', 'Burdekin', 
    'Mackay Whitsunday', 'Fitzroy', 'Burnett Mary'
]

def load_data():
    """璇诲彇宸叉湁鐨?enhm 璁＄畻缁撴灉"""
    res_df = pd.read_csv(RESULTS_DIR / 'all_residuals.csv')
    model_df = pd.read_csv(RESULTS_DIR / 'model_results.csv')
    period_df = pd.read_csv(RESULTS_DIR / 'period_results.csv')
    if (RESULTS_DIR / 'significant_reefs.csv').exists():
        sig_df = pd.read_csv(RESULTS_DIR / 'significant_reefs.csv')
    else:
        sig_df = period_df[period_df['primary_significant'] == True].merge(model_df[['reef_name', 'region']], on='reef_name')
        
    return res_df, model_df, period_df, sig_df

def plot_figure1(res_df, model_df):
    """Figure 1: Coral-COTS Regression Examples"""
    # 鎵捐娴嬬偣鏈€澶氱殑鍓?4 涓鍖?    counts = res_df.groupby('reef_name').size().sort_values(ascending=False)
    best_reefs = counts.head(4).index.tolist()
    
    # 璁剧疆涓?A4 椤甸潰鐨勪竴鍗婇珮 (瀹?8.27, 楂?5.8)
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 6.5))
    axes = axes.flatten()
    
    for idx, reef in enumerate(best_reefs):
        ax = axes[idx]
        df = res_df[res_df['reef_name'] == reef].copy()
        model = model_df[model_df['reef_name'] == reef].iloc[0]
        
        # 鎹㈢畻鍥?coral_lagged: 涔嬪墠妯″瀷鏄?y=beta*x, 鎵€浠?x=y/beta (濡傛灉beta!=0)
        beta = model['beta_coefficient']
        if beta > 0:
            coral_lag = df['predicted'] / beta
        else:
            coral_lag = np.zeros(len(df))
            
        ax.scatter(coral_lag, df['log_cots'], c=COLORS['primary'], s=40, alpha=0.7, edgecolors='white', lw=0.5)
        
        # 鎷熷悎绾?        x_line = np.linspace(coral_lag.min(), coral_lag.max(), 100) if beta>0 else np.linspace(0, 10, 100)
        y_line = beta * x_line
        ax.plot(x_line, y_line, 'r-', lw=2, label=f"$\\beta$={beta:.4f}, $R^2$={model['r_squared']:.3f}")
        
        ax.set_xlabel('Lagged Coral Cover (%)', fontweight='bold')
        ax.set_ylabel('log(COTS + 1)', fontweight='bold')
        ax.set_title(f"{reef}\n({model['region']})", fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'A4_figure1_coral_cots_regression.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'A4_figure1_coral_cots_regression.svg', bbox_inches='tight')
    plt.close(fig)

def plot_figure2(res_df, period_df):
    """Figure 2: DLS Periodogram Examples"""
    # 閫夊彇鏄捐憲涓斾富鑳介噺鏈€楂樼殑鍓?4 涓?    sig_reefs = period_df[period_df['primary_significant'] == True].sort_values('primary_power', ascending=False)
    example_reefs = sig_reefs.head(4)['reef_name'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 6.5))
    axes = axes.flatten()
    
    for idx, reef in enumerate(example_reefs):
        ax = axes[idx]
        df = res_df[res_df['reef_name'] == reef].sort_values('year')
        t = df['year'].values
        y = df['residual'].values
        
        res_info = sig_reefs[sig_reefs['reef_name'] == reef].iloc[0]
        
        # 閲嶆柊璁＄畻涓€涓?DLS 浠ヨ幏寰楃函鐭㈤噺鏇茬嚎
        frequency = np.linspace(1/30.0, 1/3.0, 1000)
        ls = LombScargle(t, y)
        power = ls.power(frequency)
        period = 1 / frequency
        
        ax.plot(period, power, color=COLORS['primary'], lw=1.5)
        ax.axvline(res_info['primary_period'], color=COLORS['secondary'], linestyle='--', lw=2,
                  label=f"Primary: {res_info['primary_period']:.1f} yr")
        
        if res_info['secondary_significant']:
            ax.axvline(res_info['secondary_period'], color=COLORS['accent'], linestyle=':', lw=2,
                      label=f"Secondary: {res_info['secondary_period']:.1f} yr")
        
        ax.set_xlabel('Period (years)', fontweight='bold')
        ax.set_ylabel('Lomb-Scargle Power', fontweight='bold')
        ax.set_title(f"{reef}\np-value: {res_info['primary_p_value']:.4f}", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(3, 30)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'A4_figure2_dls_periodograms.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'A4_figure2_dls_periodograms.svg', bbox_inches='tight')
    plt.close(fig)

def plot_figure3(sig_df):
    """Figure 3: Distribution of Significant Periods (A4 Half)"""
    fig, ax = plt.subplots(figsize=(8.27, 4.5))
    
    periods = sig_df['primary_period'].values
    ax.hist(periods, bins=np.arange(3, 31, 2), color=COLORS['primary'], edgecolor='white', alpha=0.8)
    
    # 鏇存敼涓?2-7 骞?ENSO
    ax.axvspan(2, 7, alpha=0.2, color=COLORS['accent'], label='ENSO Range (2-7 yr)')
    ax.axvspan(15, 30, alpha=0.2, color=COLORS['success'], label='IPO Range (15-30 yr)')
    
    ax.set_xlabel('Primary Period (years)', fontweight='bold')
    ax.set_ylabel('Number of Reefs', fontweight='bold')
    ax.set_title('Distribution of Significant Periods Across Reefs', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'A4_figure3_period_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'A4_figure3_period_distribution.svg', bbox_inches='tight')
    plt.close(fig)

def plot_figure4(sig_df):
    """Figure 4: Regional Period Comparison"""
    # A4 safe width
    fig, ax = plt.subplots(figsize=(8.27, 5))
    
    valid_regions = [r for r in REGION_ORDER if r in sig_df['region'].values]
    positions = range(len(valid_regions))
    
    data_by_region = [sig_df[sig_df['region'] == r]['primary_period'].dropna().values for r in valid_regions]
    
    bp = ax.boxplot(data_by_region, positions=positions, widths=0.5, patch_artist=True)
    
    box_colors = [COLORS['primary'], COLORS['success'], COLORS['accent'], COLORS['secondary'], COLORS['purple'], '#8c564b']
    for idx, patch in enumerate(bp['boxes']):
        patch.set_facecolor(box_colors[idx % len(box_colors)])
        patch.set_alpha(0.7)
        
    ax.set_xticks(positions)
    # 鐢ㄥ洖杞︽浛浠ｆ棆杞紝閬垮厤X杞撮噸鍙?    ax.set_xticklabels([r.replace(' ', '\n') for r in valid_regions], rotation=0, ha='center', fontweight='bold')
    ax.set_xlabel('Region', fontweight='bold')
    ax.set_ylabel('Primary Period (years)', fontweight='bold')
    ax.set_title('Period Distribution by GBR Region', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.axhspan(15, 30, color=COLORS['success'], alpha=0.1, label='IPO ~15-30 yr')
    ax.axhspan(2, 7, color=COLORS['accent'], alpha=0.1, label='ENSO ~2-7 yr')
    ax.legend(loc='upper right', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'A4_figure4_regional_periods.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'A4_figure4_regional_periods.svg', bbox_inches='tight')
    plt.close(fig)

def plot_figure5(model_df):
    """Figure 5: Model R2 Distribution"""
    r2_values = model_df['r_squared'].values
    
    fig, ax = plt.subplots(figsize=(8.27, 4.5))
    ax.hist(r2_values, bins=20, color=COLORS['accent'], edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(r2_values), color=COLORS['secondary'], linestyle='--', lw=2,
              label=f'Mean $R^2$ = {np.mean(r2_values):.3f}')
    ax.set_xlabel('$R^2$ (Coral-COTS Model)', fontweight='bold')
    ax.set_ylabel('Number of Reefs', fontweight='bold')
    ax.set_title('Model Fit Quality: Food Effect Explanation Power', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'A4_figure5_model_r2_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'A4_figure5_model_r2_distribution.svg', bbox_inches='tight')
    plt.close(fig)

def main():
    print("Loading pre-calculated enhm data and charting A4 versions...")
    res_df, model_df, period_df, sig_df = load_data()
    
    plot_figure1(res_df, model_df)
    plot_figure2(res_df, period_df)
    plot_figure3(sig_df)
    plot_figure4(sig_df)
    plot_figure5(model_df)
    
    print(f"All 5 diagnostic charts have been rendered successfully.")
    print(f"Directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
