# -*- coding: utf-8 -*-
"""
Paper Figures Generator
生成论文 Figure 2、Figure S2 和 Figure S5。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.stats import pearsonr
from pathlib import Path
from analysis_inputs import ecological_observations
from spectral_inference import spectrum
import matplotlib.colors as mcolors
import textwrap
from publication_style import save_publication

# 设置绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

# 配色方案
COLORS = {
    'primary': '#1f77b4',       # 蓝色
    'secondary': '#d62728',     # 红色
    'accent': '#ff7f0e',        # 橙色
    'success': '#2ca02c',       # 绿色
    'purple': '#9467bd',        # 紫色
    'gray': '#7f7f7f',          # 灰色
    'ci_band': '#a6cee3'        # 浅蓝
}

# 区域顺序 (北 -> 南)
REGION_ORDER = [
    'Cape York', 
    'Wet Tropics', 
    'Burdekin', 
    'Mackay Whitsunday', 
    'Fitzroy', 
    'Burnett Mary'
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "reef_surveys"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loading (Resued from combined_figure_generator.py)
# -----------------------------------------------------------------------------


def load_all_reef_data_from_files():
    cots, coral = ecological_observations()
    return cots, coral


def load_data():
    """加载分析结果数据"""
    cots_df, coral_df = load_all_reef_data_from_files()
    
    model_results = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "model_results.csv")
    period_results = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "period_results.csv")
    lag_analysis = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "lag_analysis.csv")
    
    # Load residuals if available (created by updated hbrs_analysis.py)
    try:
        residuals_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "all_residuals.csv")
    except:
        residuals_df = pd.DataFrame()
    
    # Merge region info if missing
    if 'region' not in period_results.columns:
        period_results = period_results.merge(model_results[['reef_name', 'region']], on='reef_name', how='left')
        
    return cots_df, coral_df, model_results, period_results, lag_analysis, residuals_df

def load_reef_lags():
    """Load reef-level optimal lags"""
    try:
        return pd.read_csv(PROJECT_ROOT / "results" / "tables" / "reef_lags.csv")
    except:
        return None

# -----------------------------------------------------------------------------
# Figure S2: Trophic lag and baseline calibration
# -----------------------------------------------------------------------------

def get_representative_reefs(model_results):
    """Select 1 representative reef per region (max observations), sorted North->South"""
    print("Selecting representative reefs (1 per region)...")
    
    selected_reefs = []
    
    for region in REGION_ORDER:
        region_reefs = model_results[model_results['region'] == region]
        if len(region_reefs) > 0:
            # Select reef with max observations
            best_reef = region_reefs.nlargest(1, 'n_observations')['reef_name'].iloc[0]
            selected_reefs.append(best_reef)
            print(f"  {region}: {best_reef} ({region_reefs.loc[region_reefs['reef_name']==best_reef, 'n_observations'].values[0]} obs)")
            
    return selected_reefs

def create_figure_s2_baseline(lag_analysis, model_results):
    """生成补充图 S2：滞后校准与基线解释力。"""
    print("Generating Figure S2: Trophic Lag and Baseline Calibration...")
    fig = plt.figure(figsize=(13, 4.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.34)
    
    # --- Panel A: Lag Profile ---
    ax_lag = fig.add_subplot(gs[0, 0])
    lags = lag_analysis['lag_years'].values
    correlations = lag_analysis['mean'].values
    standard_deviations = lag_analysis['std'].values
    
    ax_lag.axvspan(2, 4, color=COLORS['accent'], alpha=0.12,
                  label='Plausible range (2-4 yr)')
    ax_lag.errorbar(lags, correlations, yerr=standard_deviations, fmt='o-', capsize=4,
                    color=COLORS['primary'], markersize=6, lw=1.5)
    
    optimal_lag = int(lag_analysis.loc[lag_analysis['primary'], 'lag_years'].iloc[0])
    ax_lag.axvline(optimal_lag, color=COLORS['secondary'], linestyle='--', lw=2)
    ax_lag.text(optimal_lag + 0.35, np.max(correlations),
                f'Baseline: {optimal_lag} yr',
                color=COLORS['secondary'], fontweight='bold')
    common_n = int(lag_analysis['n'].min())
    ax_lag.text(0.98, 0.04, f'Common reef set: n = {common_n}',
                transform=ax_lag.transAxes, ha='right', va='bottom', fontsize=9)
    ax_lag.axhline(0, color='gray', linestyle='-', lw=0.8)
    
    ax_lag.set_xlabel('Time Lag (years)', fontsize=10.5, fontweight='bold')
    ax_lag.set_ylabel('Mean Cross-Correlation', fontsize=10.5, fontweight='bold')
    ax_lag.set_title('System-Wide Lag Profile', fontsize=11, fontweight='bold', loc='left')
    ax_lag.text(-0.08, 1.09, 'A', transform=ax_lag.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    ax_lag.legend(loc='upper right', fontsize=9, frameon=False)
    ax_lag.spines[['top', 'right']].set_visible(False)

    # --- Panel B: Lag Distribution ---
    ax_dist = fig.add_subplot(gs[0, 1])
    
    reef_lags = load_reef_lags()
    if reef_lags is not None:
        lag_counts = reef_lags['optimal_lag'].value_counts().sort_index()
        all_lags = list(lags)
        counts = [lag_counts.get(l, 0) for l in all_lags]
        
        bars = ax_dist.bar(all_lags, counts, color=COLORS['primary'], alpha=0.8)
        if optimal_lag in all_lags:
            bars[all_lags.index(optimal_lag)].set_color(COLORS['secondary'])
    else:
        ax_dist.text(0.5, 0.5, "Distribution Data Missing", ha='center')
    
    ax_dist.set_xlim(0.5, max(lags) + 0.5) # User requested to cut off empty space > 10

    ax_dist.set_xlabel('Optimal Lag (years)', fontsize=10.5, fontweight='bold')
    ax_dist.set_ylabel('Number of Reefs', fontsize=10.5, fontweight='bold')
    ax_dist.set_title('Reef-Level Lag Distribution', fontsize=11, fontweight='bold', loc='left')
    ax_dist.text(-0.08, 1.09, 'B', transform=ax_dist.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    if reef_lags is not None:
        ax_dist.text(0.98, 0.96, f'n = {len(reef_lags)} lag-screened reefs',
                     transform=ax_dist.transAxes, ha='right', va='top', fontsize=9)
    ax_dist.spines[['top', 'right']].set_visible(False)

    # --- Panel C: Model R² Distribution (Moved from Figure 3) ---
    ax_r2 = fig.add_subplot(gs[0, 2])
    r2_values = model_results['r_squared'].dropna().values
    
    ax_r2.hist(r2_values, bins=20, color=COLORS['gray'], edgecolor='white', alpha=0.7)
    mean_r2 = np.mean(r2_values)
    mean_r2_uncentered = model_results['r_squared_uncentered'].dropna().mean()
    ax_r2.axvline(mean_r2, color='k', linestyle='--', lw=2,
                  label=f'Centered mean: {mean_r2:.3f}')
    ax_r2.text(
        0.98, 0.96,
        f'n = {len(r2_values)} reefs with defined $R^2$\n'
        f'Non-centered mean: {mean_r2_uncentered:.3f}\n(RTO convention)',
        transform=ax_r2.transAxes, ha='right', va='top', fontsize=9,
    )
    ax_r2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.72),
                 fontsize=9, frameon=False)
    
    ax_r2.set_xlabel('Centered $R^2$', fontsize=10.5, fontweight='bold')
    ax_r2.set_ylabel('Count', fontsize=10.5, fontweight='bold')
    ax_r2.set_title('Baseline Explanatory Power', fontsize=11, fontweight='bold', loc='left')
    ax_r2.text(-0.08, 1.09, 'C', transform=ax_r2.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    ax_r2.spines[['top', 'right']].set_visible(False)
    
    fig.suptitle('Trophic Lag and Baseline Calibration',
                 fontsize=14, fontweight='bold', y=1.01)
    
    save_publication(fig, OUTPUT_DIR / 'FigureS2_Trophic_Lag_Baseline')
    plt.close()

# ... (Figure 2 and 3 functions remain unchanged)


# -----------------------------------------------------------------------------
# Figure 2: Periodicity (Residual Analysis)
# -----------------------------------------------------------------------------

def create_figure2_periodicity(cots_df, top4_reefs, period_results, model_results, residuals_df):
    """生成图3：周期性分析（环境驱动）- 4个周期图 + 1个分布直方图"""
    print("Generating Figure 2: Periodicity (4 Periodograms + 1 Histogram)...")
    # Increased height to 8.0 to fix the "flat" look and achieve better proportions
    fig = plt.figure(figsize=(8.27, 8.0))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.4)
    
    # --- Panel A: Residual Periodograms (Top 4) ---
    gs_top = gs[0].subgridspec(1, 4, wspace=0.08)
    
    from astropy.timeseries import LombScargle
    panel_labels = ['A', 'B', 'C', 'D']
    
    for i, reef in enumerate(top4_reefs):
        ax = fig.add_subplot(gs_top[0, i])
        

        reef_res = residuals_df.loc[residuals_df['reef_name'].eq(reef)].sort_values('time_years')
        t = reef_res['time_years'].to_numpy()
        y = reef_res['residual'].to_numpy()
        frequency, pgram = spectrum(t, y)
        periods = 1 / frequency
        title_suffix = ""


        ax.plot(periods, pgram, color=COLORS['success'], lw=1.6)

        # Period annotation
        reef_period_row = period_results[period_results['reef_name'] == reef]
        if not reef_period_row.empty:
            prim_per = reef_period_row['primary_period'].iloc[0]
            ax.axvline(prim_per, color='k', linestyle='--', lw=1.1, alpha=0.6)
            y_max = pgram.max()
            # Keep peaks at the 30-year search boundary legible inside the axes.
            if prim_per >= 25:
                label_x, label_ha = prim_per - 0.7, 'right'
            else:
                label_x, label_ha = prim_per + 0.7, 'left'
            ax.text(label_x, min(y_max + 0.02, 1.0), f'{prim_per:.1f}y',
                   fontsize=9, fontweight='bold', color='k', va='bottom', ha=label_ha,
                   clip_on=True)
        
        # Shade regions
        ax.axvspan(3, 7, color=COLORS['accent'], alpha=0.12)  # ENSO-scale
        ax.axvspan(15, 30, color=COLORS['success'], alpha=0.12)  # IPO
        
        short_name = reef.split(' (')[0]
        ax.set_title(f"{short_name}{title_suffix}", fontsize=10.5, fontweight='bold', loc='center', pad=6)
        ax.text(0.05, 0.97, panel_labels[i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
        ax.set_xlabel('Period (yr)', fontsize=10.5, fontweight='bold')
        ax.set_ylim(0, 1.08) 
        ax.set_xlim(3, 30)
        ax.set_xticks([5, 10, 20, 30])
        
        if i == 0:
            ax.set_ylabel('Normalized Power', fontsize=10.5, fontweight='bold')
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
            
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.9)

    # --- Panel E: Global Period Distribution ---
    ax_hist = fig.add_subplot(gs[1])
    sig = period_results[period_results['primary_significant'] == True]
    bins = np.arange(3, 31, 1.5)
    ax_hist.hist(period_results['primary_period'], bins=bins, color=COLORS['gray'],
                 edgecolor='white', alpha=0.45, label=f'All screened reefs (n = {len(period_results)})')
    if not sig.empty:
        ipo_n = int(sig['primary_period'].between(15, 30).sum())
        ax_hist.hist(sig['primary_period'], bins=bins, color=COLORS['primary'],
                     edgecolor='white', alpha=0.9,
                     label=f'Significant dominant periods (n = {len(sig)}; IPO band: {ipo_n}/{len(sig)})')
    ax_hist.axvspan(3, 7, color=COLORS['accent'], alpha=0.15, label='ENSO-scale (3–7 yr)')
    ax_hist.axvspan(15, 30, color=COLORS['success'], alpha=0.15, label='IPO band (15–30 yr)')
    ax_hist.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#cccccc', fontsize=9)
    
    ax_hist.set_xlabel('Dominant Period (years)', fontsize=10.5, fontweight='bold')
    ax_hist.set_ylabel('Number of Reefs', fontsize=10.5, fontweight='bold')
    ax_hist.text(0.015, 0.96, 'E', transform=ax_hist.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    ax_hist.set_xlim(2.5, 30.5)
    ax_hist.set_ylim(0, 18.5)
    
    ax_hist.grid(False)
    for spine in ax_hist.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
    
    save_publication(fig, OUTPUT_DIR / 'Figure2_Residual_Periodicity')
    plt.close()

# -----------------------------------------------------------------------------
# Figure 3: Spatiotemporal Dynamics (Hovmoller - Sorted by Latitude)
# -----------------------------------------------------------------------------




def get_top_periodic_reefs(period_results):
    """Select top 4 reefs with strongest primary periodicity signal"""
    print("Selecting top 4 periodic reefs...")
    # Filter for significant IPO periods (15-30 years) to align with story
    sig_reefs = period_results[
        (period_results['primary_significant'] == True) & 
        (period_results['primary_period'].between(15, 30))
    ].copy()
    
    if len(sig_reefs) < 4:
        # Fallback to just sorting by power
        top_reefs = period_results.sort_values('primary_power', ascending=False).head(4)['reef_name'].tolist()
    else:
        top_reefs = sig_reefs.sort_values('primary_power', ascending=False).head(4)['reef_name'].tolist()
        
    print(f"  Top 4 Periodic: {top_reefs}")
    return top_reefs

# -----------------------------------------------------------------------------
# Figure 4: Spectral Decomposition - Unmasking Environmental Signals
# -----------------------------------------------------------------------------



def create_figure4_spectral(cots_df, model_results, period_results, residuals_df):
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    panel_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for idx, (region, ax) in enumerate(zip(REGION_ORDER, axes.flat)):
        candidates = model_results.loc[model_results['region'].eq(region), 'reef_name']
        choices = period_results.loc[period_results.reef_name.isin(candidates)].sort_values(
            ['span_years', 'n_survey_dates', 'reef_name'], ascending=[False, False, True]
        )
        ax.set_title(region)
        if choices.empty:
            ax.set_visible(False)
            continue
        reef = choices.iloc[0]['reef_name']
        series = residuals_df.loc[residuals_df['reef_name'].eq(reef)].sort_values('time_years')
        time = series['time_years'].to_numpy()
        for column, label, color in [('log_cots', 'Raw matched observations', COLORS['gray']),
                                     ('residual', 'Coral baseline residual', COLORS['success'])]:
            result = spectrum(time, series[column].to_numpy())
            if result is not None:
                frequency, power = result
                ax.plot(1 / frequency, power, color=color, lw=1.5, label=label)
        ax.axvspan(15, 30, color=COLORS['success'], alpha=0.12)
        ax.set_title(f'{region} – {reef}', fontsize=10.5, fontweight='bold')
        ax.text(-0.10, 1.10, panel_letters[idx], transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top', ha='left')
        ax.set(xlim=(3, 30), ylim=(0, 1.02))
        ax.set_xlabel('Period (years)' if idx >= 4 else '')
        ax.set_ylabel('Normalized power' if idx % 2 == 0 else '')
        ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    save_publication(fig, OUTPUT_DIR / 'FigureS5_Spectral_Decomposition')
    plt.close(fig)


if __name__ == "__main__":
    print("Starting Paper Figures Generation...")
    
    cots_df, coral_df, model_results, period_results, lag_analysis, residuals_df = load_data()
    
    # 1. Representative Reefs (Longest Record)
    rep_reefs = get_representative_reefs(model_results)
    
    # 2. Top Periodic Reefs (Strongest Signal)
    top_periodic_reefs = get_top_periodic_reefs(period_results)
    
    create_figure_s2_baseline(lag_analysis, model_results)
    create_figure2_periodicity(cots_df, top_periodic_reefs, period_results, model_results, residuals_df)
    create_figure4_spectral(cots_df, model_results, period_results, residuals_df)
    
    print("\nAll figures generated successfully in:", OUTPUT_DIR)
