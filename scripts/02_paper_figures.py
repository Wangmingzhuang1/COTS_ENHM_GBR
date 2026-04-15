# -*- coding: utf-8 -*-
"""
Paper Figures Generator
鐢熸垚鐢ㄤ簬璁烘枃鐨勪笁濂楃粍鍚堝浘锛?
Figure 1: Mechanism (Lag + Regression)
Figure 2: Periodicity (Residual Analysis)
Figure 3: Spatiotemporal Dynamics (Hovmoller + Regional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib.colors as mcolors

# 璁剧疆缁樺浘椋庢牸
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

# 閰嶈壊鏂规
COLORS = {
    'primary': '#1f77b4',       # 钃濊壊
    'secondary': '#d62728',     # 绾㈣壊
    'accent': '#ff7f0e',        # 姗欒壊
    'success': '#2ca02c',       # 缁胯壊
    'purple': '#9467bd',        # 绱壊
    'gray': '#7f7f7f',          # 鐏拌壊
    'ci_band': '#a6cee3'        # 娴呰摑
}

# 鍖哄煙椤哄簭 (鍖?-> 鍗?
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
    """浠庡師濮婥SV鏂囦欢鍔犺浇鏁版嵁"""
    print("Listing files...")
    csv_files = list(DATA_DIR.glob('*.csv'))
    
    cots_files = {} 
    coral_files = {}
    
    for f in csv_files:
        fname = f.name
        if '_cots_modelled_' in fname:
            reef_name = fname.split('_Manta Tow_cots_modelled_')[0]
            if reef_name not in cots_files: cots_files[reef_name] = []
            cots_files[reef_name].append(f)
        elif '_line_chart_modelled_' in fname:
            reef_name = fname.split('_Manta Tow_line_chart_modelled_')[0]
            if reef_name not in coral_files: coral_files[reef_name] = []
            coral_files[reef_name].append(f)
            
    # Remove duplicates
    for d in [cots_files, coral_files]:
        for k, v in d.items():
            if len(v) > 1:
                primary = [f for f in v if '(1)' not in f.name]
                d[k] = primary[:1] if primary else v[:1]
                
    valid_reefs = set(cots_files.keys()) & set(coral_files.keys())
    print(f"Found {len(valid_reefs)} reefs with both COTS and Coral data")
    
    all_cots_data = []
    all_coral_data = []
    
    for reef in valid_reefs:
        try:
            # Load COTS
            cots_df = pd.read_csv(cots_files[reef][0])
            if 'date' in cots_df.columns:
                cots_df['year'] = pd.to_datetime(cots_df['date'], errors='coerce').dt.year
            elif 'Year' in cots_df.columns:
                cots_df['year'] = cots_df['Year']
            elif 'year' in cots_df.columns:
                 cots_df['year'] = cots_df['year']
            
            # Handle different column names
            if 'cots' in cots_df.columns:
                 cots_val = pd.to_numeric(cots_df['cots'], errors='coerce')
            elif 'cotsptow' in cots_df.columns:
                 cots_val = pd.to_numeric(cots_df['cotsptow'], errors='coerce')
            else:
                # print(f"Skipping {reef}: No cots column")
                continue

            cots_reef_df = pd.DataFrame({
                'reef_name': reef,
                'year': cots_df['year'],
                'cots_density': cots_val,
                'nrm_region': cots_df.get('nrm_region', 'Unknown'),
                'a_sector': cots_df.get('a_sector', 'Unknown')
            })
            all_cots_data.append(cots_reef_df)

            # Load Coral
            coral_df = pd.read_csv(coral_files[reef][0])
            if 'report_year' in coral_df.columns:
                coral_df['year'] = coral_df['report_year']
            elif 'year' in coral_df.columns:
                coral_df['year'] = coral_df['year']
            elif 'Year' in coral_df.columns:
                coral_df['year'] = coral_df['Year']
            elif 'date' in coral_df.columns:
                # Handle fractional year
                try:
                    coral_df['year'] = pd.to_numeric(coral_df['date'], errors='coerce').astype(int)
                except:
                    coral_df['year'] = pd.to_datetime(coral_df['date'], errors='coerce').dt.year
            
            if 'coral_cover' in coral_df.columns:
                coral_val = pd.to_numeric(coral_df['coral_cover'], errors='coerce')
            elif 'mean' in coral_df.columns:
                coral_val = pd.to_numeric(coral_df['mean'], errors='coerce')
            elif 'median' in coral_df.columns:
                coral_val = pd.to_numeric(coral_df['median'], errors='coerce')
            else:
                 coral_val = pd.to_numeric(coral_df.iloc[:, 1], errors='coerce')
            
            coral_reef_df = pd.DataFrame({
                'reef_name': reef,
                'year': coral_df['year'],
                'coral_cover': coral_val
            })
            all_coral_data.append(coral_reef_df)
            
        except Exception as e:
            continue
            
    cots_df_final = pd.concat(all_cots_data, ignore_index=True)
    coral_df_final = pd.concat(all_coral_data, ignore_index=True)
    
    return cots_df_final, coral_df_final

def load_data():
    """鍔犺浇鍒嗘瀽缁撴灉鏁版嵁"""
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
# Figure 1: Mechanism (Lag + Regression)
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

def create_figure1_mechanism(cots_df, coral_df, lag_analysis, model_results, rep_reefs):
    """鐢熸垚鍥?锛氭満鍒堕獙璇侊紙婊炲悗 + 鍥炲綊 + R虏鍒嗗竷锛? 3+6闈㈡澘甯冨眬"""
    print("Generating Figure 2: Mechanism (3 Top Panels + 6 Regression Panels)...")
    fig = plt.figure(figsize=(16, 14))
    # Use unified 3x3 GridSpec for perfect column alignment
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.25)
    
    # --- Panel A: Lag Profile ---
    ax_lag = fig.add_subplot(gs[0, 0])
    lags = lag_analysis['lag_years'].values
    correlations = lag_analysis['mean'].values
    std_errs = lag_analysis['std'].values
    
    ax_lag.errorbar(lags, correlations, yerr=std_errs, fmt='o-', capsize=4, 
                    color=COLORS['primary'], markersize=6, lw=1.5)
    
    optimal_idx = np.argmax(correlations)
    optimal_lag = lags[optimal_idx]
    ax_lag.axvline(optimal_lag, color=COLORS['secondary'], linestyle='--', lw=2)
    ax_lag.text(optimal_lag + 0.5, np.max(correlations), f'Optimal: {optimal_lag} yr', 
             color=COLORS['secondary'], fontweight='bold')
    ax_lag.axhline(0, color='gray', linestyle='-', lw=0.8)
    
    ax_lag.set_xlabel('Time Lag (years)', fontsize=12, fontweight='bold')
    ax_lag.set_ylabel('Cross-Correlation (Coral 鈫?COTS)', fontsize=12, fontweight='bold')
    ax_lag.set_title('(a) Temporal Lag Identification', fontsize=14, fontweight='bold', loc='left')
    ax_lag.spines[['top', 'right']].set_visible(False)

    # --- Panel B: Lag Distribution ---
    ax_dist = fig.add_subplot(gs[0, 1])
    
    reef_lags = load_reef_lags()
    if reef_lags is not None:
        lag_counts = reef_lags['optimal_lag'].value_counts().sort_index()
        all_lags = range(1, 16)
        counts = [lag_counts.get(l, 0) for l in all_lags]
        
        bars = ax_dist.bar(all_lags, counts, color=COLORS['primary'], alpha=0.8)
        if optimal_lag <= 15:
            bars[optimal_lag-1].set_color(COLORS['secondary'])
    else:
        ax_dist.text(0.5, 0.5, "Distribution Data Missing", ha='center')
    
    ax_dist.set_xlim(0.5, 10.5) # User requested to cut off empty space > 10

    ax_dist.set_xlabel('Optimal Lag (years)', fontsize=12, fontweight='bold')
    ax_dist.set_ylabel('Number of Reefs', fontsize=12, fontweight='bold')
    ax_dist.set_title('(b) Reef-Level Lag Distribution', fontsize=14, fontweight='bold', loc='left')
    ax_dist.spines[['top', 'right']].set_visible(False)

    # --- Panel C: Model R虏 Distribution (Moved from Figure 3) ---
    ax_r2 = fig.add_subplot(gs[0, 2])
    r2_values = model_results['r_squared'].dropna().values
    
    ax_r2.hist(r2_values, bins=20, color=COLORS['gray'], edgecolor='white', alpha=0.7)
    mean_r2 = np.mean(r2_values)
    ax_r2.axvline(mean_r2, color='k', linestyle='--', lw=2, label=f'Mean: {mean_r2:.2f}')
    ax_r2.legend(loc='upper right')
    
    ax_r2.set_xlabel('Food Model $R^2$', fontsize=12, fontweight='bold')
    ax_r2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax_r2.set_title('(c) Model Explanatory Power', fontsize=14, fontweight='bold', loc='left')
    ax_r2.spines[['top', 'right']].set_visible(False)

    # --- Panel D-I: Regression Examples (2x3 Grid for 6 Regions) ---
    # Replaced subgridspec with direct indexing into the unified 3x3 grid
    
    for i, reef in enumerate(rep_reefs):
        if i >= 6: break # Safety
        
        row = (i // 3) + 1 # Rows 1 and 2
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        reef_cots = cots_df[cots_df['reef_name'] == reef].copy()
        reef_coral = coral_df[coral_df['reef_name'] == reef].copy()
        
        # Simple processing
        cots_annual = reef_cots.groupby('year')['cots_density'].mean()
        coral_annual = reef_coral.groupby('year')['coral_cover'].mean()
        
        lagged_coral = coral_annual.shift(optimal_lag)
        common = cots_annual.index.intersection(lagged_coral.dropna().index)
        
        if len(common) > 0:
            x = lagged_coral[common].values
            y = np.log1p(cots_annual[common].values)
            
            # Remove NaNs
            mask = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[mask], y[mask]

            ax.scatter(x, y, alpha=0.6, c=COLORS['primary'], edgecolors='white', s=50)
            
            # Retrieve HBRS model parameters (Through-Origin Regression)
            # Find matching model result for this reef
            reef_model = model_results[model_results['reef_name'] == reef]
            if not reef_model.empty:
                beta = reef_model['beta_coefficient'].iloc[0] if 'beta_coefficient' in reef_model.columns else reef_model['beta'].iloc[0]
                r2 = reef_model['r_squared'].iloc[0]
                
                # Plot HBRS Model Line (y = beta * x)
                x_line = np.linspace(min(x), max(x), 100)
                ax.plot(x_line, beta * x_line, color=COLORS['secondary'], lw=2)
                ax.text(0.05, 0.9, f'$R^2$={r2:.2f}\n$\\beta$={beta:.2f}', transform=ax.transAxes, 
                        color=COLORS['secondary'], fontsize=10)
            else:
                # Fallback if model not found (rare)
                m, c = np.polyfit(x, y, 1)
                ax.plot(x_line, m*x_line + c, 'k--', alpha=0.5)

        # Title and cleaning
        short_name = reef.split(' (')[0]
        region = model_results[model_results['reef_name']==reef]['region'].values[0]
        # User requested to REMOVE region/reef titles
        # ax.set_title(f"{region}", fontsize=11, fontweight='bold') 
        ax.set_box_aspect(1) # Force square aspect ratio
        ax.set_xlabel('Coral (%)', fontsize=11)
        if col == 0:
            ax.set_ylabel('log(COTS + 1)', fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.spines[['top', 'right']].set_visible(False)
    
    fig.suptitle('Mechanism: Food Limitation Sets the Population Baseline', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    plt.savefig(OUTPUT_DIR / 'Figure2_Mechanism.png', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / 'Figure2_Mechanism.svg', bbox_inches='tight')
    plt.close()

# ... (Figure 2 and 3 functions remain unchanged)


# -----------------------------------------------------------------------------
# Figure 2: Periodicity (Residual Analysis)
# -----------------------------------------------------------------------------

def create_figure2_periodicity(cots_df, top4_reefs, period_results, model_results, residuals_df):
    """鐢熸垚鍥?锛氬懆鏈熸€у垎鏋愶紙鐜椹卞姩锛? 4涓懆鏈熷浘 + 1涓垎甯冪洿鏂瑰浘"""
    print("Generating Figure 3: Periodicity (4 Periodograms + 1 Histogram)...")
    # Increased height to 8.0 to fix the "flat" look and achieve better proportions
    fig = plt.figure(figsize=(8.27, 8.0))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.4)
    
    # --- Panel A: Residual Periodograms (Top 4) ---
    gs_top = gs[0].subgridspec(1, 4, wspace=0.08)
    
    from astropy.timeseries import LombScargle
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    
    for i, reef in enumerate(top4_reefs):
        ax = fig.add_subplot(gs_top[0, i])
        
        # Load EXACT residuals from HBRS analysis
        if not residuals_df.empty:
            reef_res = residuals_df[residuals_df['reef_name'] == reef].sort_values('year')
        else:
            reef_res = pd.DataFrame()
            
        if len(reef_res) < 5:
            reef_data = cots_df[cots_df['reef_name'] == reef].groupby('year')['cots_density'].mean()
            t = reef_data.index.values.astype(float)
            y = signal.detrend(np.log1p(reef_data.values))
            title_suffix = "*"
        else:
            t = reef_res['year'].values.astype(float)
            y = reef_res['residual'].values
            title_suffix = ""
        
        # Analysis parameters (3-30 years)
        min_period = 3.0
        max_period = 30.0
        frequency = np.linspace(1/max_period, 1/min_period, 1000)
        
        ls = LombScargle(t, y)
        pgram = ls.power(frequency)
        periods = 1 / frequency
        
        ax.plot(periods, pgram, color=COLORS['success'], lw=1.5)

        # Period annotation
        reef_period_row = period_results[period_results['reef_name'] == reef]
        if not reef_period_row.empty:
            prim_per = reef_period_row['primary_period'].iloc[0]
            ax.axvline(prim_per, color='k', linestyle='--', lw=1, alpha=0.5)
            y_max = pgram.max()
            ax.text(prim_per + 0.5, y_max, f'{prim_per:.1f}y', 
                   fontsize=8, color='k', va='bottom')
        
        # Shade regions
        ax.axvspan(2, 7, color=COLORS['accent'], alpha=0.1)  # ENSO
        ax.axvspan(15, 30, color=COLORS['success'], alpha=0.1)  # IPO
        
        short_name = reef.split(' (')[0]
        ax.set_title(f"{panel_labels[i]} {short_name}{title_suffix}", fontsize=9, fontweight='bold', loc='left')
        ax.set_xlabel('Period (yr)', fontsize=9)
        ax.set_ylim(0, 1.1) 
        ax.set_xlim(2, 30)
        
        if i == 0:
            ax.set_ylabel('Normalized Power', fontsize=10)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
            
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

    # --- Panel B: Global Period Distribution ---
    ax_hist = fig.add_subplot(gs[1])
    sig = period_results[period_results['primary_significant'] == True]
    periods_data = sig['primary_period'].dropna().values
    
    ax_hist.hist(periods_data, bins=np.arange(3, 31, 1.5), color=COLORS['success'], 
                 edgecolor='white', alpha=0.7)
    
    ax_hist.axvspan(2, 7, color=COLORS['accent'], alpha=0.15, label='ENSO (2-7yr)')
    ax_hist.axvspan(15, 30, color=COLORS['success'], alpha=0.15, label='IPO (15-30yr)')
    ax_hist.legend(loc='upper right', frameon=True, fontsize=10)
    
    ax_hist.set_xlabel('Significant Primary Period (years)', fontsize=11, fontweight='bold')
    ax_hist.set_ylabel('Number of Reefs', fontsize=11, fontweight='bold')
    ax_hist.set_title('(e) Period Distribution of Significant Reefs', fontsize=12, fontweight='bold', loc='left')
    
    ax_hist.grid(False)
    for spine in ax_hist.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure3_Periodicity.png', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / 'Figure3_Periodicity.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure3_Periodicity.svg', bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Figure 3: Spatiotemporal Dynamics (Hovmoller - Sorted by Latitude)
# -----------------------------------------------------------------------------

def load_coordinates_for_hovmoller():
    """浠巗ite_lon_lat.csv鍔犺浇鍧愭爣"""
    import re
    coord_file = PROJECT_ROOT / "data" / "spatial" / "site_lon_lat.csv"
    df = pd.read_csv(coord_file, encoding='gbk')
    df.columns = ['site', 'coords'] + list(df.columns[2:])
    
    # Robust parse_dms function
    def parse_dms(dms_str):
        if pd.isna(dms_str) or not dms_str:
            return None, None
            
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
    # FIX: Rename mismatched reefs before processing
    df['site'] = df['site'].replace({'Kangaroo Reef': 'Kangaroo Reef West'})

    for _, row in df.iterrows():
        lat, lon = parse_dms(row['coords'])
        coords_parsed.append({'reef_name': row['site'], 'lat': lat, 'lon': lon})
    
    return pd.DataFrame(coords_parsed).dropna()

def create_figure3_dynamics(cots_df, period_results):
    """鐢熸垚鍥?锛氭椂绌哄姩鍔涘锛圚ovmoller鍥?鎸夌簿纭含搴︽帓搴忥級"""
    print("Generating Figure 3: Spatiotemporal Dynamics (Latitude-sorted)...")
    
    # Load coordinates
    coords = load_coordinates_for_hovmoller()
    
    # Merge COTS data with coordinates
    cots_with_coords = cots_df.merge(coords, on='reef_name', how='inner')
    
    if len(cots_with_coords) == 0:
        print("  No COTS data with coordinates, falling back to region order.")
        return
    
    # Create pivot table
    ptable = cots_with_coords.pivot_table(
        index=['lat', 'nrm_region', 'reef_name'], 
        columns='year', 
        values='cots_density', 
        aggfunc='mean'
    )
    
    # Sort by latitude (North to South = less negative to more negative)
    ptable = ptable.reset_index()
    ptable = ptable.sort_values('lat', ascending=False)  # North first
    
    # Clean up matrix for plotting
    matrix_data = ptable.drop(columns=['lat', 'nrm_region', 'reef_name'])
    matrix_values = np.log1p(matrix_data.values)
    
    # Get latitudes for y-axis (convert to positive)
    latitudes = np.abs(ptable['lat'].values)
    regions = ptable['nrm_region'].tolist()
    
    # --- Plotting ---
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.25)
    
    # Panel A: Hovmoller
    ax_hov = fig.add_subplot(gs[0])
    
    # Create colormap
    cmap = plt.cm.YlOrRd
    years = matrix_data.columns.values
    
    img = ax_hov.imshow(matrix_values, aspect='auto', cmap=cmap, 
                       interpolation='nearest', origin='upper',
                       extent=[years.min(), years.max(), latitudes.max(), latitudes.min()])
    
    # Add region change dividers
    current_reg = None
    for i, reg in enumerate(regions):
        if reg != current_reg:
            if i > 0:
                ax_hov.axhline(latitudes[i], color='white', linestyle='-', lw=1, alpha=0.8)
            current_reg = reg
    
    # Y-axis as latitude (positive values, North at top)
    ax_hov.set_ylabel('Latitude (掳S)', fontweight='bold')
    ax_hov.invert_yaxis()  # Keep North at top
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax_hov, pad=0.02)
    cbar.set_label('log(COTS Density + 1)', rotation=270, labelpad=15, fontsize=11)
    
    ax_hov.set_xlabel('Year', fontweight='bold')
    ax_hov.set_title('(A) Spatiotemporal Outbreak Propagation (North to South)', 
                    fontweight='bold', loc='left')
    
    # Add "Traveling Wave" annotation arrow
    start_year, start_lat = 1995, latitudes[5]  # Near North
    end_year, end_lat = 2015, latitudes[-10]  # Near South
    
    ax_hov.annotate("", xy=(end_year, end_lat), xytext=(start_year, start_lat),
                   arrowprops=dict(arrowstyle="->", color="blue", lw=3, alpha=0.7))
    ax_hov.text((start_year+end_year)/2, (start_lat+end_lat)/2 + 1, "Outbreak Wave", 
               rotation=35, color="blue", fontweight='bold', ha='center', va='bottom')

    # Panel B: Regional Period Boxplot
    ax_box = fig.add_subplot(gs[1])
    sig = period_results[period_results['primary_significant'] == True]
    
    data = []
    labels = []
    for reg in REGION_ORDER:
        reg_data = sig[sig['region'] == reg]['primary_period'].dropna().values
        if len(reg_data) > 0:
            data.append(reg_data)
            labels.append(reg)
            
    if data:
        ax_box.boxplot(data, labels=labels)
        ax_box.set_ylabel('Dominant Period (years)', fontweight='bold')
        ax_box.set_title('(B) Regional Periodicity Characteristics', fontweight='bold', loc='left')
        ax_box.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax_box.axhspan(15, 30, color='gray', alpha=0.15, label='IPO (15-30y)')
        plt.setp(ax_box.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    fig.suptitle('Spatiotemporal Dynamics: The "Traveling Wave"', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    plt.savefig(OUTPUT_DIR / 'Figure3_Spatiotemporal.png', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / 'Figure3_Spatiotemporal.svg', bbox_inches='tight')
    plt.close()
    
    print(f"  Hovmoller sorted by latitude (range: {latitudes.max():.1f}S to {latitudes.min():.1f}S)")

# -----------------------------------------------------------------------------
# Main Execution
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
    """鐢熸垚琛ュ厖鍥? 棰戣氨鍒嗚В瀵规瘮 - 灞曠ず鐢熺墿鍓ョ濡備綍鎻ず鐜鍛ㄦ湡淇″彿"""
    print("\n  Creating SuppFig: Spectral Decomposition...")
    
    from scipy import signal as scipy_signal
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # 姣忎釜鍖哄煙閫変竴涓唬琛ㄧ锛堜紭鍏堥€夋嫨鏈夊畬鏁存畫宸暟鎹殑锛?
    region_representatives = {}
    region_stats = {}  # 璁板綍姣忎釜鍖哄煙鐨勭粺璁′俊鎭?
    
    for region in REGION_ORDER:
        region_reefs = model_results[model_results['region'] == region]
        if len(region_reefs) == 0:
            continue
            
        candidates = None # Reset candidates for this region
            
        # 閫夋嫨鏈夋樉钁楀懆鏈熶笖鏈夋畫宸暟鎹殑绀?
        reef_periods = period_results[
            (period_results['reef_name'].isin(region_reefs['reef_name'])) &
            (period_results['primary_significant'] == True)
        ]
        
        if len(reef_periods) > 0:
            # 浼樺厛閫夋嫨鏈夋畫宸暟鎹殑绀?
            candidates = []
            for _, row in reef_periods.iterrows():
                reef_name = row['reef_name']
                reef_res = residuals_df[residuals_df['reef_name'] == reef_name] if residuals_df is not None else None
                has_residual = reef_res is not None and len(reef_res) > 5
                candidates.append({
                    'reef_name': reef_name,
                    'power': row['primary_power'],
                    'has_residual': has_residual
                })
            
            # 鎸?has_residual, power 鎺掑簭
            candidates_df = pd.DataFrame(candidates)
            best_reef = candidates_df.sort_values(['has_residual', 'power'], ascending=[False, False]).iloc[0]['reef_name']
        
        # 鈿狅笍 FALLBACK: 濡傛灉娌℃湁鏄捐憲鍛ㄦ湡鐨勶紝灏濊瘯鎵句换鎰忔湁娈嬪樊鐨?
        if candidates is None or (isinstance(candidates, list) and len(candidates) > 0 and not candidates[0]['has_residual']):
             # 灏濊瘯鍦ㄦ暣涓尯鍩熸壘鏈夋畫宸殑
             fallback_reefs = region_reefs['reef_name'].unique()
             for r in fallback_reefs:
                 r_res = residuals_df[residuals_df['reef_name'] == r] if residuals_df is not None else None
                 if r_res is not None and len(r_res) > 10:
                     best_reef = r
                     print(f"    Region {region}: Fallback to {r} (Has Residuals)")
                     break
            
        region_representatives[region] = best_reef
        region_stats[region] = {'reef_name': best_reef}
    
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    
    for idx, region in enumerate(REGION_ORDER):
        if region not in region_representatives:
            continue
        if idx >= 6:
            break
            
        ax = axes[idx]
        reef_name = region_representatives[region]
        
        # 鑾峰彇鍘熷COTS鏁版嵁
        reef_cots = cots_df[cots_df['reef_name'] == reef_name].groupby('year')['cots_density'].mean()
        if len(reef_cots) < 10:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', transform=ax.transAxes)
            continue
        
        # 鑾峰彇娈嬪樊鏁版嵁
        reef_res = residuals_df[residuals_df['reef_name'] == reef_name] if residuals_df is not None else None
        if reef_res is not None and len(reef_res) > 0:
            res_annual = reef_res.groupby('year')['residual'].mean()
        else:
            res_annual = None
        
        # 璁＄畻棰戣氨 - 鍘熷鏁版嵁
        years = reef_cots.index.values
        vals = np.log1p(reef_cots.values)
        vals_detrend = scipy_signal.detrend(vals)
        
        # Lomb-Scargle
        periods = np.linspace(2, 30, 200)
        freqs = 1.0 / periods
        power_raw = scipy_signal.lombscargle(years, vals_detrend, 2*np.pi*freqs, normalize=True)
        
        # 娈嬪樊棰戣氨
        power_res = None
        n_res = 0
        if res_annual is not None and len(res_annual) > 5:
            common_years = np.intersect1d(years, res_annual.index.values)
            if len(common_years) > 5:
                res_vals = res_annual.loc[common_years].values
                res_detrend = scipy_signal.detrend(res_vals)
                power_res = scipy_signal.lombscargle(common_years, res_detrend, 2*np.pi*freqs, normalize=True)
                n_res = len(common_years)
        
        # 缁樺浘 - 鍘熷鏁版嵁锛堢伆鑹茶櫄绾匡級
        ax.plot(periods, power_raw / power_raw.max(), color='gray', linestyle='--', lw=2, 
                alpha=0.6, label=f'Raw COTS (n={len(years)})')
        
        # 缁樺浘 - 鐜娈嬪樊锛堣摑鑹插疄绾匡級
        if power_res is not None:
            ax.plot(periods, power_res / power_res.max(), color=COLORS['primary'], 
                    lw=2.5, label=f'Env. Residual (n={n_res})')
        else:
            # 鍗充娇娌℃湁娈嬪樊鏁版嵁锛屼篃鏄剧ず璇存槑鏂囨湰
            ax.text(0.98, 0.95, 'No residual data', 
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=9, style='italic', color='gray')
        
        # IPO 棰戞楂樹寒
        ax.axvspan(15, 30, color=COLORS['success'], alpha=0.12, zorder=0)
        ax.text(22.5, 0.95, 'IPO\n(15-30y)', ha='center', va='top',
               fontsize=9, color=COLORS['success'], fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=COLORS['success'], alpha=0.8))
        
        ax.set_xlim(2, 30)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel('Period (years)', fontsize=11)
        ax.set_ylabel('Normalized Power', fontsize=11)
        
        short_name = reef_name.split(' (')[0]
        if len(short_name) > 18:
            short_name = short_name[:18] + '...'
        ax.set_title(f'({panel_labels[idx]}) {region}\n{short_name}', 
                    fontweight='bold', fontsize=11, loc='left')
        
        # 娣诲姞鍥句緥锛堜粎绗竴涓潰鏉匡級
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True)
        
        # 娣诲姞缃戞牸
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    fig.suptitle('Spectral Decomposition: Biological Stripping Unmasks Environmental Periodicity', 
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'SuppFig_Spectral_Decomposition.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'SuppFig_Spectral_Decomposition.svg', bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: SuppFig_Spectral_Decomposition.png and .pdf")
    print(f"    Regions analyzed: {len(region_representatives)}")

if __name__ == "__main__":
    print("Starting Paper Figures Generation...")
    
    cots_df, coral_df, model_results, period_results, lag_analysis, residuals_df = load_data()
    
    # 1. Representative Reefs (Longest Record)
    rep_reefs = get_representative_reefs(model_results)
    
    # 2. Top Periodic Reefs (Strongest Signal)
    top_periodic_reefs = get_top_periodic_reefs(period_results)
    
    create_figure1_mechanism(cots_df, coral_df, lag_analysis, model_results, rep_reefs)
    create_figure2_periodicity(cots_df, top_periodic_reefs, period_results, model_results, residuals_df)
    create_figure4_spectral(cots_df, model_results, period_results, residuals_df)
    
    print("\nAll figures generated successfully in:", OUTPUT_DIR)
