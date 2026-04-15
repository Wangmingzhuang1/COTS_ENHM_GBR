#!/usr/bin/env python3
"""
Biological-Stripping & Residual Spectral Analysis (enhm)
鐢ㄤ簬鍒嗘瀽澶у牎绀侀暱妫樻捣鏄?COTS)鐖嗗彂瑙勫緥

鍥涙鍒嗘瀽妗嗘灦锛?
1. 鏁版嵁灞傜骇涓庢竻娲?- 浣跨敤绀佸尯绾у埆Manta Tow鏁版嵁
2. 鐢熸€侀浂鍋囪妯″瀷 (ENHM) - 鍓ョ椋熺墿闄愬埗鏁堝簲
3. 娈嬪樊DLS鍛ㄦ湡鍒嗘瀽 (enhm) - 鎻愬彇鐜椹卞姩鍛ㄦ湡
4. 鏃剁┖妯″紡璇嗗埆 - 鍒嗘瀽鐖嗗彂浼犳挱鐗瑰緛
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is editable in AI
import warnings
warnings.filterwarnings('ignore')

# 璁剧疆璺緞 (鑴氭湰鍦?scripts/ 鐩綍)
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
BASE_DIR = PROJECT_ROOT / 'data' / 'reef_surveys'  # Reef CSV data
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR = PROJECT_ROOT / 'results' / 'diagnostics'
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 姝ラ涓€锛氭暟鎹眰绾т笌娓呮礂 (Data Selection & Hierarchy)
# =============================================================================

def load_all_reef_data():
    """
    鍔犺浇鎵€鏈夌鍖烘暟鎹紝璇嗗埆閲嶅鏂囦欢锛岀瓫閫夋湁COTS璁板綍鐨勭鍖?
    
    Returns:
        reef_data: dict, {reef_name: {'cots': df, 'coral': df}}
    """
    print("=" * 60)
    print("姝ラ1: 鏁版嵁灞傜骇涓庢竻娲?)
    print("=" * 60)
    
    # 鑾峰彇鎵€鏈塁SV鏂囦欢
    csv_files = list(BASE_DIR.glob('*.csv'))
    
    # 鍒嗙被鏂囦欢
    cots_files = {}  # reef_name -> [file_paths]
    coral_files = {}  # reef_name -> [file_paths]
    
    for f in csv_files:
        fname = f.name
        # 璇嗗埆COTS鏂囦欢
        if '_cots_modelled_' in fname:
            # 鎻愬彇绀佸尯鍚嶇О
            reef_name = fname.split('_Manta Tow_cots_modelled_')[0]
            if reef_name not in cots_files:
                cots_files[reef_name] = []
            cots_files[reef_name].append(f)
        # 璇嗗埆鐝婄憵瑕嗙洊鏂囦欢 (line_chart)
        elif '_line_chart_modelled_' in fname:
            reef_name = fname.split('_Manta Tow_line_chart_modelled_')[0]
            if reef_name not in coral_files:
                coral_files[reef_name] = []
            coral_files[reef_name].append(f)
    
    print(f"鍙戠幇 {len(cots_files)} 涓鍖虹殑COTS鏁版嵁")
    print(f"鍙戠幇 {len(coral_files)} 涓鍖虹殑鐝婄憵鏁版嵁")
    
    # 璇嗗埆閲嶅鏂囦欢骞舵竻鐞?
    duplicates_removed = 0
    for reef_name, files in cots_files.items():
        if len(files) > 1:
            # 淇濈暀涓嶅甫(1)鐨勬枃浠?
            primary = [f for f in files if '(1)' not in f.name]
            if primary:
                cots_files[reef_name] = primary[:1]
                duplicates_removed += len(files) - 1
            else:
                cots_files[reef_name] = files[:1]
                
    for reef_name, files in coral_files.items():
        if len(files) > 1:
            primary = [f for f in files if '(1)' not in f.name]
            if primary:
                coral_files[reef_name] = primary[:1]
                duplicates_removed += len(files) - 1
            else:
                coral_files[reef_name] = files[:1]
    
    print(f"绉婚櫎閲嶅鏂囦欢: {duplicates_removed} 涓?)
    
    # 鎵惧嚭鍚屾椂鏈塁OTS鍜岀強鐟氭暟鎹殑绀佸尯
    valid_reefs = set(cots_files.keys()) & set(coral_files.keys())
    incomplete_reefs = (set(cots_files.keys()) | set(coral_files.keys())) - valid_reefs
    
    print(f"鍚屾椂鎷ユ湁COTS鍜岀強鐟氭暟鎹殑瀹屾暣绀佸尯: {len(valid_reefs)} 涓?)
    print(f"鏁版嵁涓嶅畬鏁寸殑绀佸尯(宸叉帓闄?: {len(incomplete_reefs)} 涓?)
    
    # 鍔犺浇鏁版嵁骞剁瓫閫夋湁COTS璁板綍鐨勭鍖?
    reef_data = {}
    reefs_with_cots = 0
    reefs_without_cots = 0
    
    for reef_name in valid_reefs:
        try:
            cots_df = pd.read_csv(cots_files[reef_name][0])
            coral_df = pd.read_csv(coral_files[reef_name][0])
            
            # 妫€鏌ユ槸鍚︽湁COTS > 0鐨勮褰?
            if 'cots' in cots_df.columns:
                max_cots = cots_df['cots'].max()
            elif 'cotsptow' in cots_df.columns:
                max_cots = cots_df['cotsptow'].astype(float).max()
            else:
                continue
                
            if max_cots > 0:
                reef_data[reef_name] = {
                    'cots': cots_df,
                    'coral': coral_df
                }
                reefs_with_cots += 1
            else:
                reefs_without_cots += 1
                
        except Exception as e:
            print(f"  璀﹀憡: 鏃犳硶鍔犺浇 {reef_name}: {e}")
            continue
    
    print(f"\n绛涢€夌粨鏋?")
    print(f"  鍘嗗彶鏈塁OTS鐖嗗彂璁板綍(>0)鐨勭鍖? {reefs_with_cots} 涓?)
    print(f"  浠庢湭璁板綍鍒癈OTS鐨勭鍖?宸叉帓闄?: {reefs_without_cots} 涓?)
    print(f"  鏈€缁堢敤浜庡垎鏋愮殑绀佸尯鏁伴噺: {len(reef_data)} 涓?)
    
    return reef_data


def merge_reef_timeseries(reef_data):
    """
    鍚堝苟姣忎釜绀佸尯鐨凜OTS鍜岀強鐟氭椂闂村簭鍒楁暟鎹?
    
    Returns:
        merged_data: dict, {reef_name: DataFrame with year, cots, coral columns}
    """
    merged_data = {}
    
    for reef_name, data in reef_data.items():
        cots_df = data['cots'].copy()
        coral_df = data['coral'].copy()
        
        # 鎻愬彇鍏抽敭鍒?
        # COTS: year, cotsptow OR cots
        year_col = 'year' if 'year' in cots_df.columns else 'Year'
        cots_df.rename(columns={year_col: 'year'}, inplace=True)
        cots_df['year'] = cots_df['year'].astype(int)
        
        cots_col = 'cots' if 'cots' in cots_df.columns else 'cotsptow'
        if cots_col not in cots_df.columns:
             print(f"Skipping {reef_name}: No cots/cotsptow column")
             continue
        cots_df['cots_density'] = cots_df[cots_col].apply(pd.to_numeric, errors='coerce')
        
        # Coral: report_year, median (HC)
        if 'report_year' in coral_df.columns:
             coral_year_col = 'report_year'
        elif 'year' in coral_df.columns:
             coral_year_col = 'year'
        elif 'Year' in coral_df.columns:
             coral_year_col = 'Year'
        elif 'date' in coral_df.columns:
             # Try parsing date
             try:
                 coral_df['parsed_year'] = pd.to_numeric(coral_df['date'], errors='coerce').astype(int)
             except:
                 coral_df['parsed_year'] = pd.to_datetime(coral_df['date'], errors='coerce').dt.year
             coral_year_col = 'parsed_year'
        else:
             print(f"Skipping {reef_name}: No year column in Coral")
             continue
             
        coral_df.rename(columns={coral_year_col: 'year'}, inplace=True)
        coral_df['year'] = coral_df['year'].astype(int)

        if 'median' in coral_df.columns:
             coral_val_col = 'median'
        elif 'coral_cover' in coral_df.columns:
             coral_val_col = 'coral_cover'
        elif 'mean' in coral_df.columns:
             coral_val_col = 'mean'
        else:
             # Fallback
             coral_val_col = coral_df.columns[1]

        coral_df['coral_cover'] = coral_df[coral_val_col].apply(pd.to_numeric, errors='coerce')
        
        # 鍩轰簬骞翠唤鍚堝苟
        cots_yearly = cots_df.groupby('year')['cots_density'].mean().reset_index()
        coral_yearly = coral_df.groupby('year')['coral_cover'].mean().reset_index()
        
        merged = pd.merge(cots_yearly, coral_yearly, on='year', how='inner')
        
        if len(merged) >= 5:  # 鑷冲皯闇€瑕?涓暟鎹偣 (Methodology: >= 5 years)
            # 娣诲姞鍏冩暟鎹?
            merged['reef_name'] = reef_name
            if 'nrm_region' in cots_df.columns:
                merged['region'] = cots_df['nrm_region'].iloc[0]
            else:
                merged['region'] = 'Unknown'
            if 'a_sector' in cots_df.columns:
                merged['sector'] = cots_df['a_sector'].iloc[0]
            else:
                merged['sector'] = 'Unknown'
                
            merged_data[reef_name] = merged
    
    print(f"\n鍚堝苟鍚庢湁鏁堢鍖烘暟閲? {len(merged_data)}")
    
    return merged_data


# =============================================================================
# 姝ラ浜岋細鏋勫缓鐢熸€侀浂鍋囪妯″瀷 (Ecological Null Hypothesis Model)
# =============================================================================

def find_optimal_lag(merged_data, max_lag=10, method='cross_correlation'):
    """
    浠庣珯鐐规暟鎹腑鑷姩璁＄畻鏈€浼樻粸鍚庡勾鏁?
    
    鏂规硶锛?
    1. cross_correlation: 璁＄畻姣忎釜绀佸尯COTS涓庢粸鍚庣強鐟氱殑浜ゅ弶鐩稿叧
    2. regression_r2: 娴嬭瘯涓嶅悓婊炲悗涓嬬殑鍥炲綊R^2
    
    Returns:
        optimal_lag: int, 鏁翠綋鏈€浼樻粸鍚庡勾鏁?
        reef_lags: dict, 姣忎釜绀佸尯鐨勬渶浼樻粸鍚?
        lag_analysis: dict, 婊炲悗鍒嗘瀽璇︾粏缁撴灉
    """
    print("\n" + "-" * 50)
    print("璁＄畻鏈€浼樻粸鍚庡勾鏁?(Optimal Lag Calculation)")
    print("-" * 50)
    
    reef_lags = {}
    all_correlations = {lag: [] for lag in range(1, max_lag + 1)}
    all_r2_scores = {lag: [] for lag in range(1, max_lag + 1)}
    
    for reef_name, df in sorted(merged_data.items()):
        df = df.sort_values('year').reset_index(drop=True)
        
        # 瀵规暟鍙樻崲COTS
        log_cots = np.log1p(df['cots_density'].values)
        coral = df['coral_cover'].values
        
        if len(df) < max_lag + 5: # 瑕佹眰鏈€灏忛暱搴︿负 15 (max_lag 10 + 5)
            continue
        
        best_lag = 1
        best_score = -np.inf
        
        for lag in range(1, max_lag + 1):
            if lag >= len(df) - 2:
                continue
                
            # 鍒涘缓婊炲悗鏁版嵁
            coral_lagged = coral[:-lag]
            cots_current = log_cots[lag:]
            
            if len(coral_lagged) < 3:
                continue
            
            # 鏂规硶1: 浜ゅ弶鐩稿叧
            if method == 'cross_correlation':
                corr = np.corrcoef(coral_lagged, cots_current)[0, 1]
                if not np.isnan(corr):
                    all_correlations[lag].append(corr)
                    if corr > best_score:
                        best_score = corr
                        best_lag = lag
            
            # 鏂规硶2: 鍥炲綊R虏
            else:
                X = coral_lagged.reshape(-1, 1)
                y = cots_current
                beta = np.sum(X.flatten() * y) / (np.sum(X.flatten() ** 2) + 1e-10)
                y_pred = beta * X.flatten()
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                all_r2_scores[lag].append(r2)
                if r2 > best_score:
                    best_score = r2
                    best_lag = lag
        
        reef_lags[reef_name] = {
            'optimal_lag': best_lag,
            'score': best_score
        }
    
    # 璁＄畻鍚勬粸鍚庡勾鏁扮殑骞冲潎鐩稿叧/R^2
    lag_means = {}
    for lag in range(1, max_lag + 1):
        if method == 'cross_correlation':
            values = all_correlations[lag]
        else:
            values = all_r2_scores[lag]
        
        if values:
            lag_means[lag] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'n': len(values)
            }
    
    # 鎵惧嚭鏁翠綋鏈€浼樻粸鍚?
    if lag_means:
        optimal_lag = max(lag_means.keys(), key=lambda x: lag_means[x]['mean'])
    else:
        optimal_lag = 5  # 榛樿鍊?
    
    # 缁熻鍚勭鍖烘渶浼樻粸鍚庡垎甯?
    reef_optimal_lags = [v['optimal_lag'] for v in reef_lags.values()]
    lag_distribution = {}
    for lag in range(1, max_lag + 1):
        count = sum(1 for l in reef_optimal_lags if l == lag)
        if count > 0:
            lag_distribution[lag] = count
    
    print(f"鍒嗘瀽绀佸尯鏁? {len(reef_lags)}")
    print(f"浣跨敤鏂规硶: {method}")
    print(f"\n鍚勬粸鍚庡勾鏁扮殑骞冲潎鐩稿叧绯绘暟:")
    for lag in sorted(lag_means.keys()):
        stats = lag_means[lag]
        print(f"  Lag {lag:2d}骞? r = {stats['mean']:+.4f} (std={stats['std']:.4f}, n={stats['n']})")
    
    print(f"\n鍚勭鍖烘渶浼樻粸鍚庡垎甯?")
    for lag, count in sorted(lag_distribution.items()):
        pct = 100 * count / len(reef_lags)
        print(f"  Lag {lag:2d}骞? {count:3d} 绀佸尯 ({pct:.1f}%)")
    
    print(f"\n>>> 鏁翠綋鏈€浼樻粸鍚庡勾鏁? {optimal_lag} 骞?<<<")
    print(f"    (骞冲潎鐩稿叧绯绘暟: {lag_means[optimal_lag]['mean']:.4f})")
    
    # 缁樺埗婊炲悗鍒嗘瀽鍥?
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 宸﹀浘: 鍚勬粸鍚庣殑骞冲潎鐩稿叧
    lags = sorted(lag_means.keys())
    means = [lag_means[l]['mean'] for l in lags]
    stds = [lag_means[l]['std'] for l in lags]
    
    axes[0].errorbar(lags, means, yerr=stds, fmt='o-', capsize=4, 
                     color='steelblue', markersize=8)
    axes[0].axvline(optimal_lag, color='red', linestyle='--', lw=2, 
                    label=f'Optimal: {optimal_lag} years')
    axes[0].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[0].set_xlabel('Lag (years)', fontsize=12)
    axes[0].set_ylabel('Mean Correlation (Coral 鈫?COTS)', fontsize=12)
    axes[0].set_title('Cross-Correlation by Lag', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 鍙冲浘: 绀佸尯鏈€浼樻粸鍚庡垎甯?
    lag_counts = [lag_distribution.get(l, 0) for l in range(1, max_lag + 1)]
    colors = ['red' if l == optimal_lag else 'steelblue' for l in range(1, max_lag + 1)]
    axes[1].bar(range(1, max_lag + 1), lag_counts, color=colors, edgecolor='white')
    axes[1].set_xlabel('Optimal Lag (years)', fontsize=12)
    axes[1].set_ylabel('Number of Reefs', fontsize=12)
    axes[1].set_title('Distribution of Reef-Level Optimal Lags', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(DIAGNOSTICS_DIR / 'figure0_optimal_lag_analysis.png', dpi=300, bbox_inches='tight')
    fig.savefig(DIAGNOSTICS_DIR / 'figure0_optimal_lag_analysis.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  淇濆瓨: figure0_optimal_lag_analysis.png")
    
    # Save reef-level lags
    reef_lag_data = []
    for reef, data in reef_lags.items():
        reef_lag_data.append({
            'reef_name': reef,
            'optimal_lag': data['optimal_lag'],
            'correlation': data['score']
        })
    pd.DataFrame(reef_lag_data).to_csv(RESULTS_DIR / 'reef_lags.csv', index=False)
    print(f"  淇濆瓨: reef_lags.csv")
    
    lag_analysis = {
        'optimal_lag': optimal_lag,
        'lag_means': lag_means,
        'lag_distribution': lag_distribution,
        'method': method
    }
    
    return optimal_lag, reef_lags, lag_analysis


def fit_coral_cots_model(merged_data, lag_years=None):
    """
    鏋勫缓鐢熸€侀浂鍋囪妯″瀷锛屽墺绂婚鐗╅檺鍒舵晥搴?
    
    妯″瀷: N_t = 尾 * Coral_{t-蟿} + 蔚_t
    鍏朵腑:
        N_t = log(COTS_t + 1)
        尾 = 鐝婄憵-COTS鍏崇郴绯绘暟
        蟿 = 婊炲悗骞存暟
        蔚_t = 娈嬪樊 (鐜椹卞姩淇″彿)
    
    Returns:
        residual_data: dict, {reef_name: DataFrame with residuals}
        model_results: dict, 妯″瀷鎷熷悎缁撴灉
    """
    print("\n" + "=" * 60)
    print("姝ラ2: 鏋勫缓鐢熸€侀浂鍋囪妯″瀷 (ENHM)")
    print("=" * 60)
    print(f"浣跨敤婊炲悗骞存暟 tau = {lag_years} 骞?)
    
    residual_data = {}
    model_results = {}
    
    for reef_name, df in merged_data.items():
        df = df.sort_values('year').reset_index(drop=True)
        
        # 鍒涘缓婊炲悗鐝婄憵瑕嗙洊
        df['coral_lagged'] = df['coral_cover'].shift(lag_years)
        
        # 瀵规暟鍙樻崲COTS瀵嗗害 (澶勭悊闆惰啫鑳€)
        df['log_cots'] = np.log1p(df['cots_density'])
        
        # 绉婚櫎NA琛?
        valid_df = df.dropna(subset=['coral_lagged', 'log_cots']).copy()
        
        if len(valid_df) < 3:
            continue
        
        # 绾挎€у洖褰? log_cots ~ coral_lagged (寮哄埗杩囧師鐐?
        X = valid_df['coral_lagged'].values.reshape(-1, 1)
        y = valid_df['log_cots'].values
        
        # ENHM 鏍稿績浼樺寲: 璁＄畻钀ュ吇杞寲绯绘暟 (Trophic Conversion Coefficient)
        # 寮哄埗杩囧師鐐?(Intercept=0) 涓斿繀椤婚潪璐?(Beta >= 0)
        # 鐢熸€佸鎰忎箟: 鐝婄憵澧炲姞涓嶅彲鑳藉鑷存捣鏄熷噺灏?(Beta < 0 鏄棤鎰忎箟鐨?
        numerator = np.sum(X.flatten() * y)
        denominator = np.sum(X.flatten() ** 2)
        
        if denominator > 0:
            beta = numerator / denominator
        else:
            beta = 0
            
        # 鏂藉姞闈炶礋绾︽潫 (Ecological Constraint)
        if beta < 0:
            beta = 0
        
        # 棰勬祴鍊?(Biological Baseline) 鍜?娈嬪樊 (Environmental Anomaly)
        y_pred = beta * X.flatten()
        residuals = y - y_pred
        
        # 璁＄畻R^2 (瀵逛簬杩囧師鐐瑰洖褰掞紝浣跨敤鍘熷瀹氫箟)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum(y ** 2) # RTO鐨勬爣鍑哛2瀹氫箟
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # 淇濆瓨缁撴灉
        valid_df = valid_df.copy()
        valid_df['predicted'] = y_pred
        valid_df['residual'] = residuals
        
        residual_data[reef_name] = valid_df
        model_results[reef_name] = {
            'beta': beta,
            'r_squared': r_squared,
            'n_obs': len(valid_df),
            'region': df['region'].iloc[0],
            'sector': df['sector'].iloc[0]
        }
    
    # 姹囨€荤粺璁?
    r2_values = [r['r_squared'] for r in model_results.values()]
    beta_values = [r['beta'] for r in model_results.values()]
    
    print(f"\n妯″瀷鎷熷悎缁撴灉姹囨€?")
    print(f"  鎴愬姛鎷熷悎绀佸尯鏁? {len(model_results)}")
    print(f"  R^2 鍧囧€? {np.mean(r2_values):.3f} (std: {np.std(r2_values):.3f})")
    print(f"  beta (鐝婄憵绯绘暟) 鍧囧€? {np.mean(beta_values):.4f}")
    
    # Load standardized residuals from enhm outputV for downstream figures
    all_res_list = []
    for reef, df in residual_data.items():
        temp = df.copy()
        temp['reef_name'] = reef
        all_res_list.append(temp[['reef_name', 'year', 'residual', 'predicted', 'log_cots']])
    
    pd.concat(all_res_list).to_csv(RESULTS_DIR / 'all_residuals.csv', index=False)
    print(f"  淇濆瓨: all_residuals.csv (鐢ㄤ簬鍚庣画缁樺浘)")
    
    return residual_data, model_results


# =============================================================================
# 姝ラ涓夛細娈嬪樊鐨凞ouble Lomb-Scargle (DLS) 鍛ㄦ湡鍒嗘瀽
# =============================================================================

def double_lomb_scargle_analysis(residual_data, n_bootstrap=1000, significance_level=0.05):
    """
    瀵规瘡涓鍖虹殑娈嬪樊搴忓垪杩涜Double Lomb-Scargle鍛ㄦ湡鍒嗘瀽
    
    DLS鏂规硶浼樺娍:
    1. 澶勭悊闈炲潎鍖€閲囨牱鏁版嵁
    2. 涓嶉渶瑕佹彃鍊?
    3. 鍙垎绂诲涓彔鍔犲懆鏈?
    
    鏄捐憲鎬ф楠?
    浣跨敤鐧藉櫔澹版帓鍒楁楠?(White Noise / Permutation Test)
    """
    print("\n" + "=" * 60)
    print("姝ラ3: 娈嬪樊Double Lomb-Scargle (DLS) 鍛ㄦ湡鍒嗘瀽")
    print("      (鏄捐憲鎬ф楠? 鐧藉櫔澹?闅忔満缃崲闆跺亣璁?")
    print("=" * 60)
    
    # 璁剧疆闅忔満绉嶅瓙纭繚缁撴灉鍙噸澶?(Critical for reproducibility)
    np.random.seed(42)
    
    period_results = {}
    
    # 瀹氫箟鎰熷叴瓒ｇ殑鍛ㄦ湡鑼冨洿 (3-30骞?
    min_period = 3.0
    max_period = 30.0
    
    for reef_name, df in sorted(residual_data.items()):
        # 鎻愬彇鏃堕棿鍜屾畫宸?
        t = df['year'].values.astype(float)
        y = df['residual'].values
        
        if len(t) < 5:
            continue
        
        # 妫€鏌ユ畫宸柟宸紙鎺掗櫎鍏ㄩ浂鎴栬繎闆跺簭鍒楋級
        if np.std(y) < 1e-10:
            print(f"  璺宠繃 {reef_name}: 娈嬪樊鏂瑰樊涓洪浂")
            continue
        
        # 鍒涘缓棰戠巼缃戞牸
        frequency = np.linspace(1/max_period, 1/min_period, 1000)
        
        # 璁＄畻Lomb-Scargle鍛ㄦ湡鍥?
        ls = LombScargle(t, y)
        power = ls.power(frequency)
        
        # 鎵句富宄?
        peak_idx = np.argmax(power)
        peak_freq = frequency[peak_idx]
        peak_period = 1 / peak_freq
        peak_power = power[peak_idx]
        
        # Bootstrap鏄捐憲鎬ф楠?(White Noise)
        bootstrap_powers = []
        for _ in range(n_bootstrap):
            y_shuffled = np.random.permutation(y)
            ls_boot = LombScargle(t, y_shuffled)
            power_boot = ls_boot.power(frequency)
            bootstrap_powers.append(np.max(power_boot))
        
        bootstrap_powers = np.array(bootstrap_powers)
        p_value = np.mean(bootstrap_powers >= peak_power)
        
        # 鍒ゆ柇鏄捐憲鎬?
        is_significant = p_value < significance_level
        
        # 璁＄畻FAP (False Alarm Probability)
        try:
            fap = ls.false_alarm_probability(peak_power)
        except:
            fap = p_value
        
        # Double Lomb-Scargle: 绉婚櫎涓诲懆鏈熷悗瀵绘壘娆＄骇鍛ㄦ湡
        if is_significant:
            # 鎷熷悎骞剁Щ闄や富鍛ㄦ湡
            y_fit = ls.model(t, peak_freq)
            y_residual2 = y - y_fit
            
            # 绗簩娆S鍒嗘瀽
            ls2 = LombScargle(t, y_residual2)
            power2 = ls2.power(frequency)
            
            peak_idx2 = np.argmax(power2)
            peak_freq2 = frequency[peak_idx2]
            peak_period2 = 1 / peak_freq2
            peak_power2 = power2[peak_idx2]
            
            # Bootstrap妫€楠岀浜屽懆鏈?(White Noise)
            bootstrap_powers2 = []
            for _ in range(min(n_bootstrap, 500)):
                y_shuffled2 = np.random.permutation(y_residual2)
                ls_boot2 = LombScargle(t, y_shuffled2)
                power_boot2 = ls_boot2.power(frequency)
                bootstrap_powers2.append(np.max(power_boot2))
            
            p_value2 = np.mean(np.array(bootstrap_powers2) >= peak_power2)
            is_significant2 = p_value2 < significance_level
        else:
            peak_period2 = np.nan
            peak_power2 = np.nan
            p_value2 = 1.0
            is_significant2 = False
        
        period_results[reef_name] = {
            'time': t,
            'residual': y,
            'frequency': frequency,
            'power': power,
            'primary_period': peak_period,
            'primary_power': peak_power,
            'primary_p_value': p_value,
            'primary_significant': is_significant,
            'secondary_period': peak_period2,
            'secondary_power': peak_power2,
            'secondary_p_value': p_value2,
            'secondary_significant': is_significant2,
            'fap': fap
        }
    
    # 姹囨€荤粨鏋?
    significant_reefs = [r for r, v in period_results.items() if v['primary_significant']]
    all_periods = [v['primary_period'] for v in period_results.values() if v['primary_significant']]
    
    # Save period_results.csv for downstream scripts
    print(f"  Saving period_results.csv...")
    period_data = []
    for reef_name, res in period_results.items():
        period_data.append({
            'reef_name': reef_name,
            'primary_period': res['primary_period'],
            'primary_power': res['primary_power'],
            'primary_p_value': res['primary_p_value'],
            'primary_significant': res['primary_significant'],
            'secondary_period': res['secondary_period'],
            'secondary_power': res['secondary_power'],
            'secondary_significant': res['secondary_significant']
        })
    pd.DataFrame(period_data).to_csv(RESULTS_DIR / 'period_results.csv', index=False)
    
    print(f"\nDLS鍒嗘瀽缁撴灉姹囨€?")
    print(f"  鍒嗘瀽绀佸尯鏁? {len(period_results)}")
    print(f"  鏄捐憲鍛ㄦ湡绀佸尯鏁? {len(significant_reefs)} ({100*len(significant_reefs)/len(period_results):.1f}%)")
    
    if all_periods:
        print(f"  涓诲懆鏈熻寖鍥? {np.min(all_periods):.1f} - {np.max(all_periods):.1f} 骞?)
        print(f"  涓诲懆鏈熷潎鍊? {np.mean(all_periods):.1f} 骞?(std: {np.std(all_periods):.1f})")
        print(f"  涓诲懆鏈熶腑浣嶆暟: {np.median(all_periods):.1f} 骞?)
    
    return period_results


# =============================================================================
# 姝ラ鍥涳細鏃剁┖妯″紡璇嗗埆涓庣幆澧冨叧鑱?
# =============================================================================

def analyze_spatiotemporal_patterns(period_results, model_results, residual_data):
    """
    缁熻鍛ㄦ湡棰戠巼鍒嗗竷锛屽垎鏋愭椂绌轰紶鎾ā寮?
    
    Returns:
        pattern_summary: dict, 鏃剁┖妯″紡鍒嗘瀽缁撴灉
    """
    print("\n" + "=" * 60)
    print("姝ラ4: 鏃剁┖妯″紡璇嗗埆涓庣幆澧冨叧鑱?)
    print("=" * 60)
    
    # 鏀堕泦鏈夋樉钁楀懆鏈熺殑绀佸尯鏁版嵁
    sig_data = []
    for reef_name, result in period_results.items():
        if result['primary_significant']:
            model = model_results.get(reef_name, {})
            sig_data.append({
                'reef_name': reef_name,
                'region': model.get('region', 'Unknown'),
                'sector': model.get('sector', 'Unknown'),
                'primary_period': result['primary_period'],
                'primary_power': result['primary_power'],
                'secondary_period': result['secondary_period'] if result['secondary_significant'] else np.nan,
                'r_squared': model.get('r_squared', np.nan)
            })
    
    sig_df = pd.DataFrame(sig_data)
    
    if len(sig_df) == 0:
        print("璀﹀憡: 娌℃湁鍙戠幇鏄捐憲鍛ㄦ湡")
        return {}
    
    # 鎸夊尯鍩熷垎缁勭粺璁?
    print("\n鎸夊尯鍩熺粺璁″懆鏈熷垎甯?")
    region_stats = sig_df.groupby('region').agg({
        'primary_period': ['mean', 'std', 'count'],
        'r_squared': 'mean'
    }).round(2)
    print(region_stats)
    
    # 鍛ㄦ湡鍒嗗竷鐩存柟鍥惧垎鏋?
    periods = sig_df['primary_period'].values
    
    # 璇嗗埆涓昏鍛ㄦ湡宄板€?
    hist, bin_edges = np.histogram(periods, bins=np.arange(3, 31, 2))
    peak_bins = np.argsort(hist)[-3:]  # 鎵惧嚭鍓?涓嘲鍊?
    peak_periods = [(bin_edges[i] + bin_edges[i+1])/2 for i in peak_bins]
    
    print(f"\n涓昏鍛ㄦ湡宄板€间綅缃? {sorted(peak_periods)}")
    
    # ENSO (~5骞? 鍜?IPO (15-30骞? 鍛ㄦ湡妫€娴?
    enso_range = (4, 7)
    ipo_range = (15, 30)
    
    enso_count = np.sum((periods >= enso_range[0]) & (periods <= enso_range[1]))
    ipo_count = np.sum((periods >= ipo_range[0]) & (periods <= ipo_range[1]))
    
    print(f"\nENSO鍛ㄦ湡鑼冨洿 ({enso_range[0]}-{enso_range[1]}骞?: {enso_count} 涓鍖?({100*enso_count/len(periods):.1f}%)")
    print(f"IPO鍛ㄦ湡鑼冨洿 ({ipo_range[0]}-{ipo_range[1]}骞?: {ipo_count} 涓鍖?({100*ipo_count/len(periods):.1f}%)")
    
    pattern_summary = {
        'significant_reefs_df': sig_df,
        'region_stats': region_stats,
        'period_distribution': {
            'hist': hist,
            'bin_edges': bin_edges,
            'peak_periods': sorted(peak_periods)
        },
        'climate_signal': {
            'enso_count': enso_count,
            'enso_percentage': 100 * enso_count / len(periods),
            'ipo_count': ipo_count,
            'ipo_percentage': 100 * ipo_count / len(periods)
        }
    }
    
    return pattern_summary


# =============================================================================
# 鍙鍖栧嚱鏁?
# =============================================================================

def plot_results(residual_data, model_results, period_results, pattern_summary):
    """
    鐢熸垚鍒嗘瀽缁撴灉鍙鍖?
    """
    print("\n" + "=" * 60)
    print("鐢熸垚鍙鍖栫粨鏋?)
    print("=" * 60)
    
    # Figure 1: 妯″瀷鎷熷悎鍜屾畫宸ず渚?(閫夋嫨鏁版嵁鏈€澶氱殑绀佸尯)
    best_reefs = sorted(residual_data.keys(), 
                       key=lambda x: len(residual_data[x]), 
                       reverse=True)[:4]
    
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, reef_name in enumerate(best_reefs):
        ax = axes[idx]
        df = residual_data[reef_name]
        model = model_results[reef_name]
        
        ax.scatter(df['coral_lagged'], df['log_cots'], 
                  c='steelblue', s=60, alpha=0.7, edgecolors='white', lw=0.5)
        
        # 鎷熷悎绾?
        x_line = np.linspace(df['coral_lagged'].min(), df['coral_lagged'].max(), 100)
        y_line = model['beta'] * x_line
        ax.plot(x_line, y_line, 'r-', lw=2, label=f"$\\beta$={model['beta']:.4f}, $R^2$={model['r_squared']:.3f}")
        
        ax.set_xlabel('Lagged Coral Cover (%)', fontsize=12)
        ax.set_ylabel('log(COTS + 1)', fontsize=12)
        ax.set_title(f"{reef_name}\n({model['region']})", fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    fig1.savefig(DIAGNOSTICS_DIR / 'figure1_coral_cots_regression.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  淇濆瓨: figure1_coral_cots_regression.png")
    
    # Figure 2: DLS鍛ㄦ湡鍥剧ず渚?
    sig_reefs = [r for r, v in period_results.items() if v['primary_significant']]
    if sig_reefs:
        example_reefs = sig_reefs[:4]
        
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, reef_name in enumerate(example_reefs):
            ax = axes[idx]
            result = period_results[reef_name]
            
            period = 1 / result['frequency']
            ax.plot(period, result['power'], 'b-', lw=1.5)
            ax.axvline(result['primary_period'], color='r', linestyle='--', lw=2,
                      label=f"Primary: {result['primary_period']:.1f} yr")
            
            if result['secondary_significant']:
                ax.axvline(result['secondary_period'], color='orange', linestyle=':', lw=2,
                          label=f"Secondary: {result['secondary_period']:.1f} yr")
            
            ax.set_xlabel('Period (years)', fontsize=12)
            ax.set_ylabel('Spectral Power', fontsize=12)
            ax.set_title(f"DLS Periodogram: {reef_name}", fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            
            ax.set_xlabel('Period (years)', fontsize=11)
            ax.set_ylabel('Lomb-Scargle Power', fontsize=11)
            ax.set_title(f"{reef_name}\np-value: {result['primary_p_value']:.4f}", fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.set_xlim(3, 30)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2.savefig(DIAGNOSTICS_DIR / 'figure2_dls_periodograms.png', dpi=300, bbox_inches='tight')
        fig2.savefig(DIAGNOSTICS_DIR / 'figure2_dls_periodograms.svg', bbox_inches='tight')
        plt.close(fig2)
        print(f"  淇濆瓨: figure2_dls_periodograms.png")
    
    # Figure 3: 鍛ㄦ湡鍒嗗竷鐩存柟鍥?
    if 'period_distribution' in pattern_summary:
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        periods = pattern_summary['significant_reefs_df']['primary_period'].values
        
        ax.hist(periods, bins=np.arange(3, 31, 2), color='steelblue', 
               edgecolor='white', alpha=0.8)
        
        # 鏍囨敞ENSO鍜孖PO鑼冨洿
        ax.axvspan(4, 7, alpha=0.2, color='orange', label='ENSO Range (4-7 yr)')
        ax.axvspan(14, 20, alpha=0.2, color='green', label='IPO Range (14-20 yr)')
        
        ax.set_xlabel('Primary Period (years)', fontsize=12)
        ax.set_ylabel('Number of Reefs', fontsize=12)
        ax.set_title('Distribution of Significant Periods Across Reefs\n(After Coral-Food Effect Removal)', fontsize=14)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig3.savefig(DIAGNOSTICS_DIR / 'figure3_period_distribution.png', dpi=300, bbox_inches='tight')
        fig3.savefig(DIAGNOSTICS_DIR / 'figure3_period_distribution.svg', bbox_inches='tight')
        plt.close(fig3)
        print(f"  淇濆瓨: figure3_period_distribution.png")
    
    # Figure 4: 鍖哄煙鍛ㄦ湡姣旇緝
    if 'significant_reefs_df' in pattern_summary:
        sig_df = pattern_summary['significant_reefs_df']
        
        if len(sig_df['region'].unique()) > 1:
            fig4, ax = plt.subplots(figsize=(12, 7))
            
            regions = sig_df['region'].unique()
            positions = range(len(regions))
            
            data_by_region = [sig_df[sig_df['region'] == r]['primary_period'].values 
                             for r in regions]
            
            bp = ax.boxplot(data_by_region, positions=positions, widths=0.6,
                           patch_artist=True)
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.set_xlabel('Region', fontsize=12)
            ax.set_ylabel('Primary Period (years)', fontsize=12)
            ax.set_title('Period Distribution by GBR Region', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 娣诲姞ENSO/IPO鍙傝€冪嚎
            ax.axhline(5.5, color='orange', linestyle='--', alpha=0.7, label='ENSO ~5.5 yr')
            ax.axhline(17, color='green', linestyle='--', alpha=0.7, label='IPO ~17 yr')
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            fig4.savefig(DIAGNOSTICS_DIR / 'figure4_regional_periods.png', dpi=300, bbox_inches='tight')
            fig4.savefig(DIAGNOSTICS_DIR / 'figure4_regional_periods.svg', bbox_inches='tight')
            plt.close(fig4)
            print(f"  淇濆瓨: figure4_regional_periods.png")
    
    # Figure 5: R虏鍒嗗竷 (妯″瀷瑙ｉ噴鍔?
    r2_values = [v['r_squared'] for v in model_results.values()]
    
    fig5, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r2_values, bins=20, color='coral', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(r2_values), color='darkred', linestyle='--', lw=2,
              label=f'Mean $R^2$ = {np.mean(r2_values):.3f}')
    ax.set_xlabel('$R^2$ (Coral-COTS Model)', fontsize=12)
    ax.set_ylabel('Number of Reefs', fontsize=12)
    ax.set_title('Model Fit Quality: Food Effect Explanation Power', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig5.savefig(DIAGNOSTICS_DIR / 'figure5_model_r2_distribution.png', dpi=300, bbox_inches='tight')
    fig5.savefig(DIAGNOSTICS_DIR / 'figure5_model_r2_distribution.svg', bbox_inches='tight')
    plt.close(fig5)
    print(f"  淇濆瓨: figure5_model_r2_distribution.png")


def save_results(residual_data, model_results, period_results, pattern_summary):
    """
    淇濆瓨鍒嗘瀽缁撴灉鍒癈SV鏂囦欢
    """
    print("\n" + "=" * 60)
    print("淇濆瓨鍒嗘瀽缁撴灉")
    print("=" * 60)
    
    # 鑾峰彇婊炲悗骞存暟
    lag_years = pattern_summary.get('lag_analysis', {}).get('optimal_lag', 'N/A')
    
    # 淇濆瓨妯″瀷缁撴灉
    model_df = pd.DataFrame([
        {
            'reef_name': k,
            'region': v['region'],
            'sector': v['sector'],
            'n_observations': v['n_obs'],
            'beta_coefficient': v['beta'],
            'r_squared': v['r_squared']
        }
        for k, v in model_results.items()
    ])
    model_df.to_csv(RESULTS_DIR / 'model_results.csv', index=False)
    print(f"  淇濆瓨: model_results.csv ({len(model_df)} 绀佸尯)")
    
    # 淇濆瓨鍛ㄦ湡鍒嗘瀽缁撴灉
    period_df = pd.DataFrame([
        {
            'reef_name': k,
            'primary_period': v['primary_period'],
            'primary_power': v['primary_power'],
            'primary_p_value': v['primary_p_value'],
            'primary_significant': v['primary_significant'],
            'secondary_period': v['secondary_period'],
            'secondary_significant': v['secondary_significant'],
            'fap': v['fap']
        }
        for k, v in period_results.items()
    ])
    period_df.to_csv(RESULTS_DIR / 'period_results.csv', index=False)
    print(f"  淇濆瓨: period_results.csv ({len(period_df)} 绀佸尯)")
    
    # 淇濆瓨婊炲悗鍒嗘瀽缁撴灉
    if 'lag_analysis' in pattern_summary:
        lag_info = pattern_summary['lag_analysis']
        lag_df = pd.DataFrame([
            {'lag_years': lag, **stats}
            for lag, stats in lag_info.get('lag_means', {}).items()
        ])
        lag_df.to_csv(RESULTS_DIR / 'lag_analysis.csv', index=False)
        print(f"  淇濆瓨: lag_analysis.csv")
    
    # 淇濆瓨缁煎悎鍒嗘瀽缁撴灉
    if 'significant_reefs_df' in pattern_summary:
        pattern_summary['significant_reefs_df'].to_csv(
            RESULTS_DIR / 'significant_reefs.csv', index=False
        )
        print(f"  淇濆瓨: significant_reefs.csv")
    
    # 淇濆瓨姹囨€绘姤鍛?
    with open(RESULTS_DIR / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("鐢熺墿鍓ョ-娈嬪樊棰戣氨鍒嗘瀽 (enhm) 姹囨€绘姤鍛奬n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 鏁版嵁姒傚喌\n")
        f.write(f"   - 鍒嗘瀽绀佸尯鎬绘暟: {len(model_results)}\n")
        f.write(f"   - 妯″瀷鎷熷悎鎴愬姛: {len(residual_data)}\n")
        f.write(f"   - 鍛ㄦ湡鍒嗘瀽瀹屾垚: {len(period_results)}\n\n")
        
        f.write("2. 椋熺墿闄愬埗妯″瀷 (ENHM)\n")
        r2_vals = [v['r_squared'] for v in model_results.values()]
        f.write(f"   - 骞冲潎R虏: {np.mean(r2_vals):.3f} (std: {np.std(r2_vals):.3f})\n")
        f.write(f"   - 鏈€浼樻粸鍚庡勾鏁? {lag_years}骞?(浠庣珯鐐规暟鎹绠?\n\n")
        
        f.write("3. 娈嬪樊鍛ㄦ湡鍒嗘瀽 (DLS)\n")
        sig_count = sum(1 for v in period_results.values() if v['primary_significant'])
        f.write(f"   - 鏄捐憲鍛ㄦ湡绀佸尯: {sig_count}/{len(period_results)}\n")
        
        if 'climate_signal' in pattern_summary:
            cs = pattern_summary['climate_signal']
            f.write(f"   - ENSO鑼冨洿(4-7yr): {cs['enso_count']} 绀佸尯 ({cs['enso_percentage']:.1f}%)\n")
            f.write(f"   - IPO鑼冨洿(15-30yr): {cs['ipo_count']} 绀佸尯 ({cs['ipo_percentage']:.1f}%)\n")
    
    print(f"  淇濆瓨: analysis_summary.txt")


# =============================================================================
# 涓荤▼搴?
# =============================================================================

def main():
    """
    鎵ц瀹屾暣鐨凥BRS鍒嗘瀽娴佺▼
    """
    print("\n" + "=" * 70)
    print(" 鐢熺墿鍓ョ-娈嬪樊棰戣氨鍒嗘瀽 (enhm) ")
    print(" Biological-Stripping & Residual Spectral Analysis")
    print("=" * 70)
    
    # 姝ラ1: 鏁版嵁娓呮礂涓庡姞杞?
    reef_data = load_all_reef_data()
    merged_data = merge_reef_timeseries(reef_data)
    
    # 姝ラ2: 璁＄畻鏈€浼樻粸鍚庡勾鏁板苟鏋勫缓鍒嗗眰鐘舵€佺┖闂存ā鍨?
    optimal_lag, reef_lags, lag_analysis = find_optimal_lag(merged_data, max_lag=10)
    residual_data, model_results = fit_coral_cots_model(merged_data, lag_years=optimal_lag)
    
    # 姝ラ3: DLS鍛ㄦ湡鍒嗘瀽
    period_results = double_lomb_scargle_analysis(residual_data, n_bootstrap=500)
    
    # 姝ラ4: 鏃剁┖妯″紡鍒嗘瀽
    pattern_summary = analyze_spatiotemporal_patterns(period_results, model_results, residual_data)
    pattern_summary['lag_analysis'] = lag_analysis
    
    # 鍙鍖栦笌淇濆瓨缁撴灉
    plot_results(residual_data, model_results, period_results, pattern_summary)
    save_results(residual_data, model_results, period_results, pattern_summary)
    
    print("\n" + "=" * 70)
    print(" enhm鍒嗘瀽瀹屾垚! ")
    print(f" 缁撴灉淇濆瓨鑷? {RESULTS_DIR}")
    print("=" * 70)
    
    return residual_data, model_results, period_results, pattern_summary


if __name__ == '__main__':
    main()
