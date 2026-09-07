#!/usr/bin/env python3
"""
Biological-Stripping & Residual Spectral Analysis (BSRA)
用于分析大堡礁长棘海星(COTS)爆发规律

四步分析框架：
1. 数据层级与清洗 - 使用礁区级别Manta Tow数据
2. 生态零假设模型 (ENHM) - 剥离食物限制效应
3. 残差DLS周期分析 (BSRA) - 提取环境驱动周期
4. 时空模式识别 - 分析爆发传播特征
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

# 设置路径 (脚本在 scripts/ 目录)
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
BASE_DIR = PROJECT_ROOT / 'data' / 'reef_surveys'  # Reef CSV data
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR = PROJECT_ROOT / 'results' / 'diagnostics'
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 步骤一：数据层级与清洗 (Data Selection & Hierarchy)
# =============================================================================

def load_all_reef_data():
    """
    加载所有礁区数据，识别重复文件，筛选有COTS记录的礁区
    
    Returns:
        reef_data: dict, {reef_name: {'cots': df, 'coral': df}}
    """
    print("=" * 60)
    print("步骤1: 数据层级与清洗")
    print("=" * 60)
    
    # 获取所有CSV文件
    # Stable ordering makes row-oriented CSV outputs reproducible across filesystems.
    csv_files = sorted(BASE_DIR.glob('*.csv'), key=lambda path: path.name.casefold())
    
    # 分类文件
    cots_files = {}  # reef_name -> [file_paths]
    coral_files = {}  # reef_name -> [file_paths]
    
    for f in csv_files:
        fname = f.name
        # 识别COTS文件
        if '_cots_modelled_' in fname:
            # 提取礁区名称
            reef_name = fname.split('_Manta Tow_cots_modelled_')[0]
            if reef_name not in cots_files:
                cots_files[reef_name] = []
            cots_files[reef_name].append(f)
        # 识别珊瑚覆盖文件 (line_chart)
        elif '_line_chart_modelled_' in fname:
            reef_name = fname.split('_Manta Tow_line_chart_modelled_')[0]
            if reef_name not in coral_files:
                coral_files[reef_name] = []
            coral_files[reef_name].append(f)
    
    print(f"发现 {len(cots_files)} 个礁区的COTS数据")
    print(f"发现 {len(coral_files)} 个礁区的珊瑚数据")
    
    # 识别重复文件并清理
    duplicates_removed = 0
    for reef_name, files in cots_files.items():
        if len(files) > 1:
            # 保留不带(1)的文件
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
    
    print(f"移除重复文件: {duplicates_removed} 个")
    
    # 找出同时有COTS和珊瑚数据的礁区
    valid_reefs = set(cots_files.keys()) & set(coral_files.keys())
    incomplete_reefs = (set(cots_files.keys()) | set(coral_files.keys())) - valid_reefs
    
    print(f"同时拥有COTS和珊瑚数据的完整礁区: {len(valid_reefs)} 个")
    print(f"数据不完整的礁区(已排除): {len(incomplete_reefs)} 个")
    
    # 加载数据并筛选有COTS记录的礁区
    reef_data = {}
    reefs_with_cots = 0
    reefs_without_cots = 0
    
    for reef_name in valid_reefs:
        try:
            cots_df = pd.read_csv(cots_files[reef_name][0])
            coral_df = pd.read_csv(coral_files[reef_name][0])
            
            # 检查是否有COTS > 0的记录
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
            print(f"  警告: 无法加载 {reef_name}: {e}")
            continue
    
    print(f"\n筛选结果:")
    print(f"  历史有COTS爆发记录(>0)的礁区: {reefs_with_cots} 个")
    print(f"  从未记录到COTS的礁区(已排除): {reefs_without_cots} 个")
    print(f"  最终用于分析的礁区数量: {len(reef_data)} 个")
    
    return reef_data


def merge_reef_timeseries(reef_data):
    """
    合并每个礁区的COTS和珊瑚时间序列数据
    
    Returns:
        merged_data: dict, {reef_name: DataFrame with year, cots, coral columns}
    """
    merged_data = {}
    
    for reef_name, data in reef_data.items():
        cots_df = data['cots'].copy()
        coral_df = data['coral'].copy()
        
        # 提取关键列
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
        
        # 基于年份合并
        cots_yearly = cots_df.groupby('year')['cots_density'].mean().reset_index()
        coral_yearly = coral_df.groupby('year')['coral_cover'].mean().reset_index()
        
        # Keep years observed by either series. Calendar-lag alignment is done
        # explicitly later, so coral observations are not discarded merely
        # because COTS was not surveyed in the same report year.
        merged = pd.merge(cots_yearly, coral_yearly, on='year', how='outer')
        merged = merged.dropna(subset=['year']).sort_values('year').reset_index(drop=True)
        
        if merged['cots_density'].notna().sum() >= 5 and merged['coral_cover'].notna().sum() >= 5:
            # 添加元数据
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
    
    print(f"\n合并后有效礁区数量: {len(merged_data)}")
    
    return merged_data


# =============================================================================
# 步骤二：构建生态零假设模型 (Ecological Null Hypothesis Model)
# =============================================================================

def align_calendar_lag(df, lag_years):
    """Pair COTS at year t with coral cover observed at exactly t-lag_years."""
    response = df[['year', 'cots_density']].dropna().copy()
    predictor = df[['year', 'coral_cover']].dropna().copy()
    predictor['year'] = predictor['year'] + int(lag_years)
    predictor = predictor.rename(columns={'coral_cover': 'coral_lagged'})
    aligned = response.merge(predictor, on='year', how='inner')
    return aligned.sort_values('year').reset_index(drop=True)


def find_optimal_lag(merged_data, max_lag=10, method='cross_correlation'):
    """
    从站点数据中自动计算最优滞后年数
    
    方法：
    1. cross_correlation: 计算每个礁区COTS与滞后珊瑚的交叉相关
    2. regression_r2: 测试不同滞后下的回归R^2
    
    Returns:
        optimal_lag: int, 整体最优滞后年数
        reef_lags: dict, 每个礁区的最优滞后
        lag_analysis: dict, 滞后分析详细结果
    """
    print("\n" + "-" * 50)
    print("计算最优滞后年数 (Optimal Lag Calculation)")
    print("-" * 50)
    
    min_pairs = 5
    reef_lags = {}
    reef_scores = {}
    all_correlations = {lag: [] for lag in range(1, max_lag + 1)}
    all_r2_scores = {lag: [] for lag in range(1, max_lag + 1)}
    
    for reef_name, df in sorted(merged_data.items()):
        scores = {}
        
        for lag in range(1, max_lag + 1):
            aligned = align_calendar_lag(df, lag)
            if len(aligned) < min_pairs:
                continue

            coral_lagged = aligned['coral_lagged'].to_numpy(dtype=float)
            cots_current = np.log1p(aligned['cots_density'].to_numpy(dtype=float))
            if np.std(coral_lagged) <= 1e-12 or np.std(cots_current) <= 1e-12:
                continue
            
            if method == 'cross_correlation':
                corr = np.corrcoef(coral_lagged, cots_current)[0, 1]
                if not np.isnan(corr):
                    scores[lag] = float(corr)
            else:
                X = coral_lagged
                y = cots_current
                beta = np.sum(X * y) / (np.sum(X ** 2) + 1e-10)
                y_pred = beta * X
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                scores[lag] = float(r2)

        if scores:
            best_lag = max(scores, key=scores.get)
            reef_scores[reef_name] = scores
            reef_lags[reef_name] = {
                'optimal_lag': best_lag,
                'score': scores[best_lag]
            }

    # Compare system-wide lag means on the same reefs at every candidate lag.
    # This avoids changing reef composition from masquerading as a lag effect.
    comparison_reefs = [
        reef for reef, scores in reef_scores.items()
        if len(scores) == max_lag
    ]
    for reef in comparison_reefs:
        for lag, score in reef_scores[reef].items():
            if method == 'cross_correlation':
                all_correlations[lag].append(score)
            else:
                all_r2_scores[lag].append(score)
    
    # 计算各滞后年数的平均相关/R^2
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
    
    # 找出整体最优滞后
    if lag_means:
        optimal_lag = max(lag_means.keys(), key=lambda x: lag_means[x]['mean'])
    else:
        optimal_lag = 5  # 默认值
    
    # 统计各礁区最优滞后分布
    reef_optimal_lags = [v['optimal_lag'] for v in reef_lags.values()]
    lag_distribution = {}
    for lag in range(1, max_lag + 1):
        count = sum(1 for l in reef_optimal_lags if l == lag)
        if count > 0:
            lag_distribution[lag] = count
    
    print(f"具有至少一个可用滞后的礁区数: {len(reef_lags)}")
    print(f"用于全滞后公平比较的共同礁区数: {len(comparison_reefs)}")
    print(f"使用方法: {method}")
    print(f"\n各滞后年数的平均相关系数:")
    for lag in sorted(lag_means.keys()):
        stats = lag_means[lag]
        print(f"  Lag {lag:2d}年: r = {stats['mean']:+.4f} (std={stats['std']:.4f}, n={stats['n']})")
    
    print(f"\n各礁区最优滞后分布:")
    for lag, count in sorted(lag_distribution.items()):
        pct = 100 * count / len(reef_lags)
        print(f"  Lag {lag:2d}年: {count:3d} 礁区 ({pct:.1f}%)")
    
    print(f"\n>>> 整体最优滞后年数: {optimal_lag} 年 <<<")
    print(f"    (平均相关系数: {lag_means[optimal_lag]['mean']:.4f})")
    
    # 绘制滞后分析图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 各滞后的平均相关
    lags = sorted(lag_means.keys())
    means = [lag_means[l]['mean'] for l in lags]
    stds = [lag_means[l]['std'] for l in lags]
    
    axes[0].errorbar(lags, means, yerr=stds, fmt='o-', capsize=4, 
                     color='steelblue', markersize=8)
    axes[0].axvline(optimal_lag, color='red', linestyle='--', lw=2, 
                    label=f'Optimal: {optimal_lag} years')
    axes[0].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[0].set_xlabel('Lag (years)', fontsize=12)
    axes[0].set_ylabel('Mean Correlation (Coral → COTS)', fontsize=12)
    axes[0].set_title('Cross-Correlation by Lag', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 右图: 礁区最优滞后分布
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
    print(f"  保存: figure0_optimal_lag_analysis.png")
    
    # Save reef-level lags
    reef_lag_data = []
    for reef, data in reef_lags.items():
        reef_lag_data.append({
            'reef_name': reef,
            'optimal_lag': data['optimal_lag'],
            'correlation': data['score']
        })
    pd.DataFrame(reef_lag_data).to_csv(RESULTS_DIR / 'reef_lags.csv', index=False)
    print(f"  保存: reef_lags.csv")
    
    lag_analysis = {
        'optimal_lag': optimal_lag,
        'lag_means': lag_means,
        'lag_distribution': lag_distribution,
        'method': method
    }
    
    return optimal_lag, reef_lags, lag_analysis


def fit_coral_cots_model(merged_data, lag_years=None):
    """
    构建生态零假设模型，剥离食物限制效应
    
    模型: N_t = β * Coral_{t-τ} + ε_t
    其中:
        N_t = log(COTS_t + 1)
        β = 珊瑚-COTS关系系数
        τ = 滞后年数
        ε_t = 残差 (环境驱动信号)
    
    Returns:
        residual_data: dict, {reef_name: DataFrame with residuals}
        model_results: dict, 模型拟合结果
    """
    print("\n" + "=" * 60)
    print("步骤2: 构建生态零假设模型 (ENHM)")
    print("=" * 60)
    print(f"使用滞后年数 tau = {lag_years} 年")
    
    residual_data = {}
    model_results = {}
    
    for reef_name, df in merged_data.items():
        df = df.sort_values('year').reset_index(drop=True)

        # Match by calendar year: COTS(t) is paired only with coral(t-tau).
        valid_df = align_calendar_lag(df, lag_years)
        valid_df['log_cots'] = np.log1p(valid_df['cots_density'])
        
        if len(valid_df) < 3:
            continue
        
        # 线性回归: log_cots ~ coral_lagged (强制过原点)
        X = valid_df['coral_lagged'].values.reshape(-1, 1)
        y = valid_df['log_cots'].values
        
        # ENHM 核心优化: 计算营养转化系数 (Trophic Conversion Coefficient)
        # 强制过原点 (Intercept=0) 且必须非负 (Beta >= 0)
        # 生态学意义: 珊瑚增加不可能导致海星减少 (Beta < 0 是无意义的)
        numerator = np.sum(X.flatten() * y)
        denominator = np.sum(X.flatten() ** 2)
        
        if denominator > 0:
            beta = numerator / denominator
        else:
            beta = 0
            
        # 施加非负约束 (Ecological Constraint)
        if beta < 0:
            beta = 0
        
        # 预测值 (Biological Baseline) 和 残差 (Environmental Anomaly)
        y_pred = beta * X.flatten()
        residuals = y - y_pred
        
        # Report conventional centered R² as the variance-explained diagnostic.
        # Keep the uncentered R² separately because it is sometimes reported for
        # regression through the origin, but it must not be called variance explained.
        ss_res = np.sum(residuals ** 2)
        ss_tot_centered = np.sum((y - np.mean(y)) ** 2)
        ss_tot_uncentered = np.sum(y ** 2)
        r_squared = 1 - ss_res / ss_tot_centered if ss_tot_centered > 0 else np.nan
        r_squared_uncentered = (
            1 - ss_res / ss_tot_uncentered if ss_tot_uncentered > 0 else np.nan
        )
        
        # 保存结果
        valid_df = valid_df.copy()
        valid_df['predicted'] = y_pred
        valid_df['residual'] = residuals
        
        residual_data[reef_name] = valid_df
        model_results[reef_name] = {
            'beta': beta,
            'r_squared': r_squared,
            'r_squared_uncentered': r_squared_uncentered,
            'n_obs': len(valid_df),
            'region': df['region'].iloc[0],
            'sector': df['sector'].iloc[0]
        }
    
    # 汇总统计
    r2_values = [r['r_squared'] for r in model_results.values() if np.isfinite(r['r_squared'])]
    uncentered_r2_values = [
        r['r_squared_uncentered']
        for r in model_results.values()
        if np.isfinite(r['r_squared_uncentered'])
    ]
    beta_values = [r['beta'] for r in model_results.values()]
    
    print(f"\n模型拟合结果汇总:")
    print(f"  成功拟合礁区数: {len(model_results)}")
    print(f"  中心化 R^2 均值: {np.mean(r2_values):.3f} (std: {np.std(r2_values):.3f})")
    print(
        f"  非中心化 R^2 均值: {np.mean(uncentered_r2_values):.3f} "
        f"(std: {np.std(uncentered_r2_values):.3f})"
    )
    print(f"  beta (珊瑚系数) 均值: {np.mean(beta_values):.4f}")
    
    # Load standardized residuals from BSRA outputV for downstream figures
    all_res_list = []
    for reef, df in residual_data.items():
        temp = df.copy()
        temp['reef_name'] = reef
        all_res_list.append(temp[['reef_name', 'year', 'residual', 'predicted', 'log_cots']])
    
    pd.concat(all_res_list).to_csv(RESULTS_DIR / 'all_residuals.csv', index=False)
    print(f"  保存: all_residuals.csv (用于后续绘图)")
    
    return residual_data, model_results


# =============================================================================
# 步骤三：残差的Double Lomb-Scargle (DLS) 周期分析
# =============================================================================

def double_lomb_scargle_analysis(residual_data, n_bootstrap=1000, significance_level=0.05):
    """
    对每个礁区的残差序列进行Double Lomb-Scargle周期分析
    
    DLS方法优势:
    1. 处理非均匀采样数据
    2. 不需要插值
    3. 可分离多个叠加周期
    
    显著性检验:
    使用白噪声排列检验 (White Noise / Permutation Test)
    """
    print("\n" + "=" * 60)
    print("步骤3: 残差Double Lomb-Scargle (DLS) 周期分析")
    print("      (显著性检验: 白噪声/随机置换零假设)")
    print("=" * 60)
    
    # 设置随机种子确保结果可重复 (Critical for reproducibility)
    np.random.seed(42)
    
    period_results = {}
    
    # 定义感兴趣的周期范围 (3-30年)
    min_period = 3.0
    max_period = 30.0
    
    for reef_name, df in sorted(residual_data.items()):
        # 提取时间和残差
        t = df['year'].values.astype(float)
        y = df['residual'].values
        
        if len(t) < 5:
            continue
        
        # 检查残差方差（排除全零或近零序列）
        if np.std(y) < 1e-10:
            print(f"  跳过 {reef_name}: 残差方差为零")
            continue
        
        # 创建频率网格
        frequency = np.linspace(1/max_period, 1/min_period, 1000)
        
        # 计算Lomb-Scargle周期图
        ls = LombScargle(t, y)
        power = ls.power(frequency)
        
        # 找主峰
        peak_idx = np.argmax(power)
        peak_freq = frequency[peak_idx]
        peak_period = 1 / peak_freq
        peak_power = power[peak_idx]
        
        # Bootstrap显著性检验 (White Noise)
        bootstrap_powers = []
        for _ in range(n_bootstrap):
            y_shuffled = np.random.permutation(y)
            ls_boot = LombScargle(t, y_shuffled)
            power_boot = ls_boot.power(frequency)
            bootstrap_powers.append(np.max(power_boot))
        
        bootstrap_powers = np.array(bootstrap_powers)
        # (k+1)/(N+1) keeps the empirical p-value strictly positive for a finite
        # number of permutations.
        p_value = (np.sum(bootstrap_powers >= peak_power) + 1) / (n_bootstrap + 1)
        
        # 判断显著性
        is_significant = p_value < significance_level
        
        # 计算FAP (False Alarm Probability)
        try:
            fap = ls.false_alarm_probability(peak_power)
        except:
            fap = p_value
        
        # Double Lomb-Scargle: 移除主周期后寻找次级周期
        if is_significant:
            # 拟合并移除主周期
            y_fit = ls.model(t, peak_freq)
            y_residual2 = y - y_fit
            
            # 第二次LS分析
            ls2 = LombScargle(t, y_residual2)
            power2 = ls2.power(frequency)
            
            peak_idx2 = np.argmax(power2)
            peak_freq2 = frequency[peak_idx2]
            peak_period2 = 1 / peak_freq2
            peak_power2 = power2[peak_idx2]
            
            # Bootstrap检验第二周期 (White Noise)
            bootstrap_powers2 = []
            for _ in range(min(n_bootstrap, 500)):
                y_shuffled2 = np.random.permutation(y_residual2)
                ls_boot2 = LombScargle(t, y_shuffled2)
                power_boot2 = ls_boot2.power(frequency)
                bootstrap_powers2.append(np.max(power_boot2))
            
            p_value2 = (np.sum(np.array(bootstrap_powers2) >= peak_power2) + 1) / (
                min(n_bootstrap, 500) + 1
            )
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
    
    # 汇总结果
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
    
    print(f"\nDLS分析结果汇总:")
    print(f"  分析礁区数: {len(period_results)}")
    print(f"  显著周期礁区数: {len(significant_reefs)} ({100*len(significant_reefs)/len(period_results):.1f}%)")
    
    if all_periods:
        print(f"  主周期范围: {np.min(all_periods):.1f} - {np.max(all_periods):.1f} 年")
        print(f"  主周期均值: {np.mean(all_periods):.1f} 年 (std: {np.std(all_periods):.1f})")
        print(f"  主周期中位数: {np.median(all_periods):.1f} 年")
    
    return period_results


# =============================================================================
# 步骤四：时空模式识别与环境关联
# =============================================================================

def analyze_spatiotemporal_patterns(period_results, model_results, residual_data):
    """
    统计周期频率分布，分析时空传播模式
    
    Returns:
        pattern_summary: dict, 时空模式分析结果
    """
    print("\n" + "=" * 60)
    print("步骤4: 时空模式识别与环境关联")
    print("=" * 60)
    
    # 收集有显著周期的礁区数据
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
        print("警告: 没有发现显著周期")
        return {}
    
    # 按区域分组统计
    print("\n按区域统计周期分布:")
    region_stats = sig_df.groupby('region').agg({
        'primary_period': ['mean', 'std', 'count'],
        'r_squared': 'mean'
    }).round(2)
    print(region_stats)
    
    # 周期分布直方图分析
    periods = sig_df['primary_period'].values
    
    # 识别主要周期峰值
    hist, bin_edges = np.histogram(periods, bins=np.arange(3, 31, 2))
    peak_bins = np.argsort(hist)[-3:]  # 找出前3个峰值
    peak_periods = [(bin_edges[i] + bin_edges[i+1])/2 for i in peak_bins]
    
    print(f"\n主要周期峰值位置: {sorted(peak_periods)}")
    
    # ENSO (~5年) 和 IPO (15-30年) 周期检测
    enso_range = (3, 7)
    ipo_range = (15, 30)
    
    enso_count = np.sum((periods >= enso_range[0]) & (periods <= enso_range[1]))
    ipo_count = np.sum((periods >= ipo_range[0]) & (periods <= ipo_range[1]))
    
    print(f"\nENSO周期范围 ({enso_range[0]}-{enso_range[1]}年): {enso_count} 个礁区 ({100*enso_count/len(periods):.1f}%)")
    print(f"IPO周期范围 ({ipo_range[0]}-{ipo_range[1]}年): {ipo_count} 个礁区 ({100*ipo_count/len(periods):.1f}%)")
    
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
# 可视化函数
# =============================================================================

def plot_results(residual_data, model_results, period_results, pattern_summary):
    """
    生成分析结果可视化
    """
    print("\n" + "=" * 60)
    print("生成可视化结果")
    print("=" * 60)
    
    # Figure 1: 模型拟合和残差示例 (选择数据最多的礁区)
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
        
        # 拟合线
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
    print(f"  保存: figure1_coral_cots_regression.png")
    
    # Figure 2: DLS周期图示例
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
        print(f"  保存: figure2_dls_periodograms.png")
    
    # Figure 3: 周期分布直方图
    if 'period_distribution' in pattern_summary:
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        periods = pattern_summary['significant_reefs_df']['primary_period'].values
        
        ax.hist(periods, bins=np.arange(3, 31, 2), color='steelblue', 
               edgecolor='white', alpha=0.8)
        
        # 标注ENSO和IPO范围
        ax.axvspan(3, 7, alpha=0.2, color='orange', label='ENSO-scale Range (3-7 yr)')
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
        print(f"  保存: figure3_period_distribution.png")
    
    # Figure 4: 区域周期比较
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
            
            # 添加ENSO/IPO参考线
            ax.axhline(5.5, color='orange', linestyle='--', alpha=0.7, label='ENSO ~5.5 yr')
            ax.axhline(17, color='green', linestyle='--', alpha=0.7, label='IPO ~17 yr')
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            fig4.savefig(DIAGNOSTICS_DIR / 'figure4_regional_periods.png', dpi=300, bbox_inches='tight')
            fig4.savefig(DIAGNOSTICS_DIR / 'figure4_regional_periods.svg', bbox_inches='tight')
            plt.close(fig4)
            print(f"  保存: figure4_regional_periods.png")
    
    # Figure 5: R²分布 (模型解释力)
    r2_values = [v['r_squared'] for v in model_results.values() if np.isfinite(v['r_squared'])]
    
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
    print(f"  保存: figure5_model_r2_distribution.png")


def save_results(residual_data, model_results, period_results, pattern_summary):
    """
    保存分析结果到CSV文件
    """
    print("\n" + "=" * 60)
    print("保存分析结果")
    print("=" * 60)
    
    # 获取滞后年数
    lag_years = pattern_summary.get('lag_analysis', {}).get('optimal_lag', 'N/A')
    
    # 保存模型结果
    model_df = pd.DataFrame([
        {
            'reef_name': k,
            'region': v['region'],
            'sector': v['sector'],
            'n_observations': v['n_obs'],
            'beta_coefficient': v['beta'],
            'r_squared': v['r_squared'],
            'r_squared_uncentered': v['r_squared_uncentered']
        }
        for k, v in model_results.items()
    ])
    model_df.to_csv(RESULTS_DIR / 'model_results.csv', index=False)
    print(f"  保存: model_results.csv ({len(model_df)} 礁区)")
    
    # 保存周期分析结果
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
    print(f"  保存: period_results.csv ({len(period_df)} 礁区)")
    
    # 保存滞后分析结果
    if 'lag_analysis' in pattern_summary:
        lag_info = pattern_summary['lag_analysis']
        lag_df = pd.DataFrame([
            {'lag_years': lag, **stats}
            for lag, stats in lag_info.get('lag_means', {}).items()
        ])
        lag_df.to_csv(RESULTS_DIR / 'lag_analysis.csv', index=False)
        print(f"  保存: lag_analysis.csv")
    
    # 保存综合分析结果
    if 'significant_reefs_df' in pattern_summary:
        pattern_summary['significant_reefs_df'].to_csv(
            RESULTS_DIR / 'significant_reefs.csv', index=False
        )
        print(f"  保存: significant_reefs.csv")
    
    # 保存汇总报告
    with open(RESULTS_DIR / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("生物剥离-残差频谱分析 (BSRA) 汇总报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 数据概况\n")
        f.write(f"   - 分析礁区总数: {len(model_results)}\n")
        f.write(f"   - 模型拟合成功: {len(residual_data)}\n")
        f.write(f"   - 周期分析完成: {len(period_results)}\n\n")
        
        f.write("2. 食物限制模型 (ENHM)\n")
        r2_vals = [v['r_squared'] for v in model_results.values() if np.isfinite(v['r_squared'])]
        uncentered_r2_vals = [
            v['r_squared_uncentered']
            for v in model_results.values()
            if np.isfinite(v['r_squared_uncentered'])
        ]
        f.write(f"   - 平均中心化R²: {np.mean(r2_vals):.3f} (std: {np.std(r2_vals):.3f})\n")
        f.write(
            f"   - 平均非中心化R²: {np.mean(uncentered_r2_vals):.3f} "
            f"(std: {np.std(uncentered_r2_vals):.3f})\n"
        )
        f.write(f"   - 最优滞后年数: {lag_years}年 (从站点数据计算)\n\n")
        
        f.write("3. 残差周期分析 (DLS)\n")
        sig_count = sum(1 for v in period_results.values() if v['primary_significant'])
        f.write(f"   - 显著周期礁区: {sig_count}/{len(period_results)}\n")
        
        if 'climate_signal' in pattern_summary:
            cs = pattern_summary['climate_signal']
            f.write(f"   - ENSO范围(4-7yr): {cs['enso_count']} 礁区 ({cs['enso_percentage']:.1f}%)\n")
            f.write(f"   - IPO范围(15-30yr): {cs['ipo_count']} 礁区 ({cs['ipo_percentage']:.1f}%)\n")
    
    print(f"  保存: analysis_summary.txt")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """
    执行完整的HBRS分析流程
    """
    print("\n" + "=" * 70)
    print(" 生物剥离-残差频谱分析 (BSRA) ")
    print(" Biological-Stripping & Residual Spectral Analysis")
    print("=" * 70)
    
    # 步骤1: 数据清洗与加载
    reef_data = load_all_reef_data()
    merged_data = merge_reef_timeseries(reef_data)
    
    # 步骤2: 计算最优滞后年数并构建分层状态空间模型
    optimal_lag, reef_lags, lag_analysis = find_optimal_lag(merged_data, max_lag=10)
    residual_data, model_results = fit_coral_cots_model(merged_data, lag_years=optimal_lag)
    
    # 步骤3: DLS周期分析
    period_results = double_lomb_scargle_analysis(residual_data, n_bootstrap=5000)
    
    # 步骤4: 时空模式分析
    pattern_summary = analyze_spatiotemporal_patterns(period_results, model_results, residual_data)
    pattern_summary['lag_analysis'] = lag_analysis
    
    # 可视化与保存结果
    plot_results(residual_data, model_results, period_results, pattern_summary)
    save_results(residual_data, model_results, period_results, pattern_summary)
    
    print("\n" + "=" * 70)
    print(" BSRA分析完成! ")
    print(f" 结果保存至: {RESULTS_DIR}")
    print("=" * 70)
    
    return residual_data, model_results, period_results, pattern_summary


if __name__ == '__main__':
    main()
