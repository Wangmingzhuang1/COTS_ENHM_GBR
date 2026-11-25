#!/usr/bin/env python3
"""
Biological-Stripping & Residual Spectral Analysis (BSRA)
用于分析大堡礁长棘海星(COTS)爆发规律

四步分析框架：
1. 数据层级与清洗 - 使用礁区级别Manta Tow数据
2. 生态零假设模型 (ENHM) - 剥离食物限制效应
3. 残差 Lomb-Scargle 周期分析 (BSRA) - 提取环境驱动周期
4. 时空模式识别 - 分析爆发传播特征
"""

import numpy as np
import pandas as pd
from pathlib import Path
from analysis_inputs import align_survey_lag, ecological_observations, reef_series
from config import ECOLOGY_OBSERVATIONS
from spectral_inference import (
    block_permutation_samples,
    holm_adjust,
    peak_p_value,
    red_noise_samples,
    spectrum,
)

PRIMARY_LAG_YEARS = 3
LAG_TOLERANCE_DAYS = 183

# 设置路径 (脚本在 scripts/ 目录)
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
BASE_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR = PROJECT_ROOT / 'results' / 'diagnostics'
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 步骤一：数据层级与清洗 (Data Selection & Hierarchy)
# =============================================================================


def find_optimal_lag(merged_data, max_lag=4):
    """Describe lag associations on identical response dates across candidates."""
    candidates = list(range(1, max_lag + 1))
    site_scores = {}
    for reef_name, frame in sorted(merged_data.items()):
        matches = [align_survey_lag(frame, lag, LAG_TOLERANCE_DAYS).set_index('survey_date') for lag in candidates]
        common = matches[0].index
        for match in matches[1:]:
            common = common.intersection(match.index)
        if len(common) < 5:
            continue
        y = matches[0].loc[common, 'log_cots'].to_numpy()
        x = [match.loc[common, 'coral_lagged'].to_numpy() for match in matches]
        if np.std(y) <= 1e-12 or any(np.std(value) <= 1e-12 for value in x):
            continue
        site_scores[reef_name] = {
            lag: float(np.corrcoef(value, y)[0, 1]) for lag, value in zip(candidates, x)
        }

    lag_means = {}
    reef_lags = {}
    if site_scores:
        for lag in candidates:
            values = [scores[lag] for scores in site_scores.values()]
            lag_means[lag] = {'mean': np.mean(values), 'std': np.std(values), 'n': len(values)}
        for reef_name, scores in site_scores.items():
            best = max(scores, key=scores.get)
            reef_lags[reef_name] = {'optimal_lag': best, 'correlation': scores[best]}
    reef_lag_frame = pd.DataFrame.from_dict(reef_lags, orient='index').rename_axis('reef_name').reset_index()
    reef_lag_frame.reindex(columns=['reef_name', 'optimal_lag', 'correlation']).to_csv(RESULTS_DIR / 'reef_lags.csv', index=False)
    print(f'Dated lag diagnostic: {len(site_scores)} reefs with common COTS survey dates')
    return {
        'optimal_lag': max(lag_means, key=lambda lag: lag_means[lag]['mean']) if lag_means else None,
        'primary_lag': PRIMARY_LAG_YEARS,
        'lag_means': lag_means,
    }


def fit_coral_cots_model(merged_data, lag_years=None):
    """
    构建生态零假设模型，剥离食物限制效应
    
    模型: N_t = β * Coral_{t-τ} + ε_t
    其中:
        N_t = log(COTS_t + 1)
        β = 珊瑚-COTS关系系数
        τ = 滞后年数
        ε_t = 珊瑚关联基线未解释的丰度变化
    
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
        valid_df = align_survey_lag(df, lag_years, LAG_TOLERANCE_DAYS)
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
        
        # Fit and residual on the same dated observations
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
        all_res_list.append(temp[['reef_name', 'year', 'report_year', 'survey_date', 'time_years',
                                  'coral_date', 'actual_lag_days', 'cots_density', 'coral_lagged',
                                  'source_file', 'source_row', 'coral_source_file', 'coral_source_row',
                                  'residual', 'predicted', 'log_cots']])
    
    pd.concat(all_res_list).to_csv(RESULTS_DIR / 'all_residuals.csv', index=False)
    print(f"  保存: all_residuals.csv (用于后续绘图)")
    
    return residual_data, model_results


# =============================================================================
# 步骤三：残差的 Lomb-Scargle 周期分析
# =============================================================================


def screen_residual_periods(residual_data, n_bootstrap=5000, significance_level=0.05):
    """Screen dominant residual periods under white-noise permutation (primary),
    plus 3-year calendar-block permutation and continuous-time AR(1) red-noise nulls."""
    rng = np.random.default_rng(42)
    results = {}
    for reef_name, frame in sorted(residual_data.items()):
        time = frame['time_years'].to_numpy()
        values = frame['residual'].to_numpy()
        result = spectrum(time, values)
        if result is None:
            continue
        frequency, power = result
        primary = int(np.nanargmax(power))
        permuted = np.asarray([rng.permutation(values) for _ in range(n_bootstrap)])
        blocked = block_permutation_samples(time, values, rng, n_bootstrap, block_years=3.0)
        simulated, decay = red_noise_samples(time, values, rng, n_bootstrap)
        wn_p = peak_p_value(time, frequency, power[primary], permuted)
        blk_p = peak_p_value(time, frequency, power[primary], blocked)
        ar1_p = peak_p_value(time, frequency, power[primary], simulated)
        results[reef_name] = {
            'time': time, 'residual': values, 'frequency': frequency, 'power': power,
            'primary_period': 1.0 / frequency[primary], 'primary_power': float(power[primary]),
            'primary_p_value': wn_p,
            'white_noise_p_value': wn_p,
            'block_p_value': blk_p,
            'ar1_p_value': ar1_p,
            'span_years': float(np.ptp(time)), 'n_survey_dates': len(time), 'red_noise_decay_years': decay,
            'primary_significant': wn_p < significance_level,
            'secondary_period': np.nan, 'secondary_significant': False, 'fap': np.nan,
        }
    ar1_adj = holm_adjust([value['ar1_p_value'] for value in results.values()])
    for value, p_adjusted in zip(results.values(), ar1_adj):
        value['ar1_p_adjusted'] = float(p_adjusted)
        value['primary_p_adjusted'] = float(p_adjusted)
    return results


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
    
    sig_df = pd.DataFrame(sig_data, columns=['reef_name', 'region', 'sector', 'primary_period', 'primary_power', 'secondary_period', 'r_squared'])
    
    if len(sig_df) == 0:
        print("没有礁区通过连续时间红噪声与礁间 Holm 校正")
        return {'significant_reefs_df': sig_df}
    
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

def save_results(residual_data, model_results, period_results, pattern_summary):
    """
    保存分析结果到CSV文件
    """
    print("\n" + "=" * 60)
    print("保存分析结果")
    print("=" * 60)
    
    # 获取滞后年数
    lag_years = pattern_summary['lag_analysis']['primary_lag']
    
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
            'white_noise_p_value': v['white_noise_p_value'],
            'block_p_value': v['block_p_value'],
            'ar1_p_value': v['ar1_p_value'],
            'ar1_p_adjusted': v['ar1_p_adjusted'],
            'primary_p_adjusted': v['primary_p_adjusted'],
            'span_years': v['span_years'],
            'n_survey_dates': v['n_survey_dates'],
            'red_noise_decay_years': v['red_noise_decay_years'],
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
            {'lag_years': lag, 'primary': lag == lag_years, **stats}
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
        f.write(f"   - 名义基准滞后: {lag_years}年 ± {LAG_TOLERANCE_DAYS}天，按实际调查日期配对\n")
        f.write(f"   - 生态输入: {ECOLOGY_OBSERVATIONS}\n\n")
        
        f.write("3. 残差周期分析（主筛选：白噪声置换 N=5000；附 3 年块置换与连续时间 AR(1) 零假设层级）\n")
        sig_count = sum(1 for v in period_results.values() if v['primary_significant'])
        blk_count = sum(1 for v in period_results.values() if v['block_p_value'] < 0.05)
        ar1_count = sum(1 for v in period_results.values() if v['ar1_p_value'] < 0.05)
        f.write(f"   - 显著周期礁区 (白噪声 p<0.05): {sig_count}/{len(period_results)}\n")
        f.write(f"   - 3年块置换检出 (p<0.05): {blk_count}/{len(period_results)}\n")
        f.write(f"   - 连续时间AR(1)保守上界 (p<0.05): {ar1_count}/{len(period_results)}\n")
        
        long_detected = sum(v['primary_significant'] and 15 <= v['primary_period'] <= 30 for v in period_results.values())
        f.write(f"   - 15–30 年频段检出: {long_detected} 礁区\n")
        if 'climate_signal' in pattern_summary:
            cs = pattern_summary['climate_signal']
            f.write(f"   - 3–7 年频段: {cs['enso_count']} 礁区 ({cs['enso_percentage']:.1f}%)\n")
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
    cots, coral = ecological_observations()
    print(f'Validated AIMS inputs: {len(cots)} COTS surveys; {len(coral)} manta coral surveys')
    merged_data = reef_series()
    
    # 步骤2: 计算最优滞后年数并构建分层状态空间模型
    lag_analysis = find_optimal_lag(merged_data)
    residual_data, model_results = fit_coral_cots_model(merged_data, lag_years=PRIMARY_LAG_YEARS)
    
    # 步骤3: DLS周期分析
    period_results = screen_residual_periods(residual_data, n_bootstrap=5000)
    
    # 步骤4: 时空模式分析
    pattern_summary = analyze_spatiotemporal_patterns(period_results, model_results, residual_data)
    pattern_summary['lag_analysis'] = lag_analysis
    
    # 可视化与保存结果
    save_results(residual_data, model_results, period_results, pattern_summary)
    
    print("\n" + "=" * 70)
    print(" BSRA分析完成! ")
    print(f" 结果保存至: {RESULTS_DIR}")
    print("=" * 70)
    
    return residual_data, model_results, period_results, pattern_summary


if __name__ == '__main__':
    main()
