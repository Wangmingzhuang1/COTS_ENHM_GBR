#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a traceable manuscript-code consistency audit.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DATA_DIR = PROJECT_ROOT / "data"
MANUSCRIPT_PATH = PROJECT_ROOT.parent / "manuscript" / "PNAS" / "manuscript_en.md"
SI_PATH = PROJECT_ROOT.parent / "manuscript" / "PNAS" / "si_appendix_en.md"


def parse_dms(dms_str):
    if pd.isna(dms_str) or not dms_str:
        return None, None

    def parse_part(part_str):
        nums = re.findall(r"(\d+(?:\.\d+)?)", part_str)
        direction = re.search(r"([NSEW])", part_str)
        if not direction:
            return None
        direction = direction.group(1)
        vals = []
        if len(nums) == 3:
            vals = [float(n) for n in nums]
        elif len(nums) == 2:
            n1, n2 = nums[0], nums[1]
            if len(n1) == 5 and float(n1) > 10000:
                vals = [float(n1[:3]), float(n1[3:]), float(n2)]
            elif len(n2) == 4 and float(n2) > 60:
                vals = [float(n1), float(n2[:2]), float(n2[2:])]
            else:
                vals = [float(n1), float(n2), 0.0]
        elif len(nums) == 1:
            vals = [float(nums[0]), 0.0, 0.0]
        else:
            return None
        deg = vals[0] + vals[1] / 60 + vals[2] / 3600
        if direction in ["S", "W"]:
            deg = -deg
        return deg

    match = re.match(r"(.*[SN])\s*(.*[EW])", dms_str.strip().upper())
    if not match:
        return None, None
    return parse_part(match.group(1)), parse_part(match.group(2))


def formal_figure5_metrics(period_df):
    coord_df = pd.read_csv(DATA_DIR / "spatial" / "site_lon_lat.csv", encoding="gbk")
    coord_df.columns = ["site", "coords"] + list(coord_df.columns[2:])
    coord_df["site"] = coord_df["site"].replace({"Kangaroo Reef": "Kangaroo Reef West"})
    coord_rows = []
    for _, row in coord_df.iterrows():
        lat, lon = parse_dms(row["coords"])
        if lat is not None:
            coord_rows.append({"reef_name": row["site"], "lat": lat, "lon": lon})
    coords = pd.DataFrame(coord_rows)

    residual_df = pd.read_csv(TABLES_DIR / "all_residuals.csv")

    target_reefs = period_df[
        (period_df["primary_significant"] == True) & (period_df["primary_period"].between(15, 30))
    ].merge(coords, on="reef_name", how="inner")

    phases = []
    lats = []
    for _, row in target_reefs.iterrows():
        ts = residual_df[residual_df["reef_name"] == row["reef_name"]].sort_values("year")
        if len(ts) < 5:
            continue
        t = ts["year"].to_numpy(dtype=float)
        y = ts["residual"].to_numpy(dtype=float)
        if np.std(y) <= 1e-12:
            continue
        omega = 2 * np.pi / row["primary_period"]

        def sine_model(t_values, amplitude, phase):
            return amplitude * np.sin(omega * t_values + phase)

        try:
            popt, _ = optimize.curve_fit(sine_model, t, y, p0=[np.std(y), 0], maxfev=2000)
            phases.append(np.angle(np.exp(1j * popt[1])))
            lats.append(abs(row["lat"]))
        except Exception:
            continue

    if len(phases) < 3:
        return {"n": 0, "r2": np.nan, "p_value": np.nan, "speed_km_per_year": np.nan}

    lats = np.array(lats)
    phases = np.array(phases)
    idx = np.argsort(lats)
    regression = stats.linregress(lats[idx], np.unwrap(phases[idx]))
    mean_period = target_reefs["primary_period"].mean()
    years_per_degree = abs(regression.slope * mean_period / (2 * np.pi))
    speed = 111.0 / years_per_degree if years_per_degree > 0 else np.nan
    return {
        "n": int(len(phases)),
        "r2": float(regression.rvalue**2),
        "p_value": float(regression.pvalue),
        "speed_km_per_year": float(speed),
    }


def status(ok):
    return "一致" if ok else "缺失/冲突"


def main():
    manuscript_text = MANUSCRIPT_PATH.read_text(encoding="utf-8", errors="ignore")
    si_text = SI_PATH.read_text(encoding="utf-8", errors="ignore")
    model_df = pd.read_csv(TABLES_DIR / "model_results.csv")
    period_df = pd.read_csv(TABLES_DIR / "period_results.csv")
    lag_df = pd.read_csv(TABLES_DIR / "lag_analysis.csv")
    sig_df = pd.read_csv(TABLES_DIR / "significant_reefs.csv")
    summary_text = (TABLES_DIR / "analysis_summary.txt").read_text(encoding="utf-8", errors="ignore")

    ipo_sig = int(
        period_df[
            (period_df["primary_significant"] == True) & (period_df["primary_period"].between(15, 30))
        ].shape[0]
    )
    best_lag = int(lag_df.loc[lag_df["mean"].idxmax(), "lag_years"])
    fig5 = formal_figure5_metrics(period_df)

    # Values explicitly claimed by the current manuscript text. Update here when
    # the manuscript is revised; the audit then flags manuscript-result drift.
    MANUSCRIPT_CLAIMS = {
        "fitted_reefs": 83,
        "period_analyses": 57,
        "significant_reefs": 16,
        "ipo_band_reefs": 13,
        "trophic_lag_years": 3,
        "phase_front_speed_km_per_year": 33.75,
    }

    rows = [
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 1",
            "script": "04_cots_overview.py",
            "output_files": "Figure1_Combined_COTS_Overview.png;Figure1_Combined_COTS_Overview.svg",
            "evidence": "results/figures",
            "status": status(
                "Figure 1" in manuscript_text
                and
                (FIGURES_DIR / "Figure1_Combined_COTS_Overview.png").exists()
                and (FIGURES_DIR / "Figure1_Combined_COTS_Overview.svg").exists()
            ),
            "notes": "研究区与长期暴发概况（论文 Figure 1）",
        },
        {
            "item_type": "Supplementary",
            "manuscript_ref": "Figure S2",
            "script": "02_paper_figures.py",
            "output_files": "FigureS2_Trophic_Lag_Baseline.png;FigureS2_Trophic_Lag_Baseline.svg;FigureS2_Trophic_Lag_Baseline.pdf",
            "evidence": "results/figures",
            "status": status("Figure S2" in si_text and (FIGURES_DIR / "FigureS2_Trophic_Lag_Baseline.png").exists()),
            "notes": "滞后与食物限制基线（论文 Figure S2）",
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 2",
            "script": "02_paper_figures.py",
            "output_files": "Figure2_Residual_Periodicity.png;Figure2_Residual_Periodicity.svg;Figure2_Residual_Periodicity.pdf",
            "evidence": "results/figures",
            "status": status("Figure 2" in manuscript_text and (FIGURES_DIR / "Figure2_Residual_Periodicity.png").exists()),
            "notes": "残差周期分析（论文Figure 2）",
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure S3",
            "script": "04_spatial_heterogeneity.py",
            "output_files": "FigureS3_Spatial_Heterogeneity.png;FigureS3_Spatial_Heterogeneity.svg;FigureS3_Spatial_Heterogeneity.pdf",
            "evidence": "results/figures",
            "status": status("Figure S3" in si_text and (FIGURES_DIR / "FigureS3_Spatial_Heterogeneity.png").exists()),
            "notes": "空间异质性（论文Figure S3）",
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 3",
            "script": "05_propagation.py",
            "output_files": "Figure3_Propagation_Dynamics.png;Figure3_Propagation_Dynamics.svg",
            "evidence": "results/figures;results/tables/figure5_summary.csv",
            "status": status("Figure 3" in manuscript_text and (FIGURES_DIR / "Figure3_Propagation_Dynamics.png").exists()),
            "notes": f"正式流程速度={fig5['speed_km_per_year']:.2f} km/yr; n={fig5['n']}; R2={fig5['r2']:.3f}; p={fig5['p_value']:.4f}",
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 4",
            "script": "06_mechanism_fig6.py",
            "output_files": "Figure4_Hydroclimatic_Coupling.png;Figure4_Hydroclimatic_Coupling.svg",
            "evidence": "results/figures",
            "status": status("Figure 4" in manuscript_text and (FIGURES_DIR / "Figure4_Hydroclimatic_Coupling.png").exists()),
            "notes": "IPO-水文-生物级联机制",
        },
        {
            "item_type": "Supplementary",
            "manuscript_ref": "Figure S4",
            "script": "08_model_comparison.py",
            "output_files": "FigureS4_Model_Comparison.png;FigureS4_Model_Comparison.svg;FigureS4_Model_Comparison.pdf",
            "evidence": "results/figures;results/tables/model_validation_summary.csv",
            "status": status("Figure S4" in si_text and (FIGURES_DIR / "FigureS4_Model_Comparison.png").exists()),
            "notes": "ENHM vs GAM/XGBoost",
        },
        {
            "item_type": "Supplementary",
            "manuscript_ref": "Figure S5",
            "script": "02_paper_figures.py",
            "output_files": "FigureS5_Spectral_Decomposition.png;FigureS5_Spectral_Decomposition.svg;FigureS5_Spectral_Decomposition.pdf",
            "evidence": "results/figures",
            "status": status("Figure S5" in si_text and (FIGURES_DIR / "FigureS5_Spectral_Decomposition.png").exists()),
            "notes": "频谱分解补图",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": f"{MANUSCRIPT_CLAIMS['fitted_reefs']} fitted reefs",
            "script": "01_enhm_analysis.py",
            "output_files": "model_results.csv",
            "evidence": "results/tables/model_results.csv",
            "status": status(
                len(model_df) == MANUSCRIPT_CLAIMS["fitted_reefs"]
                and f"{MANUSCRIPT_CLAIMS['fitted_reefs']} fitted" in manuscript_text
            ),
            "notes": f"当前值={len(model_df)}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": f"{MANUSCRIPT_CLAIMS['period_analyses']} period analyses",
            "script": "01_enhm_analysis.py",
            "output_files": "period_results.csv",
            "evidence": "results/tables/period_results.csv",
            "status": status(
                len(period_df) == MANUSCRIPT_CLAIMS["period_analyses"]
                and f"{MANUSCRIPT_CLAIMS['period_analyses']} analyzable" in manuscript_text
            ),
            "notes": f"当前值={len(period_df)}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": f"{MANUSCRIPT_CLAIMS['significant_reefs']} significant reefs",
            "script": "01_enhm_analysis.py",
            "output_files": "significant_reefs.csv",
            "evidence": "results/tables/significant_reefs.csv",
            "status": status(
                len(sig_df) == MANUSCRIPT_CLAIMS["significant_reefs"]
                and f"{MANUSCRIPT_CLAIMS['significant_reefs']} reefs showed significant" in manuscript_text
            ),
            "notes": f"当前值={len(sig_df)}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": f"{MANUSCRIPT_CLAIMS['ipo_band_reefs']} IPO-band reefs",
            "script": "01_enhm_analysis.py",
            "output_files": "period_results.csv",
            "evidence": "results/tables/period_results.csv",
            "status": status(
                ipo_sig == MANUSCRIPT_CLAIMS["ipo_band_reefs"]
                and f"including {MANUSCRIPT_CLAIMS['ipo_band_reefs']} with IPO-band" in manuscript_text
            ),
            "notes": f"当前值={ipo_sig}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": f"{MANUSCRIPT_CLAIMS['trophic_lag_years']}-year lag",
            "script": "01_enhm_analysis.py",
            "output_files": "lag_analysis.csv;reef_lags.csv",
            "evidence": "results/tables/lag_analysis.csv",
            "status": status(
                best_lag == MANUSCRIPT_CLAIMS["trophic_lag_years"]
                and f"{MANUSCRIPT_CLAIMS['trophic_lag_years']}-year default" in manuscript_text
            ),
            "notes": f"当前值={best_lag}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": f"{MANUSCRIPT_CLAIMS['phase_front_speed_km_per_year']:.2f} km/yr",
            "script": "05_propagation.py",
            "output_files": "figure5_summary.csv",
            "evidence": "results/tables/figure5_summary.csv;results/audit_242/velocity_audit_summary.csv",
            "status": status(
                abs(fig5["speed_km_per_year"] - MANUSCRIPT_CLAIMS["phase_front_speed_km_per_year"]) < 0.5
                and f"{MANUSCRIPT_CLAIMS['phase_front_speed_km_per_year']:.2f} km/year" in manuscript_text
            ),
            "notes": f"正式Figure5={fig5['speed_km_per_year']:.2f} km/yr; audit仅作历史审计",
        },
    ]

    traceability_df = pd.DataFrame(rows)
    traceability_path = TABLES_DIR / "manuscript_traceability.csv"
    traceability_df.to_csv(traceability_path, index=False, encoding="utf-8-sig")

    consistent = int((traceability_df["status"] == "一致").sum())
    total = int(len(traceability_df))

    report_lines = [
        "论文-代码一致性核查报告",
        "=" * 60,
        f"论文文件: {MANUSCRIPT_PATH}",
        f"核查时间基于当前结果目录: {TABLES_DIR}",
        "",
        "高风险结论复核:",
        f"- 拟合礁区: {len(model_df)}",
        f"- 完成周期分析: {len(period_df)}",
        f"- 显著周期礁区: {len(sig_df)}",
        f"- IPO频段显著礁区: {ipo_sig}",
        f"- 最优滞后: {best_lag} 年",
        f"- 正式Figure 5传播速度: {fig5['speed_km_per_year']:.2f} km/yr",
        f"- 正式Figure 5回归: n={fig5['n']}, R2={fig5['r2']:.3f}, p={fig5['p_value']:.4f}",
        "",
        f"总项数: {total}",
        f"一致项: {consistent}",
        f"异常项: {total - consistent}",
        "",
        "逐项状态见 manuscript_traceability.csv",
    ]

    report_path = TABLES_DIR / "manuscript_consistency_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 60)
    print("论文-代码一致性核查完成")
    print(f"  保存: {traceability_path.name}")
    print(f"  保存: {report_path.name}")
    print(f"  一致项: {consistent}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
