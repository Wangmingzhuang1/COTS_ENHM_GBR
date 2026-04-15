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
MANUSCRIPT_PATH = PROJECT_ROOT.parent / "manuscript" / "manuscript.txt"


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

    cots_rows = []
    for path in sorted((DATA_DIR / "reef_surveys").glob("*_cots_modelled_*.csv")):
        df = pd.read_csv(path)
        reef_name = path.name.split("_Manta Tow_cots_modelled_")[0]
        year_col = "year" if "year" in df.columns else ("Year" if "Year" in df.columns else None)
        val_col = "cotsptow" if "cotsptow" in df.columns else ("cots" if "cots" in df.columns else None)
        if not year_col or not val_col:
            continue
        for _, row in df.iterrows():
            year = pd.to_numeric(row[year_col], errors="coerce")
            val = pd.to_numeric(row[val_col], errors="coerce")
            if pd.isna(year) or pd.isna(val):
                continue
            cots_rows.append({"reef_name": reef_name, "year": int(year), "cots_density": float(val)})
    cots_df = pd.DataFrame(cots_rows)

    target_reefs = period_df[
        (period_df["primary_significant"] == True) & (period_df["primary_period"].between(15, 30))
    ].merge(coords, on="reef_name", how="inner")

    phases = []
    lats = []
    for _, row in target_reefs.iterrows():
        ts = cots_df[cots_df["reef_name"] == row["reef_name"]].groupby("year")["cots_density"].mean()
        if len(ts) < 5:
            continue
        t = ts.index.values
        y = signal.detrend(np.log1p(ts.values))
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
    return "涓€鑷? if ok else "缂哄け/鍐茬獊"


def main():
    manuscript_text = MANUSCRIPT_PATH.read_text(encoding="utf-8", errors="ignore")
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

    rows = [
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 1",
            "script": "03_study_area_map.py",
            "output_files": "Figure1_Study_Area.png;Figure1_Study_Area.svg",
            "evidence": "results/figures",
            "status": status(
                "Figure 1" in manuscript_text
                and (FIGURES_DIR / "Figure1_Study_Area.png").exists()
                and (FIGURES_DIR / "Figure1_Study_Area.svg").exists()
            ),
            "notes": "鐮旂┒鍖轰富鍥?,
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 2",
            "script": "02_paper_figures.py",
            "output_files": "Figure2_Mechanism.png;Figure2_Mechanism.svg",
            "evidence": "results/figures",
            "status": status("Figure 2" in manuscript_text and (FIGURES_DIR / "Figure2_Mechanism.png").exists()),
            "notes": "婊炲悗涓庨鐗╅檺鍒舵満鍒?,
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 3",
            "script": "02_paper_figures.py",
            "output_files": "Figure3_Periodicity.png;Figure3_Periodicity.svg",
            "evidence": "results/figures",
            "status": status("Figure 3" in manuscript_text and (FIGURES_DIR / "Figure3_Periodicity.png").exists()),
            "notes": "娈嬪樊鍛ㄦ湡鍒嗘瀽",
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 4",
            "script": "04_spatial_heterogeneity.py",
            "output_files": "Figure4_Spatial_Heterogeneity.png;Figure4_Spatial_Heterogeneity.svg",
            "evidence": "results/figures",
            "status": status("Figure 4" in manuscript_text and (FIGURES_DIR / "Figure4_Spatial_Heterogeneity.png").exists()),
            "notes": "绌洪棿寮傝川鎬?,
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 5",
            "script": "05_propagation.py",
            "output_files": "Figure5_Propagation_Dynamics.png;Figure5_Propagation_Dynamics.svg",
            "evidence": "results/figures;results/tables/figure5_summary.csv",
            "status": status("Figure 5" in manuscript_text and (FIGURES_DIR / "Figure5_Propagation_Dynamics.png").exists()),
            "notes": f"姝ｅ紡娴佺▼閫熷害={fig5['speed_km_per_year']:.2f} km/yr; n={fig5['n']}; R2={fig5['r2']:.3f}; p={fig5['p_value']:.4f}",
        },
        {
            "item_type": "Figure",
            "manuscript_ref": "Figure 6",
            "script": "06_mechanism_fig6.py",
            "output_files": "Figure6_Ultimate_Mechanism.png;Figure6_Ultimate_Mechanism.svg",
            "evidence": "results/figures",
            "status": status("Figure 6" in manuscript_text and (FIGURES_DIR / "Figure6_Ultimate_Mechanism.png").exists()),
            "notes": "IPO-姘存枃-鐢熺墿绾ц仈鏈哄埗",
        },
        {
            "item_type": "Supplementary",
            "manuscript_ref": "Figure S1",
            "script": "08_model_comparison.py",
            "output_files": "SuppFig_Model_Comparison.png;SuppFig_Model_Comparison.svg;SuppFig_Model_Comparison.pdf",
            "evidence": "results/figures;results/tables/model_validation_summary.csv",
            "status": status("Figure S1" in manuscript_text and (FIGURES_DIR / "SuppFig_Model_Comparison.png").exists()),
            "notes": "ENHM vs GAM/XGBoost",
        },
        {
            "item_type": "Supplementary",
            "manuscript_ref": "Figure S2",
            "script": "02_paper_figures.py",
            "output_files": "SuppFig_Spectral_Decomposition.png;SuppFig_Spectral_Decomposition.svg",
            "evidence": "results/figures",
            "status": status("Figure S2" in manuscript_text and (FIGURES_DIR / "SuppFig_Spectral_Decomposition.png").exists()),
            "notes": "棰戣氨鍒嗚В琛ュ浘",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": "104 reefs",
            "script": "01_enhm_analysis.py",
            "output_files": "model_results.csv",
            "evidence": "results/tables/model_results.csv",
            "status": status(len(model_df) == 104 and "104" in manuscript_text and "104" in summary_text),
            "notes": f"褰撳墠鍊?{len(model_df)}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": "98 period analyses",
            "script": "01_enhm_analysis.py",
            "output_files": "period_results.csv",
            "evidence": "results/tables/period_results.csv",
            "status": status(len(period_df) == 98 and "98" in manuscript_text and "98" in summary_text),
            "notes": f"褰撳墠鍊?{len(period_df)}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": "24 significant reefs",
            "script": "01_enhm_analysis.py",
            "output_files": "significant_reefs.csv",
            "evidence": "results/tables/significant_reefs.csv",
            "status": status(len(sig_df) == 24 and "24" in manuscript_text and "24" in summary_text),
            "notes": f"褰撳墠鍊?{len(sig_df)}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": "18 IPO-band reefs",
            "script": "01_enhm_analysis.py",
            "output_files": "period_results.csv",
            "evidence": "results/tables/period_results.csv",
            "status": status(ipo_sig == 18 and "18" in manuscript_text and "18" in summary_text),
            "notes": f"褰撳墠鍊?{ipo_sig}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": "2-year lag",
            "script": "01_enhm_analysis.py",
            "output_files": "lag_analysis.csv;reef_lags.csv",
            "evidence": "results/tables/lag_analysis.csv",
            "status": status(best_lag == 2 and "2 骞? in manuscript_text),
            "notes": f"褰撳墠鍊?{best_lag}",
        },
        {
            "item_type": "Statistic",
            "manuscript_ref": "242 km/yr",
            "script": "05_propagation.py",
            "output_files": "figure5_summary.csv",
            "evidence": "results/tables/figure5_summary.csv;results/audit_242/velocity_audit_summary.csv",
            "status": status(abs(fig5["speed_km_per_year"] - 242.0) < 2.0 and "242 km/骞? in manuscript_text),
            "notes": f"姝ｅ紡Figure5={fig5['speed_km_per_year']:.2f} km/yr; audit浠呬綔鍘嗗彶瀹¤",
        },
    ]

    traceability_df = pd.DataFrame(rows)
    traceability_path = TABLES_DIR / "manuscript_traceability.csv"
    traceability_df.to_csv(traceability_path, index=False, encoding="utf-8-sig")

    consistent = int((traceability_df["status"] == "涓€鑷?).sum())
    total = int(len(traceability_df))

    report_lines = [
        "璁烘枃-浠ｇ爜涓€鑷存€ф牳鏌ユ姤鍛?,
        "=" * 60,
        f"璁烘枃鏂囦欢: {MANUSCRIPT_PATH}",
        f"鏍告煡鏃堕棿鍩轰簬褰撳墠缁撴灉鐩綍: {TABLES_DIR}",
        "",
        "楂橀闄╃粨璁哄鏍?",
        f"- 104涓鍖? {len(model_df)}",
        f"- 98涓畬鎴愬懆鏈熷垎鏋? {len(period_df)}",
        f"- 24涓樉钁楀懆鏈熺鍖? {len(sig_df)}",
        f"- 18涓狪PO棰戞鏄捐憲绀佸尯: {ipo_sig}",
        f"- 鏈€浼樻粸鍚? {best_lag} 骞?,
        f"- 姝ｅ紡Figure 5浼犳挱閫熷害: {fig5['speed_km_per_year']:.2f} km/yr",
        f"- 姝ｅ紡Figure 5鍥炲綊: n={fig5['n']}, R2={fig5['r2']:.3f}, p={fig5['p_value']:.4f}",
        "",
        f"鎬婚」鏁? {total}",
        f"涓€鑷撮」: {consistent}",
        f"寮傚父椤? {total - consistent}",
        "",
        "閫愰」鐘舵€佽 manuscript_traceability.csv",
    ]

    report_path = TABLES_DIR / "manuscript_consistency_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 60)
    print("璁烘枃-浠ｇ爜涓€鑷存€ф牳鏌ュ畬鎴?)
    print(f"  淇濆瓨: {traceability_path.name}")
    print(f"  淇濆瓨: {report_path.name}")
    print(f"  涓€鑷撮」: {consistent}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
