from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from scripts.analysis_inputs import monthly_ipo, trailing_month_mean
from pnas_aug.config import FIGURES_DIR, SUMMARY_DIR, TABLES_DIR
from pnas_aug.scripts._common import fit_linear_model, monthly_nino34, reef_residual_with_region, zscore

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["svg.fonttype"] = "none"


SPECS = {
    "IPO-only": ["ipo_z"],
    "ENSO-only": ["enso_z"],
    "Trend-only": ["trend_z"],
    "IPO+ENSO": ["ipo_z", "enso_z"],
    "IPO+Trend": ["ipo_z", "trend_z"],
    "ENSO+Trend": ["enso_z", "trend_z"],
    "IPO+ENSO+Trend": ["ipo_z", "enso_z", "trend_z"],
}


def main() -> None:
    data = reef_residual_with_region()
    dates = pd.to_datetime(data["survey_date"])
    data["ipo"] = trailing_month_mean(monthly_ipo(), dates)
    data["enso"] = trailing_month_mean(monthly_nino34(), dates)
    data["trend"] = data["time_years"]
    data = data.dropna(subset=["residual", "ipo", "enso", "trend"]).copy()
    data[["reef_name", "survey_date", "residual", "ipo", "enso", "trend"]].to_csv(
        TABLES_DIR / "reef_climate_model_input.csv", index=False
    )

    rows = []
    for reef_name, reef_df in data.groupby("reef_name"):
        reef_df = reef_df.sort_values("survey_date").copy()
        if len(reef_df) < 10 or reef_df[["residual", "ipo", "enso", "trend"]].std().le(1e-12).any():
            continue
        reef_df["target"] = zscore(reef_df["residual"])
        reef_df["ipo_z"] = zscore(reef_df["ipo"])
        reef_df["enso_z"] = zscore(reef_df["enso"])
        reef_df["trend_z"] = zscore(reef_df["trend"])

        fits = {label: fit_linear_model(reef_df["target"], reef_df[cols]) for label, cols in SPECS.items()}
        best_single = max(
            ["IPO-only", "ENSO-only", "Trend-only"],
            key=lambda label: fits[label]["adj_r2"],
        )

        rows.append(
            {
                "reef_name": reef_name,
                "region": reef_df["region"].iloc[0],
                "n_years": len(reef_df),
                "adj_r2_ipo_only": fits["IPO-only"]["adj_r2"],
                "adj_r2_enso_only": fits["ENSO-only"]["adj_r2"],
                "adj_r2_trend_only": fits["Trend-only"]["adj_r2"],
                "adj_r2_ipo_enso_trend": fits["IPO+ENSO+Trend"]["adj_r2"],
                "delta_ipo_over_enso_trend": fits["IPO+ENSO+Trend"]["adj_r2"] - fits["ENSO+Trend"]["adj_r2"],
                "delta_enso_over_ipo_trend": fits["IPO+ENSO+Trend"]["adj_r2"] - fits["IPO+Trend"]["adj_r2"],
                "delta_trend_over_ipo_enso": fits["IPO+ENSO+Trend"]["adj_r2"] - fits["IPO+ENSO"]["adj_r2"],
                "coef_ipo_full": fits["IPO+ENSO+Trend"]["coefficients"].get("ipo_z", 0.0),
                "coef_enso_full": fits["IPO+ENSO+Trend"]["coefficients"].get("enso_z", 0.0),
                "coef_trend_full": fits["IPO+ENSO+Trend"]["coefficients"].get("trend_z", 0.0),
                "best_single_driver": best_single,
            }
        )

    reef_df = pd.DataFrame(rows).sort_values("reef_name")
    reef_df.to_csv(TABLES_DIR / "reef_level_exclusivity.csv", index=False)

    agg = pd.DataFrame(
        [
            {
                "metric": "Median adjusted R2",
                "IPO-only": reef_df["adj_r2_ipo_only"].median(),
                "ENSO-only": reef_df["adj_r2_enso_only"].median(),
                "Trend-only": reef_df["adj_r2_trend_only"].median(),
                "IPO+ENSO+Trend": reef_df["adj_r2_ipo_enso_trend"].median(),
            },
            {
                "metric": "Mean adjusted R2",
                "IPO-only": reef_df["adj_r2_ipo_only"].mean(),
                "ENSO-only": reef_df["adj_r2_enso_only"].mean(),
                "Trend-only": reef_df["adj_r2_trend_only"].mean(),
                "IPO+ENSO+Trend": reef_df["adj_r2_ipo_enso_trend"].mean(),
            },
            {
                "metric": "Best single-driver reef count",
                "IPO-only": int((reef_df["best_single_driver"] == "IPO-only").sum()),
                "ENSO-only": int((reef_df["best_single_driver"] == "ENSO-only").sum()),
                "Trend-only": int((reef_df["best_single_driver"] == "Trend-only").sum()),
                "IPO+ENSO+Trend": len(reef_df),
            },
            {
                "metric": "Median delta in full model",
                "IPO-only": reef_df["delta_ipo_over_enso_trend"].median(),
                "ENSO-only": reef_df["delta_enso_over_ipo_trend"].median(),
                "Trend-only": reef_df["delta_trend_over_ipo_enso"].median(),
                "IPO+ENSO+Trend": 0.0,
            },
        ]
    )
    agg.to_csv(TABLES_DIR / "reef_level_exclusivity_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ["adj_r2_ipo_only", "adj_r2_enso_only", "adj_r2_trend_only", "adj_r2_ipo_enso_trend"]
    labels = ["IPO-only", "ENSO-only", "Trend-only", "Full"]
    axes[0].boxplot([reef_df[m].dropna() for m in metrics], tick_labels=labels, patch_artist=True)
    axes[0].set_title("(a) Reef-Level Model Comparison", loc="left", fontweight="bold")
    axes[0].set_ylabel("Adjusted R²")

    counts = reef_df["best_single_driver"].value_counts().reindex(["IPO-only", "ENSO-only", "Trend-only"]).fillna(0)
    axes[1].bar(counts.index, counts.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_title("(b) Best Single Driver by Reef", loc="left", fontweight="bold")
    axes[1].set_ylabel("Reef count")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reef_level_exclusivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "reef_level_exclusivity.svg", bbox_inches="tight")
    plt.close(fig)

    text = [
        "# PNAS Augmentation: Reef-Level Exclusivity",
        "",
        f"- 纳入 reef 数: {len(reef_df)}",
        f"- IPO-only median adjusted R2: {reef_df['adj_r2_ipo_only'].median():.3f}",
        f"- ENSO-only median adjusted R2: {reef_df['adj_r2_enso_only'].median():.3f}",
        f"- Trend-only median adjusted R2: {reef_df['adj_r2_trend_only'].median():.3f}",
        f"- Full model median adjusted R2: {reef_df['adj_r2_ipo_enso_trend'].median():.3f}",
        f"- IPO best-single-driver reef count: {int((reef_df['best_single_driver'] == 'IPO-only').sum())}",
        f"- ENSO best-single-driver reef count: {int((reef_df['best_single_driver'] == 'ENSO-only').sum())}",
        f"- Trend best-single-driver reef count: {int((reef_df['best_single_driver'] == 'Trend-only').sum())}",
        f"- IPO median delta over ENSO+Trend: {reef_df['delta_ipo_over_enso_trend'].median():.3f}",
        "",
        "判读规则:",
        "- 若 IPO-only 在 reef-level 仍普遍弱于 Trend-only 或 ENSO-only，说明“为什么是 IPO”仍未站稳。",
        "- 若 IPO 在 full model 中对 ENSO+Trend 的增量中位数接近 0 或为负，则不适合按强证据版冲 PNAS。",
        "- 该模块用于排除 aggregation artifact，因为它不再依赖 domain-mean 聚合。",
    ]
    (SUMMARY_DIR / "reef_level_exclusivity.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
