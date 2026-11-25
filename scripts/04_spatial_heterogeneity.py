#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S3: Spatial Heterogeneity in Baseline Fit and Residual Periodicity

This is the formalized extraction of the old archived study-area script
function `update_figure5_with_basemap()`, which was the correct historical source
for manuscript Figure S3.

Panel (a): spatial distribution of ENHM baseline explanatory power (R^2)
Panel (b): spatial distribution of significant dominant residual periods
"""

from pathlib import Path
from analysis_inputs import reef_coordinates
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.switch_backend("Agg")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 300
plt.rcParams["svg.fonttype"] = "none"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)





def load_results():
    model_df = pd.read_csv(TABLES_DIR / "model_results.csv")
    period_df = pd.read_csv(TABLES_DIR / "period_results.csv")
    coords_df = reef_coordinates()

    merged = model_df.merge(
        period_df[
            ["reef_name", "primary_period", "primary_significant", "primary_p_value", "primary_power"]
        ],
        on="reef_name",
        how="left",
    ).merge(coords_df, on="reef_name", how="inner")

    return merged


def add_basemap(ax):
    ax.set_extent([142, 154, -25, -10], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="#404040", linewidth=0.6)
    ax.add_feature(cfeature.OCEAN, facecolor="#e0f2f1", alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#263238")
    ax.set_xticks(range(142, 156, 4), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-25, -9, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", which="major", direction="out", length=5, width=1, labelsize=11)


def main():
    from publication_style import save_publication
    from matplotlib.colors import TwoSlopeNorm
    print("Generating Figure S3: Spatial Heterogeneity...")
    df = load_results()
    print(f"  Loaded {len(df)} reefs with coordinates and BSRA outputs.")

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    add_basemap(ax1)
    sc1 = ax1.scatter(
        df["lon"],
        df["lat"],
        c=df["r_squared"],
        cmap="RdBu_r",
        s=90,
        edgecolors="white",
        linewidths=0.8,
        norm=TwoSlopeNorm(vmin=float(df['r_squared'].min()), vcenter=0,
                          vmax=float(df['r_squared'].max())),
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    cbar1 = plt.colorbar(sc1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.82)
    cbar1.set_label(r"Centered $R^2$ (coral-linked baseline)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Coral-linked baseline fit", fontsize=11, fontweight="bold", pad=6)
    ax1.text(-0.08, 1.07, "A", transform=ax1.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    add_basemap(ax2)
    ax2.yaxis.set_ticklabels([])

    tested = df.dropna(subset=["primary_period"])
    omitted = df.loc[df["primary_period"].isna()]
    ax2.scatter(omitted["lon"], omitted["lat"], c="lightgray", s=38,
                transform=ccrs.PlateCarree(), zorder=4, label="No spectral estimate")
    selected = tested["primary_significant"].eq(True)
    sc2 = ax2.scatter(
        tested["lon"], tested["lat"], c=tested["primary_period"], cmap="plasma", s=70,
        edgecolors=np.where(selected, "black", "white"), linewidths=np.where(selected, 1.8, 0.5),
        vmin=3, vmax=30, transform=ccrs.PlateCarree(), zorder=5,
    )
    cbar2 = plt.colorbar(sc2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar2.set_label("Dominant period (years)", fontsize=10.5, fontweight="bold")
    ax2.scatter([], [], c="none", edgecolors="black", linewidths=1.5, s=70, label="Significant ($p$ < 0.05)")
    ax2.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white", framealpha=0.9)
    ax2.set_title("Residual periodicity", fontsize=11, fontweight="bold", pad=6)
    ax2.text(-0.08, 1.07, "B", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    fig.suptitle("Spatial patterns of baseline fit and residual periodicity", fontsize=13, fontweight="bold", y=0.77)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    png = OUTPUT_DIR / "FigureS3_Spatial_Heterogeneity.png"
    pdf = OUTPUT_DIR / "FigureS3_Spatial_Heterogeneity.pdf"
    svg = OUTPUT_DIR / "FigureS3_Spatial_Heterogeneity.svg"
    save_publication(fig, png)
    plt.close(fig)

    print(f"  Saved: {png.name}")
    print(f"  Significant reefs shown: {int(selected.sum())}")


if __name__ == "__main__":
    main()
