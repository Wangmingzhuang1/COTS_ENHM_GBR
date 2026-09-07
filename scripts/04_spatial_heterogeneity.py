#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S3: Spatial Heterogeneity of COTS Drivers

This is the formalized extraction of the old archived study-area script
function `update_figure5_with_basemap()`, which was the correct historical source
for manuscript Figure S3.

Panel (a): spatial distribution of ENHM model explanatory power (R^2)
Panel (b): spatial distribution of significant dominant environmental periods
"""

from pathlib import Path
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
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


def load_coordinates():
    coord_file = DATA_DIR / "spatial" / "site_lon_lat.csv"
    df = pd.read_csv(coord_file, encoding="gbk")
    df.columns = ["site", "coords"] + list(df.columns[2:])
    df["site"] = df["site"].replace({"Kangaroo Reef": "Kangaroo Reef West"})

    rows = []
    for _, row in df.iterrows():
        lat, lon = parse_dms(row["coords"])
        if lat is None:
            continue
        rows.append({"reef_name": row["site"], "lat": lat, "lon": lon})

    return pd.DataFrame(rows)


def load_results():
    model_df = pd.read_csv(TABLES_DIR / "model_results.csv")
    period_df = pd.read_csv(TABLES_DIR / "period_results.csv")
    coords_df = load_coordinates()

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
        cmap="viridis",
        s=90,
        edgecolors="white",
        linewidths=0.8,
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    cbar1 = plt.colorbar(sc1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.82)
    cbar1.set_label(r"$R^2$ (Food Limitation Model)", fontsize=10)
    ax1.set_title("(a) Food Limitation Strength", fontsize=12, fontweight="bold")

    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    add_basemap(ax2)
    ax2.yaxis.set_ticklabels([])

    sig_df = df[df["primary_significant"] == True].copy()
    nonsig_df = df[df["primary_significant"] != True].copy()

    ax2.scatter(
        nonsig_df["lon"],
        nonsig_df["lat"],
        c="lightgray",
        s=40,
        edgecolors="gray",
        linewidths=0.3,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=4,
        label="Not Significant",
    )

    if not sig_df.empty:
        sc2 = ax2.scatter(
            sig_df["lon"],
            sig_df["lat"],
            c=sig_df["primary_period"],
            cmap="plasma",
            s=100,
            edgecolors="k",
            linewidths=0.5,
            vmin=0,
            vmax=30,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        cbar2 = plt.colorbar(sc2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8)
        cbar2.set_label("Dominant Period (Years)", fontsize=10)
        cbar2.set_ticks([5, 10, 15, 20, 25, 30])
        cbar2.set_ticklabels(["5y\n(ENSO)", "10y", "15y", "20y", "25y\n(IPO)", "30y"])

    ax2.set_title("(b) Environmental Periodicity", fontsize=12, fontweight="bold")
    fig.suptitle("Spatial Heterogeneity of COTS Drivers", fontsize=14, fontweight="bold", y=0.95)

    plt.tight_layout()
    png = OUTPUT_DIR / "FigureS3_Spatial_Heterogeneity.png"
    pdf = OUTPUT_DIR / "FigureS3_Spatial_Heterogeneity.pdf"
    svg = OUTPUT_DIR / "FigureS3_Spatial_Heterogeneity.svg"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {png.name}")
    print(f"  Significant reefs shown: {len(sig_df)}")


if __name__ == "__main__":
    main()
