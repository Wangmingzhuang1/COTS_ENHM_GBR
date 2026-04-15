#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified current-context analysis for manuscript discussion.

This script is independent from the formal pipeline. It provides:
1. Whole-domain current statistics
2. Propagation-corridor current statistics
3. Neg IPO / Pos IPO grouped comparisons
4. Same-unit comparison against the apparent ecological propagation speed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import CURRENTS_FILE, IPO_FILE, RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "current_context"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "current_context_summary.csv"
OUTPUT_TXT = OUTPUT_DIR / "current_context_summary.txt"

PROPAGATION_SPEED_KM_PER_YEAR = 241.96531468283558
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Study domain used by the figure
DOMAIN_BOUNDS = {
    "lon_min": 142.0,
    "lon_max": 154.0,
    "lat_min": -25.0,
    "lat_max": -10.0,
}

# A simple southward propagation corridor approximation across the central GBR shelf
CORRIDOR_BOUNDS = {
    "lon_min": 144.0,
    "lon_max": 147.0,
    "lat_min": -20.0,
    "lat_max": -14.0,
}


def to_km_per_year(speed_ms: float) -> float:
    return float(speed_ms) * SECONDS_PER_YEAR / 1000.0


def infer_coordinate_names(da: xr.DataArray) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    return lat_name, lon_name


def load_ipo_phase_years() -> tuple[list[int], list[int]]:
    try:
        df = pd.read_csv(IPO_FILE, encoding="utf-8")
    except Exception:
        df = pd.read_csv(IPO_FILE, encoding="gbk")

    year_col = df.columns[0]
    monthly_cols = list(df.columns[1:13])
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col]).copy()
    df[year_col] = df[year_col].astype(int)

    for col in monthly_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ipo_annual"] = df[monthly_cols].mean(axis=1)
    df = df.dropna(subset=["ipo_annual"]).copy()

    mean_val = float(df["ipo_annual"].mean())
    std_val = float(df["ipo_annual"].std())
    neg_years = df[df["ipo_annual"] < (mean_val - 0.5 * std_val)][year_col].astype(int).tolist()
    pos_years = df[df["ipo_annual"] > (mean_val + 0.5 * std_val)][year_col].astype(int).tolist()
    return neg_years, pos_years


def load_surface_current() -> xr.DataArray:
    ds = xr.open_dataset(CURRENTS_FILE)
    v_var = next((v for v in ["vo", "v", "northward_velocity"] if v in ds.variables), None)
    if v_var is None:
        raise RuntimeError(f"No northward current variable found in {CURRENTS_FILE}")

    da = ds[v_var]
    if "depth" in da.dims:
        da = da.isel(depth=0)
    return da


def subset_box(da: xr.DataArray, bounds: dict[str, float]) -> xr.DataArray:
    lat_name, lon_name = infer_coordinate_names(da)
    return da.sel(
        {
            lat_name: slice(bounds["lat_min"], bounds["lat_max"]),
            lon_name: slice(bounds["lon_min"], bounds["lon_max"]),
        }
    )


def summarize_region(
    da: xr.DataArray,
    region_name: str,
    neg_years: list[int],
    pos_years: list[int],
) -> list[dict[str, object]]:
    region = subset_box(da, DOMAIN_BOUNDS if region_name == "domain" else CORRIDOR_BOUNDS)
    if "time" not in region.dims:
        raise RuntimeError("Current data must contain time dimension")

    space_dims = [d for d in region.dims if d != "time"]
    region_abs = np.abs(region)
    annual = region_abs.resample(time="YE").mean()
    annual_series = annual.mean(dim=space_dims).to_series()
    annual_series.index = annual_series.index.year

    valid_neg = annual_series[annual_series.index.isin(neg_years)]
    valid_pos = annual_series[annual_series.index.isin(pos_years)]
    overall = annual_series.dropna()

    rows = []
    for phase_name, series in [
        ("all_years", overall),
        ("neg_ipo", valid_neg),
        ("pos_ipo", valid_pos),
    ]:
        if len(series) == 0:
            mean_ms = np.nan
            std_ms = np.nan
            n_years = 0
        else:
            mean_ms = float(series.mean())
            std_ms = float(series.std())
            n_years = int(len(series))

        mean_kmy = to_km_per_year(mean_ms) if not np.isnan(mean_ms) else np.nan
        rows.append(
            {
                "region": region_name,
                "phase": phase_name,
                "n_years": n_years,
                "mean_speed_ms": mean_ms,
                "std_speed_ms": std_ms,
                "mean_speed_km_per_year": mean_kmy,
                "propagation_speed_km_per_year": PROPAGATION_SPEED_KM_PER_YEAR,
                "propagation_to_current_ratio": (
                    PROPAGATION_SPEED_KM_PER_YEAR / mean_kmy if mean_kmy and not np.isnan(mean_kmy) else np.nan
                ),
                "days_for_current_to_cover_242km": (
                    PROPAGATION_SPEED_KM_PER_YEAR * 1000.0 / mean_ms / 86400.0
                    if mean_ms and not np.isnan(mean_ms)
                    else np.nan
                ),
            }
        )
    return rows


def build_summary() -> pd.DataFrame:
    neg_years, pos_years = load_ipo_phase_years()
    da = load_surface_current()

    rows = []
    rows.extend(summarize_region(da, "domain", neg_years, pos_years))
    rows.extend(summarize_region(da, "corridor", neg_years, pos_years))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    return df


def write_text_summary(df: pd.DataFrame) -> str:
    lines = [
        "Unified current-context summary",
        "=" * 60,
        f"Apparent ecological propagation speed: {PROPAGATION_SPEED_KM_PER_YEAR:.2f} km/yr",
        "",
        "Interpretation note:",
        "- The current metrics are physical flow speeds.",
        "- The 242 km/yr metric is an apparent ecological phase-front speed.",
        "- They are comparable as scale/context, not as one-to-one equivalent transport velocities.",
        "",
    ]

    for region in ["domain", "corridor"]:
        sub = df[df["region"] == region].copy()
        lines.append(f"{region.upper()} summary")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['phase']}: n={int(row['n_years'])}, "
                f"mean={row['mean_speed_ms']:.4f} m/s, "
                f"{row['mean_speed_km_per_year']:.1f} km/yr, "
                f"242/current={100 * row['propagation_to_current_ratio']:.2f}%, "
                f"242 km by current in {row['days_for_current_to_cover_242km']:.2f} days"
            )
        lines.append("")

    corridor_all = df[(df["region"] == "corridor") & (df["phase"] == "all_years")].iloc[0]
    lines.extend(
        [
            "Discussion guidance",
            "- The corridor value can be compared with 242 km/yr as a scale reference.",
            "- If corridor current speed is much larger than 242 km/yr, the result supports ecological-wave propagation rather than direct advection speed.",
            "- If Neg IPO and Pos IPO corridor speeds differ, they can be discussed as changing connectivity background, not as deterministic outbreak speed.",
            "",
            "Suggested usage:",
            "- Use corridor metrics in Discussion.",
            "- Use domain metrics as background context.",
            "- Avoid stating that the 242 km/yr wave speed equals EAC transport speed.",
            f"- Current best single context number: corridor all-years = {corridor_all['mean_speed_ms']:.4f} m/s ({corridor_all['mean_speed_km_per_year']:.1f} km/yr).",
        ]
    )
    text = "\n".join(lines)
    OUTPUT_TXT.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    df = build_summary()
    text = write_text_summary(df)
    print(text)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
