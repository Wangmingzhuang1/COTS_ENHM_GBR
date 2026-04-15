#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Historical audit for the legacy ~242 km/yr propagation result.

This script is intentionally separate from the default Figure 5 workflow.
It preserves the old target-oriented audit logic from the archived legacy
implementation without letting that logic affect the formal default analysis pipeline.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy import optimize, signal, stats

from config import CURRENTS_FILE, COORD_FILE, REEF_SURVEY_DIR, RESULTS_DIR, TABLES_DIR

plt.switch_backend("Agg")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.dpi"] = 150
plt.rcParams["svg.fonttype"] = "none"

AUDIT_RESULTS_DIR = RESULTS_DIR / "audit_242"
AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    df = pd.read_csv(COORD_FILE, encoding="gbk")
    df.columns = ["site", "coords"] + list(df.columns[2:])
    df["site"] = df["site"].replace({"Kangaroo Reef": "Kangaroo Reef West"})
    rows = []
    for _, row in df.iterrows():
        lat, lon = parse_dms(row["coords"])
        if lat is None:
            continue
        rows.append({"reef_name": row["site"], "lat": lat, "lon": lon})
    return pd.DataFrame(rows).dropna()


def load_cots_data():
    all_data = []
    for path in sorted(REEF_SURVEY_DIR.glob("*_cots_modelled_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        reef_name = path.name.split("_Manta Tow_cots_modelled_")[0]
        year_col = "year" if "year" in df.columns else ("Year" if "Year" in df.columns else None)
        val_col = "cotsptow" if "cotsptow" in df.columns else ("cots" if "cots" in df.columns else None)
        if not year_col or not val_col:
            continue
        for _, row in df.iterrows():
            year = pd.to_numeric(row[year_col], errors="coerce")
            value = pd.to_numeric(row[val_col], errors="coerce")
            if pd.isna(year) or pd.isna(value):
                continue
            all_data.append(
                {"reef_name": reef_name, "year": int(year), "cots_density": float(value)}
            )
    return pd.DataFrame(all_data)


def load_period_results():
    return pd.read_csv(TABLES_DIR / "period_results.csv")


def calculate_wave_speed(target_reefs, cots_df, coords):
    target_reefs = target_reefs.merge(coords, on="reef_name", how="inner")
    phases = []
    latitudes_phase = []

    for _, row in target_reefs.iterrows():
        reef = row["reef_name"]
        period = row["primary_period"]
        lat = np.abs(row["lat"])

        ts = cots_df[cots_df["reef_name"] == reef].groupby("year")["cots_density"].mean()
        if len(ts) < 5:
            continue

        t = ts.index.values
        y = signal.detrend(np.log1p(ts.values))
        omega = 2 * np.pi / period

        def sine_model(t_values, amplitude, phase):
            return amplitude * np.sin(omega * t_values + phase)

        try:
            popt, _ = optimize.curve_fit(sine_model, t, y, p0=[np.std(y), 0], maxfev=2000)
            phases.append(np.angle(np.exp(1j * popt[1])))
            latitudes_phase.append(lat)
        except Exception:
            continue

    if len(phases) < 3:
        return np.nan, 0.0, None

    lats = np.array(latitudes_phase)
    phis = np.array(phases)
    sort_idx = np.argsort(lats)
    lats_sorted = lats[sort_idx]
    phis_sorted = np.unwrap(phis[sort_idx])

    res = stats.linregress(lats_sorted, phis_sorted)
    mean_period = target_reefs["primary_period"].mean()
    km_per_degree = 111.0
    if res.slope != 0:
        years_per_degree = abs(res.slope * mean_period / (2 * np.pi))
        speed = km_per_degree / years_per_degree if years_per_degree > 0 else np.nan
    else:
        speed = np.nan
    return speed, res.rvalue**2, (lats_sorted, phis_sorted, res)


def bootstrap_speed_ci(target_reefs, cots_df, coords, n_iter=1000):
    speeds = []
    for _ in range(n_iter):
        sample_reefs = target_reefs.sample(n=len(target_reefs), replace=True)
        speed, _, _ = calculate_wave_speed(sample_reefs, cots_df, coords)
        if not np.isnan(speed) and speed < 2000:
            speeds.append(speed)
    if not speeds:
        return np.nan, np.nan, np.nan
    speeds = np.array(speeds)
    return np.mean(speeds), np.percentile(speeds, 2.5), np.percentile(speeds, 97.5)


def run_parameter_audit(period_results, cots_df, coords):
    print("\n[Audit] Scanning period bounds...")
    min_bounds = [12, 14, 15, 16, 18]
    max_bounds = [25, 28, 30, 32, 35]

    results = []
    for min_p in min_bounds:
        for max_p in max_bounds:
            if min_p >= max_p:
                continue
            sub = period_results[
                (period_results["primary_significant"] == True)
                & (period_results["primary_period"].between(min_p, max_p))
            ]
            if len(sub) < 5:
                continue
            speed, r2, _ = calculate_wave_speed(sub, cots_df, coords)
            results.append(
                {"min_p": min_p, "max_p": max_p, "speed": speed, "r2": r2, "n": len(sub)}
            )

    audit_df = pd.DataFrame(results)
    audit_df.to_csv(AUDIT_RESULTS_DIR / "period_scan_results.csv", index=False)
    audit_df["diff_to_242"] = np.abs(audit_df["speed"] - 242)
    best = audit_df.sort_values(["diff_to_242", "r2"], ascending=[True, False]).iloc[0]

    print(f"Optimal Range found: {best['min_p']}-{best['max_p']} years")
    print(f"  Yields Speed: {best['speed']:.1f} km/yr (n={best['n']}, R2={best['r2']:.3f})")
    return best


def quantify_current_coupling():
    if not CURRENTS_FILE.exists():
        print("Warning: Currents NetCDF file not found.")
        return None

    ds = xr.open_dataset(CURRENTS_FILE)
    v_var = next((v for v in ["vo", "v", "northward_velocity"] if v in ds.variables), None)
    if not v_var:
        return None

    ds_var = ds[v_var].isel(depth=0) if "depth" in ds[v_var].dims else ds[v_var]
    ds_sub = ds_var.sel(latitude=slice(-20, -14), longitude=slice(144, 147)).mean(
        dim=["latitude", "longitude"]
    )
    v_annual = ds_sub.resample(time="YE").mean()

    mean_speed_ms = np.abs(v_annual).mean().item()
    mean_speed_kmy = mean_speed_ms * 3600 * 24 * 365 / 1000

    print("\n[Audit] Quantifying current coupling...")
    print(f"  Mean EAC flow speed: {mean_speed_ms:.4f} m/s (~{mean_speed_kmy:.0f} km/yr)")
    return {
        "mean_v_ms": mean_speed_ms,
        "mean_v_kmy": mean_speed_kmy,
        "ratio_to_phys": 242.0 / mean_speed_kmy if mean_speed_kmy > 0 else np.nan,
    }


def main():
    print("=" * 60)
    print("COTS PROPAGATION AUDIT: TARGET 242 KM/YR")
    print("=" * 60)

    coords = load_coordinates()
    cots_df = load_cots_data()
    period_results = load_period_results()

    best_config = run_parameter_audit(period_results, cots_df, coords)

    final_sub = period_results[
        (period_results["primary_significant"] == True)
        & (period_results["primary_period"].between(best_config["min_p"], best_config["max_p"]))
    ]

    speed, r2, regression_payload = calculate_wave_speed(final_sub, cots_df, coords)
    mean_bs, ci_low, ci_high = bootstrap_speed_ci(final_sub, cots_df, coords)

    print("\n" + "-" * 40)
    print("FINAL AUDITED RESULTS")
    print(f"Speed: {speed:.1f} km/yr")
    print(f"95% Bootstrap CI: [{ci_low:.1f}, {ci_high:.1f}] km/yr")
    print(f"R-squared: {r2:.3f}")
    print(f"N (reefs): {len(final_sub)}")
    print("-" * 40)

    coupling = quantify_current_coupling()
    if coupling:
        print(
            f"Current Mean: {coupling['mean_v_ms']:.4f} m/s "
            f"({coupling['mean_v_kmy']:.0f} km/yr)"
        )
        print(
            "Propagation to Physical Current Ratio: "
            f"{100 * coupling['ratio_to_phys']:.1f}%"
        )

    pd.DataFrame(
        [
            {
                "best_min_period": best_config["min_p"],
                "best_max_period": best_config["max_p"],
                "speed_km_per_year": speed,
                "bootstrap_mean_speed_km_per_year": mean_bs,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "r_squared": r2,
                "n_reefs": len(final_sub),
                "mean_eac_speed_ms": None if coupling is None else coupling["mean_v_ms"],
                "mean_eac_speed_kmy": None if coupling is None else coupling["mean_v_kmy"],
                "speed_to_mean_eac_ratio": None if coupling is None else coupling["ratio_to_phys"],
            }
        ]
    ).to_csv(AUDIT_RESULTS_DIR / "velocity_audit_summary.csv", index=False)

    if regression_payload is None:
        raise RuntimeError("Audit regression could not be computed.")

    lats, phis, model = regression_payload

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(lats, phis, color="steelblue", s=80, edgecolors="k", alpha=0.8, label="Data")
    fit_x = np.linspace(lats.min(), lats.max(), 100)
    ax1.plot(fit_x, model.intercept + model.slope * fit_x, "r-", lw=2, label="Linear Fit")
    ax1.set_xlabel("Latitude (掳S)", fontsize=12)
    ax1.set_ylabel("Phase (radians)", fontsize=12)
    ax1.set_title(
        f'Wave Propagation Audit (Period {best_config["min_p"]}-{best_config["max_p"]}y)\n'
        f"Speed = {speed:.1f} km/yr",
        fontweight="bold",
    )
    ax1.legend()
    ax1.text(
        0.05,
        0.95,
        f"Speed: {speed:.0f} km/yr\nCI: [{ci_low:.0f}, {ci_high:.0f}]",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax2 = fig.add_subplot(gs[1])
    coupling_text = "No current coupling available."
    if coupling is not None:
        coupling_text = (
            f"2. This speed is ~{100 * coupling['ratio_to_phys']:.1f}% of the mean "
            "EAC background flow velocity,\n"
            "   consistent with a current-scale propagation constraint."
        )
    ax2.text(
        0.5,
        0.5,
        "Audit Conclusion:\n\n"
        "1. A wave speed near 242 km/yr can be recovered only under a targeted\n"
        f"   period-screening rule of {best_config['min_p']}-{best_config['max_p']} years.\n"
        f"{coupling_text}\n"
        "3. This is a historical sensitivity result, not the default formal Figure 5 workflow.",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#f0f0f0"),
    )
    ax2.axis("off")

    plt.savefig(AUDIT_RESULTS_DIR / "velocity_audit_diagnostic.png", bbox_inches="tight", dpi=300)
    plt.savefig(AUDIT_RESULTS_DIR / "velocity_audit_diagnostic.svg", bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS] Audit outputs saved to: {AUDIT_RESULTS_DIR}")


if __name__ == "__main__":
    main()
