#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent robustness comparison for spectral significance tests.

This script does not modify the formal pipeline or overwrite formal outputs.
It compares several null models on the same residual series:
1. White-noise permutation
2. Block permutation
3. AR(1) surrogate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

from config import TABLES_DIR


RESIDUALS_PATH = TABLES_DIR / "all_residuals.csv"
FORMAL_PERIODS_PATH = TABLES_DIR / "period_results.csv"
OUTPUT_CSV = TABLES_DIR / "robustness_null_comparison.csv"
OUTPUT_TXT = TABLES_DIR / "robustness_null_comparison.txt"

MIN_PERIOD = 3.0
MAX_PERIOD = 30.0
N_FREQUENCIES = 1000
N_BOOTSTRAP = 500
SIGNIFICANCE_LEVEL = 0.05
BLOCK_SIZE = 3
RANDOM_SEED = 42


@dataclass
class SpectralResult:
    reef_name: str
    n_obs: int
    best_period: float
    best_power: float
    perm_p: float
    block_p: float
    ar1_p: float
    perm_significant: bool
    block_significant: bool
    ar1_significant: bool
    perm_ipo: bool
    block_ipo: bool
    ar1_ipo: bool


def load_residual_series() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(RESIDUALS_PATH)
    grouped = {}
    for reef_name, reef_df in df.groupby("reef_name"):
        reef_df = reef_df.sort_values("year").reset_index(drop=True)
        if len(reef_df) >= 5 and reef_df["residual"].std() > 1e-10:
            grouped[reef_name] = reef_df
    return grouped


def block_permute(y: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if len(y) <= block_size:
        return rng.permutation(y)
    blocks = [y[i:i + block_size] for i in range(0, len(y), block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def fit_ar1_phi(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    x = y[:-1]
    z = y[1:]
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0
    phi = float(np.dot(x, z) / denom)
    return float(np.clip(phi, -0.95, 0.95))


def ar1_surrogate(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(y) < 3:
        return rng.permutation(y)
    centered = y - np.mean(y)
    phi = fit_ar1_phi(centered)
    eps = centered[1:] - phi * centered[:-1]
    sigma = float(np.std(eps, ddof=1)) if len(eps) > 1 else float(np.std(centered))
    sigma = max(sigma, 1e-8)
    sim = np.zeros_like(centered, dtype=float)
    stationary_std = sigma / np.sqrt(max(1e-6, 1.0 - phi ** 2))
    sim[0] = rng.normal(0.0, stationary_std)
    for i in range(1, len(sim)):
        sim[i] = phi * sim[i - 1] + rng.normal(0.0, sigma)
    sim = sim * (np.std(centered) / max(np.std(sim), 1e-8))
    return sim + np.mean(y)


def best_period_and_power(t: np.ndarray, y: np.ndarray, frequency: np.ndarray) -> tuple[float, float]:
    ls = LombScargle(t, y)
    power = ls.power(frequency)
    peak_idx = int(np.argmax(power))
    return float(1.0 / frequency[peak_idx]), float(power[peak_idx])


def empirical_p_value(
    t: np.ndarray,
    observed_power: float,
    frequency: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    mode: str,
) -> float:
    maxima = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        if mode == "perm":
            surrogate = rng.permutation(y)
        elif mode == "block":
            surrogate = block_permute(y, BLOCK_SIZE, rng)
        elif mode == "ar1":
            surrogate = ar1_surrogate(y, rng)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        ls = LombScargle(t, surrogate)
        maxima[i] = float(np.max(ls.power(frequency)))
    exceed = int(np.sum(maxima >= observed_power))
    return (exceed + 1.0) / (N_BOOTSTRAP + 1.0)


def run() -> pd.DataFrame:
    residual_series = load_residual_series()
    frequency = np.linspace(1.0 / MAX_PERIOD, 1.0 / MIN_PERIOD, N_FREQUENCIES)
    rng = np.random.default_rng(RANDOM_SEED)
    results: list[SpectralResult] = []

    for reef_name, reef_df in sorted(residual_series.items()):
        t = reef_df["year"].to_numpy(dtype=float)
        y = reef_df["residual"].to_numpy(dtype=float)
        best_period, best_power = best_period_and_power(t, y, frequency)

        perm_p = empirical_p_value(t, best_power, frequency, y, rng, "perm")
        block_p = empirical_p_value(t, best_power, frequency, y, rng, "block")
        ar1_p = empirical_p_value(t, best_power, frequency, y, rng, "ar1")

        perm_sig = perm_p < SIGNIFICANCE_LEVEL
        block_sig = block_p < SIGNIFICANCE_LEVEL
        ar1_sig = ar1_p < SIGNIFICANCE_LEVEL
        in_ipo = 15.0 <= best_period <= 30.0

        results.append(
            SpectralResult(
                reef_name=reef_name,
                n_obs=len(reef_df),
                best_period=best_period,
                best_power=best_power,
                perm_p=perm_p,
                block_p=block_p,
                ar1_p=ar1_p,
                perm_significant=perm_sig,
                block_significant=block_sig,
                ar1_significant=ar1_sig,
                perm_ipo=perm_sig and in_ipo,
                block_ipo=block_sig and in_ipo,
                ar1_ipo=ar1_sig and in_ipo,
            )
        )

    df = pd.DataFrame([vars(item) for item in results])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    return df


def summarize(df: pd.DataFrame) -> str:
    formal_df = pd.read_csv(FORMAL_PERIODS_PATH)
    formal_sig = int(formal_df["primary_significant"].sum())
    formal_ipo = int(((formal_df["primary_significant"] == True) & formal_df["primary_period"].between(15, 30)).sum())

    lines = [
        "Robustness comparison for spectral significance",
        "=" * 60,
        f"Input residual series: {len(df)} reefs",
        f"Frequency search window: {MIN_PERIOD:.1f}-{MAX_PERIOD:.1f} years",
        f"Bootstrap draws per null model: {N_BOOTSTRAP}",
        f"Block size: {BLOCK_SIZE}",
        "",
        "Formal pipeline reference",
        f"- Significant reefs: {formal_sig}",
        f"- IPO-band reefs: {formal_ipo}",
        "",
        "Independent comparison",
        f"- White-noise permutation significant reefs: {int(df['perm_significant'].sum())}",
        f"- White-noise permutation IPO-band reefs: {int(df['perm_ipo'].sum())}",
        f"- Block permutation significant reefs: {int(df['block_significant'].sum())}",
        f"- Block permutation IPO-band reefs: {int(df['block_ipo'].sum())}",
        f"- AR(1) surrogate significant reefs: {int(df['ar1_significant'].sum())}",
        f"- AR(1) surrogate IPO-band reefs: {int(df['ar1_ipo'].sum())}",
        "",
        "Mean empirical p-values",
        f"- White-noise permutation: {df['perm_p'].mean():.4f}",
        f"- Block permutation: {df['block_p'].mean():.4f}",
        f"- AR(1) surrogate: {df['ar1_p'].mean():.4f}",
        "",
        "Most sensitive reefs (largest p-value inflation under AR(1))",
    ]

    temp = df.copy()
    temp["delta_ar1_vs_perm"] = temp["ar1_p"] - temp["perm_p"]
    top = temp.sort_values("delta_ar1_vs_perm", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"- {row['reef_name']}: period={row['best_period']:.2f}y, "
            f"perm_p={row['perm_p']:.4f}, ar1_p={row['ar1_p']:.4f}, delta={row['delta_ar1_vs_perm']:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    df = run()
    summary = summarize(df)
    OUTPUT_TXT.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
