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
from scipy import stats

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


def block_permute(
    t: np.ndarray,
    y: np.ndarray,
    block_size_years: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute contiguous calendar-year blocks while preserving within-block order."""
    if len(y) <= 1:
        return rng.permutation(y)
    block_ids = ((t - np.min(t)) // block_size_years).astype(int)
    blocks = [y[block_ids == block_id] for block_id in np.unique(block_ids)]
    if len(blocks) <= 1:
        return rng.permutation(y)
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
    n_draws: int = N_BOOTSTRAP,
) -> float:
    maxima = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        if mode == "perm":
            surrogate = rng.permutation(y)
        elif mode == "block":
            surrogate = block_permute(t, y, BLOCK_SIZE, rng)
        elif mode == "ar1":
            surrogate = ar1_surrogate(y, rng)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        ls = LombScargle(t, surrogate)
        maxima[i] = float(np.max(ls.power(frequency)))
    exceed = int(np.sum(maxima >= observed_power))
    return (exceed + 1.0) / (n_draws + 1.0)


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
        f"Calendar block width: {BLOCK_SIZE} years",
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


STABILITY_SEEDS = (42, 7, 2026)
STABILITY_DRAWS = 2000
PHASE_NULL_ITERS = 500


def white_noise_significant(seed: int, n_draws: int) -> dict:
    """Significant reefs and their dominant periods under one permutation seed."""
    residual_series = load_residual_series()
    frequency = np.linspace(1.0 / MAX_PERIOD, 1.0 / MIN_PERIOD, N_FREQUENCIES)
    rng = np.random.default_rng(seed)
    out = {}
    for reef_name, reef_df in sorted(residual_series.items()):
        t = reef_df["year"].to_numpy(dtype=float)
        y = reef_df["residual"].to_numpy(dtype=float)
        best_period, best_power = best_period_and_power(t, y, frequency)
        p = empirical_p_value(t, best_power, frequency, y, rng, "perm", n_draws=n_draws)
        if p < SIGNIFICANCE_LEVEL:
            out[reef_name] = best_period
    return out


def stability_check() -> pd.DataFrame:
    sets = {seed: white_noise_significant(seed, STABILITY_DRAWS) for seed in STABILITY_SEEDS}
    base = sets[STABILITY_SEEDS[0]]
    intersection = set.intersection(*[set(sig) for sig in sets.values()])
    rows = []
    for seed, sig in sets.items():
        n_ipo = sum(1 for period in sig.values() if 15.0 <= period <= 30.0)
        union = set(base) | set(sig)
        rows.append(
            {
                "seed": seed,
                "n_significant": len(sig),
                "n_ipo_band": n_ipo,
                "jaccard_vs_seed42": len(set(base) & set(sig)) / len(union) if union else np.nan,
                "n_intersection_all_seeds": len(intersection),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "white_noise_stability.csv", index=False)
    return out


def _load_latitudes() -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_mod", Path(__file__).with_name("12_consistency_audit.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    coord_df = pd.read_csv(mod.DATA_DIR / "spatial" / "site_lon_lat.csv", encoding="gbk")
    coord_df.columns = ["site", "coords"] + list(coord_df.columns[2:])
    coord_df["site"] = coord_df["site"].replace({"Kangaroo Reef": "Kangaroo Reef West"})
    out = {}
    for _, row in coord_df.iterrows():
        lat, _ = mod.parse_dms(row["coords"])
        if lat is not None:
            out[row["site"]] = float(lat)
    return out


def _sine_phase(t: np.ndarray, y: np.ndarray, period: float) -> float:
    from scipy import optimize

    omega = 2 * np.pi / period

    def sine(tv, amplitude, phase):
        return amplitude * np.sin(omega * tv + phase)

    popt, _ = optimize.curve_fit(sine, t, y, p0=[np.std(y), 0.0], maxfev=2000)
    return float(np.angle(np.exp(1j * popt[1])))


def _phase_slope_r2(obs) -> float:
    lats = np.array([lat for lat, _ in obs])
    phis = np.array([phase for _, phase in obs])
    idx = np.argsort(lats)
    res = stats.linregress(lats[idx], np.unwrap(phis[idx]))
    return float(res.rvalue ** 2)


def phase_slope_null() -> pd.DataFrame:
    """Test whether the phase-latitude organization exceeds reddened/short-memory nulls."""
    periods = pd.read_csv(FORMAL_PERIODS_PATH)
    targets = periods[
        (periods["primary_significant"] == True)
        & periods["primary_period"].between(15, 30)
    ]
    lats = _load_latitudes()
    residual_series = load_residual_series()
    reef_info = []
    obs = []
    for _, row in targets.iterrows():
        reef = row["reef_name"]
        if reef not in residual_series or reef not in lats:
            continue
        rdf = residual_series[reef]
        t = rdf["year"].to_numpy(dtype=float)
        y = rdf["residual"].to_numpy(dtype=float)
        if len(t) < 5 or np.std(y) <= 1e-12:
            continue
        try:
            phase = _sine_phase(t, y, float(row["primary_period"]))
        except Exception:
            continue
        reef_info.append((reef, abs(lats[reef]), float(row["primary_period"]), t, y))
        obs.append((abs(lats[reef]), phase))
    if len(obs) < 4:
        return pd.DataFrame()
    r2_obs = _phase_slope_r2(obs)
    rows = []
    for mode in ("ar1", "block"):
        rng = np.random.default_rng(RANDOM_SEED)
        nulls = []
        for _ in range(PHASE_NULL_ITERS):
            sims = []
            ok = True
            for reef, lat, period, t, y in reef_info:
                ys = ar1_surrogate(y, rng) if mode == "ar1" else block_permute(t, y, BLOCK_SIZE, rng)
                try:
                    sims.append((lat, _sine_phase(t, ys, period)))
                except Exception:
                    ok = False
                    break
            if ok:
                nulls.append(_phase_slope_r2(sims))
        if nulls:
            nulls = np.array(nulls)
            rows.append(
                {
                    "null_type": mode,
                    "n_iterations": int(len(nulls)),
                    "n_reefs": len(reef_info),
                    "r2_observed": r2_obs,
                    "null_r2_mean": float(nulls.mean()),
                    "null_r2_q95": float(np.quantile(nulls, 0.95)),
                    "p_value": float((np.sum(nulls >= r2_obs) + 1) / (len(nulls) + 1)),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(TABLES_DIR / "phase_slope_null.csv", index=False)
    return out


def main() -> None:
    df = run()
    stability = stability_check()
    phase_null = phase_slope_null()
    summary = summarize(df)
    extra = ["", f"White-noise significance stability ({STABILITY_DRAWS} draws per seed)"]
    for _, row in stability.iterrows():
        extra.append(
            f"- seed {int(row['seed'])}: significant={int(row['n_significant'])}, "
            f"IPO-band={int(row['n_ipo_band'])}, Jaccard vs seed 42={row['jaccard_vs_seed42']:.3f}"
        )
    extra.append(f"- Intersection across all seeds: {int(stability['n_intersection_all_seeds'].iloc[0])} reefs")
    extra.extend(["", "Phase-latitude slope null (surrogates preserve red/short-memory structure)"])
    for _, row in phase_null.iterrows():
        extra.append(
            f"- {row['null_type']}: R2_obs={row['r2_observed']:.3f}, null mean={row['null_r2_mean']:.3f}, "
            f"q95={row['null_r2_q95']:.3f}, p={row['p_value']:.4f} "
            f"({int(row['n_iterations'])} iters, {int(row['n_reefs'])} reefs)"
        )
    summary = summary + "\n" + "\n".join(extra)
    OUTPUT_TXT.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
