"""Comprehensive robustness checks:
1. White-noise, 3-yr calendar-block, and continuous-time AR(1) spectral null hierarchy.
2. Unwrap-free circular-linear spatial phase-latitude null tests (AR(1), 3-yr block, uniform).
3. Baseline sensitivity on the identical 58 screened reefs: Raw log(1+COTS) vs ENHM (RTO) vs OLS (intercept).
4. Dated lag sensitivity on the full 113-reef dataset: lag=2 yr vs lag=3 yr.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, stats

from analysis_inputs import align_survey_lag, ecological_observations, reef_coordinates, reef_series
from config import FIGURES_DIR, TABLES_DIR
from publication_style import apply_base_style, save_publication
from spectral_inference import (
    block_permutation_samples,
    peak_p_value,
    red_noise_samples,
    spectrum,
)


def _circular_linear_fit(series_list, abs_lats, mean_period, ks_grid=None):
    """Unwrap-free circular-linear fit maximizing resultant length R(k) = |mean(exp(i*(theta_peak - k*lat)))|."""
    if ks_grid is None:
        ks_grid = np.linspace(-2.0, 2.0, 801)
    omega = 2.0 * np.pi / mean_period
    thetas = []
    for t, y in series_list:
        X = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
        coef, _, _, _ = np.linalg.lstsq(X, y - np.mean(y), rcond=None)
        # y ~ A*cos(w*t - theta_peak) => theta_peak = atan2(sin_coef, cos_coef) = w * t_peak (mod 2pi)
        thetas.append(float(np.arctan2(coef[1], coef[0])))
    thetas = np.asarray(thetas)
    R_vals = np.abs(np.mean(np.exp(1j * (thetas[:, None] - abs_lats[:, None] * ks_grid[None, :])), axis=0))
    best_i = int(np.argmax(R_vals))
    return float(ks_grid[best_i]), float(R_vals[best_i])


def run_phase_slope_null(periods, residuals, coords, n_iter=500, seed=42):
    ipo = periods[
        periods["primary_significant"].eq(True) & periods["primary_period"].between(15.0, 30.0)
    ].merge(coords, on="reef_name", how="inner")
    if len(ipo) < 4:
        return pd.DataFrame()

    mean_period = float(ipo["primary_period"].mean())
    series_list = []
    abs_lats = []
    for _, row in ipo.iterrows():
        sub = residuals.loc[residuals["reef_name"].eq(row["reef_name"])].sort_values("time_years")
        t = sub["time_years"].to_numpy(dtype=float) + 1970.0
        y = sub["residual"].to_numpy(dtype=float)
        if len(t) >= 5 and np.std(y) > 1e-12:
            series_list.append((t, y))
            abs_lats.append(abs(float(row["lat"])))
    abs_lats = np.asarray(abs_lats)
    ks_fine = np.linspace(-2.0, 2.0, 4001)
    k_init, _ = _circular_linear_fit(series_list, abs_lats, mean_period, ks_fine)
    # Local refinement for observed k and R
    omega = 2.0 * np.pi / mean_period
    thetas_obs = []
    for t, y in series_list:
        X = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
        coef, _, _, _ = np.linalg.lstsq(X, y - np.mean(y), rcond=None)
        thetas_obs.append(float(np.arctan2(coef[1], coef[0])))
    thetas_obs = np.asarray(thetas_obs)
    res_opt = optimize.minimize_scalar(
        lambda k: -float(np.abs(np.mean(np.exp(1j * (thetas_obs - k * abs_lats))))),
        bounds=(k_init - 0.01, k_init + 0.01),
        method="bounded",
    )
    k_obs = float(res_opt.x)
    R_obs = float(-res_opt.fun)
    speed_obs = float(111.0 / (abs(k_obs) * mean_period / (2.0 * np.pi))) if abs(k_obs) > 0 else np.nan

    rng = np.random.default_rng(seed)
    ks_null = np.linspace(-2.0, 2.0, 401)
    rows = []
    for null_name in ("ar1", "block", "uniform_circular"):
        if null_name == "uniform_circular":
            n_u = 2000
            null_R = np.empty(n_u, dtype=float)
            for m in range(n_u):
                th_r = rng.uniform(-np.pi, np.pi, size=len(abs_lats))
                null_R[m] = float(
                    np.max(np.abs(np.mean(np.exp(1j * (th_r[:, None] - abs_lats[:, None] * ks_null[None, :])), axis=0)))
                )
            iters_used = n_u
        else:
            surr_by_reef = []
            for t, y in series_list:
                if null_name == "ar1":
                    draws, _ = red_noise_samples(t, y, rng, n_iter)
                else:
                    draws = block_permutation_samples(t, y, rng, n_iter, block_years=3.0)
                surr_by_reef.append(draws)
            null_R = np.empty(n_iter, dtype=float)
            for m in range(n_iter):
                step_series = [(series_list[i][0], surr_by_reef[i][m]) for i in range(len(series_list))]
                _, null_R[m] = _circular_linear_fit(step_series, abs_lats, mean_period, ks_null)
            iters_used = n_iter

        p_val = (1 + int(np.sum(null_R >= R_obs - 1e-12))) / (1 + iters_used)
        rows.append({
            "null_model": null_name,
            "n_reefs": len(series_list),
            "n_iter": iters_used,
            "k_obs_rad_per_deg": k_obs,
            "speed_km_per_year": speed_obs,
            "R_obs": R_obs,
            "null_mean_R": float(np.mean(null_R)),
            "null_q95_R": float(np.quantile(null_R, 0.95)),
            "p_value": float(p_val),
        })
    return pd.DataFrame(rows)


def run_baseline_sensitivity(residuals, n_boot=5000, seed=42):
    """Compare spectral IPO concentration across Raw log1p(COTS), ENHM (RTO), and OLS (intercept, beta>=0)."""
    rows = []
    for label in ("raw_log_cots", "enhm_rto", "ols_intercept_nonneg"):
        rng = np.random.default_rng(seed)
        tested, sig_n, ipo_n, enso_n = 0, 0, 0, 0
        for _, grp in sorted(residuals.groupby("reef_name")):
            grp = grp.sort_values("time_years")
            t = grp["time_years"].to_numpy(dtype=float)
            y_raw = grp["log_cots"].to_numpy(dtype=float)
            x = grp["coral_lagged"].to_numpy(dtype=float)
            if label == "raw_log_cots":
                vals = y_raw
            elif label == "enhm_rto":
                vals = grp["residual"].to_numpy(dtype=float)
            else:
                if np.std(x) > 1e-12:
                    sl, ic, _, _, _ = stats.linregress(x, y_raw)
                    if sl < 0:
                        vals = y_raw - np.mean(y_raw)
                    else:
                        vals = y_raw - (ic + sl * x)
                else:
                    vals = y_raw - np.mean(y_raw)
            sp = spectrum(t, vals)
            if sp is None:
                continue
            tested += 1
            freq, pwr = sp
            idx = int(np.nanargmax(pwr))
            perms = np.asarray([rng.permutation(vals) for _ in range(n_boot)])
            p = peak_p_value(t, freq, float(pwr[idx]), perms)
            per = 1.0 / freq[idx]
            if p < 0.05:
                sig_n += 1
                if 15.0 <= per <= 30.0:
                    ipo_n += 1
                elif 3.0 <= per <= 7.0:
                    enso_n += 1
        rows.append({
            "baseline_model": label,
            "tested_reefs": tested,
            "significant_reefs": sig_n,
            "ipo_band_reefs": ipo_n,
            "ipo_fraction": ipo_n / sig_n if sig_n else np.nan,
            "enso_band_reefs": enso_n,
        })
    return pd.DataFrame(rows)


def run_lag_sensitivity(coords, n_boot=5000, seed=42):
    """Compare dated lag=2 yr vs lag=3 yr across the full 113-reef dataset."""
    groups = reef_series()
    coords_idx = coords.set_index("reef_name")
    rows = []
    for lag_y in (2, 3):
        rng = np.random.default_rng(seed)
        fitted, tested, sig_n, ipo_n = 0, 0, 0, 0
        ipo_series, ipo_lats, ipo_T = [], [], []
        for reef, df in sorted(groups.items()):
            v = align_survey_lag(df, lag_y, 183)
            if len(v) < 3:
                continue
            fitted += 1
            x = v["coral_lagged"].to_numpy(dtype=float)
            y = v["log_cots"].to_numpy(dtype=float)
            beta = max(0.0, float(np.sum(x * y) / np.sum(x * x))) if np.sum(x * x) > 0 else 0.0
            res = y - beta * x
            t = v["time_years"].to_numpy(dtype=float)
            sp = spectrum(t, res)
            if sp is None:
                continue
            tested += 1
            freq, pwr = sp
            idx = int(np.nanargmax(pwr))
            perms = np.asarray([rng.permutation(res) for _ in range(n_boot)])
            p = peak_p_value(t, freq, float(pwr[idx]), perms)
            per = 1.0 / freq[idx]
            if p < 0.05:
                sig_n += 1
                if 15.0 <= per <= 30.0:
                    ipo_n += 1
                    ipo_series.append((t + 1970.0, res))
                    ipo_lats.append(abs(float(coords_idx.loc[reef, "lat"])))
                    ipo_T.append(per)
        if len(ipo_series) >= 4:
            mean_T = float(np.mean(ipo_T))
            k_best, R_best = _circular_linear_fit(ipo_series, np.asarray(ipo_lats), mean_T, np.linspace(-2.0, 2.0, 2001))
            speed = float(111.0 / (abs(k_best) * mean_T / (2.0 * np.pi))) if abs(k_best) > 0 else np.nan
            direction = "North-to-South" if k_best > 0 else "South-to-North"
        else:
            mean_T, k_best, R_best, speed, direction = np.nan, np.nan, np.nan, np.nan, "NA"
        rows.append({
            "lag_years": lag_y,
            "fitted_reefs": fitted,
            "screened_reefs": tested,
            "significant_reefs": sig_n,
            "ipo_band_reefs": ipo_n,
            "ipo_fraction": ipo_n / sig_n if sig_n else np.nan,
            "mean_ipo_period": mean_T,
            "k_rad_per_deg": k_best,
            "resultant_R": R_best,
            "direction": direction,
            "speed_km_per_year": speed,
        })
    return pd.DataFrame(rows)


def _extract_phases(series_list, period):
    omega = 2.0 * np.pi / period
    thetas = []
    for t, y in series_list:
        X = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
        coef, _, _, _ = np.linalg.lstsq(X, y - np.mean(y), rcond=None)
        thetas.append(float(np.arctan2(coef[1], coef[0])))
    return np.asarray(thetas)


def _scan_resultant(thetas, abs_lats, ks_grid, weights=None):
    z = np.exp(1j * (thetas[:, None] - abs_lats[:, None] * ks_grid[None, :]))
    if weights is None:
        return np.abs(np.mean(z, axis=0))
    w = weights / np.sum(weights)
    return np.abs(np.sum(w[:, None] * z, axis=0))


def _test_directional_nulls(series_list, abs_lats, period, R_obs, k_min=0.05, k_max=2.0,
                            weights=None, n_iter=500, seed=42):
    rng = np.random.default_rng(seed)
    ks_n = np.linspace(k_min, k_max, 401)
    w = None if weights is None else (weights / np.sum(weights))
    n_u = 2000
    exceed_u = 0
    for _ in range(n_u):
        th = rng.uniform(-np.pi, np.pi, size=len(abs_lats))
        z = np.exp(1j * (th[:, None] - abs_lats[:, None] * ks_n[None, :]))
        Rm = float(np.max(np.abs(np.mean(z, axis=0) if w is None else np.sum(w[:, None] * z, axis=0))))
        if Rm >= R_obs - 1e-12:
            exceed_u += 1
    out = {"p_uniform": (1 + exceed_u) / (1 + n_u)}
    for name in ("block", "ar1"):
        surr = []
        for t, y in series_list:
            if name == "ar1":
                d, _ = red_noise_samples(t, y, rng, n_iter)
            else:
                d = block_permutation_samples(t, y, rng, n_iter, block_years=3.0)
            surr.append(d)
        exceed = 0
        for m in range(n_iter):
            step = [(series_list[i][0], surr[i][m]) for i in range(len(series_list))]
            th = _extract_phases(step, period)
            Rv = _scan_resultant(th, abs_lats, ks_n, weights=weights)
            if float(np.max(Rv)) >= R_obs - 1e-12:
                exceed += 1
        out[f"p_{name}"] = (1 + exceed) / (1 + n_iter)
    return out


def run_phase_aliasing_and_unconditional_diagnostics(periods, residuals, coords, n_iter=500, seed=42):
    """Quantify all-58-reef phase organization conditional on the direction branch and 2-yr lag 2pi spatial aliasing,
    and generate SI Figure S5 (FigureS6_Phase_Sensitivity)."""
    coords_idx = coords.set_index("reef_name")
    ipo12 = periods[
        periods["primary_significant"].eq(True) & periods["primary_period"].between(15.0, 30.0)
    ].copy()
    mean_T12 = float(ipo12["primary_period"].mean())
    ipo12_names = set(ipo12["reef_name"])

    # 1. Primary 12 reefs
    s12, l12 = [], []
    for _, r in ipo12.iterrows():
        sub = residuals.loc[residuals["reef_name"].eq(r["reef_name"])].sort_values("time_years")
        s12.append((sub["time_years"].to_numpy(dtype=float) + 1970.0, sub["residual"].to_numpy(dtype=float)))
        l12.append(abs(float(coords_idx.loc[r["reef_name"], "lat"])))
    l12 = np.asarray(l12)
    th12 = _extract_phases(s12, mean_T12)

    # 2. All 58 analyzable reefs (all-reef & harmonic-power weighted)
    s58, l58, w58, is12_58 = [], [], [], []
    omega12 = 2.0 * np.pi / mean_T12
    for _, r in periods.iterrows():
        reef = r["reef_name"]
        if reef not in coords_idx.index:
            continue
        sub = residuals.loc[residuals["reef_name"].eq(reef)].sort_values("time_years")
        t = sub["time_years"].to_numpy(dtype=float) + 1970.0
        y = sub["residual"].to_numpy(dtype=float)
        if len(t) < 5 or np.std(y) <= 1e-12:
            continue
        X = np.column_stack([np.cos(omega12 * t), np.sin(omega12 * t)])
        yc = y - np.mean(y)
        coef, _, _, _ = np.linalg.lstsq(X, yc, rcond=None)
        r2_h = float(np.sum((X @ coef) ** 2) / np.sum(yc ** 2))
        s58.append((t, y))
        l58.append(abs(float(coords_idx.loc[reef, "lat"])))
        w58.append(r2_h)
        is12_58.append(reef in ipo12_names)
    l58 = np.asarray(l58)
    w58 = np.asarray(w58)
    is12_58 = np.asarray(is12_58)
    th58 = _extract_phases(s58, mean_T12)

    # 3. 2-year lag IPO-band reefs (17 reefs)
    groups = reef_series()
    rng2 = np.random.default_rng(seed)
    s2, l2, T2_list = [], [], []
    for reef, df in sorted(groups.items()):
        v = align_survey_lag(df, 2, 183)
        if len(v) < 3:
            continue
        x = v["coral_lagged"].to_numpy(dtype=float)
        y = v["log_cots"].to_numpy(dtype=float)
        beta = max(0.0, float(np.sum(x * y) / np.sum(x * x))) if np.sum(x * x) > 0 else 0.0
        res = y - beta * x
        t = v["time_years"].to_numpy(dtype=float)
        sp = spectrum(t, res)
        if sp is None:
            continue
        freq, pwr = sp
        idx = int(np.nanargmax(pwr))
        perms = np.asarray([rng2.permutation(res) for _ in range(5000)])
        p = peak_p_value(t, freq, float(pwr[idx]), perms)
        per = 1.0 / freq[idx]
        if p < 0.05 and 15.0 <= per <= 30.0:
            s2.append((t + 1970.0, res))
            l2.append(abs(float(coords_idx.loc[reef, "lat"])))
            T2_list.append(per)
    l2 = np.asarray(l2)
    mean_T2 = float(np.mean(T2_list))
    th2 = _extract_phases(s2, mean_T2)

    ks = np.linspace(-1.8, 1.8, 3601)
    pos_mask = ks >= 0.05
    neg_mask = ks <= -0.05

    datasets = [
        ("primary_12_lag3", "Primary 12 reefs (3-yr lag)", s12, l12, th12, mean_T12, None, "#c0392b", "-", 2.0),
        ("all_58_weighted_lag3", "All 58 reefs, power-weighted (3-yr lag)", s58, l58, th58, mean_T12, w58, "#1f77b4", "-", 1.9),
        ("all_58_unweighted_lag3", "All 58 reefs, unweighted (3-yr lag)", s58, l58, th58, mean_T12, None, "#4d6475", "--", 1.5),
        ("lag2_17_reefs", "17 IPO-band reefs (2-yr lag)", s2, l2, th2, mean_T2, None, "#2ca02c", "-", 1.8),
    ]

    diag_rows = []
    curves = {}
    fits_ns = {}
    for key, label, slist, lats, thetas, T_val, wt, color, ls, lw in datasets:
        R_curve = _scan_resultant(thetas, lats, ks, weights=wt)
        curves[key] = (R_curve, label, color, ls, lw)
        ip = np.where(pos_mask)[0][int(np.argmax(R_curve[pos_mask]))]
        in_ = np.where(neg_mask)[0][int(np.argmax(R_curve[neg_mask]))]
        kp, Rp = float(ks[ip]), float(R_curve[ip])
        kn, Rn = float(ks[in_]), float(R_curve[in_])
        spd_p = float(111.0 / (abs(kp) * T_val / (2.0 * np.pi)))
        nulls_p = _test_directional_nulls(slist, lats, T_val, Rp, k_min=0.05, k_max=2.0,
                                          weights=wt, n_iter=n_iter, seed=seed)
        fits_ns[key] = (kp, Rp, spd_p, nulls_p, T_val)
        diag_rows.append({
            "dataset": key,
            "n_reefs": len(lats),
            "mean_period_years": T_val,
            "k_north_to_south": kp,
            "R_north_to_south": Rp,
            "speed_km_per_year_ns": spd_p,
            "p_uniform_ns": nulls_p["p_uniform"],
            "p_block_ns": nulls_p["p_block"],
            "p_ar1_ns": nulls_p["p_ar1"],
            "k_alias_neg": kn,
            "R_alias_neg": Rn,
            "delta_k_alias": kp - kn,
        })
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(TABLES_DIR / "phase_sensitivity_diagnostics.csv", index=False)

    # --- Plot SI Figure S5 (saved as FigureS6_Phase_Sensitivity) ---
    apply_base_style()
    fig = plt.figure(figsize=(8.27, 11.69))
    ax_a = fig.add_axes([0.11, 0.565, 0.83, 0.355])
    ax_b = fig.add_axes([0.11, 0.110, 0.37, 0.345])
    ax_c = fig.add_axes([0.57, 0.110, 0.37, 0.345])

    # Panel A: R(k) spectra revealing the convergent N->S physical peak and 2pi approx. spatial-alias peak
    ax_a.axvspan(0.0, 1.8, color="#2ca02c", alpha=0.07)
    ax_a.axvline(0.0, color="#666666", lw=0.9, ls=":")
    ax_a.axvspan(0.95, 1.25, color="#c0392b", alpha=0.08)
    ax_a.axvspan(-0.80, -0.55, color="#7f7f7f", alpha=0.10)
    for key, _, _, _, _, _, _, _, _, _ in datasets:
        R_curve, label, color, ls, lw = curves[key]
        kp, Rp, spd_p, nulls_p, _ = fits_ns[key]
        ax_a.plot(ks, R_curve, color=color, ls=ls, lw=lw,
                  label=f"{label} (N→S: {spd_p:.1f} km yr$^{{-1}}$, AR(1) $p$ = {nulls_p['p_ar1']:.3f})")
        ax_a.scatter([kp], [Rp], color=color, s=36, zorder=6, edgecolors="white", linewidths=0.6)

    ax_a.text(1.10, 1.055, "N→S branch constraint\n(30.1–32.1 km yr$^{-1}$)",
              ha="center", va="top", fontsize=9, fontweight="bold", color="#902018")
    ax_a.text(-0.68, 1.055, "2$\\pi$ approx. spatial-alias peak\n($\\Delta k \\approx 2\\pi / 3.7^\\circ \\approx 1.70$)",
              ha="center", va="top", fontsize=9, fontweight="bold", color="#444444")
    ax_a.set_xlim(-1.8, 1.8)
    ax_a.set_ylim(0.0, 1.10)
    ax_a.set_xlabel("Latitudinal phase gradient $k$ (rad deg$^{-1}$; $k > 0$: North → South)")
    ax_a.set_ylabel("Mean resultant length $R(k)$")
    ax_a.set_title("Circular–linear resultant spectra across subsets and lags", loc="left", pad=6)
    ax_a.text(-0.08, 1.08, "A", transform=ax_a.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    ax_a.legend(loc="lower center", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=8.5)
    ax_a.spines[["top", "right"]].set_visible(False)

    def _plot_branch(ax, lats, thetas, kp, Rp, spd, p_ar1, p_blk, T_val, title_str, panel_lbl,
                     weights=None, highlight_mask=None, pt_color="#1f77b4"):
        w = np.ones_like(lats) if weights is None else weights
        z_m = np.sum(w * np.exp(1j * (thetas - kp * lats))) / np.sum(w)
        th0 = float(np.angle(z_m))
        fit_at = th0 + kp * lats
        resid = np.angle(np.exp(1j * (thetas - fit_at)))
        rad_to_yr = T_val / (2.0 * np.pi)
        y_off = th0 + kp * lats.min()
        y_obs = (fit_at + resid - y_off) * rad_to_yr
        x_line = np.linspace(lats.min(), lats.max(), 100)
        y_line = ((th0 + kp * x_line) - y_off) * rad_to_yr

        # Residual bootstrap CI
        rng_b = np.random.default_rng(42)
        boot = np.empty((1000, len(x_line)))
        y_hat = fit_at - y_off
        for bi in range(1000):
            rb = rng_b.choice(resid, size=len(lats), replace=True)
            yb = y_hat + rb
            sl_b, ic_b, _, _, _ = stats.linregress(lats, yb)
            boot[bi] = (ic_b + sl_b * x_line) * rad_to_yr
        ax.fill_between(x_line, np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0),
                        color="#a6cee3", alpha=0.45, label="95% bootstrap CI")
        ax.plot(x_line, y_line, color="#c0392b", lw=1.9, label="Circular–linear fit")

        if highlight_mask is not None:
            sizes = 20 + 75 * (w / max(np.max(w), 1e-6))
            ax.scatter(lats[~highlight_mask], y_obs[~highlight_mask], s=sizes[~highlight_mask],
                       color="#5b7c99", alpha=0.65, edgecolors="white", linewidths=0.5, zorder=4,
                       label="Other screened reefs (n = 46)")
            ax.scatter(lats[highlight_mask], y_obs[highlight_mask], s=sizes[highlight_mask] + 10,
                       color="#c0392b", alpha=0.95, edgecolors="black", linewidths=0.7, zorder=5,
                       label="Primary 12 IPO-band reefs")
        else:
            ax.scatter(lats, y_obs, s=44, color=pt_color, edgecolors="white", linewidths=0.6, zorder=5)

        ax.set_xlabel("Latitude (°S)")
        ax.set_ylabel("Relative peak timing (yr)")
        ax.set_title(title_str, loc="left", pad=6)
        ax.text(-0.14, 1.08, panel_lbl, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
        ax.text(
            0.04, 0.96,
            f"n = {len(lats)} reefs ($k$ = {kp:+.3f})\n"
            f"$R$ = {Rp:.3f} (AR(1) $p$ = {p_ar1:.3f}, Block $p$ = {p_blk:.3f})\n"
            f"Speed = {spd:.1f} km yr$^{{-1}}$ (mean $T$ = {T_val:.1f} yr)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
            bbox=dict(facecolor="white", alpha=0.90, edgecolor="#cccccc", linewidth=0.6),
        )
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, edgecolor="#cccccc", fontsize=8.0)
        ax.spines[["top", "right"]].set_visible(False)

    kp_w, Rp_w, spd_w, null_w, Tw = fits_ns["all_58_weighted_lag3"]
    _plot_branch(ax_b, l58, th58, kp_w, Rp_w, spd_w, null_w["p_ar1"], null_w["p_block"], Tw,
                 "All 58 screened reefs (no pre-selection)", "B",
                 weights=w58, highlight_mask=is12_58)

    kp_2, Rp_2, spd_2, null_2, T2 = fits_ns["lag2_17_reefs"]
    _plot_branch(ax_c, l2, th2, kp_2, Rp_2, spd_2, null_2["p_ar1"], null_2["p_block"], T2,
                 "2-yr lag IPO reefs (N→S branch)", "C",
                 pt_color="#2ca02c")

    save_publication(fig, FIGURES_DIR / "FigureS6_Phase_Sensitivity")
    plt.close(fig)
    return diag_df


def export_table_s1(periods, coords):
    """Export complete Table S1 for all 58 spectrally screened reefs."""
    models = pd.read_csv(TABLES_DIR / "model_results.csv")
    cots_df, _ = ecological_observations()
    reg_map = cots_df[["reef_name", "nrm_region"]].drop_duplicates()
    df = (
        periods.merge(models[["reef_name", "n_observations", "r_squared", "r_squared_uncentered"]], on="reef_name", how="left")
        .merge(coords[["reef_name", "lat", "lon"]], on="reef_name", how="left")
        .merge(reg_map, on="reef_name", how="left")
        .sort_values(["lat"], ascending=False)  # North (-10) to South (-25)
        .reset_index(drop=True)
    )
    out = df[[
        "reef_name", "nrm_region", "lat", "lon", "n_survey_dates", "span_years",
        "r_squared", "primary_period", "white_noise_p_value", "block_p_value", "ar1_p_value",
    ]].copy()
    out.to_csv(TABLES_DIR / "table_s1_screened_reefs.csv", index=False)
    return out


def main():
    periods = pd.read_csv(TABLES_DIR / "period_results.csv")
    residuals = pd.read_csv(TABLES_DIR / "all_residuals.csv")
    coords = reef_coordinates()

    long_band = periods["primary_period"].between(15, 30)
    rows = []
    for method, column in (
        ("white_noise_nominal", "white_noise_p_value"),
        ("calendar_block_3yr", "block_p_value"),
        ("continuous_time_ar1_nominal", "ar1_p_value"),
        ("continuous_time_ar1_holm", "ar1_p_adjusted"),
    ):
        selected = periods[column].lt(0.05)
        rows.append({
            "null_and_decision": method,
            "tested_reefs": len(periods),
            "detected_reefs": int(selected.sum()),
            "detections_15_30_years": int((selected & long_band).sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "robustness_null_comparison.csv", index=False)

    phase_null = run_phase_slope_null(periods, residuals, coords, n_iter=500, seed=42)
    phase_null.to_csv(TABLES_DIR / "phase_slope_null.csv", index=False)

    baseline_sens = run_baseline_sensitivity(residuals, n_boot=5000, seed=42)
    baseline_sens.to_csv(TABLES_DIR / "baseline_sensitivity_comparison.csv", index=False)

    lag_sens = run_lag_sensitivity(coords, n_boot=5000, seed=42)
    lag_sens.to_csv(TABLES_DIR / "lag_sensitivity_dated.csv", index=False)

    phase_diag = run_phase_aliasing_and_unconditional_diagnostics(periods, residuals, coords, n_iter=500, seed=42)
    table_s1 = export_table_s1(periods, coords)

    lines = [
        "1. Spectral null hierarchy:",
        summary.to_string(index=False),
        "",
        "2. Unwrap-free circular-linear spatial phase null:",
        phase_null.to_string(index=False) if not phase_null.empty else "Insufficient reefs",
        "",
        "3. Baseline model sensitivity (identical 58 reefs, N=5000):",
        baseline_sens.to_string(index=False),
        "",
        "4. Dated lag sensitivity (full 113-reef dataset, N=5000):",
        lag_sens.to_string(index=False),
        "",
        "5. All-58-reef (direction-constrained) & 2-yr lag spatial-alias diagnostics:",
        phase_diag.to_string(index=False),
        "",
        f"6. Table S1 exported: {len(table_s1)} spectrally screened reefs",
    ]
    text = "\n".join(lines) + "\n"
    (TABLES_DIR / "robustness_null_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
