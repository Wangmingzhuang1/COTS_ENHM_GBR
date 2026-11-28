"""Comprehensive robustness checks:
1. White-noise, 3-yr calendar-block, and continuous-time AR(1) spectral null hierarchy.
2. Unwrap-free circular-linear spatial phase-latitude null tests (AR(1), 3-yr block, uniform).
3. Baseline sensitivity on the identical 58 screened reefs: Raw log(1+COTS) vs ENHM (RTO) vs OLS (intercept).
4. Dated lag sensitivity on the full 113-reef dataset: lag=2 yr vs lag=3 yr.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.signal import butter, filtfilt

from analysis_inputs import align_survey_lag, ecological_observations, ipo_phase_years, monthly_ipo, reef_coordinates, reef_series
from config import DATA_DIR, FIGURES_DIR, RESULTS_DIR, TABLES_DIR
from publication_style import apply_base_style, save_publication
from spectral_inference import (
    batch_lomb_scargle_power,
    block_permutation_samples,
    peak_p_value,
    red_noise_samples,
    reef_rng,
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
            rng_u = np.random.default_rng(seed)
            lats_u = np.sort(abs_lats)
            n_u = 2000
            null_R = np.empty(n_u, dtype=float)
            for m in range(n_u):
                th_r = rng_u.uniform(-np.pi, np.pi, size=len(lats_u))
                null_R[m] = float(
                    np.max(np.abs(np.mean(np.exp(1j * (th_r[:, None] - lats_u[:, None] * ks_null[None, :])), axis=0)))
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


def _baseline_series(grp, label):
    y_raw = grp["log_cots"].to_numpy(dtype=float)
    x = grp["coral_lagged"].to_numpy(dtype=float)
    if label == "raw_log_cots":
        return y_raw
    if label == "enhm_rto":
        return grp["residual"].to_numpy(dtype=float)
    if label == "ols_intercept_nonneg":
        if np.std(x) > 1e-12:
            sl, ic, _, _, _ = stats.linregress(x, y_raw)
            return y_raw - np.mean(y_raw) if sl < 0 else y_raw - (ic + sl * x)
        return y_raw - np.mean(y_raw)
    if label == "log_coral_ols":
        xl = np.log1p(np.maximum(0.0, x))
        if np.std(xl) > 1e-12:
            sl, ic, _, _, _ = stats.linregress(xl, y_raw)
            return y_raw - np.mean(y_raw) if sl < 0 else y_raw - (ic + sl * xl)
        return y_raw - np.mean(y_raw)
    raise ValueError(label)


def run_baseline_sensitivity(residuals, coords, mean_period, n_boot=10000, n_phase_iter=500, seed=42):
    """Compare spectral IPO concentration AND all-58-reef circular-linear phase progression
    across Raw log1p(COTS), ENHM (RTO), OLS (intercept, beta>=0), and Nonlinear log-coral OLS."""
    coords_idx = coords.set_index("reef_name")
    omega = 2.0 * np.pi / mean_period
    ks_ns = np.linspace(0.05, 2.0, 1951)
    rows = []
    for label in ("raw_log_cots", "enhm_rto", "ols_intercept_nonneg", "log_coral_ols"):
        tested, sig_n, ipo_n, enso_n = 0, 0, 0, 0
        s_list, l_list, w_list = [], [], []
        for reef_name, grp in sorted(residuals.groupby("reef_name")):
            grp = grp.sort_values("time_years")
            t = grp["time_years"].to_numpy(dtype=float)
            vals = _baseline_series(grp, label)
            sp = spectrum(t, vals)
            if sp is None:
                continue
            tested += 1
            freq, pwr = sp
            idx = int(np.nanargmax(pwr))
            rng_r = reef_rng(reef_name, "wn", seed)
            perms = np.asarray([rng_r.permutation(vals) for _ in range(n_boot)])
            p = peak_p_value(t, freq, float(pwr[idx]), perms)
            per = 1.0 / freq[idx]
            if p < 0.05:
                sig_n += 1
                if 15.0 <= per <= 30.0:
                    ipo_n += 1
                elif 3.0 <= per <= 7.0:
                    enso_n += 1
            if reef_name in coords_idx.index:
                t_yr = t + 1970.0
                X = np.column_stack([np.cos(omega * t_yr), np.sin(omega * t_yr)])
                yc = vals - np.mean(vals)
                coef, _, _, _ = np.linalg.lstsq(X, yc, rcond=None)
                r2_h = float(np.sum((X @ coef) ** 2) / np.sum(yc ** 2))
                s_list.append((t_yr, vals))
                l_list.append(abs(float(coords_idx.loc[reef_name, "lat"])))
                w_list.append(r2_h)

        l_arr = np.asarray(l_list)
        w_arr = np.asarray(w_list)
        th_arr = _extract_phases(s_list, mean_period)
        R_curve = _scan_resultant(th_arr, l_arr, ks_ns, weights=w_arr)
        bi = int(np.argmax(R_curve))
        kp, Rp = float(ks_ns[bi]), float(R_curve[bi])
        spd = float(111.0 / (abs(kp) * mean_period / (2.0 * np.pi)))
        nulls = _test_directional_nulls(s_list, l_arr, mean_period, Rp, k_min=0.05, k_max=2.0,
                                        weights=w_arr, n_iter=n_phase_iter, seed=seed)
        rows.append({
            "baseline_model": label,
            "tested_reefs": tested,
            "significant_reefs": sig_n,
            "ipo_band_reefs": ipo_n,
            "ipo_fraction": ipo_n / sig_n if sig_n else np.nan,
            "enso_band_reefs": enso_n,
            "all58_k_ns": kp,
            "all58_speed_km_yr": spd,
            "all58_R_weighted": Rp,
            "all58_p_ar1": nulls["p_ar1"],
            "all58_p_block": nulls["p_block"],
        })
    return pd.DataFrame(rows)


def run_lag_sensitivity(coords, n_boot=10000, seed=42):
    """Compare dated lag=2 yr vs lag=3 yr across the full 113-reef dataset."""
    groups = reef_series()
    coords_idx = coords.set_index("reef_name")
    rows = []
    for lag_y in (2, 3):
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
            rng_r = reef_rng(reef, "wn", seed)
            perms = np.asarray([rng_r.permutation(res) for _ in range(n_boot)])
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


def run_field_significance(periods, residuals, n_iter=1000, seed=42):
    """Uniform-frequency-grid field significance for 58-reef IPO-band fractional power
    and empirical white-noise null for IPO-band concentration among top-K peaks."""
    rng = np.random.default_rng(seed)
    n_reefs = len(periods)
    obs_fracs = np.zeros(n_reefs, dtype=float)
    wn_fracs = np.zeros((n_iter, n_reefs), dtype=float)
    blk_fracs = np.zeros((n_iter, n_reefs), dtype=float)
    ar1_fracs = np.zeros((n_iter, n_reefs), dtype=float)
    wn_peak_ipo = np.zeros((n_iter, n_reefs), dtype=bool)
    wn_peak_pwr = np.zeros((n_iter, n_reefs), dtype=float)

    obs_sig = periods["primary_significant"].eq(True)
    obs_k = int(obs_sig.sum())
    obs_ipo_in_sig = int((obs_sig & periods["primary_period"].between(15.0, 30.0)).sum())

    for i, rname in enumerate(periods["reef_name"]):
        sub = residuals.loc[residuals["reef_name"].eq(rname)].sort_values("time_years")
        t = sub["time_years"].to_numpy(dtype=float)
        y = sub["residual"].to_numpy(dtype=float)
        freq, pwr = spectrum(t, y)
        per = 1.0 / freq
        ipo_mask = (per >= 15.0) & (per <= 30.0)
        obs_fracs[i] = float(np.sum(pwr[ipo_mask]) / np.sum(pwr)) if ipo_mask.any() else 0.0

        wn_s = np.asarray([rng.permutation(y) for _ in range(n_iter)])
        p_wn = batch_lomb_scargle_power(t, freq, wn_s)
        wn_fracs[:, i] = np.sum(p_wn[:, ipo_mask], axis=1) / np.maximum(np.sum(p_wn, axis=1), 1e-12) if ipo_mask.any() else 0.0
        peak_idx = np.argmax(p_wn, axis=1)
        wn_peak_ipo[:, i] = (per[peak_idx] >= 15.0) & (per[peak_idx] <= 30.0)
        wn_peak_pwr[:, i] = p_wn[np.arange(n_iter), peak_idx]

        blk_s = block_permutation_samples(t, y, rng, n_iter, block_years=3.0)
        p_blk = batch_lomb_scargle_power(t, freq, blk_s)
        blk_fracs[:, i] = np.sum(p_blk[:, ipo_mask], axis=1) / np.maximum(np.sum(p_blk, axis=1), 1e-12) if ipo_mask.any() else 0.0

        ar1_s, _ = red_noise_samples(t, y, rng, n_iter)
        p_ar1 = batch_lomb_scargle_power(t, freq, ar1_s)
        ar1_fracs[:, i] = np.sum(p_ar1[:, ipo_mask], axis=1) / np.maximum(np.sum(p_ar1, axis=1), 1e-12) if ipo_mask.any() else 0.0

    obs_mean = float(obs_fracs.mean())
    rows = []
    for lbl, mat in (("white_noise_power_fraction", wn_fracs),
                     ("block_3yr_power_fraction", blk_fracs),
                     ("ar1_power_fraction", ar1_fracs)):
        sim_m = mat.mean(axis=1)
        p_val = (1 + int(np.sum(sim_m >= obs_mean - 1e-12))) / (1 + n_iter)
        rows.append({
            "metric": lbl,
            "n_iter": n_iter,
            "observed": obs_mean,
            "null_mean": float(sim_m.mean()),
            "null_q95": float(np.quantile(sim_m, 0.95)),
            "p_value": float(p_val),
        })

    top_k_ipo = np.array([wn_peak_ipo[s, np.argsort(wn_peak_pwr[s])[-obs_k:]].sum() for s in range(n_iter)])
    p_topk = (1 + int(np.sum(top_k_ipo >= obs_ipo_in_sig))) / (1 + n_iter)
    rows.append({
        "metric": f"wn_uniform_freq_top{obs_k}_ipo_count",
        "n_iter": n_iter,
        "observed": float(obs_ipo_in_sig),
        "null_mean": float(top_k_ipo.mean()),
        "null_q95": float(np.quantile(top_k_ipo, 0.95)),
        "p_value": float(p_topk),
    })
    rows.append({
        "metric": "wn_uniform_freq_all58_peak_ipo_rate",
        "n_iter": n_iter,
        "observed": float(obs_ipo_in_sig / obs_k) if obs_k else np.nan,
        "null_mean": float(wn_peak_ipo.mean()),
        "null_q95": float(np.quantile(wn_peak_ipo.mean(axis=1), 0.95)),
        "p_value": float(p_topk),
    })
    return pd.DataFrame(rows)


def _bretherton_corr(x, y, n_controls=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    r, _ = stats.pearsonr(x, y)
    df_raw = max(1.0, float(n - 2 - n_controls))
    t_raw = r * np.sqrt(df_raw / max(1e-12, 1.0 - r * r))
    p_raw = float(2.0 * stats.t.sf(abs(t_raw), df=df_raw))
    r1x = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r1y = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    neff = max(float(3 + n_controls), min(float(n), n * (1.0 - r1x * r1y) / (1.0 + r1x * r1y)))
    df_eff = max(1.0, neff - 2.0 - n_controls)
    t_stat = r * np.sqrt(df_eff / max(1e-12, 1.0 - r * r))
    p_eff = float(2.0 * stats.t.sf(abs(t_stat), df=df_eff))
    return float(r), p_raw, p_eff, float(neff), n


def run_environmental_sensitivity(periods, residuals):
    """Compute NDJF spawning-season Chl-a (raw & ex-Nino3.4 partial against contemporaneous NDJF TPI),
    13-yr low-pass TPI rivers (with circular phase-shift test), NDJF GLORYS meridional velocity,
    and 113/92/58 reef subset velocity consistency."""
    neg_yrs, pos_yrs, ipo_s = ipo_phase_years()
    ipo_s = ipo_s.sort_index().dropna()
    b_lp, a_lp = butter(3, 1.0 / 6.5, btype="low")
    ipo_lp = pd.Series(filtfilt(b_lp, a_lp, ipo_s.values), index=ipo_s.index)
    lp_mu, lp_sd = float(ipo_lp.mean()), float(ipo_lp.std())
    lp_neg = set(ipo_lp[ipo_lp < lp_mu - 0.5 * lp_sd].index)
    lp_pos = set(ipo_lp[ipo_lp > lp_mu + 0.5 * lp_sd].index)

    m_ipo = monthly_ipo()
    df_ipo_m = pd.DataFrame({"tpi": m_ipo, "year": m_ipo.index.year, "month": m_ipo.index.month})
    df_ipo_m["syr"] = np.where(df_ipo_m["month"] >= 11, df_ipo_m["year"] + 1, df_ipo_m["year"])
    tpi_ndjf = df_ipo_m[df_ipo_m["month"].isin([11, 12, 1, 2])].groupby("syr").filter(lambda g: len(g) == 4).groupby("syr")["tpi"].mean()

    nino_wide = pd.read_csv(DATA_DIR / "climate/enso_nino34_monthly.csv")
    months_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                  "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    nl = nino_wide.melt(id_vars=["year"], value_vars=list(months_map.keys()), var_name="m", value_name="val")
    nl["month"] = nl["m"].map(months_map)
    nl["val"] = pd.to_numeric(nl["val"], errors="coerce")
    nl = nl[nl["val"] > -90].dropna()
    clim = nl[nl["year"].between(1981, 2010)].groupby("month")["val"].mean()
    nl["anom"] = nl["val"] - nl["month"].map(clim)
    nino_ann = nl.groupby("year").filter(lambda g: len(g) == 12).groupby("year")["anom"].mean()
    nl["syr"] = np.where(nl["month"] >= 11, nl["year"] + 1, nl["year"])
    nino_ndjf = nl[nl["month"].isin([11, 12, 1, 2])].groupby("syr").filter(lambda g: len(g) == 4).groupby("syr")["anom"].mean()

    rows = []
    # 1. Annual & NDJF Chl-a vs contemporaneous TPI and ex-Nino3.4 partial (with df = Neff - 3)
    chl_ann_df = pd.read_csv(RESULTS_DIR / "source_data/chlorophyll_tpi_annual.csv", index_col=0)
    m_ann = pd.concat([chl_ann_df, nino_ann.rename("nino")], axis=1).dropna()
    r, pr, pe, neff, n = _bretherton_corr(m_ann["chlorophyll_mg_m3"], m_ann["unfiltered_tpi"], n_controls=0)
    rows.append({"category": "chl_annual_vs_tpi", "n": n, "value": r, "p_raw": pr, "p_eff_or_mw": pe, "detail": f"neff={neff:.1f};df=neff-2"})
    c_res = m_ann["chlorophyll_mg_m3"] - np.polyval(np.polyfit(m_ann["nino"], m_ann["chlorophyll_mg_m3"], 1), m_ann["nino"])
    t_res = m_ann["unfiltered_tpi"] - np.polyval(np.polyfit(m_ann["nino"], m_ann["unfiltered_tpi"], 1), m_ann["nino"])
    r, pr, pe, neff, n = _bretherton_corr(c_res, t_res, n_controls=1)
    rows.append({"category": "chl_annual_vs_tpi_partial_ex_nino34", "n": n, "value": r, "p_raw": pr, "p_eff_or_mw": pe, "detail": f"neff={neff:.1f};df=neff-3"})

    with xr.open_dataset(DATA_DIR / "climate/GBR_chlorophyll_monthly_1998_2025.nc") as ds_c:
        vname = [v for v in ds_c.data_vars if "chl" in v.lower()][0]
        da = ds_c[vname].sel(latitude=slice(-25, -10), longitude=slice(142, 154))
        lat_n = [d for d in da.dims if "lat" in d.lower()][0]
        lon_n = [d for d in da.dims if "lon" in d.lower()][0]
        w = np.cos(np.deg2rad(da[lat_n]))
        ts_m = da.weighted(w).mean(dim=[lat_n, lon_n]).to_series().dropna()
        ts_m.index = pd.to_datetime(ts_m.index)
        df_cm = pd.DataFrame({"chl": ts_m, "year": ts_m.index.year, "month": ts_m.index.month})
        df_cm["syr"] = np.where(df_cm["month"] >= 11, df_cm["year"] + 1, df_cm["year"])
        chl_ndjf = df_cm[df_cm["month"].isin([11, 12, 1, 2])].groupby("syr").filter(lambda g: len(g) == 4).groupby("syr")["chl"].mean()
    m_nd = pd.concat([chl_ndjf.rename("chl"), tpi_ndjf.rename("tpi_ndjf"), nino_ndjf.rename("nino")], axis=1).dropna()
    r_tn, _, _, _, _ = _bretherton_corr(m_nd["tpi_ndjf"], m_nd["nino"], n_controls=0)
    r_cn, pr_cn, pe_cn, ne_cn, n_cn = _bretherton_corr(m_nd["chl"], m_nd["nino"], n_controls=0)
    rows.append({"category": "chl_ndjf_vs_nino34", "n": n_cn, "value": r_cn, "p_raw": pr_cn, "p_eff_or_mw": pe_cn, "detail": f"neff={ne_cn:.1f};corr_tpi_nino34={r_tn:.3f}"})
    r, pr, pe, neff, n = _bretherton_corr(m_nd["chl"], m_nd["tpi_ndjf"], n_controls=0)
    rows.append({"category": "chl_ndjf_vs_tpi", "n": n, "value": r, "p_raw": pr, "p_eff_or_mw": pe, "detail": f"neff={neff:.1f};contemporaneous_NDJF_TPI"})
    yr_x = np.arange(len(m_nd), dtype=float)
    cd = m_nd["chl"] - np.polyval(np.polyfit(yr_x, m_nd["chl"], 1), yr_x)
    td = m_nd["tpi_ndjf"] - np.polyval(np.polyfit(yr_x, m_nd["tpi_ndjf"], 1), yr_x)
    r_dt, pr_dt, pe_dt, ne_dt, n_dt = _bretherton_corr(cd, td, n_controls=0)
    rows.append({"category": "chl_ndjf_vs_tpi_detrended", "n": n_dt, "value": r_dt, "p_raw": pr_dt, "p_eff_or_mw": pe_dt, "detail": f"neff={ne_dt:.1f};linearly_detrended_NDJF"})
    cr = m_nd["chl"] - np.polyval(np.polyfit(m_nd["nino"], m_nd["chl"], 1), m_nd["nino"])
    tr = m_nd["tpi_ndjf"] - np.polyval(np.polyfit(m_nd["nino"], m_nd["tpi_ndjf"], 1), m_nd["nino"])
    r, pr, pe, neff, n = _bretherton_corr(cr, tr, n_controls=1)
    rows.append({"category": "chl_ndjf_vs_tpi_partial_ex_nino34", "n": n, "value": r, "p_raw": pr, "p_eff_or_mw": pe, "detail": f"neff={neff:.1f};df=neff-3;corr_tpi_nino34={r_tn:.3f};contemporaneous_NDJF"})
    coords_92 = reef_coordinates()
    coords_92 = coords_92[coords_92["reef_name"].isin(residuals["reef_name"].unique())]
    with xr.open_dataset(DATA_DIR / "climate/GBR_chlorophyll_monthly_1998_2025.nc") as ds_c:
        vname = [v for v in ds_c.data_vars if "chl" in v.lower()][0]
        da_r = ds_c[vname].sel(latitude=slice(-25, -10), longitude=slice(142, 154))
        lat_n = [d for d in da_r.dims if "lat" in d.lower()][0]
        lon_n = [d for d in da_r.dims if "lon" in d.lower()][0]
        rlat_c = xr.DataArray(coords_92["lat"].to_numpy(), dims="reef")
        rlon_c = xr.DataArray(coords_92["lon"].to_numpy(), dims="reef")
        ts_reef = da_r.sel({lat_n: rlat_c, lon_n: rlon_c}, method="nearest").mean("reef").to_series().dropna()
        ts_reef.index = pd.to_datetime(ts_reef.index)
        df_rm = pd.DataFrame({"chl": ts_reef, "year": ts_reef.index.year, "month": ts_reef.index.month})
        df_rm["syr"] = np.where(df_rm["month"] >= 11, df_rm["year"] + 1, df_rm["year"])
        chl_ndjf_reef = df_rm[df_rm["month"].isin([11, 12, 1, 2])].groupby("syr").filter(lambda g: len(g) == 4).groupby("syr")["chl"].mean()
    m_rf = pd.concat([chl_ndjf_reef.rename("chl"), tpi_ndjf.rename("tpi_ndjf")], axis=1).dropna()
    r_rf, pr_rf, pe_rf, ne_rf, n_rf = _bretherton_corr(m_rf["chl"], m_rf["tpi_ndjf"], n_controls=0)
    rows.append({"category": "chl_ndjf_enhm92_reef_pixels_vs_tpi", "n": n_rf, "value": r_rf, "p_raw": pr_rf, "p_eff_or_mw": pe_rf, "detail": f"neff={ne_rf:.1f};midshelf_reef_pixels_only"})

    # 2. 13-yr low-pass TPI on three rivers (with circular time-shift p-value accounting for low-pass autocorrelation)
    rivers_df = pd.read_csv(RESULTS_DIR / "source_data/river_discharge_annual_m3_s.csv", index_col=0)
    for rcol in ("Pascoe", "Burdekin", "Fitzroy"):
        s = rivers_df[rcol].dropna()
        qn = s[s.index.isin(lp_neg)]
        qp = s[s.index.isin(lp_pos)]
        obs_d = float(qn.mean() - qp.mean())
        _, pmw = stats.mannwhitneyu(qn, qp, alternative="two-sided")
        common_y = sorted(s.index.intersection(ipo_lp.index))
        qv = s.loc[common_y].to_numpy(dtype=float)
        lv = ipo_lp.loc[common_y].to_numpy(dtype=float)
        sh_ex = 0
        for sh in range(len(common_y)):
            lr = np.roll(lv, sh)
            nm = lr < lp_mu - 0.5 * lp_sd
            pm = lr > lp_mu + 0.5 * lp_sd
            if nm.any() and pm.any() and abs(float(qv[nm].mean() - qv[pm].mean())) >= abs(obs_d) - 1e-12:
                sh_ex += 1
        p_shift = float(sh_ex / len(common_y))
        rows.append({
            "category": f"river_{rcol.lower()}_lp13_diff_m3_s",
            "n": len(qn) + len(qp),
            "value": obs_d,
            "p_raw": float(pmw),
            "p_eff_or_mw": p_shift,
            "detail": f"neg_n={len(qn)},pos_n={len(qp)};mwu_p={pmw:.4f};circ_shift_p={p_shift:.4f}",
        })

    # 3. GLORYS vo subset consistency (113 vs 92 vs 58) & NDJF
    rc = pd.read_csv(RESULTS_DIR / "source_data/reef_current_phase_composites.csv")
    enhm92 = set(residuals["reef_name"].unique())
    scr58 = set(periods["reef_name"].unique())
    for lbl, sset in (("all113", set(rc["reef_name"])), ("enhm92", enhm92), ("screened58", scr58)):
        sub = rc[rc["reef_name"].isin(sset)]
        cent = sub[sub["lat"].abs().between(16.0, 20.0)]
        burd = sub[sub["nrm_region"].eq("Burdekin")]
        pos_n = int((sub["difference_vo_m_s"] > 0).sum())
        rows.append({
            "category": f"glorys_annual_{lbl}",
            "n": len(sub),
            "value": float(100.0 * pos_n / len(sub)),
            "p_raw": float(cent["positive_vo_m_s"].mean()),
            "p_eff_or_mw": float(cent["negative_vo_m_s"].mean()),
            "detail": f"pos_diff={pos_n}/{len(sub)};burd={burd['positive_vo_m_s'].mean():+.4f}->{burd['negative_vo_m_s'].mean():+.4f}",
        })

    nd_neg_yrs = set(tpi_ndjf[tpi_ndjf < tpi_ndjf.mean() - 0.5 * tpi_ndjf.std()].index)
    nd_pos_yrs = set(tpi_ndjf[tpi_ndjf > tpi_ndjf.mean() + 0.5 * tpi_ndjf.std()].index)
    with xr.open_dataset(DATA_DIR / "climate/GBR_currents_monthly_1993_2025.nc") as ds_v:
        vo = ds_v.vo.squeeze("depth", drop=True)
        dts = pd.to_datetime(vo.time.values)
        ndjf_m = dts.month.isin([11, 12, 1, 2])
        vo_nd = vo.sel(time=ndjf_m)
        syrs = np.where(dts[ndjf_m].month >= 11, dts[ndjf_m].year + 1, dts[ndjf_m].year)
        vo_nd = vo_nd.assign_coords(syr=("time", syrs)).groupby("syr").mean("time")
        cnts = pd.Series(syrs).value_counts()
        vo_nd = vo_nd.sel(syr=vo_nd.syr.isin(cnts[cnts == 4].index))
        vn = vo_nd.sel(syr=vo_nd.syr.isin(list(nd_neg_yrs))).mean("syr")
        vp = vo_nd.sel(syr=vo_nd.syr.isin(list(nd_pos_yrs))).mean("syr")
        rlat = xr.DataArray(rc["lat"].to_numpy(), dims="reef")
        rlon = xr.DataArray(rc["lon"].to_numpy(), dims="reef")
        rc["ndjf_pos"] = vp.sel(latitude=rlat, longitude=rlon, method="nearest").values
        rc["ndjf_neg"] = vn.sel(latitude=rlat, longitude=rlon, method="nearest").values
        cent = rc[rc["lat"].abs().between(16.0, 20.0)]
        burd = rc[rc["nrm_region"].eq("Burdekin")]
        c_pos, c_neg = float(cent["ndjf_pos"].mean()), float(cent["ndjf_neg"].mean())
        b_pos, b_neg = float(burd["ndjf_pos"].mean()), float(burd["ndjf_neg"].mean())
        n_neg_nd = int(vo_nd.syr.isin(list(nd_neg_yrs)).sum())
        n_pos_nd = int(vo_nd.syr.isin(list(nd_pos_yrs)).sum())
        cent_lat = xr.DataArray(cent["lat"].to_numpy(), dims="reef")
        cent_lon = xr.DataArray(cent["lon"].to_numpy(), dims="reef")
        s_nd_cent = vo_nd.sel(latitude=cent_lat, longitude=cent_lon, method="nearest").mean("reef").to_series()
        _, p_mwu_nd = stats.mannwhitneyu(s_nd_cent[s_nd_cent.index.isin(nd_neg_yrs)], s_nd_cent[s_nd_cent.index.isin(nd_pos_yrs)], alternative="two-sided")
        m_vc = pd.concat([s_nd_cent.rename("vo"), tpi_ndjf.rename("tpi")], axis=1).dropna()
        r_vc, _, p_neff_vc, _, _ = _bretherton_corr(m_vc["vo"], m_vc["tpi"], n_controls=0)
        rows.append({
            "category": "glorys_ndjf_central_16_20S_weakening_pct",
            "n": len(cent),
            "value": float(100.0 * (c_pos - c_neg) / c_pos),
            "p_raw": c_pos,
            "p_eff_or_mw": c_neg,
            "detail": f"contemporaneous_NDJF_TPI(n={n_neg_nd}_vs_{n_pos_nd});mwu_p={p_mwu_nd:.4f};cont_r={r_vc:+.3f}(p_neff={p_neff_vc:.4f});burd_weakening_pct={100.0*(b_pos-b_neg)/b_pos:.1f}({b_pos:+.4f}->{b_neg:+.4f})",
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

    # 3. 2-year lag IPO-band reefs (using deterministic per-reef RNG at N=10000)
    groups = reef_series()
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
        rng2 = reef_rng(reef, "wn", seed)
        perms = np.asarray([rng2.permutation(res) for _ in range(10000)])
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
        ("primary_12_lag3", f"Primary {len(l12)} IPO reefs (3-yr lag)", s12, l12, th12, mean_T12, None, "#c0392b", "-", 2.0),
        ("all_58_weighted_lag3", f"All {len(l58)} reefs, power-weighted (3-yr lag)", s58, l58, th58, mean_T12, w58, "#1f77b4", "-", 1.9),
        ("all_58_unweighted_lag3", f"All {len(l58)} reefs, unweighted (3-yr lag)", s58, l58, th58, mean_T12, None, "#4d6475", "--", 1.5),
        ("lag2_17_reefs", f"{len(l2)} IPO-band reefs (2-yr lag)", s2, l2, th2, mean_T2, None, "#2ca02c", "-", 1.8),
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
    # Also evaluate full unscreened reef sets (all 80 nonzero-variance ENHM-fitted reefs & all 112 raw COTS reefs with >=5 surveys)
    cots_all, _ = ecological_observations()
    s80, l80, w80 = [], [], []
    for rn, grp in sorted(residuals.groupby("reef_name")):
        if rn not in coords_idx.index:
            continue
        grp = grp.sort_values("time_years")
        t = grp["time_years"].to_numpy(dtype=float) + 1970.0
        y = grp["residual"].to_numpy(dtype=float)
        if len(t) < 3 or np.ptp(t) <= 3.0 or np.std(y) <= 1e-10:
            continue
        X = np.column_stack([np.cos(omega12 * t), np.sin(omega12 * t)])
        yc = y - np.mean(y)
        coef, _, _, _ = np.linalg.lstsq(X, yc, rcond=None)
        s80.append((t, y))
        l80.append(abs(float(coords_idx.loc[rn, "lat"])))
        w80.append(min(1.0, float(np.sum((X @ coef) ** 2) / np.sum(yc ** 2))))
    s112, l112, w112 = [], [], []
    for rn, grp in sorted(cots_all.groupby("reef_name")):
        if rn not in coords_idx.index:
            continue
        grp = grp.sort_values("time_years")
        t = grp["time_years"].to_numpy(dtype=float) + 1970.0
        y = np.log1p(grp["cots_density"].to_numpy(dtype=float))
        if len(t) < 5 or np.ptp(t) <= 3.0 or np.std(y) <= 1e-10:
            continue
        X = np.column_stack([np.cos(omega12 * t), np.sin(omega12 * t)])
        yc = y - np.mean(y)
        coef, _, _, _ = np.linalg.lstsq(X, yc, rcond=None)
        s112.append((t, y))
        l112.append(abs(float(coords_idx.loc[rn, "lat"])))
        w112.append(min(1.0, float(np.sum((X @ coef) ** 2) / np.sum(yc ** 2))))
    extra_rows = []
    nc_mask = l58 <= 20.5
    s49 = [s58[i] for i in range(len(s58)) if nc_mask[i]]
    l49 = l58[nc_mask]
    w49 = w58[nc_mask]
    for ex_key, slist, llist, wlist in (
        ("north_central_le20.5S_weighted_lag3", s49, l49, w49),
        ("north_central_le20.5S_unweighted_lag3", s49, l49, None),
        ("all_80_enhm_fitted_lag3", s80, np.asarray(l80), np.asarray(w80)),
        ("all_112_raw_monitored", s112, np.asarray(l112), np.asarray(w112)),
    ):
        ths_ex = _extract_phases(slist, mean_T12)
        Rc_ex = _scan_resultant(ths_ex, llist, ks, weights=wlist)
        ip_ex = np.where(pos_mask)[0][int(np.argmax(Rc_ex[pos_mask]))]
        in_ex = np.where(neg_mask)[0][int(np.argmax(Rc_ex[neg_mask]))]
        kp_ex, Rp_ex = float(ks[ip_ex]), float(Rc_ex[ip_ex])
        kn_ex, Rn_ex = float(ks[in_ex]), float(Rc_ex[in_ex])
        spd_ex = float(111.0 / (abs(kp_ex) * mean_T12 / (2.0 * np.pi)))
        nulls_ex = _test_directional_nulls(slist, llist, mean_T12, Rp_ex, k_min=0.05, k_max=2.0,
                                           weights=wlist, n_iter=n_iter, seed=seed)
        extra_rows.append({
            "dataset": ex_key,
            "n_reefs": len(llist),
            "mean_period_years": mean_T12,
            "k_north_to_south": kp_ex,
            "R_north_to_south": Rp_ex,
            "speed_km_per_year_ns": spd_ex,
            "p_uniform_ns": nulls_ex["p_uniform"],
            "p_block_ns": nulls_ex["p_block"],
            "p_ar1_ns": nulls_ex["p_ar1"],
            "k_alias_neg": kn_ex,
            "R_alias_neg": Rn_ex,
            "delta_k_alias": kp_ex - kn_ex,
        })
    diag_df = pd.concat([diag_df, pd.DataFrame(extra_rows)], ignore_index=True)
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
                  label=f"{label} (N→S: {spd_p:.1f} km yr⁻¹, AR(1) p = {nulls_p['p_ar1']:.3f})")
        ax_a.scatter([kp], [Rp], color=color, s=36, zorder=6, edgecolors="white", linewidths=0.6)

    ax_a.text(1.10, 1.075, "N→S a priori branch\n(30–32 km yr⁻¹)",
              ha="center", va="top", fontsize=9, fontweight="bold", color="#902018")
    ax_a.text(-0.68, 1.075, "2π approx. spatial-alias peak\n(Δk ≈ 2π / 3.7° ≈ 1.70)",
              ha="center", va="top", fontsize=9, fontweight="bold", color="#444444")
    ax_a.set_xlim(-1.8, 1.8)
    ax_a.set_ylim(0.0, 1.12)
    ax_a.set_xlabel("Latitudinal phase gradient k (rad deg⁻¹; k > 0: North → South)")
    ax_a.set_ylabel("Mean resultant length R(k)")
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

        # Weighted residual bootstrap CI matching fitted weights
        rng_b = np.random.default_rng(42)
        boot = np.empty((1000, len(x_line)))
        y_hat = fit_at - y_off
        w_n = w / np.sum(w)
        resid_c = resid - float(np.sum(w_n * resid))
        X_l = np.column_stack([np.ones_like(lats), lats])
        XtW = X_l.T * w_n[None, :]
        XtWX_inv = np.linalg.inv(XtW @ X_l)
        for bi in range(1000):
            ib = rng_b.choice(len(lats), size=len(lats), replace=True)
            yb = y_hat + resid_c[ib]
            bb = XtWX_inv @ (XtW @ yb)
            boot[bi] = (bb[0] + bb[1] * x_line) * rad_to_yr
        ax.fill_between(x_line, np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0),
                        color="#a6cee3", alpha=0.45, label="95% bootstrap CI")
        ax.plot(x_line, y_line, color="#c0392b", lw=1.9, label="Circular–linear fit")

        if highlight_mask is not None:
            sizes = 20 + 75 * (w / max(np.max(w), 1e-6))
            ax.scatter(lats[~highlight_mask], y_obs[~highlight_mask], s=sizes[~highlight_mask],
                       color="#5b7c99", alpha=0.65, edgecolors="white", linewidths=0.5, zorder=4,
                       label=f"Other screened reefs (n = {int((~highlight_mask).sum())})")
            ax.scatter(lats[highlight_mask], y_obs[highlight_mask], s=sizes[highlight_mask] + 10,
                       color="#c0392b", alpha=0.95, edgecolors="black", linewidths=0.7, zorder=5,
                       label=f"Primary {int(highlight_mask.sum())} IPO-band reefs")
        else:
            ax.scatter(lats, y_obs, s=44, color=pt_color, edgecolors="white", linewidths=0.6, zorder=5)

        ax.set_xlabel("Latitude (°S)")
        ax.set_ylabel("Relative peak timing (yr)")
        ax.set_title(title_str, loc="left", pad=6)
        ax.text(-0.14, 1.08, panel_lbl, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
        ax.text(
            0.04, 0.96,
            f"n = {len(lats)} reefs (k = {kp:+.3f})\n"
            f"R = {Rp:.3f} (AR(1) p = {p_ar1:.3f}, Block p = {p_blk:.3f})\n"
            f"Speed = {spd:.1f} km yr⁻¹ (mean T = {T_val:.1f} yr)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
            bbox=dict(facecolor="white", alpha=0.90, edgecolor="#cccccc", linewidth=0.6),
        )
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, edgecolor="#cccccc", fontsize=8.0)
        ax.spines[["top", "right"]].set_visible(False)

    # Panel B: Primary 12 IPO-band reefs (high-SNR verification zoom, complementing Main Text Fig. 3B all-58)
    kp_12, Rp_12, spd_12, null_12, T12 = fits_ns["primary_12_lag3"]
    _plot_branch(ax_b, l12, th12, kp_12, Rp_12, spd_12, null_12["p_ar1"], null_12["p_block"], T12,
                 f"Primary {len(l12)} IPO-band reefs (3-yr lag)", "B",
                 pt_color="#c0392b")

    kp_2, Rp_2, spd_2, null_2, T2 = fits_ns["lag2_17_reefs"]
    _plot_branch(ax_c, l2, th2, kp_2, Rp_2, spd_2, null_2["p_ar1"], null_2["p_block"], T2,
                 f"2-yr lag IPO reefs (n = {len(l2)}, N→S branch)", "C",
                 pt_color="#2ca02c")

    save_publication(fig, FIGURES_DIR / "FigureS6_Phase_Sensitivity")
    plt.close(fig)
    return diag_df


def run_fixed_period_phase_sensitivity(residuals, coords, periods_to_test=(18.0, 19.0, 20.0, 21.0, 21.3, 22.0, 23.0, 24.0, 25.0), n_iter=500, seed=42):
    """Evaluate all-58-reef weighted circular-linear phase coherence across a priori fixed periods T=18-25 yr."""
    coords_idx = coords.set_index("reef_name")
    s_list, l_list = [], []
    for rn, grp in sorted(residuals.groupby("reef_name")):
        grp = grp.sort_values("time_years")
        t = grp["time_years"].to_numpy(dtype=float)
        vals = grp["residual"].to_numpy(dtype=float)
        if spectrum(t, vals) is None or rn not in coords_idx.index:
            continue
        s_list.append((t + 1970.0, vals))
        l_list.append(abs(float(coords_idx.loc[rn, "lat"])))
    l_arr = np.asarray(l_list)
    ks_ns = np.linspace(0.05, 2.0, 1951)
    ks_sn = np.linspace(-2.0, -0.05, 1951)
    rows = []
    for T in periods_to_test:
        omega = 2.0 * np.pi / float(T)
        w_list = []
        for t_yr, vals in s_list:
            X = np.column_stack([np.cos(omega * t_yr), np.sin(omega * t_yr)])
            yc = vals - np.mean(vals)
            coef, _, _, _ = np.linalg.lstsq(X, yc, rcond=None)
            w_list.append(float(np.sum((X @ coef) ** 2) / np.sum(yc ** 2)))
        w_arr = np.asarray(w_list)
        th_arr = _extract_phases(s_list, float(T))
        R_ns = _scan_resultant(th_arr, l_arr, ks_ns, weights=w_arr)
        R_sn = _scan_resultant(th_arr, l_arr, ks_sn, weights=w_arr)
        bi = int(np.argmax(R_ns)); kp, Rp = float(ks_ns[bi]), float(R_ns[bi])
        bi_s = int(np.argmax(R_sn)); km, Rm = float(ks_sn[bi_s]), float(R_sn[bi_s])
        spd = float(111.0 / (abs(kp) * float(T) / (2.0 * np.pi)))
        nulls = _test_directional_nulls(s_list, l_arr, float(T), Rp, k_min=0.05, k_max=2.0,
                                        weights=w_arr, n_iter=n_iter, seed=seed)
        rows.append({
            "period_yr": float(T),
            "k_ns_rad_per_deg": kp,
            "speed_km_per_yr": spd,
            "R_ns_weighted": Rp,
            "k_alias_rad_per_deg": km,
            "R_alias_weighted": Rm,
            "p_ar1": nulls["p_ar1"],
            "p_block": nulls["p_block"],
        })
    return pd.DataFrame(rows)


def run_unfolded_timedomain_lag_test(residuals, coords, max_lag=7, n_perm=5000, seed=42):
    """Unfolded time-domain cross-correlation within a half-cycle window (|lag|<=7 yr) breaking circular 2pi aliasing."""
    cots_df, _ = ecological_observations()
    cots_m = cots_df.merge(coords[["reef_name", "lat"]], on="reef_name", how="inner").copy()
    cots_m["abs_lat"] = cots_m["lat"].abs()
    cots_m["val"] = np.log1p(pd.to_numeric(cots_m["cots_density"], errors="coerce"))
    cots_m = cots_m.dropna(subset=["val", "year"])

    res_m = residuals.merge(coords[["reef_name", "lat"]], on="reef_name", how="inner").copy()
    res_m["abs_lat"] = res_m["lat"].abs()

    yrs = np.arange(1985, 2026)
    rows = []
    for ds_name, df_in, col in (("raw_113_reefs", cots_m, "val"), ("enhm_residuals", res_m, "residual")):
        for pair_name, ba, bb in (
            ("NorthInit_14_16.5S_to_Central_16.5_20S", (14.0, 16.5), (16.5, 20.0)),
            ("Central_16.5_20S_to_South_20_23.5S", (16.5, 20.0), (20.0, 23.5)),
        ):
            a = df_in[df_in["abs_lat"].between(*ba)].groupby("year")[col].mean().reindex(yrs).interpolate(limit=2).values
            b = df_in[df_in["abs_lat"].between(*bb)].groupby("year")[col].mean().reindex(yrs).interpolate(limit=2).values

            def _corr_at(arr_a, arr_b, lag):
                x, y = (arr_a[:len(yrs) - lag], arr_b[lag:]) if lag >= 0 else (arr_a[-lag:], arr_b[:len(yrs) + lag])
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 6:
                    return np.nan, int(m.sum()), x[m], y[m]
                return float(stats.pearsonr(x[m], y[m])[0]), int(m.sum()), x[m], y[m]

            lag_rs = {l: _corr_at(a, b, l) for l in range(-max_lag, max_lag + 1)}
            best_pos_lag = max(range(1, max_lag + 1), key=lambda l: lag_rs[l][0] if np.isfinite(lag_rs[l][0]) else -2.0)
            best_neg_lag = max(range(-max_lag, 0), key=lambda l: lag_rs[l][0] if np.isfinite(lag_rs[l][0]) else -2.0)
            r_obs, n_obs, xm, ym = lag_rs[best_pos_lag]
            r_neg = lag_rs[best_neg_lag][0]
            r1x = float(np.corrcoef(xm[:-1], xm[1:])[0, 1])
            r1y = float(np.corrcoef(ym[:-1], ym[1:])[0, 1])
            neff = max(3.0, min(float(n_obs), n_obs * (1.0 - r1x * r1y) / (1.0 + r1x * r1y)))
            tstat = r_obs * np.sqrt((neff - 2.0) / max(1e-12, 1.0 - r_obs ** 2))
            p_neff = float(2.0 * stats.t.sf(abs(tstat), df=neff - 2.0))

            blocks = [b[i:i + 3] for i in range(0, len(b), 3)]
            rng = np.random.default_rng(seed)
            ex_dir = 0
            for _ in range(n_perm):
                bp = np.concatenate([blocks[k] for k in rng.permutation(len(blocks))])[:len(b)]
                rp_max = max(_corr_at(a, bp, l)[0] for l in range(1, max_lag + 1))
                if np.isfinite(rp_max) and rp_max >= r_obs - 1e-12:
                    ex_dir += 1
            p_blk = float((ex_dir + 1) / (n_perm + 1))
            rows.append({
                "dataset": ds_name,
                "band_pair": pair_name,
                "best_ns_lag_yr": int(best_pos_lag),
                "r_best_ns": r_obs,
                "n_years": n_obs,
                "neff": neff,
                "p_neff_two_sided": p_neff,
                "p_block_3yr_one_sided": p_blk,
                "best_sn_lag_yr": int(best_neg_lag),
                "r_best_sn": r_neg,
            })
    return pd.DataFrame(rows)


def _stationary_bootstrap_diff_p(v_arr, neg, pos, diff_obs, mean_block=3.0, n_perm=20000, seed=42):
    """Boundary-free stationary bootstrap (Politis & Romano 1994) with geometric block length L=mean_block."""
    n = len(v_arr)
    if n < 4 or neg.sum() == 0 or pos.sum() == 0:
        return np.nan
    p_start = 1.0 / float(mean_block)
    rng = np.random.default_rng(seed)
    idx = np.empty((n_perm, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_perm)
    starts = rng.random((n_perm, n - 1)) < p_start
    rand_pos = rng.integers(0, n, size=(n_perm, n - 1))
    for t in range(1, n):
        idx[:, t] = np.where(starts[:, t - 1], rand_pos[:, t - 1], (idx[:, t - 1] + 1) % n)
    surr = v_arr[idx]
    diffs = surr[:, neg].mean(axis=1) - surr[:, pos].mean(axis=1)
    ex = int(np.sum(np.abs(diffs) >= abs(diff_obs) - 1e-12))
    return float((ex + 1) / (n_perm + 1))


def _ebisuzaki_phase_rand_diff_p(v_arr, neg, pos, diff_obs, n_perm=20000, seed=42):
    """Parameter-free Ebisuzaki (1997) Fourier phase randomization preserving the exact empirical power spectrum."""
    n = len(v_arr)
    if n < 4 or neg.sum() == 0 or pos.sum() == 0:
        return np.nan
    v_centered = v_arr - float(np.mean(v_arr))
    F = np.fft.rfft(v_centered)
    amp = np.abs(F)
    m = len(amp)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_perm, m))
    phases[:, 0] = 0.0
    if n % 2 == 0:
        phases[:, -1] = 0.0
    F_surr = amp[None, :] * np.exp(1j * phases)
    if n % 2 == 0:
        F_surr[:, -1] = F[-1] * rng.choice([-1.0, 1.0], size=n_perm)
    surr = np.fft.irfft(F_surr, n=n, axis=1) + float(np.mean(v_arr))
    diffs = surr[:, neg].mean(axis=1) - surr[:, pos].mean(axis=1)
    ex = int(np.sum(np.abs(diffs) >= abs(diff_obs) - 1e-12))
    return float((ex + 1) / (n_perm + 1))


def run_direct_ipo_phase_locking_check(periods, residuals, coords, v_speed=31.2, n_perm=20000, seed=42):
    """Evaluate all-92 ENHM-fitted and all-113 monitored reef window contrasts against NOAA TPI physical zero (1971-2000 ref)."""
    _, _, ipo_s = ipo_phase_years()
    ipo_s = ipo_s.sort_index().dropna()
    clim_mean = float(ipo_s.loc[1971:2000].mean()) if 1971 in ipo_s.index else 0.0
    ipo_map = {int(k): float(v - clim_mean) for k, v in ipo_s.items()}

    df92 = residuals.merge(coords[["reef_name", "lat"]], on="reef_name", how="inner").copy()
    df92["abs_lat"] = df92["lat"].abs()
    df92["anom"] = df92["residual"] - df92.groupby("reef_name")["residual"].transform("mean")

    cots_all, _ = ecological_observations()
    df113 = cots_all.merge(coords[["reef_name", "lat"]], on="reef_name", how="inner").copy()
    df113["abs_lat"] = df113["lat"].abs()
    df113["log_cots"] = np.log1p(pd.to_numeric(df113["cots_density"], errors="coerce"))
    df113["anom"] = df113["log_cots"] - df113.groupby("reef_name")["log_cots"].transform("mean")

    rows = []

    def _window_contrast(zdf, col, test_lbl, reg_lbl, y_max=2025):
        za = zdf[zdf["year"].between(1983, y_max)].groupby("year")[col].mean().reset_index()
        za["tpi"] = za["year"].map(ipo_map)
        za = za.dropna().sort_values("year")
        neg = (za["tpi"] < 0.0).to_numpy()
        pos = (za["tpi"] > 0.0).to_numpy()
        v_arr = za[col].to_numpy(dtype=float)
        diff = float(v_arr[neg].mean() - v_arr[pos].mean())
        _, p_mwu = stats.mannwhitneyu(v_arr[neg], v_arr[pos], alternative="two-sided")
        p_sb = _stationary_bootstrap_diff_p(v_arr, neg, pos, diff, mean_block=3.0, n_perm=n_perm, seed=seed)
        p_eb = _ebisuzaki_phase_rand_diff_p(v_arr, neg, pos, diff, n_perm=n_perm, seed=seed)
        rows.append({
            "test_type": test_lbl,
            "region_or_lag": reg_lbl,
            "statistic": diff,
            "p_param_or_mwu": float(p_mwu),
            "p_stat_boot_L3_or_neff": p_sb,
            "p_ebisuzaki_phase": p_eb,
            "n_obs": int(neg.sum() + pos.sum()),
        })

    # 1. All 92 ENHM-fitted reefs (reef-centered anomalies, 1983-2025 full span & 1983-2019 pre-bleaching/culling)
    _window_contrast(df92, "anom", "Window_ENHM92_TPI_neg_minus_pos", "All_92_reefs_1983_2025", 2025)
    _window_contrast(df92, "anom", "Window_ENHM92_TPI_neg_minus_pos", "All_92_reefs_1983_2019_pre2020", 2019)
    for zname, lo, hi in (
        ("NorthInit_14_16.5S_92reefs", 14.0, 16.5),
        ("Central_16_20S_92reefs", 16.0, 20.0),
        ("SouthFitzroy_20_23.5S_92reefs", 20.0, 23.5),
    ):
        _window_contrast(df92[df92["abs_lat"].between(lo, hi)], "anom", "Window_ENHM92_TPI_neg_minus_pos", zname, 2025)

    # 2. All 113 monitored reefs (raw within-reef anomalies)
    _window_contrast(df113, "anom", "Window_Raw113_TPI_neg_minus_pos", "All_113_reefs_1983_2025", 2025)
    _window_contrast(df113[df113["abs_lat"].between(20.0, 23.5)], "anom", "Window_Raw113_TPI_neg_minus_pos", "SouthFitzroy_20_23.5S_113reefs", 2025)

    # 3. Biologically plausible lagged correlation checks (TPI leads adult COTS by +2..+4 yr) on de-lagged 58-reef & all-92 residuals
    screened = df92[df92["reef_name"].isin(periods["reef_name"])].copy()
    t_cal = 1970.0 + screened["time_years"].to_numpy(dtype=float)
    screened["t_aligned"] = np.round(t_cal - (screened["abs_lat"].to_numpy(dtype=float) - 15.0) * 111.0 / float(v_speed)).astype(int)
    ann_delag = screened.groupby("t_aligned")["residual"].mean().sort_index()
    for span_lbl, yr_max in (("1983_2025", 2025), ("1983_2019", 2019)):
        sub_s = ann_delag[(ann_delag.index >= 1983) & (ann_delag.index <= yr_max)]
        for lag_y in (2, 3, 4):
            t_vec = np.array([ipo_map.get(int(yr) - lag_y, np.nan) for yr in sub_s.index])
            r_l, p_r, p_e, _, n_l = _bretherton_corr(sub_s.to_numpy(dtype=float), t_vec, n_controls=0)
            rows.append({
                "test_type": f"Lagged_Delagged58_vs_TPI_{span_lbl}",
                "region_or_lag": f"TPI_leads_+{lag_y}yr",
                "statistic": r_l,
                "p_param_or_mwu": p_r,
                "p_stat_boot_L3_or_neff": p_e,
                "p_ebisuzaki_phase": np.nan,
                "n_obs": n_l,
            })

    ann92 = df92[df92["year"].between(1983, 2025)].groupby("year")["anom"].mean().sort_index()
    for lag_y in (2, 3, 4):
        tpi_vec = np.array([ipo_map.get(int(yr) - lag_y, np.nan) for yr in ann92.index])
        r_lin, p_raw_lin, p_neff_lin, _, n_lin = _bretherton_corr(ann92.to_numpy(dtype=float), tpi_vec, n_controls=0)
        rows.append({
            "test_type": "Lagged_ENHM92_vs_annual_TPI",
            "region_or_lag": f"All_92_reefs_TPI_leads_+{lag_y}yr",
            "statistic": r_lin,
            "p_param_or_mwu": p_raw_lin,
            "p_stat_boot_L3_or_neff": p_neff_lin,
            "p_ebisuzaki_phase": np.nan,
            "n_obs": n_lin,
        })
    return pd.DataFrame(rows)


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
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-environment", action="store_true", help="Run ecology diagnostics without external monthly grids")
    args = parser.parse_args()
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

    ipo12_mean_T = float(periods[periods["primary_significant"].eq(True) & periods["primary_period"].between(15.0, 30.0)]["primary_period"].mean())
    baseline_sens = run_baseline_sensitivity(residuals, coords, ipo12_mean_T, n_boot=10000, n_phase_iter=500, seed=42)
    baseline_sens.to_csv(TABLES_DIR / "baseline_sensitivity_comparison.csv", index=False)

    lag_sens = run_lag_sensitivity(coords, n_boot=10000, seed=42)
    lag_sens.to_csv(TABLES_DIR / "lag_sensitivity_dated.csv", index=False)

    field_sig = run_field_significance(periods, residuals, n_iter=1000, seed=42)
    field_sig.to_csv(TABLES_DIR / "spectral_field_significance.csv", index=False)

    env_sens = pd.DataFrame()
    if not args.skip_environment:
        env_sens = run_environmental_sensitivity(periods, residuals)
        env_sens.to_csv(TABLES_DIR / "ndjf_and_tpi_sensitivity.csv", index=False)

    phase_diag = run_phase_aliasing_and_unconditional_diagnostics(periods, residuals, coords, n_iter=500, seed=42)
    fixed_t = run_fixed_period_phase_sensitivity(residuals, coords, n_iter=500, seed=42)
    fixed_t.to_csv(TABLES_DIR / "fixed_period_phase_sensitivity.csv", index=False)
    unfolded_lag = run_unfolded_timedomain_lag_test(residuals, coords, max_lag=7, n_perm=5000, seed=42)
    unfolded_lag.to_csv(TABLES_DIR / "unfolded_timedomain_lag_test.csv", index=False)
    direct_lock = run_direct_ipo_phase_locking_check(periods, residuals, coords, v_speed=31.2, n_perm=20000, seed=42)
    direct_lock.to_csv(TABLES_DIR / "direct_ipo_phase_locking_check.csv", index=False)
    table_s1 = export_table_s1(periods, coords)

    lines = [
        "1. Spectral null hierarchy (N=10000 per-reef deterministic seeds):",
        summary.to_string(index=False),
        "",
        "2. Uniform-frequency field significance (N=1000 ensembles):",
        field_sig.to_string(index=False),
        "",
        "3. Unwrap-free circular-linear spatial phase null (primary IPO subset):",
        phase_null.to_string(index=False) if not phase_null.empty else "Insufficient reefs",
        "",
        "4. Baseline model sensitivity + all-58-reef phase convergence (N=10000):",
        baseline_sens.to_string(index=False),
        "",
        "5. Dated lag sensitivity (full 113-reef dataset, N=10000):",
        lag_sens.to_string(index=False),
        "",
        "6. All-58-reef (direction-constrained) & 2-yr lag spatial-alias diagnostics:",
        phase_diag.to_string(index=False),
        "",
        "7. Environmental sensitivity (NDJF Chl-a ex-Nino3.4, LP13 rivers, GLORYS subsets & NDJF):",
        env_sens.to_string(index=False),
        "",
        f"8. Table S1 exported: {len(table_s1)} spectrally screened reefs",
        "",
        "9. A priori fixed evaluation period sensitivity (T=18-25 yr, all 58 reefs weighted):",
        fixed_t.to_string(index=False),
        "",
        "10. Unfolded time-domain lag cross-correlation (half-cycle |lag|<=7 yr):",
        unfolded_lag.to_string(index=False),
        "",
        "11. Direct 1:1 IPO phase-locking diagnostics (Test A de-lagged & Test B regional composites):",
        direct_lock.to_string(index=False),
    ]
    text = "\n".join(lines) + "\n"
    (TABLES_DIR / "robustness_null_comparison.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
