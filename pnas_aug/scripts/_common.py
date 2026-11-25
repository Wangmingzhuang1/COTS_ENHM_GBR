from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from scripts.config import CLIMATE_DIR
from pnas_aug.config import ALL_RESIDUALS, MODEL_RESULTS

def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def fetch_noaa_nino34(cache_file: Path | None = None) -> pd.DataFrame:
    cache_file = cache_file or (CLIMATE_DIR / "enso_nino34_monthly.csv")
    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        url = "https://psl.noaa.gov/data/correlation/nina34.data"
        text = urllib.request.urlopen(url, timeout=20).read().decode("utf-8")
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 13:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            values = [float(x) for x in parts[1:13]]
            if all(v == -99.99 for v in values):
                continue
            rows.append([year, *values])
        tidy = pd.DataFrame(
            rows,
            columns=["year", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        )
        tidy.to_csv(cache_file, index=False)
        df = tidy
    if "year" not in df.columns:
        year_col = df.columns[0]
        month_cols = df.columns[1:13]
        df = df[[year_col, *month_cols]].copy()
        df.columns = ["year", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    month_cols = [c for c in df.columns if c != "year"]
    for col in month_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["year", *month_cols]].dropna(subset=["year"])


def monthly_nino34() -> pd.Series:
    nino = fetch_noaa_nino34().set_index("year")
    nino = nino.mask((nino < 10) | (nino > 40))
    anomalies = nino - nino.loc[1981:2010].mean()
    dates = pd.to_datetime({"year": np.repeat(nino.index.astype(int), 12),
                            "month": np.tile(np.arange(1, 13), len(nino)), "day": 1})
    return pd.Series(anomalies.to_numpy().ravel(), index=pd.DatetimeIndex(dates), name="nino34")


def fit_linear_model(y: pd.Series, predictors: pd.DataFrame) -> dict:
    X = predictors.copy()
    X.insert(0, "intercept", 1.0)
    x = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    beta, _, _, _ = np.linalg.lstsq(x, yv, rcond=None)
    fitted = x @ beta
    resid = yv - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    n = len(yv)
    p = x.shape[1] - 1
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)
    rmse = float(np.sqrt(np.mean(resid**2)))
    aic = n * np.log(max(ss_res / max(n, 1), 1e-12)) + 2 * x.shape[1]
    return {
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "aic": aic,
        "coefficients": dict(zip(X.columns, beta)),
        "fitted": fitted,
    }


def reef_residual_with_region() -> pd.DataFrame:
    residuals = pd.read_csv(ALL_RESIDUALS)
    model = pd.read_csv(MODEL_RESULTS)[["reef_name", "region", "sector"]].drop_duplicates()
    merged = residuals.merge(model, on="reef_name", how="left")
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
    merged["residual"] = pd.to_numeric(merged["residual"], errors="coerce")
    return merged.dropna(subset=["year", "residual"]).copy()
