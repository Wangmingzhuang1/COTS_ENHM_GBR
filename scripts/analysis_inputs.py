"""Validated ecological inputs and shared climate conventions for Star_DLS."""

from functools import lru_cache

import numpy as np
import pandas as pd

if __package__:
    from .config import ECOLOGY_OBSERVATIONS, ECOLOGY_REEF_SITES, IPO_FILE, RUNOFF_DIR
else:
    from config import ECOLOGY_OBSERVATIONS, ECOLOGY_REEF_SITES, IPO_FILE, RUNOFF_DIR


@lru_cache(maxsize=1)
def ecological_observations():
    source = pd.read_csv(ECOLOGY_OBSERVATIONS, low_memory=False, parse_dates=["survey_date"])
    manta = source.loc[source["method"].eq("manta")].copy()
    cots = manta.loc[manta["measure"].eq("COTS_density")].copy()
    coral = manta.loc[
        manta["measure"].eq("HC_cover") & manta["project_code"].eq("LTMP") & manta["depth"].eq(9)
    ].copy()
    for label, frame in [("COTS", cots), ("manta hard coral", coral)]:
        if frame.empty or frame[["reef_key", "domain_name", "survey_date", "source_file", "source_row"]].isna().any().any():
            raise ValueError(f"Empty or incomplete {label} input in {ECOLOGY_OBSERVATIONS}")
        if frame.duplicated(["reef_key", "survey_date"]).any():
            raise ValueError(f"Duplicate {label} reef and survey date in {ECOLOGY_OBSERVATIONS}")

    if not cots["unit"].eq("COTS per tow").all() or not cots["tows"].gt(0).all() or not cots["cots"].ge(0).all():
        raise ValueError("Invalid COTS count, tow effort, or unit")
    cots["cots_density"] = cots["cots"] / cots["tows"]
    if not np.allclose(cots["cots_density"], cots["cotsptow"], atol=0.0050001, rtol=0):
        raise ValueError("COTS count/effort does not agree with the rounded published density")
    coral["coral_cover"] = coral["mean"]
    if not coral["unit"].eq("percent cover").all() or not coral["coral_cover"].between(0, 100).all():
        raise ValueError("Invalid hard coral cover or unit")

    names = cots[["reef_key", "domain_name"]].drop_duplicates()
    if names["reef_key"].duplicated().any() or names["domain_name"].duplicated().any():
        raise ValueError("Reef keys and COTS names must map one-to-one")
    for frame in (cots, coral):
        frame["year"] = frame["survey_date"].dt.year
        frame["time_years"] = (frame["survey_date"] - pd.Timestamp("1970-01-01")).dt.days / 365.2425
    cots["report_year"] = manta.loc[cots.index, "year"].astype(int)
    cots["reef_name"] = cots["domain_name"]
    coral["reef_name"] = coral["reef_key"].map(names.set_index("reef_key")["domain_name"])
    return cots.sort_values(["reef_key", "survey_date"]), coral.sort_values(["reef_key", "survey_date"])


def reef_coordinates():
    cots, _ = ecological_observations()
    coords = pd.read_csv(ECOLOGY_REEF_SITES).rename(columns={"latitude": "lat", "longitude": "lon"})
    matched = cots[["reef_key", "reef_name"]].drop_duplicates().merge(
        coords[["reef_key", "lat", "lon"]], on="reef_key", how="left", validate="one_to_one"
    )
    if not (matched["lat"].between(-25, -10) & matched["lon"].between(142, 155)).all():
        raise ValueError("Missing or out-of-domain coordinates for AIMS COTS reefs")
    return matched[["reef_name", "lat", "lon"]]


def reef_series():
    cots, coral = ecological_observations()
    groups = {}
    for name, observations in cots.groupby("reef_name"):
        cover = coral.loc[coral["reef_name"].eq(name)]
        if len(observations) < 5 or len(cover) < 5:
            continue
        response = observations[["survey_date", "year", "report_year", "time_years", "cots_density", "source_file", "source_row"]]
        predictor = cover[["survey_date", "coral_cover", "source_file", "source_row"]].rename(
            columns={"source_file": "coral_source_file", "source_row": "coral_source_row"}
        )
        frame = response.merge(predictor, on="survey_date", how="outer", validate="one_to_one").sort_values("survey_date")
        frame["log_cots"] = np.log1p(frame["cots_density"])
        frame["region"] = observations["nrm_region"].iloc[0]
        frame["sector"] = observations["a_sector"].iloc[0]
        groups[name] = frame
    return groups


def align_survey_lag(frame, lag_years, tolerance_days=183):
    """Match the nearest dated coral survey to the nominal prior anniversary."""
    response = frame.dropna(subset=["cots_density"])[
        ["survey_date", "year", "report_year", "time_years", "cots_density", "log_cots", "source_file", "source_row"]
    ].copy()
    predictor = frame.dropna(subset=["coral_cover"])[
        ["survey_date", "coral_cover", "coral_source_file", "coral_source_row"]
    ].rename(columns={"survey_date": "coral_date", "coral_cover": "coral_lagged"})
    response["target_date"] = response["survey_date"] - pd.DateOffset(years=int(lag_years))
    tolerance = pd.Timedelta(days=tolerance_days if lag_years else 0)
    pairs = pd.merge_asof(response.sort_values("target_date"), predictor.sort_values("coral_date"),
                          left_on="target_date", right_on="coral_date", direction="nearest", tolerance=tolerance)
    pairs = pairs.dropna(subset=["coral_date"]).copy()
    if lag_years:
        pairs = pairs.loc[pairs["coral_date"] < pairs["survey_date"]].copy()
    pairs["actual_lag_days"] = (pairs["survey_date"] - pairs["coral_date"]).dt.days
    return pairs.sort_values("survey_date").reset_index(drop=True)


@lru_cache(maxsize=1)
def monthly_ipo():
    values = pd.read_csv(IPO_FILE, encoding="gbk")
    years = pd.to_numeric(values.iloc[:, 0], errors="coerce")
    months = values.iloc[:, 1:13].apply(pd.to_numeric, errors="coerce").mask(lambda x: x.abs() >= 10)
    dates = pd.to_datetime({"year": np.repeat(years.to_numpy(), 12), "month": np.tile(np.arange(1, 13), len(years)), "day": 1}, errors="coerce")
    result = pd.Series(months.to_numpy().ravel(), index=pd.DatetimeIndex(dates), name="ipo").dropna().sort_index()
    result = result.loc[result.index.notna()]
    if not result.index.is_unique:
        raise ValueError("Duplicate months in IPO input")
    return result


def annual_ipo():
    grouped = monthly_ipo().resample("YE")
    series = grouped.mean().where(grouped.count().eq(12)).dropna()
    return pd.DataFrame({"year": series.index.year, "ipo": series.to_numpy()})


def ipo_phase_years():
    series = annual_ipo().set_index("year")["ipo"]
    delta = series - series.mean()
    width = 0.5 * series.std()
    return delta.index[delta < -width].tolist(), delta.index[delta > width].tolist(), series


def trailing_month_mean(monthly, survey_dates, months=12):
    """Mean over complete calendar months preceding each survey."""
    series = monthly.asfreq("MS").rolling(months, min_periods=months).mean()
    last_month = pd.to_datetime(survey_dates).dt.to_period("M").dt.to_timestamp() - pd.offsets.MonthBegin(1)
    return last_month.map(series)


def annual_discharge(river):
    paths = sorted((RUNOFF_DIR / river).glob("*.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one monthly discharge file for {river}")
    frame = pd.read_csv(paths[0], comment="#", names=["timestamp", "value", "quality", "interpolation"])
    dates = pd.to_datetime(frame["timestamp"], utc=True, errors="raise").dt.tz_convert("Australia/Brisbane")
    frame["year"] = dates.dt.year
    frame["month"] = dates.dt.month
    frame["days"] = dates.dt.days_in_month
    if frame.duplicated(["year", "month"]).any():
        raise ValueError(f"Duplicate monthly discharge records for {river}")
    valid = frame.loc[frame["quality"].isin([10, 90, 110]) & frame["value"].ge(0)].copy()
    valid["weighted_flow"] = valid["value"] * valid["days"]
    sums = valid.groupby("year").agg(months=("month", "size"), volume=("weighted_flow", "sum"), days=("days", "sum"))
    return (sums["volume"] / sums["days"]).where(sums["months"].eq(12)).dropna().rename("runoff")
