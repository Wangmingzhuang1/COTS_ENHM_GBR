"""Assess observed COTS growth and threshold crossings using antecedent climate."""

import numpy as np
import pandas as pd
from scipy.stats import t

from scripts.analysis_inputs import ecological_observations, monthly_ipo, trailing_month_mean
from scripts.config import TABLES_DIR
from pnas_aug.scripts._common import monthly_nino34


def build_transitions():
    cots, coral = ecological_observations()
    cots = cots[["reef_key", "survey_date", "cots_density", "source_file", "source_row"]].copy()
    groups = cots.groupby("reef_key", sort=False)
    for column in ("survey_date", "cots_density", "source_file", "source_row"):
        cots[f"next_{column}"] = groups[column].shift(-1)
    cots["gap_days"] = (cots["next_survey_date"] - cots["survey_date"]).dt.days
    cots = cots.loc[cots["gap_days"].between(183, 548)].copy()
    cover = coral[["reef_key", "survey_date", "coral_cover", "source_file", "source_row"]].rename(
        columns={"survey_date": "coral_date", "source_file": "coral_source_file", "source_row": "coral_source_row"}
    )
    data = pd.merge_asof(cots.sort_values("survey_date"), cover.sort_values("coral_date"),
                         left_on="survey_date", right_on="coral_date", by="reef_key",
                         direction="backward", tolerance=pd.Timedelta(days=365))
    data["start_year"] = data["survey_date"].dt.year
    data["gap_years"] = data["gap_days"] / 365.2425
    data["initial_log"] = np.log1p(data["cots_density"])
    data["annual_log_growth"] = (np.log1p(data["next_cots_density"]) - data["initial_log"]) / data["gap_years"]
    data["linear_time"] = (data["survey_date"] - pd.Timestamp("1980-01-01")).dt.days / 365.2425
    data["ipo_60_months"] = trailing_month_mean(monthly_ipo(), data["survey_date"], months=60)
    data["ipo_12_months"] = trailing_month_mean(monthly_ipo(), data["survey_date"])
    data["nino34_12_months"] = trailing_month_mean(monthly_nino34(), data["survey_date"])
    data = data.dropna(subset=["ipo_60_months", "ipo_12_months", "nino34_12_months"])
    for col in ("ipo_60_months", "ipo_12_months"):
        data[f"negative_{col}_sd"] = -(data[col] - data[col].mean()) / data[col].std(ddof=1)
    return data.sort_values(["reef_key", "survey_date"])


def fit_effect(data, endpoint, exposure, adjust_cover=True, bandwidth=4):
    """Within-reef slope with a Bartlett HAC covariance over common calendar-year scores."""
    variables = [exposure, "initial_log", "gap_years", "linear_time", "nino34_12_months"]
    if adjust_cover:
        variables.append("coral_cover")
    data = data.dropna(subset=variables + [endpoint]).copy()
    data = data.loc[data.groupby("reef_key")[endpoint].transform("size").ge(2)]
    group = data.groupby("reef_key")
    columns = variables + [endpoint]
    within = data[columns] - group[columns].transform("mean")
    x, y = within[variables].to_numpy(), within[endpoint].to_numpy()
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != len(variables):
        raise ValueError(f"Rank-deficient within-reef model: {endpoint}, {exposure}")
    score = pd.DataFrame(x * (y - x @ beta)[:, None], index=data["start_year"])
    years = np.arange(data["start_year"].min(), data["start_year"].max() + 1)
    annual = score.groupby(level=0).sum().reindex(years, fill_value=0).to_numpy()
    meat = annual.T @ annual
    for lag in range(1, bandwidth + 1):
        cross = annual[lag:].T @ annual[:-lag]
        meat += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
    year_clusters = data["start_year"].nunique()
    n_reefs = data["reef_key"].nunique()
    inverse = np.linalg.inv(x.T @ x)
    correction = year_clusters / (year_clusters - 1) * (len(data) - 1) / (len(data) - n_reefs - rank)
    variance = float((correction * inverse @ meat @ inverse)[0, 0])
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"Invalid calendar-year HAC variance: {endpoint}, {exposure}")
    error = np.sqrt(variance)
    critical = t.ppf(0.975, year_clusters - 1)
    return {"estimate": beta[0], "ci_low": beta[0] - critical * error, "ci_high": beta[0] + critical * error,
            "p_two_sided": 2 * t.sf(abs(beta[0] / error), year_clusters - 1), "n_transitions": len(data),
            "n_reefs": n_reefs, "n_start_years": year_clusters,
            "events": int(data[endpoint].sum()) if endpoint == "crossing" else np.nan}


def main():
    data = build_transitions()
    data.to_csv(TABLES_DIR / "ipo_outbreak_transition_input.csv", index=False)
    outputs = []
    for outcome, threshold in [("growth", None), ("active_outbreak", 1.0), ("incipient_outbreak", 0.22)]:
        sample = data
        endpoint = "annual_log_growth"
        if threshold is not None:
            sample = data.loc[data["cots_density"].lt(threshold)].copy()
            sample["crossing"] = sample["next_cots_density"].ge(threshold).astype(float)
            endpoint = "crossing"
        for window, cover in [(60, True), (60, False), (12, True)]:
            estimate = fit_effect(sample, endpoint, f"negative_ipo_{window}_months_sd", adjust_cover=cover)
            if threshold is not None:
                for column in ("estimate", "ci_low", "ci_high"):
                    estimate[column] *= 100
            outputs.append({"outcome": outcome, "threshold_cots_per_tow": threshold,
                            "ipo_window_months": window, "baseline_coral_adjustment": cover,
                            "unit_per_lower_ipo_sd": "percentage_points" if threshold else "log1p_density_per_year",
                            **estimate})
    effects = pd.DataFrame(outputs)
    effects.to_csv(TABLES_DIR / "ipo_outbreak_transition_effects.csv", index=False)
    print(effects.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
