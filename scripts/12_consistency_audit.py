"""Verify the rebuilt inputs, paired records, and published spectral tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_inputs import ecological_observations
from config import ECOLOGY_DATA_DIR, TABLES_DIR
from spectral_inference import holm_adjust, spectrum


def check(condition, label):
    if not condition:
        raise ValueError(f"Audit failed: {label}")


def check_raw_exports(observations, columns, date_column):
    for filename, group in observations.groupby("source_file"):
        path = (ECOLOGY_DATA_DIR / Path(filename).relative_to("data")).resolve()
        check(path.is_relative_to(ECOLOGY_DATA_DIR.resolve()), f"source path inside the ecology snapshot: {filename}")
        published = pd.read_csv(path).iloc[group["source_row"].astype(int).to_numpy() - 2]
        check(np.array_equal(published["domain_name"], group["domain_name"]), f"raw reef identity: {filename}")
        check(np.allclose(published[date_column], group["raw_decimal_date"], atol=1e-8, rtol=0),
              f"raw survey date: {filename}")
        check(np.allclose(published[columns], group[columns], atol=1e-8, rtol=0), f"raw measurement: {filename}")


def main():
    cots, coral = ecological_observations()
    check_raw_exports(cots, ["cots", "tows", "cotsptow"], "survey_date")
    check_raw_exports(coral, ["mean"], "date")
    residuals = pd.read_csv(TABLES_DIR / "all_residuals.csv", parse_dates=["survey_date", "coral_date"])
    models = pd.read_csv(TABLES_DIR / "model_results.csv").set_index("reef_name")
    periods = pd.read_csv(TABLES_DIR / "period_results.csv")
    significant = pd.read_csv(TABLES_DIR / "significant_reefs.csv")
    primary_lag = int(pd.read_csv(TABLES_DIR / "lag_analysis.csv").loc[lambda f: f["primary"], "lag_years"].iloc[0])
    dated = residuals.merge(
        cots[["source_file", "source_row", "survey_date", "reef_name", "cots_density"]],
        on=["source_file", "source_row", "survey_date", "reef_name"], validate="one_to_one",
        suffixes=("", "_source"),
    ).merge(
        coral.rename(columns={"source_file": "coral_source_file", "source_row": "coral_source_row", "survey_date": "coral_date"})[["coral_source_file", "coral_source_row", "coral_date", "reef_name", "coral_cover"]],
        on=["coral_source_file", "coral_source_row", "coral_date", "reef_name"], validate="many_to_one",
    )
    check(len(dated) == len(residuals), "every fitted observation resolves to both raw source rows")
    check(np.allclose(dated["cots_density"], dated["cots_density_source"]), "COTS per tow")
    check(np.allclose(dated["coral_lagged"], dated["coral_cover"]), "manta hard coral mean")
    target_date = dated["survey_date"] - pd.DateOffset(years=primary_lag)
    check((dated["coral_date"] < dated["survey_date"]).all()
          and ((dated["coral_date"] - target_date).dt.days.abs() <= 183).all(), "dated lag and time direction")
    check(np.allclose(dated["log_cots"], np.log1p(dated["cots_density"])), "log-transformed per-tow density")
    check(np.allclose(dated["predicted"], dated["reef_name"].map(models["beta_coefficient"]) * dated["coral_lagged"]),
          "fitted baseline coefficients")
    check(np.allclose(dated["log_cots"] - dated["predicted"], dated["residual"]), "residual arithmetic")
    check(dated.groupby("reef_name").size().sort_index().equals(models["n_observations"].sort_index()),
          "model and residual reef counts")
    expected = {}
    for reef, frame in dated.sort_values("time_years").groupby("reef_name"):
        values = spectrum(frame["time_years"].to_numpy(), frame["residual"].to_numpy())
        if values is not None:
            frequencies, power = values
            expected[reef] = 1 / frequencies[np.nanargmax(power)]
    check(set(expected) == set(periods["reef_name"]), "spectral eligibility reproduces")
    check(np.allclose(periods["reef_name"].map(expected), periods["primary_period"]), "peak values reproduce")
    check(np.allclose(holm_adjust(periods["ar1_p_value"]), periods["ar1_p_adjusted"]), "Holm correction on AR(1)")
    sig_names = set(periods.loc[periods["white_noise_p_value"].lt(0.05), "reef_name"])
    ipo_names = set(
        periods.loc[
            periods["white_noise_p_value"].lt(0.05) & periods["primary_period"].between(15, 30),
            "reef_name",
        ]
    )
    check(sig_names == set(significant["reef_name"])
          == set(periods.loc[periods["primary_significant"], "reef_name"]), "significant subset")
    lines = [
        "AIMS source-to-result audit passed",
        f"Source COTS records: {len(cots)}",
        f"Source manta hard coral records: {len(coral)}",
        f"Dated fitted observations: {len(dated)}",
        f"Fitted reefs: {len(models)}",
        f"Screened reefs: {len(periods)}",
        f"Significant reefs (white-noise p<0.05): {len(sig_names)}",
        f"IPO-band significant reefs (15-30 yr): {len(ipo_names)}",
    ]
    (TABLES_DIR / "manuscript_consistency_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
