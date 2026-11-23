import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MIN_N = 8
COVERAGE = 0.60
MIN_RHO = 0.30
MAX_Q = 0.20
MAX_CORR = 0.85
MAX_FEATURES = 4
MIN_STABLE = 2


# ---------------------------------------------------------
# STATISTICAL UTILITIES
# ---------------------------------------------------------

def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction."""

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(n)
    running_min = 1.0

    for k in range(n - 1, -1, -1):
        index = order[k]

        running_min = min(
            running_min,
            p_values[index] * n / (k + 1)
        )

        adjusted[index] = min(running_min, 1.0)

    return adjusted


def spearman_correlation(x, y):
    """
    Calculate pairwise-complete Spearman correlation.

    Missing observations are excluded, not imputed.
    """

    paired = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    n = len(paired)

    if (
        n < MIN_N
        or paired["x"].nunique() < 2
        or paired["y"].nunique() < 2
    ):
        return np.nan, np.nan, n

    rho, p_value = spearmanr(
        paired["x"],
        paired["y"]
    )

    if not np.isfinite(rho) or not np.isfinite(p_value):
        return np.nan, np.nan, n

    return float(rho), float(p_value), n


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def validate_inputs(weather, yield_data):

    weather_required = {
        "district_code",
        "year",
        "weather_column",
        "anomaly_z",
        "is_target_year",
        "target_year"
    }

    yield_required = {
        "district_code",
        "district_name",
        "year",
        "crop",
        "is_aggregate",
        "is_seasonal_component",
        "is_group_category",
        "modelling_role",
        "use_in_modelling",
        "eligible_for_baseline",
        "yield_anomaly_zscore",
        "anomaly_status"
    }

    missing_weather = weather_required - set(weather.columns)
    missing_yield = yield_required - set(yield_data.columns)

    if missing_weather or missing_yield:
        raise ValueError(
            f"Missing columns:\n"
            f"Weather: {sorted(missing_weather)}\n"
            f"Yield: {sorted(missing_yield)}"
        )

    if weather.duplicated(
        ["district_code", "year", "weather_column"]
    ).any():
        raise ValueError("Duplicate weather observation keys detected.")

    if yield_data.duplicated(
        ["district_code", "year", "crop"]
    ).any():
        raise ValueError("Duplicate yield observation keys detected.")

    if not (
        weather["target_year"].dropna().astype(int)
        ==
        weather.loc[
            weather["target_year"].notna(),
            "year"
        ].astype(int)
    ).all():
        raise ValueError("Weather target_year/year mismatch.")

    if not weather["is_target_year"].fillna(False).astype(bool).all():
        raise ValueError(
            "Expected forward-chaining target-year weather observations only."
        )


# ---------------------------------------------------------
# ELIGIBILITY FILTERING
# ---------------------------------------------------------

def filter_eligible_crops(data):

    mask = (
        data["eligible_for_baseline"].fillna(False).astype(bool)
        & data["use_in_modelling"].fillna(False).astype(bool)
        & ~data["is_aggregate"].fillna(False).astype(bool)
        & ~data["is_seasonal_component"].fillna(False).astype(bool)
        & ~data["is_group_category"].fillna(False).astype(bool)
        & data["modelling_role"].eq("include")
        & data["anomaly_status"].eq("calculated")
        & data["yield_anomaly_zscore"].notna()
    )

    return data.loc[mask].copy()


def get_eligible_crop_list(data):

    mask = (
        data["eligible_for_baseline"].fillna(False).astype(bool)
        & data["use_in_modelling"].fillna(False).astype(bool)
        & ~data["is_aggregate"].fillna(False).astype(bool)
        & ~data["is_seasonal_component"].fillna(False).astype(bool)
        & ~data["is_group_category"].fillna(False).astype(bool)
        & data["modelling_role"].eq("include")
    )

    return sorted(
        data.loc[mask, "crop"].dropna().unique()
    )


# ---------------------------------------------------------
# FEATURE SELECTION FOR ONE CROP / LOYO FOLD
# ---------------------------------------------------------

def select_features_for_fold(
    training_data,
    feature_columns,
    crop,
    heldout_year
):

    sample_count = len(training_data)

    if sample_count < MIN_N:
        return (
            [],
            [],
            "insufficient_training_samples"
        )

    minimum_pairs = max(
        MIN_N,
        int(np.ceil(COVERAGE * sample_count))
    )

    statistics = []

    # Calculate crop-specific weather/yield relationships.
    for feature in feature_columns:

        rho, p_value, pair_count = spearman_correlation(
            training_data[feature],
            training_data["yield_anomaly_zscore"]
        )

        if (
            np.isfinite(rho)
            and pair_count >= minimum_pairs
        ):

            statistics.append({
                "crop": crop,
                "heldout_year": heldout_year,
                "feature": feature,
                "training_rows": sample_count,
                "paired_n": pair_count,
                "spearman_rho": rho,
                "p_value": p_value,
                "min_paired_n": minimum_pairs
            })

    if not statistics:
        return (
            [],
            [],
            "no_features_with_sufficient_data"
        )

    # Multiple-testing correction.
    q_values = benjamini_hochberg(
        [item["p_value"] for item in statistics]
    )

    for item, q_value in zip(statistics, q_values):

        item["bh_q_value"] = float(q_value)

        item["passes_screen"] = (
            abs(item["spearman_rho"]) >= MIN_RHO
            and q_value <= MAX_Q
        )

    # Rank statistically relevant predictors.
    candidates = sorted(
        [
            item for item in statistics
            if item["passes_screen"]
        ],
        key=lambda item: (
            item["bh_q_value"],
            -abs(item["spearman_rho"]),
            item["feature"]
        )
    )

    selected = []

    # Multicollinearity filtering.
    for candidate in candidates:

        redundant = False

        for previous in selected:

            rho, _, _ = spearman_correlation(
                training_data[candidate["feature"]],
                training_data[previous["feature"]]
            )

            if (
                np.isfinite(rho)
                and abs(rho) > MAX_CORR
            ):
                redundant = True
                break

        candidate["rejected_for_multicollinearity"] = redundant
        candidate["dropped_for_max_features"] = False

        if not redundant:
            if len(selected) < MAX_FEATURES:
                selected.append(candidate)
            else:
                candidate["dropped_for_max_features"] = True

    status = (
        "selected"
        if selected
        else "no_predictors_passed_screen"
    )

    return selected, statistics, status



# ---------------------------------------------------------
# CROP-FEATURE REPORT (crop_feature_selection_v1.csv)
# ---------------------------------------------------------

def build_crop_feature_report(
    crops,
    feature_columns,
    scores_df,
    fold_df,
    consensus_df,
    scorable_years
):
    """
    One row per crop x weather feature, aggregated over the scorable
    LOYO folds only (held-out year in scorable_years), so that the
    selection counts match crop_feature_selection_consensus.csv.

    Statistics summarise per-fold training-set values. Features with no
    evaluable fold keep their row with empty statistics; missing is not
    zero.
    """

    if len(scores_df):
        scorable_scores = scores_df[
            scores_df["heldout_year"].isin(scorable_years)
        ]
    else:
        scorable_scores = scores_df

    empty_stats = {
        "mean_spearman_rho": np.nan,
        "min_spearman_rho": np.nan,
        "max_spearman_rho": np.nan,
        "mean_abs_spearman_rho": np.nan,
        "rho_sign_consistent": np.nan,
        "min_bh_q_value": np.nan,
        "mean_bh_q_value": np.nan,
        "max_bh_q_value": np.nan,
        "min_paired_n": np.nan,
        "max_paired_n": np.nan,
        "folds_passing_screen": 0,
        "folds_rejected_multicollinearity": 0,
        "folds_dropped_for_max_features": 0,
        "selected_in_heldout_years": "",
    }

    rows = []

    for crop in crops:

        n_folds = int(
            (
                fold_df["crop"].eq(crop)
                & fold_df["heldout_year"].isin(scorable_years)
            ).sum()
        )

        consensus_row = consensus_df[consensus_df["crop"].eq(crop)].iloc[0]

        for feature in feature_columns:

            window, indicator = feature.split("__", 1)

            if len(scorable_scores):
                subset = scorable_scores[
                    scorable_scores["crop"].eq(crop)
                    & scorable_scores["feature"].eq(feature)
                ]
            else:
                subset = scorable_scores

            n_evaluated = len(subset)
            n_selected = int(subset["selected"].sum()) if n_evaluated else 0

            if n_evaluated:
                rho = subset["spearman_rho"]
                signs = np.sign(rho)
                stats = {
                    "mean_spearman_rho": float(rho.mean()),
                    "min_spearman_rho": float(rho.min()),
                    "max_spearman_rho": float(rho.max()),
                    "mean_abs_spearman_rho": float(rho.abs().mean()),
                    "rho_sign_consistent": bool(
                        (signs > 0).all() or (signs < 0).all()
                    ),
                    "min_bh_q_value": float(subset["bh_q_value"].min()),
                    "mean_bh_q_value": float(subset["bh_q_value"].mean()),
                    "max_bh_q_value": float(subset["bh_q_value"].max()),
                    "min_paired_n": int(subset["paired_n"].min()),
                    "max_paired_n": int(subset["paired_n"].max()),
                    "folds_passing_screen": int(subset["passes_screen"].sum()),
                    "folds_rejected_multicollinearity": int(
                        subset["rejected_for_multicollinearity"].sum()
                    ),
                    "folds_dropped_for_max_features": int(
                        subset["dropped_for_max_features"].sum()
                    ),
                    "selected_in_heldout_years": ";".join(
                        str(int(y))
                        for y in sorted(
                            subset.loc[subset["selected"], "heldout_year"]
                        )
                    ),
                }
            else:
                stats = dict(empty_stats)

            rows.append({
                "crop": crop,
                "feature": feature,
                "season_window": window,
                "weather_indicator": indicator,
                "usable_yield_anomaly_rows": int(
                    consensus_row["usable_yield_anomaly_rows"]
                ),
                "district_count": int(consensus_row["district_count"]),
                "scorable_loyo_folds": n_folds,
                "folds_evaluated": n_evaluated,
                "folds_selected": n_selected,
                "selection_frequency_fraction": (
                    n_selected / n_folds if n_folds else np.nan
                ),
                "is_stable_predictor": n_selected >= MIN_STABLE,
                **stats,
                "crop_selection_status": consensus_row["selection_status"],
                "interpretation": "statistical_association_only",
            })

    report = pd.DataFrame(rows)

    # Integrity checks specific to this deliverable.
    if report.duplicated(["crop", "feature"]).any():
        raise ValueError(
            "crop_feature_selection_v1: duplicate crop-feature keys."
        )

    if len(report) != len(crops) * len(feature_columns):
        raise ValueError(
            "crop_feature_selection_v1: expected one row per crop-feature."
        )

    for row in consensus_df.itertuples():
        expected = {
            item.split(":")[0]
            for item in str(row.selection_frequency).split(";")
            if item and item != "nan"
        }
        actual = set(
            report.loc[
                report["crop"].eq(row.crop) & report["is_stable_predictor"],
                "feature"
            ]
        )
        # The consensus file lists at most MAX_FEATURES names but its
        # stable_predictor_count is untruncated; v1 must match both.
        if (
            len(actual) != int(row.stable_predictor_count)
            or not expected.issubset(actual)
        ):
            raise ValueError(
                "crop_feature_selection_v1: stable predictors differ from "
                f"consensus for {row.crop}."
            )

    return report


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def run_feature_selection(
    weather_path,
    yield_path,
    output_directory
):

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    weather = pd.read_csv(weather_path)
    yield_data = pd.read_csv(yield_path)

    validate_inputs(weather, yield_data)

    # Convert long-form weather anomalies to wide format.
    weather_wide = (
        weather.pivot(
            index=["district_code", "year"],
            columns="weather_column",
            values="anomaly_z"
        )
        .reset_index()
    )

    feature_columns = sorted(
        column
        for column in weather_wide.columns
        if column not in {"district_code", "year"}
    )

    if len(feature_columns) != 78:
        raise ValueError(
            f"Expected 78 weather features, found {len(feature_columns)}."
        )

    # Keep relevant yield metadata.
    yield_columns = [
        "district_code",
        "district_name",
        "year",
        "crop",
        "is_aggregate",
        "is_seasonal_component",
        "is_group_category",
        "modelling_role",
        "use_in_modelling",
        "eligible_for_baseline",
        "yield_anomaly_zscore",
        "anomaly_status"
    ]

    # Join weather and yield observations.
    merged = yield_data[yield_columns].merge(
        weather_wide,
        on=["district_code", "year"],
        how="left",
        validate="many_to_one"
    )

    eligible_data = filter_eligible_crops(merged)

    years = sorted(
        yield_data["year"].dropna().astype(int).unique()
    )

    scorable_years = sorted(
        eligible_data["year"].astype(int).unique()
    )

    crops = get_eligible_crop_list(merged)

    fold_results = []
    feature_scores = []
    validation_results = []

    # -----------------------------------------------------
    # CROP-SPECIFIC LOYO FEATURE SELECTION
    # -----------------------------------------------------

    for crop in crops:

        crop_data = eligible_data[
            eligible_data["crop"].eq(crop)
        ]

        for heldout_year in years:

            # Critical leakage control:
            # Remove the held-out year before feature selection.
            training_data = crop_data[
                crop_data["year"] != heldout_year
            ]

            heldout_data = crop_data[
                crop_data["year"] == heldout_year
            ]

            selected, statistics, status = select_features_for_fold(
                training_data=training_data,
                feature_columns=feature_columns,
                crop=crop,
                heldout_year=heldout_year
            )

            selected_names = [
                item["feature"]
                for item in selected
            ]

            # Measured, not asserted: any row shared by the training
            # frame passed to the scorer and the held-out frame.
            heldout_rows_in_training = bool(
                training_data.index.intersection(heldout_data.index).size
                or (training_data["year"] == heldout_year).any()
            )

            training_years = ";".join(
                map(
                    str,
                    sorted(
                        training_data["year"].astype(int).unique()
                    )
                )
            )

            fold_results.append({

                "crop": crop,
                "heldout_year": heldout_year,
                "training_years": training_years,
                "training_sample_count": len(training_data),
                "heldout_sample_count": len(heldout_data),
                "selection_status": status,
                "validation_status": (
                    "scorable"
                    if len(heldout_data)
                    else "no_eligible_heldout_yield_anomaly"
                ),
                "selected_feature_count": len(selected_names),
                "selected_features": ";".join(selected_names)

            })

            validation_results.append({

                "crop": crop,
                "heldout_year": heldout_year,
                "training_years": training_years,
                "training_sample_count": len(training_data),
                "heldout_sample_count": len(heldout_data),
                "heldout_label_available": bool(len(heldout_data)),
                "heldout_features_used_in_selection": heldout_rows_in_training,
                "validation_status": (
                    "scorable"
                    if len(heldout_data)
                    else "no_eligible_heldout_yield_anomaly"
                )

            })

            for item in statistics:

                item["selected"] = (
                    item["feature"] in selected_names
                )

                item["rejected_for_multicollinearity"] = (
                    item.get(
                        "rejected_for_multicollinearity",
                        False
                    )
                )

                item["dropped_for_max_features"] = (
                    item.get(
                        "dropped_for_max_features",
                        False
                    )
                )

                feature_scores.append(item)

    # -----------------------------------------------------
    # CROP-LEVEL CONSENSUS
    # -----------------------------------------------------

    fold_df = pd.DataFrame(fold_results)
    scores_df = pd.DataFrame(feature_scores)
    validation_df = pd.DataFrame(validation_results)

    consensus_results = []

    for crop in crops:

        crop_folds = fold_df[
            fold_df["crop"].eq(crop)
            & fold_df["heldout_year"].isin(scorable_years)
        ]

        selection_counts = {}

        for cell in crop_folds["selected_features"]:

            for feature in [
                value for value in str(cell).split(";")
                if value
            ]:

                selection_counts[feature] = (
                    selection_counts.get(feature, 0) + 1
                )

        stable_features = sorted(
            [
                (feature, count)
                for feature, count in selection_counts.items()
                if count >= MIN_STABLE
            ],
            key=lambda item: (-item[1], item[0])
        )

        crop_observations = eligible_data[
            eligible_data["crop"].eq(crop)
        ]

        if (
            len(crop_folds) == 0
            or len(crop_observations) < MIN_N
        ):

            status = "insufficient_evidence"

        elif stable_features:

            status = "stable_predictors_identified"

        else:

            status = "no_stable_predictors_across_folds"

        consensus_results.append({

            "crop": crop,
            "usable_yield_anomaly_rows": len(crop_observations),
            "district_count": crop_observations["district_code"].nunique(),
            "scorable_loyo_folds": len(crop_folds),
            "stable_predictor_count": len(stable_features),
            "selected_predictors": ";".join(
                feature
                for feature, count in stable_features[:MAX_FEATURES]
            ),
            "selection_frequency": ";".join(
                f"{feature}:{count}/{len(crop_folds)}"
                for feature, count in stable_features[:MAX_FEATURES]
            ),
            "selection_status": status

        })

    consensus_df = pd.DataFrame(consensus_results)

    report_df = build_crop_feature_report(
        crops=crops,
        feature_columns=feature_columns,
        scores_df=scores_df,
        fold_df=fold_df,
        consensus_df=consensus_df,
        scorable_years=scorable_years
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    checks = [

        (
            "weather_feature_count",
            len(feature_columns) == weather["weather_column"].nunique()
            and len(feature_columns) == 78,
            f"{len(feature_columns)} unique features; expected 78 "
            f"(source file has {weather['weather_column'].nunique()})"
        ),

        (
            "weather_key_uniqueness",
            not weather.duplicated(
                ["district_code", "year", "weather_column"]
            ).any(),
            "Unique district-year-feature keys"
        ),

        (
            "yield_key_uniqueness",
            not yield_data.duplicated(
                ["district_code", "year", "crop"]
            ).any(),
            "Unique district-year-crop keys"
        ),

        (
            "heldout_year_excluded_from_training",
            all(
                int(row.heldout_year)
                not in [
                    int(value)
                    for value in str(row.training_years).split(";")
                    if value
                ]
                for row in fold_df.itertuples()
            )
            and not validation_df[
                "heldout_features_used_in_selection"
            ].any(),
            "Held-out year excluded from every crop-fold training set"
        ),

        (
            "heldout_features_not_used",
            not validation_df[
                "heldout_features_used_in_selection"
            ].any(),
            "Held-out observations are not used for feature scoring"
        ),

        (
            "missing_values_not_imputed",
            int(weather_wide[feature_columns].notna().sum().sum())
            == int(weather["anomaly_z"].notna().sum())
            and int(merged["yield_anomaly_zscore"].notna().sum())
            == int(yield_data["yield_anomaly_zscore"].notna().sum()),
            "Non-missing counts unchanged by pivot/merge; "
            "pairwise-complete observations only; no imputation"
        ),

        (
            "selected_feature_limit",
            bool(
                (fold_df["selected_feature_count"] <= MAX_FEATURES).all()
            ),
            f"At most {MAX_FEATURES} features per crop-fold"
        ),

        (
            "scorable_years",
            len(scorable_years) > 0
            and set(scorable_years).issubset(
                set(weather["year"].astype(int))
            )
            and scorable_years == sorted(
                eligible_data["year"].astype(int).unique()
            ),
            f"Usable target years derived from inputs: {scorable_years}"
        )

    ]

    checks_df = pd.DataFrame(
        checks,
        columns=["check", "passed", "detail"]
    )

    checks_df["status"] = np.where(
        checks_df["passed"],
        "PASS",
        "FAIL"
    )

    checks_df = checks_df[
        ["check", "status", "detail"]
    ]

    if (checks_df["status"] != "PASS").any():

        raise RuntimeError(
            "Validation failed:\n"
            + checks_df.to_string(index=False)
        )

    # -----------------------------------------------------
    # EXPORT OUTPUTS
    # -----------------------------------------------------

    fold_df.to_csv(
        output_directory / "crop_feature_selection_by_fold.csv",
        index=False
    )

    scores_df.to_csv(
        output_directory / "crop_feature_scores.csv",
        index=False
    )

    consensus_df.to_csv(
        output_directory / "crop_feature_selection_consensus.csv",
        index=False
    )

    report_df.to_csv(
        output_directory / "crop_feature_selection_v1.csv",
        index=False
    )

    validation_df.to_csv(
        output_directory / "crop_feature_selection_validation.csv",
        index=False
    )

    checks_df.to_csv(
        output_directory / "validation_checks.csv",
        index=False
    )

    print("\nISSUE 5 FEATURE SELECTION COMPLETED")
    print("-----------------------------------")

    print(f"Eligible observations: {len(eligible_data)}")
    print(f"Eligible crops: {len(crops)}")
    print(f"Weather features: {len(feature_columns)}")
    print(f"Scorable years: {scorable_years}")

    print("\nCROP-LEVEL CONSENSUS")
    print(consensus_df.to_string(index=False))

    print(f"\ncrop_feature_selection_v1.csv rows: {len(report_df)}")

    print("\nVALIDATION CHECKS")
    print(checks_df.to_string(index=False))

    print(
        f"\nOutputs saved to: {output_directory.resolve()}"
    )


# ---------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Issue 5: Crop-specific weather feature selection"
    )

    parser.add_argument(
        "--weather",
        required=True,
        help="Path to forward-chaining weather anomaly CSV"
    )

    parser.add_argument(
        "--yield-anomalies",
        required=True,
        help="Path to forward-chaining yield anomaly CSV"
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory for generated outputs"
    )

    arguments = parser.parse_args()

    run_feature_selection(
        weather_path=arguments.weather,
        yield_path=arguments.yield_anomalies,
        output_directory=arguments.outdir
    )
