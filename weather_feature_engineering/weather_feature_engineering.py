#!/usr/bin/env python3
"""
QWeatherRiskF - Issue #3: Weather Feature Engineering & Anomaly Generation

Input:
    crop_long_v1.csv (or another crop-long CSV with district/year and weather
    columns named <window>__<indicator>)

Outputs:
    district_weather_normals_v1.csv
    weather_anomalies_v1.csv
    weather_stress_features_v1.csv
    validation_tables/validation_checks.csv
    validation_tables/weather_missingness.csv
    validation_tables/weather_feature_inventory.csv

Method:
    * Deduplicate crop-long data to one record per district-year after verifying
      that repeated weather values are identical across crop rows.
    * Calculate a separate leave-one-year-out (LOYO) normal for each held-out
      year, district, seasonal window, and weather indicator.
    * The held-out year's value is excluded from its own normal.
    * Preserve missing observations; never impute missing weather values.
    * Stress thresholds are transparent, configurable screening heuristics,
      not official agronomic thresholds.

Example:
    python weather_feature_engineering.py --input crop_long_v1.csv --outdir .
"""

import argparse
import os
import re
import hashlib
import numpy as np
import pandas as pd

ID_COLS = ["district_code", "district_name", "year"]
RAIN_DEFICIT_PCT = -20.0
RAIN_EXCESS_PCT = 20.0
EXTREME_RAIN_Z = 2.0
DRY_SPELL_Z = 1.5
HEAT_STRESS_Z = 1.5
MIN_TRAINING_OBSERVATIONS = 2


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_weather_columns(columns):
    parsed = []
    for col in columns:
        if "__" not in col:
            continue
        window, indicator = col.split("__", 1)
        if window and indicator:
            parsed.append((col, window, indicator))
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input crop-long CSV")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    validation_dir = os.path.join(args.outdir, "validation_tables")
    os.makedirs(validation_dir, exist_ok=True)

    source_hash_before = sha256_file(args.input)
    source = pd.read_csv(args.input)

    required = set(ID_COLS)
    missing_required = sorted(required - set(source.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    weather_specs = parse_weather_columns(source.columns)
    if not weather_specs:
        raise ValueError("No weather columns matching <window>__<indicator> found.")

    weather_cols = [x[0] for x in weather_specs]
    unique_keys = ["district_code", "year"]

    checks = []
    def add_check(name, passed, detail):
        checks.append({
            "check": name,
            "result": "PASS" if passed else "FAIL",
            "detail": str(detail)
        })

    duplicate_source_keys = int(source.duplicated(
        ["district_code", "year", "crop"] if "crop" in source.columns else unique_keys
    ).sum())
    add_check(
        "Crop-long source key uniqueness",
        duplicate_source_keys == 0,
        f"{duplicate_source_keys} duplicate source keys"
    )

    # Ensure weather observations repeated across crop rows are consistent.
    nunique = source.groupby(unique_keys, dropna=False)[weather_cols].nunique(dropna=False)
    inconsistent_cells = int((nunique > 1).sum().sum())
    add_check(
        "Weather values consistent across crop rows",
        inconsistent_cells == 0,
        f"{inconsistent_cells} district-year/weather-column combinations differ"
    )
    if inconsistent_cells:
        raise ValueError(
            "Repeated weather values are inconsistent within at least one district-year. "
            "Resolve this before calculating normals."
        )

    # Retain one weather record per district-year. Keep district_name from first row.
    weather = (
        source[ID_COLS + weather_cols]
        .sort_values(unique_keys)
        .drop_duplicates(unique_keys, keep="first")
        .reset_index(drop=True)
    )

    duplicate_weather_keys = int(weather.duplicated(unique_keys).sum())
    add_check(
        "One weather record per district-year",
        duplicate_weather_keys == 0,
        f"{len(weather)} unique district-year rows"
    )

    years = sorted(int(y) for y in weather["year"].dropna().unique())
    districts = weather[["district_code", "district_name"]].drop_duplicates()

    # Convert the 78 weather columns into a long observation table.
    long_parts = []
    for col, window, indicator in weather_specs:
        part = weather[ID_COLS].copy()
        part["seasonal_window"] = window
        part["indicator"] = indicator
        part["weather_column"] = col
        part["actual_value"] = pd.to_numeric(weather[col], errors="coerce").values
        long_parts.append(part)
    observations = pd.concat(long_parts, ignore_index=True)

    normals_rows = []
    anomaly_rows = []

    # LOYO: each held-out year receives a normal calculated only from other years.
    for year in years:
        train = observations[observations["year"] != year]

        group_cols = ["district_code", "district_name", "seasonal_window", "indicator", "weather_column"]
        stats = (
            train.groupby(group_cols, dropna=False)["actual_value"]
            .agg(
                normal_mean="mean",
                normal_std="std",
                n_training_valid="count",
                training_min="min",
                training_max="max"
            )
            .reset_index()
        )
        stats["held_out_year"] = year
        stats["training_years"] = ";".join(str(y) for y in years if y != year)
        stats["normal_status"] = np.where(
            stats["n_training_valid"] >= MIN_TRAINING_OBSERVATIONS,
            "available",
            "insufficient_training_history"
        )
        normals_rows.append(stats)

        # Apply this fold's training-only normal to BOTH training and validation
        # observations. Thus no held-out-year weather enters any training feature.
        fold = observations.merge(stats, on=group_cols, how="left", validate="many_to_one")
        fold["validation_fold_year"] = year
        fold["is_validation_year"] = fold["year"] == year
        fold["anomaly"] = fold["actual_value"] - fold["normal_mean"]
        fold["anomaly_pct"] = np.where(
            fold["normal_mean"].notna() & (fold["normal_mean"] != 0),
            (fold["anomaly"] / fold["normal_mean"]) * 100.0,
            np.nan
        )
        fold["anomaly_z"] = np.where(
            fold["normal_std"].notna() & (fold["normal_std"] > 0),
            fold["anomaly"] / fold["normal_std"],
            np.nan
        )
        fold["actual_missing"] = fold["actual_value"].isna()
        fold["normal_missing"] = fold["normal_mean"].isna()
        fold["normal_std_zero_or_missing"] = (
            fold["normal_std"].isna() | (fold["normal_std"] == 0)
        )
        anomaly_rows.append(fold)

    normals = pd.concat(normals_rows, ignore_index=True)
    anomalies = pd.concat(anomaly_rows, ignore_index=True)

    # Stress feature table: one row per district-year per validation fold.
    stress_parts = [
        weather[ID_COLS].assign(validation_fold_year=fold_year)
        for fold_year in years
    ]
    stress = pd.concat(stress_parts, ignore_index=True)
    stress["is_validation_year"] = stress["year"] == stress["validation_fold_year"]
    anomaly_lookup = anomalies.set_index(
        ["district_code", "year", "weather_column", "validation_fold_year"]
    )

    def get_series(column, field):
        vals = []
        for _, row in stress.iterrows():
            key = (
                row["district_code"], int(row["year"]), column,
                int(row["validation_fold_year"])
            )
            try:
                vals.append(anomaly_lookup.loc[key, field])
            except KeyError:
                vals.append(np.nan)
        return pd.Series(vals, index=stress.index, dtype="float64")

    # Preserve the original weather context as well as derived indicators.
    stress_columns = {}
    for col in weather_cols:
        stress_columns[col] = np.tile(pd.to_numeric(weather[col], errors="coerce").values, len(years))
        stress_columns[f"{col}__anomaly"] = get_series(col, "anomaly").values
        stress_columns[f"{col}__anomaly_pct"] = get_series(col, "anomaly_pct").values
        stress_columns[f"{col}__anomaly_z"] = get_series(col, "anomaly_z").values
    stress = pd.concat([stress, pd.DataFrame(stress_columns)], axis=1)

    # Explicit, provisional stress flags based on fold-specific training normals.
    stress_rule_rows = []
    for window in sorted({w for _, w, _ in weather_specs}):
        rain = f"{window}__rain_sum"
        max5 = f"{window}__max_5day_rain"
        dry = f"{window}__longest_dry"
        tmax = f"{window}__tmax_mean"
        hot = f"{window}__hot_days"

        if rain in weather_cols:
            pct_col = f"{rain}__anomaly_pct"
            stress[f"{window}__rainfall_deficit_flag"] = (
                stress[pct_col] <= RAIN_DEFICIT_PCT
            ).where(stress[pct_col].notna(), pd.NA).astype("boolean")
            stress[f"{window}__rainfall_excess_flag"] = (
                stress[pct_col] >= RAIN_EXCESS_PCT
            ).where(stress[pct_col].notna(), pd.NA).astype("boolean")
            stress_rule_rows.append({
                "seasonal_window": window, "feature": "rainfall_deficit",
                "rule": f"rain_sum percentage anomaly <= {RAIN_DEFICIT_PCT}%",
                "interpretation": "Provisional screening flag; not an official crop-specific threshold."
            })
            stress_rule_rows.append({
                "seasonal_window": window, "feature": "rainfall_excess",
                "rule": f"rain_sum percentage anomaly >= {RAIN_EXCESS_PCT}%",
                "interpretation": "Provisional screening flag; not an official crop-specific threshold."
            })

        if max5 in weather_cols:
            zcol = f"{max5}__anomaly_z"
            stress[f"{window}__extreme_rainfall_flag"] = (
                stress[zcol] >= EXTREME_RAIN_Z
            ).where(stress[zcol].notna(), pd.NA).astype("boolean")
            stress_rule_rows.append({
                "seasonal_window": window, "feature": "extreme_rainfall",
                "rule": f"max_5day_rain anomaly z-score >= {EXTREME_RAIN_Z}",
                "interpretation": "Statistical screening proxy; does not establish an official extreme-rain event."
            })

        if dry in weather_cols:
            zcol = f"{dry}__anomaly_z"
            stress[f"{window}__dry_spell_stress_flag"] = (
                stress[zcol] >= DRY_SPELL_Z
            ).where(stress[zcol].notna(), pd.NA).astype("boolean")
            stress_rule_rows.append({
                "seasonal_window": window, "feature": "dry_spell_stress",
                "rule": f"longest_dry anomaly z-score >= {DRY_SPELL_Z}",
                "interpretation": "Higher-than-normal longest dry spell; provisional statistical proxy."
            })

        if tmax in weather_cols or hot in weather_cols:
            candidates = []
            if tmax in weather_cols:
                candidates.append(stress[f"{tmax}__anomaly_z"])
            if hot in weather_cols:
                candidates.append(stress[f"{hot}__anomaly_z"])
            combined = pd.concat(candidates, axis=1)
            stress[f"{window}__heat_stress_flag"] = (
                combined.ge(HEAT_STRESS_Z).any(axis=1)
            ).where(combined.notna().any(axis=1), pd.NA).astype("boolean")
            stress_rule_rows.append({
                "seasonal_window": window, "feature": "heat_stress",
                "rule": f"tmax_mean or hot_days anomaly z-score >= {HEAT_STRESS_Z}",
                "interpretation": "Provisional statistical proxy; crop-stage-specific thresholds require agronomic review."
            })

    # Missingness inventory from unique district-year observations.
    missingness = (
        observations.groupby(["seasonal_window", "indicator", "weather_column"])
        .agg(
            district_year_rows=("actual_value", "size"),
            valid_observations=("actual_value", "count"),
            missing_observations=("actual_value", lambda s: int(s.isna().sum()))
        )
        .reset_index()
    )
    missingness["missing_pct"] = (
        missingness["missing_observations"] / missingness["district_year_rows"] * 100
    )

    feature_inventory = pd.DataFrame(
        weather_specs, columns=["weather_column", "seasonal_window", "indicator"]
    )
    feature_inventory["source_units"] = "Not encoded in column name; verify source metadata before interpreting units."

    # Structural and leakage validation.
    # Independently verify that each fold excludes its held-out year.
    expected_training_years = {
        year: sorted(y for y in years if y != year)
        for year in years
    }

    actual_training_years = (
        normals.groupby("held_out_year")["training_years"]
        .unique()
    )

    exclusion_valid = True

    for held_out_year in years:
        expected = ";".join(
            str(y) for y in expected_training_years[held_out_year]
        )

        actual_values = actual_training_years.get(held_out_year, [])

        if len(actual_values) != 1 or actual_values[0] != expected:
            exclusion_valid = False
            break

    add_check(
        "No held-out year enters its fold's training normals",
        exclusion_valid,
        "Verified training_years for every held-out-year fold."
    )

    
    # Independently recompute every fold-specific normal from raw observations.
    # This checks the actual mean, standard deviation and valid observation count.
    independent_normal_check = True

    observation_groups = {
        key: group
        for key, group in observations.groupby(
            group_cols, dropna=False
        )
    }

    for _, normal_row in normals.iterrows():
        key = tuple(normal_row[col] for col in group_cols)

        group = observation_groups[key]

        training_values = pd.to_numeric(
            group.loc[
                group["year"] != normal_row["held_out_year"],
                "actual_value"
            ],
            errors="coerce"
        ).dropna().to_numpy(dtype=float)

        expected_count = len(training_values)

        expected_mean = (
            float(np.mean(training_values))
            if expected_count > 0 else np.nan
        )

        expected_std = (
            float(np.std(training_values, ddof=1))
            if expected_count >= 2 else np.nan
        )

        mean_matches = np.isclose(
            normal_row["normal_mean"],
            expected_mean,
            rtol=1e-9,
            atol=1e-10,
            equal_nan=True
        )

        std_matches = np.isclose(
            normal_row["normal_std"],
            expected_std,
            rtol=1e-9,
            atol=1e-10,
            equal_nan=True
        )

        count_matches = (
            int(normal_row["n_training_valid"]) == expected_count
        )

        if not (mean_matches and std_matches and count_matches):
            independent_normal_check = False
            break

    add_check(
        "Independent recomputation of fold normals",
        independent_normal_check,
        "Independently verified normal means, sample standard deviations, "
        "and valid training counts for all normal rows."
    )


    normals.to_csv(os.path.join(args.outdir, "district_weather_normals_v1.csv"), index=False)
        # Additional structural and data-quality checks.

    expected_normals = len(districts) * len(weather_specs) * len(years)
    expected_anomalies = len(weather) * len(weather_specs) * len(years)
    expected_stress = len(weather) * len(years)

    # 1. Check expected number of district weather normals.
    add_check(
        "Expected number of weather normal records",
        len(normals) == expected_normals,
        f"{len(normals)} rows; expected {expected_normals}"
    )

    # 2. Check expected number of anomaly records.
    add_check(
        "Expected number of weather anomaly records",
        len(anomalies) == expected_anomalies,
        f"{len(anomalies)} rows; expected {expected_anomalies}"
    )

    # 3. Check expected number of stress records.
    add_check(
        "Expected number of weather stress records",
        len(stress) == expected_stress,
        f"{len(stress)} rows; expected {expected_stress}"
    )

    # 4. Check missing weather values are preserved.
    expected_missing = int(observations["actual_value"].isna().sum()) * len(years)
    actual_missing = int(anomalies["actual_missing"].sum())

    add_check(
        "Missing weather values remain missing",
        actual_missing == expected_missing,
        f"{actual_missing} missing values; expected {expected_missing}"
    )

    # 5. Check expected weather feature count.
    add_check(
        "Weather feature count is 78",
        len(weather_specs) == 78,
        f"{len(weather_specs)} features; expected 78"
    )

    # 6. Check that normals remain constant within each fold.
    normal_consistency = (
        anomalies.groupby(
            ["validation_fold_year", "district_code",
             "seasonal_window", "indicator"]
        )["normal_mean"].nunique(dropna=False) <= 1
    ).all()

    add_check(
        "Fold-specific normals remain consistent",
        normal_consistency,
        "Each district-feature group uses one normal throughout each fold."
    )
    anomalies.to_csv(os.path.join(args.outdir, "weather_anomalies_v1.csv"), index=False)
    stress.to_csv(os.path.join(args.outdir, "weather_stress_features_v1.csv"), index=False)
    pd.DataFrame(checks).to_csv(
        os.path.join(validation_dir, "validation_checks.csv"), index=False
    )
    missingness.to_csv(
        os.path.join(validation_dir, "weather_missingness.csv"), index=False
    )
    feature_inventory.to_csv(
        os.path.join(validation_dir, "weather_feature_inventory.csv"), index=False
    )
    pd.DataFrame(stress_rule_rows).to_csv(
        os.path.join(validation_dir, "stress_rule_definitions.csv"), index=False
    )

    if not all(c["result"] == "PASS" for c in checks):
        raise AssertionError("One or more structural validation checks failed.")

    print(pd.DataFrame(checks).to_string(index=False))
    print(f"District-year rows: {len(weather)}")
    print(f"Weather features: {len(weather_specs)}")
    print(f"Normals rows: {len(normals)}")
    print(f"Anomaly rows: {len(anomalies)}")
    print(f"Stress rows: {len(stress)}")


if __name__ == "__main__":
    main()
