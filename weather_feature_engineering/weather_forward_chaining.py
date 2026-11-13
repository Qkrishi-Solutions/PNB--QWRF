
#!/usr/bin/env python3
"""Issue 3: Prospective forward-chaining weather anomaly generation."""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

ID_COLS = ["district_code", "district_name", "year"]
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
        if "__" in col:
            window, indicator = col.split("__", 1)
            if window and indicator:
                parsed.append((col, window, indicator))
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    validation_dir = os.path.join(args.outdir, "validation_tables")
    os.makedirs(validation_dir, exist_ok=True)

    source_hash = sha256_file(args.input)
    source = pd.read_csv(args.input)

    missing = sorted(set(ID_COLS) - set(source.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    weather_specs = parse_weather_columns(source.columns)
    if not weather_specs:
        raise ValueError("No weather columns matching <window>__<indicator> found.")

    weather_cols = [item[0] for item in weather_specs]
    unique_keys = ["district_code", "year"]

    checks = []

    def add_check(name, passed, detail):
        checks.append({
            "check": name,
            "result": "PASS" if passed else "FAIL",
            "detail": str(detail),
        })

    source_key_cols = (
        ["district_code", "year", "crop"]
        if "crop" in source.columns else unique_keys
    )
    duplicate_source_keys = int(source.duplicated(source_key_cols).sum())
    add_check(
        "Crop-long source key uniqueness",
        duplicate_source_keys == 0,
        f"{duplicate_source_keys} duplicate source keys",
    )

    # Verify repeated district-year weather values before deduplication.
    nunique = source.groupby(unique_keys, dropna=False)[weather_cols].nunique(
        dropna=False
    )
    inconsistent = int((nunique > 1).sum().sum())

    add_check(
        "Weather values consistent across crop rows",
        inconsistent == 0,
        f"{inconsistent} inconsistent district-year/weather-column combinations",
    )

    if inconsistent:
        raise ValueError("Inconsistent repeated weather values found.")

    weather = (
        source[ID_COLS + weather_cols]
        .sort_values(unique_keys)
        .drop_duplicates(unique_keys, keep="first")
        .reset_index(drop=True)
    )

    add_check(
        "One weather record per district-year",
        not weather.duplicated(unique_keys).any(),
        f"{len(weather)} district-year rows",
    )

    years = sorted(int(y) for y in weather["year"].dropna().unique())
    if not years:
        raise ValueError("No valid years found.")

    # Convert weather columns into a long-format observation table.
    long_parts = []

    for col, window, indicator in weather_specs:
        part = weather[ID_COLS].copy()
        part["seasonal_window"] = window
        part["indicator"] = indicator
        part["weather_column"] = col
        part["actual_value"] = pd.to_numeric(
            weather[col], errors="coerce"
        ).to_numpy()
        long_parts.append(part)

    observations = pd.concat(long_parts, ignore_index=True)

    group_cols = [
        "district_code",
        "district_name",
        "seasonal_window",
        "indicator",
        "weather_column",
    ]

    normals_rows = []
    anomaly_rows = []

    # Forward chaining: train only on years strictly before the target year.
    for target_year in years:
        previous_years = [y for y in years if y < target_year]

        target = observations[
            observations["year"] == target_year
        ].copy()

        training = observations[
            observations["year"] < target_year
        ].copy()

        if len(previous_years) > 0:
            stats = (
                training.groupby(group_cols, dropna=False)["actual_value"]
                .agg(
                    candidate_mean="mean",
                    candidate_std="std",
                    n_training_valid="count",
                    training_min="min",
                    training_max="max",
                )
                .reset_index()
            )
        else:
            stats = target[group_cols].drop_duplicates().copy()
            stats["candidate_mean"] = np.nan
            stats["candidate_std"] = np.nan
            stats["n_training_valid"] = 0
            stats["training_min"] = np.nan
            stats["training_max"] = np.nan

        stats["target_year"] = target_year
        stats["training_years"] = ";".join(map(str, previous_years))

        stats["normal_status"] = np.where(
            stats["n_training_valid"] >= MIN_TRAINING_OBSERVATIONS,
            "available",
            "insufficient_training_history",
        )

        available = stats["normal_status"] == "available"

        # A baseline is usable only with at least two valid observations.
        stats["normal_mean"] = stats["candidate_mean"].where(available)
        stats["normal_std"] = stats["candidate_std"].where(available)

        stats = stats.drop(
            columns=["candidate_mean", "candidate_std"]
       )

        normals_rows.append(stats)

        # Generate anomalies for the target year only.
        fold = target.merge(
            stats,
            on=group_cols,
            how="left",
            validate="one_to_one",
        )

        fold["target_year"] = target_year
        fold["is_target_year"] = fold["year"] == target_year

        fold["anomaly"] = (
            fold["actual_value"] - fold["normal_mean"]
        )

        fold["anomaly_pct"] = np.where(
            fold["normal_mean"].notna()
            & (fold["normal_mean"] != 0),
            fold["anomaly"] / fold["normal_mean"] * 100.0,
            np.nan,
        )

        fold["anomaly_z"] = np.where(
            fold["normal_std"].notna()
            & (fold["normal_std"] > 0),
            fold["anomaly"] / fold["normal_std"],
            np.nan,
        )

        fold["actual_missing"] = fold["actual_value"].isna()
        fold["normal_missing"] = fold["normal_mean"].isna()

        fold["normal_std_zero_or_missing"] = (
            fold["normal_std"].isna()
            | (fold["normal_std"] == 0)
        )

        anomaly_rows.append(fold)

    normals = pd.concat(normals_rows, ignore_index=True)
    anomalies = pd.concat(anomaly_rows, ignore_index=True)

    # ---------------------------------------------------------
    # Independent validation
    # ---------------------------------------------------------

    expected_rows = len(weather) * len(weather_specs)
    expected_missing = int(observations["actual_value"].isna().sum())

    unique_keys_valid = not anomalies.duplicated(
        ["district_code", "year", "weather_column"]
    ).any()

    add_check(
        "Expected anomaly row count and unique keys",
        len(anomalies) == expected_rows and unique_keys_valid,
        f"{len(anomalies)} rows; expected {expected_rows}",
    )

    # Confirm every target uses strictly earlier calendar years.
    chronology_valid = True

    for _, row in normals[
        ["target_year", "training_years"]
    ].drop_duplicates().iterrows():

        listed_years = [
            int(y) for y in row["training_years"].split(";") if y
        ]

        expected_years = [
            y for y in years if y < int(row["target_year"])
        ]

        if listed_years != expected_years:
            chronology_valid = False
            break

        if any(y >= int(row["target_year"]) for y in listed_years):
            chronology_valid = False
            break

    add_check(
        "Only strictly earlier years enter training",
        chronology_valid,
        "Training years verified for every target year.",
    )

    # Independently recompute every normal.
    observation_groups = {
        key: group
        for key, group in observations.groupby(group_cols, dropna=False)
    }

    independent_valid = True

    for _, normal_row in normals.iterrows():
        key = tuple(normal_row[col] for col in group_cols)
        group = observation_groups[key]

        values = pd.to_numeric(
            group.loc[
                group["year"] < normal_row["target_year"],
                "actual_value",
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        count = len(values)

        expected_mean = (
            float(np.mean(values)) if count >= 2 else np.nan
        )

        expected_std = (
            float(np.std(values, ddof=1)) if count >= 2 else np.nan
        )

        mean_ok = np.isclose(
            normal_row["normal_mean"],
            expected_mean,
            rtol=1e-9,
            atol=1e-10,
            equal_nan=True,
        )

        std_ok = np.isclose(
            normal_row["normal_std"],
            expected_std,
            rtol=1e-9,
            atol=1e-10,
            equal_nan=True,
        )

        count_ok = int(normal_row["n_training_valid"]) == count

        if not (mean_ok and std_ok and count_ok):
            independent_valid = False
            break

    add_check(
        "Independent recomputation of normals",
        independent_valid,
        "Normal means, sample standard deviations and counts independently verified.",
    )

    # Insufficient history must not produce usable normals.
    insufficient = normals["n_training_valid"] < MIN_TRAINING_OBSERVATIONS

    insufficient_valid = (
        normals.loc[insufficient, "normal_mean"].isna().all()
        and normals.loc[insufficient, "normal_std"].isna().all()
    )

    add_check(
        "Insufficient history produces unavailable normals",
        insufficient_valid,
        f"{int(insufficient.sum())} normal rows have insufficient history.",
    )

    # Check anomaly formula independently.
    expected_anomaly = (
        anomalies["actual_value"] - anomalies["normal_mean"]
    )

    formula_valid = np.allclose(
        anomalies["anomaly"].to_numpy(dtype=float),
        expected_anomaly.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True,
    )

    add_check(
        "Anomaly formula",
        formula_valid,
        "Anomaly equals actual value minus training normal mean.",
    )

    # Missing source values must remain missing.
    missing_valid = (
        int(anomalies["actual_missing"].sum()) == expected_missing
    )

    add_check(
        "Missing weather observations remain missing",
        missing_valid,
        f"{int(anomalies['actual_missing'].sum())} missing values; expected {expected_missing}",
    )

    # Only target-year observations should be in the anomaly output.
    target_only_valid = (
        anomalies["is_target_year"].all()
        and (anomalies["year"] == anomalies["target_year"]).all()
    )

    add_check(
        "Anomaly output contains target-year observations only",
        target_only_valid,
        "Each anomaly is generated for its corresponding target year.",
    )

    # Zero or unavailable standard deviation cannot produce z-scores.
    invalid_std = (
        anomalies["normal_std"].isna()
        | (anomalies["normal_std"] == 0)
    )

    zscore_valid = anomalies.loc[
        invalid_std, "anomaly_z"
    ].isna().all()

    add_check(
        "Zero or unavailable standard deviation produces no z-score",
        zscore_valid,
        "Z-scores remain missing when standard deviation is zero or unavailable.",
    )

    source_unchanged = source_hash == sha256_file(args.input)

    add_check(
        "Input source remains unchanged",
        source_unchanged,
        "Source SHA-256 hash is unchanged.",
    )

    # ---------------------------------------------------------
    # Save separate forward-chaining outputs.
    # Existing LOYO outputs are not overwritten.
    # ---------------------------------------------------------

    normals.to_csv(
        os.path.join(
            args.outdir,
            "district_weather_normals_forward_chaining_v1.csv",
        ),
        index=False,
    )

    anomalies.to_csv(
        os.path.join(
            args.outdir,
            "weather_anomalies_forward_chaining_v1.csv",
        ),
        index=False,
    )

    checks_df = pd.DataFrame(checks)

    checks_df.to_csv(
        os.path.join(
            validation_dir,
            "forward_chaining_validation_checks.csv",
        ),
        index=False,
    )

    report_path = os.path.join(
        args.outdir,
        "forward_chaining_validation_report.txt",
    )

    with open(report_path, "w", encoding="utf-8") as report:
        report.write(
            "Issue 3 - Forward-Chaining Weather Anomaly Validation\n\n"
        )
        report.write(
            "Method: Each target year uses only strictly earlier years.\n"
        )
        report.write(f"Years: {years}\n")
        report.write(
            f"Minimum training observations: {MIN_TRAINING_OBSERVATIONS}\n"
        )
        report.write(f"District-year rows: {len(weather)}\n")
        report.write(f"Weather features: {len(weather_specs)}\n")
        report.write(f"Normal rows: {len(normals)}\n")
        report.write(f"Anomaly rows: {len(anomalies)}\n")
        report.write(
            f"Insufficient-history normal rows: {int(insufficient.sum())}\n\n"
        )
        report.write(checks_df.to_string(index=False))
        report.write("\n")

    print(checks_df.to_string(index=False))
    print(f"\nDistrict-year rows: {len(weather)}")
    print(f"Weather features: {len(weather_specs)}")
    print(f"Forward normals: {len(normals)}")
    print(f"Forward anomalies: {len(anomalies)}")
    print(f"Insufficient-history normals: {int(insufficient.sum())}")

    if not checks_df["result"].eq("PASS").all():
        raise AssertionError(
            "One or more validation checks failed. "
            "Review forward_chaining_validation_checks.csv."
        )


if __name__ == "__main__":
    main()
