#!/usr/bin/env python3
"""QWeatherRiskF Issue #4: district-crop LOYO and forward yield baselines/anomalies.

The script preserves source rows and documents exclusions; it does not
repair inconsistent source yield, area, or production values.
"""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["district_code", "crop", "year"]
REQUIRED = [
    "district_code", "district_name", "year", "crop", "crop_name",
    "yield_kg_per_ha", "modelling_role", "flag_nan_filled_with_zero",
    "area_1000_ha", "production_1000_tons",
]
FLAGS = [
    "flag_nan_filled_with_zero", "flag_yield_mismatch", "flag_yield_jump",
    "flag_estimated_by_allocation", "flag_negative_value",
]
MIN_TRAINING_OBSERVATIONS = 2
LOW_Z_THRESHOLD = -2.0
HIGH_Z_THRESHOLD = 2.0


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
        "": False, "nan": False, "none": False,
    }
    return series.astype(str).str.strip().str.lower().map(values).fillna(False).astype(bool)


def add_check(checks, name, passed, detail):
    checks.append({
        "check": name,
        "result": "PASS" if bool(passed) else "FAIL",
        "detail": str(detail),
    })


def load_and_prepare(path):
    source = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(source.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = source.copy()
    for col in FLAGS:
        if col in data.columns:
            data[col] = as_bool(data[col])
        else:
            data[col] = False

    for col in ["yield_kg_per_ha", "area_1000_ha", "production_1000_tons"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["year"] = pd.to_numeric(data["year"], errors="raise").astype(int)

    if data[KEYS].isna().any().any():
        raise ValueError("District, crop, and year keys cannot be missing.")
    duplicates = int(data.duplicated(KEYS).sum())
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate district-crop-year rows.")

    role_ok = data["modelling_role"].astype(str).str.strip().eq("include")
    yield_ok = (
        data["yield_kg_per_ha"].notna()
        & np.isfinite(data["yield_kg_per_ha"].fillna(np.nan))
        & data["yield_kg_per_ha"].gt(0)
    )
    not_filled = ~data["flag_nan_filled_with_zero"]
    positive_area_zero_production = (
        data["area_1000_ha"].gt(0) & data["production_1000_tons"].eq(0)
    ).fillna(False)

    reasons = []
    for idx, row in data.iterrows():
        why = []
        if not role_ok.loc[idx]:
            why.append("excluded_modelling_role")
        y = row["yield_kg_per_ha"]
        if pd.isna(y):
            why.append("missing_yield")
        elif not np.isfinite(y):
            why.append("non_finite_yield")
        elif y <= 0:
            why.append("zero_or_negative_yield")
        if not not_filled.loc[idx]:
            why.append("filled_zero_flag")
        if positive_area_zero_production.loc[idx]:
            why.append("positive_area_zero_production")
        reasons.append(";".join(why) if why else "eligible")

    data["eligibility_reason"] = reasons
    data["eligible_for_baseline"] = (
        role_ok & yield_ok & not_filled & ~positive_area_zero_production
    )
    return source, data


def build_normals(data, years):
    eligible = data[data["eligible_for_baseline"]]
    groups = data[
        ["district_code", "district_name", "crop", "crop_name"]
    ].drop_duplicates(["district_code", "crop"])
    records = []

    for g in groups.itertuples(index=False):
        group_rows = eligible[
            (eligible["district_code"] == g.district_code)
            & (eligible["crop"] == g.crop)
        ]
        for heldout in years:
            training = group_rows[group_rows["year"] != heldout]
            vals = training["yield_kg_per_ha"].dropna().astype(float)
            n = int(vals.count())
            records.append({
                "district_code": g.district_code,
                "district_name": g.district_name,
                "crop": g.crop,
                "crop_name": g.crop_name,
                "validation_year": int(heldout),
                "baseline_mean_kg_per_ha": float(vals.mean()) if n else np.nan,
                "baseline_std_kg_per_ha": (
                    float(vals.std(ddof=1)) if n >= MIN_TRAINING_OBSERVATIONS else np.nan
                ),
                "training_observation_count": n,
                "training_years": ",".join(
                    map(str, sorted(training["year"].astype(int).unique()))
                ),
                "baseline_status": (
                    "ready" if n >= MIN_TRAINING_OBSERVATIONS
                    else "insufficient_history"
                ),
                "minimum_training_observations": MIN_TRAINING_OBSERVATIONS,
            })
    return pd.DataFrame(records)


def build_forward_normals(data, years):
    """Use only eligible observations from years strictly before each target year."""
    eligible = data[data["eligible_for_baseline"]]
    groups = data[["district_code", "district_name", "crop", "crop_name"]].drop_duplicates(["district_code", "crop"])
    records = []
    for g in groups.itertuples(index=False):
        rows = eligible[(eligible["district_code"] == g.district_code) & (eligible["crop"] == g.crop)]
        for target in years:
            training = rows[rows["year"] < target]
            vals = training["yield_kg_per_ha"].dropna().astype(float)
            n = len(vals)
            records.append({
                "district_code": g.district_code, "district_name": g.district_name,
                "crop": g.crop, "crop_name": g.crop_name, "validation_year": int(target),
                "baseline_mean_kg_per_ha": float(vals.mean()) if n else np.nan,
                "baseline_std_kg_per_ha": float(vals.std(ddof=1)) if n >= 2 else np.nan,
                "training_observation_count": int(n),
                "training_years": ",".join(map(str, sorted(training["year"].astype(int).unique()))),
                "baseline_status": "ready" if n >= MIN_TRAINING_OBSERVATIONS else "insufficient_history",
                "minimum_training_observations": MIN_TRAINING_OBSERVATIONS,
                "baseline_method": "forward_chaining",
            })
    return pd.DataFrame(records)


def make_forward_validation(data, normals, anomalies, input_path, original_hash):
    checks = []
    eligible = data[data["eligible_for_baseline"]]
    expected_rows = data[["district_code", "crop"]].drop_duplicates().shape[0] * data["year"].nunique()
    add_check(checks, "Expected forward baseline coverage", len(normals) == expected_rows,
              f"Expected={expected_rows}; actual={len(normals)}")
    unmatched = int(anomalies["training_observation_count"].isna().sum())
    add_check(checks, "Every source row has one forward baseline", unmatched == 0,
              f"Rows without matched baseline: {unmatched}")

    future_leaks = training_failures = baseline_failures = 0
    for r in normals.itertuples(index=False):
        train = eligible[(eligible["district_code"] == r.district_code)
                         & (eligible["crop"] == r.crop)
                         & (eligible["year"] < r.validation_year)]
        vals = train["yield_kg_per_ha"].dropna().astype(float)
        years_expected = sorted(train["year"].astype(int).unique().tolist())
        years_recorded = [int(x) for x in str(r.training_years).split(",") if x]
        if any(y >= int(r.validation_year) for y in years_recorded):
            future_leaks += 1
        if years_recorded != years_expected or len(vals) != int(r.training_observation_count):
            training_failures += 1
        mean_expected = float(vals.mean()) if len(vals) else np.nan
        std_expected = float(vals.std(ddof=1)) if len(vals) >= 2 else np.nan
        mean_ok = ((pd.isna(mean_expected) and pd.isna(r.baseline_mean_kg_per_ha))
                   or (pd.notna(mean_expected) and pd.notna(r.baseline_mean_kg_per_ha)
                       and np.isclose(mean_expected, r.baseline_mean_kg_per_ha, rtol=1e-10, atol=1e-10)))
        std_ok = ((pd.isna(std_expected) and pd.isna(r.baseline_std_kg_per_ha))
                  or (pd.notna(std_expected) and pd.notna(r.baseline_std_kg_per_ha)
                      and np.isclose(std_expected, r.baseline_std_kg_per_ha, rtol=1e-10, atol=1e-10)))
        if not (mean_ok and std_ok):
            baseline_failures += 1

    add_check(checks, "No future-year leakage", future_leaks == 0,
              f"Baseline rows with current/future training years: {future_leaks}")
    add_check(checks, "Forward training years and counts agree", training_failures == 0,
              f"Inconsistent rows: {training_failures}")
    add_check(checks, "Independent forward-baseline recomputation", baseline_failures == 0,
              f"Baseline rows failing recomputation: {baseline_failures}")

    eligible_mask = anomalies["eligible_for_baseline"].astype(bool)
    enough = anomalies["training_observation_count"].fillna(0).ge(MIN_TRAINING_OBSERVATIONS)
    assess = eligible_mask & enough
    abs_expected = anomalies["yield_kg_per_ha"] - anomalies["baseline_mean_kg_per_ha"]
    pct_expected = abs_expected / anomalies["baseline_mean_kg_per_ha"] * 100.0
    z_expected = abs_expected / anomalies["baseline_std_kg_per_ha"]

    def matches(actual, expected, mask):
        a = pd.to_numeric(actual[mask], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(expected[mask], errors="coerce").to_numpy(dtype=float)
        return bool(np.allclose(a, e, rtol=1e-9, atol=1e-9, equal_nan=True))

    abs_ok = matches(anomalies["yield_anomaly_kg_per_ha"], abs_expected, assess)
    pct_mask = assess & anomalies["baseline_mean_kg_per_ha"].ne(0)
    z_mask = assess & anomalies["baseline_std_kg_per_ha"].gt(0)
    pct_ok = matches(anomalies["yield_anomaly_pct"], pct_expected, pct_mask)
    z_ok = matches(anomalies["yield_anomaly_zscore"], z_expected, z_mask)
    add_check(checks, "Forward anomaly formulas independently recomputed",
              abs_ok and pct_ok and z_ok,
              f"Absolute={abs_ok}; percentage={pct_ok}; z-score={z_ok}")

    metric_cols = ["yield_anomaly_kg_per_ha", "yield_anomaly_pct", "yield_anomaly_zscore"]
    excluded_metrics = int(anomalies.loc[~eligible_mask, metric_cols].notna().any(axis=1).sum())
    insufficient_metrics = int(anomalies.loc[eligible_mask & ~enough, metric_cols].notna().any(axis=1).sum())
    add_check(checks, "Excluded forward observations have no metrics", excluded_metrics == 0,
              f"Excluded rows with metrics: {excluded_metrics}")
    add_check(checks, "Insufficient-history forward observations have no metrics", insufficient_metrics == 0,
              f"Insufficient-history rows with metrics: {insufficient_metrics}")

    zero_std = assess & anomalies["baseline_std_kg_per_ha"].eq(0)
    zero_std_z = int(anomalies.loc[zero_std, "yield_anomaly_zscore"].notna().sum())
    add_check(checks, "Forward zero-standard-deviation cases have no z-score", zero_std_z == 0,
              f"Zero-std rows with z-score: {zero_std_z}")

    expected_status = pd.Series("ineligible_observation", index=anomalies.index, dtype="object")
    expected_status.loc[eligible_mask & ~enough] = "insufficient_history"
    expected_status.loc[zero_std] = "zero_baseline_std"
    expected_status.loc[z_mask] = "calculated"
    status_failures = int(anomalies["anomaly_status"].ne(expected_status).sum())

    expected_class = pd.Series("not_assessed", index=anomalies.index, dtype="object")
    expected_class.loc[zero_std] = "zero_baseline_std"
    expected_class.loc[z_mask] = "within_threshold"
    expected_class.loc[z_mask & anomalies["yield_anomaly_zscore"].le(LOW_Z_THRESHOLD)] = "unusually_low"
    expected_class.loc[z_mask & anomalies["yield_anomaly_zscore"].ge(HIGH_Z_THRESHOLD)] = "unusually_high"
    class_failures = int(anomalies["yield_deviation_class"].ne(expected_class).sum())
    add_check(checks, "Forward statuses and deviation classes are consistent",
              status_failures == 0 and class_failures == 0,
              f"Status mismatches={status_failures}; class mismatches={class_failures}")

    first_year = int(data["year"].min())
    first_rows = normals[normals["validation_year"].eq(first_year)]
    first_year_nonzero = int(first_rows["training_observation_count"].ne(0).sum())
    add_check(checks, "First year has no prior training observations", first_year_nonzero == 0,
              f"First-year rows with training observations: {first_year_nonzero}")

    current_hash = sha256_file(input_path)
    add_check(checks, "Input file unchanged during forward processing", current_hash == original_hash,
              f"SHA-256: {original_hash}")
    return pd.DataFrame(checks)


def build_anomalies(data, normals):
    out = data.merge(
        normals,
        left_on=["district_code", "district_name", "crop", "crop_name", "year"],
        right_on=[
            "district_code", "district_name", "crop", "crop_name",
            "validation_year",
        ],
        how="left",
        validate="one_to_one",
    )

    actual = out["yield_kg_per_ha"]
    mean = out["baseline_mean_kg_per_ha"]
    std = out["baseline_std_kg_per_ha"]
    enough = out["training_observation_count"].fillna(0).ge(MIN_TRAINING_OBSERVATIONS)
    assess = out["eligible_for_baseline"] & enough

    out["yield_anomaly_kg_per_ha"] = np.where(assess, actual - mean, np.nan)
    out["yield_anomaly_pct"] = np.where(
        assess & mean.ne(0), (actual - mean) / mean * 100.0, np.nan
    )
    out["yield_anomaly_zscore"] = np.where(
        assess & std.gt(0), (actual - mean) / std, np.nan
    )

    out["anomaly_status"] = "ineligible_observation"
    out.loc[out["eligible_for_baseline"] & ~enough, "anomaly_status"] = "insufficient_history"
    out.loc[assess & std.eq(0), "anomaly_status"] = "zero_baseline_std"
    out.loc[assess & std.gt(0), "anomaly_status"] = "calculated"

    z = out["yield_anomaly_zscore"]
    out["yield_deviation_class"] = "not_assessed"
    out.loc[out["anomaly_status"].eq("zero_baseline_std"), "yield_deviation_class"] = "zero_baseline_std"
    out.loc[out["anomaly_status"].eq("calculated"), "yield_deviation_class"] = "within_threshold"
    out.loc[out["anomaly_status"].eq("calculated") & z.le(LOW_Z_THRESHOLD),
            "yield_deviation_class"] = "unusually_low"
    out.loc[out["anomaly_status"].eq("calculated") & z.ge(HIGH_Z_THRESHOLD),
            "yield_deviation_class"] = "unusually_high"
    return out


def make_validation(source, data, normals, anomalies, input_path, original_hash):
    checks = []

    # 1. Source-key uniqueness and output row preservation.
    duplicate_count = int(source.duplicated(KEYS).sum())
    add_check(
        checks, "Unique district-crop-year source keys",
        duplicate_count == 0, f"Duplicate rows: {duplicate_count}"
    )
    add_check(
        checks, "Anomaly output preserves every source row",
        len(anomalies) == len(source),
        f"Source={len(source)}; anomalies={len(anomalies)}"
    )

    # 2. Every source row must match exactly one LOYO baseline record.
    expected_baseline_rows = (
        data[["district_code", "crop"]].drop_duplicates().shape[0]
        * data["year"].nunique()
    )
    add_check(
        checks, "Expected LOYO baseline coverage",
        len(normals) == expected_baseline_rows,
        f"Expected={expected_baseline_rows}; actual={len(normals)}"
    )
    unmatched = int(anomalies["training_observation_count"].isna().sum())
    add_check(
        checks, "Every source row has one matched LOYO baseline",
        unmatched == 0,
        f"Rows without a matched baseline: {unmatched}"
    )

    # 3. Recompute training years, counts, means and sample standard deviations.
    eligible = data[data["eligible_for_baseline"]]
    leakage_count = 0
    baseline_failures = 0
    training_year_failures = 0

    for r in normals.itertuples(index=False):
        group_rows = eligible[
            (eligible["district_code"] == r.district_code)
            & (eligible["crop"] == r.crop)
            & (eligible["year"] != r.validation_year)
        ]
        vals = group_rows["yield_kg_per_ha"].dropna().astype(float)
        actual_training_years = sorted(group_rows["year"].astype(int).unique().tolist())
        recorded_training_years = [
            int(x) for x in str(r.training_years).split(",") if x
        ]

        if int(r.validation_year) in recorded_training_years:
            leakage_count += 1

        if (
            recorded_training_years != actual_training_years
            or len(vals) != int(r.training_observation_count)
        ):
            training_year_failures += 1

        expected_mean = float(vals.mean()) if len(vals) else np.nan
        expected_std = float(vals.std(ddof=1)) if len(vals) >= 2 else np.nan

        mean_ok = (
            (pd.isna(expected_mean) and pd.isna(r.baseline_mean_kg_per_ha))
            or (
                pd.notna(expected_mean)
                and pd.notna(r.baseline_mean_kg_per_ha)
                and np.isclose(
                    expected_mean, r.baseline_mean_kg_per_ha,
                    rtol=1e-10, atol=1e-10
                )
            )
        )
        std_ok = (
            (pd.isna(expected_std) and pd.isna(r.baseline_std_kg_per_ha))
            or (
                pd.notna(expected_std)
                and pd.notna(r.baseline_std_kg_per_ha)
                and np.isclose(
                    expected_std, r.baseline_std_kg_per_ha,
                    rtol=1e-10, atol=1e-10
                )
            )
        )
        if not (mean_ok and std_ok):
            baseline_failures += 1

    add_check(
        checks, "No validation-year leakage",
        leakage_count == 0, f"Leaking baseline rows: {leakage_count}"
    )
    add_check(
        checks, "Training years and observation counts agree",
        training_year_failures == 0,
        f"Baseline rows with inconsistent training years/counts: {training_year_failures}"
    )
    add_check(
        checks, "Independent baseline recomputation",
        baseline_failures == 0,
        f"Baseline rows failing recomputation: {baseline_failures}"
    )

    # 4. Independently recompute anomaly values and their classifications.
    eligible_mask = anomalies["eligible_for_baseline"].astype(bool)
    enough_mask = (
        anomalies["training_observation_count"].fillna(0)
        .ge(MIN_TRAINING_OBSERVATIONS)
    )
    assess_mask = eligible_mask & enough_mask

    expected_abs = anomalies["yield_kg_per_ha"] - anomalies["baseline_mean_kg_per_ha"]
    expected_pct = (
        expected_abs / anomalies["baseline_mean_kg_per_ha"] * 100.0
    )
    expected_z = (
        expected_abs / anomalies["baseline_std_kg_per_ha"]
    )

    def numeric_column_matches(actual_series, expected_series, mask):
        actual_values = pd.to_numeric(
            actual_series[mask], errors="coerce"
        ).to_numpy(dtype=float)
        expected_values = pd.to_numeric(
            expected_series[mask], errors="coerce"
        ).to_numpy(dtype=float)
        return bool(np.allclose(
            actual_values, expected_values,
            rtol=1e-9, atol=1e-9, equal_nan=True
        ))

    abs_ok = numeric_column_matches(
        anomalies["yield_anomaly_kg_per_ha"], expected_abs, assess_mask
    )
    pct_mask = assess_mask & anomalies["baseline_mean_kg_per_ha"].ne(0)
    pct_ok = numeric_column_matches(
        anomalies["yield_anomaly_pct"], expected_pct, pct_mask
    )
    z_mask = assess_mask & anomalies["baseline_std_kg_per_ha"].gt(0)
    z_ok = numeric_column_matches(
        anomalies["yield_anomaly_zscore"], expected_z, z_mask
    )
    add_check(
        checks, "Anomaly formulas independently recomputed",
        abs_ok and pct_ok and z_ok,
        f"Absolute={abs_ok}; percentage={pct_ok}; z-score={z_ok}"
    )

    # 5. Ineligible and insufficient-history rows must not receive anomaly metrics.
    metric_cols = [
        "yield_anomaly_kg_per_ha",
        "yield_anomaly_pct",
        "yield_anomaly_zscore",
    ]
    ineligible_with_metrics = int(
        anomalies.loc[~eligible_mask, metric_cols].notna().any(axis=1).sum()
    )
    insufficient_mask = eligible_mask & ~enough_mask
    insufficient_with_metrics = int(
        anomalies.loc[insufficient_mask, metric_cols].notna().any(axis=1).sum()
    )
    add_check(
        checks, "Excluded observations have no anomaly metrics",
        ineligible_with_metrics == 0,
        f"Excluded rows with one or more anomaly metrics: {ineligible_with_metrics}"
    )
    add_check(
        checks, "Insufficient-history observations have no anomaly metrics",
        insufficient_with_metrics == 0,
        f"Insufficient-history rows with metrics: {insufficient_with_metrics}"
    )

    # 6. A zero baseline standard deviation must never produce a z-score.
    zero_std_mask = assess_mask & anomalies["baseline_std_kg_per_ha"].eq(0)
    zero_std_with_z = int(
        anomalies.loc[zero_std_mask, "yield_anomaly_zscore"].notna().sum()
    )
    add_check(
        checks, "Zero-standard-deviation cases have no z-score",
        zero_std_with_z == 0,
        f"Zero-std rows with a z-score: {zero_std_with_z}"
    )

    # 7. Check the anomaly status and high/low screening labels.
    expected_status = pd.Series(
        "ineligible_observation", index=anomalies.index, dtype="object"
    )
    expected_status.loc[insufficient_mask] = "insufficient_history"
    expected_status.loc[zero_std_mask] = "zero_baseline_std"
    expected_status.loc[z_mask] = "calculated"
    status_failures = int(
        anomalies["anomaly_status"].ne(expected_status).sum()
    )

    expected_class = pd.Series(
        "not_assessed", index=anomalies.index, dtype="object"
    )
    expected_class.loc[zero_std_mask] = "zero_baseline_std"
    expected_class.loc[z_mask] = "within_threshold"
    expected_class.loc[z_mask & anomalies["yield_anomaly_zscore"].le(LOW_Z_THRESHOLD)] = "unusually_low"
    expected_class.loc[z_mask & anomalies["yield_anomaly_zscore"].ge(HIGH_Z_THRESHOLD)] = "unusually_high"
    class_failures = int(
        anomalies["yield_deviation_class"].ne(expected_class).sum()
    )
    add_check(
        checks, "Anomaly statuses and deviation classes are consistent",
        status_failures == 0 and class_failures == 0,
        f"Status mismatches={status_failures}; class mismatches={class_failures}"
    )

    # 8. Eligibility must agree with the documented conservative rules.
    role_ok = data["modelling_role"].astype(str).str.strip().eq("include")
    yield_ok = (
        data["yield_kg_per_ha"].notna()
        & np.isfinite(data["yield_kg_per_ha"].fillna(np.nan))
        & data["yield_kg_per_ha"].gt(0)
    )
    not_filled = ~data["flag_nan_filled_with_zero"]
    positive_area_zero_production = (
        data["area_1000_ha"].gt(0)
        & data["production_1000_tons"].eq(0)
    ).fillna(False)
    expected_eligible = (
        role_ok & yield_ok & not_filled & ~positive_area_zero_production
    )
    eligibility_failures = int(
        data["eligible_for_baseline"].ne(expected_eligible).sum()
    )
    add_check(
        checks, "Eligibility follows documented rules",
        eligibility_failures == 0,
        f"Eligibility mismatches: {eligibility_failures}"
    )

    add_check(
        checks, "Eligible observations exist",
        int(data["eligible_for_baseline"].sum()) > 0,
        f"Eligible observations: {int(data['eligible_for_baseline'].sum())}"
    )

    # 9. Ensure the input data was not modified during processing.
    current_hash = sha256_file(input_path)
    add_check(
        checks, "Input file unchanged during processing",
        current_hash == original_hash,
        f"SHA-256: {original_hash}"
    )

    checks_df = pd.DataFrame(checks)
    missingness = (
        data.groupby(["modelling_role", "eligibility_reason"], dropna=False)
        .size().reset_index(name="record_count")
        .sort_values(["modelling_role", "eligibility_reason"])
    )
    coverage = (
        data.groupby(["district_code", "district_name", "crop", "crop_name"])
        .agg(
            source_year_count=("year", "nunique"),
            eligible_observation_count=("eligible_for_baseline", "sum"),
            first_source_year=("year", "min"),
            last_source_year=("year", "max")
        )
        .reset_index()
    )
    coverage["coverage_status"] = np.select(
        [
            coverage["eligible_observation_count"].eq(0),
            coverage["eligible_observation_count"].eq(1),
            coverage["eligible_observation_count"].ge(2)
        ],
        [
            "no_valid_observations",
            "one_valid_observation",
            "multiple_valid_observations"
        ],
        default="unknown"
    )
    return checks_df, missingness, coverage


def write_report(path, input_path, input_hash, data, normals, anomalies, checks, method="LOYO"):
    if method == "Forward chaining":
        method_description = "Forward chaining: only eligible observations from years strictly before the target year are used."
        limitation_lines = [
            "Forward chaining uses only prior years and is intended for prospective evaluation.",
            "The first target year has no prior observations; early years may have insufficient history.",
        ]
    else:
        method_description = "LOYO: the held-out year is excluded from its own district-crop baseline; other years may be used."
        limitation_lines = [
            "LOYO is retrospective and may use years later than the held-out year.",
            "It is not a prospective forecasting baseline.",
        ]

    text = [
        "QWeatherRiskF - Issue #4 Yield Feature Engineering",
        "=" * 56, "",
        f"Method: {method}",
        f"Input: {input_path}",
        f"Input SHA-256: {input_hash}",
        f"Source rows: {len(data)}",
        f"Districts: {data['district_code'].nunique()}",
        f"Crops: {data['crop'].nunique()}",
        f"Years: {', '.join(map(str, sorted(data['year'].unique())))}",
        f"Eligible observations: {int(data['eligible_for_baseline'].sum())}",
        f"Baseline records: {len(normals)}",
        f"Anomaly records: {len(anomalies)}", "",
        "Methodology",
        "-----------",
        method_description,
        f"Minimum training observations for anomaly metrics: {MIN_TRAINING_OBSERVATIONS}",
        "Mean is reported with >=1 training observation; sample std requires >=2.",
        "Absolute, percentage and z-score anomalies require >=2 training observations.",
        f"Low screening threshold: z <= {LOW_Z_THRESHOLD}",
        f"High screening threshold: z >= {HIGH_Z_THRESHOLD}", "",
        "Eligibility",
        "-----------",
        "Start with modelling_role == include.",
        "Exclude missing, non-finite, zero or negative yields.",
        "Exclude flag_nan_filled_with_zero == True.",
        "Exclude positive-area / zero-production records.",
        "Retain mismatch, yield-jump and estimated-allocation flags for review.",
        "Positive-area / zero-production rows are excluded, not corrected; review source data.", "",
        "Limitations",
        "-----------",
        *limitation_lines,
        "Z-score thresholds are statistical screening heuristics, not agronomic diagnoses.", "",
        "Validation checks",
        "-----------------",
        checks.to_string(index=False), "",
        f"Failed checks: {int(checks['result'].ne('PASS').sum())}",
    ]
    Path(path).write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Generate district-crop LOYO and forward-chaining yield baselines and anomalies."
    )
    parser.add_argument("--input", required=True, help="Input crop-long CSV")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    outdir = os.path.abspath(args.outdir)
    validation_dir = os.path.join(outdir, "validation_tables")
    os.makedirs(validation_dir, exist_ok=True)

    input_hash = sha256_file(input_path)
    source, data = load_and_prepare(input_path)
    years = sorted(data["year"].unique().tolist())

    # Retrospective LOYO pipeline.
    normals = build_normals(data, years)
    anomalies = build_anomalies(data, normals)
    checks, missingness, coverage = make_validation(
        source, data, normals, anomalies, input_path, input_hash
    )

    # Prospective forward-chaining pipeline. Its outputs remain separate from LOYO.
    forward_normals = build_forward_normals(data, years)
    forward_anomalies = build_anomalies(data, forward_normals)
    forward_checks = make_forward_validation(
        data, forward_normals, forward_anomalies, input_path, input_hash
    )

    normals.to_csv(os.path.join(outdir, "district_crop_yield_normals_v1.csv"), index=False)
    anomalies.to_csv(os.path.join(outdir, "yield_anomalies_v1.csv"), index=False)
    forward_normals.to_csv(
        os.path.join(outdir, "district_crop_yield_normals_forward_chaining_v1.csv"), index=False
    )
    forward_anomalies.to_csv(
        os.path.join(outdir, "yield_anomalies_forward_chaining_v1.csv"), index=False
    )

    checks.to_csv(os.path.join(validation_dir, "validation_checks.csv"), index=False)
    forward_checks.to_csv(
        os.path.join(validation_dir, "forward_chaining_validation_checks.csv"), index=False
    )
    missingness.to_csv(os.path.join(validation_dir, "yield_missingness.csv"), index=False)
    coverage.to_csv(os.path.join(validation_dir, "yield_coverage.csv"), index=False)

    write_report(
        os.path.join(outdir, "validation_report.txt"),
        input_path, input_hash, data, normals, anomalies, checks
    )
    write_report(
        os.path.join(outdir, "forward_chaining_validation_report.txt"),
        input_path, input_hash, data, forward_normals, forward_anomalies, forward_checks,
        method="Forward chaining"
    )

    print("\nISSUE #4 YIELD FEATURE ENGINEERING")
    print("-" * 42)
    print(f"Source rows: {len(source)}")
    print(f"Eligible observations: {int(data['eligible_for_baseline'].sum())}")
    print(f"District-crop combinations: {len(coverage)}")
    print(f"LOYO baseline records: {len(normals)}")
    print(f"LOYO anomaly records: {len(anomalies)}")
    print(f"LOYO validation PASS: {(checks['result'] == 'PASS').sum()}")
    print(f"LOYO validation FAIL: {(checks['result'] == 'FAIL').sum()}")
    print(f"Forward baseline records: {len(forward_normals)}")
    print(f"Forward anomaly records: {len(forward_anomalies)}")
    print(f"Forward validation PASS: {(forward_checks['result'] == 'PASS').sum()}")
    print(f"Forward validation FAIL: {(forward_checks['result'] == 'FAIL').sum()}")
    print(f"Output folder: {outdir}")

    if checks["result"].eq("FAIL").any() or forward_checks["result"].eq("FAIL").any():
        raise SystemExit("Validation failed. Review LOYO and forward-chaining validation outputs.")


if __name__ == "__main__":
    main()
