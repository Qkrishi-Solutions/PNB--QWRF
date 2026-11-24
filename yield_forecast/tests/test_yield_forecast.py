"""
test_yield_forecast.py - Automated Test Suite for Issue #50 ML Yield Forecasting

Comprehensive validation of:
  1. Input file existence and schema integrity.
  2. Primary key uniqueness, join cardinality, and row preservation.
  3. Crop eligibility filtering and usable record count accounting.
  4. Missing target handling (no missing-to-zero corruption).
  5. Absence of target-derived predictors or future-year leakage.
  6. Strict forward-chaining chronology (t_train < t_test).
  7. Train-only feature selection and preprocessing isolation.
  8. Climatology baseline calculation logic and fallback.
  9. Model training and inference execution (Ridge, RandomForest, XGBoost, CatBoost).
 10. Forecast timing taxonomy and honest classification (retrospective hindcast).
 11. Prediction interval mathematical validity and honest uncalibrated labeling.
 12. Identical held-out evaluation instances across all 5 models.
 13. Statistical significance reporting (no ungrounded superiority claims).
 14. Deliverable table integrity and reproducibility.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from yield_forecast.train_forecast import (
    ClimatologyBaseline,
    select_features_in_fold,
    run_training_pipeline
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def test_input_files_exist_and_non_empty():
    """Verify all upstream canonical input files exist and are populated."""
    canonical_inputs = [
        "CropLongFormatv1/crop_long_v1.csv",
        "yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv",
        "weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv",
        "District Climate Zonal Clustering/district_zone_map_v1.csv",
        "CropModelEligibility/crop_model_eligibility_v1.csv",
        "crop specific weather feature selection/crop_feature_selection_v1.csv"
    ]
    for path in canonical_inputs:
        assert os.path.exists(path), f"Required input file missing: {path}"
        assert os.path.getsize(path) > 0, f"Input file is empty: {path}"


def test_generated_dataset_schema():
    """Verify generated ML-ready dataset schema, row count, and essential columns."""
    dataset_path = "yield_forecast/data/ml_yield_dataset_v1.csv"
    assert os.path.exists(dataset_path), "Generated ML dataset missing"

    df = pd.read_csv(dataset_path)
    assert df.shape[0] == 1430, f"Expected 1430 rows, got {df.shape[0]}"
    assert df.shape[1] >= 200, f"Expected wide dataset with weather features, got {df.shape[1]} columns"

    expected_cols = [
        "district_code", "district_name", "crop", "year",
        "yield_kg_per_ha", "yield_anomaly_kg_per_ha", "yield_anomaly_pct",
        "is_eligible_crop", "has_valid_yield_anomaly", "usable_for_forward_chaining_eval",
        "climate_zone", "cluster_id"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_primary_key_uniqueness():
    """Verify primary key (district_code, crop, year) has zero duplicates."""
    df = pd.read_csv("yield_forecast/data/ml_yield_dataset_v1.csv")
    keys = ["district_code", "crop", "year"]
    dups = df.duplicated(subset=keys).sum()
    assert dups == 0, f"Found {dups} duplicate primary keys in ML dataset"


def test_join_cardinality_and_row_preservation():
    """Verify join operations did not drop or duplicate baseline rows (must equal 1,430)."""
    df = pd.read_csv("yield_forecast/data/ml_yield_dataset_v1.csv")
    assert len(df) == 1430, f"Join cardinality violated: expected 1430, got {len(df)}"
    assert df["district_code"].nunique() == 13, "Must have exactly 13 districts"
    assert df["crop"].nunique() == 22, "Must have exactly 22 crops"
    assert df["year"].nunique() == 5, "Must have exactly 5 years (2015-2019)"


def test_crop_eligibility_and_usable_counts():
    """
    Verify crop eligibility mapping and explain the exact usable record counts.
    - Exactly 11 crops are eligible (proceed_to_modelling == True), yielding 715 records.
    - 11 crops are ineligible/conditional, yielding 715 records.
    - Usable evaluation records equal exactly 384 (128 per year in 2017, 2018, 2019).
    - Discrepancy from 11 * 13 = 143 is due to 15 uncultivated district-crop pairs in AP.
    """
    df = pd.read_csv("yield_forecast/data/ml_yield_dataset_v1.csv")
    df_elig = pd.read_csv("CropModelEligibility/crop_model_eligibility_v1.csv")

    eligible_crops = set(df_elig[df_elig["proceed_to_modelling"] == True]["crop"])
    assert len(eligible_crops) == 11, f"Expected 11 eligible crops, got {len(eligible_crops)}"

    # Check is_eligible_crop boolean flag
    assert df["is_eligible_crop"].sum() == 715, "11 eligible crops * 13 districts * 5 years must equal 715"

    # Check usable evaluation records
    df_usable = df[df["usable_for_forward_chaining_eval"] == True]
    assert len(df_usable) == 384, f"Expected 384 usable evaluation records, got {len(df_usable)}"

    for yr in [2017, 2018, 2019]:
        yr_count = (df_usable["year"] == yr).sum()
        assert yr_count == 128, f"Year {yr} must have exactly 128 usable records, got {yr_count}"


def test_missing_target_handling_no_zero_imputation():
    """Verify missing yield anomalies remain NaN and are never silently converted to zero."""
    df = pd.read_csv("yield_forecast/data/ml_yield_dataset_v1.csv")

    # 2015 and 2016 must be 100% NaN in forward chaining
    df_2015_2016 = df[df["year"].isin([2015, 2016])]
    assert df_2015_2016["yield_anomaly_kg_per_ha"].isna().all(), \
        "2015-2016 forward anomalies must be NaN due to insufficient history"
    assert (df_2015_2016["yield_anomaly_kg_per_ha"] == 0).sum() == 0, \
        "Missing anomalies must not be imputed as zero"


def test_no_target_derived_predictors():
    """Verify that predictor columns do not leak target-derived measures."""
    df = pd.read_csv("yield_forecast/data/ml_yield_dataset_v1.csv")
    forbidden = [
        "yield_kg_per_ha", "yield_check_kg_per_ha", "yield_rel_diff",
        "production_1000_tons", "yield_vs_other_years_median", "yield_deviation_class",
        "forward_mean_yield_kg_per_ha", "forward_std_yield_kg_per_ha"
    ]
    candidate_weather_cols = [c for c in df.columns if c.startswith("wanom__")]
    for col in candidate_weather_cols:
        for f in forbidden:
            assert f not in col, f"Predictor {col} contains forbidden target string {f}"


def test_forward_chaining_chronology():
    """Assert all forward-chaining folds strictly enforce t_train < t_test and exactly 2 folds exist."""
    config_path = "yield_forecast/models/model_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    folds = config["validation"]["folds"]
    assert len(folds) == 2, f"Exactly 2 forward-chaining folds expected, got {len(folds)}"

    for fold in folds:
        test_year = fold["test_year"]
        for train_yr in fold["train_years"]:
            assert train_yr < test_year, f"Future leakage: train {train_yr} >= test {test_year}"


def test_feature_selection_isolation(synthetic_sample_data):
    """Verify feature selection runs independently inside training fold data without leakage."""
    df = synthetic_sample_data
    train_df = df[df["year"] == 2017]
    candidate_features = [c for c in df.columns if c.startswith("wanom__")]

    selected_feats = select_features_in_fold(
        df_train=train_df,
        candidate_features=candidate_features,
        target_col="yield_anomaly_kg_per_ha",
        max_features=2,
        max_collinearity=0.80
    )
    assert len(selected_feats) <= 2
    for feat in selected_feats:
        assert feat in candidate_features


def test_climatology_baseline_calculation(synthetic_sample_data):
    """Verify ClimatologyBaseline accurately calculates crop-specific training means."""
    df = synthetic_sample_data
    train_df = df[df["year"] == 2017].copy()
    test_df = df[df["year"] == 2018].copy()

    model = ClimatologyBaseline().fit(train_df, "yield_anomaly_kg_per_ha")
    preds = model.predict(test_df)

    for c in test_df["crop"].unique():
        expected_crop_mean = train_df[train_df["crop"] == c]["yield_anomaly_kg_per_ha"].mean()
        crop_preds = preds[test_df["crop"] == c]
        np.testing.assert_allclose(crop_preds, expected_crop_mean, rtol=1e-5)


def test_model_training_and_inference_execution(synthetic_sample_data):
    """Verify Ridge, RandomForest, XGBoost, and CatBoost train and infer without error."""
    df = synthetic_sample_data
    train_df = df[df["year"] == 2017].copy()
    test_df = df[df["year"] == 2018].copy()

    X_train = train_df[["wanom__annual_cal__rain_sum", "wanom__kharif__rain_sum"]].values
    y_train = train_df["yield_anomaly_kg_per_ha"].values
    X_test = test_df[["wanom__annual_cal__rain_sum", "wanom__kharif__rain_sum"]].values

    models = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=10, max_depth=2, n_jobs=1, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=10, depth=2, thread_count=1, random_seed=42, verbose=False, allow_writing_files=False)
    }

    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        assert len(preds) == len(test_df), f"{name} output shape mismatch"
        assert not np.isnan(preds).any(), f"{name} produced NaN predictions"


def test_forecast_timing_and_retrospective_classification():
    """
    Verify weather window features and assert that configuration and taxonomy
    explicitly declare the pipeline as a retrospective response estimation (diagnostic hindcast),
    not an advance operational early-warning forecast.
    """
    config_path = "yield_forecast/models/model_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config.get("operational_taxonomy") == "end_of_season_retrospective_response_estimation_diagnostic_hindcast", \
        "Configuration must declare retrospective hindcast taxonomy"

    df = pd.read_csv("yield_forecast/data/ml_yield_dataset_v1.csv")
    wanom_cols = [c for c in df.columns if c.startswith("wanom__")]
    windows = set(c.split("__")[1] for c in wanom_cols)
    expected_windows = {"annual_cal", "annual_crop", "kharif", "ne", "rabi_next", "rabi_prev"}
    assert windows == expected_windows, f"Expected windows {expected_windows}, got {windows}"


def test_prediction_interval_calculations_and_honest_calibration_labels():
    """
    Verify prediction intervals satisfy lower <= pred <= upper, strictly positive width,
    and are honestly labeled as uncalibrated in-sample exploratory bounds.
    """
    preds_path = "yield_forecast/results/predictions_forward_chaining_v1.csv"
    assert os.path.exists(preds_path), "Predictions file missing"
    df_preds = pd.read_csv(preds_path)

    assert (df_preds["pi_lower_80"] <= df_preds["predicted_yield_anomaly"] + 1e-6).all(), \
        "Lower bound exceeded predicted value"
    assert (df_preds["predicted_yield_anomaly"] <= df_preds["pi_upper_80"] + 1e-6).all(), \
        "Predicted value exceeded upper bound"
    assert ((df_preds["pi_upper_80"] - df_preds["pi_lower_80"]) > 0).all(), \
        "Prediction interval width must be strictly positive"

    # Verify honest labeling in prediction interval evaluation deliverable
    pi_path = "yield_forecast/results/prediction_interval_evaluation_v1.csv"
    assert os.path.exists(pi_path), "Prediction interval evaluation file missing"
    df_pi = pd.read_csv(pi_path)
    assert "calibration_status" in df_pi.columns, "Must report calibration_status"
    assert (df_pi["calibration_status"] == "uncalibrated_in_sample_exploratory").all(), \
        "Must honestly label intervals as uncalibrated in-sample exploratory"
    assert "nominal_target_coverage" in df_pi.columns
    assert "empirical_test_coverage" in df_pi.columns


def test_consistent_evaluation_observations_across_models():
    """Verify all 5 models are evaluated on the exact same 256 test observations."""
    preds_path = "yield_forecast/results/predictions_forward_chaining_v1.csv"
    df_preds = pd.read_csv(preds_path)
    models = ["Climatology", "Ridge", "RandomForest", "XGBoost", "CatBoost"]

    ref_subset = None
    for m in models:
        m_df = df_preds[df_preds["model"] == m].sort_values(
            ["fold", "district_code", "crop", "test_year"]
        ).reset_index(drop=True)
        assert len(m_df) == 256, f"Model {m} evaluated on {len(m_df)} instances, expected 256"

        key_cols = ["fold", "district_code", "crop", "test_year", "actual_yield_anomaly"]
        if ref_subset is None:
            ref_subset = m_df[key_cols]
        else:
            pd.testing.assert_frame_equal(m_df[key_cols], ref_subset)


def test_statistical_significance_reporting():
    """Verify aggregate model comparison reports p-values and does not claim unsubstantiated superiority."""
    agg_path = "yield_forecast/results/aggregate_model_comparison_v1.csv"
    df_agg = pd.read_csv(agg_path)

    assert "paired_ttest_p_value" in df_agg.columns
    assert "statistically_significant_improvement" in df_agg.columns

    xgb_row = df_agg[df_agg["model"] == "XGBoost"].iloc[0]
    assert xgb_row["statistically_significant_improvement"] == "NO", \
        "XGBoost improvement must NOT be claimed as statistically significant (p ~ 0.87)"
    assert xgb_row["paired_ttest_p_value"] > 0.05, "p-value must reflect lack of statistical significance"


def test_output_deliverables_integrity():
    """Verify all required Issue #50 deliverable tables exist and are non-empty."""
    required_results = [
        "yield_forecast/results/predictions_forward_chaining_v1.csv",
        "yield_forecast/results/fold_evaluation_metrics_v1.csv",
        "yield_forecast/results/aggregate_model_comparison_v1.csv",
        "yield_forecast/results/crop_level_evaluation_metrics_v1.csv",
        "yield_forecast/results/prediction_interval_evaluation_v1.csv"
    ]
    for path in required_results:
        assert os.path.exists(path), f"Required result file missing: {path}"
        assert os.path.getsize(path) > 0, f"Result file is empty: {path}"

    df_agg = pd.read_csv("yield_forecast/results/aggregate_model_comparison_v1.csv")
    expected_models = {"Climatology", "Ridge", "RandomForest", "XGBoost", "CatBoost"}
    assert set(df_agg["model"].unique()) == expected_models, "Not all 5 required models evaluated"
