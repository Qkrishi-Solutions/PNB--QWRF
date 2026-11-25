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
    classify_performance,
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


ML_READY = "ml_ready_dataset/ml_ready_dataset_v1.csv"  # single dataset (Issue #7)


def _eligible_crop_long_rows():
    """Expected row count derived from the inputs, never hard-coded."""
    crop_long = pd.read_csv("CropLongFormatv1/crop_long_v1.csv")
    elig = pd.read_csv("CropModelEligibility/crop_model_eligibility_v1.csv")
    ok = set(elig.loc[elig["proceed_to_modelling"].astype(bool), "crop"])
    mask = crop_long["crop"].isin(ok) & crop_long["use_in_modelling"].astype(bool)
    return crop_long[mask], ok


def test_generated_dataset_schema():
    """The pipeline reads the Issue #7 ML-ready dataset; verify schema and derived row count."""
    assert os.path.exists(ML_READY), "Issue #7 ML-ready dataset missing"
    df = pd.read_csv(ML_READY)
    expected_rows, _ = _eligible_crop_long_rows()
    assert df.shape[0] == len(expected_rows), f"Expected {len(expected_rows)} rows, got {df.shape[0]}"
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
    df = pd.read_csv(ML_READY)
    keys = ["district_code", "crop", "year"]
    dups = df.duplicated(subset=keys).sum()
    assert dups == 0, f"Found {dups} duplicate primary keys in ML dataset"


def test_join_cardinality_and_row_preservation():
    """Joins must neither drop nor duplicate the eligible crop-long rows."""
    df = pd.read_csv(ML_READY)
    expected_rows, ok = _eligible_crop_long_rows()
    assert len(df) == len(expected_rows), f"Join cardinality violated: expected {len(expected_rows)}, got {len(df)}"
    assert df["district_code"].nunique() == expected_rows["district_code"].nunique()
    assert set(df["crop"]) == ok
    assert df["year"].nunique() == expected_rows["year"].nunique()


def test_crop_eligibility_and_usable_counts():
    """
    Eligible rows and usable evaluation rows must reconcile to the source files.
    All counts are derived from inputs; none are hard-coded.
    """
    df = pd.read_csv(ML_READY)
    expected_rows, ok = _eligible_crop_long_rows()
    assert df["is_eligible_crop"].all(), "Issue #7 keeps eligible crops only"

    src_y = pd.read_csv("yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv")
    keys = ["district_code", "year", "crop"]
    src = expected_rows[keys].merge(src_y[keys + ["yield_anomaly_kg_per_ha"]], on=keys, how="left")
    expected_usable = src["yield_anomaly_kg_per_ha"].notna()

    df_usable = df[df["usable_for_forward_chaining_eval"] == True]  # noqa: E712
    assert len(df_usable) == int(expected_usable.sum())
    expected_years = set(src.loc[expected_usable, "year"])
    assert set(df_usable["year"]) == expected_years
    assert not set(df_usable["year"]) & {2015, 2016}, "No forward-chaining anomaly exists for 2015-2016"
    for yr in expected_years:
        n = int(((src["year"] == yr) & expected_usable).sum())
        assert (df_usable["year"] == yr).sum() == n


def test_missing_target_handling_no_zero_imputation():
    """Verify missing yield anomalies remain NaN and are never silently converted to zero."""
    df = pd.read_csv("ml_ready_dataset/ml_ready_dataset_v1.csv")

    # 2015 and 2016 must be 100% NaN in forward chaining
    df_2015_2016 = df[df["year"].isin([2015, 2016])]
    assert df_2015_2016["yield_anomaly_kg_per_ha"].isna().all(), \
        "2015-2016 forward anomalies must be NaN due to insufficient history"
    assert (df_2015_2016["yield_anomaly_kg_per_ha"] == 0).sum() == 0, \
        "Missing anomalies must not be imputed as zero"


def test_no_target_derived_predictors():
    """Verify that predictor columns do not leak target-derived measures."""
    df = pd.read_csv("ml_ready_dataset/ml_ready_dataset_v1.csv")
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

    df = pd.read_csv("ml_ready_dataset/ml_ready_dataset_v1.csv")
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
    """Verdict columns must be consistent with the computed p-values and differences."""
    df_agg = pd.read_csv("yield_forecast/results/aggregate_model_comparison_v1.csv")

    assert "district_cluster_wilcoxon_p_value" in df_agg.columns
    assert "statistically_significant_improvement" in df_agg.columns
    assert "scaled_mae" in df_agg.columns and "macro_crop_scaled_mae" in df_agg.columns

    for _, r in df_agg[df_agg["model"] != "Climatology"].iterrows():
        sig, verdict = classify_performance(
            r["model"], r["scaled_mae_diff_vs_baseline"], r["district_cluster_wilcoxon_p_value"]
        )
        assert r["statistically_significant_improvement"] == sig
        assert r["performance_vs_climatology"] == verdict


def test_classify_performance_is_derived_not_hardcoded():
    assert classify_performance("XGBoost", -0.1, 0.01)[0] == "YES"
    assert classify_performance("XGBoost", -0.1, 0.60)[0] == "NO"
    assert "worse" in classify_performance("XGBoost", 0.1, 0.01)[1]
    assert "Not distinguishable" in classify_performance("Ridge", 0.1, 0.60)[1]
    assert classify_performance("Climatology", 0.0, 1.0)[0] == "Reference"


def test_feature_selection_does_not_impute():
    """Screening uses pairwise-complete data; a feature with NaNs must not be filled."""
    rng = np.random.default_rng(0)
    n = 40
    y = rng.normal(size=n)
    df = pd.DataFrame({
        "good": y + rng.normal(scale=0.1, size=n),
        "gappy": y + rng.normal(scale=0.1, size=n),
        "yield_anomaly_kg_per_ha": y,
    })
    df.loc[:5, "gappy"] = np.nan
    original = df.copy()
    sel = select_features_in_fold(df, ["good", "gappy"], "yield_anomaly_kg_per_ha", min_observed_fraction=1.0)
    assert sel == ["good"]
    sel_relaxed = select_features_in_fold(df, ["good", "gappy"], "yield_anomaly_kg_per_ha",
                                          max_collinearity=0.999, min_observed_fraction=0.8)
    assert "gappy" in sel_relaxed
    pd.testing.assert_frame_equal(df, original)  # input untouched (no in-place fill)


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


def test_fit_crop_scales_uses_training_rows_only_and_falls_back():
    from yield_forecast.train_forecast import fit_crop_scales
    tr = pd.DataFrame({
        "crop": ["a", "a", "a", "b", "b", "c"],
        "yield_anomaly_kg_per_ha": [1.0, 3.0, 5.0, 10.0, 30.0, 7.0],
    })
    scale, gstd = fit_crop_scales(tr, "yield_anomaly_kg_per_ha")
    assert np.isclose(scale["a"], np.std([1.0, 3.0, 5.0]))
    assert np.isclose(scale["b"], 10.0)
    assert scale["c"] == gstd  # single observation -> zero spread -> global fallback
    assert gstd > 0


def test_target_scaling_enabled_in_config_and_documented():
    with open("yield_forecast/models/model_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["target_scaling"]["enabled"] is True
    assert cfg["target_scaling"]["method"] == "per_crop_training_fold_std"
    assert "Exploratory" in cfg["target_scaling"]["note"]


def test_scaled_predictions_are_back_in_kg_per_ha():
    """Scaled-target pipeline must still output predictions in the original units."""
    preds = pd.read_csv("yield_forecast/results/predictions_forward_chaining_v1.csv")
    assert preds["predicted_yield_anomaly"].notna().all()
    # Magnitudes are on the kg/ha scale of the actuals, not ~1 (scaled units)
    assert preds["predicted_yield_anomaly"].abs().max() > 50
    assert np.allclose(preds["residual"], preds["actual_yield_anomaly"] - preds["predicted_yield_anomaly"])


def test_crop_level_metrics_have_sample_size_flags_and_skill():
    """Roadmap M5: per-crop sample-size flags, plus crop-scaled skill against climatology."""
    df = pd.read_csv("yield_forecast/results/crop_level_evaluation_metrics_v1.csv")
    for col in ["n_districts", "min_fold_n_test", "low_sample_flag", "scaled_mae",
                "skill_vs_climatology_pct", "beats_climatology_pct"]:
        assert col in df.columns, col
    with open("yield_forecast/models/model_config.json", encoding="utf-8") as f:
        thr = json.load(f)["evaluation"]["min_fold_test_samples_for_conclusion"]
    assert ((df["min_fold_n_test"] < thr) == df["low_sample_flag"]).all()
    base = df[df["model"] == "Climatology"]
    assert (base["skill_vs_climatology_pct"] == 0).all()
    ml = df[df["model"] != "Climatology"].merge(
        base[["crop", "scaled_mae"]].rename(columns={"scaled_mae": "base_scaled"}), on="crop")
    # skill is consistent with the scaled errors, and sign agrees with which is smaller
    assert np.allclose(ml["skill_vs_climatology_pct"], 100 * (1 - ml["scaled_mae"] / ml["base_scaled"]))
    assert (np.sign(ml["skill_vs_climatology_pct"]) == np.sign(ml["base_scaled"] - ml["scaled_mae"])).all()


def test_issue5_style_predictor_coverage_is_documented_in_dataset():
    """Issue 5-style forward-chaining per-crop predictors exist for only a minority of test rows."""
    df = pd.read_csv(ML_READY)
    test_rows = df[df["usable_for_forward_chaining_eval"] & (df["year"] >= 2018)]
    covered = test_rows["fc_n_selected_predictors"].fillna(0) > 0
    assert 0 < covered.mean() < 0.5
