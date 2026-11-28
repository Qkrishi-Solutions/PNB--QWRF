#!/usr/bin/env python3
"""
issue_51_production_forecast.py - Issue #51: Provisional Production Forecast
  with Propagated Yield Uncertainty

Computes production forecasts (area × yield) using Issue #50 yield anomaly
predictions, propagating yield-side uncertainty with area prediction intervals
from Issue #49 (model-specific per-model bounds; Ridge, RandomForest, XGBoost,
HistGBR each use their own interval columns; no Ridge reuse for other models).

Operational taxonomy: end-of-season retrospective response estimation,
yield-and-area and yield-only uncertainty propagation. Data joins use the
canonical primary key: ['district_code', 'crop', 'year'].
"""

import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
YIELD_FORECAST_DIR = BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_issue50_predictions(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """Load the Issue #50 forward-chaining predictions."""
    path = os.path.join(results_dir, "predictions_forward_chaining_v1.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Issue #50 predictions not found at {path}")
    df = pd.read_csv(path)
    print(f"Loaded Issue #50 predictions: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_ml_yield_dataset() -> pd.DataFrame:
    """Load the integrated ML yield dataset with area and baseline reference."""
    path = os.path.join(YIELD_FORECAST_DIR, "..", "ml_ready_dataset", "ml_ready_dataset_v1.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ML yield dataset not found at {path}")
    df = pd.read_csv(path)
    print(f"Loaded ML yield dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_yield_anomalies_forward_chaining() -> pd.DataFrame:
    """Load forward-chaining yield anomalies with baseline reference."""
    path = os.path.join(YIELD_FORECAST_DIR, "..", "yield_feature_engineering",
                          "yield_anomalies_forward_chaining_v1.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Forward-chaining anomalies not found at {path}")
    df = pd.read_csv(path)
    print(f"Loaded forward-chaining anomalies: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_area_forecast(results_dir: str = "") -> pd.DataFrame:
    """Load area forecast predictions from Issue #49."""
    if not results_dir:
        results_dir = os.path.dirname(os.path.abspath(__file__))
    area_forecast_path = os.path.join(results_dir, "..", "area_forecast", "area_forecast_v1.csv")
    if not os.path.exists(area_forecast_path):
        print(f"  Note: Area forecast not found at {area_forecast_path}; area forecasts will be unavailable")
        return pd.DataFrame()
    area_df = pd.read_csv(area_forecast_path)
    print(f"Loaded area forecast: {area_df.shape[0]} rows, {area_df.shape[1]} columns")
    return area_df


def compute_production_forecast(
    predictions_df: pd.DataFrame,
    ml_dataset_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    area_proxy: str = "historical_area"
) -> pd.DataFrame:
    """
    Compute provisional production forecasts with area-and-yield uncertainty propagation.

    Formula:
        predicted_yield_kg_per_ha = baseline_mean_kg_per_ha + predicted_yield_anomaly_kg_per_ha
        predicted_production_1000_tons = area_1000_ha * predicted_yield_kg_per_ha / 1000

    Area interval → Production interval (combined area-and-yield):
        pi_lower_prod_1000_tons = predicted_area_1000_ha * predicted_yield_kg_per_ha / 1000
        pi_upper_prod_1000_tons = predicted_area_1000_ha * predicted_yield_kg_per_ha / 1000

    Uncertainty scope: area_and_yield when valid area model available;
      yield_only when area model not found (e.g., CatBoost).

    Monte Carlo propagation: 1,000 uniform samples per record, rho=0 independence,
    seed 42; exploratory/uncalibrated.
    """

    print("=" * 70)
    print("Issue #51: Provisional Production Forecast with Area-and-Yield Uncertainty")
    print("=" * 70)

    # ---- Validate primary keys ----
    primary_key = ["district_code", "crop", "year"]

    # The predictions file uses 'test_year'; ensure it's available as 'year' for joins
    if "test_year" in predictions_df.columns and "year" not in predictions_df.columns:
        predictions_df = predictions_df.rename(columns={"test_year": "year"})
    assert "year" in predictions_df.columns, "Predictions must have 'year' column (renamed from test_year)"
    assert "district_code" in predictions_df.columns, "Predictions must have 'district_code' column"
    assert "crop" in predictions_df.columns, "Predictions must have 'crop' column"

    # ---- Load area forecast data from Issue #49 ----
    area_df = load_area_forecast()

    # ---- Join 1: predictions with ML dataset (area + baseline_mean) ----
    # We need: area_1000_ha and baseline_mean_kg_per_ha from the ML dataset
    # Join keys: district_code, crop, year
    predictions_joined = predictions_df.merge(
        ml_dataset_df[primary_key + ["area_1000_ha", "baseline_mean_kg_per_ha"]],
        on=primary_key,
        how="left"
    )

    print(f"\nAfter joining with ML dataset (area + baseline_mean): "
          f"{predictions_joined.shape[0]} rows (dropped {predictions_df.shape[0] - predictions_joined.shape[0]})")

    # Check for missing area or baseline after join
    missing_area_ml = predictions_joined["area_1000_ha"].isna().sum()
    missing_baseline_flag = predictions_joined["baseline_mean_kg_per_ha"].isna()
    # missing_baseline as per-row flag: True where baseline_mean is NaN
    predictions_joined["missing_baseline"] = missing_baseline_flag
    if missing_area_ml > 0:
        print(f"  WARNING: {missing_area_ml} rows missing area_1000_ha after join")
    if missing_baseline_flag.sum() > 0:
        print(f"  WARNING: {int(missing_baseline_flag.sum())} rows missing baseline_mean_kg_per_ha after join")

    # ---- Join 2: area forecast predictions from Issue #49 ----
    # Add area model predictions (Climatology, Ridge, RandomForest, XGBoost)
    # Note: we do NOT include area_1000_ha from area forecast to avoid column name conflict;
    # the historical area_1000_ha from the ML dataset is used as the area proxy
    # Also load model-specific prediction intervals from Issue #49
    area_interval_cols = [
        "pred_Ridge_Delta_pi_low", "pred_Ridge_Delta_pi_high",
        "pred_RandomForest_L1_LogDelta_pi_low", "pred_RandomForest_L1_LogDelta_pi_high",
        "pred_XGBoost_L1_LogDelta_pi_low", "pred_XGBoost_L1_LogDelta_pi_high",
    ]
    available_interval_cols = [c for c in area_interval_cols if c in area_df.columns]
    if not area_df.empty:
        merge_cols = primary_key + ["pred_Climatology", "pred_Ridge_Delta",
                                      "pred_RandomForest_L1_LogDelta", "pred_XGBoost_L1_LogDelta"] + available_interval_cols
        predictions_joined = predictions_joined.merge(
            area_df[merge_cols],
            on=primary_key,
            how="left"
        )
        # After area forecast join, check for rows that still have missing area predictions
        missing_area_af = predictions_joined["pred_Climatology"].isna().sum()
        if missing_area_af > 0:
            print(f"  Note: {missing_area_af} rows missing area forecast predictions")
    else:
        predictions_joined["pred_Climatology"] = np.nan
        predictions_joined["pred_Ridge_Delta"] = np.nan
        predictions_joined["pred_RandomForest_L1_LogDelta"] = np.nan
        predictions_joined["pred_XGBoost_L1_LogDelta"] = np.nan

    # ---- Join 3: anomalies for yield_anomaly_kg_per_ha (if not already in predictions) ----
    # The predictions already contain predicted_yield_anomaly, but we need baseline_mean
    # to reconstruct actual yield. The anomalies_df has baseline_mean_kg_per_ha.
    if "baseline_mean_kg_per_ha" not in predictions_joined.columns or predictions_joined["baseline_mean_kg_per_ha"].isna().all():
        predictions_joined = predictions_joined.merge(
            anomalies_df[primary_key + ["baseline_mean_kg_per_ha"]],
            on=primary_key,
            how="left"
        )
        missing_baseline2 = predictions_joined["baseline_mean_kg_per_ha"].isna().sum()
        if missing_baseline2 > 0:
            print(f"  WARNING: {missing_baseline2} rows still missing baseline_mean after second join")

    # ---- Reconstruct predicted actual yield ----
    # predicted_yield_kg_per_ha = baseline_mean_kg_per_ha + predicted_yield_anomaly_kg_per_ha
    predictions_joined["predicted_yield_kg_per_ha"] = (
        predictions_joined["baseline_mean_kg_per_ha"] +
        predictions_joined["predicted_yield_anomaly"]
    )

    # Flag negative predicted yield (do not silently clip)
    predictions_joined["predicted_yield_flag_negative"] = (
        predictions_joined["predicted_yield_kg_per_ha"] < 0
    )

    n_negative_yield = predictions_joined["predicted_yield_flag_negative"].sum()
    if n_negative_yield > 0:
        print(f"  FLAGGED: {n_negative_yield} rows with negative predicted yield (not clipped)")

    # Convert yield anomaly intervals to actual yield bounds using baseline_mean
    # pi_lower_80 / pi_upper_80 are Issue #50 anomaly intervals (in kg/ha above/below baseline)
    # Actual yield bounds = baseline_mean + anomaly_interval
    predictions_joined["yield_lower_80"] = (
        predictions_joined["baseline_mean_kg_per_ha"] + predictions_joined["pi_lower_80"]
    )
    predictions_joined["yield_upper_80"] = (
        predictions_joined["baseline_mean_kg_per_ha"] + predictions_joined["pi_upper_80"]
    )

    # ---- Determine uncertainty scope per-row ----
    # uncertainty_scope = area_and_yield when yield_model is NOT CatBoost
    # uncertainty_scope = yield_only when yield_model IS CatBoost (no area forecast)
    predictions_joined["uncertainty_scope"] = "yield_only"
    predictions_joined.loc[
        predictions_joined["model"].isin(["Ridge", "RandomForest", "XGBoost", "Climatology"]),
        "uncertainty_scope"
    ] = "area_and_yield"

    # missing_area_forecast flag: True when yield_model is CatBoost (no area forecast available)
    # Model name pairing (e.g., Ridge area with Ridge yield) is administrative only:
    #   shared model names are a convenience for referencing similarly-named models;
    #   they do NOT imply correlated errors between area and yield forecasts.
    #   Area and yield uncertainties are propagated independently (rho=0 Monte Carlo assumption).
    #   This independence assumption is explicit unless empirical evidence supports dependence.
    predictions_joined["missing_area_forecast"] = predictions_joined["model"].isin(["CatBoost"])

    # predicted_area_1000_ha: use area forecast if available, otherwise NaN for yield_only rows
    # For area_and_yield rows: use the forecasted area (pred_Climatology, pred_Ridge_Delta, etc.)
    # predicted_area_1000_ha: use area forecast if available, otherwise NaN for yield_only rows
    # For area_and_yield rows: use the forecasted area (pred_Climatology, pred_Ridge_Delta, etc.)
    area_and_yield_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    predictions_joined.loc[area_and_yield_mask, "predicted_area_1000_ha"] = np.nan
    for model_col, model_name in [("pred_Ridge_Delta", "Ridge"),
                                   ("pred_RandomForest_L1_LogDelta", "RandomForest"),
                                   ("pred_XGBoost_L1_LogDelta", "XGBoost"),
                                   ("pred_Climatology", "Climatology")]:
        model_mask = area_and_yield_mask & (predictions_joined["model"] == model_name)
        predictions_joined.loc[model_mask, "predicted_area_1000_ha"] = predictions_joined.loc[model_mask, model_col]

    # Check for negative area predictions and flag them (after predicted_area_1000_ha is set)
    negative_area_flag = predictions_joined["predicted_area_1000_ha"] < 0
    predictions_joined["negative_area_prediction"] = negative_area_flag
    # Flag negative yield anomalies (do not silently clip)
    predictions_joined["negative_yield_anomaly"] = predictions_joined["actual_yield_anomaly"] < 0
    # Set sampling_assumption based on uncertainty scope
    predictions_joined["sampling_assumption"] = "historical_area_yield_only_historical"
    ay_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    predictions_joined.loc[ay_mask, "sampling_assumption"] = "historical_area_yield_and_yield"

    # For yield_only rows (CatBoost): use latest available pre-forecast-year area from ML dataset
    # to avoid same-year area leakage; set as conditional area forecast
    yield_only_mask = predictions_joined["uncertainty_scope"] == "yield_only"
    predictions_joined.loc[yield_only_mask, "predicted_area_1000_ha"] = np.nan
    # Compute pre-forecast year area: find the most recent year < test_year for each (district_code, crop)
    for idx in predictions_joined[yield_only_mask].index:
        dc = predictions_joined.loc[idx, "district_code"]
        crop = predictions_joined.loc[idx, "crop"]
        test_year = predictions_joined.loc[idx, "year"]
        # Look up in ML dataset for earlier years
        ml_mask = (ml_dataset_df["district_code"] == dc) & \
                  (ml_dataset_df["crop"] == crop) & \
                  (ml_dataset_df["year"] < test_year)
        ml_matches = ml_dataset_df[ml_mask].sort_values("year", ascending=False)
        if len(ml_matches) > 0:
            pre_forecast_area = ml_matches.iloc[0]["area_1000_ha"]
            predictions_joined.loc[idx, "predicted_area_1000_ha"] = pre_forecast_area
        # else: remain NaN (should not happen with sufficient data)

    # Add conditional_area_forecast flag: True when predicted_area_1000_ha differs from same-year area_1000_ha
    predictions_joined["conditional_area_forecast"] = False
    # For yield_only rows where we found a pre-forecast year area, flag as conditional
    yomask = predictions_joined["uncertainty_scope"] == "yield_only"
    has_preforecast = predictions_joined.loc[yomask, "predicted_area_1000_ha"] != predictions_joined.loc[yomask, "area_1000_ha"]
    predictions_joined.loc[yomask & has_preforecast, "conditional_area_forecast"] = True
    # For area_and_yield rows, flag when using model-predicted area vs same-year actual area
    ay_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    has_pred_area = predictions_joined.loc[ay_mask, "predicted_area_1000_ha"] != predictions_joined.loc[ay_mask, "area_1000_ha"]
    predictions_joined.loc[ay_mask & has_pred_area, "conditional_area_forecast"] = True

    # area_pi_low_80 and area_pi_high_80: area prediction interval endpoints
    # Use model-specific prediction bounds from Issue #49 per model; do not reuse Ridge predictions for other models
    predictions_joined["area_pi_low_80"] = np.nan
    predictions_joined["area_pi_high_80"] = np.nan
    ay_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    # Map each model to its own prediction interval columns from Issue #49
    area_interval_map = {
        "Ridge": ("pred_Ridge_Delta_pi_low", "pred_Ridge_Delta_pi_high"),
        "RandomForest": ("pred_RandomForest_L1_LogDelta_pi_low", "pred_RandomForest_L1_LogDelta_pi_high"),
        "XGBoost": ("pred_XGBoost_L1_LogDelta_pi_low", "pred_XGBoost_L1_LogDelta_pi_high"),
        "HistGBR": ("pred_HistGBR_L1_LogDelta_pi_low", "pred_HistGBR_L1_LogDelta_pi_high"),
    }
    for model_name, (low_col, high_col) in area_interval_map.items():
        model_mask = ay_mask & (predictions_joined["model"] == model_name)
        if low_col in predictions_joined.columns and high_col in predictions_joined.columns:
            predictions_joined.loc[model_mask, "area_pi_low_80"] = predictions_joined.loc[model_mask, low_col]
            predictions_joined.loc[model_mask, "area_pi_high_80"] = predictions_joined.loc[model_mask, high_col]
    # Enforce area_pi_low_80 <= area_pi_high_80 for all eligible rows
    interval_mask = predictions_joined["area_pi_low_80"].notna() & predictions_joined["area_pi_high_80"].notna()
    # If low > high, swap to maintain invariant; NaN bounds are preserved
    swap_mask = interval_mask & (predictions_joined["area_pi_low_80"] > predictions_joined["area_pi_high_80"])
    if swap_mask.any():
        predictions_joined.loc[swap_mask, "area_pi_low_80"], predictions_joined.loc[swap_mask, "area_pi_high_80"] = \
            predictions_joined.loc[swap_mask, "area_pi_high_80"].copy(), \
            predictions_joined.loc[swap_mask, "area_pi_low_80"].copy()
    # Climatology has no formal prediction intervals in Issue #49; intervals remain NaN
    # This is intentional: do not manufacture or reuse Ridge bounds for Climatology

    # ---- Compute predicted production ----
    # predicted_production_1000_tons = predicted_area_1000_ha * predicted_yield_kg_per_ha / 1000
    predictions_joined["predicted_production_1000_tons"] = (
        predictions_joined["predicted_area_1000_ha"] * predictions_joined["predicted_yield_kg_per_ha"] / 1000
    )

    # ---- Convert yield interval endpoints to production interval endpoints ----
    # Use the same area proxy and add baseline_mean inside the interval
    # pi_lower_prod = area * (baseline_mean + pi_lower_80) / 1000
    # pi_upper_prod = area * (baseline_mean + pi_upper_80) / 1000

    # Ensure interval columns exist - use actual yield bounds
    if "yield_lower_80" in predictions_joined.columns and "yield_upper_80" in predictions_joined.columns:
        predictions_joined["pi_lower_prod_1000_tons"] = (
            predictions_joined["predicted_area_1000_ha"] *
            (predictions_joined["yield_lower_80"]) / 1000
        )
        predictions_joined["pi_upper_prod_1000_tons"] = (
            predictions_joined["predicted_area_1000_ha"] *
            (predictions_joined["yield_upper_80"]) / 1000
        )

        # Flag negative production interval endpoints (do not silently clip)
        predictions_joined["pi_lower_prod_flag_negative"] = (
            predictions_joined["pi_lower_prod_1000_tons"] < 0
        )
        predictions_joined["pi_upper_prod_flag_negative"] = (
            predictions_joined["pi_upper_prod_1000_tons"] < 0
        )
        n_neg_lower = predictions_joined["pi_lower_prod_flag_negative"].sum()
        n_neg_upper = predictions_joined["pi_upper_prod_flag_negative"].sum()
        if n_neg_lower > 0:
            print(f"  FLAGGED: {n_neg_lower} rows with negative pi_lower_prod endpoint (not clipped)")
        if n_neg_upper > 0:
            print(f"  FLAGGED: {n_neg_upper} rows with negative pi_upper_prod endpoint (not clipped)")
    else:
        predictions_joined["pi_lower_prod_1000_tons"] = np.nan
        predictions_joined["pi_upper_prod_1000_tons"] = np.nan
        predictions_joined["pi_lower_prod_flag_negative"] = False
        predictions_joined["pi_upper_prod_flag_negative"] = False

# ---- Monte Carlo uncertainty propagation ----
    # 1,000 samples per record, rho=0 independence, seed 42
    n_mc = 1000
    np.random.seed(42)
    
    # Initialize Monte Carlo production samples
    predictions_joined["mc_production_samples"] = None
    
    # For area_and_yield rows: combine area and yield uncertainty
    ay_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    y_mask = predictions_joined["uncertainty_scope"] == "yield_only"
    
    mc_results = np.full((len(predictions_joined), n_mc), np.nan)
    
    # For area_and_yield rows: combine area and yield uncertainty
    if ay_mask.any():
        # Quantile-matched Normal sampling: estimate sigma from interval widths
        # so that Normal(mean, sigma) has the same P90-P10 width as the original
        # prediction intervals. This avoids the interval-narrowing effect that can
        # occur when Uniform[P10,P90] samples are subsequently P10/P90-extracted.
        # sigma = (P90 - P10) / (2 * 1.28155) for a Normal distribution.
        area_lo = predictions_joined.loc[ay_mask, "area_pi_low_80"].values
        area_hi = predictions_joined.loc[ay_mask, "area_pi_high_80"].values
        sigma_area = np.where(
            np.isnan(area_lo) | np.isnan(area_hi) | (area_hi <= area_lo),
            np.nan,
            (area_hi - area_lo) / (2 * 1.28155)
        )
        yield_lo = predictions_joined.loc[ay_mask, "yield_lower_80"].values
        yield_hi = predictions_joined.loc[ay_mask, "yield_upper_80"].values
        sigma_yield = np.where(
            np.isnan(yield_lo) | np.isnan(yield_hi) | (yield_hi <= yield_lo),
            np.nan,
            (yield_hi - yield_lo) / (2 * 1.28155)
        )
        
        # Use predicted point estimates as Normal means
        area_mean = predictions_joined.loc[ay_mask, "predicted_area_1000_ha"].values
        yield_mean = predictions_joined.loc[ay_mask, "predicted_yield_kg_per_ha"].values
        
        # Sample from Normal distributions (rho=0 independence)
        area_samples = np.random.normal(
            np.tile(np.where(np.isnan(area_mean), 0, area_mean), (n_mc, 1)).T,
            np.tile(np.where(np.isnan(sigma_area), 1, sigma_area), (n_mc, 1)).T
        )
        yield_samples = np.random.normal(
            np.tile(np.where(np.isnan(yield_mean), 0, yield_mean), (n_mc, 1)).T,
            np.tile(np.where(np.isnan(sigma_yield), 1, sigma_yield), (n_mc, 1)).T
        )
        
        # Row-wise: determine which rows have valid sigma (valid interval bounds)
        # Valid sigma means both bounds exist and lower <= upper
        valid_area_sigma = ~np.isnan(sigma_area)
        valid_yield_sigma = ~np.isnan(sigma_yield)
        valid_both = valid_area_sigma & valid_yield_sigma
        
        # For rows with valid sigma, keep Normal samples
        # For rows with invalid sigma, set NaN (uncertainty unavailable)
        area_samples = np.where(valid_both, area_samples, np.nan)
        yield_samples = np.where(valid_both, yield_samples, np.nan)
        
        # Compute production samples only where both area and yield uncertainty are available
        # If either is NaN, the production sample will be NaN
        # Only rows with valid uncertainty for both area and yield will have valid production samples
        mc_prod = area_samples * yield_samples / 1000
        # Assign to mc_results for area_and_yield rows - NaN where uncertainty unavailable
        mc_results[ay_mask] = mc_prod
    
    # For yield_only rows: only yield uncertainty, area is historical (fixed)
    if y_mask.any():
        # Use historical area_1000_ha as fixed, sample yield only
        hist_area = predictions_joined.loc[y_mask, "area_1000_ha"].values
        # Sample yield uniformly from actual yield bounds (converted from anomaly intervals)
        yield_lo = predictions_joined.loc[y_mask, "yield_lower_80"].values
        yield_hi = predictions_joined.loc[y_mask, "yield_upper_80"].values
        
        # Monte Carlo: 1,000 uniform samples for yield only
        yield_samples = np.random.uniform(
            np.tile(yield_lo, (n_mc, 1)).T,
            np.tile(yield_hi, (n_mc, 1)).T
        )
        # Area is fixed at historical value, so production = area_1000_ha * yield / 1000
        mc_prod = np.tile(hist_area, (n_mc, 1)).T * yield_samples / 1000
        mc_results[y_mask] = mc_prod

    predictions_joined["mc_production_samples"] = mc_results.tolist()

# ---- Derive production interval endpoints from Monte Carlo ----
    # prod_pi_low_80 = 10th percentile, prod_pi_high_80 = 90th percentile
    predictions_joined["prod_pi_low_80"] = np.nan
    predictions_joined["prod_pi_high_80"] = np.nan
    
    if ay_mask.any():
        for i in range(len(predictions_joined)):
            if ay_mask.iloc[i]:
                samples = mc_results[i]
                if not np.all(np.isnan(samples)):
                    pred_sorted = np.nanpercentile(samples, 10)
                    predictions_joined.loc[i, "prod_pi_low_80"] = round(pred_sorted, 6)
    
    if y_mask.any():
        for i in range(len(predictions_joined)):
            if y_mask.iloc[i]:
                samples = mc_results[i]
                if not np.all(np.isnan(samples)):
                    pred_sorted = np.nanpercentile(samples, 10)
                    predictions_joined.loc[i, "prod_pi_low_80"] = round(pred_sorted, 6)
    
    # Set prod_pi_high_80 (90th percentile)
    predictions_joined["prod_pi_high_80"] = np.nan
    for i in range(len(predictions_joined)):
        if ay_mask.iloc[i] or y_mask.iloc[i]:
            samples = mc_results[i]
            if not np.all(np.isnan(samples)):
                pred_sorted = np.nanpercentile(samples, 90)
                predictions_joined.loc[i, "prod_pi_high_80"] = round(pred_sorted, 6)

    # ---- interval_methodology per-row ----
    # Determine interval methodology based on uncertainty scope
    predictions_joined["interval_methodology"] = "historical_area_yield_only_historical"
    ay_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    y_mask = predictions_joined["uncertainty_scope"] == "yield_only"
    predictions_joined.loc[ay_mask, "interval_methodology"] = (
        "in_sample_residual_quantile_80pct_area_and_yield"
    )
    predictions_joined.loc[y_mask, "interval_methodology"] = (
        "in_sample_residual_quantile_80pct_yield_only_historical_area"
    )

    # ---- Empirical coverage evaluation on held-out years ----
    # Compute whether the production interval covers the observed production
    # Separate coverage by uncertainty scope (area_and_yield vs yield_only)
    empirical_coverage = None
    n_coverage_obs = None
    coverage_by_year_model = []
    coverage_scope_metrics = {}  # per-scope: numerator, denominator, exclusions, percentage
    
    if "production_1000_tons" in ml_dataset_df.columns and \
       "prod_pi_low_80" in predictions_joined.columns and \
       "prod_pi_high_80" in predictions_joined.columns:
        # Join observed production
        coverage_df = predictions_joined.merge(
            ml_dataset_df[primary_key + ["production_1000_tons"]],
            on=primary_key,
            how="left"
        )
        # Add actual_production_1000_tons to predictions_joined from the merge
        predictions_joined["actual_production_1000_tons"] = coverage_df["production_1000_tons"]
        # Check coverage ONLY for rows with valid combined intervals and actual production
        valid_combined_mask = (
            (coverage_df["prod_pi_low_80"].notna()) &
            (coverage_df["prod_pi_high_80"].notna()) &
            (coverage_df["production_1000_tons"].notna())
        )
        coverage_df_valid = coverage_df[valid_combined_mask]
        
        # Coverage for area_and_yield rows
        ay_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
        ay_valid = valid_combined_mask & ay_mask
        ay_valid_df = coverage_df[ay_valid]
        if len(ay_valid_df) > 0:
            ay_coverage_mask = (
                (ay_valid_df["prod_pi_low_80"] <= ay_valid_df["production_1000_tons"]) &
                (ay_valid_df["production_1000_tons"] <= ay_valid_df["prod_pi_high_80"])
            )
            n_ay_eligible = ay_valid_mask.sum()
            n_ay_covered = int(ay_coverage_mask.sum())
            n_ay_excluded = int((~ay_valid_mask).sum())
            ay_coverage_rate = n_ay_covered / n_ay_eligible if n_ay_eligible > 0 else None
            coverage_scope_metrics["area_and_yield"] = {
                "n_eligible": int(n_ay_eligible),
                "n_covered": n_ay_covered,
                "n_excluded": n_ay_excluded,
                "coverage_percentage": ay_coverage_rate * 100 if ay_coverage_rate is not None else None
            }
        else:
            coverage_scope_metrics["area_and_yield"] = {
                "n_eligible": 0, "n_covered": 0, "n_excluded": 0, "coverage_percentage": None
            }
        
        # Coverage for yield_only rows
        y_mask = predictions_joined["uncertainty_scope"] == "yield_only"
        y_valid = valid_combined_mask & y_mask
        y_valid_df = coverage_df[y_valid]
        if len(y_valid_df) > 0:
            y_coverage_mask = (
                (y_valid_df["prod_pi_low_80"] <= y_valid_df["production_1000_tons"]) &
                (y_valid_df["production_1000_tons"] <= y_valid_df["prod_pi_high_80"])
            )
            n_y_eligible = y_valid_mask.sum()
            n_y_covered = int(y_coverage_mask.sum())
            n_y_excluded = int((~y_valid_mask).sum())
            y_coverage_rate = n_y_covered / n_y_eligible if n_y_eligible > 0 else None
            coverage_scope_metrics["yield_only"] = {
                "n_eligible": int(n_y_eligible),
                "n_covered": n_y_covered,
                "n_excluded": n_y_excluded,
                "coverage_percentage": y_coverage_rate * 100 if y_coverage_rate is not None else None
            }
        else:
            coverage_scope_metrics["yield_only"] = {
                "n_eligible": 0, "n_covered": 0, "n_excluded": 0, "coverage_percentage": None
            }
        
        if len(coverage_df_valid) > 0:
            coverage_mask = (
                (coverage_df_valid["prod_pi_low_80"] <= coverage_df_valid["production_1000_tons"]) &
                (coverage_df_valid["production_1000_tons"] <= coverage_df_valid["prod_pi_high_80"])
            )
            empirical_coverage = coverage_mask.sum() / len(coverage_df_valid)
            n_with_prod = coverage_df_valid["production_1000_tons"].notna().sum()
            print(f"\nEmpirical production interval coverage (valid combined forecast rows with observed production): "
                  f"{empirical_coverage*100:.1f}% ({empirical_coverage:.2f} "
                  f"among {n_with_prod} observations with valid combined forecast and production)")
        else:
            n_with_prod = 0
            print("\nNo rows with valid combined forecast intervals and observed production for coverage evaluation")
        
        # Initialize prod_pi_covered_80 as nullable boolean with NA default
        predictions_joined["prod_pi_covered_80"] = pd.NA
        
        # Set coverage only for eligible rows (valid combined forecast + observed production)
        eligible_mask = valid_combined_mask.copy()
        # True when production is inside the 80% prediction interval
        in_interval = (
            (coverage_df["prod_pi_low_80"][eligible_mask] <= coverage_df["production_1000_tons"][eligible_mask]) &
            (coverage_df["production_1000_tons"][eligible_mask] <= coverage_df["prod_pi_high_80"][eligible_mask])
        )
        # Set True for covered, False for uncovered, keep NA for ineligible
        pred_covered = pd.Series(pd.NA, index=predictions_joined.index)
        pred_covered.loc[eligible_mask & in_interval] = True
        pred_covered.loc[eligible_mask & ~in_interval] = False
        predictions_joined["prod_pi_covered_80"] = pred_covered
        
        # Count coverage metrics for reporting
        n_eligible = eligible_mask.sum()
        n_covered = int(pred_covered.eq(True).sum())
        n_excluded = int((~eligible_mask).sum())
        n_total = len(predictions_joined)
        
        # Also compute production_covered_80 for backward compatibility
        # Map: True->True, False->False, NA->NaN
        prod_covered_val = predictions_joined["prod_pi_covered_80"].copy()
        prod_covered_val = prod_covered_val.where(pd.notna(prod_covered_val), None)
        predictions_joined["production_covered_80"] = prod_covered_val.values
        
        # Coverage by year and model pairing
        if len(coverage_df_valid) > 0:
            for year_val in sorted(coverage_df_valid["year"].unique()):
                year_data = coverage_df_valid[coverage_df_valid["year"] == year_val]
                for model_val in sorted(year_data["model"].unique()):
                    model_data = year_data[year_data["model"] == model_val]
                    if len(model_data) > 0:
                        cm = (
                            (model_data["prod_pi_low_80"] <= model_data["production_1000_tons"]) &
                            (model_data["production_1000_tons"] <= model_data["prod_pi_high_80"])
                        )
                        coverage_by_year_model.append({
                            "year": year_val,
                            "model": model_val,
                            "n_obs": len(model_data),
                            "coverage_rate": cm.sum() / len(model_data)
                        })

    # ---- Return coverage info alongside output ----
    coverage_info = {
        "empirical_coverage": empirical_coverage,
        "coverage_observations": n_coverage_obs,
        "coverage_by_year_model": coverage_by_year_model,
        "coverage_by_scope": coverage_scope_metrics
    }

    # ---- Select and order output columns ----
    output_cols = [
        "fold", "train_years", "year", "district_code", "district_name",
        "crop", "model",
        "actual_yield_anomaly", "predicted_yield_anomaly",
        "baseline_mean_kg_per_ha", "predicted_yield_kg_per_ha",
        "area_1000_ha", "predicted_area_1000_ha",
        "area_pi_low_80", "area_pi_high_80",
        "predicted_production_1000_tons",
        "prod_pi_low_80", "prod_pi_high_80",
        "actual_production_1000_tons", "prod_pi_covered_80", "production_covered_80",
        "uncertainty_scope", "sampling_assumption",
        "interval_methodology",
        "missing_baseline", "missing_area_forecast", "negative_area_prediction", "negative_yield_anomaly",
        "pi_lower_80", "pi_upper_80",
        "residual",
        "conditional_area_forecast"
    ]

    # Only include columns that exist
    existing_output_cols = [c for c in output_cols if c in predictions_joined.columns]
    predictions_output = predictions_joined[existing_output_cols].copy()

    return predictions_output, coverage_info


def save_production_forecast(
    df: pd.DataFrame,
    results_dir: str = RESULTS_DIR,
    filename: str = "production_forecast_v1.csv"
) -> str:
    """Save the production forecast output."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    df.to_csv(path, index=False)
    print(f"\n[SAVED] Production forecast: {path} ({len(df)} rows)")
    return path


def generate_metrics_report(df: pd.DataFrame, coverage_n: int = None, coverage_rate: float = None) -> dict:
    """Generate a metrics report from the production forecast output."""
    report = {
        "total_observations": len(df),
        "observations_with_area": int(df["area_1000_ha"].notna().sum()) if "area_1000_ha" in df.columns else 0,
        "observations_with_production": int(df["predicted_production_1000_tons"].notna().sum()) if "predicted_production_1000_tons" in df.columns else 0,
        "observations_with_yield_interval": int(df["pi_lower_80"].notna().sum()) if "pi_lower_80" in df.columns else 0,
        "negative_yield_flags": int(df["predicted_yield_flag_negative"].sum()) if "predicted_yield_flag_negative" in df.columns else 0,
        "negative_pi_lower_flags": int(df["pi_lower_prod_flag_negative"].sum()) if "pi_lower_prod_flag_negative" in df.columns else 0,
        "negative_pi_upper_flags": int(df["pi_upper_prod_flag_negative"].sum()) if "pi_upper_prod_flag_negative" in df.columns else 0,
    }

    # Coverage from pre-computed values (observed production not in output CSV to avoid leakage)
    if coverage_rate is not None:
        report["empirical_production_coverage_80"] = coverage_rate
    if coverage_n is not None:
        report["coverage_observations"] = coverage_n

    # Coverage numerator/denominator/exclusions
    n_total = len(df)
    n_eligible = int((df["prod_pi_covered_80"] != pd.NA).sum()) if "prod_pi_covered_80" in df.columns else n_total
    n_covered = int((df["prod_pi_covered_80"] == True).sum()) if "prod_pi_covered_80" in df.columns else 0
    n_excluded = n_total - n_eligible
    report["coverage_numerator"] = n_covered
    report["coverage_denominator"] = n_eligible
    report["coverage_exclusions"] = n_excluded

    # By model breakdown
    if "model" in df.columns:
        model_stats = df.groupby("model").agg(
            n_obs=("predicted_production_1000_tons", "count"),
            mean_production=("predicted_production_1000_tons", "mean"),
            coverage_rate=("prod_pi_covered_80", "mean")
        ).reset_index()
        report["by_model"] = model_stats.to_dict(orient="records")

    return report


def main():
    """Run the Issue #51 production forecast pipeline."""
    # 1. Load data
    predictions_df = load_issue50_predictions(RESULTS_DIR)
    ml_dataset_df = load_ml_yield_dataset()

    # Load anomalies for baseline_mean if needed
    try:
        anomalies_df = load_yield_anomalies_forward_chaining()
    except FileNotFoundError:
        anomalies_df = None
        print("  Note: yield_anomalies_forward_chaining_v1.csv not available; "
              "baseline_mean will be taken from ML dataset join")

    # 2. Compute production forecast
    forecast_df, coverage_info = compute_production_forecast(
        predictions_df=predictions_df,
        ml_dataset_df=ml_dataset_df,
        anomalies_df=anomalies_df,
        area_proxy="historical_area"
    )

    # 3. Save output
    output_path = save_production_forecast(forecast_df, RESULTS_DIR, "production_forecast_v1.csv")

    # 4. Generate and print metrics report
    report = generate_metrics_report(
        forecast_df,
        coverage_n=coverage_info["coverage_observations"],
        coverage_rate=coverage_info["empirical_coverage"]
    )
    print("\n" + "=" * 70)
    print("Issue #51 Metrics Report")
    print("=" * 70)
    for key, value in report.items():
        print(f"  {key}: {value}")

    # 5. Print formula summary
    print("\n" + "=" * 70)
    print("Formula Summary")
    print("=" * 70)
    print("  predicted_yield_kg_per_ha  = baseline_mean_kg_per_ha + predicted_yield_anomaly_kg_per_ha")
    print("  predicted_production_1000_tons = area_1000_ha * predicted_yield_kg_per_ha / 1000")
    print("  pi_lower_prod_1000_tons = area_1000_ha * (baseline_mean_kg_per_ha + pi_lower_80) / 1000")
    print("  pi_upper_prod_1000_tons = area_1000_ha * (baseline_mean_kg_per_ha + pi_upper_80) / 1000")
    print("  Uncertainty scope: area_and_yield (Ridge/RF/XGBoost/Climatology) or yield_only (CatBoost)")
    print("  Labels: uncalibrated_in_sample_exploratory")

    return forecast_df


if __name__ == "__main__":
    result_df = main()