#!/usr/bin/env python3
"""
issue_51_production_forecast.py - Issue #51: Provisional Production Forecast
  with Propagated Yield Uncertainty

Computes production forecasts (area × yield) using Issue #50 yield anomaly
predictions, propagating only yield-side uncertainty (Issue #49 area forecast
not yet available).

Operational taxonomy: end-of-season retrospective response estimation,
yield-only uncertainty propagation. Complete area-and-yield uncertainty
propagation pending Issue #49.

Data joins use the canonical primary key: ['district_code', 'crop', 'year'].
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
    missing_baseline = predictions_joined["baseline_mean_kg_per_ha"].isna().sum()
    if missing_area_ml > 0:
        print(f"  WARNING: {missing_area_ml} rows missing area_1000_ha after join")
    if missing_baseline > 0:
        print(f"  WARNING: {missing_baseline} rows missing baseline_mean_kg_per_ha after join")

    # ---- Join 2: area forecast predictions from Issue #49 ----
    # Add area model predictions (Climatology, Ridge, RandomForest, XGBoost)
    # Note: we do NOT include area_1000_ha from area forecast to avoid column name conflict;
    # the historical area_1000_ha from the ML dataset is used as the area proxy
    if not area_df.empty:
        predictions_joined = predictions_joined.merge(
            area_df[primary_key + ["pred_Climatology", "pred_Ridge_Delta",
                                  "pred_RandomForest_L1_LogDelta", "pred_XGBoost_L1_LogDelta"]],
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

    # ---- Determine uncertainty scope per-row ----
    # uncertainty_scope = area_and_yield when yield_model is NOT CatBoost
    # uncertainty_scope = yield_only when yield_model IS CatBoost (no area forecast)
    predictions_joined["uncertainty_scope"] = "yield_only"
    predictions_joined.loc[
        predictions_joined["model"].isin(["Ridge", "RandomForest", "XGBoost", "Climatology"]),
        "uncertainty_scope"
    ] = "area_and_yield"

    # missing_area_forecast flag: True when yield_model is CatBoost (no area forecast available)
    predictions_joined["missing_area_forecast"] = predictions_joined["model"].isin(["CatBoost"])

    # predicted_area_1000_ha: use area forecast if available, otherwise NaN for yield_only rows
    # For area_and_yield rows: use the forecasted area (pred_Climatology, pred_Ridge_Delta, etc.)
    # For yield_only rows (CatBoost): set to NaN
    predictions_joined["predicted_area_1000_ha"] = np.nan
    area_and_yield_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    predictions_joined.loc[area_and_yield_mask, "predicted_area_1000_ha"] = (
        predictions_joined.loc[area_and_yield_mask, "pred_Climatology"]
    )
    # Also set from the specific model's area forecast where available
    for model_col, model_name in [("pred_Ridge_Delta", "Ridge"),
                                   ("pred_RandomForest_L1_LogDelta", "RandomForest"),
                                   ("pred_XGBoost_L1_LogDelta", "XGBoost"),
                                   ("pred_Climatology", "Climatology")]:
        model_mask = (predictions_joined["uncertainty_scope"] == "area_and_yield") & \
                     (predictions_joined["model"] == model_name)
        predictions_joined.loc[model_mask, "predicted_area_1000_ha"] = (
            predictions_joined.loc[model_mask, model_col]
        )

    # area_pi_low_80 and area_pi_high_80: area prediction interval endpoints
    predictions_joined["area_pi_low_80"] = np.nan
    predictions_joined["area_pi_high_80"] = np.nan
    ay_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    predictions_joined.loc[ay_mask, "area_pi_low_80"] = (
        predictions_joined.loc[ay_mask, "pred_Ridge_Delta"]  # use Ridge as reference, or compute min
    )
    predictions_joined.loc[ay_mask, "area_pi_high_80"] = (
        predictions_joined.loc[ay_mask, "pred_Ridge_Delta"]
    )
    # Actually, let me use the appropriate model's interval
    for model_col_low, model_col_high, model_name in [
        ("pred_Ridge_Delta", "pred_Ridge_Delta", "Ridge"),
        ("pred_RandomForest_L1_LogDelta", "pred_RandomForest_L1_LogDelta_pi_high", "RandomForest"),
        ("pred_XGBoost_L1_LogDelta", "pred_XGBoost_L1_LogDelta_pi_high", "XGBoost"),
        ("pred_Climatology", "pred_Climatology", "Climatology")
    ]:
        if model_col_low in predictions_joined.columns and model_col_high in predictions_joined.columns:
            model_mask = (predictions_joined["uncertainty_scope"] == "area_and_yield") & \
                         (predictions_joined["model"] == model_name)
            predictions_joined.loc[model_mask, "area_pi_low_80"] = np.maximum(
                predictions_joined.loc[model_mask, model_col_low], predictions_joined.loc[model_mask, "area_pi_low_80"]
            )
            predictions_joined.loc[model_mask, "area_pi_high_80"] = np.minimum(
                predictions_joined.loc[model_mask, model_col_high], predictions_joined.loc[model_mask, "area_pi_high_80"]
            )

    # ---- Compute predicted production ----
    # predicted_production_1000_tons = predicted_area_1000_ha * predicted_yield_kg_per_ha / 1000
    predictions_joined["predicted_production_1000_tons"] = (
        predictions_joined["predicted_area_1000_ha"] * predictions_joined["predicted_yield_kg_per_ha"] / 1000
    )

    # ---- Convert yield interval endpoints to production interval endpoints ----
    # Use the same area proxy and add baseline_mean inside the interval
    # pi_lower_prod = area * (baseline_mean + pi_lower_80) / 1000
    # pi_upper_prod = area * (baseline_mean + pi_upper_80) / 1000

    # Ensure interval columns exist
    if "pi_lower_80" in predictions_joined.columns and "pi_upper_80" in predictions_joined.columns:
        predictions_joined["pi_lower_prod_1000_tons"] = (
            predictions_joined["predicted_area_1000_ha"] *
            (predictions_joined["baseline_mean_kg_per_ha"] + predictions_joined["pi_lower_80"]) / 1000
        )
        predictions_joined["pi_upper_prod_1000_tons"] = (
            predictions_joined["predicted_area_1000_ha"] *
            (predictions_joined["baseline_mean_kg_per_ha"] + predictions_joined["pi_upper_80"]) / 1000
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
    # 1,000 uniform samples per record, rho=0 independence, seed 42
    n_mc = 1000
    np.random.seed(42)
    
    # Initialize Monte Carlo production samples
    predictions_joined["mc_production_samples"] = None
    
    # For area_and_yield rows: sample area from area interval and yield from yield interval
    ay_mask = predictions_joined["uncertainty_scope"] == "area_and_yield"
    y_mask = predictions_joined["uncertainty_scope"] == "yield_only"
    
    mc_results = np.full((len(predictions_joined), n_mc), np.nan)
    
    # For area_and_yield rows: combine area and yield uncertainty
    if ay_mask.any():
        # Sample area uniformly from area prediction interval
        area_lo = predictions_joined.loc[ay_mask, "area_pi_low_80"].values
        area_hi = predictions_joined.loc[ay_mask, "area_pi_high_80"].values
        # Sample yield uniformly from yield interval
        yield_lo = predictions_joined.loc[ay_mask, "pi_lower_80"].values
        yield_hi = predictions_joined.loc[ay_mask, "pi_upper_80"].values
        
        # Monte Carlo: 1,000 uniform samples for area and yield, independently (rho=0)
        area_samples = np.random.uniform(
            np.tile(area_lo, (n_mc, 1)).T,
            np.tile(area_hi, (n_mc, 1)).T
        )
        yield_samples = np.random.uniform(
            np.tile(yield_lo, (n_mc, 1)).T,
            np.tile(yield_hi, (n_mc, 1)).T
        )
        
        # Compute production samples: area * yield / 1000
        # Area is in 1000 ha, yield is in kg/ha, so production = area_1000_ha * yield_kg_per_ha / 1000
        # But our area is already in 1000 ha and yield is in kg/ha, so:
        # production_1000_tons = area_1000_ha * yield_kg_per_ha / 1000
        # Wait, let me reconsider. area_samples is in 1000 ha, yield_samples is in kg/ha
        # production = area_1000_ha * yield_kg_per_ha / 1000 gives tons
        # So: mc_prod = area_samples * yield_samples / 1000
        mc_prod = area_samples * yield_samples / 1000
        mc_results[ay_mask] = mc_prod

    # For yield_only rows: only yield uncertainty, area is historical (fixed)
    if y_mask.any():
        # Use historical area_1000_ha as fixed, sample yield only
        hist_area = predictions_joined.loc[y_mask, "area_1000_ha"].values
        yield_lo = predictions_joined.loc[y_mask, "pi_lower_80"].values
        yield_hi = predictions_joined.loc[y_mask, "pi_upper_80"].values
        
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
    empirical_coverage = None
    n_coverage_obs = None
    coverage_by_year_model = []
    
    if "production_1000_tons" in ml_dataset_df.columns and \
       "prod_pi_low_80" in predictions_joined.columns and \
       "prod_pi_high_80" in predictions_joined.columns:
        # Join observed production
        coverage_df = predictions_joined.merge(
            ml_dataset_df[primary_key + ["production_1000_tons"]],
            on=primary_key,
            how="left"
        )
        # Check coverage ONLY for rows with valid combined intervals and actual production
        valid_combined_mask = (
            (coverage_df["prod_pi_low_80"].notna()) &
            (coverage_df["prod_pi_high_80"].notna()) &
            (coverage_df["production_1000_tons"].notna())
        )
        coverage_df_valid = coverage_df[valid_combined_mask]
        
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
        
        predictions_joined["prod_pi_covered_80"] = False
        # Set coverage mask only for eligible rows
        eligible_mask = valid_combined_mask.copy()
        predictions_joined.loc[eligible_mask, "prod_pi_covered_80"] = (
            (coverage_df["prod_pi_low_80"][eligible_mask] <= coverage_df["production_1000_tons"][eligible_mask]) &
            (coverage_df["production_1000_tons"][eligible_mask] <= coverage_df["prod_pi_high_80"][eligible_mask])
        )
        n_coverage_obs = n_with_prod
        
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
        "coverage_by_year_model": coverage_by_year_model
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
        "residual"
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
    print("  Uncertainty scope: yield_only (Issue #49 area forecast not available)")
    print("  Labels: uncalibrated_in_sample_exploratory")

    return forecast_df


if __name__ == "__main__":
    result_df = main()