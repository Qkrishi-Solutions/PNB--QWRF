#!/usr/bin/env python3
"""
build_ml_dataset.py - Issue #50 / Issue #7 ML-Ready Dataset Generator

This script compiles the canonical agricultural, yield anomaly, weather anomaly,
district climate zone, and crop eligibility datasets into a unified ML-ready table.

Inputs:
  - CropLongFormatv1/crop_long_v1.csv
  - yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv
  - weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv
  - District Climate Zonal Clustering/district_zone_map_v1.csv
  - CropModelEligibility/crop_model_eligibility_v1.csv

Output:
  - yield_forecast/data/ml_yield_dataset_v1.csv

Key guarantees:
  - Deterministic joins with key uniqueness assertions before and after.
  - Strict preservation of all 1,430 canonical rows (no silent dropping).
  - Explicit boolean indicators for modeling eligibility and target validity.
"""

import argparse
import hashlib
import os
import sys
import pandas as pd
import numpy as np


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(
    crop_long_path: str = "CropLongFormatv1/crop_long_v1.csv",
    yield_fc_path: str = "yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv",
    weather_fc_path: str = "weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv",
    climate_zone_path: str = "District Climate Zonal Clustering/district_zone_map_v1.csv",
    crop_eligibility_path: str = "CropModelEligibility/crop_model_eligibility_v1.csv",
    output_path: str = "yield_forecast/data/ml_yield_dataset_v1.csv"
) -> pd.DataFrame:
    print("=" * 70)
    print("Issue #50: Building Integrated ML-Ready Yield Dataset")
    print("=" * 70)

    # 1. Verify existence of inputs
    for name, path in [
        ("Crop Long", crop_long_path),
        ("Yield Forward Chaining", yield_fc_path),
        ("Weather Forward Chaining", weather_fc_path),
        ("Climate Zones", climate_zone_path),
        ("Crop Eligibility", crop_eligibility_path),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {name} at {path}")
        print(f"[FOUND] {name}: {path} ({compute_sha256(path)[:12]}...)")

    # 2. Load primary agricultural tables
    df_crop_long = pd.read_csv(crop_long_path)
    df_yield_fc = pd.read_csv(yield_fc_path)
    df_climate = pd.read_csv(climate_zone_path)
    df_eligibility = pd.read_csv(crop_eligibility_path)
    df_weather_fc = pd.read_csv(weather_fc_path)

    print(f"\nInitial row counts:")
    print(f"  Crop long: {len(df_crop_long)} rows")
    print(f"  Yield forward chaining: {len(df_yield_fc)} rows")
    print(f"  Climate zones: {len(df_climate)} rows")
    print(f"  Crop eligibility: {len(df_eligibility)} rows")
    print(f"  Weather forward chaining (long): {len(df_weather_fc)} rows")

    # 3. Key Uniqueness Checks
    assert not df_crop_long.duplicated(subset=['district_code', 'crop', 'year']).any(), \
        "Duplicate keys in crop_long_v1.csv"
    assert not df_yield_fc.duplicated(subset=['district_code', 'crop', 'year']).any(), \
        "Duplicate keys in yield_anomalies_forward_chaining_v1.csv"
    assert not df_climate.duplicated(subset=['district_code']).any(), \
        "Duplicate district_code in district_zone_map_v1.csv"
    assert not df_eligibility.duplicated(subset=['crop']).any(), \
        "Duplicate crop in crop_model_eligibility_v1.csv"
    assert not df_weather_fc.duplicated(subset=['district_code', 'year', 'weather_column']).any(), \
        "Duplicate keys in weather_anomalies_forward_chaining_v1.csv"

    # 4. Deterministic Weather Long-to-Wide Pivot
    print("\nPivoting long-format weather forward-chaining anomalies into wide matrix...")
    weather_pivot_anom = df_weather_fc.pivot(
        index=['district_code', 'year'],
        columns='weather_column',
        values='anomaly'
    )
    weather_pivot_anom.columns = [f"wanom__{c}" for c in weather_pivot_anom.columns]
    weather_pivot_anom = weather_pivot_anom.reset_index()

    weather_pivot_act = df_weather_fc.pivot(
        index=['district_code', 'year'],
        columns='weather_column',
        values='actual_value'
    )
    weather_pivot_act.columns = [f"wact__{c}" for c in weather_pivot_act.columns]
    weather_pivot_act = weather_pivot_act.reset_index()

    weather_wide = pd.merge(weather_pivot_anom, weather_pivot_act, on=['district_code', 'year'], how='inner')
    assert len(weather_wide) == 65, f"Expected 65 weather rows, got {len(weather_wide)}"
    print(f"Weather wide matrix created: {weather_wide.shape} (65 district-years, 156 weather indicators)")

    # 5. Base Table: Start from df_yield_fc (1,430 rows)
    base_rows = len(df_yield_fc)
    df_merged = df_yield_fc.copy()

    # Join Crop Eligibility
    elig_cols = ['crop', 'eligibility_class', 'proceed_to_modelling', 'modelling_role', 'reasons', 'usable_districts']
    df_merged = pd.merge(df_merged, df_eligibility[elig_cols].rename(columns={'reasons': 'eligibility_reasons'}), on='crop', how='left')
    assert len(df_merged) == base_rows, f"Row count changed after eligibility join: {len(df_merged)}"

    # Join Climate Zones
    clim_cols = ['district_code', 'cluster_id', 'climate_zone']
    df_merged = pd.merge(df_merged, df_climate[clim_cols], on='district_code', how='left')
    assert len(df_merged) == base_rows, f"Row count changed after climate zone join: {len(df_merged)}"

    # Join Pivoted Weather Matrix
    df_merged = pd.merge(df_merged, weather_wide, on=['district_code', 'year'], how='left')
    assert len(df_merged) == base_rows, f"Row count changed after weather join: {len(df_merged)}"

    # 6. Add Modeling Status and Validity Indicators
    df_merged['is_eligible_crop'] = df_merged['proceed_to_modelling'].astype(bool)
    df_merged['has_valid_yield_anomaly'] = df_merged['yield_anomaly_kg_per_ha'].notna()
    df_merged['usable_for_forward_chaining_eval'] = df_merged['is_eligible_crop'] & df_merged['has_valid_yield_anomaly']

    # Sort deterministically
    df_merged.sort_values(by=['crop', 'district_code', 'year'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # 7. Write to Output Path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    output_sha256 = compute_sha256(output_path)

    print("\n" + "=" * 70)
    print("Integrated ML Dataset Successfully Generated")
    print(f"Destination: {output_path}")
    print(f"Rows: {df_merged.shape[0]}")
    print(f"Columns: {df_merged.shape[1]}")
    print(f"SHA-256: {output_sha256}")
    print(f"Summary breakdown:")
    print(f"  - Total records: {len(df_merged)}")
    print(f"  - Eligible crop records: {df_merged['is_eligible_crop'].sum()} ({df_merged['is_eligible_crop'].mean()*100:.1f}%)")
    print(f"  - Records with valid forward yield anomaly: {df_merged['has_valid_yield_anomaly'].sum()}")
    print(f"  - Usable records for evaluation (eligible & valid): {df_merged['usable_for_forward_chaining_eval'].sum()}")
    print("=" * 70)

    return df_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issue #50 ML-Ready Dataset Generator")
    parser.add_argument("--crop-long", default="CropLongFormatv1/crop_long_v1.csv")
    parser.add_argument("--yield-fc", default="yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv")
    parser.add_argument("--weather-fc", default="weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv")
    parser.add_argument("--climate-zone", default="District Climate Zonal Clustering/district_zone_map_v1.csv")
    parser.add_argument("--crop-eligibility", default="CropModelEligibility/crop_model_eligibility_v1.csv")
    parser.add_argument("--output", default="yield_forecast/data/ml_yield_dataset_v1.csv")
    args = parser.parse_args()

    build_dataset(
        crop_long_path=args.crop_long,
        yield_fc_path=args.yield_fc,
        weather_fc_path=args.weather_fc,
        climate_zone_path=args.climate_zone,
        crop_eligibility_path=args.crop_eligibility,
        output_path=args.output
    )
