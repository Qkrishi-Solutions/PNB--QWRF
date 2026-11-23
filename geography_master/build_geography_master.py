"""QWeatherRiskF: Geography Master Builder (Issue #42)

Combines:
- address_expansion/pincode_reference_v1.csv (19,300 all-India pincodes)
- District Climate Zonal Clustering/district_zone_map_v1.csv (13 AP districts, climate clusters)
- PNB administrative banking hierarchy (Zone: Hyderabad, Circles: Tirupati, Vijayawada, Visakhapatnam)
- Cross-dataset district_code consistency validation against weather and yield feature engineering

Outputs:
- geography_master_v1.csv (Full pincode-grain geography master, 19,300 rows, nullable integers)
- district_geography_master_v1.csv (Canonical 13-district summary)
- geography_master_coverage_report_v1.csv (Join coverage, dataset audit, and mapping metrics)
- unmatched_ambiguous_pincodes_v1.csv (Audit of all ambiguous and unmapped pincodes)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "geography_master"

PINCODE_REF_PATH = PROJECT_ROOT / "address_expansion" / "pincode_reference_v1.csv"
CLIMATE_ZONE_PATH = PROJECT_ROOT / "District Climate Zonal Clustering" / "district_zone_map_v1.csv"

# Cross-dataset consistency targets
DATASETS_TO_CHECK = {
    "weather_anomalies_v1": PROJECT_ROOT / "weather_feature_engineering" / "weather_anomalies_v1.csv",
    "district_weather_normals_v1": PROJECT_ROOT / "weather_feature_engineering" / "district_weather_normals_v1.csv",
    "yield_anomalies_v1": PROJECT_ROOT / "yield_feature_engineering" / "yield_anomalies_v1.csv",
    "district_crop_yield_normals_v1": PROJECT_ROOT / "yield_feature_engineering" / "district_crop_yield_normals_v1.csv",
    "district_zone_map_v1": PROJECT_ROOT / "District Climate Zonal Clustering" / "district_zone_map_v1.csv",
    "pincode_reference_v1": PROJECT_ROOT / "address_expansion" / "pincode_reference_v1.csv",
}

# Provisional PNB administrative hierarchy (sourced from synthetic generator circle_of_district config)
# Note: Marked provisional until authoritative PNB bank branch circular is ingested.
PNB_ZONE_PROVISIONAL = "Hyderabad"
PNB_CIRCLE_MAP_PROVISIONAL = {
    502: "Tirupati",       # ANANTAPURAMU
    503: "Tirupati",       # CHITTOOR
    504: "Tirupati",       # YSR
    505: "Visakhapatnam",  # EAST GODAVARI
    506: "Vijayawada",     # GUNTUR
    510: "Vijayawada",     # KRISHNA
    511: "Tirupati",       # KURNOOL
    515: "Tirupati",       # SPR NELLORE
    517: "Vijayawada",     # PRAKASAM
    519: "Visakhapatnam",  # SRIKAKULAM
    520: "Visakhapatnam",  # VISAKHAPATNAM
    521: "Visakhapatnam",  # VIZIANAGARAM
    523: "Vijayawada",     # WEST GODAVARI
}

EXPECTED_DISTRICT_CODES = {
    502, 503, 504, 505, 506, 510, 511, 515, 517, 519, 520, 521, 523
}


def verify_cross_dataset_consistency(canonical_codes: set[int]) -> dict[str, dict]:
    """Verifies that district_code is consistent across every foundational dataset in the repository.
    
    Fails the build if any dataset has mismatched or unexpected district codes.
    """
    coverage_audit = {}
    print("\n--- Running Cross-Dataset District Consistency Check ---")

    for name, path in DATASETS_TO_CHECK.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dataset missing for cross-consistency check: {path}")

        df = pd.read_csv(path)
        if "district_code" not in df.columns:
            raise KeyError(f"Dataset {name} ({path.name}) is missing 'district_code' column!")

        # Extract unique valid integer district codes
        valid_codes = set(pd.to_numeric(df["district_code"], errors="coerce").dropna().astype(int).unique())

        diff = valid_codes ^ canonical_codes
        if diff:
            raise ValueError(
                f"Cross-dataset consistency failure on {name}! "
                f"Expected {canonical_codes}, but found symmetric difference: {diff}"
            )

        # Check district_name consistency if present
        if "district_name" in df.columns:
            name_mapping = df[["district_code", "district_name"]].drop_duplicates()
            # Ensure 1-to-1 mapping
            if name_mapping["district_code"].nunique() != len(name_mapping):
                raise ValueError(f"Inconsistent district names found within {name}: {name_mapping}")

        coverage_audit[name] = {
            "total_rows": len(df),
            "unique_districts": len(valid_codes),
            "canonical_match": True,
        }
        print(f"  [PASS] {name}: {len(df)} rows, 13/13 canonical district codes matched.")

    print("--- Cross-Dataset District Consistency Passed Successfully ---\n")
    return coverage_audit


def build_district_master(climate_path: Path) -> pd.DataFrame:
    if not climate_path.exists():
        raise FileNotFoundError(f"Missing climate zone map: {climate_path}")

    cz = pd.read_csv(climate_path)
    required_cols = {"district_code", "district_name", "cluster_id", "climate_zone"}
    if not required_cols.issubset(cz.columns):
        raise ValueError(f"Climate zone file missing required columns: {required_cols - set(cz.columns)}")

    cz["district_code"] = cz["district_code"].astype("Int64")
    cz_codes = set(cz["district_code"].dropna().astype(int).unique())
    if cz_codes != EXPECTED_DISTRICT_CODES:
        raise ValueError(f"Unexpected district codes in climate map: {cz_codes ^ EXPECTED_DISTRICT_CODES}")

    cz["pnb_zone_provisional"] = PNB_ZONE_PROVISIONAL
    cz["pnb_circle_provisional"] = cz["district_code"].astype(int).map(PNB_CIRCLE_MAP_PROVISIONAL)
    cz["state_code"] = 1
    cz["state_code"] = cz["state_code"].astype("Int64")
    cz["state_name"] = "ANDHRA PRADESH"
    cz["cluster_id"] = cz["cluster_id"].astype("Int64")

    cols_order = [
        "district_code",
        "district_name",
        "state_code",
        "state_name",
        "pnb_zone_provisional",
        "pnb_circle_provisional",
        "cluster_id",
        "climate_zone",
    ]
    return cz[cols_order].sort_values("district_code").reset_index(drop=True)


def build_geography_master(pincode_path: Path, district_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not pincode_path.exists():
        raise FileNotFoundError(f"Missing pincode reference: {pincode_path}")

    pin_df = pd.read_csv(pincode_path)
    initial_rows = len(pin_df)

    # Clean district_code for merge as nullable Int64
    pin_df["clean_district_code"] = pd.to_numeric(pin_df["district_code"], errors="coerce").astype("Int64")

    # Merge district-level metadata
    merged = pin_df.merge(
        district_df.rename(columns={"district_name": "canonical_district_name"}),
        left_on="clean_district_code",
        right_on="district_code",
        how="left",
        suffixes=("", "_dist"),
    )

    # Check for row multiplication
    if len(merged) != initial_rows:
        raise AssertionError(f"Row count changed during merge: {initial_rows} -> {len(merged)}")

    # Clean up columns and prioritize canonical fields
    merged["district_code"] = merged["clean_district_code"]
    merged.drop(columns=["clean_district_code", "district_code_dist"], inplace=True, errors="ignore")

    # Cast integer columns to nullable Int64 so they serialize as 502 instead of 502.0
    merged["pincode"] = merged["pincode"].astype("Int64")
    merged["state_code"] = merged["state_code"].astype("Int64")
    merged["district_code"] = merged["district_code"].astype("Int64")
    merged["cluster_id"] = merged["cluster_id"].astype("Int64")
    merged["n_source_districts"] = merged["n_source_districts"].astype("Int64")
    merged["n_states"] = merged["n_states"].astype("Int64")

    # Strict assertion: every 'mapped' row must have a district_code, circle, and climate_zone
    mapped_mask = merged["mapping_status"] == "mapped"
    mapped_rows = merged[mapped_mask]
    if mapped_rows["district_code"].isna().any():
        raise AssertionError("Found 'mapped' pincodes with null district_code!")
    if mapped_rows["pnb_circle_provisional"].isna().any():
        raise AssertionError("Found 'mapped' pincodes with null pnb_circle_provisional!")
    if mapped_rows["climate_zone"].isna().any():
        raise AssertionError("Found 'mapped' pincodes with null climate_zone!")

    # Strict assertion: ambiguous rows must NOT have a district_code
    ambiguous_mask = merged["mapping_status"].isin(["ambiguous_district", "ambiguous_state"])
    if merged[ambiguous_mask]["district_code"].notna().any():
        raise AssertionError("Found ambiguous pincodes with a guessed district_code!")

    # Final columns ordering
    master_cols = [
        "pincode",
        "mapping_status",
        "state_name",
        "state_code",
        "district_code",
        "canonical_district_name",
        "source_districts",
        "n_source_districts",
        "n_states",
        "pnb_zone_provisional",
        "pnb_circle_provisional",
        "cluster_id",
        "climate_zone",
    ]
    merged = merged[master_cols].sort_values(["pincode"]).reset_index(drop=True)

    # Ambiguous / Unmatched pincodes report
    ambiguous = merged[ambiguous_mask].copy()

    # District summary with pincode counts
    pincode_counts = (
        merged[mapped_mask]
        .groupby("district_code")
        .size()
        .to_dict()
    )
    district_summary = district_df.copy()
    district_summary["n_pincodes_mapped"] = district_summary["district_code"].map(pincode_counts).fillna(0).astype("Int64")

    return merged, district_summary, ambiguous


def generate_coverage_report(
    master_df: pd.DataFrame,
    district_summary: pd.DataFrame,
    ambiguous_df: pd.DataFrame,
    cross_dataset_audit: dict[str, dict]
) -> pd.DataFrame:
    total_pins = len(master_df)
    status_counts = master_df["mapping_status"].value_counts().to_dict()

    rows = [
        # Metric category: Pincode mapping
        {"category": "pincode_mapping", "metric": "total_pincodes_evaluated", "value": str(total_pins)},
        {"category": "pincode_mapping", "metric": "pincodes_mapped_to_ap_district", "value": str(int(status_counts.get("mapped", 0)))},
        {"category": "pincode_mapping", "metric": "pincodes_ambiguous_within_ap_district", "value": str(int(status_counts.get("ambiguous_district", 0)))},
        {"category": "pincode_mapping", "metric": "pincodes_ambiguous_border_state", "value": str(int(status_counts.get("ambiguous_state", 0)))},
        {"category": "pincode_mapping", "metric": "pincodes_outside_project_scope", "value": str(int(status_counts.get("outside_project_scope", 0)))},
        
        # Metric category: Administrative & Climate representation
        {"category": "administrative_coverage", "metric": "unique_project_districts_mapped", "value": str(int(master_df["district_code"].dropna().nunique()))},
        {"category": "administrative_coverage", "metric": "pnb_zones_represented", "value": str(int(master_df["pnb_zone_provisional"].dropna().nunique()))},
        {"category": "administrative_coverage", "metric": "pnb_circles_represented", "value": str(int(master_df["pnb_circle_provisional"].dropna().nunique()))},
        {"category": "administrative_coverage", "metric": "climate_zones_represented", "value": str(int(master_df["climate_zone"].dropna().nunique()))},
        {"category": "administrative_coverage", "metric": "min_pincodes_per_district", "value": str(int(district_summary["n_pincodes_mapped"].min()))},
        {"category": "administrative_coverage", "metric": "max_pincodes_per_district", "value": str(int(district_summary["n_pincodes_mapped"].max()))},
        {"category": "administrative_coverage", "metric": "mean_pincodes_per_district", "value": f"{district_summary['n_pincodes_mapped'].mean():.2f}"},
    ]

    # Metric category: Cross-dataset consistency audit
    for dataset_name, audit in cross_dataset_audit.items():
        rows.append({
            "category": "cross_dataset_consistency",
            "metric": f"dataset_{dataset_name}_district_match",
            "value": f"13/13 canonical districts (rows={audit['total_rows']})",
        })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run cross-dataset consistency verification across weather, yield, and climate files
    cross_dataset_audit = verify_cross_dataset_consistency(EXPECTED_DISTRICT_CODES)

    # 2. Build canonical district master
    print("Building canonical district master...")
    district_df = build_district_master(CLIMATE_ZONE_PATH)
    print(f"  Loaded {len(district_df)} districts.")

    # 3. Build full geography master
    print("Building geography master from pincode reference...")
    master_df, district_summary, ambiguous_df = build_geography_master(PINCODE_REF_PATH, district_df)
    print(f"  Processed {len(master_df)} pincodes.")

    # 4. Generate coverage and cross-dataset consistency report
    cov_report = generate_coverage_report(master_df, district_summary, ambiguous_df, cross_dataset_audit)

    # 5. Save all deliverables
    master_path = OUTPUT_DIR / "geography_master_v1.csv"
    dist_summary_path = OUTPUT_DIR / "district_geography_master_v1.csv"
    cov_path = OUTPUT_DIR / "geography_master_coverage_report_v1.csv"
    ambiguous_path = OUTPUT_DIR / "unmatched_ambiguous_pincodes_v1.csv"

    master_df.to_csv(master_path, index=False)
    district_summary.to_csv(dist_summary_path, index=False)
    cov_report.to_csv(cov_path, index=False)
    ambiguous_df.to_csv(ambiguous_path, index=False)

    print("\n=== SUCCESS: ALL DELIVERABLES GENERATED ===")
    print(f"1. Geography Master: {master_path} ({len(master_df)} rows)")
    print(f"2. District Summary: {dist_summary_path} ({len(district_summary)} rows)")
    print(f"3. Coverage Report:  {cov_path} ({len(cov_report)} metrics)")
    print(f"4. Ambiguous Audit:  {ambiguous_path} ({len(ambiguous_df)} rows)")


if __name__ == "__main__":
    main()
