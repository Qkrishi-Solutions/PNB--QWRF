"""QWeatherRiskF: Geography Master Builder (Issue #42)

Combines:
- address_expansion/pincode_reference_v1.csv (19,300 all-India pincodes)
- District Climate Zonal Clustering/district_zone_map_v1.csv (13 AP districts, climate clusters)
- PNB administrative banking hierarchy (Zone: Hyderabad, Circles: Tirupati, Vijayawada, Visakhapatnam)
- Canonical crop-weather district codes (502 to 523)

Outputs:
- geography_master_v1.csv (Full pincode-grain geography master, 19,300 rows)
- district_geography_master_v1.csv (Canonical 13-district summary)
- geography_master_coverage_report_v1.csv (Join coverage and mapping metrics)
- unmatched_ambiguous_pincodes_v1.csv (Audit of all ambiguous and unmapped pincodes)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "geography_master"

PINCODE_REF_PATH = PROJECT_ROOT / "address_expansion" / "pincode_reference_v1.csv"
CLIMATE_ZONE_PATH = PROJECT_ROOT / "District Climate Zonal Clustering" / "district_zone_map_v1.csv"

PNB_ZONE = "Hyderabad"
PNB_CIRCLE_MAP = {
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


def build_district_master(climate_path: Path) -> pd.DataFrame:
    if not climate_path.exists():
        raise FileNotFoundError(f"Missing climate zone map: {climate_path}")

    cz = pd.read_csv(climate_path)
    required_cols = {"district_code", "district_name", "cluster_id", "climate_zone"}
    if not required_cols.issubset(cz.columns):
        raise ValueError(f"Climate zone file missing required columns: {required_cols - set(cz.columns)}")

    cz["district_code"] = cz["district_code"].astype(int)
    cz_codes = set(cz["district_code"].unique())
    if cz_codes != EXPECTED_DISTRICT_CODES:
        raise ValueError(f"Unexpected district codes in climate map: {cz_codes ^ EXPECTED_DISTRICT_CODES}")

    cz["pnb_zone"] = PNB_ZONE
    cz["pnb_circle"] = cz["district_code"].map(PNB_CIRCLE_MAP)
    cz["state_code"] = 1
    cz["state_name"] = "ANDHRA PRADESH"

    cols_order = [
        "district_code",
        "district_name",
        "state_code",
        "state_name",
        "pnb_zone",
        "pnb_circle",
        "cluster_id",
        "climate_zone",
    ]
    return cz[cols_order].sort_values("district_code").reset_index(drop=True)


def build_geography_master(pincode_path: Path, district_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not pincode_path.exists():
        raise FileNotFoundError(f"Missing pincode reference: {pincode_path}")

    pin_df = pd.read_csv(pincode_path)
    initial_rows = len(pin_df)

    # Clean district_code for merge
    pin_df["clean_district_code"] = pd.to_numeric(pin_df["district_code"], errors="coerce")

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
        "pnb_zone",
        "pnb_circle",
        "cluster_id",
        "climate_zone",
    ]
    merged = merged[master_cols].sort_values(["pincode"]).reset_index(drop=True)

    # Ambiguous / Unmatched pincodes report
    ambiguous = merged[merged["mapping_status"].isin(["ambiguous_district", "ambiguous_state"])].copy()

    # District summary with pincode counts
    pincode_counts = (
        merged[merged["mapping_status"] == "mapped"]
        .groupby("district_code")
        .size()
        .to_dict()
    )
    district_summary = district_df.copy()
    district_summary["n_pincodes_mapped"] = district_summary["district_code"].map(pincode_counts).fillna(0).astype(int)

    return merged, district_summary, ambiguous


def generate_coverage_report(master_df: pd.DataFrame, district_summary: pd.DataFrame, ambiguous_df: pd.DataFrame) -> pd.DataFrame:
    total_pins = len(master_df)
    status_counts = master_df["mapping_status"].value_counts().to_dict()

    rows = [
        {"metric": "total_pincodes_evaluated", "value": total_pins},
        {"metric": "pincodes_mapped_to_ap_district", "value": int(status_counts.get("mapped", 0))},
        {"metric": "pincodes_ambiguous_within_ap_district", "value": int(status_counts.get("ambiguous_district", 0))},
        {"metric": "pincodes_ambiguous_border_state", "value": int(status_counts.get("ambiguous_state", 0))},
        {"metric": "pincodes_outside_project_scope", "value": int(status_counts.get("outside_project_scope", 0))},
        {"metric": "unique_project_districts_mapped", "value": int(master_df["district_code"].dropna().nunique())},
        {"metric": "pnb_zones_represented", "value": int(master_df["pnb_zone"].dropna().nunique())},
        {"metric": "pnb_circles_represented", "value": int(master_df["pnb_circle"].dropna().nunique())},
        {"metric": "climate_zones_represented", "value": int(master_df["climate_zone"].dropna().nunique())},
        {"metric": "min_pincodes_per_district", "value": int(district_summary["n_pincodes_mapped"].min())},
        {"metric": "max_pincodes_per_district", "value": int(district_summary["n_pincodes_mapped"].max())},
        {"metric": "mean_pincodes_per_district", "value": float(district_summary["n_pincodes_mapped"].mean())},
    ]
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building canonical district master...")
    district_df = build_district_master(CLIMATE_ZONE_PATH)
    print(f"  Loaded {len(district_df)} districts.")

    print("Building geography master from pincode reference...")
    master_df, district_summary, ambiguous_df = build_geography_master(PINCODE_REF_PATH, district_df)
    print(f"  Processed {len(master_df)} pincodes.")

    cov_report = generate_coverage_report(master_df, district_summary, ambiguous_df)

    # Save all deliverables
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
