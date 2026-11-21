"""Unit and Integration Tests for Geography Master (Issue #42).

Validates:
1. district_code uniqueness and consistency across datasets
2. Zero row multiplication (19,300 pincodes preserved exactly)
3. Unmatched/ambiguous keys properly documented without guessing
4. Banking circles and climate zone consistency
"""
import unittest
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
GEO_MASTER_PATH = REPO_ROOT / "geography_master" / "geography_master_v1.csv"
DIST_MASTER_PATH = REPO_ROOT / "geography_master" / "district_geography_master_v1.csv"
COV_REPORT_PATH = REPO_ROOT / "geography_master" / "geography_master_coverage_report_v1.csv"
AMBIGUOUS_PATH = REPO_ROOT / "geography_master" / "unmatched_ambiguous_pincodes_v1.csv"

EXPECTED_DISTRICTS = {502, 503, 504, 505, 506, 510, 511, 515, 517, 519, 520, 521, 523}


class TestGeographyMaster(unittest.TestCase):

    def test_files_exist(self):
        self.assertTrue(GEO_MASTER_PATH.exists(), "Missing geography_master_v1.csv")
        self.assertTrue(DIST_MASTER_PATH.exists(), "Missing district_geography_master_v1.csv")
        self.assertTrue(COV_REPORT_PATH.exists(), "Missing geography_master_coverage_report_v1.csv")
        self.assertTrue(AMBIGUOUS_PATH.exists(), "Missing unmatched_ambiguous_pincodes_v1.csv")

    def test_district_master_completeness(self):
        df = pd.read_csv(DIST_MASTER_PATH)
        self.assertEqual(len(df), 13, f"Expected 13 districts, got {len(df)}")
        self.assertEqual(set(df["district_code"].unique()), EXPECTED_DISTRICTS)
        self.assertTrue(df["district_code"].is_unique, "Duplicate district_code found in district master")
        self.assertEqual(set(df["pnb_zone"].unique()), {"Hyderabad"})
        self.assertEqual(set(df["pnb_circle"].unique()), {"Tirupati", "Vijayawada", "Visakhapatnam"})
        self.assertEqual(set(df["climate_zone"].unique()), {"Climate Zone 1", "Climate Zone 2", "Climate Zone 3"})
        self.assertTrue((df["n_pincodes_mapped"] > 0).all(), "Every district must have at least one mapped pincode")

    def test_geography_master_row_count(self):
        df = pd.read_csv(GEO_MASTER_PATH)
        self.assertEqual(len(df), 19300, f"Expected 19,300 rows, got {len(df)} (row multiplication detected!)")
        self.assertTrue(df["pincode"].is_unique, "Pincode column is not unique in geography_master_v1")

    def test_mapping_status_and_consistency(self):
        df = pd.read_csv(GEO_MASTER_PATH)
        mapped = df[df["mapping_status"] == "mapped"]
        self.assertEqual(len(mapped), 1207, f"Expected 1,207 mapped pincodes, got {len(mapped)}")
        self.assertTrue(mapped["district_code"].notna().all())
        self.assertEqual(set(mapped["district_code"].astype(int).unique()), EXPECTED_DISTRICTS)

        # Mapped rows must have circle, zone, climate_zone populated
        self.assertTrue(mapped["pnb_zone"].notna().all())
        self.assertTrue(mapped["pnb_circle"].notna().all())
        self.assertTrue(mapped["climate_zone"].notna().all())

        # Ambiguous rows must NOT guess a district_code
        ambiguous = df[df["mapping_status"].isin(["ambiguous_district", "ambiguous_state"])]
        self.assertEqual(len(ambiguous), 64, f"Expected 64 ambiguous pincodes, got {len(ambiguous)}")
        self.assertTrue(ambiguous["district_code"].isna().all(), "Ambiguous pincodes must not have a guessed district_code")

    def test_coverage_report_integrity(self):
        cov = pd.read_csv(COV_REPORT_PATH).set_index("metric")["value"].to_dict()
        self.assertEqual(cov["total_pincodes_evaluated"], 19300)
        self.assertEqual(cov["pincodes_mapped_to_ap_district"], 1207)
        self.assertEqual(cov["pincodes_ambiguous_within_ap_district"], 18)
        self.assertEqual(cov["pincodes_ambiguous_border_state"], 46)
        self.assertEqual(cov["pincodes_outside_project_scope"], 18029)
        self.assertEqual(cov["unique_project_districts_mapped"], 13)
        self.assertEqual(cov["pnb_circles_represented"], 3)
        self.assertEqual(cov["climate_zones_represented"], 3)


if __name__ == "__main__":
    unittest.main()
