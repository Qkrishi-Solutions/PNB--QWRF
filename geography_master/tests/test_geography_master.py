"""Unit and Integration Tests for Geography Master (Issue #42).

Validates:
1. district_code uniqueness and consistency across datasets (weather, yield, climate zones, pincodes)
2. Zero row multiplication (19,300 pincodes preserved exactly)
3. Unmatched/ambiguous keys properly documented without guessing
4. Banking circles and climate zone consistency
5. Integer serialisation (no float formatted strings like 502.0)
6. Strict assertion: every mapped row has circle and climate zone
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
        self.assertEqual(set(df["pnb_zone_provisional"].unique()), {"Hyderabad"})
        self.assertEqual(set(df["pnb_circle_provisional"].unique()), {"Tirupati", "Vijayawada", "Visakhapatnam"})
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
        self.assertTrue(mapped["pnb_zone_provisional"].notna().all(), "All mapped rows must have pnb_zone_provisional")
        self.assertTrue(mapped["pnb_circle_provisional"].notna().all(), "All mapped rows must have pnb_circle_provisional")
        self.assertTrue(mapped["climate_zone"].notna().all(), "All mapped rows must have climate_zone")

        # Ambiguous rows must NOT guess a district_code
        ambiguous = df[df["mapping_status"].isin(["ambiguous_district", "ambiguous_state"])]
        self.assertEqual(len(ambiguous), 64, f"Expected 64 ambiguous pincodes, got {len(ambiguous)}")
        self.assertTrue(ambiguous["district_code"].isna().all(), "Ambiguous pincodes must not have a guessed district_code")

    def test_integer_formatting_in_csv(self):
        # Read raw lines to verify integer formatting (no trailing .0)
        with open(GEO_MASTER_PATH) as f:
            for line in f:
                if ",mapped," in line:
                    parts = line.strip().split(",")
                    # parts: pincode,mapping_status,state_name,state_code,district_code,...
                    pincode, status, state, state_code, dist_code = parts[:5]
                    self.assertNotIn(".0", pincode, f"pincode should be integer formatted, got {pincode}")
                    self.assertNotIn(".0", state_code, f"state_code should be integer formatted, got {state_code}")
                    self.assertNotIn(".0", dist_code, f"district_code should be integer formatted, got {dist_code}")
                    break

        with open(DIST_MASTER_PATH) as f:
            header = f.readline().strip()
            first_row = f.readline().strip().split(",")
            self.assertNotIn(".0", first_row[0], f"district_code in district master should be integer, got {first_row[0]}")
            self.assertNotIn(".0", first_row[2], f"state_code in district master should be integer, got {first_row[2]}")
            self.assertNotIn(".0", first_row[6], f"cluster_id in district master should be integer, got {first_row[6]}")

    def test_coverage_report_integrity(self):
        cov = pd.read_csv(COV_REPORT_PATH).set_index("metric")["value"].to_dict()
        self.assertEqual(int(cov["total_pincodes_evaluated"]), 19300)
        self.assertEqual(int(cov["pincodes_mapped_to_ap_district"]), 1207)
        self.assertEqual(int(cov["pincodes_ambiguous_within_ap_district"]), 18)
        self.assertEqual(int(cov["pincodes_ambiguous_border_state"]), 46)
        self.assertEqual(int(cov["pincodes_outside_project_scope"]), 18029)
        self.assertEqual(int(cov["unique_project_districts_mapped"]), 13)
        self.assertEqual(int(cov["pnb_circles_represented"]), 3)
        self.assertEqual(int(cov["climate_zones_represented"]), 3)
        # Check cross-dataset consistency metrics
        self.assertIn("dataset_weather_anomalies_v1_district_match", cov)
        self.assertIn("dataset_yield_anomalies_v1_district_match", cov)
        self.assertTrue(cov["dataset_weather_anomalies_v1_district_match"].startswith("13/13"))


if __name__ == "__main__":
    unittest.main()
