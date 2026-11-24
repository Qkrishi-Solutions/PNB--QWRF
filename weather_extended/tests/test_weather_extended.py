"""
Unit tests for Extended Meteorological Ingestion (Issue #45 / Module W1)
Asserts schema compliance, primary key uniqueness, physical bounds,
and zero silent imputation for inland sea surface temperatures.
"""

import os
import unittest
import numpy as np
import pandas as pd

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(MODULE_DIR, "weather_extended_v1.csv")
REPORT_PATH = os.path.join(MODULE_DIR, "weather_extended_coverage_report_v1.csv")

CANONICAL_13_DISTRICT_CODES = {
    502, 503, 504, 505, 506, 510, 511, 515, 517, 519, 520, 521, 523
}

INLAND_DISTRICT_CODES = {502, 503, 504, 511}
COASTAL_DISTRICT_CODES = CANONICAL_13_DISTRICT_CODES - INLAND_DISTRICT_CODES


class TestWeatherExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CSV_PATH)
        cls.report = pd.read_csv(REPORT_PATH)

    def test_files_exist(self):
        self.assertTrue(os.path.exists(CSV_PATH), "weather_extended_v1.csv does not exist")
        self.assertTrue(os.path.exists(REPORT_PATH), "weather_extended_coverage_report_v1.csv does not exist")

    def test_schema_exact(self):
        expected_cols = [
            "district_code", "district_name", "date", "era5_grid_cells",
            "msl_min_hpa", "msl_mean_hpa", "wind_speed_mean_kmh",
            "wind_speed_max_kmh", "wind_gust_max_kmh", "wind_direction_deg", "sst_mean_c"
        ]
        self.assertEqual(list(self.df.columns), expected_cols)

    def test_primary_key_uniqueness(self):
        dups = self.df.duplicated(subset=["district_code", "date"]).sum()
        self.assertEqual(dups, 0, f"Found {dups} duplicate (district_code, date) rows")

    def test_all_13_districts_present(self):
        present_codes = set(self.df["district_code"].unique())
        self.assertEqual(present_codes, CANONICAL_13_DISTRICT_CODES)

    def test_physical_pressure_bounds(self):
        self.assertTrue((self.df["msl_min_hpa"] >= 900.0).all(), "Found msl_min_hpa < 900 hPa")
        self.assertTrue((self.df["msl_min_hpa"] <= 1040.0).all(), "Found msl_min_hpa > 1040 hPa")
        self.assertTrue((self.df["msl_mean_hpa"] >= self.df["msl_min_hpa"]).all(), "Found msl_mean_hpa < msl_min_hpa")

    def test_physical_wind_bounds(self):
        self.assertTrue((self.df["wind_speed_mean_kmh"] >= 0.0).all(), "Negative wind speed mean")
        self.assertTrue((self.df["wind_speed_max_kmh"] >= self.df["wind_speed_mean_kmh"]).all(), "Max wind < Mean wind")
        self.assertTrue((self.df["wind_gust_max_kmh"] >= self.df["wind_speed_max_kmh"]).all(), "Peak gust < Max sustained wind")
        self.assertTrue((self.df["wind_direction_deg"] >= 0.0).all(), "Wind direction < 0 deg")
        self.assertTrue((self.df["wind_direction_deg"] <= 360.0).all(), "Wind direction > 360 deg")

    def test_coastal_sst_coverage_and_bounds(self):
        coastal_df = self.df[self.df["district_code"].isin(COASTAL_DISTRICT_CODES)]
        self.assertEqual(coastal_df["sst_mean_c"].isna().sum(), 0, "Missing SST in coastal districts")
        self.assertTrue((coastal_df["sst_mean_c"] >= 20.0).all(), "SST < 20.0 C in Bay of Bengal")
        self.assertTrue((coastal_df["sst_mean_c"] <= 35.0).all(), "SST > 35.0 C in Bay of Bengal")

    def test_inland_sst_strictly_nan(self):
        inland_df = self.df[self.df["district_code"].isin(INLAND_DISTRICT_CODES)]
        self.assertEqual(len(inland_df), inland_df["sst_mean_c"].isna().sum(),
                         "Inland districts must have 100% NaN for SST (zero silent imputation)")


if __name__ == "__main__":
    unittest.main()
