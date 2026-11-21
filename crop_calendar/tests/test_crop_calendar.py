"""
Unit tests for Crop Calendar Deliverables (Issue #43)
Validates schema, completeness across the 17 eligible crops,
weather window alignment with Issue 3, and source citation non-emptiness.
"""

import os
import unittest
import pandas as pd

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_PATH = os.path.join(MODULE_DIR, "crop_calendar_v1.csv")
DISTRICT_PATH = os.path.join(MODULE_DIR, "district_crop_calendar_v1.csv")

VALID_WEATHER_WINDOWS = {
    "annual_cal",
    "annual_crop",
    "kharif",
    "ne",
    "rabi_next",
    "rabi_prev"
}

VALID_MONTHS = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
}

CANONICAL_13_DISTRICT_CODES = {
    502, 503, 504, 505, 506, 510, 511, 515, 517, 519, 520, 521, 523
}

# The 17 crops from Issue 5 (with sorghum explicitly split into kharif/rabi)
REQUIRED_CROPS = {
    "rice", "groundnut", "maize", "sugarcane", "pigeonpea", "sesamum",
    "chickpea", "sunflower", "finger_millet", "pearl_millet", "cotton",
    "castor", "soyabean", "rapeseed_and_mustard", "safflower", "wheat", "linseed"
}

class TestCropCalendar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_canon = pd.read_csv(CANONICAL_PATH)
        cls.df_dist = pd.read_csv(DISTRICT_PATH)

    def test_files_exist(self):
        self.assertTrue(os.path.exists(CANONICAL_PATH), "crop_calendar_v1.csv does not exist")
        self.assertTrue(os.path.exists(DISTRICT_PATH), "district_crop_calendar_v1.csv does not exist")

    def test_canonical_schema(self):
        expected_cols = [
            "crop", "season", "sowing_window_start", "sowing_window_end",
            "harvest_window_start", "harvest_window_end", "duration_days_approx",
            "weather_window_alignment", "cyclone_vulnerability_window",
            "notes_phenology", "source_reference", "status"
        ]
        self.assertEqual(list(self.df_canon.columns), expected_cols)

    def test_all_17_crops_represented(self):
        present_crops = set(self.df_canon["crop"].unique())
        for crop in REQUIRED_CROPS:
            self.assertIn(crop, present_crops, f"Required crop '{crop}' missing from crop calendar")
        # Also ensure sorghum is represented
        self.assertTrue("kharif_sorghum" in present_crops or "sorghum" in present_crops)
        self.assertTrue("rabi_sorghum" in present_crops or "sorghum" in present_crops)

    def test_no_empty_sources(self):
        self.assertTrue(self.df_canon["source_reference"].notna().all(), "Found null source_reference in canonical table")
        self.assertTrue((self.df_canon["source_reference"].str.strip() != "").all(), "Found empty source_reference string in canonical table")
        self.assertTrue(self.df_dist["source_reference"].notna().all(), "Found null source_reference in district table")
        self.assertTrue((self.df_dist["source_reference"].str.strip() != "").all(), "Found empty source_reference string in district table")

    def test_weather_window_alignment_matches_issue_3(self):
        for window in self.df_canon["weather_window_alignment"].dropna():
            self.assertIn(window, VALID_WEATHER_WINDOWS, f"Invalid weather window alignment '{window}'")
        for window in self.df_dist["weather_window_alignment"].dropna():
            self.assertIn(window, VALID_WEATHER_WINDOWS, f"Invalid district weather window alignment '{window}'")

    def test_valid_calendar_months(self):
        for col in ["sowing_window_start", "sowing_window_end", "harvest_window_start", "harvest_window_end"]:
            for month in self.df_canon[col].dropna():
                self.assertIn(month, VALID_MONTHS, f"Invalid month '{month}' in column {col}")

    def test_status_values(self):
        valid_statuses = {"verified", "unavailable"}
        for status in self.df_canon["status"].dropna():
            self.assertIn(status, valid_statuses, f"Invalid status '{status}' in canonical table")
        for status in self.df_dist["status"].dropna():
            self.assertIn(status, valid_statuses, f"Invalid status '{status}' in district table")

    def test_district_referential_integrity(self):
        for dcode in self.df_dist["district_code"].unique():
            self.assertIn(dcode, CANONICAL_13_DISTRICT_CODES, f"District code {dcode} not in canonical 13 AP districts")

    def test_unavailable_crops_explicitly_tagged(self):
        unavailable_crops = self.df_canon[self.df_canon["status"] == "unavailable"]["crop"].tolist()
        self.assertIn("wheat", unavailable_crops, "Wheat must be explicitly tagged as unavailable in AP")
        self.assertIn("linseed", unavailable_crops, "Linseed must be explicitly tagged as unavailable in AP")

if __name__ == "__main__":
    unittest.main()
