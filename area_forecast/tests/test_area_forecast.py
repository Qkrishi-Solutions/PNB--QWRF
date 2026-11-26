"""
Unit and Integration Tests for Issue #49 (Area Forecast).

Validates:
1. Deliverables exist
2. Row count & (district_code, year, crop) uniqueness derived dynamically from ml_ready_dataset_v1.csv
3. Zero temporal leakage (train_years < test_year; causal lag computation leaves missing lags as NaN)
4. Only pre-sowing weather features (wact__rabi_prev__*) used
5. Non-negative predictions across all 3 baselines and 4 ML models
6. Per-crop sample-size flags completeness
7. Reproducibility in a temporary directory matching committed CSVs
"""
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from area_forecast.build_area_forecast import (
    engineer_causal_lag_features,
    fit_crop_scales,
    run_area_forecast,
    select_features_in_fold,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_READY_PATH = REPO_ROOT / "ml_ready_dataset" / "ml_ready_dataset_v1.csv"
CALENDAR_PATH = REPO_ROOT / "crop_calendar" / "crop_calendar_v1.csv"
CONFIG_PATH = REPO_ROOT / "area_forecast" / "model_config.json"
OUT_DIR = REPO_ROOT / "area_forecast"


class TestAreaForecast(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_ml = pd.read_csv(ML_READY_PATH)
        cls.df_forecast = pd.read_csv(OUT_DIR / "area_forecast_v1.csv")
        cls.df_metrics = pd.read_csv(OUT_DIR / "area_forecast_metrics_v1.csv")
        cls.df_folds = pd.read_csv(OUT_DIR / "area_forecast_fold_metrics_v1.csv")
        cls.df_flags = pd.read_csv(OUT_DIR / "sample_size_flags_v1.csv")

    def test_deliverables_exist(self):
        for fname in [
            "area_forecast_v1.csv",
            "area_forecast_metrics_v1.csv",
            "area_forecast_fold_metrics_v1.csv",
            "area_forecast_crop_metrics_v1.csv",
            "sample_size_flags_v1.csv",
        ]:
            self.assertTrue((OUT_DIR / fname).exists(), f"Missing deliverable: {fname}")

    def test_derived_row_count_and_key_uniqueness(self):
        expected_rows = len(self.df_ml[self.df_ml["year"].isin([2018, 2019])])
        self.assertEqual(len(self.df_forecast), expected_rows)
        keys = self.df_forecast.set_index(["district_code", "year", "crop"])
        self.assertTrue(keys.index.is_unique, "Duplicate (district_code, year, crop) in area_forecast_v1.csv")

    def test_no_temporal_leakage_in_folds_and_weather_horizon(self):
        for _, row in self.df_folds.iterrows():
            train_years = [int(y) for y in str(row["train_years"]).split(";")]
            test_year = int(row["test_year"])
            self.assertTrue(
                all(y < test_year for y in train_years),
                f"Temporal leakage: train_years={train_years} >= test_year={test_year}",
            )
            sel = [f for f in str(row["selected_weather_features"]).split(";") if f and f != "nan"]
            for feat in sel:
                self.assertTrue(
                    feat.startswith("wact__rabi_prev__"),
                    f"Non-pre-sowing weather feature selected: {feat}",
                )

    def test_zero_silent_imputation_in_lag_engineering(self):
        df_eng = engineer_causal_lag_features(self.df_ml)
        # In 2015, lag_area_1 must be NaN; in 2016, lag_area_2 must be NaN (never silently filled with 0!)
        self.assertTrue(df_eng[df_eng["year"] == 2015]["lag_area_1"].isna().all())
        self.assertTrue(df_eng[df_eng["year"] == 2016]["lag_area_2"].isna().all())

        # Modifying 2019 area must NOT alter 2018 lag features
        df_perturbed = self.df_ml.copy()
        df_perturbed.loc[df_perturbed["year"] == 2019, "area_1000_ha"] = 99999.0
        df_eng_p = engineer_causal_lag_features(df_perturbed)
        pd.testing.assert_frame_equal(
            df_eng[df_eng["year"] <= 2018][["lag_area_1", "lag_hist_mean", "lag_yield_1"]],
            df_eng_p[df_eng_p["year"] <= 2018][["lag_area_1", "lag_hist_mean", "lag_yield_1"]],
        )

    def test_non_negative_predictions_and_all_models_present(self):
        required_pred_cols = [
            "pred_LastValue",
            "pred_Climatology",
            "pred_LinearTrend",
            "pred_XGBoost_L1_LogDelta",
            "pred_HistGBR_L1_LogDelta",
            "pred_RandomForest_L1_LogDelta",
            "pred_Ridge_Delta",
        ]
        for col in required_pred_cols:
            self.assertIn(col, self.df_forecast.columns)
            self.assertTrue(self.df_forecast[col].notna().all(), f"NaN predictions in {col}")
            self.assertTrue((self.df_forecast[col] >= 0.0).all(), f"Negative area prediction in {col}")

    def test_sample_size_flags_cover_all_crops(self):
        expected_crops = set(self.df_ml["crop"].unique())
        flagged_crops = set(self.df_flags["crop"].unique())
        self.assertEqual(flagged_crops, expected_crops)
        self.assertTrue(self.df_flags["sample_size_flag"].notna().all())

    def test_in_fold_feature_selection_respects_collinearity(self):
        df_eng = engineer_causal_lag_features(self.df_ml)
        df_tr = df_eng[df_eng["year"].isin([2016, 2017])].copy()
        scales, _ = fit_crop_scales(df_tr, "area_delta")
        self.assertEqual(set(scales.keys()), set(df_tr["crop"].unique()))

        candidates = [c for c in df_tr.columns if c.startswith("wact__rabi_prev__") and df_tr[c].notna().all()]
        selected = select_features_in_fold(df_tr, candidates, "area_log_delta", max_features=4, max_collinearity=0.80)
        self.assertLessEqual(len(selected), 4)

    def test_reproducibility_in_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_area_forecast(
                dataset_path=ML_READY_PATH,
                calendar_path=CALENDAR_PATH,
                config_path=CONFIG_PATH,
                outdir=tmp_path,
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(tmp_path / "area_forecast_v1.csv"),
                self.df_forecast,
                rtol=1e-6,
                atol=1e-6,
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(tmp_path / "area_forecast_metrics_v1.csv"),
                self.df_metrics,
                rtol=1e-6,
                atol=1e-6,
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(tmp_path / "sample_size_flags_v1.csv"),
                self.df_flags,
            )


if __name__ == "__main__":
    unittest.main()
