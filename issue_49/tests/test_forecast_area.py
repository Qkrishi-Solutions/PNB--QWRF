import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from issue_49 import forecast_area as model


class ForecastTests(unittest.TestCase):
    def test_missing_and_ambiguous_area_preserves_observed_zero(self):
        for value, flag, result in (
            ("", "False", (None, "missing_area")),
            ("0", "True", (None, "ambiguous_zero")),
            ("0", "False", (0, "")),
            ("1.25", "False", (1250, "")),
            ("-1", "False", (None, "negative_area")),
        ):
            with self.subTest(value=value, flag=flag):
                self.assertEqual(model.area({
                    "area_1000_ha": value, "flag_nan_filled_with_zero": flag
                }), result)

    def test_annual_cutoff_uses_earliest_season_and_district_override(self):
        state = {
            ("rice", "kharif"): {"sowing_window_start": "June"},
            ("rice", "rabi"): {"sowing_window_start": "November"},
            ("sugarcane", "annual"): {"sowing_window_start": "January"},
        }
        district = {("502", "rice", "kharif"): {"typical_sowing_window": "June 15 - July 20"}}
        self.assertEqual(model.provisional_cutoff(("502", "rice", 2018), state, district),
                         date(2018, 6, 14))
        self.assertEqual(model.provisional_cutoff(("503", "rice", 2018), state, district),
                         date(2018, 5, 31))
        self.assertEqual(model.provisional_cutoff(("502", "sugarcane", 2018), state, district),
                         date(2017, 12, 31))

    def test_future_targets_do_not_change_earlier_baseline_forecasts(self):
        series = {("502", "rice"): {
            2015: (100, ""), 2016: (120, ""), 2017: (140, ""), 2018: (1000, ""), 2019: (3000, "")
        }}
        def predict():
            return {r["method"]: (r["predicted_area_ha"], r["training_years"])
                    for r in model.forecast(series, {}, ({}, {}), {}, [2018], model.BASELINES)}
        expected = {"last_value": (140, "2015;2016;2017"),
                    "climatology": (120, "2015;2016;2017"),
                    "trend": (160, "2015;2016;2017")}
        self.assertEqual(predict(), expected)
        series["502", "rice"].update({2018: (8000, ""), 2019: (9000, "")})
        self.assertEqual(predict(), expected)

    def test_at_cutoff_release_excluded_and_weather_lag_is_explicit(self):
        target = "502", "rice", 2018
        earlier = "502", "rice", 2017
        series = {("502", "rice"): {2017: (20, ""), 2018: (30, "")}}
        cutoff = date(2018, 5, 31)
        timing = {target: {"forecast_date": cutoff, "lagged_kharif_available_on": cutoff},
                  earlier: {"area_available_on": cutoff}}
        rainfall = {("502", 2017): 800, ("502", 2018): 10000}
        self.assertEqual(model.history_before(series, target, timing), [])
        timing[earlier]["area_available_on"] = date(2018, 5, 30)
        history = model.history_before(series, target, timing)
        self.assertEqual(history, [(2017, 20)])
        self.assertIsNone(model.weather_features(target, rainfall, timing, history))
        timing[target]["lagged_kharif_available_on"] = date(2018, 5, 30)
        self.assertEqual(model.weather_features(target, rainfall, timing, history), (20, 2018, 800))

    def test_timing_evidence_rejects_rainfall_release_before_window_closes(self):
        with tempfile.TemporaryDirectory(dir=model.HERE) as folder:
            manifest = Path(folder) / "timing.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows([
                    ("district_code", "crop", "year", "forecast_date", "area_available_on",
                     "lagged_kharif_available_on", "source_reference"),
                    ("502", "rice", 2018, "2018-05-31", "2019-01-01", "2017-09-29", "fixture"),
                ])
            with self.assertRaisesRegex(ValueError, "predates the window end"):
                model.load_timing(manifest, {("502", "rice"): {2018: (100, "")}})

    def test_ridge_forecasts_ignore_future_targets_and_future_weather(self):
        series, rainfall, timing = {}, {}, {}
        for district in range(502, 507):
            code = str(district)
            series[code, "rice"] = {}
            for year in range(2015, 2020):
                series[code, "rice"][year] = (float(district + year), "")
                rainfall[code, year] = district * (year - 2014)
                timing[code, "rice", year] = {
                    "forecast_date": date(year, 5, 31),
                    "area_available_on": date(year + 1, 4, 1),
                    "lagged_kharif_available_on": date(year, 1, 1),
                }

        def predict():
            return [(r["predicted_area_ha"], r["training_years"]) for r in
                    model.forecast(series, rainfall, ({}, {}), timing, [2018], model.MODELS)]

        original = predict()
        self.assertEqual(len(original), 10)
        self.assertTrue(all(value is not None and years == "2016;2017" for value, years in original))
        for code, crop in series:
            for year in (2018, 2019):
                series[code, crop][year] = (100_000.0, "")
                rainfall[code, year] = 200_000
        self.assertEqual(predict(), original)

    def test_area_ridge_runs_without_weather_or_a_timing_manifest(self):
        series = {}
        for district in range(502, 507):
            series[str(district), "rice"] = {
                year: (float(district + year), "") for year in range(2015, 2020)
            }

        def predict():
            return list(model.forecast(series, {}, ({}, {}), {}, [2018], model.MODELS))

        forecasts = predict()
        area_only = [row for row in forecasts if row["method"] == model.MODELS[0]]
        with_weather = [row for row in forecasts if row["method"] == model.MODELS[1]]
        self.assertEqual(len(area_only), 5)
        self.assertTrue(all(row["status"] == "scored" and row["training_years"] == "2016;2017"
                            for row in area_only))
        self.assertTrue(all(row["reason"] == "unverified_or_missing_weather" for row in with_weather))
        for history in series.values():
            history[2018] = (100_000.0, "")
            history[2019] = (200_000.0, "")
        self.assertEqual([row["predicted_area_ha"] for row in predict()],
                         [row["predicted_area_ha"] for row in forecasts])

    def test_matched_comparison_excludes_row_for_all_methods(self):
        series = {
            ("502", "rice"): {2015: (100, ""), 2016: (120, ""), 2017: (140, ""), 2018: (160, "")},
            ("503", "rice"): {2015: (80, ""), 2016: (None, "ambiguous_zero"),
                              2017: (100, ""), 2018: (110, "")},
        }
        predictions = list(model.forecast(series, {}, ({}, {}), {}, [2018], model.BASELINES))
        matched = list(model.score(predictions, model.BASELINES, "matched"))
        self.assertEqual({r["n_scored"] for r in matched}, {1})
        self.assertEqual({r["n_targets"] for r in matched}, {2})
        available = list(model.score(predictions, model.BASELINES, "available", matched=False))
        self.assertEqual({r["n_scored"] for r in available if r["method"] == "last_value"}, {2})


if __name__ == "__main__":
    unittest.main()
