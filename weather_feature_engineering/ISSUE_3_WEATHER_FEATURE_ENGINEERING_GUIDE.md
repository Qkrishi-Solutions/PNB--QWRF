# Issue 3: Weather Feature Engineering & Anomaly Generation

## Purpose

This workflow derives district-specific weather normals, weather anomalies, and provisional weather-stress screening indicators from the weather measurements already present in `crop_long_v1.csv`.

It is designed for the QWeatherRiskF / PNB data pipeline. It does not alter the source dataset.

## Files in this folder

| File | Purpose |
|---|---|
| `weather_feature_engineering.py` | Reproducible extraction, fold-wise normal calculation, anomaly generation, stress screening, and validation |
| `district_weather_normals_v1.csv` | District/window/indicator normals, separately calculated for each held-out validation year |
| `weather_anomalies_v1.csv` | Actual values and anomalies for every district-year-weather feature in every validation fold |
| `weather_stress_features_v1.csv` | Raw weather context, fold-specific anomalies, and provisional stress flags |
| `weather_feature_validation_report.pdf` | Summary of method, validation results, findings, and limitations |
| `ISSUE_3_WEATHER_FEATURE_ENGINEERING_GUIDE.md` | This implementation guide |
| `validation_tables/validation_checks.csv` | Machine-readable structural and leakage checks |
| `validation_tables/weather_missingness.csv` | Missingness by seasonal window and indicator |
| `validation_tables/weather_feature_inventory.csv` | Inventory of all source weather columns |
| `validation_tables/stress_rule_definitions.csv` | Explicit definitions of provisional stress screening rules |

## Input and source handling

Input: `crop_long_v1.csv` from `Issues/CropLongFormatv1/`.

The source has 1,430 crop-long rows, but weather measurements are district-year measurements repeated across crop rows. The script first verifies that repeated weather values are identical within each `(district_code, year)` group, then extracts one weather row per district-year.

The resulting weather input contains 65 district-year rows (13 districts × 5 years) and 78 weather columns (13 indicators × 6 seasonal windows).

The source dataset is read as input, and the script does not intentionally modify the original source file.

## Weather features and seasonal windows

The 13 source indicators are:

`rain_sum`, `rainy_days`, `longest_dry`, `max_5day_rain`, `tmax_mean`, `tmin_mean`, `hot_days`, `cold_days`, `dtr_mean`, `rh_mean`, `solar_mean`, `wind_mean`, and `gdd_sum`.

The six windows are:

`annual_cal`, `annual_crop`, `kharif`, `ne`, `rabi_next`, and `rabi_prev`.

Original column names and numeric values are retained. The uploaded CSV does not include an explicit unit dictionary for the weather columns. Therefore, this implementation does not invent or convert units. Confirm the authoritative source metadata and units before interpreting thresholds or publishing unit labels.

## Normal and anomaly method

For each leave-one-year-out (LOYO) validation fold:

1. Select the four years other than the held-out validation year as the training years.
2. Calculate the mean and sample standard deviation for each district, seasonal window, and indicator using only valid training-year observations.
3. Apply that same training-only normal to every observation in the fold, including both training rows and the held-out row.
4. Calculate:
   - Absolute anomaly = actual value − training normal mean.
   - Percentage anomaly = (actual − normal mean) / normal mean × 100, when the normal mean is non-zero.
   - Standardised anomaly (z-score) = absolute anomaly / training sample standard deviation, when the standard deviation is positive.

This fold-specific approach is important. For a given validation fold, the held-out year's weather is not used in the normal used to create any training or validation feature in that fold.

`district_weather_normals_v1.csv` contains one row per district × weather feature × held-out year. `weather_anomalies_v1.csv` contains one row per district-year-weather feature-validation fold. The `validation_fold_year` field identifies the fold and `is_validation_year` identifies the held-out observation.

This is leave-one-year-out evaluation, not a chronological forecasting simulation. If the intended deployment predicts only future years from past years, a forward-chaining validation design should also be evaluated.

## Missing-value policy

Missing weather values are retained as missing. No zero filling, interpolation, forward filling, or imputation is performed.

The source contains 169 missing values across the 78 unique district-year weather features. All 169 are in `rabi_prev` for 2015: 13 indicators × 13 districts. The anomaly output repeats these missing observations across the five validation folds (845 missing fold-observation values).

A normal is marked `available` when at least two valid training observations exist. The current dataset provides sufficient training observations for all 5,070 district-feature-fold normal rows, but this is a small historical sample and should not be mistaken for a long-term climatological normal.

## Stress screening rules

The stress output includes the original weather values, absolute anomalies, percentage anomalies, z-scores, and these provisional flags where the source indicator exists:

- Rainfall deficit: `rain_sum` percentage anomaly ≤ −20%.
- Rainfall excess: `rain_sum` percentage anomaly ≥ +20%.
- Extreme-rainfall screening: `max_5day_rain` z-score ≥ +2.
- Dry-spell stress screening: `longest_dry` z-score ≥ +1.5.
- Heat-stress screening: either `tmax_mean` or `hot_days` z-score ≥ +1.5.

These thresholds are transparent initial screening heuristics, not official meteorological definitions or crop-specific agronomic thresholds. They must be reviewed with the project/domain team and calibrated to the relevant crop, growth stage, location, and source units before being used as decision rules. A missing z-score or percentage anomaly produces a missing flag, not a false flag.

## Validation results

The supplied input passed all 10 structural and data-quality validation checks.

| Validation check | Result | Detail |
|---|---|---|
| Crop-long source key uniqueness | PASS | 0 duplicate source keys |
| Weather values consistent across crop rows | PASS | 0 inconsistent district-year/weather-column combinations |
| One weather record per district-year | PASS | 65 unique district-year rows |
| No held-out year enters its fold's training normals | PASS | Training years verified for every held-out-year fold |
| Expected number of weather normal records | PASS | 5,070 rows; expected 5,070 |
| Expected number of weather anomaly records | PASS | 25,350 rows; expected 25,350 |
| Expected number of weather stress records | PASS | 325 rows; expected 325 |
| Missing weather values remain missing | PASS | 845 missing fold-observation values; expected 845 |
| Weather feature count is 78 | PASS | 78 features detected |
| Fold-specific normals remain consistent | PASS | Each district-feature group uses one normal throughout each fold |

The validation checks confirm that the expected output dimensions are generated, missing weather observations are preserved, and each validation fold uses its own training-year normal.

The validation results are also available in machine-readable form in `validation_tables/validation_checks.csv`.

## Important limitations and handoff notes

1. Only five calendar years (2015–2019) are available. Normals are short-period training baselines, not standard 30-year climate normals.
2. All `rabi_prev` features are missing for 2015. This reflects source coverage and is preserved.
3. The source file does not carry a weather-unit dictionary. Confirm units and definitions with the source data owner before final scientific interpretation.
4. The stress flags are provisional statistical screening features and require domain review.
5. The `ne` and `kharif` windows may overlap in time according to the project dataset documentation. Do not sum these windows as if they were independent.
6. Leave-one-year-out folds use the other four years, including years later than the held-out year. Use forward-chaining if the modelling question is strictly prospective.

## Reproduce the outputs

Install dependencies:

```bash
pip install pandas numpy
```

Run from this folder (or provide the appropriate input path):

```bash
py .\weather_feature_engineering\weather_feature_engineering.py --input "C:\Users\punithaqkrishi\Downloads\crop_long_v1.csv" --outdir .\weather_feature_engineering
```

The script creates the three deliverable CSVs and the validation tables. It raises an error if a structural or leakage check fails.
