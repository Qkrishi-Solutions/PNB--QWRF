# Issue 3: Weather Feature Engineering & Anomaly Generation

## Purpose

This workflow derives district-specific weather normals, weather anomalies, and provisional weather-stress screening indicators from the weather measurements already present in `crop_long_v1.csv`.

It is designed for the QWeatherRiskF / PNB data pipeline. It does not alter the source dataset.

The workflow supports two separate evaluation methodologies:

1. **Leave-One-Year-Out (LOYO):** Calculates weather normals using all years except the held-out validation year.
2. **Forward Chaining:** Calculates weather normals using only observations from years strictly earlier than the target year, simulating prospective year-by-year anomaly generation.

Both methodologies generate separate outputs and validation results. Their outputs must be evaluated and interpreted independently.

## Files in this folder

| File | Purpose |
|---|---|
| `weather_feature_engineering.py` | Existing LOYO extraction, fold-wise normal calculation, anomaly generation, stress screening, and validation |
| `weather_forward_chaining.py` | Separate forward-chaining normal calculation, target-year anomaly generation, and validation |
| `district_weather_normals_v1.csv` | LOYO district/window/indicator normals, separately calculated for each held-out validation year |
| `weather_anomalies_v1.csv` | LOYO actual values and anomalies for every district-year-weather feature in every validation fold |
| `weather_stress_features_v1.csv` | LOYO raw weather context, fold-specific anomalies, and provisional stress flags |
| `district_weather_normals_forward_chaining_v1.csv` | Forward-chaining district/window/indicator normals calculated using only previous years |
| `weather_anomalies_forward_chaining_v1.csv` | Forward-chaining actual values and anomalies for the corresponding target year |
| `weather_feature_validation_report.pdf` | Summary of the original LOYO method, validation results, findings, and limitations |
| `forward_chaining_validation_report.txt` | Summary of the forward-chaining methodology and its validation results |
| `ISSUE_3_WEATHER_FEATURE_ENGINEERING_GUIDE.md` | This implementation guide |
| `validation_tables/validation_checks.csv` | Machine-readable LOYO structural and leakage checks |
| `validation_tables/forward_chaining_validation_checks.csv` | Machine-readable forward-chaining validation checks |
| `validation_tables/weather_missingness.csv` | Missingness by seasonal window and indicator |
| `validation_tables/weather_feature_inventory.csv` | Inventory of all source weather columns |
| `validation_tables/stress_rule_definitions.csv` | Explicit definitions of provisional LOYO stress screening rules |

## Input and source handling

Input: `crop_long_v1.csv` from `Issues/CropLongFormatv1/`.

The source has 1,430 crop-long rows, but weather measurements are district-year measurements repeated across crop rows. The scripts first verify that repeated weather values are identical within each `(district_code, year)` group, then extract one weather row per district-year.

The resulting weather input contains 65 district-year rows (13 districts × 5 years) and 78 weather columns (13 indicators × 6 seasonal windows).

The source dataset is read as input, and neither implementation intentionally modifies the original source file.

## Weather features and seasonal windows

The 13 source indicators are:

`rain_sum`, `rainy_days`, `longest_dry`, `max_5day_rain`, `tmax_mean`, `tmin_mean`, `hot_days`, `cold_days`, `dtr_mean`, `rh_mean`, `solar_mean`, `wind_mean`, and `gdd_sum`.

The six windows are:

`annual_cal`, `annual_crop`, `kharif`, `ne`, `rabi_next`, and `rabi_prev`.

Original column names and numeric values are retained. The uploaded CSV does not include an explicit unit dictionary for the weather columns. Therefore, this implementation does not invent or convert units. Confirm the authoritative source metadata and units before interpreting thresholds or publishing unit labels.

## Normal and anomaly method

### Leave-One-Year-Out (LOYO)

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

This is leave-one-year-out evaluation, not a chronological forecasting simulation.

## Forward-chaining normal and anomaly method

In addition to LOYO evaluation, this workflow provides a separate forward-chaining evaluation to simulate prospective year-by-year weather anomaly generation.

For each target year:

1. Select only observations from calendar years strictly earlier than the target year.
2. Calculate the mean and sample standard deviation for each district, seasonal window, and indicator using valid previous-year observations.
3. Require at least two valid training observations before making a normal available.
4. Apply the historical normal only to the corresponding target-year observation.
5. Calculate:
   - Absolute anomaly = actual value − training normal mean.
   - Percentage anomaly = (actual − normal mean) / normal mean × 100, when the normal mean is non-zero.
   - Standardised anomaly (z-score) = absolute anomaly / training sample standard deviation, when the standard deviation is positive.

Unlike LOYO, forward chaining does not use later years to calculate the normal for an earlier target year.

The forward-chaining anomaly output contains only target-year observations. Historical training observations are not repeated as anomaly output rows.

### Forward-chaining training-year structure

For the available 2015–2019 dataset, the training periods are:

| Target year | Training years |
|---|---|
| 2015 | None |
| 2016 | 2015 |
| 2017 | 2015–2016 |
| 2018 | 2015–2017 |
| 2019 | 2015–2018 |

A normal is unavailable when fewer than two valid previous-year observations exist. Missing weather values are not imputed.

### Forward-chaining baseline availability

The forward-chaining implementation generated 5,070 normal records and 5,070 target-year anomaly records.

The baseline availability results are:

| Target year | Available normals | Insufficient normals | Total |
|---|---:|---:|---:|
| 2015 | 0 | 1,014 | 1,014 |
| 2016 | 0 | 1,014 | 1,014 |
| 2017 | 845 | 169 | 1,014 |
| 2018 | 1,014 | 0 | 1,014 |
| 2019 | 1,014 | 0 | 1,014 |
| **Total** | **2,873** | **2,197** | **5,070** |

The unavailable baselines are expected:

- **2015:** No previous-year observations are available.
- **2016:** Only one previous year is available, which does not satisfy the minimum two-observation requirement.
- **2017:** Two previous years are available, but 169 district-weather feature records have insufficient valid historical observations because the corresponding 2015 `rabi_prev` values are missing.
- **2018 and 2019:** All 1,014 normal records per year have sufficient historical observations.

The 169 unavailable 2017 baselines are preserved as insufficient history. No artificial values or imputations are introduced.

### Separate forward-chaining outputs

The forward-chaining pipeline is implemented in `weather_forward_chaining.py` and generates:

- `district_weather_normals_forward_chaining_v1.csv`
- `weather_anomalies_forward_chaining_v1.csv`
- `forward_chaining_validation_report.txt`
- `validation_tables/forward_chaining_validation_checks.csv`

The existing LOYO script and its output files remain unchanged. Both evaluation methods must be evaluated and interpreted separately.

## Missing-value policy

Missing weather values are retained as missing. No zero filling, interpolation, forward filling, or imputation is performed.

The source contains 169 missing values across the 78 unique district-year weather features. All 169 are in `rabi_prev` for 2015: 13 indicators × 13 districts.

The LOYO anomaly output repeats these missing observations across the five validation folds (845 missing fold-observation values).

For LOYO, a normal is marked `available` when at least two valid training observations exist. The current dataset provides sufficient training observations for all 5,070 district-feature-fold normal rows.

For forward chaining, the same minimum of two valid training observations applies, but the training history is restricted to previous years. Consequently, 2,197 of the 5,070 forward-chaining normal records are marked `insufficient_training_history`.

These are short-period historical training baselines and should not be mistaken for long-term climatological normals.

## Stress screening rules

The existing LOYO stress output includes the original weather values, absolute anomalies, percentage anomalies, z-scores, and these provisional flags where the source indicator exists:

- Rainfall deficit: `rain_sum` percentage anomaly ≤ −20%.
- Rainfall excess: `rain_sum` percentage anomaly ≥ +20%.
- Extreme-rainfall screening: `max_5day_rain` z-score ≥ +2.
- Dry-spell stress screening: `longest_dry` z-score ≥ +1.5.
- Heat-stress screening: either `tmax_mean` or `hot_days` z-score ≥ +1.5.

These thresholds are transparent initial screening heuristics, not official meteorological definitions or crop-specific agronomic thresholds. They must be reviewed with the project/domain team and calibrated to the relevant crop, growth stage, location, and source units before being used as decision rules.

A missing z-score or percentage anomaly produces a missing flag, not a false flag.

The separate forward-chaining implementation currently generates normals and anomalies. It does not replace or overwrite the existing LOYO stress feature output.

## Validation results

### LOYO validation

The original LOYO implementation passed all 10 structural and data-quality validation checks.

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

The LOYO validation results are available in machine-readable form in `validation_tables/validation_checks.csv`.

### Forward-chaining validation

The separate forward-chaining implementation passed all 11 validation checks on the supplied input.

| Validation check | Result | Detail |
|---|---|---|
| Crop-long source key uniqueness | PASS | 0 duplicate source keys |
| Weather values consistent across crop rows | PASS | 0 inconsistent district-year/weather-column combinations |
| One weather record per district-year | PASS | 65 unique district-year rows |
| Expected anomaly row count and unique keys | PASS | 5,070 rows; expected 5,070 |
| Only strictly earlier years enter training | PASS | Training years verified for every target year |
| Independent recomputation of normals | PASS | Means, sample standard deviations, and valid counts independently verified |
| Insufficient history produces unavailable normals | PASS | 2,197 normal records have insufficient history |
| Anomaly formula | PASS | Anomaly equals actual value minus training normal mean |
| Missing weather observations remain missing | PASS | 169 missing target observations; expected 169 |
| Anomaly output contains target-year observations only | PASS | Each anomaly is generated for its corresponding target year |
| Zero or unavailable standard deviation produces no z-score | PASS | Z-scores remain missing when standard deviation is zero or unavailable |
| Input source remains unchanged | PASS | Source SHA-256 hash is unchanged |

The forward-chaining results are available in:

- `validation_tables/forward_chaining_validation_checks.csv`
- `forward_chaining_validation_report.txt`

The checks confirm that each target year uses only strictly earlier years, baseline calculations can be independently reproduced, insufficient historical data is handled explicitly, and the original input remains unchanged.

## Important limitations and handoff notes

1. Only five calendar years (2015–2019) are available. Normals are short-period training baselines, not standard 30-year climate normals.
2. All `rabi_prev` features are missing for 2015. This reflects source coverage and is preserved.
3. The source file does not carry a weather-unit dictionary. Confirm units and definitions with the source data owner before final scientific interpretation.
4. The stress flags are provisional statistical screening features and require domain review.
5. The `ne` and `kharif` windows may overlap in time according to the project dataset documentation. Do not sum these windows as if they were independent.
6. LOYO folds use the other four years, including years later than the held-out year. Forward chaining addresses the prospective forecasting requirement by using only previous years.
7. Forward-chaining normals for early target years are unavailable when there are fewer than two valid historical observations. This is an expected limitation of the available five-year dataset.
8. A normal with zero training standard deviation cannot produce a standardised anomaly (z-score). Such z-scores remain missing.
9. The current forward-chaining pipeline generates normals and anomalies separately from LOYO. The existing stress feature generation remains part of the LOYO implementation.

## Reproduce the outputs

Install dependencies:

```bash
pip install pandas numpy
```

### Run the LOYO pipeline

Run from the repository root:

```powershell
py .\weather_feature_engineering\weather_feature_engineering.py --input "C:\Users\punithaqkrishi\Downloads\crop_long_v1.csv" --outdir ".\weather_feature_engineering"
```

This generates the existing LOYO normals, anomalies, stress features, and validation tables.

### Run the forward-chaining pipeline

Run from the repository root:

```powershell
py .\weather_feature_engineering\weather_forward_chaining.py --input "C:\Users\punithaqkrishi\Downloads\crop_long_v1.csv" --outdir ".\weather_feature_engineering"
```

This generates the separate forward-chaining normals, target-year anomalies, validation report, and validation checks.

Both scripts read the source dataset without intentionally modifying it. They raise an error if their validation checks fail.

The two pipelines can be executed independently. Running the forward-chaining pipeline does not overwrite the original LOYO outputs.
