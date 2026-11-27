# Issue #7: Integrated ML-Ready Dataset

One integrated table, **one row per `district_code` + `year` + `crop`**, that joins the engineered crop, weather, yield, selected-predictor and climate-zone data. It is the **single input for the forecasting issues (#49 to #52)**. Nothing is imputed: missing stays missing.

> This is a data-preparation deliverable. It makes no claim about prediction skill. Weather–yield relationships in it are statistical associations, not causation.

## Run

From the repository root, with Python 3.12 and the pinned packages:

```bash
pip install -r ml_ready_dataset/requirements.txt
python ml_ready_dataset/build_ml_ready_dataset.py
python -m pytest ml_ready_dataset/tests -v
```

The build stops with a non-zero exit if any validation or leakage check fails. The output is byte-identical only with the pinned versions; other pandas/numpy versions can differ in the last float digit (about 1e-32), which the tests tolerate.

## Inputs

| Input | Used for | Issue |
|---|---|---|
| `CropLongFormatv1/crop_long_v1.csv` | base rows | 1 |
| `CropModelEligibility/crop_model_eligibility_v1.csv` | which crops proceed to modelling | 2 |
| `weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv` | weather anomalies, joined at district-year | 3 |
| `yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv` | yield anomaly targets | 4 |
| `crop specific weather feature selection/crop_feature_selection_v1.csv` | crop-level LOYO predictor summary | 5 (#36) |
| `District Climate Zonal Clustering/district_zone_map_v1.csv` | climate zone | 6 |

## Outputs

| File | Content |
|---|---|
| `ml_ready_dataset_v1.csv` | the dataset (715 rows × 313 columns) |
| `ml_ready_column_dictionary_v1.csv` | every column: role, description, `baseline_leakage_safe` and `usable_as_feature` flags, why a column is unusable, availability note, timing caveat, leakage note, missing count (also rendered below) |
| `missing_data_report_v1.csv` | **every missing row and column with its cause** (rendered in the Completeness section) |
| `excluded_crop_rows_v1.csv` | the 715 crop-long rows that are not carried (crops that are not eligible), with the reason and, where one exists, the Issue 4 yield anomaly |
| `weather_loyo_anomalies_v1.csv` | leave-one-year-out weather anomalies for **all** years (`wloyo__*`), including 2015 and 2016, which have none in the main table. **Not forward-chaining safe** |
| `fc_crop_feature_selection_v1.csv`, `fc_crop_feature_selection_detail_v1.csv` | forward-chaining crop-specific predictors per crop and target year, and the correlation statistics behind them |
| `join_coverage_report_v1.csv` | matched and unmatched keys for every join |
| `unmatched_keys_v1.csv` | the unmatched keys, each with a `reason`: `GAP` (a real missing source record) or `expected` (e.g. a crop that is not eligible in Issue 2) |
| `exclusion_report_v1.csv` | crop-long rows excluded and why |
| `validation_checks_v1.csv` | all validation, leakage and coverage checks with results |
| `reproducibility_report_v1.json` | exact command, row counts, package versions, SHA-256 of inputs and output |

## How rows are selected

Base rows are crop-long rows whose crop has `proceed_to_modelling = True` (Issue 2) and `use_in_modelling = True` (Issue 1). The expected count is **derived from the inputs**, not hard-coded. In the committed run: 1,430 crop-long rows, 11 eligible crops, **715 rows kept**, 715 excluded (crops that are ineligible or only conditionally eligible).

Rows are **not** dropped for having a missing yield anomaly. Use the flags to filter:

- `has_valid_yield_anomaly`: a forward-chaining yield anomaly exists (384 rows; 2017–2019 only).
- `usable_for_forward_chaining_eval`: eligible crop and valid anomaly (384 rows; 128 in each of 2017, 2018, 2019).

## Forward-chaining crop-specific predictors (`fc_*`): the leakage-safe alternative to `loyo_*`

Issue 5's published selection (`loyo_*`) used leave-one-year-out folds, so the choice for a held-out year saw later years. The `fc_*` columns use **the same method and thresholds** (Issue 5's own `select_features_for_fold`: Spearman, Benjamini-Hochberg, multicollinearity filter, at most 4 features, `MIN_N` 8, `COVERAGE` 0.6, `MIN_RHO` 0.3, `MAX_Q` 0.2, `MAX_CORR` 0.85) and the same row filter, but for every crop and target year `T` they train **only on target years strictly before `T`**. They are therefore safe to attach to rows of year `T`.

| Column | Meaning |
|---|---|
| `fc_selection_train_years` | the earlier target years the selection used |
| `fc_selection_n_training_rows` | rows it used |
| `fc_selection_status` | `selected`, `no_predictors_passed_screen` (ran, nothing passed), `no_prior_target_years` (could not run), `crop_excluded_by_issue5_row_filter`, or Issue 5's insufficient-data statuses |
| `fc_n_selected_predictors` | 0 means it ran and nothing passed; **missing means it could not run** |
| `fc_selected_predictors` | `;`-separated, in Issue 5 feature naming (`<window>__<indicator>`, z-score anomalies) |

**Result (committed run, 55 crop-years):** 4 selected, 16 ran with nothing passing, 30 could not run (no forward-chaining target exists before 2017, so 2015 to 2017 have no earlier target years), 5 are `rabi_sorghum` (see below). The four selections are chickpea 2018 (trained on 2017 only, 11 rows) and maize, pigeonpea and sunflower 2019 (trained on 2017 to 2018, 20 to 26 rows). For maize, pigeonpea and sunflower these overlap with Issue 5's stable LOYO predictors (`annual_crop__hot_days`, `annual_crop__hot_days`, `kharif__dtr_mean`), which is a reassuring but small cross-check.

**Read this honestly:** with 11 to 26 rows and 78 candidate features, a selection is fragile, "nothing passed" is the typical outcome, and a selection is a statistical association, not a causal driver. Use `fc_*` as the per-crop feature list for year `T`, never `loyo_*`.

**Leakage tests (builder):** training years are all `< T`, and scrambling the target in every year `>= T` leaves each selection unchanged (a look-ahead mutation of the code fails the build; see the tests).

## Completeness: what is missing and why

Nothing in this dataset is silently absent. Every missing row and column below has a cause, and the same list is written to `missing_data_report_v1.csv` (checked by the build). The rows that are not carried are in `excluded_crop_rows_v1.csv`, and leave-one-year-out weather anomalies for all years (including 2015 and 2016, which have none in the main table) are in `weather_loyo_anomalies_v1.csv`. **That file is not forward-chaining safe**: its baselines use later years, so use it only for retrospective analysis.

<!-- COMPLETENESS_START (generated by build_ml_ready_dataset.py; do not edit by hand) -->

| Kind | What is missing | Count | Why | By design? | Where to look |
|---|---|---|---|---|---|
| rows | crop-long rows of castor (not carried) | 65 of 1430 | Issue 2: conditionally_eligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of cotton (not carried) | 65 of 1430 | Issue 2: conditionally_eligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of kharif_sorghum (not carried) | 65 of 1430 | Issue 2: conditionally_eligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of linseed (not carried) | 65 of 1430 | Issue 2: ineligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of minor_pulses (not carried) | 65 of 1430 | Issue 2: ineligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of oilseeds (not carried) | 65 of 1430 | Issue 2: ineligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of rapeseed_and_mustard (not carried) | 65 of 1430 | Issue 2: ineligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of safflower (not carried) | 65 of 1430 | Issue 2: ineligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of sorghum (not carried) | 65 of 1430 | Issue 2: ineligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of soyabean (not carried) | 65 of 1430 | Issue 2: ineligible | yes | excluded_crop_rows_v1.csv |
| rows | crop-long rows of wheat (not carried) | 65 of 1430 | Issue 2: ineligible | yes | excluded_crop_rows_v1.csv |
| rows | carried rows without a yield target: insufficient_history, 2015 | 129 of 715 | a forward-chaining anomaly needs two earlier years | yes | anomaly_status |
| rows | carried rows without a yield target: insufficient_history, 2016 | 132 of 715 | a forward-chaining anomaly needs two earlier years | yes | anomaly_status |
| rows | carried rows without a yield target: insufficient_history, 2017 | 3 of 715 | a forward-chaining anomaly needs two earlier years | yes | anomaly_status |
| rows | carried rows without a yield target: insufficient_history, 2018 | 2 of 715 | a forward-chaining anomaly needs two earlier years | yes | anomaly_status |
| rows | carried rows without a yield target: insufficient_history, 2019 | 5 of 715 | a forward-chaining anomaly needs two earlier years | yes | anomaly_status |
| rows | carried rows without a yield target: ineligible_observation, 2015 | 14 of 715 | zero, missing or non-positive yield; excluded from baselines | yes | anomaly_status |
| rows | carried rows without a yield target: ineligible_observation, 2016 | 11 of 715 | zero, missing or non-positive yield; excluded from baselines | yes | anomaly_status |
| rows | carried rows without a yield target: ineligible_observation, 2017 | 12 of 715 | zero, missing or non-positive yield; excluded from baselines | yes | anomaly_status |
| rows | carried rows without a yield target: ineligible_observation, 2018 | 13 of 715 | zero, missing or non-positive yield; excluded from baselines | yes | anomaly_status |
| rows | carried rows without a yield target: ineligible_observation, 2019 | 10 of 715 | zero, missing or non-positive yield; excluded from baselines | yes | anomaly_status |
| rows | district-crop pairs with no target in any year | 14 of 143 | crop not grown or not recorded in that district | yes | yield_value_status, flag_* |
| rows | rows with raw zeros: zero_ambiguous | 35 of 715 | area, production and yield all 0 and NOT flagged as filled: a true zero cannot be told from missing | AMBIGUOUS: labelled, source value kept | yield_value_status |
| rows | rows with raw zeros: zero_source_filled | 21 of 715 | the source filled a missing value with zero (flag_nan_filled_with_zero) | labelled | yield_value_status |
| rows | rows with raw zeros: zero_yield_only | 1 of 715 | yield is 0 while area or production is not | labelled | yield_value_status |
| rows | carried rows with no weather anomaly at all, 2015 | 143 of 143 | forward-chaining normals need two earlier years | yes | weather_loyo_anomalies_v1.csv holds leave-one-year-out anomalies for these years |
| rows | carried rows with no weather anomaly at all, 2016 | 143 of 143 | forward-chaining normals need two earlier years | yes | weather_loyo_anomalies_v1.csv holds leave-one-year-out anomalies for these years |
| rows | rows with a kg/ha anomaly but no z-score | 1 of 715 | baseline standard deviation is zero | yes | has_valid_yield_zscore |
| rows | rows whose z-score rests on fewer than 3 baseline observations | 128 of 383 | forward-chaining baseline has only 2 observations in the first target year | yes | yield_zscore_low_confidence |
| columns | aggregate_member_sum_area (dropped) | 715 of 715 | empty for every carried row (applies to aggregate categories only) | yes | reproducibility_report_v1.json -> dropped_empty_columns |
| columns | aggregate_member_sum_production (dropped) | 715 of 715 | empty for every carried row (applies to aggregate categories only) | yes | reproducibility_report_v1.json -> dropped_empty_columns |
| columns | wanom__kharif__cold_days (kept, unusable) | 286 of 715 | constant (zero variance): carries no information | yes | ml_ready_column_dictionary_v1.csv: usable_as_feature |
| columns | wz__kharif__cold_days (kept, unusable) | 715 of 715 | 100% missing | yes | ml_ready_column_dictionary_v1.csv: usable_as_feature |
| columns | wact__kharif__cold_days (kept, unusable) | 0 of 715 | constant (zero variance): carries no information | yes | ml_ready_column_dictionary_v1.csv: usable_as_feature |
| columns | wz__ z-score columns with gaps beyond the anomaly's | 11 of 78 | baseline standard deviation is zero (mostly the cold-days indicators) | yes | availability_note |
| columns | constant non-weather columns (kept) | 20 of 313 | provenance flags that are constant because only eligible crops are carried | yes | availability_note |
| columns | season_component (mostly empty) | 650 of 715 | applies to a subset of rows only | yes | n_missing |
| columns | component_of_aggregate (mostly empty) | 455 of 715 | applies to a subset of rows only | yes | n_missing |
| columns | flag_summary (mostly empty) | 402 of 715 | applies to a subset of rows only | yes | n_missing |
| columns | fc_n_selected_predictors (mostly empty) | 455 of 715 | applies to a subset of rows only | yes | n_missing |
| columns | raw same-year weather columns of crop-long (not carried) | 78 of 78 | identical to the wact__ columns (verified); nothing is lost | yes | wact__* |
| columns | leave-one-year-out weather anomalies (companion file, not in the main table) | 78 of 78 | not forward-chaining safe, so kept out of the model-ready table | yes | weather_loyo_anomalies_v1.csv |

<!-- COMPLETENESS_END -->

## Join coverage (committed run)

| Join | Keys | Base keys matched | Unmatched base keys |
|---|---|---|---|
| eligibility | `crop` | 11 / 11 | 0 |
| yield anomaly | `district_code`+`year`+`crop` | 715 / 715 | 0 |
| weather (district-year, deduplicated) | `district_code`+`year` | 65 / 65 | 0 |
| climate zone | `district_code` | 13 / 13 | 0 |
| crop feature selection (Issue 5) | `crop` | **10 / 11** | **1: `rabi_sorghum`** |

**Known gap:** `rabi_sorghum` is eligible under Issue 2 but has no Issue 5 result. The cause is a rule mismatch: crop-long marks it `is_seasonal_component = True`, which Issue 5's row filter excludes (`crop_excluded_by_issue5_row_filter`). Either Issue 2 or Issue 5 should change; this dataset does not decide. Separately, Issue 5 contains seven crops (castor, cotton, linseed, rapeseed_and_mustard, safflower, soyabean, wheat) that Issue 2 does not mark eligible. The two issues used different eligible sets. Its `loyo_*` columns stay **missing** for `rabi_sorghum`; nothing is filled in. This should be reconciled upstream.

## Column roles (see `ml_ready_column_dictionary_v1.csv`)

| Role | Columns | Use |
|---|---|---|
| key | 3 | `district_code`, `year`, `crop` |
| feature_weather_anomaly | 78 | `wanom__<window>__<indicator>`: forward-chaining anomaly (actual minus normal from earlier years) |
| feature_weather_anomaly_z | 78 | `wz__...`: same anomaly as a z-score (the form Issue 5 used) |
| feature_weather_actual | 78 | `wact__...`: target-year weather value. **Units are not fully defined in the source; do not assume them** |
| feature_context | 2 | `climate_zone` (a similarity group, not a risk level), `cluster_id` (a label, not a rank) |
| target | 5 | `yield_anomaly_kg_per_ha`, `yield_anomaly_pct`, `yield_anomaly_zscore`, `yield_deviation_class`, `anomaly_status` |
| target_baseline_meta | 10 | how each anomaly baseline was formed |
| outcome_year_measurement | 6 | area, production, yield measured in the target year |
| selection_info_loyo | 4 | Issue 5 leave-one-year-out predictor summary (**not** forward-chaining safe; see warning) |
| selection_info_forward_chaining | 5 | `fc_*`: the same selection using only earlier target years (leakage-safe) |
| target_quality_flag | 3 | `yield_value_status` (valid, missing, or which kind of zero), `has_valid_yield_zscore`, `yield_zscore_low_confidence` |
| eligibility_flag / data_quality_flag / metadata | 6 / 21 / 14 | filtering and traceability |

**Baseline-leakage-safe set (236 columns):** the `wanom__`, `wz__` and `wact__` columns, plus `climate_zone` and `crop`. "Safe" means only that the anomaly baselines use earlier years than the target year. It does **not** mean the weather would be available at a forecast origin: the source does not define the window dates, and `annual_cal`, `annual_crop`, `ne` and `rabi_next` may extend past harvest, so treat them as retrospective unless a window end is confirmed to precede the forecast origin. Everything else is a target, a target-year outcome, or selection information and must not be used as a feature.

> **Warning on `loyo_*` columns (use `fc_*` instead).** Issue 5 selected predictors with leave-one-year-out folds over 2017–2019, so the selection saw the held-out year and later years. They are included because the issue asks for them, but they are **not forward-chaining safe** (marked `safe = no`). Re-select features inside each forward-chaining fold.

The same-year raw weather columns that crop-long also carries are dropped after verifying they are identical to `wact__*`.

<!-- COLUMN_DICTIONARY_START (generated by build_ml_ready_dataset.py; do not edit by hand) -->

### Column dictionary

313 columns. `safe` = baseline-leakage safe (see the timing caveat below the table).

| Column | Role | Safe | Usable | Missing | Description |
|---|---|---|---|---|---|
| `wanom__<window>__<indicator>` (78 columns) | feature_weather_anomaly | yes | 77 of 78 | 286-429 | Issue 3 forward-chaining anomaly (actual minus normal from earlier years) |
| `wz__<window>__<indicator>` (78 columns) | feature_weather_anomaly_z | yes | 77 of 78 | 286-715 | Issue 3 forward-chaining anomaly z-score |
| `wact__<window>__<indicator>` (78 columns) | feature_weather_actual | yes | 77 of 78 | 0-143 | Issue 3 target-year weather value (units not defined in source) |
| `district_code` | key | no | n/a | 0 | Row key |
| `district_name` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `year` | key | no | n/a | 0 | Row key |
| `crop` | key | yes | yes | 0 | Row key; usable as a categorical feature |
| `crop_name` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `crop_group` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `is_aggregate` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `is_seasonal_component` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `season_component` | metadata | no | n/a | 650 | Descriptive attribute from the crop-long table |
| `component_of_aggregate` | metadata | no | n/a | 455 | Descriptive attribute from the crop-long table |
| `is_group_category` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `modelling_role` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `use_in_modelling` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `area_1000_ha` | outcome_year_measurement | no | n/a | 0 | Crop outcome measured in the target year |
| `production_1000_tons` | outcome_year_measurement | no | n/a | 0 | Crop outcome measured in the target year |
| `yield_kg_per_ha` | outcome_year_measurement | no | n/a | 0 | Crop outcome measured in the target year |
| `yield_check_kg_per_ha` | outcome_year_measurement | no | n/a | 61 | Crop outcome measured in the target year |
| `yield_rel_diff` | outcome_year_measurement | no | n/a | 65 | Crop outcome measured in the target year |
| `yield_vs_other_years_median` | outcome_year_measurement | no | n/a | 277 | Crop outcome measured in the target year |
| `flag_nan_filled_with_zero` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_any_missing` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_area_zero` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_production_zero` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_yield_zero` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_all_zero` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_zero_area_positive_production` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_positive_area_zero_production` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_positive_area_zero_yield` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_negative_value` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_yield_mismatch` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_low_area_lt_1000ha` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_yield_jump` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_aggregate_mismatch` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_sugarcane_unit_corrected` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_cotton_2019_suspect` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_minor_pulses_not_reported` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_changed_by_fix_plan` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_estimated_by_allocation` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `n_flags` | data_quality_flag | no | n/a | 0 | Issue 1 data-quality flag |
| `flag_summary` | data_quality_flag | no | n/a | 402 | Issue 1 data-quality flag |
| `state_code` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `state_name` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `crop_dist_name` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `rain_valid_area_fraction` | metadata | no | n/a | 0 | Descriptive attribute from the crop-long table |
| `eligibility_class` | eligibility_flag | no | n/a | 0 | Eligibility and usability flags |
| `proceed_to_modelling` | eligibility_flag | no | n/a | 0 | Eligibility and usability flags |
| `eligibility_reasons` | eligibility_flag | no | n/a | 0 | Eligibility and usability flags |
| `eligibility_reason` | target_baseline_meta | no | n/a | 0 | How the yield anomaly baseline was formed |
| `eligible_for_baseline` | target_baseline_meta | no | n/a | 0 | How the yield anomaly baseline was formed |
| `validation_year` | target_baseline_meta | no | n/a | 0 | How the yield anomaly baseline was formed |
| `baseline_mean_kg_per_ha` | target_baseline_meta | no | n/a | 181 | How the yield anomaly baseline was formed |
| `baseline_std_kg_per_ha` | target_baseline_meta | no | n/a | 329 | How the yield anomaly baseline was formed |
| `training_observation_count` | target_baseline_meta | no | n/a | 0 | How the yield anomaly baseline was formed |
| `training_years` | target_baseline_meta | no | n/a | 181 | How the yield anomaly baseline was formed |
| `baseline_status` | target_baseline_meta | no | n/a | 0 | How the yield anomaly baseline was formed |
| `minimum_training_observations` | target_baseline_meta | no | n/a | 0 | How the yield anomaly baseline was formed |
| `baseline_method` | target_baseline_meta | no | n/a | 0 | How the yield anomaly baseline was formed |
| `yield_anomaly_kg_per_ha` | target | no | n/a | 331 | Issue 4 forward-chaining yield anomaly outcome |
| `yield_anomaly_pct` | target | no | n/a | 331 | Issue 4 forward-chaining yield anomaly outcome |
| `yield_anomaly_zscore` | target | no | n/a | 332 | Issue 4 forward-chaining yield anomaly outcome |
| `anomaly_status` | target | no | n/a | 0 | Issue 4 forward-chaining yield anomaly outcome |
| `yield_deviation_class` | target | no | n/a | 0 | Issue 4 forward-chaining yield anomaly outcome |
| `cluster_id` | feature_context | no | n/a | 0 | Issue 6 cluster label |
| `climate_zone` | feature_context | yes | yes | 0 | Issue 6 climate zone (a similarity group, not a risk level) |
| `loyo_crop_selection_status` | selection_info_loyo | no | n/a | 65 | Issue 5 leave-one-year-out predictor summary |
| `loyo_n_stable_predictors` | selection_info_loyo | no | n/a | 65 | Issue 5 leave-one-year-out predictor summary |
| `loyo_stable_predictors` | selection_info_loyo | no | n/a | 65 | Issue 5 leave-one-year-out predictor summary |
| `loyo_selection_forward_chaining_safe` | selection_info_loyo | no | n/a | 0 | Issue 5 leave-one-year-out predictor summary |
| `is_eligible_crop` | eligibility_flag | no | n/a | 0 | Eligibility and usability flags |
| `has_valid_yield_anomaly` | eligibility_flag | no | n/a | 0 | Eligibility and usability flags |
| `usable_for_forward_chaining_eval` | eligibility_flag | no | n/a | 0 | Eligibility and usability flags |
| `yield_value_status` | target_quality_flag | no | n/a | 0 | Quality flag for the yield value or the yield z-score |
| `has_valid_yield_zscore` | target_quality_flag | no | n/a | 0 | Quality flag for the yield value or the yield z-score |
| `yield_zscore_low_confidence` | target_quality_flag | no | n/a | 0 | Quality flag for the yield value or the yield z-score |
| `fc_selection_train_years` | selection_info_forward_chaining | no | n/a | 0 | Per-crop weather predictors selected with Issue 5's method using only target years before this row's year |
| `fc_selection_n_training_rows` | selection_info_forward_chaining | no | n/a | 0 | Per-crop weather predictors selected with Issue 5's method using only target years before this row's year |
| `fc_selection_status` | selection_info_forward_chaining | no | n/a | 0 | Per-crop weather predictors selected with Issue 5's method using only target years before this row's year |
| `fc_n_selected_predictors` | selection_info_forward_chaining | no | n/a | 455 | Per-crop weather predictors selected with Issue 5's method using only target years before this row's year |
| `fc_selected_predictors` | selection_info_forward_chaining | no | n/a | 0 | Per-crop weather predictors selected with Issue 5's method using only target years before this row's year |

Weather windows (6): `annual_cal`, `annual_crop`, `kharif`, `ne`, `rabi_next`, `rabi_prev`. Indicators (13): `cold_days`, `dtr_mean`, `gdd_sum`, `hot_days`, `longest_dry`, `max_5day_rain`, `rain_sum`, `rainy_days`, `rh_mean`, `solar_mean`, `tmax_mean`, `tmin_mean`, `wind_mean`.

**Timing caveat for the weather columns:** Baseline-leakage safe only. Window dates are not defined in the source, and annual/ne/rabi_next windows may extend past the crop's harvest, so availability at a forecast origin is NOT established. Treat as retrospective unless the window end is confirmed to precede the forecast origin.

**Unusable feature columns** (listed so nobody feeds them to a model by mistake): `wanom__kharif__cold_days` (constant (zero variance): carries no information), `wz__kharif__cold_days` (100% missing), `wact__kharif__cold_days` (constant (zero variance): carries no information)

<!-- COLUMN_DICTIONARY_END -->

## Validation and leakage checks (36 of 36 pass)

Recorded in `validation_checks_v1.csv`.

- **Keys and counts:** all inputs unique at their own grain; output unique on (`district_code`, `year`, `crop`); row count equals the eligible crop-long rows derived from the inputs; weather deduplicated to one row per district-year.
- **Missing values:** weather anomaly NaN counts equal the source; yield anomaly NaN counts equal the source; no anomaly of exactly zero is created where the status is not `calculated`. Missing source data, insufficient history and a true zero stay distinct.
- **Leakage (forward chaining):**
  - every weather baseline's training years are strictly earlier than the target year;
  - every yield baseline's training years are strictly earlier than the target year;
  - every available weather normal is **recomputed** from its listed earlier years and matches;
  - anomaly equals actual minus normal;
  - the safe set is checked by **provenance**: it must come only from the weather join, `climate_zone` and `crop`, and must not overlap any column sourced from the yield file, any target-year outcome column, or any Issue 5 (`loyo_*`) column;
  - every column is documented and every weather feature carries the timing caveat;
  - the `fc_*` selections train only on target years before the row's year, and are unchanged when later years' targets are scrambled.
- **Completeness (nothing silently absent):**
  - every crop-long row is either carried or listed in `excluded_crop_rows_v1.csv` (carried + excluded = all rows, no overlap);
  - every carried row states why it has or lacks a yield target;
  - rows whose raw area, production and yield are all 0 and are not flagged as filled are labelled `zero_ambiguous` and equal an independent recount from crop-long (source values are kept, not changed);
  - the low-confidence z-score flag equals "the baseline has fewer than 3 observations";
  - the leave-one-year-out weather companion equals an independent recomputation (actual minus the mean of the other years);
  - the missing-data report reconciles to the excluded rows.
- **Mutation-tested:** the builder was run on tampered inputs (a baseline containing the target year, a tampered normal, a duplicate key) and exited non-zero each time.

## Known limitations

- 13 districts and 5 years; forward-chaining targets and anomalies exist only for 2017–2019. The 384 usable rows contain only 39 distinct weather observations (weather repeats across crops), so effective sample size is far smaller than the row count.
- One kept row has `anomaly_status = zero_baseline_std`: its kg/ha anomaly is computable but its z-score is undefined (the baseline has no spread). It is counted in the 384 rows with a valid anomaly; drop it if a z-score target is used.
- 2015 and 2016 have no forward-chaining anomalies; 2017 `rabi_prev` is missing because the 2015 `rabi_prev` source values are missing. These are kept as missing in the main table; `weather_loyo_anomalies_v1.csv` has leave-one-year-out anomalies for those years for retrospective use only.
- **Raw zeros:** 35 carried rows record area, production and yield as 0 without the source flagging them as filled, so "not grown" cannot be told from a true zero. They are labelled `zero_ambiguous` and are never used as baselines (`eligible_for_baseline` is false), but do not read their raw yield as a real zero.
- **Z-scores:** the 2017 yield z-scores rest on a 2-observation baseline (128 rows flagged `yield_zscore_low_confidence`) and are far less stable than 2018 (3 observations) or 2019 (4); the largest is 122. Prefer the kg/ha anomaly, or exclude flagged rows. `wz__` columns are also undefined where a weather feature has zero baseline spread (mostly the cold-days indicators); the three `kharif__cold_days` columns carry no information and are marked unusable in the dictionary.
- **Yield versus production over area:** the reported `yield_kg_per_ha` differs from `production_1000_tons / area_1000_ha x 1000` by more than 1% on 105 rows, almost all with an area under 500 ha. The source rounds area and production to two decimals, so the gap is rounding, not an error; `flag_yield_mismatch` fires only above about 5% (31 rows, all flagged). `yield_kg_per_ha` is the source-reported value and is not recomputed. `flag_low_area_lt_1000ha` means 0 < area < 1000 ha (zero area is a separate flag).
- **Not carried by design:** crops that are not eligible (see `excluded_crop_rows_v1.csv`; cotton, castor and kharif sorghum are only *conditionally* eligible), and the weather normals, percent anomalies and stress heuristics from Issue 3.
- `kharif` and `ne` windows overlap. They are not independent and must not be added.
- Weather units are not fully defined in the source.
- Issue 5 and Issue 2 eligible-crop sets differ (see Known gap).
- Reproducibility is for the pinned input files (SHA-256 recorded in `reproducibility_report_v1.json`) and, for byte-identical output, the pinned package versions.
- Weather timing relative to a forecast origin is not established (see the caveat above).
