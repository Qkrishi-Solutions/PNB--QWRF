# Issue #7: Integrated ML-Ready Dataset

One integrated table, **one row per `district_code` + `year` + `crop`**, that joins the engineered crop, weather, yield, selected-predictor and climate-zone data. It is the **single input for the forecasting issues (#49 to #52)**. Nothing is imputed: missing stays missing.

> This is a data-preparation deliverable. It makes no claim about prediction skill. Weather–yield relationships in it are statistical associations, not causation.

## Run

From the repository root (Python 3.12, `pandas`, `numpy`):

```bash
python ml_ready_dataset/build_ml_ready_dataset.py
python -m pytest ml_ready_dataset/tests -v
```

The build stops with a non-zero exit if any validation or leakage check fails.

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
| `ml_ready_dataset_v1.csv` | the dataset (715 rows × 307 columns) |
| `ml_ready_column_dictionary_v1.csv` | every column: role, description, forward-chaining-safe flag, leakage note, missing count |
| `join_coverage_report_v1.csv` | matched and unmatched keys for every join |
| `unmatched_keys_v1.csv` | the unmatched keys themselves |
| `exclusion_report_v1.csv` | crop-long rows excluded and why |
| `validation_checks_v1.csv` | all validation, leakage and coverage checks with results |
| `reproducibility_report_v1.json` | exact command, row counts, package versions, SHA-256 of inputs and output |

## How rows are selected

Base rows are crop-long rows whose crop has `proceed_to_modelling = True` (Issue 2) and `use_in_modelling = True` (Issue 1). The expected count is **derived from the inputs**, not hard-coded. In the committed run: 1,430 crop-long rows, 11 eligible crops, **715 rows kept**, 715 excluded (crops that are ineligible or only conditionally eligible).

Rows are **not** dropped for having a missing yield anomaly. Use the flags to filter:

- `has_valid_yield_anomaly`: a forward-chaining yield anomaly exists (384 rows; 2017–2019 only).
- `usable_for_forward_chaining_eval`: eligible crop and valid anomaly (384 rows; 128 in each of 2017, 2018, 2019).

## Join coverage (committed run)

| Join | Keys | Base keys matched | Unmatched base keys |
|---|---|---|---|
| eligibility | `crop` | 11 / 11 | 0 |
| yield anomaly | `district_code`+`year`+`crop` | 715 / 715 | 0 |
| weather (district-year, deduplicated) | `district_code`+`year` | 65 / 65 | 0 |
| climate zone | `district_code` | 13 / 13 | 0 |
| crop feature selection (Issue 5) | `crop` | **10 / 11** | **1: `rabi_sorghum`** |

**Known gap:** `rabi_sorghum` is eligible under Issue 2 but has no Issue 5 result, and Issue 5 contains seven crops (castor, cotton, linseed, rapeseed_and_mustard, safflower, soyabean, wheat) that Issue 2 does not mark eligible. The two issues used different eligible sets. Its `loyo_*` columns stay **missing** for `rabi_sorghum`; nothing is filled in. This should be reconciled upstream.

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
| selection_info_loyo | 4 | Issue 5 predictor summary (see warning) |
| eligibility_flag / data_quality_flag / metadata | 6 / 21 / 16 | filtering and traceability |

**Forward-chaining safe feature set (236 columns):** the `wanom__`, `wz__` and `wact__` columns, plus `climate_zone` and `crop`. Everything else is a target, a target-year outcome, or selection information and must not be used as a feature.

> **Warning on `loyo_*` columns.** Issue 5 selected predictors with leave-one-year-out folds over 2017–2019, so the selection saw the held-out year and later years. They are included because the issue asks for them, but they are **not forward-chaining safe**. Re-select features inside each forward-chaining fold.

The same-year raw weather columns that crop-long also carries are dropped after verifying they are identical to `wact__*`.

## Validation and leakage checks (25 of 25 pass)

Recorded in `validation_checks_v1.csv`.

- **Keys and counts:** all inputs unique at their own grain; output unique on (`district_code`, `year`, `crop`); row count equals the eligible crop-long rows derived from the inputs; weather deduplicated to one row per district-year.
- **Missing values:** weather anomaly NaN counts equal the source; yield anomaly NaN counts equal the source; no anomaly of exactly zero is created where the status is not `calculated`. Missing source data, insufficient history and a true zero stay distinct.
- **Leakage (forward chaining):**
  - every weather baseline's training years are strictly earlier than the target year;
  - every yield baseline's training years are strictly earlier than the target year;
  - every available weather normal is **recomputed** from its listed earlier years and matches;
  - anomaly equals actual minus normal;
  - the safe feature set contains no target-derived or `loyo_*` columns;
  - `loyo_*` is flagged not forward-chaining safe.

## Known limitations

- 13 districts and 5 years; forward-chaining targets and anomalies exist only for 2017–2019. The 384 usable rows contain only 39 distinct weather observations (weather repeats across crops), so effective sample size is far smaller than the row count.
- One kept row has `anomaly_status = zero_baseline_std`: its kg/ha anomaly is computable but its z-score is undefined (the baseline has no spread). It is counted in the 384 rows with a valid anomaly; drop it if a z-score target is used.
- 2015 and 2016 have no forward-chaining anomalies; 2017 `rabi_prev` is missing because the 2015 `rabi_prev` source values are missing. These are kept as missing.
- `kharif` and `ne` windows overlap. They are not independent and must not be added.
- Weather units are not fully defined in the source.
- Issue 5 and Issue 2 eligible-crop sets differ (see Known gap).
- Reproducibility is for the pinned input files (SHA-256 recorded in `reproducibility_report_v1.json`).
