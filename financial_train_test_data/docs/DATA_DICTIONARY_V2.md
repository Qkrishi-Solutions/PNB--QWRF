# Data Dictionary: synthetic_training_v2

> **SYNTHETIC. Not PNB data.** Generated from `v2/synthetic_training_v2_data_dictionary.csv`; that CSV is the source of truth.

52 columns in the rich files (`synthetic_training_v2_train.csv` / `_test.csv`). The `Financial_data_PNB_SYNTHETIC_*.csv` files hold only the first 20 original fields (see `PNB_FIELD_DICTIONARY_DRAFT.md`).

## Role legend

| Role | Meaning |
|---|---|
| `label` | Marker column. |
| `split` | train or test. |
| `feature` | Model input. |
| `feature_raw_date` | Raw date; prefer the derived day-count features. |
| `identifier_exclude` | Identifier or personal-style field. Do not use as a feature. |
| `geo_id` | Location id. Do not use as a feature. |
| `post_event_exclude` | Observed after the weather event. Exclude for early-warning use. |
| `leakage_exclude` | Determined by the target. Exclude (target leakage). |
| `target` | Label to predict. |
| `target_binary` | Binary convenience label (NPA or not). |

## Columns

| # | Column | Role | Type | Blank in train | Description |
|---|---|---|---|---|---|
| 1 | `synthetic_flag` | `label` | str | 0.0% | Always SYNTHETIC. |
| 2 | `split` | `split` | str | 0.0% | train or test (out-of-time: by snapshot year). |
| 3 | `snapshot_year` | `feature` | int64 | 0.0% | Reporting year of the snapshot; joins that year's weather. |
| 4 | `reporting_date` | `feature_raw_date` | str | 0.0% | Snapshot reporting date. |
| 5 | `zone` | `feature` | str | 0.0% | Constant placeholder zone. |
| 6 | `circle` | `feature` | str | 0.0% | Placeholder circle grouping. |
| 7 | `branch ID` | `identifier_exclude` | str | 0.0% | Placeholder branch id; high cardinality. |
| 8 | `branch name` | `identifier_exclude` | str | 0.0% | Placeholder branch name. |
| 9 | `customer ID` | `identifier_exclude` | str | 0.0% | Synthetic customer id. |
| 10 | `customer name` | `identifier_exclude` | str | 0.0% | Synthetic name. |
| 11 | `borrowers address` | `identifier_exclude` | str | 0.0% | Synthetic address. |
| 12 | `pincode` | `geo_id` | int64 | 0.0% | Pincode mapped to the district. |
| 13 | `account number` | `identifier_exclude` | int64 | 0.0% | Synthetic account number. |
| 14 | `account open date` | `feature_raw_date` | str | 0.0% | Account open date. |
| 15 | `facility type` | `feature` | str | 0.0% | Facility type. |
| 16 | `scheme code` | `feature` | str | 0.0% | Scheme code. |
| 17 | `agri sector` | `feature` | str | 0.0% | Agricultural sector. |
| 18 | `sanction amount` | `feature` | int64 | 0.0% | Sanctioned amount (INR). |
| 19 | `rate of interest` | `feature` | float64 | 0.0% | Interest rate (%). |
| 20 | `balance outstanding` | `feature` | int64 | 0.0% | Outstanding balance (INR). |
| 21 | `loan_age_days` | `feature` | int64 | 0.0% | Days from account open date to reporting date. |
| 22 | `district_code` | `feature` | int64 | 0.0% | Project district code. |
| 23 | `district_name` | `feature` | str | 0.0% | District name. |
| 24 | `climate_zone` | `feature` | str | 0.0% | Issue 6 climate zone: similarity group, not a risk level. |
| 25 | `wx_kharif__rain_sum_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of kharif__rain_sum; blank when unavailable. Not imputed. |
| 26 | `wx_kharif__longest_dry_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of kharif__longest_dry; blank when unavailable. Not imputed. |
| 27 | `wx_annual_crop__hot_days_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of annual_crop__hot_days; blank when unavailable. Not imputed. |
| 28 | `wx_kharif__tmax_mean_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of kharif__tmax_mean; blank when unavailable. Not imputed. |
| 29 | `wx_kharif__dtr_mean_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of kharif__dtr_mean; blank when unavailable. Not imputed. |
| 30 | `wx_ne__rainy_days_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of ne__rainy_days; blank when unavailable. Not imputed. |
| 31 | `crop` | `feature` | str | 95.9% | Crop for Crop Production accounts only; else blank. |
| 32 | `yield_anomaly_zscore` | `feature` | float64 | 95.9% | Forward-chaining yield anomaly z-score; blank when not calculated. Not imputed. |
| 33 | `yield_anomaly_status` | `feature` | str | 0.0% | Status of the yield anomaly (calculated, insufficient_history, ...). |
| 34 | `crop_insurance_available` | `feature` | str | 0.0% | Y or N. |
| 35 | `insurance_scheme` | `feature` | str | 80.5% | Synthetic scheme code. |
| 36 | `insurance_amount` | `feature` | float64 | 80.5% | Insured amount (INR). |
| 37 | `insurance_premium` | `feature` | float64 | 80.5% | Premium (INR). |
| 38 | `insurance_provider` | `feature` | str | 80.5% | Synthetic insurer. |
| 39 | `insurance_coverage_status` | `feature` | str | 0.0% | Active, Expired, Pending or Not Insured. |
| 40 | `insurance_start_date` | `feature_raw_date` | str | 80.5% | Policy start. |
| 41 | `insurance_end_date` | `feature_raw_date` | str | 80.5% | Policy end. |
| 42 | `insurance_policy_id` | `identifier_exclude` | str | 80.5% | Synthetic policy id. |
| 43 | `insurance_days_to_expiry` | `feature` | float64 | 80.5% | Policy end minus reporting date (negative = expired). |
| 44 | `insurance_coverage_ratio` | `feature` | float64 | 80.5% | Insured amount / outstanding balance. |
| 45 | `insurance_claim_status` | `post_event_exclude` | str | 0.0% | Claim outcome after the weather event; excluded for early-warning use. |
| 46 | `insurance_claim_amount` | `post_event_exclude` | float64 | 96.8% | Claim amount. |
| 47 | `insurance_payout_amount` | `post_event_exclude` | float64 | 98.6% | Payout amount. |
| 48 | `insurance_payout_date` | `post_event_exclude` | str | 98.6% | Payout date. |
| 49 | `days_past_due` | `leakage_exclude` | int64 | 0.0% | Determined by the asset class (target leakage). |
| 50 | `overdue amount` | `leakage_exclude` | int64 | 0.0% | Determined by the asset class (target leakage). |
| 51 | `asset class` | `target` | str | 0.0% | Standard, SMA 0, SMA 1, SMA 2, NPA (ordered). |
| 52 | `npa_flag` | `target_binary` | int64 | 0.0% | 1 if asset class is NPA. |
