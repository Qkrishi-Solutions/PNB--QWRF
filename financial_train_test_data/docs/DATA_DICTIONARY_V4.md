# Data Dictionary: synthetic_training_v4

> **SYNTHETIC. Not PNB data.** Generated from `v4/synthetic_training_v4_data_dictionary.csv`; that CSV is the source of truth.

44 columns in the rich files (`synthetic_training_v4_train.csv` / `_test.csv`). The `Financial_data_PNB_SYNTHETIC_*.csv` files hold only the first 20 original fields (see `PNB_FIELD_DICTIONARY_DRAFT.md`).

## Role legend

| Role | Meaning |
|---|---|
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
| 1 | `split` | `split` | str | 0.0% | train or test (out-of-time: by snapshot year). |
| 2 | `snapshot_year` | `feature` | int64 | 0.0% | Reporting year of the snapshot; joins that year's weather. |
| 3 | `reporting_date` | `feature_raw_date` | str | 0.0% | Snapshot reporting date. |
| 4 | `zone` | `feature` | str | 0.0% | Constant placeholder zone. |
| 5 | `circle` | `feature` | str | 0.0% | Placeholder circle grouping. |
| 6 | `branch ID` | `identifier_exclude` | str | 0.0% | Placeholder branch id; high cardinality. |
| 7 | `branch name` | `identifier_exclude` | str | 0.0% | Placeholder branch name. |
| 8 | `customer ID` | `identifier_exclude` | str | 0.0% | Synthetic customer id. |
| 9 | `customer name` | `identifier_exclude` | str | 0.0% | Synthetic name. |
| 10 | `borrowers address` | `identifier_exclude` | str | 0.0% | Synthetic address. |
| 11 | `pincode` | `geo_id` | int64 | 0.0% | Pincode mapped to the district. |
| 12 | `account number` | `identifier_exclude` | int64 | 0.0% | Synthetic account number. |
| 13 | `account open date` | `feature_raw_date` | str | 0.0% | Account open date. |
| 14 | `facility type` | `feature` | str | 0.0% | Facility type. |
| 15 | `scheme code` | `feature` | str | 0.0% | Scheme code. |
| 16 | `agri sector` | `feature` | str | 0.0% | Agricultural sector. |
| 17 | `sanction amount` | `feature` | int64 | 0.0% | Sanctioned amount (INR). |
| 18 | `rate of interest` | `feature` | float64 | 0.0% | Interest rate (%). |
| 19 | `balance outstanding` | `feature` | int64 | 0.0% | Outstanding balance (INR). |
| 20 | `loan_age_days` | `feature` | int64 | 0.0% | Days from account open date to reporting date. |
| 21 | `district_code` | `feature` | int64 | 0.0% | Project district code. |
| 22 | `district_name` | `feature` | str | 0.0% | District name. |
| 23 | `climate_zone` | `feature` | str | 0.0% | Issue 6 climate zone: similarity group, not a risk level. |
| 24 | `wx_kharif__rain_sum_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of kharif__rain_sum; blank when unavailable. Not imputed. |
| 25 | `wx_kharif__longest_dry_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of kharif__longest_dry; blank when unavailable. Not imputed. |
| 26 | `wx_annual_crop__hot_days_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of annual_crop__hot_days; blank when unavailable. Not imputed. |
| 27 | `wx_kharif__tmax_mean_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of kharif__tmax_mean; blank when unavailable. Not imputed. |
| 28 | `wx_kharif__dtr_mean_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of kharif__dtr_mean; blank when unavailable. Not imputed. |
| 29 | `wx_ne__rainy_days_z` | `feature` | float64 | 0.0% | Forward-chaining anomaly z-score of ne__rainy_days; blank when unavailable. Not imputed. |
| 30 | `crop` | `feature` | str | 54.9% | Crop for Crop Production accounts only; else blank. |
| 31 | `yield_anomaly_zscore` | `feature` | float64 | 55.0% | Forward-chaining yield anomaly z-score; blank when not calculated. Not imputed. |
| 32 | `yield_anomaly_status` | `feature` | str | 0.0% | Status of the yield anomaly (calculated, insufficient_history, ...). |
| 33 | `crop_insurance_available` | `feature` | str | 0.0% | Y or N. |
| 34 | `insurance_amount` | `feature` | float64 | 55.1% | Insured amount (INR). |
| 35 | `insurance_coverage_status` | `feature` | str | 0.0% | Active, Expired, Pending or Not Insured. |
| 36 | `insurance_end_date` | `feature_raw_date` | str | 55.1% | Policy end. |
| 37 | `insurance_days_to_expiry` | `feature` | float64 | 55.1% | Policy end minus reporting date (negative = expired). |
| 38 | `insurance_coverage_ratio` | `feature` | float64 | 55.1% | Insured amount / outstanding balance. |
| 39 | `insurance_claim_status` | `post_event_exclude` | str | 0.0% | Claim outcome after the weather event; excluded for early-warning use. |
| 40 | `insurance_payout_amount` | `post_event_exclude` | float64 | 95.9% | Payout amount. |
| 41 | `days_past_due` | `leakage_exclude` | int64 | 0.0% | Determined by the asset class (target leakage). |
| 42 | `overdue amount` | `leakage_exclude` | int64 | 0.0% | Determined by the asset class (target leakage). |
| 43 | `asset class` | `target` | str | 0.0% | Standard, SMA 0, SMA 1, SMA 2, NPA (ordered). |
| 44 | `npa_flag` | `target_binary` | int64 | 0.0% | 1 if asset class is NPA. |
