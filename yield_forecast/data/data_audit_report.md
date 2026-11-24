# Issue #50 — Comprehensive Input Data Audit Report

**Date:** 24 November 2026
**Context:** QWeatherRiskF / FinQrop Agricultural Risk Platform
**Target:** GitHub Issue #50 (ML Yield Forecasting Pipeline)
**Branch:** `issue-50-yield-forecast`

---

## 1. Executive Summary

This report documents the verification and statistical audit of the input datasets available in the repository for training and evaluating predictive crop yield-anomaly models (Ridge, RandomForest, XGBoost, and CatBoost).

All findings are based on actual inspected files on branch `issue-50-yield-forecast`. No synthetic or real data were altered or fabricated.

---

## 2. Inventory and Verification of Input Datasets

| Dataset Identifier | Canonical File Path | Format / Shape | Unique Keys Verified | Primary Function in Issue #50 |
|---|---|---|---|---|
| **Crop Baseline** | `CropLongFormatv1/crop_long_v1.csv` | Wide-long, 1,430 rows $\times$ 124 cols | `(district_code, crop, year)` (1,430 unique) | Base agricultural production & raw weather |
| **Yield Baseline & Anomalies** | `yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv` | Long tabular, 1,430 rows $\times$ 139 cols | `(district_code, crop, year)` (1,430 unique) | Forward-chaining yield anomaly target (`yield_anomaly_kg_per_ha`) |
| **Weather Anomalies** | `weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv` | Long format, 5,070 rows $\times$ 22 cols | `(district_code, year, weather_column)` (5,070 unique) | 78 forward-chaining weather anomaly features |
| **Climate Zones** | `District Climate Zonal Clustering/district_zone_map_v1.csv` | Tabular, 13 rows $\times$ 5 cols | `district_code` (13 unique) | $K=3$ Climate zone clustering context |
| **Crop Eligibility** | `CropModelEligibility/crop_model_eligibility_v1.csv` | Tabular, 22 rows $\times$ 25 cols | `crop` (22 unique) | Crop screening filter (11 eligible crops) |
| **Feature Selection Reference** | `crop specific weather feature selection/crop_feature_selection_v1.csv` | Tabular, 1,326 rows $\times$ 27 cols | `(crop, feature)` (1,326 unique) | Issue #5 reference benchmark for feature stability |

---

## 3. Detailed Data Integrity & Schema Checks

### 3.1 Composite Key Uniqueness
- **Agricultural Primary Key:** `(district_code, crop, year)`.
  - In `crop_long_v1.csv`: 1,430 records, exactly 1,430 unique keys. Duplicates: **0**.
  - In `yield_anomalies_forward_chaining_v1.csv`: 1,430 records, exactly 1,430 unique keys. Duplicates: **0**.
- **Weather Primary Key:** `(district_code, year, weather_column)`.
  - In `weather_anomalies_forward_chaining_v1.csv`: 5,070 records (13 districts $\times$ 5 years $\times$ 78 indicators). Duplicates: **0**.
- **Climate Zone Key:** `district_code` (13 districts). Duplicates: **0**.
- **Crop Eligibility Key:** `crop` (22 crops). Duplicates: **0**.

### 3.2 Geographic and Temporal Coverage
- **Districts (13 historical Andhra Pradesh districts):**
  - Codes: `502, 503, 504, 505, 506, 510, 511, 515, 517, 519, 520, 521, 523`.
  - Names: Anantapuramu, Chittoor, East Godavari, Guntur, Krishna, Kurnool, Prakasam, SPR Nellore, Srikakulam, Visakhapatnam, Vizianagaram, West Godavari, YSR.
  - Coverage across all 13 districts is 100% complete across all inputs.
- **Years (5 years):** `2015, 2016, 2017, 2018, 2019`.

### 3.3 Crop Eligibility (Issue #2 Decisions)
Of the 22 agricultural crop categories:
- **11 Crops are Eligible (`proceed_to_modelling == True`, `modelling_role == include`):**
  - `groundnut` (65 total obs, 39 forward-chaining anomaly obs)
  - `maize` (65 total obs, 39 forward-chaining anomaly obs)
  - `rice` (65 total obs, 39 forward-chaining anomaly obs)
  - `sugarcane` (65 total obs, 39 forward-chaining anomaly obs)
  - `pigeonpea` (65 total obs, 39 forward-chaining anomaly obs)
  - `sesamum` (65 total obs, 36 forward-chaining anomaly obs)
  - `finger_millet` (65 total obs, 30 forward-chaining anomaly obs)
  - `chickpea` (65 total obs, 33 forward-chaining anomaly obs)
  - `pearl_millet` (65 total obs, 33 forward-chaining anomaly obs)
  - `sunflower` (65 total obs, 32 forward-chaining anomaly obs)
  - `rabi_sorghum` (65 total obs, 26 forward-chaining anomaly obs)
- **3 Crops are Conditionally Eligible (`proceed_to_modelling == False`, borderline $N$):**
  - `cotton` (32 anomaly obs), `kharif_sorghum` (32 anomaly obs), `castor` (18 anomaly obs).
- **8 Crops are Ineligible:**
  - `oilseeds` (aggregate), `sorghum` (aggregate), `minor_pulses` (group category), `soyabean`, `rapeseed_and_mustard`, `linseed`, `safflower`, `wheat`.

### 3.4 Missingness Distribution
- **Raw Yield (`yield_kg_per_ha`):**
  - 1,404 valid values out of 1,430 (26 missing values = 1.8%, occurring exclusively in `minor_pulses` in 2018 and 2019 due to non-reporting by state DES).
- **Forward-Chaining Yield Anomaly (`yield_anomaly_kg_per_ha`):**
  - 2015: 286 / 286 missing (100% missing; requires $\ge 2$ prior training years).
  - 2016: 286 / 286 missing (100% missing; only 1 prior training year).
  - 2017: 121 missing, 165 calculated (108 ineligible crop rows, 13 insufficient local history).
  - 2018: 123 missing, 163 calculated (119 ineligible crop rows, 4 insufficient local history).
  - 2019: 123 missing, 163 calculated (115 ineligible crop rows, 8 insufficient local history).
  - Total valid anomaly observations across all crops: **491**.
  - Total valid anomaly observations for the 11 eligible crops: **384** (128 in 2017, 128 in 2018, 128 in 2019).
- **Weather Anomaly Predictors (`wanom__*`):**
  - 2015 & 2016: 100% missing in forward-chaining mode due to baseline history requirements ($\ge 2$ years).
  - 2017: 845 available, 169 missing (the 13 `rabi_prev` indicators across 13 districts because 2015 rabi weather is unobserved).
  - 2018 & 2019: 100% available (1,014 / 1,014).

---

## 4. Leakage and Forecast-Horizon Timing Analysis

### 4.1 Strict Leakage Controls Enforced
1. **Target Leakage Prohibition:** No target-year raw yield (`yield_kg_per_ha`), target-year production, or retrospective leave-one-year-out anomaly is ever used as a predictor.
2. **Temporal Leakage Prohibition:** Preprocessing transformations (scalers, encoders) and feature selection are fitted **strictly on training fold years** ($t_{train} < t_{test}$).
3. **No Retrospective Feature Selection:** Feature selection from Issue #5 used LOYO across 2017–2019, which exposed held-out test years. In Issue #50, feature screening is computed dynamically inside each training fold using training observations only.

### 4.2 Forecast Origin and Information Timing
- **The Operational Challenge:**
  - Standard agricultural seasons in Andhra Pradesh:
    - Kharif: Sown June–July, harvested October–December.
    - Rabi: Sown October–November, harvested February–April.
  - The 78 weather indicators include full calendar year summaries (`annual_cal`), crop year summaries (`annual_crop`), and post-kharif weather (`ne`, `rabi_next`).
  - If a model uses `annual_cal__rain_sum` or `ne__*` to predict kharif yield, this represents an **end-of-season yield-weather sensitivity analysis (diagnostic/hindcast)**, NOT an advance early-warning forecast.
  - An advance early-warning forecast made at sowing (July) can only access pre-sowing weather (prior rabi / pre-monsoon).
- **Methodological Distinction:**
  Issue #50 evaluates whether weather anomalies improve yield-anomaly estimation. Because the baseline dataset aggregates weather to seasonal and annual windows, this study demonstrates predictive association assuming seasonal weather realizations are known, while strictly preserving time-series forward-chaining across calendar years.
