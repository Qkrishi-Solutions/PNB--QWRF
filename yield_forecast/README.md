# Issue #50 — ML Yield-Anomaly Response Estimation Pipeline

**Project:** PNB QWeatherRiskF (FinQrop Agricultural Risk Platform)
**Branch:** `issue-50-yield-forecast`
**Target PR Branch:** `dev`
**Module:** `yield_forecast`
**Operational Taxonomy:** End-of-season retrospective weather–yield response estimation (diagnostic hindcast)

---

> [!IMPORTANT]
> **Scientific & Operational Limitation Notice:**
> **The current five-year dataset and two held-out test years do not establish generalizable predictive superiority over climatology. The experiment is an exploratory retrospective weather–yield response estimation, not a validated operational early-warning model.**
> Furthermore, observed weather/yield empirical associations are correlational and **do not constitute proof of causation**.

---

## 1. Executive Summary & Problem Formulation

This module addresses GitHub Issue #50: evaluating whether weather-derived indicators improve the prediction of agricultural yield anomalies over a historical climatology baseline across the 13 districts of Andhra Pradesh.

### Operational Taxonomy & Information Horizon
- **Retrospective Response Estimation:** The weather anomaly indicators used (`wanom__annual_cal__*`, `wanom__kharif__*`, `wanom__ne__*`, `wanom__rabi_next__*`) span full calendar years, monsoon seasons, and post-monsoon periods.
- For Kharif crops harvested in September–November, `annual_cal` includes December weather, and `rabi_next` extends to April of the subsequent calendar year.
- At an advance operational pre-sowing origin (May–June) or mid-season origin (August), these weather observations do not yet exist.
- Therefore, this pipeline models the **retrospective statistical response of crop yields to season-long weather anomalies (diagnostic hindcast)**, not an advance pre-harvest early-warning forecast.

---

## 2. Directory Structure

```text
yield_forecast/
├── README.md                                    # This guide and technical reference
├── requirements.txt                             # Module-specific dependencies
├── build_ml_dataset.py                          # Compiles canonical inputs into ML-ready dataset
├── train_forecast.py                            # Forward-chaining evaluation pipeline
├── data/
│   ├── data_audit_report.md                     # Upstream input audit and schema verification
│   └── ml_yield_dataset_v1.csv                  # Integrated ML-ready table (1,430 rows × 305 cols)
├── models/
│   └── model_config.json                        # Hyperparameters, random seeds, and taxonomy metadata
├── results/
│   ├── predictions_forward_chaining_v1.csv      # Out-of-fold predictions & intervals (1,280 rows)
│   ├── fold_evaluation_metrics_v1.csv           # Fold-by-fold MAE, RMSE, and R²
│   ├── aggregate_model_comparison_v1.csv        # Consolidated model comparison with p-values
│   ├── crop_level_evaluation_metrics_v1.csv     # Granular per-crop evaluation breakdown
│   └── prediction_interval_evaluation_v1.csv    # Empirical test coverage vs 80% nominal target
├── docs/
│   └── issue_50_final_report.md                 # Complete technical findings and audit decisions
└── tests/
    ├── conftest.py                              # Synthetic data fixture for unit testing
    └── test_yield_forecast.py                   # Automated test suite (16 test functions)
```

---

## 3. Dataset Construction & Exclusion Accounting

The dataset generator (`build_ml_dataset.py`) merges five canonical repository inputs without row duplication or silent data loss:
1. `CropLongFormatv1/crop_long_v1.csv` (1,430 rows)
2. `yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv` (1,430 rows)
3. `District Climate Zonal Clustering/district_zone_map_v1.csv` (13 rows)
4. `CropModelEligibility/crop_model_eligibility_v1.csv` (22 rows)
5. `weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv` (5,070 long records pivoted to 65 district-year rows × 158 indicators)

### Record Accounting & Usable Sample Counts:
- **Base Table:** Exactly 1,430 rows ($13 \text{ districts} \times 22 \text{ crops} \times 5 \text{ years: 2015–2019}$).
- **Primary Key:** `['district_code', 'crop', 'year']` (0 duplicate keys).
- **Eligible Crop Records:** 715 rows (11 eligible crops with `proceed_to_modelling == True`: `groundnut`, `maize`, `rice`, `sugarcane`, `pigeonpea`, `sesamum`, `finger_millet`, `chickpea`, `pearl_millet`, `sunflower`, `rabi_sorghum`).
- **Ineligible / Conditional Crop Records:** 715 rows (11 crops with `proceed_to_modelling == False`: `kharif_sorghum`, `castor`, `cotton`, `oilseeds`, `sorghum`, `minor_pulses`, `soyabean`, `rapeseed_and_mustard`, `linseed`, `safflower`, `wheat`).
- **Forward-Chaining Target Availability:** Forward anomaly calculation requires $\ge 2$ historical prior years.
  - Years 2015 & 2016: 0 valid forward anomalies (strictly `NaN`, never imputed as 0).
  - Years 2017, 2018, 2019: Exactly 128 usable records per year.
- **Usable Evaluation Records:** Exactly **384 records** ($128 \times 3 \text{ years}$).
- **Explanation of Discrepancy from $11 \times 13 = 143$:**
  Across Andhra Pradesh, 15 district-crop pairs are naturally uncultivated or have no official historical production reports in DES AP data (e.g., chickpea in coastal Srikakulam/Chittoor; finger millet in delta canal tracts Krishna/West Godavari; rabi sorghum in North Coastal districts). Thus, exactly 128 observations exist in each target year.

---

## 4. Validation Design: Why Only Two Folds

Given the 5-year timeline (2015–2019) and the minimum requirement of 2 prior years to establish forward-chaining normals:
- **2015–2016:** Pure baseline history ($T < 2$). Anomaly target is undefined (`NaN`).
- **2017:** First year with valid forward anomaly targets ($N=128$). Serves as the initial training set.
- **2018:** Second anomaly year ($N=128$).
- **2019:** Third anomaly year ($N=128$).

Under strict prospective forward chaining without lookahead:
- **Fold 1:** Train strictly on 2017 ($N=128$) $\to$ Test on 2018 ($N=128$).
- **Fold 2:** Train strictly on 2017 + 2018 ($N=256$) $\to$ Test on 2019 ($N=128$).
- Exactly **two forward-chaining test folds** are mathematically possible.
- Total out-of-fold test instances per model: **256 observations**.

---

## 5. Model Comparison & Statistical Evaluation

All five models are evaluated on the exact same 256 held-out out-of-fold instances:

| Model | Evaluated Samples | Aggregate MAE (kg/ha) | Aggregate RMSE (kg/ha) | $\Delta$ MAE vs Base | Paired $t$-test $p$-value | Statistically Significant? | Empirical 80% PI Coverage | Performance Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Climatology Baseline** | 256 | **1,341.91** | **3,787.98** | 0.00 | 1.0000 | Reference | 71.9% | Reference Baseline |
| **XGBoost** | 256 | **1,332.01** | **3,764.76** | **-9.91** (-0.74%) | 0.8709 | **NO** | 73.8% | Statistically Indistinguishable ($p=0.87$) |
| **CatBoost** | 256 | 1,368.18 | 4,016.13 | +26.27 (+1.96%) | 0.7623 | **NO** | 70.7% | Inferior to Climatology ($p=0.76$) |
| **RandomForest** | 256 | 1,443.02 | 3,802.90 | +101.11 (+7.53%) | 0.0774 | **NO** | 80.1% | Inferior to Climatology ($p=0.08$) |
| **Ridge Regression** | 256 | 2,017.91 | 3,851.50 | +676.00 (+50.38%) | < 0.0001 | **NO** | 50.4% | Inferior to Climatology ($p < 0.0001$) |

### Honest Statistical Findings:
1. **XGBoost Margin:** While XGBoost achieved an aggregate MAE of 1,332.01 kg/ha compared to Climatology's 1,341.91 kg/ha (a reduction of 9.91 kg/ha or 0.74%), a two-tailed paired $t$-test on absolute residuals reveals $t = -0.163, p = 0.8709$. This delta is **statistically indistinguishable from zero**.
2. **Fold Reversals:** In Fold 1 (2018), XGBoost outperformed Climatology by $-233.82$ kg/ha. In Fold 2 (2019), Climatology outperformed XGBoost by $+214.01$ kg/ha. The models do not demonstrate robust out-of-fold generalization across years.
3. **No Model Beats Climatology:** No classical ML model demonstrates a statistically significant improvement over the crop-mean climatology baseline on this dataset.

---

## 6. Prediction Intervals & Honest Calibration Status

- **Method:** In-sample residual quantiles: $q_{80} = \text{Quantile}_{0.80}(|y_{\text{train}} - \hat{y}_{\text{train}}|)$.
- **Calibration Status:** Honestly designated as **`uncalibrated_in_sample_exploratory`**.
- **Reason for In-Sample Calculation:** In Fold 1, only a single year (2017, $N=128$) is available for training across 11 crops. Carving out a held-out calibration split within a single year would severely degrade model training and produce highly noisy calibration thresholds.
- **Empirical Undercoverage:**
  - Complex models (Ridge: 50.4%, CatBoost: 70.7%, XGBoost: 73.8%) overfit the small training sample, producing artificially narrow residual quantiles that fail to achieve the nominal 80% coverage on out-of-fold test data.
  - Random Forest achieved 80.1% coverage primarily due to aggressive tree constraints (`max_depth=3`, `min_samples_leaf=3`).
- **Recommendation:** True conformal out-of-fold calibration requires longer historical time series ($\ge 15$ years) with dedicated multi-year calibration folds.

---

## 7. Status of Issue #50 Mitigations

| Mitigation Strategy | Status | Location / Implementation Details | Evaluation Finding |
| :--- | :---: | :--- | :--- |
| **1. Extend History (Issue #44)** | **DEFERRED** | Researched in Issue #44 documentation; not yet merged into canonical `crop_long_v1.csv`. | Pending upstream merge of multi-decadal historical series. |
| **2. Climate-Zone Crop Pooling** | **PARTIAL** | Implemented as categorical context feature (`climate_zone` OHE) in `train_forecast.py#L194-L197`. | Conditioning on zone is active; explicit hierarchical multi-task shrinkage deferred due to $N=128$. |
| **3. Shrinkage Toward Trend** | **DEFERRED / BLOCKED** | Climatology acts as shrinkage to crop mean. Explicit trend fitting is blocked by short series ($T=3$). | Estimating a polynomial or linear trend on 2–3 anomaly years introduces severe estimation variance. |

---

## 8. Quickstart & Reproduction Commands

All commands run using Python in the project's virtual environment:

### Step 1: Install Dependencies
```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

### Step 2: Build Integrated ML Dataset
```powershell
.venv\Scripts\python.exe yield_forecast/build_ml_dataset.py
```
*Expected Output: `yield_forecast/data/ml_yield_dataset_v1.csv` ($1,430 \text{ rows} \times 305 \text{ columns}$, SHA-256 `b2e930eda0a1...`)*

### Step 3: Run Training & Forward-Chaining Pipeline
```powershell
.venv\Scripts\python.exe yield_forecast/train_forecast.py
```
*Expected Output: Generates all 5 result CSVs in `yield_forecast/results/`.*

### Step 4: Run Automated Test Suite
```powershell
.venv\Scripts\python.exe -m pytest yield_forecast/tests/test_yield_forecast.py -v
```
*Expected Output: 16 passed in ~10 seconds.*
