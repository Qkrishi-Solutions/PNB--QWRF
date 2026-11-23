# QWeatherRiskF

**Hybrid classical-quantum agricultural risk and decision support pipeline for Andhra Pradesh.**

QWeatherRiskF merges district-level crop statistics with seasonal weather features, forecasts yield anomalies with classical machine learning, converts forecasts into profit and risk scores, and allocates crops under land, water and budget limits using QAOA, benchmarked against Gurobi.

> **Status: INITIAL** The data layer is in progress. The finance layer, risk engine and optimisation are in design. QAOA is evaluated against an exact classical solver and results are reported as measured.

---

## What it does

```
Sources → Ingestion and merge → Features and zones → ML forecast
        → Risk and finance → Optimisation (QAOA, Gurobi benchmark) → Decision support
```

| Stage | What happens |
|---|---|
| 1. Data layer | Crop statistics (ICRISAT), rainfall (IMD), climate (ERA5-Land), prices and costs |
| 2. Ingestion and merge | Clean, harmonise district codes, join on district and year |
| 3. Features and zones | Seasonal weather windows, anomalies against each district's normal, district clustering |
| 4. ML forecasting | Predict yield anomalies per crop with Ridge, Random Forest, XGBoost and CatBoost |
| 5. Risk and finance | Expected production, revenue and profit; weather, yield and financial risk scores |
| 6. Optimisation | QUBO → Ising → QAOA for crop allocation, with Gurobi as the exact benchmark |
| 7. Decision support | Allocation, expected outcomes and risk reports |

Classical ML answers *what will happen*, the risk and finance layer answers *what it means*, the QUBO defines *what decision to optimise*, and QAOA searches for the allocation. Quantum computing is used only for optimisation, never for prediction.

## Data

| Source | Content | Used for |
|---|---|---|
| ICRISAT District Level Database (unapportioned) | Area, production, yield by crop, district and year | Targets and crop history |
| IMD 0.25° gridded rainfall | Daily rainfall | Rainfall features |
| ERA5-Land reanalysis | Temperature, humidity, wind, solar radiation | Climate features |
| Agmarknet, MSP, cost-of-cultivation data | Crop price and cost | Financial impact (not yet integrated) |

**Working dataset: `merged_district_year_v1b.csv`**

- 65 rows: 13 districts × 5 years (2015–2019), joined on `district_code` + `year`.
- 160 columns: identifiers, 23 crops × (area, production, yield), 6 area-only columns, a rainfall-coverage column, and 78 weather columns (6 seasonal windows × 13 features).
- Missing values are left blank, never filled. ICRISAT's `-1` and `-500` codes are set to blank.
- See the Scientific Data Guide for column definitions, units, caveats and known data issues.

## Merged Crop–Weather Datasets

The project contains multiple versions of the merged district-year crop–weather
dataset. Each version represents a different processing stage and should not be
treated as interchangeable.

| Dataset | Description | Status / Recommended use |
|---|---|---|
| `merged_district_year_v1b.csv` | Initial crop–weather merge using district and year mapping. Original rainfall missingness is preserved. | Baseline version for reproducibility and comparison. |
| `merged_district_year_v1b_filledv2.csv` | `v1b` updated using the filled Srikakulam rainfall and recomputed annual weather features. | Intermediate weather-imputation version. Use when specifically analysing the Srikakulam rainfall reconstruction. |
| `merged_district_year_v1c.csv` | Updated merged dataset with completed rainfall values and corrections identified during data-quality and EDA checks, including suspected crop-value scaling issues. | Main updated merged dataset for analysis and comparison. |
| `merged_cleaned_v1d.csv` | Cleaned and refined version incorporating fixes and missing-value corrections identified in earlier versions. | Latest and preferred merged dataset for new analysis, subject to team validation. |



### Dataset lineage

```text
merged_district_year_v1b.csv
        |
        |-- Srikakulam rainfall reconstruction
        |       └── merged_district_year_v1b_filledv2.csv
        |
        |-- Data-quality and EDA corrections
                └── merged_district_year_v1c.csv
                        |
                        └── Missing-value corrections and final cleaning
                                └── merged_cleaned_v1d.csv
```

## Canonical Issue Deliverables & Artifact Paths

Canonical locations for foundational feature engineering and data processing deliverables:

| Issue | Component | Canonical Path | Primary Outputs / Scripts | Description |
|---|---|---|---|---|
| **Issue 1** | Crop Long Format Dataset | `CropLongFormatv1/` | `crop_long_v1.csv`<br>`crop_restructure_validate.py` | Restructured unpivoted crop-long dataset (1,430 rows × 124 columns), change logs (`v1f_change_log.csv`), and validation reports. |
| **Issue 2** | Crop Model Eligibility | `CropModelEligibility/` | `crop_model_eligibility_v1.csv`<br>`crop_observation_coverage_summary_v1.csv` | Multi-crop suitability classification, observation coverage thresholds, and evaluation report. |
| **Issue 6** | Climate Zonal Clustering | `District Climate Zonal Clustering/` | `district_climate_profiles_v1.csv`<br>`district_zone_map_v1.csv`<br>`district_climate_clustering.py` | Long-term district climate profiling, districts grouped by climate-feature similarity (3 zones), and clustering metrics (`clustering_evaluation_v1.csv`). |

> **Note on Issue 1 canonical location and duplicate cleanup:**
> - Canonical files reside in `CropLongFormatv1/` (`crop_long_v1.csv`).
> - The legacy duplicate folder `Issues/CropLongFormatv1/` has been resolved on `dev`. Removal on `Main` is queued to take effect when `dev` is released/merged into `Main`.
> - Checksum verification between `CropLongFormatv1/` and legacy `Issues/CropLongFormatv1/` confirmed 100% byte-level identity across all 8 files (SHA-256):
>
> | File | Status | SHA-256 Checksum |
> |---|:---:|---|
> | `crop_long_v1.csv` | Byte-identical | `e0e57b0fc90f12498436f081978b6fd26836c8863b202eb2baf6c23c8522f1ad` |
> | `crop_restructure_validate.py` | Byte-identical | `2a245fd9b7b34aa5e6a418ca94d790a1d0a0e769ee5f1e61c287402ceac86bef` |
> | `merged_cleaned_v1f.csv` | Byte-identical | `d299847c01b87ea725623aa5192e34e6bd540064323c5fe7a8cecf1410279ab7` |
> | `v1f_change_log.csv` | Byte-identical | `d9e84dc194d86bd75e45712508a885d5fdcbcef0ba0f287088add727cb17e698` |
> | `ISSUE_1_CROP_LONG_FORMAT_GUIDE.md` | Byte-identical | `21b459d92f086c2ae442cd4e89db27c97bb3eee8acb3eb6e1227f910b16e26a7` |
> | `crop_data_validation_report.pdf` | Byte-identical | `95184ad56ca0e0fa44af39c4509a60c4765f4e8f2643c07c1bf1a23657c859ac` |
> | `QWeatherRiskF_Data_Fixes_Report.pdf` | Byte-identical | `34e11d88f420f799ecd4e859cc4312baa26b5b6fbf99f3224ad7aa00f05b5430` |
> | `QWeatherRiskF_Verification_and_Fix_Plan.pdf` | Byte-identical | `78196a3269c34f3ee63c6f904f30b82bbbc7e30711bb7c036e8654500fcff325` |

## Method summary

- **Target:** yield anomaly, meaning yield minus that district's average yield. District averages are computed from training years only, to avoid leakage.
- **Features:** two to four weather anomalies per crop, chosen from agronomic reasoning (for example Kharif rainfall, longest dry spell and north-east monsoon rainfall for groundnut).
- **Models:** Ridge, Random Forest, XGBoost, CatBoost, each compared with a baseline that predicts the district's usual yield.
- **Validation:** leave-one-year-out (five folds). Skill is `1 − model error / baseline error`, with a bootstrap 95% interval. Zero means no better than the baseline.

## Known limitations

- Only five years (2015–2019) for 13 districts, so about four or five independent years. Intervals are wide and the results are early evidence.
- 2016 and 2018 were poor years across the whole state, so a large part of the signal is shared by all districts.
- Srikakulam rainfall is blank (only 75% of its area has valid IMD cells).
- The ICRISAT year convention (crop year starting in that year) is tentative and must be confirmed from its documentation.
- Thresholds such as hot day (35 °C), rainy day (2.5 mm) and base temperature (10 °C) are placeholders.
- Some crops have data breaks (sugarcane 2018, cotton 2019) and the oilseeds aggregate has implausible yields; these are excluded until checked.
- ERA5-Land temperature has not yet been checked against IMD station temperature.
- QAOA is simulated and limited to roughly 20 to 25 qubits.

## Roadmap

1. Confirm the ICRISAT year convention and extend the weather record back to about 1990.
2. Add price, cost and water data.
3. Build the feature script (anomalies, seasonal windows, zones).
4. Tune and validate models per crop; add SHAP and yield ranges.
5. Build the risk engine and financial-impact module.
6. Build the QUBO and run QAOA and Gurobi on pilot-size instances.
7. Reports and dashboard.

## Reproducibility

Fixed random seeds, versioned data and code, and no gap-filling. Every reported result should be reproducible from the dataset version and script in this repository.

## Licence and ownership

© [Your name / organisation]. All rights reserved. Ownership and permitted use are governed by the agreement with the client. If the repository is published, add a licence (for example MIT or Apache-2.0) after confirming with the owner.

Do not commit client data, credentials or Gurobi licence files.

## Data acknowledgements

- ICRISAT District Level Database.
- India Meteorological Department (IMD) gridded rainfall.
- ERA5-Land, Copernicus Climate Change Service (C3S) / ECMWF.

## Setup and tests

### Requirements
- Python 3.12
- UV (recommended)
- Dependencies listed in the root requirements.txt

### Install and run tests

Run all four test suites using:

`uv run --python 3.12 --with-requirements requirements.txt python run_tests.py`

The test runner executes:

1. address_expansion/tests
2. financial_data_validation/tests
3. synthetic_portfolio/tests
4. synthetic_training_dataset/tests

The runner stops and returns a non-zero exit code if any suite fails. A successful run prints ALL TEST SUITES PASSED.
