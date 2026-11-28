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
├── train_forecast.py                            # Forward-chaining evaluation pipeline (scaled-target by default)
├── folds.py                                     # Year-agnostic fold generation and validation
├── experiment_scaled_shrinkage.py               # Exploratory: target scaling and shrinkage blend
├── experiment_lodo.py                           # Exploratory: forward-chaining leave-one-district-out
├── experiment_calibrated_intervals.py           # Exploratory: calibrated 80% intervals (out-of-fold, per-crop)
├── experiment_five_year_loyo.py                 # Exploratory: five-year nested leave-one-year-out (not forecasting)
├── data/
│   └── data_audit_report.md                     # Upstream input audit (written against the earlier PR-local table)
├── models/
│   └── model_config.json                        # Hyperparameters, random seeds, and taxonomy metadata
├── results/
│   ├── predictions_forward_chaining_v1.csv      # Out-of-fold predictions & intervals (1,280 rows)
│   ├── fold_evaluation_metrics_v1.csv           # Fold-by-fold MAE, RMSE, and R²
│   ├── aggregate_model_comparison_v1.csv        # Consolidated model comparison with p-values
│   ├── crop_level_evaluation_metrics_v1.csv     # Granular per-crop evaluation breakdown
│   ├── prediction_interval_evaluation_v1.csv    # Empirical test coverage vs 80% nominal target
│   └── experiments/                             # raw_target_v1/ run, shrinkage and LODO outputs
├── docs/
│   ├── issue_50_final_report.md                 # Complete technical findings and audit decisions
│   └── RETRAINING_ON_LONGER_HISTORY.md          # How to rebuild and retrain on more years
└── tests/
    ├── conftest.py                              # Synthetic data fixture for unit testing
    └── test_yield_forecast.py                   # Automated test suite (53 test functions)
```

---

## 3. Dataset Construction & Exclusion Accounting

The pipeline reads the **single shared dataset from Issue #7**, `ml_ready_dataset/ml_ready_dataset_v1.csv`, built by `ml_ready_dataset/build_ml_ready_dataset.py` (see `ml_ready_dataset/README.md` for the column dictionary, join-coverage report, and leakage checks). This module no longer builds its own table.

### Record Accounting & Usable Sample Counts (derived from the inputs, not hard-coded):
- **Base rows:** the eligible crop-long rows: crops with `proceed_to_modelling == True` (Issue 2) and `use_in_modelling == True` (Issue 1). Committed run: 1,430 crop-long rows, 11 eligible crops, **715 rows kept**, 715 excluded.
- **Primary Key:** `['district_code', 'year', 'crop']` (0 duplicate keys).
- **Eligible crops:** `groundnut`, `maize`, `rice`, `sugarcane`, `pigeonpea`, `sesamum`, `finger_millet`, `chickpea`, `pearl_millet`, `sunflower`, `rabi_sorghum`.
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

### Validation modes: the pipeline is year-agnostic

The folds are no longer hand-written. `yield_forecast/folds.py` builds them from the years present in the dataset, and the pipeline stops with a clear message if a fold would leak, has no test rows, or has too few training rows. Set `validation.mode` in `models/model_config.json`:

| Mode | What it does |
|---|---|
| `expanding` | **the committed design.** Every usable year after the first is a test year, trained on all earlier years. For our data: train 2015-2017 to test 2018, then train 2015-2018 to test 2019 |
| `fixed_origin` | train on all years up to `train_end_year`, test each later year that has targets (for us `train_end_year` 2017 would train both folds on the 2017 rows alone, which weakens Fold 2) |
| `explicit` | hand-listed folds (a config with only a `folds` list still works) |

Forward-chaining targets exist only from the third data year, so **2015 and 2016 contribute no training rows**: Fold 1 trains on the 128 rows from 2017, and Fold 2 on 256 rows (2017 and 2018). The run prints, for every fold, which years actually supply training rows.

It was tested on a synthetic 12-year dataset with a planted weather signal (`tests/test_multi_year.py`). To retrain on a longer real history, follow `docs/RETRAINING_ON_LONGER_HISTORY.md`. That test shows the code works on more years, not what a real longer-history result will be.

---

## 5. Model Comparison & Statistical Evaluation

All five models are evaluated on the exact same 256 held-out out-of-fold instances:

Models are trained on the yield anomaly divided by each crop's training-fold standard deviation and their predictions are converted back to kg/ha (`target_scaling` in `models/model_config.json`; `--no-scale-target` restores the raw-target run, saved in `results/experiments/raw_target_v1/`).

| Model | Samples | Aggregate MAE (kg/ha) | Aggregate RMSE (kg/ha) | Scaled MAE (x crop train SD) | $\Delta$ MAE vs Base | District-clustered Wilcoxon $p$ | Better than climatology? | 80% PI coverage (legacy in-sample rule) | Verdict (derived by code) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Climatology Baseline** | 256 | 1,341.91 | 3,787.98 | 1.033 | 0.00 | 1.0000 | Reference | 71.9% | Reference |
| **XGBoost** | 256 | 1,122.88 | 3,304.92 | 0.938 | -219.03 (-16.32%) | 0.0007 | YES | 74.2% | Significantly better |
| **RandomForest** | 256 | 1,174.28 | 3,527.83 | 0.950 | -167.64 (-12.49%) | 0.0017 | YES | 74.2% | Significantly better |
| **CatBoost** | 256 | 1,180.39 | 3,494.07 | 0.956 | -161.53 (-12.04%) | 0.0046 | YES | 72.7% | Significantly better |
| **Ridge Regression** | 256 | 1,360.56 | 4,032.09 | 0.986 | +18.65 (+1.39%) | 0.1099 | NO | 73.8% | Not distinguishable |

Results are reproducible only with the exact package versions pinned in `yield_forecast/requirements.txt` (Python 3.12).

### Honest Statistical Findings:
1. **Target scaling drives the result.** With the raw kg/ha target (saved in `results/experiments/raw_target_v1/`) every model was significantly *worse* than climatology (scaled MAE 1.31-4.59 vs 1.03). Sugarcane's anomaly SD (~10,300 kg/ha) is about ten times that of rabi sorghum (~970), so a pooled fit on the raw target is mostly learning sugarcane. Scaling each crop's target by its training-fold SD reverses the ranking.
2. **Size of the gain is modest.** The tree models cut crop-scaled error by about 7-9% versus climatology and beat it on 56-59% of individual rows. Ridge is not distinguishable from climatology.
3. **Selected after seeing the test years.** The scaling idea was adopted after inspecting the 2018-2019 results, and several variants were compared. The p-values are therefore optimistic. A forward-chaining leave-one-district-out check (`experiment_lodo.py`, 12-13 districts better for CatBoost/XGBoost, +7-8% skill) supports it but reuses the same two test years. A five-year check (below) did **not** confirm the size of the gain: it shrinks to about 2-3% and is not significant.
4. **Significance test.** Per-district mean differences in scaled absolute error are tested with a two-sided Wilcoxon signed-rank test across the 13 districts (rows within a district, year and crop are clustered). With 13 districts the smallest attainable $p$ is 0.000244. Only two test years exist, so $p$-values are indicative.
5. **Fold reversal.** XGBoost beats climatology in both folds on pooled MAE; RandomForest, CatBoost and Ridge beat it in Fold 1 (2018) but not Fold 2 (2019).
6. **Not a usable predictor yet.** $R^2$ is negative for every model (best: XGBoost -0.312). Direction accuracy (above/below normal) is about 42-49%, i.e. chance. The models reduce the size of errors but do not predict the sign of the anomaly.

### Per-crop results (crop-scaled error, skill vs climatology)

Skill is `100 x (1 - model scaled MAE / climatology scaled MAE)` per crop; positive means better than the crop-mean baseline. A crop is flagged **low-sample** when its smallest fold has fewer than 10 test rows (project threshold in `model_config.json`, above Issue 5's `MIN_N` of 8); read flagged crops with caution. Per-crop numbers rest on 16 to 26 test rows, so they are descriptive, not significance-tested.

| Crop | Test rows | Districts | Min fold rows | Low-sample flag | CatBoost | RandomForest | Ridge | XGBoost | XGBoost beats climatology on |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| chickpea | 22 | 11 | 11 | no | +6.6% | +7.5% | +6.3% | **+7.9%** | 55% of rows |
| finger_millet | 20 | 10 | 10 | no | -1.9% | -2.9% | -1.8% | **-2.4%** | 45% of rows |
| groundnut | 26 | 13 | 13 | no | +10.3% | +11.3% | +5.1% | **+11.6%** | 62% of rows |
| maize | 26 | 13 | 13 | no | +24.6% | +23.8% | +8.6% | **+27.3%** | 77% of rows |
| pearl_millet | 22 | 11 | 11 | no | +6.8% | +7.5% | +12.9% | **+8.3%** | 45% of rows |
| pigeonpea | 26 | 13 | 13 | no | +2.3% | +4.5% | +4.1% | **+4.0%** | 50% of rows |
| rabi_sorghum | 16 | 8 | 8 | **yes** | -0.8% | +0.1% | -2.3% | **+5.9%** | 75% of rows |
| rice | 26 | 13 | 13 | no | +0.3% | +1.5% | +6.4% | **+1.7%** | 54% of rows |
| sesamum | 24 | 12 | 12 | no | -1.1% | -2.5% | -6.1% | **-3.2%** | 42% of rows |
| sugarcane | 26 | 13 | 13 | no | +12.0% | +12.6% | -5.3% | **+17.2%** | 65% of rows |
| sunflower | 22 | 11 | 11 | no | +18.6% | +19.7% | +13.9% | **+20.9%** | 86% of rows |

XGBoost improves on climatology in 9 of 11 crops. The gain is concentrated (maize, sunflower, sugarcane) and two crops (finger_millet, sesamum) are slightly worse than climatology, so the pooled average hides real heterogeneity.

**Why selection is pooled, not per crop.** Issue #7's forward-chaining version of Issue 5's per-crop selection (`fc_*`) yields at least one predictor for only 48 of the 256 test rows (19%), and none for 16 rows (`rabi_sorghum`, which Issue 5's row filter drops). A per-crop-predictor model could therefore act on under a fifth of the evaluation, so the pipeline selects features pooled across crops inside each forward-chaining fold.

### Five-year check (exploratory; retrospective response estimation, **not forecasting**)

Only 2015-2019 are available, and forward chaining scores just two of those years. `experiment_five_year_loyo.py` re-estimates the weather-yield response on **all five years** with nested leave-one-year-out validation: for each held-out year, every anomaly (target and weather) is recomputed from raw values with a baseline that excludes both the row's own year and the held-out year, so training anomalies never contain the held-out year. Because the baselines use later years, this track **cannot support a forecasting claim**; it estimates how strongly weather relates to yield.

Recomputation check: the held-out-year anomalies match the source leave-one-year-out files (`yield_anomalies_v1.csv`, `weather_anomalies_v1.csv`) to about 1e-13 on 642 yield and 4901 weather values compared, with none missing.

| Model | Raw-target skill | Scaled-target skill | Scaled: years better than climatology | Scaled: district Wilcoxon p | Verdict (scaled) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| XGBoost | -9.0% | **+2.6%** | 3 of 5 | 0.127 | Not distinguishable from Climatology; scaled MAE lower |
| CatBoost | -5.6% | **+2.8%** | 3 of 5 | 0.146 | Not distinguishable from Climatology; scaled MAE lower |
| RandomForest | -14.1% | **+2.6%** | 3 of 5 | 0.146 | Not distinguishable from Climatology; scaled MAE lower |
| Ridge | -118.6% | **+2.4%** | 4 of 5 | 0.168 | Not distinguishable from Climatology; scaled MAE lower |

XGBoost's scaled-target skill by held-out year: 2015: -0.3%, 2016: -0.2%, 2017: +5.4%, 2018: +4.0%, 2019: +3.1%.

**What this changes.** (1) Target scaling is still essential: with the raw target no model beats climatology. (2) But the gain over climatology across five years is **about +2 to +3%, not the +9% seen on 2018-2019**, it is **not statistically distinguishable from zero** (p 0.13 to 0.17), and the tree models are better in only 3 of 5 held-out years (2017 to 2019) and marginally worse in 2015 and 2016 (Ridge: 4 of 5). The headline forward-chaining gain should therefore be read as largely specific to those two test years and optimistic. The honest summary is: weather adds a small, unconfirmed improvement over a crop-mean baseline.

### Calibrated prediction intervals (exploratory; `experiment_calibrated_intervals.py`)

The pipeline's 80% intervals use in-sample training residuals and one width in kg/ha for every crop, and they cover only about 72-74%. The experiment compares four ways of building them on the same folds, features and models (nothing re-tuned). The best, `oof_scaled`, is a split-conformal interval built from **out-of-fold residuals** (leave-one-district-out inside the training rows only, feature selection repeated inside each inner fold), **normalised by each crop's training standard deviation** so every crop gets its own width. The test year is never used to calibrate (a test confirms the bounds do not change when the test targets change).

| Model | Coverage now -> calibrated | Worst crop's coverage now -> calibrated | Crops within 10 pts of 80% | Interval score (lower is better) | Coverage 2018 / 2019 (calibrated) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Climatology | 71.9% -> **74.2%** | 4% -> **50%** | 2 of 11 -> **8 of 11** | 7.41 -> **4.80** | 66% / 83% |
| XGBoost | 74.2% -> **76.2%** | 8% -> **59%** | 1 of 11 -> **8 of 11** | 6.46 -> **4.55** | 66% / 86% |
| RandomForest | 74.2% -> **74.2%** | 15% -> **59%** | 2 of 11 -> **8 of 11** | 6.62 -> **4.62** | 65% / 84% |
| CatBoost | 72.7% -> **75.4%** | 12% -> **59%** | 2 of 11 -> **8 of 11** | 6.68 -> **4.64** | 66% / 85% |
| Ridge | 73.8% -> **73.0%** | 8% -> **59%** | 3 of 11 -> **6 of 11** | 7.35 -> **4.83** | 64% / 82% |

**What it shows**
- **Per-crop calibration is the real fix.** The current rule looks acceptable overall but is badly wrong by crop: its worst crop is covered only 4-15% of the time (sugarcane's intervals are far too narrow) while small crops are covered almost always (far too wide). The calibrated version brings the worst crop to 50-59% and puts 6-8 of 11 crops within 10 points of 80%.
- **Sharper at the same coverage.** Intervals are about half as wide in crop-std units (2.7-2.9 against 4.9-5.6), and the interval score, which penalises both width and misses, improves by 30-35% for every model.
- **Overall coverage is still short of 80%** (73-76%) and this is a limit, not a bug. The calibration data come from one training year (2017), so they cannot see a **year-level shift**: errors on the 2018 test year were about 1.43-1.51x the calibration error (coverage 64-66%), while 2019 was about 0.92-0.99x (coverage 82-86%, at or above nominal). More training years allow leave-one-year-out calibration inside the training window, which would capture that shift.
- Two test years and 13 districts, so this is indicative. It is now the pipeline's default interval (the `pi_*_80` columns, labelled `oof_conformal_per_crop_exploratory`); the old in-sample interval is kept in the `*_in_sample` columns.

---

## 6. Prediction Intervals & Honest Calibration Status

- **Default method (`oof_conformal_per_crop`):** prediction $\pm\, q \times \sigma_{\text{crop}}$, where $\sigma_{\text{crop}}$ is the crop's training-fold standard deviation and $q$ is the split-conformal quantile of **out-of-fold** residuals divided by $\sigma_{\text{crop}}$. The out-of-fold residuals come from leave-one-district-out cross-validation **inside the training rows only**, with feature selection repeated inside each inner fold. The test year is never used to calibrate (tested).
- **Status label:** **`oof_conformal_per_crop_exploratory`**. "Calibrated" describes the method, not the outcome: coverage is **measured on the test years and reported, and it falls short of 80%**.
- **Measured coverage vs the 80% target (pooled, 256 test rows):** Climatology 74.2%, XGBoost 76.2%, RandomForest 74.2%, CatBoost 75.4%, Ridge 73.0%. By year: 2018 64-66% and 2019 82-86%.
- **Per crop:** the worst crop's coverage is 50-59% (the legacy in-sample interval: 4-15%). Per-crop coverage is in `crop_level_evaluation_metrics_v1.csv` (`pi_coverage_80`).
- **Legacy interval kept for comparison:** the old in-sample rule (prediction $\pm$ the 80th percentile of absolute in-sample training residuals, one width in kg/ha for every crop) is in the `*_in_sample` columns. Its pooled coverage was 72-74%, and it is far too narrow for sugarcane and far too wide for small crops. Interval score (lower is better, crop-std units): Climatology 7.41 -> 4.80, XGBoost 6.46 -> 4.55, RandomForest 6.62 -> 4.62, CatBoost 6.68 -> 4.64, Ridge 7.35 -> 4.83 (legacy -> default, 30-35% better).
- **Why coverage is still short:** the calibration data come from a single training year (2017), so they cannot reveal a **year-level shift**. The 2018 test errors were about 1.5x the calibration error, while 2019 matched it.
- **Switching back:** set `prediction_intervals.method` to `in_sample_residual_quantile` in `models/model_config.json`.
- **Recommendation:** with a longer history, calibrate with leave-one-year-out inside the training window so the year-level shift is visible (see `docs/RETRAINING_ON_LONGER_HISTORY.md`).

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

### Step 2: Build the Issue #7 ML-Ready Dataset
```bash
python ml_ready_dataset/build_ml_ready_dataset.py
```
*Expected Output: `ml_ready_dataset/ml_ready_dataset_v1.csv` (715 rows × 313 columns), plus coverage, validation and reproducibility reports in `ml_ready_dataset/`. All 36 checks must pass.*

### Step 3: Run Training & Forward-Chaining Pipeline
```powershell
python yield_forecast/train_forecast.py                      # scaled-target (default)
python yield_forecast/train_forecast.py --no-scale-target \
  --results-dir yield_forecast/results/experiments/raw_target_v1   # raw-target comparison
```
*Expected Output: 5 result CSVs per run. Requires Python 3.12 and `pip install -r yield_forecast/requirements.txt`.*

### Step 4: Run Automated Test Suite
```powershell
python -m pytest yield_forecast/tests -v
```
*Expected Output: 53 passed.*

---

## 9. Issue #51 — Provisional Production Forecast with Area-and-Yield Uncertainty (Integration of Issue #49)

### Objective
This pipeline produces a provisional production forecast (`production_forecast_v1.csv`) by integrating:
- **Issue #49 area forecasts** (286 rows from `area_forecast/area_forecast_v1.csv`: 13 districts × 22 crop combinations, `pred_Climatology`, `pred_Ridge_Delta`, `pred_RandomForest_L1_LogDelta`, `pred_XGBoost_L1_LogDelta`) with
- **Issue #50 yield predictions** (715-row eligible-for-baseline dataset from `ml_ready_dataset/ml_ready_dataset_v1.csv`) to compute `predicted_production_1000_tons = predicted_area_1000_ha × predicted_yield_kg_per_ha / 1000`.

Two uncertainty scopes are propagated:

| Scope | Records | Area forecast | CatBoost model |
|---|---|---|---|
| `area_and_yield` | 1,024 | Issue #49 area forecast (interval `area_pi_low_80` / `area_pi_high_80`) + yield interval `pi_lower_80` / `pi_upper_80` | N/A (excluded) |
| `yield_only` | 256 | **Missing** — `missing_area_forecast=True`, `predicted_area_1000_ha = NaN` | CatBoost only |

### Input Files & Exact Reproduction Commands

| Artifact | Source | Description |
|---|---|---|
| `area_forecast/area_forecast_v1.csv` | 286 rows | Issue #49 forward-chaining area forecasts (Climatology, Ridge, RandomForest, XGBoost deltas) |
| `ml_ready_dataset/ml_ready_dataset_v1.csv` | 715 rows | Eligible-for-baseline dataset (Issue #7) joined on `['district_code', 'crop', 'year']` |
| `yield_forecast/models/model_config.json` | — | Operational taxonomy: `end_of_season_retrospective_response_estimation_diagnostic_hindcast` |
| `yield_forecast/issue_51_production_forecast.py` | — | Main pipeline script |
| `yield_forecast/tests/test_issue_51_production_forecast.py` | — | 19 test functions (11 original fixes + 8 integration tests) |

**Reproduction command:**
```powershell
python yield_forecast/issue_51_production_forecast.py
```
*Expected Output:*
- `[SAVED] Production forecast: yield_forecast/results/production_forecast_v1.csv (1280 rows)`
- Empirical production interval coverage reported (30.6% combined area_and_yield; 27.2% overall including yield_only)
- All 19 Issue #51 tests pass: `python -m pytest yield_forecast/tests/test_issue_51_production_forecast.py -v`

### Production Formula & Units

```
predicted_yield_kg_per_ha = baseline_mean_kg_per_ha + predicted_yield_anomaly_kg_per_ha
predicted_production_1000_tons = predicted_area_1000_ha × predicted_yield_kg_per_ha / 1000
```

- Area: 1,000 hectares (1000 ha)
- Yield: kilograms per hectare (kg/ha)
- Production: 1,000 tons (1000 tons); the `/1000` converts kg to tons given area is in 1000 ha

### Output Files & Actual Roles

| File | Rows | Columns | uncertainty_scope | Role |
|---|---|---|---|---|
| `production_forecast_v1.csv` | **1,280** | 25 | area_and_yield: 1,024; yield_only: 256 | **Current pipeline output**. Contains `prod_pi_low_80`, `prod_pi_high_80`, `prod_pi_covered_80`. Production formula applies only to area_and_yield rows (1,024). Yield_only rows have NaN predicted_area_1000_ha. |
| `production_forecast_combined_v1.csv` | 1,280 | 25 | yield_only: 1,280 | **Preserved fallback** (pre-Issue #49). All rows have `uncertainty_scope=yield_only`; no area forecasts are joined. This file exists as a reference/rollback; the current pipeline produces `production_forecast_v1.csv`. |

Column key (production_forecast_v1.csv):
`fold`, `train_years`, `year`, `district_code`, `district_name`, `crop`, `model`, `actual_yield_anomaly`, `predicted_yield_anomaly`, `baseline_mean_kg_per_ha`, `predicted_yield_kg_per_ha`, `area_1000_ha`, `predicted_area_1000_ha`, `area_pi_low_80`, `area_pi_high_80`, `predicted_production_1000_tons`, `prod_pi_low_80`, `prod_pi_high_80`, `prod_pi_covered_80`, `uncertainty_scope`, `interval_methodology`, `missing_area_forecast`, `pi_lower_80`, `pi_upper_80`, `residual`

### Two Uncertainty Scopes

- **`area_and_yield` (1,024 records):** Both area and yield intervals are propagated via Monte Carlo (1,000 uniform samples per record, rho=0 independence, seed 42). Production = area × yield / 1000. Area interval columns: `area_pi_low_80`, `area_pi_high_80`. Yield interval columns: `pi_lower_80`, `pi_upper_80`.
- **`yield_only` (256 CatBoost records):** Area forecast from Issue #49 is **not available**; `missing_area_forecast=True` and `predicted_area_1000_ha = NaN`. Only yield uncertainty is sampled. Production uses historical (baseline) area_1000_ha as fixed, yielding `mc_production_samples` with yield-only variation.

### Monte Carlo Assumptions

- **1,000 uniform samples** per record
- **Seed 42** for reproducibility
- **rho=0 independence:** area and yield sampled independently (no correlation)
- **Interval sampling:** area uniformly from `[area_pi_low_80, area_pi_high_80]`; yield uniformly from `[pi_lower_80, pi_upper_80]` (area_and_yield only)
- **Yield-only:** historical `area_1000_ha` fixed; yield sampled from `[pi_lower_80, pi_upper_80]`
- Output: `mc_production_samples` column (lists of 1,000 floats, stored in CSV as Python `list` objects); derived percentiles produce `prod_pi_low_80` (10th percentile) and `prod_pi_high_80` (90th percentile)

### Coverage

**Combined area-and-yield (1,024 records):**
- Numerator: 313 covered observations
- Rate: **30.6%**

**Year breakdown (area_and_yield only):**
- Year 2018: 210/512 = **41.0%**
- Year 2019: 103/512 = **20.1%**

**Overall 27.2% figure:** This is the coverage across **all 1,280 rows** (area_and_yield + yield_only combined), using the `prod_pi_covered_80` column which is set only for eligible combined-forecast rows. It is **not** the combined-only metric (30.6%); the overall figure includes yield_only rows where coverage is computed differently.

**Important:** Intervals are **exploratory and uncalibrated**. Do not claim 80% calibrated coverage or quantum advantage. The legacy Issue #50 calibrated intervals achieved 72-76% coverage on 256 test rows; Issue #51 intervals are wider (reflecting area uncertainty) and deliberately uncalibrated for prospective use.

### Exploratory & Uncalibrated Statement

- Prediction intervals are **exploratory**, not calibrated to 80%
- Coverage is **reported and computed** on held-out test years, but intervals are uncalibrated
- Do not claim calibrated coverage or quantum advantage over climatology
- The pipeline is a **provisional production forecast** for Issue #51, pending Issue #49 area forecast integration maturity

### Reproduction & Test Commands (Verified)

```powershell
# Run the pipeline
python yield_forecast/issue_51_production_forecast.py

# Run all 19 Issue #51 tests
python -m pytest yield_forecast/tests/test_issue_51_production_forecast.py -v
# Expected: 19 passed in ~1.3s
```

### Limitations, Missing-Data Handling & Scope

- **5-year dataset limitation:** 2015–2019 only; no generalizable predictive superiority over climatology established
- **CatBoost missing area forecast:** 256 yield-only rows have `missing_area_forecast=True`; `predicted_area_1000_ha = NaN`; production uses historical (baseline) area as fixed
- **Area forecast integration:** Issue #49 area forecasts (286 rows) joined via `['district_code', 'crop', 'year']`; 0 rows dropped in the join
- **Retrospective hindock scope:** Operational taxonomy declares `end_of_season_retrospective_response_estimation_diagnostic_hindcast`; not an advance forecast
- **NaN propagation:** Missing values from left-joins propagate as NaN; never silently imputed to zero
- **Monte Carlo 1000 simulations:** Uniform sampling, rho=0, seed 42; exploratory only
- **No quantum advantage claimed:** Intervals are wider than calibrated Issue #50 intervals; coverage reported but not calibrated to 80%
