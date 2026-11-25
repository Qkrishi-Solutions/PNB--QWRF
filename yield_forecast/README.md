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
├── experiment_scaled_shrinkage.py               # Exploratory: target scaling and shrinkage blend
├── experiment_lodo.py                           # Exploratory: forward-chaining leave-one-district-out
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
│   └── issue_50_final_report.md                 # Complete technical findings and audit decisions
└── tests/
    ├── conftest.py                              # Synthetic data fixture for unit testing
    └── test_yield_forecast.py                   # Automated test suite (23 test functions)
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

---

## 5. Model Comparison & Statistical Evaluation

All five models are evaluated on the exact same 256 held-out out-of-fold instances:

Models are trained on the yield anomaly divided by each crop's training-fold standard deviation and their predictions are converted back to kg/ha (`target_scaling` in `models/model_config.json`; `--no-scale-target` restores the raw-target run, saved in `results/experiments/raw_target_v1/`).

| Model | Samples | Aggregate MAE (kg/ha) | Aggregate RMSE (kg/ha) | Scaled MAE (x crop train SD) | $\Delta$ MAE vs Base | District-clustered Wilcoxon $p$ | Better than climatology? | Empirical 80% PI Coverage | Verdict (derived by code) |
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
3. **Selected after seeing the test years.** The scaling idea was adopted after inspecting the 2018-2019 results, and several variants were compared. The p-values are therefore optimistic. A forward-chaining leave-one-district-out check (`experiment_lodo.py`, 12-13 districts better for CatBoost/XGBoost, +7-8% skill) supports it but reuses the same two test years. It needs confirmation on new years (Issue #44).
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

---

## 6. Prediction Intervals & Honest Calibration Status

- **Method:** In-sample residual quantiles: $q_{80} = \text{Quantile}_{0.80}(|y_{\text{train}} - \hat{y}_{\text{train}}|)$.
- **Calibration Status:** Honestly designated as **`uncalibrated_in_sample_exploratory`**.
- **Reason for In-Sample Calculation:** In Fold 1, only a single year (2017, $N=128$) is available for training across 11 crops. Carving out a held-out calibration split within a single year would severely degrade model training and produce highly noisy calibration thresholds.
- **Empirical Coverage vs 80% target:** 71.9% (Climatology), 72.7% (CatBoost), 73.8% (Ridge), 74.2% (RandomForest and XGBoost). All remain below nominal; the intervals are in-sample and uncalibrated.
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

### Step 2: Build the Issue #7 ML-Ready Dataset
```bash
python ml_ready_dataset/build_ml_ready_dataset.py
```
*Expected Output: `ml_ready_dataset/ml_ready_dataset_v1.csv` (715 rows × 312 columns), plus coverage, validation and reproducibility reports in `ml_ready_dataset/`. All 29 checks must pass.*

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
*Expected Output: 23 passed.*
