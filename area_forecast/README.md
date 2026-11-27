# Area Forecast Pipeline (Issue #49)

**Module:** `area_forecast/`  
**Target Variable:** `area_1000_ha` (Sown area in 1,000 hectares)  
**Inputs:**
- `ml_ready_dataset/ml_ready_dataset_v1.csv` (Issue #7: 715 rows across 13 AP districts, 5 years, 11 eligible crops)
- `crop_calendar/crop_calendar_v1.csv` (Issue #43: Crop season & weather-window alignments)

> [!IMPORTANT]
> **Honest Baseline Comparison Notice:**
> Across all 286 out-of-time test observations (2018 and 2019), **none of the candidate ML models beat the naive `LastValue` (Persistence) baseline on overall MAE** (`3.7708` vs `3.8277` for `XGBoost_L1_LogDelta`, Wilcoxon $p = 0.5879$; `3.8496` for `HistGBR_L1_LogDelta`, $p = 0.5879$). `XGBoost_L1_LogDelta` improves overall **RMSE** (`10.4949` vs `10.5727`) and $R^2$ (`0.9796` vs `0.9793`) over `LastValue`, and edges out `LastValue` in Fold 2 (2019 MAE `3.3485` vs `3.3628`). However, over only 13 districts and 2 held-out test years, ML improvements over persistence are modest and not statistically conclusive.

---

## 1. Deliverables

| File | Grain / Rows | Description |
|---|---|---|
| `build_area_forecast.py` | — | Reproducible pre-sowing forward-chaining pipeline evaluating 3 baselines and 4 ML models. |
| `model_config.json` | — | Hyperparameter grids, forward-chaining folds, feature-selection settings, and uncertainty config. |
| `area_forecast_v1.csv` | 286 rows | Out-of-time predictions (`13 districts × 11 crops × 2 test years`) across all baselines, ML models, and 80% exploratory intervals. |
| `area_forecast_metrics_v1.csv` | 7 rows | Aggregate MAE, RMSE, MedAE, WAPE (%), sMAPE (%), $R^2$, scaled MAE, skill vs `LastValue`, district-clustered Wilcoxon $p$-values, and PI coverage. |
| `area_forecast_fold_metrics_v1.csv` | 14 rows | Per-fold metrics, tuned best hyperparameters, and dynamically selected `wact__rabi_prev__*` features. |
| `area_forecast_crop_metrics_v1.csv` | 11 rows | Per-crop MAE across all baselines and models, identifying the best model per crop. |
| `sample_size_flags_v1.csv` | 11 rows | Per-crop observation counts, zero-area counts, mean/std, and sample-size suitability flags. |
| `tests/test_area_forecast.py` | 8 tests | Automated tests verifying key uniqueness, zero temporal leakage, zero silent imputation, causal weather horizon, non-negativity, and temp-dir reproducibility. |

---

## 2. Methodology & Safeguards

1. **Three Mandatory Naive Baselines (strictly `year < test_year`):**
   - **LastValue (Persistence):** $\hat{A}_{d,c,T} = A_{d,c,T-1}$.
   - **Climatology:** Expanding historical mean of `area_1000_ha` per `(district_code, crop)` over `year < test_year`.
   - **LinearTrend:** Ordinary least squares trend per `(district_code, crop)` over `year < test_year`, clipped at $\ge 0$.
2. **Causal Feature Engineering & Zero Silent Imputation:**
   - **Strictly Causal Lags ($\le T-1$):** `lag_area_1`, `log_lag_area_1`, `lag_area_2`, `lag_area_diff`, `lag_hist_mean`, `lag_mean_reversion`, `lag_hist_max`, `lag_is_zero`, `lag_yield_1`, and `lag_prod_1`.
   - **Zero Imputation:** `XGBoost_L1_LogDelta` and `HistGBR_L1_LogDelta` handle missing early-history lags (`2016`) and zero-area yields natively without median/zero imputation; `RandomForest_L1_LogDelta` and `Ridge_Delta` restrict numeric inputs to the 100%-observed subset (`PURE_LAG_COLS` + selected pre-sowing weather).
   - **Pre-Sowing Weather Horizon:** Candidate weather predictors are restricted exclusively to prior-season **`wact__rabi_prev__*`** (100% observed in 2016–2019, preceding Kharif sowing), screened inside `df_train` only via `select_features_in_fold` (Spearman $|\rho|$ + collinearity $|r| \le 0.80$, max 4 features).
3. **Target Transformation, Loss Alignment & Inner Hyperparameter Tuning:**
   - **Log-Delta Target (`log1p(A_T) - log1p(A_{T-1})`):** Evaluated with **L1 / MAE loss** (`XGBoost`, `HistGBR`, `RandomForest`) and tuned via `GridSearchCV` using `GroupKFold` (grouped by `year`) strictly inside `df_train`.
   - **Additive Delta Target (`area_1000_ha - lag_area_1`):** Evaluated with `Ridge(alpha=10.0)` using Issue #50's per-crop training-fold `std` scaling (`fit_crop_scales`).
4. **Forward-Chaining Folds:**
   - **Fold 1:** Train `[2016, 2017]` ($N=286$) $\rightarrow$ Test `2018` ($N=143$)
   - **Fold 2:** Train `[2016, 2017, 2018]` ($N=429$) $\rightarrow$ Test `2019` ($N=143$)

---

## 3. Observed Results (`area_forecast_metrics_v1.csv`)

| Model / Baseline | MAE (1,000 ha) | RMSE (1,000 ha) | MedAE | WAPE (%) | sMAPE (%) | $R^2$ | Skill vs LastValue (%) | Wilcoxon $p$ | 80% PI Cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LastValue (Baseline)** | **3.7708** | 10.5727 | **0.5650** | **12.48%** | **37.60%** | 0.9793 | 0.00% | 1.0000 | — |
| **XGBoost_L1_LogDelta** | 3.8277 | **10.4949** | 0.5868 | 12.67% | 43.90% | **0.9796** | -1.51% | 0.5879 | 83.22% |
| **HistGBR_L1_LogDelta** | 3.8496 | 10.7193 | 0.5692 | 12.75% | 38.86% | 0.9787 | -2.09% | 0.5879 | 84.27% |
| **RandomForest_L1_LogDelta** | 3.8986 | 10.8495 | 0.6029 | 12.91% | 48.43% | 0.9782 | -3.39% | 0.8394 | 81.47% |
| **LinearTrend (Baseline)** | 4.2343 | 10.6156 | 0.5767 | 14.02% | 48.51% | 0.9791 | -12.29% | 0.0398 | — |
| **Climatology (Baseline)** | 4.5468 | 11.8898 | 0.9029 | 15.05% | 46.60% | 0.9738 | -20.58% | 0.1677 | — |
| **Ridge_Delta** | 6.5984 | 17.1522 | 1.3933 | 21.85% | 75.07% | 0.9455 | -74.99% | 0.0002 | 76.22% |

---

## 4. Run & Test Commands

```bash
# Generate all Issue #49 deliverables from the repository root:
python area_forecast/build_area_forecast.py

# Run automated validation and zero-leakage tests:
python -m unittest area_forecast/tests/test_area_forecast.py -v
```
