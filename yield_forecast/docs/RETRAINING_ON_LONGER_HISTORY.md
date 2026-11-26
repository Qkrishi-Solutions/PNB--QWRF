# Retraining the Yield Pipeline on a Longer History

**Audience:** anyone who has more years of crop and weather data than the 2015 to 2019 we developed on, for example the bank's team or the data owner.
**What this is:** a step-by-step guide to rebuilding the dataset and retraining on a longer history, what carries over, what must be redone, and what to decide again.
**What this is not:** a promise that results will improve. Our results come from three years of targets and are exploratory. A longer history is the test we could not run.

---

## 1. What the pipeline now does by itself

The Issue 50 pipeline (`yield_forecast/train_forecast.py`) no longer contains hand-written years. It reads the years that are in the dataset and **builds its validation folds from them** (`yield_forecast/folds.py`). It checks that every training year is earlier than its test year, that each test year has targets, and that each fold has enough training rows. If a rule is broken it stops with a clear message.

We tested this on a **synthetic 12-year dataset (2008 to 2019) with a planted weather signal**: the pipeline built the right folds, trained only on earlier years, ran every model, and recovered the signal. We also confirmed that those tests fail when the signal is removed or when the folds are made to leak. **That test used synthetic data. It shows the code works on a longer history, not what the real result will be.**

On our own data the refactor changed no result: every output file is byte-identical to before.

## 2. What carries over, and what does not

| Carries over | Does **not** carry over |
|---|---|
| The dataset builder and its checks | The fitted models |
| The training pipeline, fold generator and metrics | The three features we selected |
| The validation design: train on earlier years, test on later ones | Every performance number (+9%, +5.6%, +2.6%) |
| The test suite and the leakage tests | The small model settings and the 3-feature limit (chosen for about 128 rows) |
| The lessons: scale the target per crop, select features inside each fold, compare with a baseline, never impute | Our conclusion that the gain is "unconfirmed" (it describes a small sample) |

**Retrain from scratch.** Do not reuse our fitted models, selected features or settings as if they were validated.

## 3. Choosing the validation mode

Set `validation` in `yield_forecast/models/model_config.json`.

| Mode | Use it when | What it does |
|---|---|---|
| `expanding` | **recommended for a long history** | every year with targets that has at least `min_train_years_with_targets` earlier years with targets is a test year; each fold trains on all earlier years |
| `fixed_origin` | you want one frozen training window (the design committed for our data) | train on all years up to `train_end_year`; test on each later year that has targets |
| `explicit` | you need hand-picked folds | list them under `validation.folds` (an old config with only a `folds` list still works) |

Examples:

```json
"validation": {"method": "forward_chaining", "mode": "expanding",
               "min_train_years_with_targets": 3, "min_training_rows": 8}
```
```json
"validation": {"method": "forward_chaining", "mode": "fixed_origin",
               "train_end_year": 2017, "min_training_rows": 8}
```

Optional for `expanding`: `first_test_year` and `last_test_year` to limit which years are scored.

**Why `min_train_years_with_targets` matters.** A forward-chaining target needs at least two earlier years, so the first two data years never have targets, and the third has no earlier year with a target to train on. A fold trained on one year (as ours were) is very weak; with a longer history, require several years (for example 3 to 5) before scoring a test year.

**How the pipeline reports it.** At the start of a run it prints, for every fold, the test year, the number of test rows, and the training rows with the years they actually come from. Always read this block. A configured training year with no targets contributes no rows (this is why both of our folds train on 2017 only).

## 4. Step by step

Run from the repository root. Package versions are pinned in `yield_forecast/requirements.txt` and `ml_ready_dataset/requirements.txt` (Python 3.12). Commands marked **(not tested on a longer history)** are year-agnostic in my reading of the code but have not been run on more years.

### Step 1. Supply the data in the same structure
- **Crop table:** one row per district, year and crop, in the structure of `crop_long_v1.csv`, including area, production, yield and the data-quality flags.
- **Weather:** per district-year, with the same 13 indicators x 6 windows (78 columns) as `weather_stress_features_v1.csv` / the crop-long table.
- A single **consistent district key** across the whole history (see Section 6).
- A **unit and window-date dictionary** for the weather columns. We do not have one, and it matters for interpretation.

### Step 2. Crop-long table (Issue 1) **(review required)**
```bash
python CropLongFormatv1/crop_restructure_validate.py --input <wide_file.csv> --outdir CropLongFormatv1
```
**Warning:** this script contains corrections tied to specific years in our data: a sugarcane unit fix for years up to 2017, a flag for cotton in 2019, and a minor-pulses flag from 2018. **They are data fixes for our file, not general rules.** A longer history needs its own review of units and outliers; do not assume these fixes apply to other years.

### Step 3. Crop eligibility (Issue 2)
The repository holds the Issue 2 **outputs** (`crop_model_eligibility_v1.csv` and a PDF report) but **I could not find a script that regenerates them.** With more years the set of eligible crops can change. Eligibility must be reassessed, and the Issue 7 builder reads whatever `proceed_to_modelling` column you give it.

### Step 4. Weather and yield anomalies (Issues 3 and 4) **(not tested on a longer history)**
```bash
python weather_feature_engineering/weather_forward_chaining.py --input <crop_long_v1.csv> --outdir weather_feature_engineering
python yield_feature_engineering/yield_feature_engineering.py --input <crop_long_v1.csv> --outdir yield_feature_engineering
```
Both derive their years from the data and use only strictly earlier years for each target year (minimum two). The leave-one-year-out versions (`weather_feature_engineering.py` and the LOYO outputs) are needed only by the optional all-years experiment.

### Step 5. Crop-specific selection (Issue 5) and climate zones (Issue 6)
- Issue 5 (`crop specific weather feature selection/feature_selection.py`) feeds only the informational `loyo_*` columns of the dataset; the training pipeline does not use them. Regenerate it if you want those columns to match the new data.
- Issue 6 (district climate zones) should be **recomputed**: more years change the climate profiles, and the pipeline uses the zone as a feature. Follow that folder's own instructions; I have not verified its command line.

### Step 6. Build the ML-ready dataset (Issue 7)
```bash
python ml_ready_dataset/build_ml_ready_dataset.py \
  --crop-long <crop_long_v1.csv> --eligibility <eligibility.csv> \
  --weather <weather_anomalies_forward_chaining_v1.csv> \
  --yield-anomalies <yield_anomalies_forward_chaining_v1.csv> \
  --feature-selection <crop_feature_selection_v1.csv> --zones <district_zone_map_v1.csv> \
  --outdir ml_ready_dataset
```
The builder derives its counts from the inputs, verifies every join, and runs 36 checks including the leakage and completeness checks; it exits non-zero if any fails. It also writes `missing_data_report_v1.csv`, `excluded_crop_rows_v1.csv` and `weather_loyo_anomalies_v1.csv`, so every gap in the data is listed with its cause (read the report first on new data). **The tests in `ml_ready_dataset/tests` contain assertions tied to our five years** (for example about `rabi_sorghum` and years up to 2017). On new data those tests should be updated, not deleted.

### Step 7. Choose the validation mode and train
Edit `validation` (Section 3), then:
```bash
python yield_forecast/train_forecast.py
```
Read the fold block it prints first.

### Step 8. Run the checks
```bash
python -m pytest yield_forecast/tests -q
```
Several tests compare the committed results with the committed dataset and will fail on new data until you regenerate the results. The new `test_multi_year.py` tests use synthetic data and should pass unchanged.

### Step 9. Optional experiments
```bash
python -m yield_forecast.experiment_scaled_shrinkage
python -m yield_forecast.experiment_lodo
python -m yield_forecast.experiment_five_year_loyo
```
The last one uses all available years with nested leave-one-year-out validation. It needs the leave-one-year-out source files (Step 4) and is **not forecasting**: its baselines use later years.

## 5. Decisions to revisit with more data

With a longer history these choices can be tested properly instead of assumed. Do this with **nested validation inside the training years**, never by tuning against the final test years.

| Choice | Our setting | Why to revisit |
|---|---|---|
| Features per fold | 3, ranked by Spearman correlation with a collinearity filter, **no significance control** | with more rows, add false-discovery control and consider per-crop selection (it covered only 19% of our test rows) |
| Model sizes | 10 to 50 trees or rounds, depth 2 to 3, Ridge alpha 10 | chosen for about 128 rows; tune with nested temporal validation |
| Per-crop target scaling | divide by the crop's training standard deviation | adopted after seeing our test years; confirm it holds on many test years |
| Baseline | per-crop mean of training anomalies | with more years, consider a trend or a rolling window |
| Prediction intervals | in-sample residual quantile, uncalibrated | with many folds, calibrate from out-of-fold residuals. On our data out-of-fold, per-crop-scaled conformal intervals were sharper and far more even across crops, but a single training year cannot reveal a year-level shift (coverage stayed about 72-75%); with several training years calibrate with leave-one-year-out inside the training window (see `experiment_calibrated_intervals.py`) |
| Minimum observations for a baseline | 2 earlier years | a longer history supports a higher minimum and more stable normals |
| Crop eligibility thresholds | Issue 2 report | more years can change which crops qualify |
| Climate zones | 3 zones from 2015 to 2019 | recompute on the longer history |
| Weather windows | six windows, dates not documented | confirm which windows are known before harvest if an advance forecast is intended |

## 6. Cautions

- **Leakage.** Never use the leave-one-year-out anomalies as targets in a forward-chaining run; their baselines contain later years. The pipeline's fold validation protects against a leaking fold, not against leaking features.
- **Do not change the design after seeing test results.** Fix the mode, thresholds and settings first, or you will repeat the optimism we found in our own +9%.
- **Missing data stays missing.** Do not fill gaps to enlarge the training set.
- **A consistent district key.** Administrative boundaries and codes can change over a long history (for example Andhra Pradesh's 2022 reorganisation from 13 to 26 districts). Each district-year must map to one stable key, or the baselines will mix different areas.
- **Crops and definitions change** (new crops, merged categories, unit changes). Re-run the eligibility assessment and review the data-quality flags.
- **Sample size is the real limit.** Count distinct weather observations (district-years), not rows: weather repeats across crops. We had 39.
- **Association is not causation**, and a selected feature is not a causal driver.

## 7. What to report from a longer run

1. The fold table (test years, training rows, years used).
2. Skill against the baseline **by test year** and the number of years the model beats the baseline, not only the pooled figure.
3. Per-crop skill with the low-sample flags.
4. The district-clustered test results and the number of test years behind them.
5. Interval coverage against the nominal level, with how the intervals were calibrated.
6. The settings and the commit used, the package versions, and the dataset hash from `ml_ready_dataset/reproducibility_report_v1.json`.
7. The limitations that still apply.

## 8. What is verified and what is not

| Verified | Not verified |
|---|---|
| The pipeline builds folds from the data, trains only on earlier years, and recovers a planted signal on a synthetic 12-year history (tests in `yield_forecast/tests/test_multi_year.py`) | Behaviour on real data with a different structure, missingness or crop set |
| On our data the refactor changed no result (every output file byte-identical) | The Issue 1, 2, 5 and 6 steps on a longer history; the Issue 3 and 4 scripts are year-agnostic in reading but untested |
| The fold generator rejects leaking, empty or thin folds | Run time on a large history (not measured) |
