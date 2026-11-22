# Issue 5 — Crop-specific weather feature selection: report

Covers Issue 5 and the Issue 36 foundation fix (adds the required
`crop_feature_selection_v1.csv`).

## 1. Inputs

| Input | Path |
|---|---|
| Forward-chaining weather anomalies | `weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv` |
| Forward-chaining yield anomalies | `yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv` |

## 2. Re-run command (from the repository root)

```bash
python3 -m pip install pandas numpy scipy

python3 "crop specific weather feature selection/feature_selection.py" \
  --weather weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv \
  --yield-anomalies yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv \
  --outdir "crop specific weather feature selection"
```

Verified with Python 3 / pandas 3.1.1 / scipy 1.18.1.

## 3. Method (plain language)

For each eligible crop, each of the 78 weather anomaly features
(13 indicators x 6 seasonal windows) is compared with the crop's yield
anomaly z-score using Spearman rank correlation on pairwise-complete
observations (no imputation). P-values are adjusted with
Benjamini-Hochberg FDR. A feature passes the screen if
`|rho| >= 0.30` and `q <= 0.20`. Passing features are ranked and any
feature with `|rho| > 0.85` against an already-chosen feature is dropped
(multicollinearity). At most 4 features are kept per fold.

Selection is repeated in each Leave-One-Year-Out (LOYO) fold with the
held-out year removed from correlation, significance and selection. A
feature is a **stable predictor** for a crop if it is selected in at
least 2 of the 3 scorable folds (held-out years 2017, 2018, 2019).

Settings: `MIN_N=8`, `COVERAGE=0.60`, `MIN_RHO=0.30`, `MAX_Q=0.20`,
`MAX_CORR=0.85`, `MAX_FEATURES=4`, `MIN_STABLE=2`.

## 4. Important limitations

- **LOYO selection is not prospective forecasting.** The weather
  anomalies are forward-chaining (reference built only from earlier
  years), but the selection folds are LOYO, so a fold's training set can
  contain years *later* than the held-out year. Do not describe this as a
  chronological forecast.
- **Stable predictors are statistical associations only.** They are not
  causal drivers, and a feature that was not selected does not prove
  that the variable has no agricultural effect.
- This is exploratory feature selection, not a validated yield
  prediction model. Samples are small (13 districts, 3 scorable years;
  several crops have 1-8 usable rows).
- `ne` and `kharif` windows overlap; they are not independent.
- Source weather units are not defined in the inputs; none are assumed.
- Missing values are never treated as zero or imputed.

## 5. Outputs

| File | Role |
|---|---|
| `crop_feature_selection_v1.csv` | **Required Issue 5 deliverable.** One row per crop x feature (see section 6). |
| `crop_feature_selection_consensus.csv` | One row per crop: stable predictors and status. |
| `crop_feature_selection_by_fold.csv` | Selected features per crop per LOYO fold. |
| `crop_feature_scores.csv` | Per fold/feature rho, p, q and screen flags. |
| `crop_feature_selection_validation.csv` | Per fold leakage-control record. |
| `validation_checks.csv` | The 8 validation checks. |

The five original outputs are preserved; `crop_feature_selection_v1.csv`
was added.

## 6. Schema of `crop_feature_selection_v1.csv`

Grain: one row per (`crop`, `feature`). 17 eligible crops x 78 features
= 1,326 rows. All statistics are over the **scorable** LOYO folds
(held-out year 2017, 2018, 2019) so counts match the consensus file.
Per-fold statistics are computed on that fold's training rows.

| Column | Meaning |
|---|---|
| `crop` | Eligible crop. |
| `feature` | `<season_window>__<weather_indicator>` anomaly z-score. |
| `season_window` / `weather_indicator` | The two parts of `feature`. |
| `usable_yield_anomaly_rows` | Usable yield-anomaly rows for the crop (sample size). |
| `district_count` | Distinct districts for the crop. |
| `scorable_loyo_folds` | Folds with a scorable held-out year (3). |
| `folds_evaluated` | Folds in which the feature had enough pairs to be scored. |
| `folds_selected` | Folds in which the feature was selected. |
| `selection_frequency` | `folds_selected / scorable_loyo_folds`. |
| `is_stable_predictor` | Selected in at least `MIN_STABLE` (2) folds; matches the consensus file. |
| `mean/min/max_spearman_rho` | Spearman rho across evaluated folds. |
| `mean_abs_spearman_rho` | Mean of `|rho|`. |
| `rho_sign_consistent` | True if rho kept one sign in all evaluated folds. |
| `min/mean/max_bh_q_value` | Benjamini-Hochberg FDR q-value across evaluated folds. |
| `min/max_paired_n` | Pairs used in the correlation across folds. |
| `folds_passing_screen` | Folds where `|rho| >= 0.30` and `q <= 0.20`. |
| `folds_rejected_multicollinearity` | Folds where it passed but was dropped as redundant. |
| `selected_in_heldout_years` | Held-out years (`;`-separated) in which it was selected. |
| `crop_selection_status` | Crop status from the consensus file. |
| `interpretation` | Always `statistical_association_only`. |

Features never evaluated (insufficient pairs) keep their row with empty
statistics and `folds_evaluated = 0`; empty means "not computed", not zero.

## 7. Results of the recorded run

- Eligible observations 436; eligible crops 17; weather features 78;
  scorable years 2017, 2018, 2019.
- All 8 validation checks: PASS.
- `crop_feature_selection_v1.csv`: 1,326 rows, no duplicate
  (`crop`, `feature`) keys, stable predictors identical to the
  consensus file.

| Crop | Stable predictor | Selected in | Mean rho | Sign consistent |
|---|---|---|---|---|
| cotton | `ne__rainy_days` | 2/3 | 0.456 | No (-0.05 to 0.71) |
| maize | `annual_crop__hot_days` | 2/3 | -0.526 | Yes |
| pigeonpea | `annual_crop__hot_days` | 2/3 | -0.605 | Yes |
| sunflower | `kharif__dtr_mean` | 2/3 | -0.455 | No (-0.79 to 0.16) |

Linseed, safflower and wheat have too few usable observations
(`insufficient_evidence`). The other 10 crops have no stable predictor
under the current criteria. Cotton and sunflower flip rho sign across
folds, so treat them with extra caution.

## 8. Known discrepancy

Re-running on the current inputs reproduces the committed
`crop_feature_selection_by_fold.csv`, `_consensus.csv`,
`_validation.csv` and `validation_checks.csv` exactly, and every
selection/flag in `crop_feature_scores.csv`. However the committed
`crop_feature_scores.csv` has `spearman_rho`, `p_value` and
`bh_q_value` values that differ from a fresh run by up to about 0.035,
0.063 and 0.052 respectively (the inputs are unchanged between `Main`
and `dev`), so it was probably generated with a different scipy/tie
handling. It was kept unchanged as the original output;
`crop_feature_selection_v1.csv` uses the fresh run. No selection
decision is affected.
