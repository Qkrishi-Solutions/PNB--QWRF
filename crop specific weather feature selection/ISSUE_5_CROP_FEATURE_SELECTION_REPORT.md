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
| `crop_feature_scores.csv` | Per fold/feature rho, p, q and screen flags, including `dropped_for_max_features`. |
| `crop_feature_selection_validation.csv` | Per fold leakage-control record. |
| `validation_checks.csv` | The 8 validation checks. |

The five original outputs keep their names and columns;
`crop_feature_selection_v1.csv` was added. All six files are regenerated
together by the one command in section 2, and a second run is
byte-identical.

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
| `selection_frequency_fraction` | `folds_selected / scorable_loyo_folds`, numeric (0 to 1). Not the same as the text `selection_frequency` column (`feature:count/n`) in the consensus file. |
| `is_stable_predictor` | `folds_selected >= MIN_STABLE` (2), computed from the numeric counts. Cross-checked against the consensus file's untruncated `stable_predictor_count`. |
| `mean/min/max_spearman_rho` | Spearman rho across evaluated folds. |
| `mean_abs_spearman_rho` | Mean of `|rho|`. |
| `rho_sign_consistent` | True if rho kept one sign in all evaluated folds. |
| `min/mean/max_bh_q_value` | Benjamini-Hochberg FDR q-value across evaluated folds. |
| `min/max_paired_n` | Pairs used in the correlation across folds. |
| `folds_passing_screen` | Folds where `|rho| >= 0.30` and `q <= 0.20`. |
| `folds_rejected_multicollinearity` | Folds where it passed the screen but was dropped as redundant. |
| `folds_dropped_for_max_features` | Folds where it passed the screen and was not redundant, but was dropped because `MAX_FEATURES` (4) selected features were already chosen. |
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

## 8. Validation checks

The 8 checks in `validation_checks.csv` keep their names. Their logic was
tightened so they test the data and not constants:

- `weather_feature_count`: 78 features, and equal to the number of
  distinct `weather_column` values in the weather file.
- `heldout_year_excluded_from_training` and `heldout_features_not_used`:
  the held-out flag is measured (training frame shares no rows or year
  with the held-out frame) and no longer set to a constant `False`.
- `missing_values_not_imputed`: non-missing counts are unchanged by the
  pivot and merge (weather anomalies and yield z-scores).
- `scorable_years`: derived from the inputs (non-empty, within the weather
  years) and no longer a hardcoded `[2017, 2018, 2019]`, so valid new data
  does not abort the run. For the current inputs it is still 2017-2019.

Additionally, building `crop_feature_selection_v1.csv` raises an error if
it has duplicate keys, is not exactly crops x features, or its stable
predictors disagree with the consensus file.

## 9. Change in `crop_feature_scores.csv` values

The committed `crop_feature_scores.csv` (from PR #14) had `spearman_rho`,
`p_value` and `bh_q_value` values that differ from a fresh run on the same
inputs by up to about 0.035, 0.063 and 0.052. The cause was not identified
(probably a different scipy or tie handling when it was first generated).
This version regenerates the file so that every shipped output comes from
one reproducible run. Every selection decision and screen flag
(`selected`, `passes_screen`, `rejected_for_multicollinearity`) is
identical to before, and so are the by-fold, consensus and validation
files. The scores file also gains one column, `dropped_for_max_features`
(769 fold-feature rows). Anyone who cited old rho, p or q values from that
file should re-read them.
