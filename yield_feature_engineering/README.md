# Issue #4 - Yield Feature Engineering

## Purpose

Generate district-crop-specific historical yield baselines and yield anomalies using two separate methods:

1. Leave-One-Year-Out (LOYO)
2. Forward Chaining

## Run from the repository root (PowerShell)

```powershell
py .\yield_feature_engineering\yield_feature_engineering.py `
  --input "C:\Users\punithaqkrishi\Downloads\crop_long_v1.csv" `
  --outdir ".\yield_feature_engineering"
```

## Eligibility rules

Start with `modelling_role == include`.

Exclude:
- Missing, non-finite, zero or negative yields.
- Rows marked `flag_nan_filled_with_zero`.
- Rows with positive area but zero production.

Excluded rows remain in the anomaly outputs with an `eligibility_reason`.

Mismatch, yield-jump and estimated-allocation flags are retained for review and do not automatically exclude otherwise eligible rows.

Positive-area / zero-production records are excluded, not corrected. The source data should be reviewed separately.

## LOYO calculations

For each district-crop pair and held-out year, calculate the baseline using eligible observations from all other years. The held-out year is excluded from its own baseline.

- Baseline mean: available with at least one training observation.
- Sample standard deviation: available with at least two observations.
- Absolute, percentage and z-score anomalies: require at least two observations.
- Z-score is unavailable when the baseline standard deviation is zero.
- Low deviation threshold: z <= -2.
- High deviation threshold: z >= +2.

These thresholds are statistical screening heuristics, not agronomic diagnoses.

## Forward Chaining calculations

For each district-crop pair and target year, calculate the baseline using only eligible observations from years strictly earlier than the target year.

No current-year or future-year observations are used.

Example:
- 2015: no prior training observations.
- 2016: uses 2015.
- 2017: uses 2015 and 2016.
- 2018: uses 2015 through 2017.
- 2019: uses 2015 through 2018.

The same minimum observation requirements and anomaly calculations are applied.

## Outputs

### LOYO outputs
- `district_crop_yield_normals_v1.csv`
- `yield_anomalies_v1.csv`
- `validation_report.txt`

### Forward Chaining outputs
- `district_crop_yield_normals_forward_chaining_v1.csv`
- `yield_anomalies_forward_chaining_v1.csv`
- `forward_chaining_validation_report.txt`

### Validation and data-quality outputs
- `validation_tables/validation_checks.csv`
- `validation_tables/forward_chaining_validation_checks.csv`
- `validation_tables/yield_missingness.csv`
- `validation_tables/yield_coverage.csv`

## Validation

LOYO and Forward Chaining have separate validation suites.

Checks include:
- Unique district-crop-year source keys.
- Expected baseline coverage.
- Source-row preservation.
- Baseline matching.
- Training-year and observation-count consistency.
- No validation-year leakage in LOYO.
- No current-year or future-year leakage in Forward Chaining.
- Independent baseline recomputation.
- Independent anomaly formula verification.
- Eligibility consistency.
- Handling of excluded and insufficient-history observations.
- Zero-standard-deviation handling.
- Input file integrity.

## Limitations

LOYO is retrospective and may use years later than the held-out year. It is not a prospective forecasting baseline.

Forward Chaining uses only prior years and is intended for prospective evaluation. Early years may have insufficient historical observations.

The eligibility rules and statistical thresholds should be reviewed with agricultural domain experts before interpreting anomalies as agronomic findings.