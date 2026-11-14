# Synthetic Training Dataset (v2): Ruleset and Limitations

> **Label: SYNTHETIC. This is not PNB data.** A development fixture for building and testing an asset-class prediction pipeline.
> **Extends:** `synthetic_portfolio/` (v1 generator, config and ruleset). v1 is unchanged.

## 1. What it is

About 11,000 synthetic loan-account rows: **~10,000 train** (snapshot years 2017 and 2018) and **~1,000 test** (snapshot year 2019). Exact counts are in `synthetic_training_report_v2.csv` (actual run: 9,999 train, 1,001 test; customer-grouped assignment makes counts approximate).

**Target:** `asset class` (Standard, SMA 0, SMA 1, SMA 2, NPA; ordered). A binary `npa_flag` is included for convenience.

## 2. How it differs from v1, and why

| | v1 (`synthetic_portfolio`) | v2 (this folder) |
|---|---|---|
| Purpose | Plumbing test at volume | Training/test data for a prediction pipeline |
| Asset class | Drawn independently of every other field (rule P8: no engineered association) | **Derived from a risk score** (section 4). P8 is deliberately set aside for v2 |
| Time | One reporting date (2026-10-31) | Snapshot years 2017-2019, so real weather and yield anomalies can be joined |
| Insurance | None | 13 insurance fields |
| Split | None | Out-of-time: train 2017+2018, test 2019 |

Why the change: with v1's independent draws there is nothing to learn. v2 injects documented relationships so the pipeline can be exercised end to end. **A model trained on v2 recovers the injected rules. It is not evidence about real borrowers, real weather-credit links or PNB's portfolio.**

## 3. Inputs

| Input | Used for |
|---|---|
| `synthetic_portfolio` generator and `synthetic_config_v1.json` | Geography, branches, customers, products, sanction, rate, asset-class mix (11,000 accounts) |
| `weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv` | District-year anomaly z-scores, 6 columns. Real project data, joined by `district_code` and snapshot year. Missing values stay missing |
| `yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv` | Crop area weights; yield anomaly z-score only where status is `calculated` |
| `District Climate Zonal Clustering/district_zone_map_v1.csv` | `climate_zone` (similarity group, not a risk level) |
| `address_expansion/pincode_reference_v1.csv` | Mapped pincodes per district |

## 4. Generation rules

1. **Accounts.** 11,000 accounts from the v1 generator (seeded). District counts follow v1 (2011 population, largest remainder).
2. **Snapshot year.** Each customer (with all their accounts) is assigned to one year, stratified by district, in proportion 5000:5000:1000. No customer appears in both train and test. `reporting_date` = `<year>-10-31`. Account open date is redrawn uniformly between 2012-01-01 and 90 days before the reporting date.
3. **Weather.** District-year anomaly z-scores for `kharif__rain_sum`, `kharif__longest_dry`, `annual_crop__hot_days`, `kharif__tmax_mean`, `kharif__dtr_mean`, `ne__rainy_days` appear as `wx_*_z` columns. Four of them (signs in config) form a stress index used **only inside the generator**: mean of sign x z clipped to +/-3. The index itself is not a column.
4. **Crop.** Only `Crop Production` accounts carry a crop (about 4% of accounts, inherited from the v1 product mix), drawn by district-year area share. Yield z-score is attached only when calculated; otherwise blank.
5. **Insurance.** Availability by sector (config). Status Active 70 / Expired 22 / Pending 8 of insured. Policy term 364 days. Insured amount 0.5-1.2 x sanction; premium 2-5% of insured amount. Dates are generated so that status, start, end and reporting date agree. Claims occur only for Active or Expired policies under stress (sector sensitivity x stress index above threshold, or yield z at or below -1), with claim and payout rules in config. Scheme and insurer names are synthetic placeholders (`SYN-*`), not real products.
6. **Risk score** = weather x sector sensitivity x stress index + yield shortfall + uncovered agri account (uninsured or pending) + expired policy + coverage shortfall + standardised log sanction + standardised rate + standardised loan age + Gaussian noise. All coefficients are in `synthetic_training_config_v2.json` and are **assumptions, not estimates from PNB data**. The achieved share of score variance from the signal terms is recorded in the report (0.4375 in the run below).
7. **Asset class.** Accounts are ranked by score over the whole file; the lowest 36% are Standard, then 20% SMA 0, 16% SMA 1, 12% SMA 2, highest 16% NPA (v1's stress-heavy mix, not PNB's real asset quality). Ranking is pooled across years, so a harsh weather year yields more NPA than a mild one (NPA share: 2017 8.1%, 2018 24.6%, 2019 12.1%). This gives a realistic distribution shift between train and test.
8. **DPD and overdue.** Drawn within the class band (v1 ranges) after the class is set.

## 5. Column roles (see `synthetic_training_v2_data_dictionary.csv`)

| Role | Columns | Use |
|---|---|---|
| `feature` | loan, geography, weather, yield, insurance state, derived ratios | Model inputs |
| `feature_raw_date` | reporting, open, insurance start/end dates | Prefer the derived `loan_age_days` and `insurance_days_to_expiry` |
| `identifier_exclude`, `geo_id` | customer, account, policy, branch, name, address, pincode | Do not use as features |
| `leakage_exclude` | `days_past_due`, `overdue amount` | Determined by the target. Exclude |
| `post_event_exclude` | claim status/amount, payout amount/date | Observed after the weather event. Exclude for early-warning use |
| `target`, `target_binary` | `asset class`, `npa_flag` | Labels |

## 6. Validation (actual results)

Command: `python3 synthetic_training_dataset/verify_training_dataset.py --datadir <outdir> --baseline`

- 18 of 18 counted rule checks pass over all rows (labelling, out-of-time split, no customer overlap, unique account numbers, X1-X5 financial rules, insurance date/status/claim/payout consistency, crop and yield rules, weather values equal the source).
- The existing `validate_loan_portfolio.py` accepts the 20 original columns: 56 PASS, 1 INFO, no errors.
- `pytest synthetic_training_dataset/tests`: 8 passed.
- Baseline gradient-boosting sanity check on 30 feature columns (pipeline check, not a finding):

| Setup | Accuracy | Macro-F1 |
|---|---|---|
| Out-of-time (train 2017-18, test 2019) | 0.411 | 0.278 |
| Same, labels shuffled (control) | 0.338 | 0.166 |
| Majority-class baseline | 0.386 | - |
| In-time random 80/20 inside 2017-18 only | 0.435 | 0.330 |
| Same, labels shuffled (control) | 0.322 | 0.174 |
| Majority-class baseline | 0.357 | - |

The label is learnable above the shuffled control, but only modestly: five ordered classes, about 56% noise in the score, and the weather effect enters through only 13 district-year vectors per year. Lower `risk_score.noise_sd` to make it more learnable; raise it to make it harder.

## 7. Known limitations

- Synthetic effects were injected by design. Do not report any association found here as a finding.
- Weather and yield anomalies are real project data, but each snapshot year has only 13 distinct weather vectors, so weather effects are learnable from about 26 train district-years. The test year adds 13 more.
- 2017 anomalies rely on a forward-chaining history of two years or fewer, so they are noisy; values are used as supplied.
- Only about 4% of accounts have a crop, so crop and yield features are sparse. Non-Agri accounts are about 36%, inherited from v1.
- Insurance uptake (about 20% of accounts), status mix and claim rules are assumptions, not public references or PNB figures. RBI/NABARD benchmarks are not yet applied to district allocation.
- Accounts are independent snapshots, not a panel: the same account is not followed across years. No repayment history.
- `climate_zone` is a similarity group and has no effect in the risk score; it is a feature with no injected signal.
- `zone` is a constant column.

## 8. Reproduce

```bash
python3 synthetic_training_dataset/generate_training_dataset.py \
  --zone-map "District Climate Zonal Clustering/district_zone_map_v1.csv" \
  --outdir financial_train_test_data

python3 synthetic_training_dataset/verify_training_dataset.py \
  --datadir financial_train_test_data --baseline

python3 -m pytest synthetic_training_dataset/tests -q
```

Outputs in `financial_train_test_data/` (not committed): `Financial_data_PNB_SYNTHETIC_train.csv` and `_test.csv` (exact 20-column Financial_data_PNB layout), plus the richer `synthetic_training_v2_train.csv`, `synthetic_training_v2_test.csv` and `synthetic_training_v2_data_dictionary.csv`. The aggregate report `synthetic_training_report_v2.csv` is written to this folder (counts and hashes only, no row-level values). Same config and code give identical output (seed 2026).

Requirements: Python 3.12, pandas, numpy; `scikit-learn` only for `--baseline`.
