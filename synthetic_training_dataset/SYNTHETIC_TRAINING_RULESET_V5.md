# Synthetic Training Dataset (v5): Minimal Insurance Fields, 2015 to 2019

> **Label: SYNTHETIC. This is not PNB data.** PNB supplies field definitions only, so this is simulated data for model development and testing.
> **Approval status:** v5 changes the insurance fields and date ranges after v4. Its approval is not recorded. v3 and v4 are unchanged and remain available.
> Everything not listed here is unchanged from v4 (`SYNTHETIC_TRAINING_RULESET_V4.md`): four agri sectors, class mix 30/20/18/16/16 in every year, LOYO weather and yield anomalies, seed 2026.

## 1. What changed from v4

### 1.1 Insurance: three fields only

| Written | Meaning |
|---|---|
| `crop_insurance_available` | `Y` if the account has crop insurance, otherwise `N` |
| `insurance_start_date` | Policy start date (blank when `N`) |
| `insurance_end_date` | Policy expiry date (blank when `N`) |

No longer written: `insurance_amount`, `insurance_coverage_status`, `insurance_claim_status`, `insurance_payout_amount`, and the derived `insurance_days_to_expiry` and `insurance_coverage_ratio`. Policy id, scheme, premium, provider, claim amount and payout date were already absent in v3 and v4.

How the dates are built: the generator still draws an internal coverage state (Active 70%, Expired 22%, Pending 8% of insured accounts, relative to the reporting date) to place start and end dates. That state is not written. A model can derive "expired at reporting date" from `insurance_end_date` and `reporting_date`.

Risk-score effect: the `coverage_shortfall` term (needs an insured amount) is set to 0. The `uninsured_agri` and `expired_policy` terms remain. Claims no longer exist in the data, so claim-based effects are gone.

### 1.2 Time frame: every date inside 2015 to 2019

| Item | v4 | **v5** |
|---|---|---|
| Reporting dates | 2015-10-31 to 2019-10-31 | Same |
| Account open dates | From 2012-01-01 | **From 2015-01-01** |
| Insurance end dates | Active policies could run into 2020 | **Capped at 2019-12-31** |
| Train / test | Train 2015 to 2017, test 2018 to 2019 | Same |

Consequences to be aware of:
- **Loans are younger.** Accounts cannot be opened before 2015, so `loan_age_days` is small in the 2015 snapshot (accounts opened between 2015-01-01 and early August 2015). Account age therefore tells less in 2015 than in later years.
- **Capped end dates.** For insured accounts whose 364-day policy would end after 2019-12-31, the end date is cut to 2019-12-31, so a few policies are shorter than 364 days. End is always after start.

## 2. Verification

`verify_training_dataset.py` adds four checks: V20 (only the three insurance fields are written), V20b (`Y` or `N` only), V20c (insured rows have start and end date, uninsured rows have neither) and V21 (every date inside 2015-01-01 to 2019-12-31). Checks that need dropped columns (V10, V11, V13 to V16) are reported as SKIP, never PASS.

## 3. Results (actual run, seed 2026)

- Rows: **10,002 train (2015 to 2017), 998 test (2018 and 2019)**; 39 columns in the rich files, plus the PNB-layout files for the same accounts: **23 columns = the 20 PNB loan fields + the 3 PNB insurance fields** (`crop_insurance_available`, `insurance_start_date`, `insurance_end_date`, appended after `asset class`). In v3 and v4 the PNB-layout files had 20 columns.
- Rows per year: 2015 3,330; 2016 3,337; 2017 3,335; 2018 498; 2019 500. Class mix 30/20/18/16/16 in every year.
- Insured share overall 45.0%: Crop Production 70.5%, Fisheries 40.4%, Dairy 14.9%, Poultry 14.6%. By year: 45.8%, 43.4%, 46.0%, 44.4%, 44.2% (2015 to 2019).
- Date ranges: account open 2015-01-01 to 2019-07-31; insurance start 2015-01-01 to 2019-12-14; insurance end 2015-02-15 to 2019-12-31; reporting 2015-10-31 to 2019-10-31.
- Verifier: 16 passed, 0 failed, 7 skipped (dropped columns).
- Existing `validate_loan_portfolio.py` on the 23-column files with the new `loan_portfolio_schema_v5.json` (v4 schema plus the 3 insurance columns, all optional): no errors. It WARNs that the two insurance date columns have blanks (5,493 train rows); that is expected, since uninsured accounts have no policy dates.
- Tests: 37 pass (`python3 -m pytest synthetic_training_dataset/tests synthetic_portfolio/tests`).
- Baseline gradient boosting, train 2015 to 2017, test 2018 to 2019: accuracy 0.453 (majority class 0.300), macro-F1 0.418 (shuffled-label control 0.099). This is a pipeline check on rules the generator injected, not a finding about real borrowers.

## 4. Limitations

- Insurance take-up rates (70% crop, 40% fisheries, 15% dairy and poultry) and the Active/Expired/Pending split are assumptions with no public reference.
- With claims removed, the data cannot show insurance paying out after a weather event.
- Insurance is only "has a policy and when", so insurance adequacy (amount against outstanding) cannot be modelled from this data.
- Labels come from a documented risk score; a model recovers those rules. No causal claim about weather and default.

## 5. Reproduce

```bash
python3 synthetic_training_dataset/generate_training_dataset.py --config synthetic_training_dataset/synthetic_training_config_v5.json
python3 synthetic_training_dataset/verify_training_dataset.py --config synthetic_training_dataset/synthetic_training_config_v5.json --baseline
```

Output goes to `financial_train_test_data/v5/`.
