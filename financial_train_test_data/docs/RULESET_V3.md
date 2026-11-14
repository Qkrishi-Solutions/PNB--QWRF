# Synthetic Training Dataset (v3): Changes from v2

> **Label: SYNTHETIC. This is not PNB data.** PNB supplies field definitions only, so this simulated data is the deliverable for development, demos and UAT-style testing. It is not a statement about PNB's real portfolio, and any model trained on it recovers the rules in `synthetic_training_config_v3.json`.
> Everything not listed here is unchanged from `SYNTHETIC_TRAINING_RULESET_V2.md`. v2 files are kept in `financial_train_test_data/v2/` for comparison.

## 1. What changed and why

| # | Change | Why | Where |
|---|---|---|---|
| 1 | **6 insurance fields instead of 13** | Keep only the fields that matter for the requirements: insured vs uninsured, expiry, coverage vs exposure, claims and payouts | `insurance_output_fields` |
| 2 | **Asset-class mix 30 / 20 / 18 / 16 / 16** (Standard, SMA 0, SMA 1, SMA 2, NPA) | More balanced classes give a model more examples of each class. **Chosen for learnability, not realism.** Real agri books have far more Standard | `asset_class_percent_override` |
| 3 | **Crop Production 32% of accounts (was 4%); Non-Agri 12% (was 36%)** | Puts the crop and yield-anomaly signal on about 3,500 accounts instead of 440. Uses only the 17 product combinations already in v1. **A synthetic choice, not a PNB or public figure** | `product_mix_percent_override` |
| 4 | **Class cut-offs applied within each snapshot year** | With a pooled ranking, a harsh weather year absorbed most NPA (NPA 1.8% / 32.9% / 2.2% in 2017 / 2018 / 2019 in the first v3 attempt), so the test year looked nothing like training. Now every year has the target mix, and weather differentiates districts and sectors within a year | `rank_within_snapshot_year` |
| 5 | **Lower noise: `noise_sd` 0.25 (was 0.6)** | More learnable. Signal share of the score is 0.91 (was 0.44). Going lower (0.15 gave 0.97) makes the label an almost exact formula | `risk_score.noise_sd` |

## 2. The 6 insurance fields

| Field | Meaning | Role |
|---|---|---|
| `crop_insurance_available` | Y or N | feature |
| `insurance_amount` | Insured amount (INR) | feature |
| `insurance_end_date` | Policy end | feature (raw date) |
| `insurance_coverage_status` | Active, Expired, Pending, Not Insured | feature |
| `insurance_claim_status` | No Claim, Filed, Approved, Rejected, Paid | `post_event_exclude` |
| `insurance_payout_amount` | Amount paid | `post_event_exclude` |

Derived features computed from them before the fields are written: `insurance_days_to_expiry` and `insurance_coverage_ratio` (insured amount / outstanding balance). The generator still builds all 13 fields internally, because claim and status rules need start dates and amounts. The other 7 (policy id, scheme, premium, start date, provider, claim amount, payout date) are not written.

## 3. Results (actual run, seed 2026)

- Rows: **10,001 train, 999 test** (out-of-time: train 2017 and 2018, test 2019). 45 columns in the rich files.
- Asset-class mix: train and test both 30 / 20 / 18 / 16 / 16 (NPA 16.0% in every year).
- Insured accounts 39.1%; coverage: Not Insured 6,704, Active 2,974, Expired 958, Pending 364. Claims: Paid 369, Approved 212, Filed 177, Rejected 124.
- Accounts with a crop: 3,520 (3,511 with a calculated yield anomaly).
- Verification: **15 passed, 0 failed, 3 skipped** (V12, V14, V15 need columns v3 no longer writes: start date, premium, claim amount; they are reported as SKIP, not PASS). The existing loan validator accepts the 20-column files: 56 PASS, 1 INFO, no errors, on both train and test.
- Tests: 18 passed (`pytest synthetic_training_dataset/tests synthetic_portfolio/tests`).

Baseline gradient boosting on 27 feature columns, trained on 2017-18, scored on 2019:

| | Accuracy | Macro-F1 |
|---|---|---|
| Model | 0.469 | 0.396 |
| Shuffled-label control | 0.290 | 0.100 |
| Majority-class baseline | 0.299 | - |

For reference, v2 scored accuracy 0.411 and macro-F1 0.278 against a majority baseline of 0.386.

How `noise_sd` affects learnability (first v3 attempt, pooled ranking, so not directly comparable with the table above): macro-F1 out-of-time 0.31 / 0.38 / 0.43 / 0.47 at noise 0.5 / 0.35 / 0.25 / 0.15. Edit `risk_score.noise_sd` to change it.

## 4. Limitations (in addition to v2's)

- The labels are my assumptions. Because class cut-offs are applied per year, the data cannot show that a bad weather year produces more NPA. Year-level weather affects only who is ranked worst within a year.
- The 30/20/18/16/16 mix and the 32% Crop Production share are for learnability. Do not quote either as realistic or as PNB's asset quality or lending mix.
- Insurance take-up, status mix and claim rules are assumptions without a public reference. RBI/NABARD district benchmarks are still not applied.
- Dropping 7 insurance fields means the start-date, premium and claim-amount rules are not checkable from the written files, only from the in-memory tests.
- Each snapshot year has only 13 distinct weather vectors, and accounts are independent snapshots with no repayment history.

## 5. Reproduce

```bash
python3 synthetic_training_dataset/generate_training_dataset.py \
  --config synthetic_training_dataset/synthetic_training_config_v3.json \
  --zone-map "District Climate Zonal Clustering/district_zone_map_v1.csv"

python3 synthetic_training_dataset/verify_training_dataset.py --baseline
```

Outputs in `financial_train_test_data/v3/` (committed on explicit request, issue #32; all SYNTHETIC): `Financial_data_PNB_SYNTHETIC_train.csv` and `_test.csv` (exact 20-column Financial_data_PNB layout), `synthetic_training_v3_train.csv`, `synthetic_training_v3_test.csv` and `synthetic_training_v3_data_dictionary.csv`. The aggregate report is `synthetic_training_report_v3.csv` in this folder. v2: pass `--config .../synthetic_training_config_v2.json` (outputs go to `financial_train_test_data/v2/`).
