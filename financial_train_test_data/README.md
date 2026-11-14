# financial_train_test_data

> **SYNTHETIC. This is not PNB customer or portfolio data.** PNB supplied field definitions only. Every row here was simulated from those fields and documented assumptions, for training and testing a loan asset-class model.
> **Approval:** reported as reviewed and approved by PNB (per the project owner). No approval reference is recorded here yet.
> **Purpose:** learnability of the model. Class mix and effect sizes are tuned for that, not for realism. Do not quote any figure here as PNB's real asset quality, insurance coverage or lending mix.

## Current version: v3

| Item | Value |
|---|---|
| Train rows | 10,001 (snapshot years 2017 and 2018) |
| Test rows | 999 (snapshot year 2019); out-of-time split, no customer on both sides |
| Target | `asset class` (Standard, SMA 0, SMA 1, SMA 2, NPA); `npa_flag` is the binary version |
| Class mix | 30 / 20 / 18 / 16 / 16 in every year |
| Insurance fields | 6 (`crop_insurance_available`, `insurance_amount`, `insurance_end_date`, `insurance_coverage_status`, `insurance_claim_status`, `insurance_payout_amount`) plus derived days-to-expiry and coverage ratio |
| Seed | 2026 (same config and code give identical files) |
| Baseline check (2017-18 to 2019) | accuracy 0.469 (majority class 0.299), macro-F1 0.396 (shuffled labels 0.100) |

## Layout

```
financial_train_test_data/
├── README.md                     this file
├── docs/                         all documentation (Markdown)
│   ├── DATA_DICTIONARY_V3.md     column-by-column dictionary of the v3 rich files (roles, types, blanks)
│   ├── DATA_DICTIONARY_V2.md     same for v2
│   ├── PNB_FIELD_DICTIONARY_DRAFT.md   the 20 fields PNB supplied (draft dictionary, no row values)
│   ├── RULESET_V3.md             v3 rules, choices, results, limitations
│   ├── RULESET_V2.md             v2 rules (risk score, insurance, split)
│   ├── RULESET_V1_BASE_PORTFOLIO.md    base 20-field portfolio generator rules
│   └── ASSUMPTIONS_V1_BASE_PORTFOLIO.md  base generator assumptions
├── v3/                           CURRENT
│   ├── Financial_data_PNB_SYNTHETIC_train.csv   exact 20-column Financial_data_PNB layout
│   ├── Financial_data_PNB_SYNTHETIC_test.csv
│   ├── synthetic_training_v3_train.csv          45 columns: loan, weather, yield, crop, insurance, targets
│   ├── synthetic_training_v3_test.csv
│   └── synthetic_training_v3_data_dictionary.csv   machine-readable dictionary (source of truth)
└── v2/                           previous version, kept for comparison (same file pattern, 13 insurance fields)
```

## Which file to use

- **Same format as the original `Financial_data_PNB.csv`:** `v3/Financial_data_PNB_SYNTHETIC_*.csv` (20 columns, no labels or extras).
- **Training a model:** `v3/synthetic_training_v3_train.csv` and `_test.csv`. Choose columns from the data dictionary by `role`:
  - use `feature`;
  - do **not** use `identifier_exclude`, `geo_id`, `leakage_exclude` (`days_past_due`, `overdue amount` are determined by the target) or `post_event_exclude` (claim status, payout);
  - `feature_raw_date` columns are better replaced by `loan_age_days` and `insurance_days_to_expiry`.
- **Missing values are not imputed.** Blank means not applicable or not calculated (for example no crop, no yield anomaly, or no insurance policy).

## Known limits

- Labels come from a documented risk score, so a model recovers those rules. It is not evidence about real borrowers or a real weather-to-default link.
- Because class cut-offs are applied within each year, the data cannot show that a bad weather year raises NPA overall.
- Only about 13 distinct weather vectors per year; accounts are independent snapshots with no repayment history.
- Insurance take-up, claims and effect sizes are assumptions without a public reference.

Full detail: `docs/RULESET_V3.md`.

## Regenerate (from the repository root)

```bash
python3 synthetic_training_dataset/generate_training_dataset.py \
  --config synthetic_training_dataset/synthetic_training_config_v3.json \
  --zone-map "District Climate Zonal Clustering/district_zone_map_v1.csv"

python3 synthetic_training_dataset/verify_training_dataset.py --baseline
```

Source code, configs and tests live in `synthetic_training_dataset/` (branch `feat/synthetic-dataset-train-test`). The `docs/` copies were taken from that folder and from `synthetic_portfolio/`; if they disagree, the source folders win.
