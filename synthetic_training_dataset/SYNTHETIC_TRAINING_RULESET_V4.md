# Synthetic Training Dataset (v4): Four Agri Sectors

> **Label: SYNTHETIC. This is not PNB data.** PNB supplies field definitions only, so this is simulated data for model development and testing.
> **Approval status:** v3 was reported approved by PNB (per the project owner). **v4 changes the `agri sector` domain and its approval is not recorded.** v3 is unchanged and remains available.
> Everything not listed here is unchanged from v3 (`SYNTHETIC_TRAINING_RULESET_V3.md`).

## 1. What changed from v3

The `agri sector` column now holds **only four values**:

| Value | Meaning | Share of accounts |
|---|---|---|
| `Crop Production` | Crop agriculture | 45% |
| `Dairy & Animal Husbandry` | Dairy | 25% |
| `Fisheries` | **Prawn and shrimp farming only** | 20% |
| `Poultry` | Poultry | 10% |

Removed: Horticulture, Irrigation, Farm Mechanization, Non-Agri.

| # | Detail | Why |
|---|---|---|
| 1 | `Crop Production`, `Dairy & Animal Husbandry` and `Poultry` keep the original PNB labels | They are already in the PNB field domain, so existing validation still applies to them |
| 2 | `Fisheries` is a **new** value, not in the original 25-row sample | Requested. The prawn/shrimp restriction is a documentation rule: there is no species column, and the generator does not model species |
| 3 | `Fisheries` reuses existing scheme and facility combinations (`KCC02` Kisan Credit Card, `AGRI01` Cash Credit, `TL101` Term Loan) | No new scheme codes are invented. Which schemes PNB really uses for fisheries is **unknown and must be confirmed** |
| 4 | 9 product combinations (scheme, facility, sector), weights sum to 100 | See `product_mix_percent_override` in `synthetic_training_config_v4.json`. Weights are synthetic choices, tuned for learnability |
| 5 | Weather sensitivity: Crop 1.0, Fisheries 0.6, Dairy 0.3, Poultry 0.2 | Synthetic assumptions. Fisheries (coastal prawn/shrimp ponds) is set above Dairy and Poultry as an assumption, not a measured effect |
| 6 | Insurance availability: Crop 70%, Fisheries 40%, Dairy 15%, Poultry 15% | Synthetic assumptions, no public reference |
| 7 | Only `Crop Production` carries a crop and yield anomaly | Fisheries, Dairy and Poultry have no crop yield data in the project |

## 2. Validator schema for v4

The PNB-derived validator schema (`financial_data_validation/loan_portfolio_schema_v1.json`) allows seven sector values and **does not include `Fisheries`**. Run against it, the v4 files fail the allowed-values check (189 of 999 test rows). v4 therefore ships `loan_portfolio_schema_v4.json` (a copy with the sector list narrowed to the four values), used via `--schema`. The Issue 24 validator and schema themselves are not modified. If PNB confirms `Fisheries` as a real value, the main schema should be updated.

## 3. Results (actual run, seed 2026)

- Rows: **10,001 train, 999 test** (out-of-time: train 2017-2018, test 2019); 44 columns in the rich files, no `synthetic_flag` column.
- Class mix 30 / 20 / 18 / 16 / 16 in every year (NPA 16.0% in 2017, 2018 and 2019).
- Sector counts: Crop Production 4,952; Dairy 2,752; Fisheries 2,196; Poultry 1,100. Accounts with a crop and a calculated yield anomaly: 4,946.
- Insured accounts 45.0% (Crop 71%, Fisheries 38%, Poultry 16%, Dairy 15%).
- Verification: **15 passed, 0 failed, 4 skipped** (V1 label column, V12, V14, V15: their columns are not written). New check V19 (only the four allowed sectors, each present) passes and fails when an unexpected sector is injected.
- Validator with the v4 schema on the 20-column files: 56 PASS, 1 INFO, no errors (train and test).
- Tests: 25 passed (`pytest synthetic_training_dataset/tests synthetic_portfolio/tests`).

Baseline gradient boosting on the rich features, trained 2017-18, scored 2019:

| | Accuracy | Macro-F1 |
|---|---|---|
| Model | 0.528 | 0.474 |
| Shuffled-label control | 0.298 | 0.094 |
| Majority-class baseline | 0.299 | - |

(v3 scored accuracy 0.469 and macro-F1 0.396.) The gain comes mostly from the larger Crop Production share (45% vs 32%), which puts the yield-anomaly signal on more rows.

## 4. Limitations (in addition to v3's)

- Fisheries NPA is lower than the other sectors in this data (10% vs 17-19%) because it is more often insured (38% vs 15% for Dairy) and has no yield term in the score. That is an artefact of the assumptions here, not a statement about real fisheries lending.
- Species (prawn or shrimp) is not a column and has no effect on any value.
- The 45/25/20/10 sector mix and all sector-level effects are synthetic choices, not PNB or public figures.
- All v3 limitations still apply: labels follow a documented risk score, class cut-offs are applied within each year, accounts are independent snapshots, and the data is not evidence about real borrowers.

## 5. Reproduce

```bash
python3 synthetic_training_dataset/generate_training_dataset.py \
  --config synthetic_training_dataset/synthetic_training_config_v4.json \
  --zone-map "District Climate Zonal Clustering/district_zone_map_v1.csv"

python3 synthetic_training_dataset/verify_training_dataset.py \
  --config synthetic_training_dataset/synthetic_training_config_v4.json --baseline

python3 financial_data_validation/validate_loan_portfolio.py \
  --input financial_train_test_data/v4/Financial_data_PNB_SYNTHETIC_train.csv \
  --schema synthetic_training_dataset/loan_portfolio_schema_v4.json \
  --pincode-reference address_expansion/pincode_reference_v1.csv
```

Outputs go to `financial_train_test_data/v4/`: `Financial_data_PNB_SYNTHETIC_train.csv` and `_test.csv` (20-column layout), `synthetic_training_v4_train.csv`, `_test.csv` and `synthetic_training_v4_data_dictionary.csv`.
