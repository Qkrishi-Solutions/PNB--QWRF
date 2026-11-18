# Synthetic Training Dataset (v4): Four Agri Sectors

> **Label: SYNTHETIC. This is not PNB data.** PNB supplies field definitions only, so this is simulated data for model development and testing.
> **Approval status:** v3 was reported approved by PNB (per the project owner). **v4 changes the `agri sector` domain and its approval is not recorded.** v3 is unchanged and remains available.
> Everything not listed here (sections 1, 1b) is unchanged from v3 (`SYNTHETIC_TRAINING_RULESET_V3.md`).

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

## 1b. Dates, train/test split and anomalies (changed)

| Item | v3 | **v4 (now)** |
|---|---|---|
| Training reporting dates | 2017-10-31 and 2018-10-31 | **2015-10-31, 2016-10-31 and 2017-10-31** |
| Test reporting dates | 2019-10-31 | **2018-10-31 and 2019-10-31** |
| Weather and yield anomalies | Forward chaining (earlier years only) | **Leave-one-year-out (LOYO)** |

**Why the test range is 2018 to 2019.**
- It comes after the training years, so the test is genuinely out-of-time. Testing on 2015 to 2017 would be an in-time test.
- The weather data ends in 2019, so nothing later is available.
- One test year is not reliable. With the same model, accuracy is 0.404 on 2018 alone and 0.536 on 2019 alone, because each year has only 13 distinct weather patterns. Two years average out some of that luck.

**Why the anomalies changed to LOYO.** Forward-chaining anomalies need at least two earlier years, so they do not exist for 2015 or 2016 (0% of weather and yield values available). Training rows dated 2015 and 2016, two thirds of the training set, would have had no weather or yield features. LOYO anomalies exist for every year (100% of the six weather features and about 94% of yield anomalies).

**Trade-off.** A LOYO normal for a year uses the other years, including later ones. So a training row's weather anomaly was computed with information from years that are later than it, and the test-year anomalies use the training years. Only weather and yield inputs are affected, never the labels, but the data is no longer a strictly chronological forecasting setup. The `anomaly_method` setting in the config (`loyo` here, `forward_chaining` for v3) records which is used.

**Other dates.** Only the reporting date follows the 2015 to 2017 and 2018 to 2019 ranges. Account open dates still start in 2012, so loans have realistic age (at least 90 days before the reporting date). Insurance end dates fall after the reporting date for active policies, so some run into 2018 (training) and 2020 (test).

## 2. Validator schema for v4

The PNB-derived validator schema (`financial_data_validation/loan_portfolio_schema_v1.json`) allows seven sector values and **does not include `Fisheries`**. Run against it, the v4 files fail the allowed-values check (182 of 998 test rows). v4 therefore ships `loan_portfolio_schema_v4.json` (a copy with the sector list narrowed to the four values), used via `--schema`. The Issue 24 validator and schema themselves are not modified. If PNB confirms `Fisheries` as a real value, the main schema should be updated.

## 3. Results (actual run, seed 2026)

- Generated on the final Issue 25 base generator, same as the regenerated v3.
- Rows: **10,002 train (reporting dates 2015 to 2017), 998 test (2018 to 2019)**; out-of-time, 44 columns in the rich files, no `synthetic_flag` column. Rows per year: 2015 3,330; 2016 3,337; 2017 3,335; 2018 498; 2019 500.
- Class mix 30 / 20 / 18 / 16 / 16 in every year (NPA 16.0% in each of 2015 to 2019).
- Weather features: 0% blank in train and test. Yield anomaly present for 4,949 of the 4,952 accounts with a crop.
- Sector counts: Crop Production 4,952; Dairy 2,752; Fisheries 2,196; Poultry 1,100. Accounts with a crop: 4,952.
- Insured accounts 44.2% (Crop 69%, Fisheries 40%, Poultry 15%, Dairy 14%).
- Verification: **15 passed, 0 failed, 4 skipped** (V1 label column, V12, V14, V15: their columns are not written). New check V19 (only the four allowed sectors, each present) passes and fails when an unexpected sector is injected.
- Validator with the v4 schema on the 20-column files: 56 PASS, 1 INFO, no errors (train and test).
- Tests: 30 passed (`pytest synthetic_training_dataset/tests synthetic_portfolio/tests`).
- Reproducibility: regenerating v4 gives byte-identical files (checked by hash).

Baseline gradient boosting on the rich features, trained on 2015 to 2017, scored on 2018 to 2019:

| | Accuracy | Macro-F1 |
|---|---|---|
| Model | 0.470 | 0.420 |
| Shuffled-label control | 0.302 | 0.102 |
| Majority-class baseline | 0.300 | - |

By test year: 2018 alone accuracy 0.404, macro-F1 0.355; 2019 alone accuracy 0.536, macro-F1 0.479.

The regenerated v3 (trained 2017-18, tested 2019) scores accuracy 0.472 and macro-F1 0.410. With about 1,000 test rows the noise on accuracy is about 1.6 points, so **v4 is not shown to be more or less learnable than v3**. The two are not directly comparable, because the train and test years and the anomaly method differ.

## 4. Limitations (in addition to v3's)

- Fisheries NPA is lower than the other sectors in this data (about 11% vs 13-20%) because it is more often insured (40% vs 14% for Dairy) and has no yield term in the score. That is an artefact of the assumptions here, not a statement about real fisheries lending.
- Species (prawn or shrimp) is not a column and has no effect on any value.
- The 45/25/20/10 sector mix and all sector-level effects are synthetic choices, not PNB or public figures.
- LOYO anomalies use information from other years, including later ones (section 1b), so v4 is not a strictly chronological forecasting setup.
- Test accuracy varies a lot by test year (0.404 for 2018, 0.536 for 2019), so report both years together and do not quote one year alone.
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
