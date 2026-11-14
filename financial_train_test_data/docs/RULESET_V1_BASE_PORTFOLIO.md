# Synthetic Loan Portfolio: Generation Ruleset (v1)

> **Label:** SYNTHETIC. This is not PNB data. Every output of this ruleset is a development test fixture.
> **Issue:** #25. **Status:** DRAFT. Values in `synthetic_config_v1.json` are assumptions until PNB confirms definitions.
> **Layout source:** `financial_data_requirements/DATA_DICTIONARY.md` (20 fields) and `financial_data_validation/loan_portfolio_schema_v1.json`.

## 1. Purpose and non-purpose

**Purpose:** test the pipeline (address expansion, pincode→district mapping, column validation, district aggregation and the district join) at realistic volume (5,000 rows) before PNB data access is approved.

**Not a purpose:** estimating real exposure, real delinquency or real risk. No result computed on this data may be quoted as a finding about PNB's portfolio, and no model may be trained on it for use on real data.

## 2. Governing principles

| # | Rule |
|---|---|
| P1 | **Only the number of accounts varies by district.** Every other distribution is identical in all 13 districts (issue #25 brief). |
| P2 | **Reproducible.** One seed (42) in the config. Same config + same code = byte-identical output. |
| P3 | **Config-driven.** Every number lives in `synthetic_config_v1.json`. No constants hidden in code. |
| P4 | **No real data.** No real names, account numbers, customer IDs or addresses. Nothing derived from `Financial_data_PNB.csv` row values beyond the marginal ranges and shares recorded in the config. |
| P5 | **Coherent rows.** Each row must pass the cross-field rules in section 6, not just the per-column rules. |
| P6 | **Labelled everywhere.** Filename, README header and the first line of any report say SYNTHETIC. |
| P7 | **Must pass the validator.** The generated file must exit 0 on `validate_loan_portfolio.py` with no errors (warnings are reviewed, not ignored). |
| P8 | **Do not engineer the answer.** Nothing is injected to make weather and loan stress correlate. Any association found must be absent or accidental, and is not evidence of anything. |

## 3. Volume and allocation

1. `total_accounts` = 5,000 (config).
2. Allocate accounts to the 13 districts in proportion to **Census 2011 district population** (config `district_population_2011`) using the **largest-remainder method**, so counts are integers and sum to exactly 5,000.
3. Check: sum of district counts = 5,000; every district ≥ 1; the order of districts by count matches the order by population.
4. Document the population source and the limitation that population is a proxy for lending volume, not a measure of it. Census 2011 districts are pre-bifurcation boundaries mapped to the project's 13 districts; state any mapping done.

## 4. Geography rules (per account)

| Field | Rule |
|---|---|
| `zone` | Constant `Hyderabad` (config `zone`). |
| `circle` | Looked up from the district (`circle_of_district`): Vijayawada, Visakhapatnam or Tirupati. Placeholder grouping, not PNB's structure. |
| `pincode` | Drawn at random from pincodes with status `mapped` to that district in `address_expansion/pincode_reference_v1.csv`. Never from ambiguous, out-of-scope or unmapped pincodes. |
| `borrowers address` | Four comma-separated parts in the sample layout: house, area, locality, city + pincode. City comes from `city_of_district`. House/area/locality are generic placeholder strings (e.g. `H.No 12-3`, `Gandhi Nagar`), never real addresses. |
| `branch ID`, `branch name` | Branches are created inside a district, about `accounts_per_branch` (60) accounts each (minimum 1 branch per district). A branch belongs to exactly one district. Branch ID matches `[A-Z]{2}\d{3}`; the name is a placeholder such as `<City> Main`, `<City> Branch 2`. |

**Round-trip rule:** running the generated file through `expand_addresses.py` must map every row to the district it was generated for (100% `mapped`, zero city conflicts). Because the pincode is drawn from the same reference, this check is circular. It tests plumbing, not mapping accuracy, and must be described that way.

## 5. Per-column rules

Common distributions are **the same in every district** (P1).

| Field | Rule |
|---|---|
| `reporting_date` | Constant `2026-10-31` (config). Format `YYYY-MM-DD`. |
| `customer ID` | `[A-Z]{4}\d{5}`, synthetic, unique per customer. Most customers hold one account; a small share (config, default ≤10%) hold two, in the same district. |
| `customer name` | Generated from synthetic Andhra Pradesh given-name and surname lists (see open decision O1). Never a real person. May be left blank for a documented share to test missing-value handling (default 0%). |
| `account number` | 11 digits, unique across the file, random, not sequential, no check-digit meaning. |
| `account open date` | Uniform within `open_date_range` (2020-06-27 to 2025-03-17). Never after `reporting_date`. |
| product triple (`scheme code`, `facility type`, `agri sector`) | Drawn **as a unit** from `product_mix_percent` (17 combinations, weights sum to 100). Never drawn independently per column. Only combinations present in the table may appear. |
| `sanction amount` | Drawn from `sanction_amount_weights` (₹1 lakh to ₹25 lakh bands, discrete values). Always > 0. |
| `rate of interest` | Uniform within `interest_rate_range` (7.22 to 12.59), rounded to 2 decimals. |
| `asset class` | Drawn from `asset_class_percent` (Standard 36, SMA 0 20, SMA 1 16, SMA 2 12, NPA 16). This mix is **deliberately stress-heavy**, copied from the temporary sample so that every class has enough rows to test. It is **not** a claim about PNB's real asset quality. |
| `days_past_due` | Drawn uniformly within the band of the account's asset class (`dpd_range`): Standard 0, SMA 0 1–30, SMA 1 31–60, SMA 2 61–90, NPA 91–365. |
| `balance outstanding` | `sanction amount × u`, with `u` uniform in `outstanding_to_sanction_range` (0.203 to 0.95). Rounded to whole rupees. |
| `overdue amount` | Standard: exactly 0. Other classes: `balance outstanding × v`, with `v` uniform within the class range `overdue_to_outstanding_range`. Rounded to whole rupees. |

## 6. Cross-field rules (every row must satisfy)

| ID | Rule | Validator severity |
|---|---|---|
| X1 | `balance outstanding` ≤ `sanction amount` | error |
| X2 | `overdue amount` ≤ `balance outstanding` | error |
| X3 | `asset class` agrees with `days_past_due` band (RBI-style, to be confirmed by PNB) | error |
| X4 | `days_past_due` = 0 ⇔ `overdue amount` = 0 | error |
| X5 | `account open date` ≤ `reporting_date` | error |
| X6 | Product triple is one of the 17 allowed combinations | error |
| X7 | `pincode` maps to the same district as `city` in the address and as the branch's district | error |
| X8 | Branch belongs to exactly one district; circle is consistent with that district | error |
| X9 | `account number` unique; `customer ID` format valid | error / warning |

## 7. Order of generation

1. Read config; set the seed (a single `numpy.random.default_rng(42)`; all draws come from it in a fixed order).
2. Allocate accounts to districts (section 3).
3. Create branches per district.
4. For each account, in district order then account order: branch → pincode → address → customer → account number → open date → product triple → sanction → rate → asset class → DPD → outstanding → overdue.
5. Sort by district, branch, account number for stable output.
6. Write the CSV, the generation report and the checksum.

Fixed order matters: changing the order of draws changes every later value, even with the same seed.

## 8. Outputs

| File | Where | Tracked in Git? |
|---|---|---|
| `synthetic_portfolio_v1.csv` (SYNTHETIC) | `data/private/synthetic/` | No, regenerated on demand |
| `synthetic_generation_report_v1.csv` (counts by district, product, asset class; config hash; seed; row count; SHA-256 of the output) | `synthetic_portfolio/` | Yes (aggregate counts only) |
| `SYNTHETIC_DATA_ASSUMPTIONS.md` | `synthetic_portfolio/` | Yes |

The CSV is synthetic and could be tracked, but is regenerated from the seed so there is no need, and keeping it out of Git avoids it ever being mistaken for portfolio data.

## 9. Acceptance tests (the generator is not done until all pass)

| # | Test |
|---|---|
| T1 | Row count = 5,000; district counts sum to 5,000; matches the largest-remainder allocation. |
| T2 | Same seed twice = identical file (SHA-256 equal). A different seed = different file. |
| T3 | Validator exit code 0; zero `error` findings. |
| T4 | All cross-field rules X1–X9 hold for every row (counted, not sampled). |
| T5 | Within tolerance, the product mix, asset-class mix, sanction-amount mix and the ranges for rate, DPD and outstanding/sanction are the same in every district. Report the largest deviation. Small districts will show sampling noise (about 100 rows); state the tolerance used. |
| T6 | Round trip through `expand_addresses.py`: all rows `mapped`, zero city conflicts. |
| T7 | No cell value equals any value in `Financial_data_PNB.csv` for account number, customer ID or customer name. |
| T8 | Output contains no pincode with status other than `mapped`. |
| T9 | The generation report holds counts and hashes only, no customer-level cell values. |

## 10. Prohibited

- Using real PNB values, or the 25-row sample's identifiers, names or addresses.
- Presenting the output as PNB data, or its asset-class mix as PNB's asset quality.
- Tuning any distribution to produce a desired weather/loan association (P8).
- Varying anything other than account counts by district without updating this ruleset and the issue.
- Adding fields that are not among the 20 in the data dictionary.
- Committing the generated CSV to a public location without a SYNTHETIC label.

## 11. Known limitations to state in `SYNTHETIC_DATA_ASSUMPTIONS.md`

- District allocation by 2011 population is a proxy; real lending is driven by agricultural activity, branch network and credit demand.
- Identical distributions in all districts mean the data **cannot** show district differences in repayment. It can only test that the plumbing works.
- No time dimension: one reporting date, no repayment history, no disbursement dates, no crop, no insurance. The planned cyclone, January-issuance and insurance analyses cannot be tested on it.
- The DPD→asset-class bands are RBI-style and unconfirmed. Crop-season NPA rules are not modelled.
- Rate, amount and ratio ranges come from 25 sample rows. They are indicative only.

## 12. Open decisions (answer before building the generator)

| # | Decision | Default if no answer |
|---|---|---|
| O1 | Replace the sample's customer names with synthetic Andhra Pradesh names? | Yes: synthetic names. |
| O2 | Keep the stress-heavy asset-class mix (36/20/16/12/16), or use a more typical mix (e.g. Standard ~85%) for realism? | Keep as is; label it clearly. |
| O3 | Share of customers with two accounts. | 5%. |
| O4 | Output location: `data/private/synthetic/` (untracked) or tracked in Git? | Untracked. |
| O5 | Tolerance for T5 per-district equality. | ±10 percentage points for districts under 150 rows; ±5 above. |
