# Synthetic Loan Portfolio: Assumptions and Results (v1)

**SYNTHETIC. Not PNB data.** A development test fixture for issue #25. Rules: `SYNTHETIC_DATA_RULESET.md`. All parameters: `synthetic_config_v1.json`.

## How to reproduce

```bash
# from the repository root
python synthetic_portfolio/generate_synthetic_portfolio.py --avoid-from Financial_data_PNB.csv
python synthetic_portfolio/verify_synthetic_portfolio.py --avoid-from Financial_data_PNB.csv
python financial_data_validation/validate_loan_portfolio.py --input data/private/synthetic/SYNTHETIC_not_PNB_data_portfolio_v1.csv --pincode-reference address_expansion/pincode_reference_v1.csv
python -m pytest synthetic_portfolio/tests
```

`--avoid-from` is optional. It stops the generator reusing any account number, customer ID or name found in the private 25-row sample. The generated CSV goes to `data/private/synthetic/` (untracked). The tracked `synthetic_generation_report_v1.csv` holds counts and hashes only.

## Assumptions

| # | Assumption | Basis |
|---|---|---|
| A1 | 5,000 accounts in total | Large enough to test volume and every asset class; arbitrary otherwise. |
| A2 | Accounts per district are proportional to 2011 Census district population (largest remainder) | Population is a convenient proxy. It is not a measure of agricultural lending. 2011 boundaries pre-date the 2022 re-organisation; the 13 project districts are the older ones. |
| A3 | Only the district account count varies; every other distribution is identical in each district | Issue #25 brief. |
| A4 | Product mix (17 scheme/facility/sector combinations) copied from the 25-row sample's facility and sector shares | Scheme shares differ from the sample. Indicative only. |
| A5 | Asset class mix 36 / 20 / 16 / 12 / 16 (Standard / SMA 0 / SMA 1 / SMA 2 / NPA) | Stress-heavy on purpose so every class is testable. **Not** PNB's asset quality. |
| A6 | DPD bands Standard 0, SMA 0 1–30, SMA 1 31–60, SMA 2 61–90, NPA 91–365 | RBI-style, to be confirmed by PNB. Crop-season NPA rules are not modelled. |
| A7 | Rate 7.22–12.59 %, outstanding 20.3–95 % of sanction, overdue share per class, sanction amounts ₹1 lakh–₹25 lakh | Ranges seen in the 25-row sample. Indicative only. |
| A8 | 5 % of customers hold two accounts, in the same district, branch, address and pincode | Default O3. |
| A9 | Customer names are generated from synthetic Andhra Pradesh given-name and surname lists | Default O1. |
| A10 | Branches: about 60 accounts each, placeholder names and IDs | Not PNB's branch network. |
| A11 | Circle: placeholder grouping of the 13 districts into 3 circles; zone is Hyderabad | As agreed for the sample. |

## Method refinement against the ruleset

Product combination, asset class and sanction amount are **apportioned exactly** within each district (largest remainder) and then shuffled, rather than drawn independently at random. This keeps the per-district shares within 0.3 percentage points of the config instead of the several points that random draws would give in a district of about 240 rows. The three shuffles are independent, so the attributes are not correlated with each other.

## Results (seed 42)

| Check | Result |
|---|---|
| Rows | 5,000 in 13 districts (237 to 522 per district), 85 branches, 4,761 customers |
| Reproducibility (T2) | Two runs give identical SHA-256 (`18912212d3a6c0716867a3cf282f10dd5393a8976243404fde8d2657abd39dd0`) |
| Validator (T3) | Not run in this handoff review because the repository column-validator script was not supplied as an executable file. See `synthetic_portfolio/TEST_RESULTS.md`. |
| Cross-field rules X1–X9 (T4) | 0 violations on 5,000 rows |
| Equal distributions across districts (T5) | Largest deviation 0.31 percentage points (product, asset class, sanction amount); numeric ranges pass |
| Address round trip (T6) | Not run in this handoff review because the repository address-expansion script was not supplied as an executable file. Pincode membership was checked directly against the supplied Issue #23 reference; see `synthetic_portfolio/TEST_RESULTS.md`. |
| No pincode other than `mapped` (T8) | Pass |
| No overlap with the real sample's identifiers or names (T7) | 0 collisions (checked against the supplied 25-row sample) |
| Report holds no cell values (T9) | By construction: counts and hashes only |
| Unit tests | 8 passed in this package review |

The T6 round trip is **circular**: pincodes were drawn from the same reference the mapper uses. It tests plumbing, not mapping accuracy.

## Limitations

- Identical distributions in all districts mean this data **cannot** show district differences in repayment. It tests plumbing only.
- One reporting date. No disbursement dates, repayment history, crop, insurance or cyclone data, so those analyses cannot be tested here.
- Population is only a proxy for lending volume.
- Rate, amount and ratio ranges come from 25 sample rows.
- Any association between this data and weather is accidental and means nothing (rule P8).

## Labelling and schema compatibility

The generated filename starts with `SYNTHETIC_not_PNB_data_`, and a companion `SYNTHETIC_DATA_LABEL.txt` explicitly states **SYNTHETIC - not PNB data**. The CSV's first line must remain the exact 20-column field header for the existing column validator; inserting a comment/metadata line into the CSV would break that schema unless the validator is changed. This is an explicit compromise against the issue's literal "filename and header" wording.

## Source and version notes

- District population weights: 2011 Census populations for the 13 pre-2022 Andhra Pradesh districts, transcribed in `synthetic_config_v1.json`; verify the official Census source before external publication.
- Pincode reference: `address_expansion/pincode_reference_v1.csv`; the generator uses only `mapping_status=mapped`. The original provenance of this project's reference must be checked in the address-expansion deliverable.
- Generator version: v1.1 (revised 2026-11-13), NumPy `default_rng`, seed 42. Exact reproducibility also requires pinned NumPy/Pandas versions and unchanged reference files.
- The 25-row sample is used only to avoid identifier/name collisions and to justify documented ranges; do not redistribute it in this ZIP.

## Review status (2026-11-13)

The packaged CSV was regenerated with the supplied Issue #23 pincode reference and the 13 project district codes. The regenerated SHA-256 exactly matches the packaged CSV; all 8 unit tests and the synthetic-specific verifier passed. **The repository column-validator and address-expansion scripts still need to be run inside the repository.** The CSV preserves the 20 schema column names; the explicit synthetic label is carried by the filename, companion label file and documentation to avoid breaking schema validation.


## Supplied Issue #23 reference snapshot

This package includes the uploaded `pincode_reference_v1.csv` used by the generator at `address_expansion/pincode_reference_v1.csv`. Its reported build has 19,300 unique pincodes, 1,207 mapped Andhra Pradesh pincodes, 18 ambiguous-district pincodes, 46 ambiguous-state pincodes and coverage of all 13 project districts. The raw India Post directory is not included. Confirm the upstream data.gov.in licence before external redistribution, as noted in the Issue #23 source documentation.
