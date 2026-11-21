# Loan portfolio column validation (Issue #24)

## Objective
Check every loan-portfolio file against the field layout in
`financial_data_requirements/DATA_DICTIONARY.md` before it is used, so problems are found
consistently and early. The validator never changes, imputes or corrects data.

## Files
| File | Purpose |
|---|---|
| `loan_portfolio_schema_v1.json` | Schema: columns, types, formats, allowed values, ranges, asset-class bands |
| `validate_loan_portfolio.py` | Validator and command-line entry point |
| `tests/` | 27 unit tests (synthetic data only) |

## Commands (run from the repository root)
```bash
python3 financial_data_validation/validate_loan_portfolio.py \
  --input <loan portfolio csv> \
  --pincode-reference address_expansion/pincode_reference_v1.csv \
  --outdir data/private/validation

python3 -m pytest financial_data_validation/tests
```
`--pincode-reference` is optional; without it the pincode-district checks are not run.

**Exit codes:** `0` no error-severity check failed (warnings allowed), `1` at least one error-severity
check failed, `2` input could not be read.

## Report
`validation_report.csv`, one row per check: `check_id, column, severity, status, failing_rows,
total_rows, first_failing_rows, detail`. It contains counts and data-row numbers (from 1, header
excluded) only, never cell values, so it holds no personal data. Status is `PASS`, `FAIL`
(error-severity check failed), `WARN` (warning-severity check failed), `INFO` or `SKIPPED`
(a needed column is absent).

## Checks
| Group | Checks | Severity |
|---|---|---|
| Structure | required columns present; unexpected columns | error; warning |
| Per column | missing values (required columns); valid date; valid number; whole number; minimum / maximum; allowed values (facility type, scheme code, agri sector, asset class); pincode format | error |
| Per column | ID formats (branch ID, customer ID, account number) | warning (observed patterns, unconfirmed) |
| Cross-field | asset class vs days-past-due band; overdue <= outstanding; overdue > 0 iff days past due > 0; open date not after reporting date | error |
| Cross-field | outstanding <= sanction (limits may legitimately exceed it) | warning |
| Keys | account number unique; duplicate rows; branch ID to one branch name; circle to one zone | error; warning; warning; warning |
| Information | number of distinct reporting dates | info |
| Pincode (optional) | pincode found in reference; pincode in Andhra Pradesh (project scope); pincode ambiguous across districts or states | error; error; warning |

Missing values in optional columns (customer name, address) are reported as `INFO`, not failures.

## Assumptions
- The asset-class bands (Standard 0, SMA 0 1-30, SMA 1 31-60, SMA 2 61-90, NPA over 90 days) are the
  commonly published RBI buckets and match the 25-row sample; they live in the schema file and must be
  confirmed by PNB. The RBI crop-season rules for agricultural NPA were not verified here and are not
  modelled.
- Allowed values for facility type, scheme code and agri sector are those observed in the sample; PNB must
  supply the full lists. Until then a new valid code is reported as a failure, which is intentional.
- Observed ID formats are warnings because the real extract may differ (for example account numbers may
  be longer than the 11 digits in the sample).
- Dates are `YYYY-MM-DD`; currency and rate units are not checked (not yet defined by PNB).

## Results on the temporary sample (25 rows, Andhra Pradesh only)
57 checks: 56 pass, 1 informational (single reporting date), 0 warnings, 0 failures. Exit code 0.
Run on the earlier out-of-state version of the sample, the validator fails `pincode_in_project_state`
for all 25 rows and passes everything else.

## Limitations
- Allowed-value lists and bands come from a 25-row sample and are proposals.
- Pincode checks only test the reference source; they do not prove the pincode belongs to the borrower.
- The checks cover structure and consistency, not whether the values are true.
