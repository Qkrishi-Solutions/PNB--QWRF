# Issue #25 verification record

Executed on 2026-11-13.

## 1. Synthetic dataset checks

| Check | Result |
|---|---|
| Synthetic records generated | 5,000 |
| Dataset columns | 20 |
| Project districts represented | 13/13 |
| Pincode reference rows | 19,300 |
| Independent standard-library audit | 15 passed, 0 failed |
| District account distribution | 5,000 accounts across 13 districts |

## 2. Independent audit

Executed using:

`synthetic_portfolio/audit_synthetic_portfolio_stdlib.py`

Results:
- 5,000 records checked.
- 15 checks passed.
- 0 checks failed.
- All 13 project districts represented.
- Pincode mapping checked against the supplied reference.

Generated reports:
- `data/private/stdlib_audit/stdlib_audit_report.csv`
- `data/private/stdlib_audit/stdlib_audit_summary.json`

These reports are private validation outputs and are not committed.

## 3. Official integration checks

| Check | Status |
|---|---|
| Official financial data validator | Blocked |
| Address expansion script | Blocked |
| Repository pytest suite | Blocked |

The scripts were present and execution was attempted. However, Python Pandas could not load its compiled `_cyutility` extension because Windows Application Control blocked the file.

Therefore, the official validator, address expansion and repository test suite have not been successfully verified in this environment.

They must be executed in an approved environment where Pandas can load successfully.

## 4. Synthetic data labelling

The generated CSV uses the original 20-column data dictionary header to preserve validator compatibility.

Synthetic identification is currently provided through:
- The filename.
- The companion synthetic data label file.
- The README.
- The assumptions document.

The literal synthetic label in the CSV header remains to be resolved against the validator's header requirements.

## 5. Verification limitations

The independent standard-library audit is supplementary. It does not replace the official financial data validator, address expansion integration or repository test suite.

The dataset is synthetic and must never be represented as actual PNB borrower data or PNB results.
