# SYNTHETIC - not PNB data

Issue #25 development/test fixture. This is **not real PNB data, not PNB portfolio performance, and not suitable for credit decisions**.

## Package contents

- `generate_synthetic_portfolio.py` — deterministic generator (v1.1).
- `synthetic_config_v1.json` — seed and documented generation parameters.
- `verify_synthetic_portfolio.py` — synthetic-specific acceptance checks.
- `synthetic_generation_report_v1.csv` — aggregate counts and hashes only.
- `SYNTHETIC_DATA_ASSUMPTIONS.md` — at `financial_data_requirements/`, as requested by Issue #25.
- `../address_expansion/pincode_reference_v1.csv` — exact Issue #23 reference snapshot used for this package.
- `project_district_codes_v1.csv` — 13 project district-code/name lookup rows derived from the supplied crop data.
- `data/private/synthetic/SYNTHETIC_not_PNB_data_portfolio_v1.csv` — 5,000-row synthetic fixture.
- `data/private/synthetic/SYNTHETIC_DATA_LABEL.txt` — explicit synthetic label.
- `TEST_RESULTS.md` — checks actually executed and outstanding integration checks.

## Requirements

Python 3.10+; install from repository root:

```powershell
python -m pip install -r synthetic_portfolio/requirements.txt
```

## Reproduce the packaged dataset

From the repository root, with the supplied reference snapshot and district lookup in place:

```powershell
python synthetic_portfolio/generate_synthetic_portfolio.py `
  --pincode-reference address_expansion/pincode_reference_v1.csv `
  --project-districts synthetic_portfolio/project_district_codes_v1.csv `
  --avoid-from Financial_data_PNB.csv
```

`--avoid-from` is optional and should point only to an authorised private sample. It is used only to prevent generated account numbers, customer IDs and customer names from colliding with values in that sample. Never commit or redistribute the private sample.

Run synthetic-specific checks and tests:

```powershell
python synthetic_portfolio/verify_synthetic_portfolio.py `
  --input data/private/synthetic/SYNTHETIC_not_PNB_data_portfolio_v1.csv `
  --pincode-reference address_expansion/pincode_reference_v1.csv `
  --project-districts synthetic_portfolio/project_district_codes_v1.csv `
  --avoid-from Financial_data_PNB.csv
python -m pytest synthetic_portfolio/tests -q
```

Run the repository's column validator and address expansion separately using the project versions of those scripts. Those scripts were not included in this handoff ZIP, so their execution is not claimed here. Example column-validator command (from repository root):

```powershell
python financial_data_validation/validate_loan_portfolio.py `
  --input data/private/synthetic/SYNTHETIC_not_PNB_data_portfolio_v1.csv `
  --pincode-reference address_expansion/pincode_reference_v1.csv
```

## Schema and synthetic label

The CSV retains the exact 20 field names in the PNB-provided draft data dictionary so the existing schema validator can read it. The filename and companion `SYNTHETIC_DATA_LABEL.txt` carry the explicit **SYNTHETIC - not PNB data** label. A comment row or an extra metadata column is deliberately not inserted into the 20-column CSV because that would alter the schema and may cause the existing validator to reject it. The README and assumptions document also label the fixture clearly.

## Privacy

The ZIP contains synthetic rows and a public/reference pincode lookup only. It does not contain the 25-row private sample. Generated data belongs under `data/private/` and should remain untracked in Git.
