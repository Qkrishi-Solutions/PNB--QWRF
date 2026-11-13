# Issue #25 verification record

Executed on 2026-11-13 against the user-supplied Issue #23 pincode reference, project district data, and temporary 25-row sample (used only to avoid identifier/name collisions).

| Check | Result |
|---|---|
| Generator unit tests | 8 passed |
| Synthetic records | 5,000 rows; 20 columns |
| District allocation | 13/13 districts; exact configured allocation |
| Cross-field checks X1-X9 | 0 violations |
| Distribution tolerance T5 | Largest deviation 0.31 percentage points |
| Numeric ranges | 0 out-of-range values |
| Pincodes | All generated pincodes have `mapped` status in supplied reference |
| Pincode diversity | 1,175 distinct pincodes |
| Collision check | 0 collisions against supplied sample identifiers/names |
| Reproducibility | Regenerated file SHA-256 exactly matches packaged CSV |

Packaged CSV SHA-256: `18912212d3a6c0716867a3cf282f10dd5393a8976243404fde8d2657abd39dd0`.

**Not verified in this package review:** the repository's `financial_data_validation/validate_loan_portfolio.py` and `address_expansion/expand_addresses.py` were not supplied as executable files in this handoff. Run those repository scripts after copying the package into the repository. The 20-column header was preserved for validator compatibility; synthetic labelling is in the filename, companion label file, README and assumptions document.
