# Address expansion and pincode-to-district mapping (Issue #23)

## Objective
Expand the free-text borrower address of the loan portfolio into structured parts and map each
pincode to a project `district_code` / `state_code`, so loan records can be joined to the
district-keyed weather and yield data (13 Andhra Pradesh districts).

## Files
| File | Purpose |
|---|---|
| `build_pincode_reference.py` | Builds `pincode_reference_v1.csv` from the India Post pincode directory |
| `expand_addresses.py` | Parses addresses, joins the pincode reference, writes outputs |
| `pincode_reference_v1.csv` | One row per pincode (19,300 rows): state, source district(s), project district code, mapping status |
| `pincode_reference_build_report_v1.csv` | Build counts and the SHA-256 of the raw input |
| `tests/` | Unit tests (synthetic data only) |

## Commands (run from the repository root)
```bash
# 1. Rebuild the reference (only needed when the source directory changes)
python3 address_expansion/build_pincode_reference.py \
  --raw <path to pincode.csv> \
  --districts weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv

# 2. Expand addresses (input and outputs stay in a private, untracked folder)
python3 address_expansion/expand_addresses.py \
  --input <path to loan portfolio csv> \
  --outdir data/private/address_expansion

# 3. Tests
python3 -m pytest address_expansion/tests
```

## Reference data source
- Directory: India Post "All India Pincode Directory" (data.gov.in), used via the community mirror
  `github.com/dropdevrahul/pincodes-india`, file `pincode.csv`.
- Mirror commit `d817d3e652e8167a70507350d9a8e4acda9113ef` (committed 2024-06-30), retrieved 2026-11-13.
- Raw file SHA-256: `84af12fa29adddedfa9addcf46546000d89d2e2899b2993dc88050fafc8861e2`
  (157,126 post-office rows, 19,300 unique pincodes). The raw file is not committed.
- Licence: the mirror repository is MIT. The underlying data is government open data; sources
  differ on the exact licence name (Government Open Data Licence - India vs Open Government
  Licence - India), so **confirm the licence on data.gov.in before external distribution**.

## Method
1. **Reference build:** collapse post offices to one row per pincode. Normalise district names
   (upper case, punctuation removed). Map four India Post spellings to project spellings:
   `ANANTAPUR` to `ANANTAPURAMU`, `SPSR NELLORE` to `SPR NELLORE`, `VISAKHAPATANAM` to
   `VISAKHAPATNAM`, `Y.S.R.` to `YSR`. Project district codes are read from the project CSV
   (not hardcoded). Project `state_code` = 1 for Andhra Pradesh, as in `crop_long_v1.csv`;
   other states have no project code.
2. **Address parsing:** split on commas. Four parts are read as house/plot, area, locality, city
   (layout observed in the sample). Any other part count is labelled `unexpected_part_count`
   and left unparsed. A 6-digit number inside the text is extracted as `address_pincode` only
   if there is exactly one distinct value.
3. **Pincode join:** many-to-one on the 6-digit `pincode` column; the row count is asserted
   unchanged.
4. **Cross-checks:** address city vs the pincode's district (`consistent`, `conflict`,
   `not_comparable`); embedded address pincode vs the pincode column.

## Status definitions (`mapping_status`)
| Status | Meaning |
|---|---|
| `mapped` | Andhra Pradesh pincode with exactly one district; `district_code` filled |
| `ambiguous_district` | Pincode spans more than one district; candidates kept in `project_district_codes`, `district_code` blank |
| `ambiguous_state` | Pincode spans more than one state; left unmapped |
| `outside_project_scope` | Single non-Andhra Pradesh state; not part of the 13 project districts |
| `not_in_reference` | Valid-format pincode absent from the directory |
| `invalid_pincode` | Missing or not a 6-digit number starting 1-9 |

## Results on the reference build (actual)
| Item | Count |
|---|---|
| Raw rows / unique pincodes | 157,126 / 19,300 |
| Invalid-format pincodes | 0 |
| Andhra Pradesh `mapped` | 1,207 |
| Andhra Pradesh `ambiguous_district` | 18 |
| Pincodes shared with another state (`ambiguous_state`) | 46 (18 involve Andhra Pradesh: 17 Telangana, 1 Puducherry) |
| Project districts covered | 13 of 13 |

## Results on the temporary sample file (25 rows)
All 25 addresses parsed; all 25 pincodes found in the reference; all 25 are
`outside_project_scope` (pincodes outside Andhra Pradesh), so 0% mapped to a project district.
City check: 18 consistent, 7 not comparable, 0 conflict. The sample has no pincode inside the
address text. District mapping for Andhra Pradesh is therefore verified only on synthetic data.

## Assumptions
- The four-part address layout seen in the sample holds for the real extract.
- The India Post district for a pincode is the district of the borrower or branch (PNB to confirm
  what `pincode` refers to).
- The directory uses the same 13 pre-2022 Andhra Pradesh districts as the project data; this was
  confirmed for the mirror used. If PNB reports the current 26 districts, a crosswalk is needed.

## Limitations
- The mirror is dated June 2024 and is not the live source; pincode-district assignments change.
- 36 Andhra Pradesh-linked pincodes cannot be mapped uniquely (18 ambiguous by district, 18 by state).
- The city check can only flag a conflict when the city is itself a district name; localities are
  `not_comparable`.
- A district match shows where a loan is recorded, not where the farm or weather exposure is.

## Privacy
`address_expanded.csv` contains address text and account numbers. It is written to
`data/private/` and must not be committed. `address_expansion_report.csv` holds counts only.
