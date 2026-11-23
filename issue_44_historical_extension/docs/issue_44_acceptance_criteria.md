# Issue #44 — Acceptance Criteria Audit

**Project:** PNB QWeatherRiskF (Qkrishi-Solutions/PNB--QWRF)
**Assessment date:** 22 November 2026
**Assessment type:** Evidence-based historical data availability audit
**Overall status:** PARTIAL — historical extension not complete

## 1. Acceptance criteria status

| # | Acceptance criterion | Status | Evidence and explanation | Remaining requirement |
|---|---|---|---|---|
| 1 | Preserve the existing 2015–2019 baseline and existing Issue #3/#4 outputs | PARTIAL | Baseline hash was recorded and remained unchanged in the final investigation evidence. Existing pipeline outputs and validation artifacts were inspected. | Maintain reproducible baseline hash verification and preserve the original files during future ingestion. |
| 2 | Verify existing weather and yield baseline outputs | PASS | Existing baseline outputs and validation artifacts were inspected. Weather forward-chaining report records 65 district-years and 78 indicators, with 12 checks passing. Yield forward-chaining report records 1,430 rows, 286 district–crop combinations and 12 checks passing. | No additional baseline requirement identified within the inspected artifacts. |
| 3 | Preserve missing values without silent imputation or zero-filling | PASS | Existing validation evidence records missing observations as missing. No new historical values were imputed or zero-filled during this investigation. | Apply the same rule to any future ingestion and document missingness explicitly. |
| 4 | Confirm forward-chaining scorable years | PASS | Existing yield anomaly evidence shows calculated observations in 2017 (164), 2018 (162) and 2019 (163). 2015 and 2016 do not have calculated anomalies. | Reassess the forward-chaining windows only if a validated extended dataset is introduced. |
| 5 | Investigate agricultural source availability, including ICRISAT and DES APY | PARTIAL | ICRISAT access was unsuccessful due to a transport-level failure. DES APY was reachable, but controlled retrieval of complete target records was inconclusive. Earlier investigation notes report selected 2020-21 agricultural area records; supporting phase-level audit files are not included in this repository, so the reported count is not reproducible here. | Obtain reproducible, data-bearing source records with definitions, units, years and provenance. |
| 6 | Investigate weather source availability, including IMD and ERA5/ERA5-Land | PARTIAL | IMD 2020 rainfall NetCDF was retrieved and inspected at file/header/metadata level. ERA5 and ERA5-Land documentation was assessed, but sample data were not validated. | Validate actual weather values, decoded time, missingness, spatial coverage, licensing and source compatibility. |
| 7 | Produce agricultural crop x district x year coverage evidence | PARTIAL | The existing baseline matrix covers 13 districts and 22 crops for 2015-2019. Earlier investigation notes report selected 2020-21 area records, but their phase-level audit evidence is not included in this repository and the reported count is not reproducible here. They do not provide complete verified extended coverage. | Produce a complete evidence-backed extension matrix with verified crop identity, district identity, year, season, unit and source provenance. |
| 8 | Produce weather coverage and feature compatibility evidence | PARTIAL | The IMD 2020 artifact establishes partial gridded-source retrieval. The baseline weather feature inventory contains 78 features. The detailed compatibility assessment table and the previously reported 281-row breakdown are not included in this repository contribution and are not reproducible here. | Validate additional weather source contents and demonstrate compatibility with baseline feature definitions and district geography. |
| 9 | Establish historical district boundaries and district/crop crosswalks | FAIL | The 13 historical district mappings remain unresolved. The six crop mapping entries also contain unresolved identity or product-form questions. No authoritative 2015–2019 geometry and evidence-backed crosswalk was established. | Obtain authoritative historical boundaries, verify district codes and identities, resolve crop mappings and validate spatial equivalence. |
| 10 | Ingest additional data through existing pipelines and rerun Issues #3/#4/#5 | NOT VERIFIED / BLOCKED | No additional agricultural or district-level weather records passed ingestion-readiness checks. No new data were ingested, and no extended-data pipeline reruns were performed. | Complete source validation, mapping, schema alignment and approved ingestion before rerunning the pipelines. |

## 2. Additional agricultural evidence

Earlier investigation notes report:

- 728 selected 2020-21 agricultural area records across earlier investigation phases. The supporting phase-level audit files are not included in this repository; the count and phase breakdown are not reproducible here.
- 224 selected yield/production sample records, with only 6 independently visually rechecked in the later audit. The supporting phase-level audit files are not included in this repository; the count and audit breakdown are not reproducible here.
- 991 candidate district-wise table pages scanned across 2020-21 to 2024-25.
- 0 newly verified per-value records for 2021-2025.
- 0 additional agricultural records currently cleared for ingestion.

The reported 728 selected area records are historical investigation notes, not independently reproducible repository evidence. They must not be treated as complete district-by-crop coverage. State totals remain separate from district observations.

The reported 224 yield/production samples are historical investigation notes, not independently reproducible repository evidence. They must not be described as fully independently verified. Earlier notes recorded an audit verdict of ACCEPT WITH CORRECTIONS, but the supporting audit evidence is not included in this repository.

## 3. Additional weather evidence

The IMD 2020 NetCDF file was retrieved and inspected at the metadata level.

Recorded properties:

- File size: 25,501,500 bytes.
- SHA-256: `2E567F0F71F3A1936BA11D61A0214FC3927C7B969EE5669D14959AA0B6B026DC`.
- Grid resolution: 0.25° × 0.25°.
- Variable: `RAINFALL`.
- Units: mm.
- Time dimension: 366 records.

Full data-array validation and district aggregation were not performed.

ERA5 and ERA5-Land sample data were not validated.

## 4. Interpretation of status

- **PASS:** Evidence supports the criterion within the stated scope.
- **PARTIAL:** Some evidence exists, but the complete criterion is not established.
- **FAIL:** A required prerequisite has been assessed and is not satisfied.
- **NOT VERIFIED / BLOCKED:** The required implementation or validation has not been performed because prerequisite evidence is missing.

A PARTIAL or NOT VERIFIED status must not be presented as successful historical extension.

## 5. Overall conclusion

**Overall acceptance status: PARTIAL.**

The investigation provides evidence about the existing baseline, selected agricultural records, weather source retrieval, feature compatibility and unresolved mapping requirements.

However, the central objective of Issue #44 — verified additional agricultural and weather coverage aligned to the historical district geography — has not been achieved.

No additional records were ingested, and Issues #3, #4 and #5 were not rerun on an extended dataset.

The documentation contribution can be reviewed independently as an investigation milestone, if the maintainers accept incremental documentation PRs.

**Issue #44 should remain open until the historical-extension requirements are satisfied and validated.**
