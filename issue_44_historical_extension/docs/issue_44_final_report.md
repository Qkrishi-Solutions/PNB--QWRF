# Issue #44 — Historical Data Availability Investigation Report

**Project:** PNB QWeatherRiskF (Qkrishi-Solutions/PNB--QWRF)
**Investigation date:** 22 November 2026
**Proposed contribution:** Documentation-only investigation and availability assessment
**Target branch:** `dev`
**Working branch:** `issue-44-historical-data-availability`

## 1. Executive summary

Issue #44 aims to extend the historical agricultural and weather data coverage beyond the existing 2015–2019 baseline and 13 historical Andhra Pradesh districts.

The investigation examined agricultural data sources, weather data sources, district and crop mapping requirements, baseline preservation, and compatibility with the existing QWeatherRiskF weather and yield feature structures.

**Overall result: PARTIAL — historical extension is not ready for ingestion.**

The investigation produced useful evidence:

- The existing 2015–2019 baseline covers 13 districts and 22 crop categories.
- Selected 2020–21 agricultural area records were audited, but they do not constitute a complete district × crop extension.
- Additional agricultural observations for 2021–2025 were not independently verified at the individual-value level.
- An IMD 2020 gridded rainfall NetCDF file was retrieved and inspected at the file and metadata level. Its rainfall values were not fully validated.
- District and crop crosswalks remain unresolved.
- The weather compatibility assessment covers all 79 baseline weather features, but no feature has been verified as compatible with an additional source.
- No additional data were ingested, no district-level weather aggregation was performed, and Issues #3, #4 and #5 were not rerun on an extended dataset.

The existing baseline was preserved. No synthetic or fabricated observations were introduced, and missing values were not silently imputed or replaced with zero.

**This report does not claim that Issue #44's historical-extension acceptance criteria are complete.**

## 2. Existing baseline

The existing baseline remains the reference dataset for this investigation.

| Item | Baseline |
|---|---|
| Historical period | 2015–2019 |
| Districts | 13 |
| Crop categories | 22 |
| Agricultural source rows | 1,430 |
| District–crop combinations | 286 |
| Weather district-year observations | 65 |
| Weather indicators | 78 |
| Weather feature inventory | 79, including `rain_valid_area_fraction` |

The original agricultural baseline file was recorded with the following SHA-256:

`4765429E3ACB7CF3EB9E64226BABC9D29D910FB8DEB7D0C42AC48E3BADDF727D`

Existing Issue #3 and Issue #4 outputs and validation artifacts were inspected. The forward-chaining yield analysis has calculated anomalies for 2017, 2018 and 2019. The years 2015 and 2016 do not have calculated forward-chaining anomalies because of the historical training requirements.

Existing baseline validation results must not be interpreted as validation of newly discovered historical data.

## 3. Agricultural data availability investigation

### 3.1 Sources investigated

The investigation examined:

- ICRISAT District Level Database (DLD).
- Directorate of Economics and Statistics (DES), Andhra Pradesh, Area, Production and Yield (APY) portal.
- Available agricultural tables and PDF reports for years beyond the existing baseline.

ICRISAT access was unsuccessful during the investigation because of a transport-level failure. This does not establish that ICRISAT has no relevant data.

The DES APY portal was reachable, and year and district selection options were observed. However, controlled retrieval of a complete, data-bearing result for the required outside-baseline district and crop combinations was not conclusively completed. One observed submission returned “No Data Found.”

### 3.2 Selected 2020–21 agricultural area records

A total of **728 selected 2020–21 agricultural area records** were audited across the investigation phases:

- 140 records from Phase 15D/15D.1.
- 84 records from Phase 15E.2/15E.3.
- 504 records from Phase 15F/15F.1.

These selected area records were visually checked against source-rendered PDF pages, and source-total checks passed in the previous phase audits where performed.

Important limitations:

- These are selected records, not a complete district × crop × season dataset.
- Andhra Pradesh state totals are separate entities and must not be treated as district observations.
- Crop and district crosswalks to the existing baseline are unresolved.
- Their presence does not establish that the records can safely be merged into the existing QWeatherRiskF baseline.

### 3.3 Yield and production sample limitations

The investigation identified 224 selected yield/production sample records.

The audit status must be interpreted carefully:

- The previous audit verdict was “ACCEPT WITH CORRECTIONS.”
- Only 6 records were independently visually rechecked in the later audit.
- 218 records were not independently rechecked in that audit.
- 13 numerical checks passed where testable.
- 5 numerical checks were not testable.

Therefore, the 224 samples must not be described as 224 fully independently verified observations.

### 3.4 Later-year agricultural investigation

A total of 991 candidate district-wise table pages were scanned across 2020–21 to 2024–25.

These are scanned pages, not extracted or verified agricultural observations.

The final evidence assessment identified:

- Newly verified per-value records for 2021–2025: **0**.
- Additional agricultural records safe for ingestion: **0**.

The investigation therefore does not establish a complete extended agricultural coverage matrix.

## 4. Agricultural district and crop mapping

The additional source records cannot be aligned safely to the existing baseline without evidence-backed district and crop mappings.

The mapping assessment identified:

- 13 district mapping entries, all unresolved.
- 6 crop mapping entries, with unresolved crop identity or product-form questions.

Examples of unresolved crop equivalence include:

- Jowar versus sorghum.
- Bajra versus pearl millet.
- Ragi versus finger millet.
- Red gram versus pigeonpea.
- Cotton area versus cotton lint product forms.

Similar names are not sufficient evidence of identical crop definitions, statistical units or product forms.

District name similarity is also insufficient to establish historical boundary equivalence. Current or reorganized district identities must not be mapped automatically to the 13 historical baseline districts.

**Agricultural extension decision: NO-GO for ingestion until source records, definitions, units, seasons and district/crop crosswalks are verified.**

## 5. Weather data availability investigation

### 5.1 IMD 2020 rainfall artifact

An IMD 0.25-degree gridded daily rainfall NetCDF file was retrieved:

`RF25_ind2020_rfp25.nc`

Recorded evidence:

| Property | Observed value |
|---|---|
| Year | 2020 |
| File size | 25,501,500 bytes |
| SHA-256 | `2E567F0F71F3A1936BA11D61A0214FC3927C7B969EE5669D14959AA0B6B026DC` |
| Format | NetCDF classic |
| Grid resolution | 0.25° × 0.25° |
| Grid dimensions | 135 × 129 |
| Variable | `RAINFALL` |
| Units | mm |
| Time dimension | 366 records |
| Missing/fill value | -999.0 |

The file signature, size, header and metadata were inspected.

**Validation limitation:** Full rainfall-array values, decoded time values, missingness distribution and district-level coverage were not validated. The existence of a valid NetCDF header does not establish that the data are ready for ingestion.

No district-level aggregation was performed.

### 5.2 ERA5 and ERA5-Land

ERA5 and ERA5-Land source documentation was investigated.

However:

- No ERA5-Land sample file was downloaded and validated.
- No ERA5 sample file was downloaded and validated.
- Dataset-level availability information is not equivalent to verified project-ready data.

### 5.3 Weather compatibility assessment

The corrected weather compatibility assessment represents all **79 baseline weather features**.

| Assessment status | Feature rows |
|---|---:|
| NOT_TESTED | 241 |
| INCOMPATIBLE | 40 |
| VERIFIED_COMPATIBLE | 0 |
| **Total** | **281** |

A feature was not considered compatible merely because its name or unit appeared similar to a baseline feature.

No additional weather source has passed the required content, unit, temporal-definition, spatial-alignment and schema compatibility checks.

**Weather extension decision: NO-GO for district-level ingestion at this stage.**

## 6. Historical district boundary limitations

The 13 baseline district codes are:

`502, 503, 504, 505, 506, 510, 511, 515, 517, 519, 520, 521, 523`

The investigation did not establish an authoritative historical 2015–2019 boundary geometry and evidence-backed crosswalk suitable for extending the baseline.

The geoBoundaries candidate was not accepted as a verified substitute for historical district boundaries.

This is a critical prerequisite for weather aggregation. Gridded weather data cannot be treated as district-level observations without defensible historical geometry and aggregation rules.

Coastal cells, partial grid cells and the Srikakulam/grid-edge case also require attention in any future aggregation.

## 7. Impact on Issues #3, #4 and #5

No additional dataset passed the required ingestion-readiness checks.

Consequently:

- Issue #3 weather feature engineering was not rerun on an extended dataset.
- Issue #4 yield anomaly analysis was not rerun on an extended dataset.
- Issue #5 crop-specific feature selection was not rerun on an extended dataset.

If compatible data become available later, the team will need to review:

1. Year ranges and validation folds.
2. Forward-chaining training and evaluation windows.
3. District and crop mapping.
4. Feature definitions, units and missingness.
5. Dataset schema and versioning.
6. Validation reports and reproducibility.

Existing baseline outputs remain the reference and must not be silently overwritten.

## 8. Data integrity and handling

The investigation followed these restrictions:

- No real PNB customer or portfolio data were used.
- No synthetic values were represented as historical observations.
- Missing values were not silently imputed or converted to zero.
- State totals were not treated as district observations.
- No district-level weather aggregation was performed without validated boundaries.
- No additional data were ingested into the baseline.
- No Issue #3/#4/#5 extended-data reruns were performed.

The original baseline SHA-256 recorded for `crop_long_v1.csv` remained unchanged in the final investigation evidence.

## 9. Final readiness assessment

| Workstream | Result |
|---|---|
| Existing baseline verification | Completed for inspected artifacts |
| Agricultural source investigation | Partial |
| Selected 2020–21 area-record audit | Completed for selected records |
| Complete extended agricultural coverage | Not verified |
| Yield/production sample verification | Partial |
| Weather source investigation | Partial |
| IMD 2020 file retrieval | Completed |
| IMD full data-value validation | Not completed |
| ERA5/ERA5-Land sample validation | Not completed |
| Historical district boundary crosswalk | Not verified |
| Weather feature compatibility | Assessed; no feature verified compatible |
| New data ingestion | Not performed |
| Issue #3/#4/#5 extended-data reruns | Not performed |

## 10. Conclusion and recommendation

The investigation has established useful source availability evidence and identified the main technical blockers to extending QWeatherRiskF historical coverage.

However, no additional agricultural or weather dataset has passed all the required checks for safe district-level ingestion.

The recommended outcome is a documentation-only investigation contribution, subject to team review.

**Issue #44 remains open for the historical-extension objective.**

The next implementation phase should focus on obtaining data-bearing agricultural records, resolving historical district and crop mappings, validating weather source contents, and defining a reproducible district-level aggregation process.

Only after those prerequisites pass should additional records be ingested and Issues #3, #4 and #5 be rerun.
