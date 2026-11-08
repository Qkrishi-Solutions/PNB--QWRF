# Issue 1: Crop-long restructuring and validation

## Purpose

Issue 1 converts the wide crop-weather table into a crop-long dataset for
crop-level analysis and predictive modelling. The original wide source is
preserved. The long output has one record for each
`district_code` + `year` + `crop` combination, while district metadata and
weather features are carried onto every crop row for that district-year.

## Deliverables in this folder

| File | Purpose | Status |
|---|---|---|
| `crop_restructure_validate.py` | Reproducible wide-to-long transformation, quality flags, provenance flags, and acceptance checks | Processing and validation script |
| `crop_long_v1.csv` | Final crop-long deliverable generated from the cleaned/fixed crop dataset | **Recommended output for downstream crop-level analysis** |
| `crop_data_validation_report.pdf` | Validation results for restructuring, round-trip preservation, missing/zero values, aggregates, and consistency checks | Validation report |
| `QWeatherRiskF_Verification_and_Fix_Plan.pdf` | Reasoning and decision plan for cotton 2019, oilseeds, minor pulses, and other suspected data issues | Investigation and decision record |
| `QWeatherRiskF_Data_Fixes_Report.pdf` | What was changed, why it was changed, and which observations remain estimated or excluded | Data-fix report |
| `merged_cleaned_v1f.csv` | Corrected wide upstream dataset used for the final crop-long workflow | Fixed wide source |
| `v1f_change_log.csv` | Cell-level record of every v1f change, including old value, new value, action, and reason | Fix provenance |

## Processing flow

```text
merged_cleaned_v1d.csv
        |
        | verification of suspected crop issues
        v
merged_cleaned_v1f.csv
        |
        | crop_restructure_validate.py
        | identify crop columns, reshape, flag, validate
        v
crop_long_v1.csv
```

The final long file contains 1,430 records: 65 district-year observations
(13 districts × 5 years) × 22 crop entries. The current CSV contains 124
columns, including crop measurements, district metadata, weather features,
modelling-role fields, and validation flags.

## Restructuring method

The script identifies crop-specific columns using the naming pattern:

```text
<crop>_<measure>_<unit>
```

The three measures are:

- `area_1000_ha`
- `production_1000_tons`
- `yield_kg_per_ha`

For each crop, the script creates one long table and then joins the
district-year metadata and all weather columns using a validated many-to-one
merge. Weather values are therefore repeated across the crop rows belonging
to the same district-year; this is intentional and keeps crop-level modelling
aligned with the same weather exposure.

The script also verifies the source SHA-256 checksum before and after
processing, so the input wide dataset is not modified.

## Quality and provenance flags

The output retains the original numeric values and adds explicit flags for:

- missing crop measurements;
- zero area, production, or yield;
- all-zero observations;
- zero area with positive production;
- positive area with zero production or yield;
- negative values;
- reported yield versus `production / area × 1000` mismatch;
- low area;
- unusually large year-to-year yield jumps;
- aggregate mismatch;
- values filled from a prior missing-value version;
- values changed by the v1f fix plan; and
- values estimated by allocation rather than directly sourced.

This means a flagged observation is not silently discarded. Analysts can
choose whether to exclude it, model it, or report it depending on the task.

## Fixes applied before restructuring

### Sugarcane

Sugarcane production and yield unit errors for 2015–2017 were corrected by a
factor of 10. The fix report states that the corrected values match official
totals within approximately 0.1%. This correction from the earlier v1c work
is retained in `merged_cleaned_v1f.csv` and marked in the long output.

### Cotton 2019

Cotton area was retained, while production and yield were corrected for five
districts using official agricultural statistics. For eight districts where
district-level official values were not available, production was estimated
by allocating the official combined total using area multiplied by prior
yield as the weight; yield was then recomputed from production and area.
These eight observations are explicitly marked as estimated and must not be
treated as direct measurements.

### Minor pulses

The 2018–2019 zero values were interpreted as “not reported,” not as measured
zero production. The affected cells were set to missing in v1f and are
flagged in the long output. Minor pulses remain available for documentation
but are excluded from modelling by role.

### Oilseeds

The oilseeds aggregate is retained for traceability but is not repaired or
recommended for modelling. Its composition does not match the listed member
crops, and the investigation indicates that it includes additional crops
such as oil palm. Use the individual oilseed crops instead.

## Aggregates and seasonal components

The long dataset retains and identifies non-leaf crop categories:

| Category | Treatment |
|---|---|
| `sorghum` | Aggregate retained; its members are `kharif_sorghum` and `rabi_sorghum` |
| `kharif_sorghum` | Seasonal component retained and marked as part of sorghum |
| `rabi_sorghum` | Seasonal component retained and marked as part of sorghum |
| `oilseeds` | Aggregate retained for traceability but excluded from modelling because its member sum is inconsistent |
| `minor_pulses` | Group category retained with missing/not-reported flags and excluded from modelling |

The columns `is_aggregate`, `is_seasonal_component`,
`season_component`, `component_of_aggregate`, `is_group_category`,
`modelling_role`, and `use_in_modelling` make these decisions machine-readable.

## Validation and acceptance

The validation report confirms the required structural properties:

- one row per district-year-crop;
- no duplicate `(district_code, year, crop)` keys;
- exact round-trip preservation of source crop values;
- weather features preserved unchanged;
- weather values consistent across crops within a district-year; and
- source checksum unchanged by processing.

The report also documents data-quality findings rather than hiding them.
These include missing/filled provenance, zero-area observations, production
and yield inconsistencies, yield jumps, and aggregate mismatches. The
presence of a quality flag does not mean the restructuring failed; it records
an observation requiring interpretation.

## Recommended file for future work

Use **`crop_long_v1.csv`** as the primary dataset for future crop-level
analysis and predictive modelling from this issue. It is the final
crop-long deliverable and includes the v1f fixes and all required flags.

Before modelling:

1. filter using `use_in_modelling` or the documented `modelling_role`;
2. exclude aggregates and group categories unless the analysis explicitly
   requires them;
3. inspect `flag_estimated_by_allocation` for cotton 2019;
4. inspect missing and zero flags instead of interpreting every zero as a
   measured zero; and
5. record the exact filename and filtering rules in the analysis output.

The wide files remain available for audit and reproducibility. Do not edit
`merged_cleaned_v1f.csv` or the original merged datasets in place; create a
new version and update the change log when future corrections are required.
