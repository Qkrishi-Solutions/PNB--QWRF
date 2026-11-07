# Merged crop-weather data

## Purpose

This folder contains annual district-year datasets created by mapping the
ICRISAT crop data to the weather data on district and year. It is the main
handoff area for datasets used by downstream feature engineering and modelling.

## Version history

| File | Meaning | Recommended use |
|---|---|---|
| `merged_district_year_v1b.csv` | First documented merged crop-weather table; original rainfall gaps are retained | Baseline comparison and reproducibility |
| `merged_district_year_v1b_filledv2.csv` | `v1b` with Srikakulam annual weather features recomputed from the filled daily weather file | Use when the analysis requires the filled Srikakulam weather values |
| `merged_district_year_v1c.csv` | Later merged revision with rainfall completion and additional data-quality corrections | Historical comparison; confirm the associated report before use |
| `merged_cleaned_v1d.csv` | Latest cleaned merged revision, incorporating fixes to issues identified in earlier versions | Current final candidate, pending team confirmation |

The version labels must not be treated as interchangeable. Analysis results
should always record the exact input filename.

## Mapping and aggregation

The merge uses district identity and year. Weather observations are first
aggregated to annual/agricultural windows, then joined to the crop
district-year observations. The daily district rainfall aggregation uses the
area of each grid-cell intersection with the district, rather than a simple
unweighted average.

The annual weather features include rainfall totals, rainy days, longest dry
spell, maximum five-day rainfall, temperature summaries, hot/cold days,
diurnal temperature range, humidity, solar radiation, wind, and
growing-degree-day totals. Crop-year and seasonal windows are retained as
separate feature groups.

## District-code mapping

The project harmonizes district names/codes before joining. The join key is
the harmonized district identifier plus year; it is not a raw string match
between uncleaned source names. The mapping table and any future district
renaming must be preserved with the processing report so that the join can be
reproduced.

## Scientific guide

`QWeatherRiskF — Merged District-Year Dataset Scientific Data Guide (v1b).pdf`
is the detailed reference for column definitions, units, formulas, source
limitations, and known data issues. This README is the short operational guide
for selecting a file.

## Which file should I use?

- For reproducing the original baseline: use `merged_district_year_v1b.csv`.
- For specifically testing the Srikakulam filled-rainfall workflow: use
  `merged_district_year_v1b_filledv2.csv`.
- For new downstream analysis: start with `merged_cleaned_v1d.csv`, but
  confirm with the team that its supporting quality report is complete.
- Do not delete the older versions; they are needed to explain changes and
  reproduce earlier results.

## Required metadata for future versions

Every new merged dataset should have a short report stating:

- input crop and weather files;
- district mapping and join counts;
- year range and row/column counts;
- missing-value treatment;
- formula or threshold changes;
- data-quality corrections;
- validation results; and
- whether the file is a baseline, intermediate, or recommended release.
