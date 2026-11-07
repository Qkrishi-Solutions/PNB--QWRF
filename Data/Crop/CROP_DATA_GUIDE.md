# Crop data

## Purpose

This folder contains the crop-history dataset used as the agricultural
component of QWeatherRiskF. It supplies district-year crop observations that
can be joined with annual weather features.

## Current file

| File | Description | Status |
|---|---|---|
| `ICRISAT-District Level Data (2).csv` | ICRISAT district-level crop statistics, including crop area, production, and yield fields | Source data used by the current pipeline |

The dataset is treated as the crop-history source for the merged
district-year tables. Its crop, district, and year fields are mapped to the
weather data before annual feature aggregation.

## Important interpretation notes

- The project currently uses district and year as the main merge keys.
- The ICRISAT year convention (crop year beginning in the named year) should
  be confirmed against the source documentation before final publication.
- Source missing-value codes such as `-1` and `-500` are interpreted as
  missing in the derived merged datasets rather than as measured values.
- Some crop series require quality review. Known concerns include a sugarcane
  break, a cotton break, and implausible values in an oilseeds aggregate.

## Status and future updates

This folder currently contains one source file only. If a cleaned or
standardized crop file is added, retain the original source and document:

1. the input filename;
2. every transformation or missing-value rule;
3. the output filename and version;
4. validation checks and known limitations; and
5. which file collaborators should use.
