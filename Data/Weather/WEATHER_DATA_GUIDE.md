# Weather data

## Purpose

This folder contains daily district-level weather data used to create annual
features for the crop-weather datasets. It combines IMD gridded rainfall with
ERA5-Land weather variables for Andhra Pradesh districts.

## Files

| File | Description | Recommended use |
|---|---|---|
| `ap_district_daily_historical_weather_2015_2021.csv` | Original daily historical export with IMD rainfall missingness preserved | Use when analysing the original observations or source coverage |
| `ap_district_daily_historical_weather_2015_2021_filledv2.csv` | Filled version used for the Srikakulam-complete merged weather update | Use for the filled Srikakulam workflow, with imputation flags retained |

The daily coverage is 13 districts from 2015-01-01 through 2021-12-31,
with 33,241 expected rows. The structural integration QA recorded a complete
IMD/ERA5 join and no missing ERA5 variables. The original file has 2,557
missing rainfall rows, all associated with Srikakulam's persistent spatial
coverage gap.

## Sources and variables

- **IMD RF25:** daily gridded rainfall, aggregated to districts using
  grid-cell/district intersection areas.
- **ERA5-Land:** daily minimum and maximum temperature, mean temperature,
  relative humidity, wind speed, and solar radiation.
- The daily files retain provenance columns such as source, source file,
  valid-area coverage, and rainfall-imputation status where available.

## Srikakulam reconstruction

The filled version reconstructs the seven persistent Srikakulam IMD cells
that were encoded as missing sentinel values (`-999`/`-9999`). The method is:

1. identify affected Srikakulam grid cells;
2. estimate missing daily rainfall with inverse-distance weighting (IDW);
3. use up to the four nearest valid cells within 0.5 degrees;
4. use a distance power of 2;
5. clamp reconstructed rainfall to non-negative values; and
6. repeat the area-weighted district aggregation.

The original rainfall values are not overwritten. Reconstructed observations
are marked with `rainfall_imputed`, and the filled source is labelled
`IMD_RF25_IDW_SRIKAKULAM_GAP_FILL`.

## Parameter experiment and validation

The IDW settings were selected after comparing candidate reconstruction
configurations with leave-one-cell-out/day cross-validation on observed
Srikakulam cells. The final configuration is the documented four-neighbour,
0.5-degree, power-2 setting. The validation report records the selected
configuration and its error statistics; future parameter changes must be
recorded there rather than silently changing the dataset.

For the annual merged weather features, the current thresholds are:

| Feature rule | Value |
|---|---:|
| Rainy day | rainfall >= 2.5 mm |
| Dry day | rainfall < 1.0 mm |
| Hot day | maximum temperature >= 35 C |
| Cold day | minimum temperature <= 15 C |
| GDD base temperature | 10 C |

The annual windows include calendar year, June-May crop year, Kharif
(June-October), northeast monsoon (October-December), and Rabi
(November-March) windows. These are agronomic processing choices, not claims
that the thresholds are universally optimal.

## Status

- Original weather export: **available; source missingness preserved**.
- Filled weather export: **available; Srikakulam rainfall reconstructed**.
- Structural QA: **passed**, but structural checks do not prove
  meteorological accuracy.
- The filled version still carries uncertainty because reconstructed rainfall
  is estimated rather than directly observed.
- Any future weather file must state whether values are observed, reconstructed,
  or mixed, and must preserve the corresponding provenance flags.
