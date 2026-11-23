# Extended Meteorological Ingestion Data Dictionary (Issue #45)

## 1. Overview
This dictionary defines the variables in `weather_extended_v1.csv` produced by **Module W1** under **Phase 2 (Objective 3: Severe-weather and cyclone intelligence)** of the QWeatherRiskF project.

All physical variables are derived from the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 single-levels reanalysis dataset (`reanalysis-era5-single-levels`) on a 0.25° grid, intersected with the 13 canonical district polygons of Andhra Pradesh and a 75 km offshore marine buffer into the Bay of Bengal.

---

## 2. Canonical Column Definitions

| Column Name | Type | Physical Unit | ECMWF Parameter | Physical Validity Bounds | Description |
|---|---|---|---|---|---|
| `district_code` | int | - | - | `[502, 523]` | 3-digit canonical district code matching Issue #42 geography master (`geography_master_v1.csv`). |
| `district_name` | str | - | - | 13 AP names | Standardized uppercase district name. |
| `date` | str | `YYYY-MM-DD` | - | `2015-01-01` to `2015-12-31` | Observation date. |
| `era5_grid_cells` | int | count | - | `[4, 18]` | Number of 0.25° grid points intersecting the district boundary polygon. |
| `msl_min_hpa` | float | $\text{hPa}$ / $\text{mb}$ | `msl` (Mean sea level pressure, ID 151) | `[900.0, 1050.0]` | Daily minimum sea level pressure across the district polygon ($\text{Pa} / 100.0$). Key indicator of cyclonic depression ($<1000\text{ hPa}$) and severe cyclonic storms ($<990\text{ hPa}$). |
| `msl_mean_hpa` | float | $\text{hPa}$ / $\text{mb}$ | `msl` (ID 151) | `[950.0, 1040.0]` | Daily spatial mean sea level pressure across the district. |
| `wind_speed_mean_kmh` | float | $\text{km/h}$ | `u10`, `v10` (IDs 165, 166) | `[0.0, 150.0]` | Daily mean 10-meter wind speed: $\sqrt{u_{10}^2 + v_{10}^2} \times 3.6$. |
| `wind_speed_max_kmh` | float | $\text{km/h}$ | `u10`, `v10` (IDs 165, 166) | `[0.0, 250.0]` | Daily maximum sustained 10-meter wind speed. |
| `wind_gust_max_kmh` | float | $\text{km/h}$ | `fg10` (10m wind gust, ID 49) | `[0.0, 300.0]` | Daily peak 10-meter wind gust: $fg10 \times 3.6$. Critical threshold for crop lodging and structural damage ($>65\text{ km/h}$). |
| `wind_direction_deg` | float | Degrees ($^\circ$) | `u10`, `v10` | `[0.0, 360.0)` | Prevailing meteorological wind direction from true North clockwise ($0^\circ = \text{North}, 90^\circ = \text{East}, 180^\circ = \text{South}, 270^\circ = \text{West}$). Calculated as $\text{mod}(270 - \text{atan2}(\bar{v}_{10}, \bar{u}_{10}) \times \frac{180}{\pi}, 360)$. |
| `sst_mean_c` | float | $^\circ\text{C}$ | `sst` (Sea surface temperature, ID 34) | `[20.0, 35.0]` or `NaN` | Daily mean sea surface temperature in the adjacent Bay of Bengal (within 75 km offshore marine buffer) for coastal districts. **Strictly NaN for inland districts.** |

---

## 3. Data Governance & Imputation Policy

1. **Zero Silent Imputation:**
   - In accordance with project policy (*"Missing is not zero"*), inland districts (`ANANTAPURAMU`, `CHITTOOR`, `KURNOOL`, `YSR`) have **no sea surface interface**. Their `sst_mean_c` is explicitly set to `NaN` and never imputed as 0.0 or filled from distant waters.
2. **Marine Buffer Specification:**
   - Coastal districts (`EAST GODAVARI`, `GUNTUR`, `KRISHNA`, `SPR NELLORE`, `PRAKASAM`, `SRIKAKULAM`, `VISAKHAPATNAM`, `VIZIANAGARAM`, `WEST GODAVARI`) sample ocean cells within an EPSG:32644 projected buffer of 75,000 meters (75 km) from the district coastline into the Bay of Bengal.
3. **Unique Composite Key:**
   - Every row is strictly unique on `(district_code, date)`.
