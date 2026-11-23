# Extended Meteorological Ingestion Layer (`weather_extended`)

## 1. Overview
This module implements **Module W1** of the QWeatherRiskF architecture (Issue #45), providing extended atmospheric and oceanic variables for severe-weather risk screening (Module M4 / Issue #48).

It bridges the gap between land-surface agronomic weather (rainfall, temperature) and extreme cyclonic events by tracking:
- Atmospheric pressure drops (Mean Sea Level Pressure)
- Sustained wind speeds and extreme gusts
- Meteorological wind direction
- Bay of Bengal Sea Surface Temperature (SST)

---

## 2. Deliverables Summary

| File | Description | Rows / Coverage | Source |
|---|---|---|---|
| `weather_extended_v1.csv` | Canonical daily extended weather dataset | 4,745 rows (2015 full year) | ECMWF ERA5 Single-Levels |
| `weather_extended_coverage_report_v1.csv` | Missingness and physical validity audit report | 13 districts | Automated |
| `WEATHER_EXTENDED_DATA_DICTIONARY.md` | Exact parameter definitions, units, formulas, and physical bounds | - | ECMWF Documentation |
| `build_weather_extended.py` | Standalone reproducible pipeline script | - | Python 3 |
| `tests/test_weather_extended.py` | Unit test suite verifying schema, key uniqueness, and physics | 8 unit tests | Python 3 |

---

## 3. Geographic & Relational Integrity

- **Foreign Key Alignment:** All 13 district records strictly link 1-to-1 with `geography_master_v1.csv` using canonical `district_code` (`502` to `523`) and uppercase `district_name`.
- **Inland vs Coastal Separation:**
  - **Coastal Districts (9):** `EAST GODAVARI`, `GUNTUR`, `KRISHNA`, `SPR NELLORE`, `PRAKASAM`, `SRIKAKULAM`, `VISAKHAPATNAM`, `VIZIANAGARAM`, `WEST GODAVARI` have 100% valid SST values.
  - **Inland Districts (4):** `ANANTAPURAMU`, `CHITTOOR`, `KURNOOL`, `YSR` are strictly `NaN` for SST (zero silent imputation).

---

## 4. Verification

Run automated tests using standard library `unittest`:
```powershell
python -m unittest weather_extended/tests/test_weather_extended.py
```
