# Crop Calendar Data Layer (`crop_calendar`)

## 1. Overview
This module implements the canonical crop calendar required by **Issue #43** and the **Phase 2 Forecasting & Feature Engineering** pipeline (Modules M1, M2, and Issue #49) for the QWeatherRiskF project.

It provides verified sowing, harvesting, and phenological windows across the 17 eligible crops cultivated in Andhra Pradesh, mapped strictly to the six seasonal weather windows defined in Issue #3:
- `kharif` (South-West Monsoon)
- `ne` (North-East Monsoon)
- `rabi_next` (Winter / Rabi cycle)
- `rabi_prev` (Prior cycle winter baseline)
- `annual_crop` (Crop year: June 1 – May 31)
- `annual_cal` (Calendar year: January 1 – December 31)

---

## 2. Deliverables

| File | Description | Rows | Source |
|---|---|---|---|
| `crop_calendar_v1.csv` | Canonical statewide crop $\times$ season lookup table | 25 | ANGRAU / DES AP |
| `district_crop_calendar_v1.csv` | District-level agronomic variations across 13 canonical AP districts | 24 | ICAR-CRIDA District Contingency Plans |
| `crop_calendar_notes.md` | In-depth agronomic reference documentation, citations, and phenological risk notes | - | ANGRAU, ICAR, DES AP |
| `build_crop_calendar.py` | Standalone reproducible generation script | - | Python 3 |
| `tests/test_crop_calendar.py` | Comprehensive verification test suite using standard library `unittest` | - | Python 3 |

---

## 3. Schema Definitions

### `crop_calendar_v1.csv`
- `crop`: Lowercase crop identifier matching Issue 2 eligibility and Issue 5 consensus (`rice`, `groundnut`, `maize`, `sugarcane`, `pigeonpea`, `sesamum`, `chickpea`, `sunflower`, `finger_millet`, `pearl_millet`, `cotton`, `castor`, `kharif_sorghum`, `rabi_sorghum`, `soyabean`, `rapeseed_and_mustard`, `safflower`, `wheat`, `linseed`).
- `season`: Agricultural season (`kharif`, `rabi`, `annual`).
- `sowing_window_start`: Starting calendar month of sowing/nursery.
- `sowing_window_end`: Ending calendar month of sowing/transplanting.
- `harvest_window_start`: Starting calendar month of harvest.
- `harvest_window_end`: Ending calendar month of harvest.
- `duration_days_approx`: Approximate days to physiological maturity.
- `weather_window_alignment`: Project weather window from Issue 3 (`kharif`, `ne`, `rabi_next`, `annual_crop`, `annual_cal`).
- `cyclone_vulnerability_window`: Critical cyclone/flood risk window (`ne_oct_nov`, `sw_sept`, `none`).
- `notes_phenology`: Field observations, moisture constraints, and vegetative sensitivity.
- `source_reference`: Explicit public citation (ANGRAU / ICAR / DES AP).
- `status`: Verification status (`verified` or `unavailable`).

### `district_crop_calendar_v1.csv`
- `district_code`: 3-digit canonical district code (Issue 42 geography master).
- `district_name`: Standardized district name.
- `climate_zone`: Climate cluster assignment (Issue 6 / Issue 42).
- `crop`: Standardized crop identifier.
- `season`: Season identifier.
- `primary_agro_system`: Dominant agricultural system (e.g., Rainfed dryland, Canal irrigated delta, Zero-till rice fallows).
- `typical_sowing_window`: District-specific operational sowing window.
- `typical_harvest_window`: District-specific operational harvest window.
- `weather_window_alignment`: Aligned Issue 3 weather window.
- `district_specific_notes`: Local agricultural characteristics.
- `source_reference`: Specific ICAR-CRIDA district contingency plan citation.
- `status`: Verification status.

---

## 4. Governance Compliance

- **No Silent Imputation:** Missing is not zero. Crops with negligible or non-viable cultivation in Andhra Pradesh (`wheat`, `linseed`) are explicitly categorized as `unavailable` rather than assigned synthetic or guessed dates.
- **Authoritative Sourcing:** 100% of rows carry cited public agronomic sources from state agricultural universities (ANGRAU) or national dryland agriculture institutes (ICAR-CRIDA).
- **Relational Integrity:** District codes in `district_crop_calendar_v1.csv` strictly join 1-to-1 with `geography_master_v1.csv`.
