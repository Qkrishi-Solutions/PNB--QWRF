# QWeatherRiskF

**Hybrid classical-quantum agricultural risk and decision support pipeline for Andhra Pradesh.**

QWeatherRiskF merges district-level crop statistics with seasonal weather features, forecasts yield anomalies with classical machine learning, converts forecasts into profit and risk scores, and allocates crops under land, water and budget limits using QAOA, benchmarked against Gurobi.

> **Status: INITIAL** The data layer is in progress. The finance layer, risk engine and optimisation are in design. QAOA is evaluated against an exact classical solver and results are reported as measured.

---

## What it does

```
Sources → Ingestion and merge → Features and zones → ML forecast
        → Risk and finance → Optimisation (QAOA, Gurobi benchmark) → Decision support
```

| Stage | What happens |
|---|---|
| 1. Data layer | Crop statistics (ICRISAT), rainfall (IMD), climate (ERA5-Land), prices and costs |
| 2. Ingestion and merge | Clean, harmonise district codes, join on district and year |
| 3. Features and zones | Seasonal weather windows, anomalies against each district's normal, district clustering |
| 4. ML forecasting | Predict yield anomalies per crop with Ridge, Random Forest, XGBoost and CatBoost |
| 5. Risk and finance | Expected production, revenue and profit; weather, yield and financial risk scores |
| 6. Optimisation | QUBO → Ising → QAOA for crop allocation, with Gurobi as the exact benchmark |
| 7. Decision support | Allocation, expected outcomes and risk reports |

Classical ML answers *what will happen*, the risk and finance layer answers *what it means*, the QUBO defines *what decision to optimise*, and QAOA searches for the allocation. Quantum computing is used only for optimisation, never for prediction.

## Data

| Source | Content | Used for |
|---|---|---|
| ICRISAT District Level Database (unapportioned) | Area, production, yield by crop, district and year | Targets and crop history |
| IMD 0.25° gridded rainfall | Daily rainfall | Rainfall features |
| ERA5-Land reanalysis | Temperature, humidity, wind, solar radiation | Climate features |
| Agmarknet, MSP, cost-of-cultivation data | Crop price and cost | Financial impact (not yet integrated) |

**Working dataset: `merged_district_year_v1b.csv`**

- 65 rows: 13 districts × 5 years (2015–2019), joined on `district_code` + `year`.
- 160 columns: identifiers, 23 crops × (area, production, yield), 6 area-only columns, a rainfall-coverage column, and 78 weather columns (6 seasonal windows × 13 features).
- Missing values are left blank, never filled. ICRISAT's `-1` and `-500` codes are set to blank.
- See the Scientific Data Guide for column definitions, units, caveats and known data issues.


## Method summary

- **Target:** yield anomaly, meaning yield minus that district's average yield. District averages are computed from training years only, to avoid leakage.
- **Features:** two to four weather anomalies per crop, chosen from agronomic reasoning (for example Kharif rainfall, longest dry spell and north-east monsoon rainfall for groundnut).
- **Models:** Ridge, Random Forest, XGBoost, CatBoost, each compared with a baseline that predicts the district's usual yield.
- **Validation:** leave-one-year-out (five folds). Skill is `1 − model error / baseline error`, with a bootstrap 95% interval. Zero means no better than the baseline.

## Known limitations

- Only five years (2015–2019) for 13 districts, so about four or five independent years. Intervals are wide and the results are early evidence.
- 2016 and 2018 were poor years across the whole state, so a large part of the signal is shared by all districts.
- Srikakulam rainfall is blank (only 75% of its area has valid IMD cells).
- The ICRISAT year convention (crop year starting in that year) is tentative and must be confirmed from its documentation.
- Thresholds such as hot day (35 °C), rainy day (2.5 mm) and base temperature (10 °C) are placeholders.
- Some crops have data breaks (sugarcane 2018, cotton 2019) and the oilseeds aggregate has implausible yields; these are excluded until checked.
- ERA5-Land temperature has not yet been checked against IMD station temperature.
- QAOA is simulated and limited to roughly 20 to 25 qubits.

## Roadmap

1. Confirm the ICRISAT year convention and extend the weather record back to about 1990.
2. Add price, cost and water data.
3. Build the feature script (anomalies, seasonal windows, zones).
4. Tune and validate models per crop; add SHAP and yield ranges.
5. Build the risk engine and financial-impact module.
6. Build the QUBO and run QAOA and Gurobi on pilot-size instances.
7. Reports and dashboard.

## Reproducibility

Fixed random seeds, versioned data and code, and no gap-filling. Every reported result should be reproducible from the dataset version and script in this repository.

## Licence and ownership

© [Your name / organisation]. All rights reserved. Ownership and permitted use are governed by the agreement with the client. If the repository is published, add a licence (for example MIT or Apache-2.0) after confirming with the owner.

Do not commit client data, credentials or Gurobi licence files.

## Data acknowledgements

- ICRISAT District Level Database.
- India Meteorological Department (IMD) gridded rainfall.
- ERA5-Land, Copernicus Climate Change Service (C3S) / ECMWF.

