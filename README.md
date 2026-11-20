# QWeatherRiskF

**Weather-risk and agricultural analytics for Andhra Pradesh agricultural lending, with a planned hybrid classical-quantum decision layer.**

QWeatherRiskF merges district-level crop statistics with seasonal weather features, selects crop-specific weather predictors, groups districts into climate zones, and is being extended to forecast crop yield and relate weather stress to a loan portfolio. Classical machine learning predicts. A later optimisation layer (QAOA, benchmarked against an exact classical solver) is planned for decision support.

> **Status: data layer and feature engineering complete (Issues 1 to 6). Financial data foundation complete on synthetic data (Issues 21 to 25, 32). ML forecasting, risk, optimisation and alerts are planned, not built.**
>
> The financial data in this repository is **synthetic**. PNB supplied field definitions only. Nothing here is PNB customer or portfolio data, and nothing here is a validated crop-loss, default or NPA prediction.

---

## What it does

```
Sources → Ingestion and merge → Features and zones → ML forecast
        → Exposure and risk → Scenarios and optimisation (QAOA, exact benchmark) → Decision support
```

| Stage | What happens | Status |
|---|---|---|
| 1. Data layer | Crop statistics (ICRISAT), rainfall (IMD), climate (ERA5-Land), pincode reference, synthetic loan portfolio | Done for the current scope |
| 2. Ingestion and merge | Clean, harmonise district codes, join on district and year | Done |
| 3. Features and zones | Seasonal weather windows, anomalies against each district's normal, crop-specific feature selection, district clustering | Done |
| 4. ML forecasting | Area and yield forecasts per crop, production with intervals | Planned (issues below) |
| 5. Exposure and risk | Loan exposure, insurance coverage, exposure at risk | Planned |
| 6. Scenarios and optimisation | Scenario set, QUBO, Ising, QAOA with exact benchmark | Planned |
| 7. Decision support | Alerts, dashboards, reports | Planned |

Quantum computing is used only for optimisation, never for prediction. The full plan is in [`final docs/QWRF_ROADMAP_AND_HYBRID_ARCHITECTURE.md`](final%20docs/QWRF_ROADMAP_AND_HYBRID_ARCHITECTURE.md).

## Repository layout

| Folder | Content | Issue |
|---|---|---|
| `CropLongFormatv1/` | Merged agricultural dataset (`merged_cleaned_v1f.csv`) and crop-long dataset (`crop_long_v1.csv`), validation reports, change log | 1 |
| `CropModelEligibility/` | Crop eligibility for modelling and observation coverage | 2 |
| `weather_feature_engineering/` | Weather features, normals and anomalies (leave-one-year-out and forward chaining), validation tables | 3 |
| `yield_feature_engineering/` | District and crop yield normals and anomalies, validation | 4 |
| `crop specific weather feature selection/` | Spearman, FDR and multicollinearity based selection of weather predictors per crop | 5 |
| `District Climate Zonal Clustering/` | District climate profiles, three climate zones, clustering evaluation | 6 |
| `financial_data_validation/` | Schema and validator for the loan portfolio layout | 21, 24 |
| `address_expansion/` | Address splitting and pincode-to-district mapping | 23 |
| `synthetic_portfolio/` | Synthetic loan portfolio generator, ruleset, assumptions, tests | 25 |
| `synthetic_training_dataset/`, `financial_train_test_data/` | Synthetic training and test datasets (v3, v4), dictionaries, rulesets | 32 |
| `final docs/` | Roadmap, SRS, data dictionaries, presentation scripts | |

`Issues/CropLongFormatv1/` is a duplicate of `CropLongFormatv1/` and is tracked for removal in issue #35.

## Data

| Source | Content | Used for |
|---|---|---|
| ICRISAT District Level Database (unapportioned) | Area, production, yield by crop, district and year | Targets and crop history |
| IMD 0.25° gridded rainfall | Daily rainfall | Rainfall features |
| ERA5-Land reanalysis | Temperature, humidity, wind, solar radiation | Climate features |
| India Post pincode directory | Pincode to district | Linking loan records to districts |
| Agmarknet, MSP, cost-of-cultivation data | Crop price and cost | Not yet integrated |

**Current agricultural datasets**

- `CropLongFormatv1/merged_cleaned_v1f.csv`: 65 rows (13 districts × 5 years, 2015 to 2019), 151 columns. This is the current merged district-year dataset. Corrections from earlier versions are recorded row by row in `CropLongFormatv1/v1f_change_log.csv`. Some 2019 cotton values are official figures or **allocation estimates**; the change log marks which.
- `CropLongFormatv1/crop_long_v1.csv`: 1,430 rows, 124 columns, one record per district, year and crop.
- Missing values are left blank, never filled. ICRISAT's `-1` and `-500` codes are set to blank.

Earlier merge versions (`v1b`, `v1b_filledv2`, `v1c`, `v1d`) are superseded and not in this repository. Use `merged_cleaned_v1f.csv`.

**Weather and yield anomalies**

- 13 weather features × 6 seasonal windows (`annual_cal`, `annual_crop`, `kharif`, `ne`, `rabi_next`, `rabi_prev`) = 78 candidate features. `kharif` and `ne` overlap; do not treat them as independent or add them.
- Two anomaly methods, kept separate: **leave-one-year-out** (cross-validation, can use later years) and **forward chaining** (target year uses only strictly earlier years, at least two valid years).
- Forward-chaining anomalies are unavailable for early years: 2015 and 2016 lack history, and 2017 `rabi_prev` is missing because 2015 `rabi_prev` is missing in the source.

**Synthetic financial data** (`financial_train_test_data/`)

- v3 and v4 train and test files in the 20-column PNB layout and a richer layout with weather and yield anomalies, climate zone, crop and insurance fields.
- Labels come from a documented risk score, and class mix is tuned for model learnability. A model trained on this recovers those rules. It is not evidence about real borrowers or a real weather-to-default link.
- v3 was reported as reviewed by PNB (per the project owner; no approval reference recorded yet). v4 changes the `agri sector` domain and its approval is not recorded.

## Method summary

- **Target:** yield anomaly (z-score against a district and crop baseline built from allowed years only).
- **Weather features:** 78 anomaly features per district and year.
- **Feature selection (Issue 5):** per crop, Spearman correlation, Benjamini-Hochberg false-discovery control, multicollinearity filter, at most four features per fold, leave-one-year-out folds. A stable predictor must be selected in at least two folds. In the recorded run, 17 crops were eligible and stable predictors were found for four (cotton, maize, pigeonpea, sunflower). Verify against the current output files before quoting.
- **Climate zones (Issue 6):** three zones over 13 districts (k=3: silhouette 0.256, mean ARI 1.0 across initialisation seeds; k=4 had a slightly higher silhouette, 0.267, but lower stability, mean ARI 0.77). Zones group similar climate. They are **not** risk labels.
- **Validation of forecasts (planned):** forward chaining only, with feature selection inside each fold, compared against a climatology baseline.
- Statistical associations are not causation. A predictor that was not selected does not prove the weather variable has no effect.

## Known limitations

- Only five years (2015 to 2019) for 13 districts. Three yield-anomaly years are scorable under forward chaining. Intervals are wide and results are exploratory.
- 2016 and 2018 were poor years across the state, so much of the signal is shared by all districts.
- Feature selection uses leave-one-year-out folds, so it is not prospective chronological forecasting.
- Srikakulam rainfall had incomplete IMD coverage (about 75% of its area). Check the data guide before relying on it.
- The ICRISAT year convention is tentative and must be confirmed from its documentation.
- The source did not provide a full unit dictionary. Units are not assumed.
- Thresholds such as hot day (35 °C), rainy day (2.5 mm) and base temperature (10 °C) are placeholders.
- ERA5-Land temperature has not been checked against IMD station temperature.
- Synthetic financial data supports pipeline development only.
- QAOA, when built, will be simulated and limited to roughly 20 to 25 qubits, with no claim of quantum advantage.

## Roadmap

Planned work is tracked as GitHub issues, in this order.

| Phase | Work | Issues |
|---|---|---|
| 0. Foundation fixes | Remove duplicate folder, add `crop_feature_selection_v1.csv`, root dependency file and test runner | #35, #36, #37 |
| 1b. Data extension | Dictionary v2, insurance schema and synthetic insurance, RBI/NABARD benchmark, geography master, crop calendar, extended history | #38 to #44 |
| 2. Severe weather | Extended weather, cyclone catalogue, IMD warnings, severe-weather indicators | #45 to #48 |
| 3. ML forecast | ML-ready dataset, area, yield and production forecasts, validation harness | #7, #49 to #52 |
| 4 onward | Exposure and insurance risk, scenario engine, QAOA optimisation, alerts and dashboards | Not yet filed |

## Reproducibility

Fixed random seeds, versioned data and code, and no gap-filling. Every reported result should be reproducible from the dataset version and script in this repository. Each issue folder documents its run command. Branch from `dev` and open pull requests to `dev`.

## Licence and ownership

Apache License 2.0 (see `LICENSE`). Ownership and permitted use are governed by the agreement with the client.

Do not commit real client data, borrower identifiers, credentials or Gurobi licence files.

## Data acknowledgements

- ICRISAT District Level Database.
- India Meteorological Department (IMD) gridded rainfall.
- ERA5-Land, Copernicus Climate Change Service (C3S) / ECMWF.
- India Post pincode directory.
