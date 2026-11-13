# QWeatherRiskF: Architecture Diagrams

Mermaid diagrams of the project architecture, derived from the repository tree (all branches) and script CLIs.

Legend: ⚠ = exists only on another branch, or not yet built/verified. Financial integration, risk scoring and optimisation are future work pending PNB approval and are not implemented.

## 1. End-to-end system architecture

```mermaid
flowchart TD
    subgraph SRC["Raw sources"]
        ICR["ICRISAT district crop stats"]
        IMD["IMD 0.25° rainfall"]
        ERA["ERA5-Land climate"]
        PNBF["PNB loan layout<br/>(20 fields, private)"]
        IND["India Post pincode directory"]
    end

    subgraph AGR["Agriculture layer (Issues 1-2)"]
        I1["CropLongFormatv1/<br/>crop_restructure_validate.py"]
        O1[("crop_long_v1.csv<br/>merged_cleaned_v1f.csv")]
        I2["CropModelEligibility/"]
        O2[("crop_model_eligibility_v1.csv<br/>crop_observation_coverage_summary_v1.csv")]
    end

    subgraph WX["Weather layer (Issue 3)"]
        I3["weather_feature_engineering.py"]
        I3b["weather_forward_chaining.py"]
        O3[("weather_stress_features_v1.csv<br/>weather_anomalies_v1.csv<br/>weather_anomalies_forward_chaining_v1.csv<br/>district_weather_normals_*.csv")]
    end

    subgraph YLD["Yield layer (Issue 4)"]
        I4["yield_feature_engineering.py"]
        O4[("yield_anomalies_v1.csv<br/>yield_anomalies_forward_chaining_v1.csv<br/>district_crop_yield_normals_*.csv")]
    end

    subgraph ANA["Analytics (Issues 5-6)"]
        I5["⚠ crop specific weather feature selection/<br/>feature_selection.py (branch issue-5)"]
        O5[("crop_feature_selection_*.csv<br/>⚠ crop_feature_selection_v1.csv missing")]
        I6["District Climate Zonal Clustering/<br/>district_climate_clustering.py"]
        O6[("district_climate_profiles_v1.csv<br/>district_zone_map_v1.csv<br/>clustering_evaluation_v1.csv")]
    end

    subgraph FIN["Financial data foundation (Issues 23-25)"]
        F0["financial_data_requirements/<br/>data dictionary + linking stance"]
        F1["financial_data_validation/<br/>validate_loan_portfolio.py<br/>+ loan_portfolio_schema_v1.json"]
        F2["address_expansion/<br/>build_pincode_reference.py<br/>expand_addresses.py"]
        F3["synthetic_portfolio/<br/>generate_synthetic_portfolio.py<br/>verify_synthetic_portfolio.py"]
        FO[("pincode_reference_v1.csv<br/>address_expanded.csv<br/>synthetic_portfolio_v1.csv<br/>validation_report.csv")]
    end

    subgraph FUT["Future, pending PNB approval"]
        J["Join on district_code + year/season"]
        R["Portfolio weather-risk analytics"]
        D["Dashboard / UAT / deployment"]
    end

    ICR --> I1 --> O1 --> I2 --> O2
    IMD --> I3
    ERA --> I3
    I3 --> I3b
    I3 --> O3
    I3b --> O3
    O1 --> I4
    O2 --> I4
    I4 --> O4
    O3 --> I5
    O4 --> I5
    I5 --> O5
    O3 --> I6 --> O6

    IND --> F2
    PNBF --> F0
    PNBF --> F1
    F1 --> F2
    F2 --> F3
    F1 --> FO
    F2 --> FO
    F3 --> FO

    O3 --> J
    O4 --> J
    O5 --> J
    O6 --> J
    FO -->|"pincode → district_code"| J
    J --> R --> D
```

## 2. Weather and yield pipeline with anomaly methods

```mermaid
flowchart LR
    W["Raw weather<br/>13 districts × 2015-2019"] --> FE["13 indicators × 6 windows<br/>= 78 features"]
    FE --> N1["LOYO normals"]
    FE --> N2["Forward-chaining normals<br/>(≥2 earlier years)"]
    N1 --> A1["weather_anomalies_v1"]
    N2 --> A2["weather_anomalies_forward_chaining_v1"]
    C["crop_long + eligibility"] --> Y["Yield anomaly<br/>(district × crop × year)"]
    Y --> YA["yield_anomalies_forward_chaining_v1<br/>status = calculated only"]
    A2 --> FS["Issue 5: Spearman + BH-FDR<br/>+ collinearity filter<br/>LOYO folds, max 4/crop-fold<br/>consensus ≥ 2 folds"]
    YA --> FS
    FS --> P["Stable crop predictors<br/>(exploratory association only)"]
    FE --> CL["Issue 6: 54 climate features<br/>→ K=3 clustering"]
    CL --> Z["district_zone_map_v1<br/>(climate zones, not risk levels)"]
```

Notes:
- LOYO excludes only the held-out year and can use later years. Forward chaining uses strictly earlier years. Issue 5 combines forward-chaining anomalies with LOYO folds, so it is not prospective forecasting.
- Selected predictors are statistical associations, not causal drivers.

## 3. Financial data and the district-key bridge

```mermaid
flowchart TD
    L["Loan record<br/>zone, circle, branch, pincode,<br/>address, DPD, asset class ..."] --> V["Validate vs JSON schema<br/>(DPD band ↔ asset class,<br/>overdue ≤ outstanding, unique account)"]
    V --> AE["Split address into 4 parts<br/>extract address pincode<br/>compare city vs pincode district"]
    PIN["pincode_reference_v1.csv<br/>(India Post → 13 AP district_codes)"] --> AE
    AE --> M{"Pincode maps<br/>to a district?"}
    M -->|yes, labelled| DK["district_code"]
    M -->|no / conflict| U["Flagged, never guessed"]
    DK --> WX2["Weather anomalies<br/>climate zone<br/>yield anomalies"]
    SYN["Synthetic generator<br/>(AP-only, config + ruleset)"] -.->|"dev/test only, labelled synthetic"| V
    SEC["data/private/ (not committed)"] -.-> L
```

## 4. Repo and branch workflow

```mermaid
gitGraph
    commit id: "Main"
    branch dev
    commit id: "integration"
    branch issue-3-weather-feature-engineering
    branch issue-4-yield-feature-engineering
    branch issue-5-crop-feature-selection
    branch issue-6-district-climate-zoning
    branch issue-23-address-expansion
    branch issue-24-loan-portfolio-validation
    branch issue-25-synthetic-portfolio
    checkout dev
    merge issue-3-weather-feature-engineering
    merge issue-4-yield-feature-engineering
```

Illustrative only: which branches are merged into `dev` was not verified.

## Known discrepancies

- Issue 5 (`crop specific weather feature selection/`) lives on `issue-5-crop-feature-selection`, not in every checkout.
- Issues 1, 2 and 6 were untracked in the author's local checkout. `CropLongFormatv1/` is duplicated under `Issues/` there.
- The root README mentions ML forecasting, risk scoring and QAOA/Gurobi optimisation. None of that exists in the repo yet.
- The README names `merged_cleaned_v1d.csv`; the repo contains `merged_cleaned_v1f.csv`.
