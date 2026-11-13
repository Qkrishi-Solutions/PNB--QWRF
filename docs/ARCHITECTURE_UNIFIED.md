# QWeatherRiskF: Unified Architecture

One diagram covering the whole project: sources, built layers (Issues 1-6, 23-25), and future stages.

Legend: solid = implemented in repo; ⚠ = on another branch or has a gap; dashed/grey "Future" = not built, pending PNB approval.

```mermaid
flowchart TD
    %% ---------- Sources ----------
    subgraph SRC["1. Sources"]
        ICR["ICRISAT crop stats"]
        IMD["IMD rainfall"]
        ERA["ERA5-Land climate"]
        PNBF["PNB loan layout<br/>(private, 20 fields)"]
        IND["India Post pincode directory"]
    end

    %% ---------- Agriculture ----------
    subgraph AGR["2. Agriculture (Issues 1-2)"]
        I1["crop_restructure_validate.py"]
        O1[("crop_long_v1.csv")]
        I2["Eligibility assessment"]
        O2[("crop_model_eligibility_v1.csv")]
    end

    %% ---------- Weather ----------
    subgraph WX["3. Weather (Issue 3)"]
        I3["weather_feature_engineering.py<br/>13 indicators x 6 windows = 78"]
        I3b["weather_forward_chaining.py"]
        O3[("weather_stress_features_v1.csv<br/>weather_anomalies_*_v1.csv")]
    end

    %% ---------- Yield ----------
    subgraph YLD["4. Yield (Issue 4)"]
        I4["yield_feature_engineering.py<br/>LOYO + forward chaining"]
        O4[("yield_anomalies_*_v1.csv")]
    end

    %% ---------- Analytics ----------
    subgraph ANA["5. Analytics (Issues 5-6)"]
        I5["⚠ feature_selection.py (branch issue-5)<br/>Spearman + BH-FDR + collinearity, LOYO folds"]
        O5[("crop_feature_selection_*.csv<br/>⚠ _v1.csv name missing")]
        I6["district_climate_clustering.py<br/>54 features, K=3"]
        O6[("district_zone_map_v1.csv<br/>district_climate_profiles_v1.csv")]
    end

    %% ---------- Financial ----------
    subgraph FIN["6. Financial foundation (Issues 23-25)"]
        F1["validate_loan_portfolio.py<br/>+ JSON schema"]
        F2["address_expansion<br/>split address, extract pincode"]
        F3["synthetic_portfolio<br/>(AP-only, dev/test, labelled synthetic)"]
        PIN[("pincode_reference_v1.csv")]
        DK(["district_code<br/>(pincode bridge)"])
    end

    %% ---------- Future ----------
    subgraph FUT["7. Future (pending PNB approval)"]
        J["Join: district_code + season/year"]
        R["Portfolio weather-risk analytics"]
        D["Dashboard, UAT, deployment"]
    end

    %% ---------- Edges ----------
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

    PNBF --> F1 --> F2
    IND --> PIN --> F2
    F3 -.->|"test input only"| F1
    F2 --> DK

    O3 --> J
    O4 --> J
    O5 --> J
    O6 --> J
    DK --> J
    J -.-> R -.-> D

    %% ---------- Styling ----------
    classDef done fill:#e6f4ea,stroke:#2e7d32,color:#111;
    classDef gap fill:#fff4e5,stroke:#ef6c00,color:#111;
    classDef future fill:#eeeeee,stroke:#888,stroke-dasharray:4 3,color:#444;
    class I1,O1,I2,O2,I3,I3b,O3,I4,O4,I6,O6,F1,F2,F3,PIN,DK,ICR,IMD,ERA,PNBF,IND done;
    class I5,O5 gap;
    class J,R,D future;
```

## Reading notes

- Weather and yield are joined by district and year. The financial side connects only through `district_code`, derived from the loan pincode.
- Climate zones group similar districts. They are not risk levels.
- Issue 5 outputs are exploratory associations (13 districts, 3 scorable years), not a validated prediction model or proof of causation.
- LOYO uses later years in training; forward chaining uses only earlier years. Issue 5 mixes forward-chaining anomalies with LOYO folds, so it is not prospective forecasting.
- Everything in section 7 is planned work. No real PNB financial data is in the repo.
