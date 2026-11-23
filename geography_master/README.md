# Geography Master Data Layer (Issue #42)

## 1. Overview
The **Geography Master** provides a unified geographic reference for the QWeatherRiskF project. It connects India Post pincodes, administrative banking divisions of Punjab National Bank (PNB), and climate zonal clustering into a single relational schema.

This connects:
- **Banking / Loan Portfolio Data:** Keyed by `pincode`, `pnb_zone_provisional` (Hyderabad), and `pnb_circle_provisional` (Tirupati, Vijayawada, Visakhapatnam).
- **Agricultural & Meteorological Data:** Keyed by `district_code` (13 historical districts of Andhra Pradesh, codes 502–523).
- **Climate Zonal Data:** Keyed by `climate_zone` (Climate Zone 1, 2, 3).

> **Important Note on PNB Administrative Divisions:**
> The banking hierarchy columns (`pnb_zone_provisional`, `pnb_circle_provisional`) are derived from the synthetic loan portfolio generator configuration (`circle_of_district`) and are explicitly designated as **provisional placeholders**. They do not constitute an audited, official PNB administrative directory. When official PNB administrative circulars or branch master datasets become available, these provisional columns will be reconciled against official bank sources.

---

## 2. Deliverables & File Manifest

| File | Rows | Description |
|---|:---:|---|
| `geography_master_v1.csv` | 19,300 | Full pincode-grain master table covering all India Post pincodes with canonical AP district codes, provisional PNB circles/zones, and climate zones. Integer codes are formatted as nullable integers (`502`, not `502.0`). |
| `district_geography_master_v1.csv` | 13 | Canonical 13-district summary with mapped pincode counts, provisional banking circles, and climate cluster assignments. |
| `geography_master_coverage_report_v1.csv` | 18 | Summary metrics covering mapping status breakdown, join integrity, district distribution, and cross-dataset consistency audit. |
| `unmatched_ambiguous_pincodes_v1.csv` | 64 | Audit of ambiguous pincodes spanning district or state boundaries (unassigned, not guessed). |
| `build_geography_master.py` | — | Deterministic script generating the geography master, enforcing cross-dataset validation, and generating audit tables. |
| `tests/test_geography_master.py` | — | Automated regression and validation tests enforcing acceptance criteria. |

---

## 3. Geographic & Administrative Hierarchy

```text
[All-India Pincode (19,300)]
         │
         ├── outside_project_scope (18,029) ──> Non-AP states (Unmapped)
         ├── ambiguous_district / state (64) ──> Multi-boundary postal zones (Unassigned, Not Guessed)
         │
         └── mapped (1,207 AP Pincodes)
                 │
                 ▼
      [district_code (502 to 523)] ── 13 Canonical Districts
                 │
                 ├── [PNB Administrative Zone]   ──> Hyderabad (Provisional)
                 ├── [PNB Administrative Circle] ──> Tirupati | Vijayawada | Visakhapatnam (Provisional)
                 └── [Climate Zonal Cluster]     ──> Climate Zone 1 | Climate Zone 2 | Climate Zone 3
```

### The 13 Canonical Districts of Andhra Pradesh

| District Code | Canonical District Name | PNB Zone (Provisional) | PNB Circle (Provisional) | Climate Zone (Issue 6) | Mapped Pincodes |
|:---:|---|---|---|---|:---:|
| **502** | ANANTAPURAMU | Hyderabad | Tirupati | Climate Zone 3 | 93 |
| **503** | CHITTOOR | Hyderabad | Tirupati | Climate Zone 1 | 99 |
| **504** | YSR | Hyderabad | Tirupati | Climate Zone 3 | 83 |
| **505** | EAST GODAVARI | Hyderabad | Visakhapatnam | Climate Zone 2 | 126 |
| **506** | GUNTUR | Hyderabad | Vijayawada | Climate Zone 3 | 112 |
| **510** | KRISHNA | Hyderabad | Vijayawada | Climate Zone 2 | 118 |
| **511** | KURNOOL | Hyderabad | Tirupati | Climate Zone 3 | 85 |
| **515** | SPR NELLORE | Hyderabad | Tirupati | Climate Zone 1 | 81 |
| **517** | PRAKASAM | Hyderabad | Vijayawada | Climate Zone 3 | 101 |
| **519** | SRIKAKULAM | Hyderabad | Visakhapatnam | Climate Zone 2 | 63 |
| **520** | VISAKHAPATNAM | Hyderabad | Visakhapatnam | Climate Zone 2 | 83 |
| **521** | VIZIANAGARAM | Hyderabad | Visakhapatnam | Climate Zone 2 | 52 |
| **523** | WEST GODAVARI | Hyderabad | Vijayawada | Climate Zone 2 | 111 |

---

## 4. Cross-Dataset Consistency & Validation Audit

The build script `build_geography_master.py` programmatically inspects and enforces `district_code` and `district_name` consistency across every foundational dataset in the repository. The build fails if any dataset has a missing, extra, or mismatched district code:

| Dataset Name | Path | Total Rows | District Code Alignment |
|---|---|:---:|:---:|
| `weather_anomalies_v1.csv` | `weather_feature_engineering/` | 25,350 | **13/13 Canonical Match** |
| `district_weather_normals_v1.csv` | `weather_feature_engineering/` | 5,070 | **13/13 Canonical Match** |
| `yield_anomalies_v1.csv` | `yield_feature_engineering/` | 1,430 | **13/13 Canonical Match** |
| `district_crop_yield_normals_v1.csv` | `yield_feature_engineering/` | 1,430 | **13/13 Canonical Match** |
| `district_zone_map_v1.csv` | `District Climate Zonal Clustering/` | 13 | **13/13 Canonical Match** |
| `pincode_reference_v1.csv` | `address_expansion/` | 19,300 | **13/13 Canonical Match** |

---

## 5. Join Integrity & Non-Guessing Policy

1. **Zero Row Multiplication:**
   * Joins are verified strictly as **many-to-one** (many pincodes map to one district).
   * Total row count remains invariant at exactly **19,300 rows**.
2. **Ambiguous Postal Boundaries Policy:**
   * 18 pincodes span across two AP districts; 46 pincodes span state borders.
   * In compliance with project rules (*No silent imputation / Missing is not zero*), these 64 records have `district_code = blank (NaN)` and are catalogued in `unmatched_ambiguous_pincodes_v1.csv` with their candidate `source_districts`.
3. **Integer Serialisation:**
   * All integer identifiers (`district_code`, `state_code`, `cluster_id`, `pincode`, `n_source_districts`, `n_states`) are cast to nullable integers (`Int64`) to prevent floating-point strings like `502.0` in output CSV files.
4. **Mandatory Field Assertions:**
   * Every record with `mapping_status == 'mapped'` is strictly asserted to contain non-null `district_code`, `pnb_circle_provisional`, and `climate_zone`.

---

## 6. Execution & Verification Commands

```bash
# Generate the geography master deliverables and verify cross-dataset consistency:
python geography_master/build_geography_master.py

# Run acceptance verification tests:
python -m unittest geography_master/tests/test_geography_master.py
```
