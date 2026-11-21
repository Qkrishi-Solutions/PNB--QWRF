# Geography Master Data Layer (Issue #42)

## 1. Overview
The **Geography Master** is the single authoritative source of geographic truth for the QWeatherRiskF project. It unifies the administrative banking hierarchy of Punjab National Bank (PNB), India Post pincode directory mappings, and climate zonal clustering into a single relational schema.

This bridges the gap between:
- **Banking / Loan Portfolio Data:** Keyed by `pincode`, `zone` (Hyderabad), and `circle` (Tirupati, Vijayawada, Visakhapatnam).
- **Agricultural & Meteorological Data:** Keyed by `district_code` (13 historical districts of Andhra Pradesh, codes 502–523).
- **Climate Zonal Data:** Keyed by `climate_zone` (Climate Zone 1, 2, 3).

---

## 2. Deliverables & File Manifest

| File | Rows | Description |
|---|:---:|---|
| `geography_master_v1.csv` | 19,300 | Full pincode-grain master table covering all India Post pincodes with canonical AP district codes, PNB circles/zones, and climate zones. |
| `district_geography_master_v1.csv` | 13 | Canonical 13-district summary with mapped pincode counts and administrative/climate assignments. |
| `geography_master_coverage_report_v1.csv` | 12 | Summary metrics covering mapping status breakdown, join integrity, and district distribution. |
| `unmatched_ambiguous_pincodes_v1.csv` | 64 | Audit of ambiguous pincodes spanning district or state boundaries (unassigned, not guessed). |
| `build_geography_master.py` | — | Deterministic script generating the geography master and audit tables. |
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
                 ├── [PNB Administrative Zone]  ──> Hyderabad (covers AP portfolio)
                 ├── [PNB Administrative Circle] ──> Tirupati | Vijayawada | Visakhapatnam
                 └── [Climate Zonal Cluster]    ──> Climate Zone 1 | Climate Zone 2 | Climate Zone 3
```

### The 13 Canonical Districts of Andhra Pradesh

| District Code | Canonical District Name | PNB Zone | PNB Circle | Climate Zone (Issue 6) | Mapped Pincodes |
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

## 4. Join Integrity & Non-Guessing Policy

1. **Zero Row Multiplication:**
   * Joins are verified strictly as **many-to-one** (many pincodes map to one district).
   * Total row count remains invariant at exactly **19,300 rows**.
2. **Ambiguous Postal Boundaries Policy:**
   * 18 pincodes span across two AP districts; 46 pincodes span state borders.
   * In compliance with project rules (*No silent imputation / Missing is not zero*), these 64 records have `district_code = blank (NaN)` and are catalogued in `unmatched_ambiguous_pincodes_v1.csv` with their candidate `source_districts`.

---

## 5. Execution & Verification Commands

```bash
# Generate the geography master deliverables:
python geography_master/build_geography_master.py

# Run acceptance verification tests:
python geography_master/tests/test_geography_master.py
```
