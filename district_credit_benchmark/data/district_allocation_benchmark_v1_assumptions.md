# District allocation benchmark v1 — assumptions and provenance

## Purpose

This benchmark provides a traceable public-reference basis for allocating **synthetic accounts** across the 13-district Andhra Pradesh geography used by the QWeatherRiskF historical dataset. It does not represent actual PNB customers, accounts, borrowers, or branch-level records.

## Geography

The benchmark retains the 13 pre-4-April-2022 Andhra Pradesh districts:
Anantapur, Chittoor, East Godavari, Guntur, Krishna, Kurnool, Kadapa, Nellore, Prakasam, Srikakulam, Visakhapatnam, Vizianagaram, and West Godavari.

RBI's downloaded Table 3.9 uses the spellings `CUDDAPAH` for Kadapa and `VISHAKHAPATNAM` for Visakhapatnam. These are mapped to the project's canonical district names. The source labels are retained in the CSV.

## Public reference 1 — NABARD

- **Source:** NABARD, *Andhra Pradesh State Focus Paper 2018–19*.
- **Table:** Annexure I, district-wise sector/sub-sector-wise SFP projections 2018–19, printed pages 127–128.
- **Measure:** `Total Agriculture (A+B+C)` — projected credit potential, not actual disbursement or outstanding credit.
- **Unit:** ₹ lakh.
- **Source year:** 2018–19.
- **URL:** https://www.nabard.org/auth/writereaddata/tender/1005184203sfp%202018-19%20andhrapradesh.pdf

The 13 district values are retained from the NABARD source and the allocation share is calculated as:

`benchmark_share_d = district_NABARD_Total_Agriculture_projection_d / sum_of_13_district_projections`

The shares sum to 1 (subject to CSV display precision).

## Public reference 2 — RBI

- **Source:** Reserve Bank of India, Database on Indian Economy (DBIE).
- **Publication:** Quarterly Basic Statistical Return (BSR)-1 on Credit by Scheduled Commercial Banks (excluding Regional Rural Banks).
- **Table:** Table No. 3.9 — District-wise classification of outstanding credit of scheduled commercial banks according to detailed occupation.
- **Reference date:** 31 March 2019.
- **State filter:** Andhra Pradesh.
- **Major sector:** `I. AGRICULTURE`.
- **Sub-sectors included:** `1. Direct Finance` and `2. Indirect Finance`.
- **Measure:** Outstanding credit.
- **Original unit:** ₹ thousands.
- **Unit conversion:** ₹ crore = ₹ thousands / 10,000.
- **Portal:** https://data.rbi.org.in
- **Navigation:** Publications → Time-Series Publications → Quarterly Basic Statistical Return (BSR)-1 on Credit by Scheduled Commercial Banks (excluding Regional Rural Banks) → Quarterly BSR-1: Outstanding Credit of Scheduled Commercial Banks → Section 3: Occupation-wise classification of credit → Table No. 3.9.

### RBI aggregation method

The downloaded Excel workbook contains detailed occupation/activity rows. For each Andhra Pradesh district, the RBI value in the CSV is the sum of all numeric outstanding-credit rows under `I. AGRICULTURE`, across both Direct Finance and Indirect Finance.

The CSV preserves:
- Direct Finance amount (₹ thousands)
- Indirect Finance amount (₹ thousands)
- Combined agriculture amount (₹ thousands)
- Combined agriculture amount converted to ₹ crore

The combined figure is a derived aggregation from the detailed occupation rows, not a separately published district aggregate row. No imputation or zero-filling is used.

### RBI coverage limitation

**Coverage gap:** Regional Rural Bank agricultural credit is not included because the selected quarterly BSR-1 Table 3.9 publication excludes RRBs. This missing coverage is not treated as zero and is not imputed.

This quarterly BSR-1 publication covers Scheduled Commercial Banks **excluding Regional Rural Banks (RRBs)**. Therefore, these values must not be described as total agricultural credit across all banking institutions or as the annual BSR series including RRBs.

RBI's official description of the quarterly BSR-1 confirms that it covers SCBs excluding RRBs and collects district, occupation/activity, number of accounts, credit limit, and outstanding credit. The RBI's March 2019 annual BSR release is a different publication and includes RRBs; this benchmark uses the quarterly Table 3.9 workbook actually downloaded and inspected.

## Assumptions register

| Assumption | Tag | Explanation |
|---|---|---|
| Retain the pre-April-2022 13-district geography | public reference | Matches the project's historical district universe. RBI source district labels are mapped to the project's canonical names. |
| Use NABARD Total Agriculture (A+B+C) | public reference | District-wise 2018–19 projected agricultural credit potential. |
| Treat NABARD values as projections, not actual credit flow | public reference | The source identifies these as SFP projections/credit potential. |
| Use RBI Table 3.9 at 31 March 2019 | public reference | Official DBIE quarterly BSR-1 district-wise detailed occupation table. |
| Aggregate RBI Direct + Indirect Finance under Agriculture | synthetic choice | This is the explicit aggregation rule applied to detailed occupation rows to form one district-level agriculture measure. |
| Convert RBI ₹ thousands to ₹ crore by dividing by 10,000 | synthetic choice | Unit conversion only; original amounts are retained in the CSV for auditability. |
| Calculate district benchmark shares from NABARD values | synthetic choice | Converts public projection amounts into relative allocation weights. |
| Use shares only to guide synthetic account allocation | synthetic choice | Does not reveal actual customer/account distribution. |
| Do not equate NABARD potential with RBI outstanding credit | synthetic choice | They are different measures and must remain separate. |

## Limitations

1. NABARD values represent assessed credit potential, not actual disbursement, outstanding balances, or account counts.
2. RBI values represent outstanding credit reported by Scheduled Commercial Banks excluding RRBs in the quarterly BSR-1 publication.
3. RBI values are aggregated from detailed occupation/activity rows; they are not a direct NABARD-equivalent credit-potential measure.
4. The two source measures must not be added together or treated as interchangeable.
5. The 2018–19 / 31 March 2019 benchmark is historical and must not be described as current lending distribution.
6. These weights are a transparent synthetic allocation choice, not a validated estimate of PNB's customer distribution.
7. No real PNB customer, account, borrower, or transaction data may be added.

## Validation status

- District rows: 13
- Distinct canonical districts: 13
- NABARD numeric values: 13
- RBI district records: 13
- RBI source date: 31 March 2019
- RBI source coverage: SCBs excluding RRBs
- RBI agriculture sub-sectors: Direct Finance + Indirect Finance
- Missing RBI district totals: 0
- Allocation share sum: 1 (subject to CSV precision)
- No silent imputation: maintained
