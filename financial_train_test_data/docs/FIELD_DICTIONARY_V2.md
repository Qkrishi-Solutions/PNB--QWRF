# Field Dictionary v2

> **Status:** draft for PNB review. Nothing here is a confirmed PNB definition. No real PNB data is in this repository; all example values come from the PNB field layout, public data or synthetic files.
>
> **Machine-readable source of truth:** `field_dictionary_v2.csv` (this file is generated from it). The earlier dictionaries (`PNB_FIELD_DICTIONARY_DRAFT.md`, `DATA_DICTIONARY_V2` to `V4`, `synthetic_training_v5_data_dictionary.csv`) are unchanged.

## 1. Scope and counts

| Group | Fields | What it is |
|---|---|---|
| `pnb_loan` | 20 | The 20 loan fields supplied by PNB (field layout of the 25-row sample). |
| `pnb_insurance_supplied` | 3 | The 3 insurance fields supplied by PNB. |
| `pnb_insurance_not_supplied` | 10 | Insurance fields **named in `PNB_requirements.md` but not supplied**. Listed so nobody invents them. |
| `geography` | 4 | Project geography fields derived from pincode and public data. |
| `severe_weather_pending` | 5 | Placeholders pending issues #45 and #46. |
| `derived_weather_yield` | 9 | Weather and yield anomaly fields used in the synthetic training files. |
| `derived_other` | 4 | Other fields derived by the platform. |
| **Total** | **55** | |

### Column meanings (CSV)

| Column | Meaning |
|---|---|
| `field_name` | Standard name used by the platform. |
| `source_field_name` | Name as received from PNB (blank for non-PNB fields). |
| `group`, `status` | Group above; `supplied`, `named_not_supplied`, `derived` or `pending_dependency`. |
| `type`, `unit`, `allowed_values`, `nullable` | Format. "Observed in sample" values are not confirmed domains. Not-supplied and pending fields say "not defined" instead of guessing. |
| `sensitivity` | Low, Medium or High (personal). Values marked "(proposed)" are this project's suggestion, not PNB's. |
| `source` | `PNB` (supplied layout), `public`, `synthetic`, `derived` (computed by the platform from other fields), `not supplied`, or `public (ERA5 or IMD, planned)`. |
| `leakage_role` | Role in modelling, using the legend in section 7. |
| `notes` | Definition, caveats and what must be confirmed. |

## 2. PNB loan fields (20, supplied)

| field_name | source_field_name | type | allowed_values | sensitivity | leakage_role | notes |
|---|---|---|---|---|---|---|
| `reporting_date` | reporting_date | date | valid calendar date | Low | feature_raw_date | Date as at which balances and status are reported. Single value in the PNB sample. |
| `zone` | zone | text | PNB zone name; sample has one value (Hyderabad) | Low | feature | Administrative zone. Observed in the 25-row sample only; full list to confirm with PNB. |
| `circle` | circle | text | PNB circle name; sample has 3 placeholder values | Low | feature | One circle spans several districts. Real circle list to confirm with PNB. |
| `branch_id` | branch ID | text | pattern AA999 (sample) | Low | identifier_exclude | High cardinality; not used as a model feature. |
| `branch_name` | branch name | text | free text | Low | identifier_exclude | Optional descriptive field. |
| `customer_id` | customer ID | text | pattern AAAA99999 (sample) | High (personal) | identifier_exclude | Pseudonymise before any analysis; never used as a feature. |
| `customer_name` | customer name | text | free text | High (personal) | identifier_exclude | Exclude from analysis. |
| `borrowers_address` | borrowers address | text | free text; four comma-separated parts in sample (house, area, locality, city) | High (personal) | identifier_exclude | Exclude from analysis. |
| `pincode` | pincode | integer (6 digits) | 6-digit PIN starting 1-9 | Medium | geo_id | Location key; mapped to a project district (see pincode_mapping_status). Not a model feature. |
| `account_ref` | account number | integer (11 digits) | 11-digit number, unique per row | High (personal) | identifier_exclude | Pseudonymise before any analysis. |
| `account_open_date` | account open date | date | valid date on or before reporting_date | Low | feature_raw_date | Not the same as disbursement date. |
| `facility_type` | facility type | text | sample: Cash Credit, Term Loan, Kisan Credit Card, Working Capital, Overdraft | Low | feature | Full list to confirm with PNB. |
| `scheme_code` | scheme code | text | sample: AGRI01, AGRI09, KCC02, MSME05, TL101, WC204 | Low | feature | Full list to confirm with PNB. |
| `agri_sector` | agri sector | text | sample: Crop Production, Dairy & Animal Husbandry, Farm Mechanization, Horticulture, Irrigation, Poultry, Non-Agri | Low | feature | Full list to confirm with PNB. Fisheries appears only in the synthetic data. |
| `sanction_amount` | sanction amount | integer | >= 0 | Medium | feature | Currency and unit to confirm with PNB. |
| `interest_rate` | rate of interest | decimal | sample range 7.22 to 12.59 | Low | feature | Unit to confirm with PNB. |
| `outstanding_balance` | balance outstanding | decimal (2 dp) | >= 0 | Medium | feature | Amount outstanding at reporting_date. Currency to confirm with PNB. |
| `overdue_amount` | overdue amount | decimal | >= 0; > 0 only when days_past_due > 0 (sample) | Medium | leakage_exclude | Determined by the asset class (target leakage); exclude as a model feature. |
| `days_past_due` | days_past_due | integer | >= 0; sample range 0 to 204 | Medium | leakage_exclude | Determined by the asset class (target leakage); exclude as a model feature. |
| `asset_class` | asset class | text (ordered) | Standard, SMA 0, SMA 1, SMA 2, NPA | Medium | target | Label. DPD bands per class to confirm with PNB. |

Currency, rate and unit conventions are not stated by PNB. Amounts are INR only in the synthetic dictionaries.

## 3. Insurance fields

### 3a. Supplied by PNB (3)

| field_name | type | allowed_values | nullable | sensitivity | leakage_role | notes |
|---|---|---|---|---|---|---|
| `crop_insurance_available` | text | Y, N | no | Medium | feature | Whether insurance exists for the loan. Allowed values taken from the PNB layout; to confirm with PNB. |
| `insurance_start_date` | date | valid date; blank when crop_insurance_available = N | yes (blank when not insured) | Medium | feature_raw_date | Policy commencement date. Blank is not zero and is not imputed. |
| `insurance_end_date` | date | valid date on or after insurance_start_date; blank when not insured | yes (blank when not insured) | Medium | feature_raw_date | Policy expiry date. Blank is not zero and is not imputed. |

### 3b. Named in `PNB_requirements.md` but NOT supplied (10)

These have no type, unit or allowed values because PNB has not provided them. Do not generate, infer or backfill them as if they were PNB data. Sensitivity and leakage roles are proposals for PNB to confirm.

| field_name | sensitivity | leakage_role | notes |
|---|---|---|---|
| `insurance_policy_id` | High (identifier, proposed) | undecided until supplied | Unique insurance policy identifier. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_scheme` | Low (proposed) | undecided until supplied | Insurance scheme or product. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_amount` | Medium (proposed) | undecided until supplied | Total insured amount. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_premium` | Medium (proposed) | undecided until supplied | Premium paid. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_provider` | Low (proposed) | undecided until supplied | Insurance provider. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_coverage_status` | Medium (proposed) | undecided until supplied | Active, expired, pending, or other validated status. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_claim_status` | Medium (proposed) | post_event_exclude (proposed) | Claim status; arises after a loss event, so excluded for early-warning use. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_claim_amount` | Medium (proposed) | post_event_exclude (proposed) | Amount claimed; arises after a loss event. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_payout_amount` | Medium (proposed) | post_event_exclude (proposed) | Amount received; arises after a loss event. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_payout_date` | Medium (proposed) | post_event_exclude (proposed) | Date of payout; arises after a loss event. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |

## 4. Geography fields (4)

| field_name | type | allowed_values | nullable | source | leakage_role | notes |
|---|---|---|---|---|---|---|
| `district_code` | integer | project district code; 13 districts (502, 503, 504, 505, 506, 510, 511, 515, 517, 519, 520, 521, 523) | yes (blank unless pincode_mapping_status = mapped) | public | feature | Project district code from the crop/weather datasets. Derived from pincode via the India Post directory (address_expansion/). A project key, not a PNB field. |
| `district_name` | text | 13 project district names (upper case) | yes (blank unless mapped) | public | feature | Project spelling; India Post spellings are normalised (e.g. ANANTAPUR to ANANTAPURAMU). |
| `climate_zone` | text | Climate Zone 1, Climate Zone 2, Climate Zone 3 | yes (blank when district unmapped) | public | feature | Issue 6 clustering group of districts with similar climate. A similarity group, not a risk level; numbers are labels, not rankings. |
| `pincode_mapping_status` | text | mapped, ambiguous_district, ambiguous_state, outside_project_scope, not_in_reference, invalid_pincode, ap_district_not_in_project | no | public | geo_id | Outcome of the pincode-to-district mapping (address_expansion/). Only 'mapped' fills district_code; other statuses are reported, never guessed. |

## 5. Severe-weather fields (pending #45 and #46)

Issues #45 (extended meteorological ingestion) and #46 (cyclone event catalogue) are open and have produced no output files yet. These rows record only what those issues name. Field names, types and units must be taken from their data dictionaries once they exist, and this dictionary updated then. Units must come from source metadata, not be assumed.

| field_name | status | source | leakage_role | notes |
|---|---|---|---|---|
| `pressure` | pending_dependency | public (ERA5 or IMD, planned) | undecided until #45 is defined | PROVISIONAL placeholder: field names, units and types are defined by the owning issue, not here. Named in issue #45 (extended meteorological ingestion). |
| `wind_direction` | pending_dependency | public (ERA5 or IMD, planned) | undecided until #45 is defined | PROVISIONAL placeholder: field names, units and types are defined by the owning issue, not here. Named in issue #45. Wind speed (wind_mean) already exists in Issue 3 features. |
| `sea_surface_temperature` | pending_dependency | public (ERA5 or IMD, planned) | undecided until #45 is defined | PROVISIONAL placeholder: field names, units and types are defined by the owning issue, not here. Named in issue #45 (Bay of Bengal grid points). |
| `cyclone_event` | pending_dependency | public (ERA5 or IMD, planned) | post_event_exclude (proposed) | PROVISIONAL placeholder: field names, units and types are defined by the owning issue, not here. Event catalogue from issue #46 (cyclone_events_v1.csv). An observed event, so post-event for early-warning use. |
| `cyclone_affected_district` | pending_dependency | public (ERA5 or IMD, planned) | post_event_exclude (proposed) | PROVISIONAL placeholder: field names, units and types are defined by the owning issue, not here. District-event map from issue #46 (cyclone_district_map_v1.csv); the affected-district rule is defined there. |

## 6. Derived fields

| field_name | type | allowed_values | nullable | source | leakage_role | notes |
|---|---|---|---|---|---|---|
| `wx_kharif__rain_sum_z` | float | real number or blank | yes (blank when unavailable; not imputed) | derived | feature | Forward-chaining anomaly z-score of kharif__rain_sum; blank when the baseline is unavailable. Not imputed. Association only, not causation. Source weather units are not defined. |
| `wx_kharif__longest_dry_z` | float | real number or blank | yes (blank when unavailable; not imputed) | derived | feature | Forward-chaining anomaly z-score of kharif__longest_dry; blank when the baseline is unavailable. Not imputed. Association only, not causation. Source weather units are not defined. |
| `wx_annual_crop__hot_days_z` | float | real number or blank | yes (blank when unavailable; not imputed) | derived | feature | Forward-chaining anomaly z-score of annual_crop__hot_days; blank when the baseline is unavailable. Not imputed. Association only, not causation. Source weather units are not defined. |
| `wx_kharif__tmax_mean_z` | float | real number or blank | yes (blank when unavailable; not imputed) | derived | feature | Forward-chaining anomaly z-score of kharif__tmax_mean; blank when the baseline is unavailable. Not imputed. Association only, not causation. Source weather units are not defined. |
| `wx_kharif__dtr_mean_z` | float | real number or blank | yes (blank when unavailable; not imputed) | derived | feature | Forward-chaining anomaly z-score of kharif__dtr_mean; blank when the baseline is unavailable. Not imputed. Association only, not causation. Source weather units are not defined. |
| `wx_ne__rainy_days_z` | float | real number or blank | yes (blank when unavailable; not imputed) | derived | feature | Forward-chaining anomaly z-score of ne__rainy_days; blank when the baseline is unavailable. Not imputed. Association only, not causation. Source weather units are not defined. |
| `crop` | text | crop names from crop_long_v1; blank for non-crop accounts | yes (blank unless Crop Production) | synthetic | feature | Synthetic crop assignment for Crop Production accounts; PNB layout has no crop field. |
| `yield_anomaly_zscore` | float | real number or blank | yes (blank when not calculated; not imputed) | derived | feature | Forward-chaining yield anomaly (Issue 4). Observed after harvest, so check timing before using it for early warning. Missing is not zero. |
| `yield_anomaly_status` | text | calculated, ineligible_observation, insufficient_history, not_applicable | no | derived | feature | Why a yield anomaly is or is not available. |
| `snapshot_year` | integer | 2015 to 2019 (synthetic) | no | derived | feature | Year of reporting_date; joins that year's weather. |
| `loan_age_days` | integer | >= 0 | no | derived | feature | Days from account_open_date to reporting_date. |
| `npa_flag` | integer | 0, 1 | no | derived | target_binary | 1 if asset_class is NPA. Derived from the target; never a feature. |
| `split` | text | train, test | no | synthetic | split | Out-of-time split by snapshot year in the synthetic files. |

## 7. Leakage and exclusion roles

| Role | Meaning |
|---|---|
| `feature` | Model input. |
| `feature_raw_date` | Raw date; prefer derived day-count features. |
| `identifier_exclude` | Identifier or personal field; never a feature. |
| `geo_id` | Location key; not a feature. |
| `leakage_exclude` | Determined by the target; exclude (target leakage). |
| `post_event_exclude` | Observed after the weather event; exclude for early-warning use. |
| `target` / `target_binary` | Label and its binary form; never a feature. |
| `split` | Train/test marker. |
| `undecided until ...` | Role cannot be set until PNB supplies or the owning issue defines the field. |

### Excluded as model features because of leakage or timing

| field_name | leakage_role | notes |
|---|---|---|
| `overdue_amount` | leakage_exclude | Determined by the asset class (target leakage); exclude as a model feature. |
| `days_past_due` | leakage_exclude | Determined by the asset class (target leakage); exclude as a model feature. |
| `asset_class` | target | Label. DPD bands per class to confirm with PNB. |
| `insurance_claim_status` | post_event_exclude (proposed) | Claim status; arises after a loss event, so excluded for early-warning use. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_claim_amount` | post_event_exclude (proposed) | Amount claimed; arises after a loss event. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_payout_amount` | post_event_exclude (proposed) | Amount received; arises after a loss event. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `insurance_payout_date` | post_event_exclude (proposed) | Date of payout; arises after a loss event. Named in PNB_requirements.md Objective 2; NOT supplied by PNB. Type, unit and allowed values must come from PNB, not be inferred. |
| `cyclone_event` | post_event_exclude (proposed) | PROVISIONAL placeholder: field names, units and types are defined by the owning issue, not here. Event catalogue from issue #46 (cyclone_events_v1.csv). An observed event, so post-event for early-warning use. |
| `cyclone_affected_district` | post_event_exclude (proposed) | PROVISIONAL placeholder: field names, units and types are defined by the owning issue, not here. District-event map from issue #46 (cyclone_district_map_v1.csv); the affected-district rule is defined there. |
| `npa_flag` | target_binary | 1 if asset_class is NPA. Derived from the target; never a feature. |

### Excluded as identifiers or location keys

`branch_id`, `branch_name`, `customer_id`, `customer_name`, `borrowers_address`, `pincode`, `account_ref`, `pincode_mapping_status`.

## 8. Caveats

- Association is not causation. Weather anomalies in this dictionary do not prove crop loss, repayment behaviour or NPA outcomes.
- Missing is not zero. Blank weather, yield and insurance values are never imputed.
- Climate zones are similarity groups, not risk levels.
- Insurance fields shown as "not supplied" must not be fabricated. If PNB supplies them, update the CSV and this file together.
- `PNB_requirements.md` is not tracked in this repository; the 13 insurance field names were read from a local copy of it (Objective 2).
