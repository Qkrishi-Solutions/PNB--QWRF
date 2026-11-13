# Financial Data Foundation: Speaking Script and Linking Stance

Companion to the deck *Financial Data Foundation* (14 slides). Part 1 is the script, with timings.
Part 2 is the final stance on how the merged agriculture and weather data connects to the financial data.
Part 3 is likely questions and short answers.

Total speaking time: about 12 minutes without questions.

---

# Part 1: Script

## Slide 1: Cover (0:30)
**Say:**
"Today I will walk through the financial data work we have completed so far. I will cover what PNB supplied, how we documented and analysed it, how we split addresses into structured parts, and how we turn a pincode into a district code. That district code is what connects a loan record to our weather and yield analysis for the 13 Andhra Pradesh districts. One caveat up front: everything I show is based on a temporary 25-row sample used for mapping, not real portfolio data."

## Slide 2: Why the loan data needs a district key (1:00)
**Say:**
"Our weather and yield work is organised by district. The loan layout from PNB has zone, circle, branch and pincode, but no district. So the pincode is the bridge. A loan record gives us a pincode, the pincode gives us a district code, and the district code connects to weather anomalies, climate zones and yield. The rule we follow throughout: every mapping is checked and labelled, never guessed."

## Slide 3: What PNB supplied, 20 fields (1:00)
**Say:**
"PNB supplied 20 fields. They fall into four groups. Location: zone, circle, branch ID, branch name and pincode. Borrower: customer ID, name, address and account number. Loan: account open date, facility type, scheme code, agri sector, sanction amount and interest rate. Performance: the reporting date, outstanding balance, overdue amount, days past due and asset class. We were given the layout, not a filled dataset. A temporary 25-row file lets us test the logic, and its values are not real portfolio data."

## Slide 4: The data dictionary (1:00)
**Say:**
"For every field the dictionary records a standard name, type, the format we observed, a description, a sensitivity rating, and whether we use it in analysis. The pincode row on the left is an example. Four fields hold personal data: customer ID, name, address and account number. Name and address are not needed for analysis, so they are excluded. IDs and account numbers are used only when pseudonymised. All definitions are proposals until PNB confirms them."

## Slide 5: The sample is internally consistent (1:00)
**Say:**
"We checked the sample against the rules a loan portfolio should satisfy. All of them held on all 25 rows. The asset class fits its days-past-due band, overdue never exceeds outstanding, overdue is above zero exactly when days past due is, and every account number is unique. The chart shows the asset class mix. It is a small temporary sample, so please do not read it as a real portfolio. The band definitions are the commonly published SMA and NPA buckets, and PNB needs to confirm the ones it uses."

## Slide 6: A one-date snapshot has limits (0:45)
**Say:**
"This slide is about honesty. The sample supports the field layout, a snapshot of exposure and asset class, and the rules every file must satisfy. It cannot show trends or seasonality, because there is one reporting date. It cannot be joined at district level, because there is no district field. And 25 rows cannot support statistical conclusions. That is why the next part matters: we have to create the district key."

## Slide 7: One address field becomes four parts (1:15)
**Say:**
"The borrower address arrives as one comma-separated field. In the sample it always has four parts: house or plot, area, locality and city. We split it into those parts. If an address has a different number of parts, we label it and leave it unparsed. If one six-digit number appears inside the text, we extract it as the address pincode. We also compare the city with the pincode's district and report consistent, conflict or not comparable. The example on the slide is synthetic."

## Slide 8: Building the pincode-to-district reference (1:15)
**Say:**
"The reference comes from the India Post pincode directory: 157,126 post-office rows. We collapse that to 19,300 unique pincodes, one row each. Four district spellings differ between India Post and our project, for example Anantapur and Anantapuramu, so we alias them. Then we map to the 13 project districts. The district codes are read from the project file, not typed in, because they are not in alphabetical order. The source is a community mirror of the government directory from June 2024, and we record its hash in the build report."

## Slide 9: Mapping results for Andhra Pradesh (1:15)
**Say:**
"Of 1,243 Andhra Pradesh pincodes, 1,207 map to exactly one district. That is 97.1 percent. 18 span two districts. 18 are shared with another state: 17 with Telangana and 1 with Puducherry. For those 36 we do not pick one. We keep the candidates and leave the district code blank. All 13 project districts appear in the reference."

## Slide 10: Every pincode gets a status (0:45)
**Say:**
"Every pincode ends with one of six statuses. Only 'mapped' fills a district code. Everything else is labelled and left unmapped, so nobody downstream mistakes a guess for a fact. It also lets us report coverage honestly."

## Slide 11: Results on the temporary sample (0:45)
**Say:**
"On the temporary Andhra Pradesh sample, all 25 addresses parsed, all 25 pincodes mapped to a project district, and all 13 districts are present. There were no conflicts between city and pincode district. Please note the sample's geography was rewritten to Andhra Pradesh only: zone is Hyderabad, and the three circles are placeholders. So this shows the pipeline works end to end. It is not evidence about real PNB data."

## Slide 12: Built to be traceable (0:45)
**Say:**
"Four things make this traceable. Eleven unit tests pass, using synthetic data only. The reference records its source and file hash. Personal data stays out of Git and out of reports. And joins are checked so a lookup cannot silently add or drop rows."

## Slide 13: Assumptions and limitations (1:00)
**Say:**
"We assume the four-part address layout holds, that the pincode is the borrower's or the branch's location, and that the directory's 13 districts match ours. Limitations: the pincode source is a June 2024 mirror and the licence needs confirming; 36 pincodes cannot map to a single district; we have only verified on placeholder and synthetic data; and a district match shows where the loan is recorded, not where the farm is."

## Slide 14: Next steps (0:45)
**Say:**
"Next: review and merge the dictionary update and the address expansion, validate every incoming file against the dictionary, build a clearly labelled synthetic Andhra Pradesh test portfolio, and re-run the mapping on real PNB data once access is approved. Next I will explain how all of this connects to the weather and agriculture data. Thank you."

---

# Part 2: Final stance, linking merged agriculture and weather data to financial data

*Recommended position for the project. It is based on the datasets as they exist today. Items marked "to confirm" depend on PNB or the project owner.*

## 2.1 The stance in one paragraph
We connect financial data to the merged agriculture and weather data **at the district level, through `district_code`**, using the pincode-to-district mapping. The 13 Andhra Pradesh districts are the common key across weather, yield, climate zones and, once mapped, the loan portfolio. Loan accounts are **aggregated to district (and later district × period)** before they are joined to weather and yield features. We do **not** link an individual loan to an individual crop or farm unless PNB's data identifies the crop, because the current layout does not. We treat the resulting links as **associations to be explored**, not proof that weather caused repayment outcomes.

## 2.2 The chain
```
Loan account (20 PNB fields)
   |  pincode (6-digit)  [validated]
   v
Pincode reference (India Post, 19,300 pincodes)
   |  status: mapped / ambiguous_* / outside_project_scope / not_in_reference / invalid
   v
district_code  (13 Andhra Pradesh districts)
   |
   +--> District climate zone          (Issue 6, grain: district)
   +--> Weather anomaly features       (Issue 3, grain: district x year x window x indicator)
   +--> Yield anomalies and crop data  (Issues 1, 2, 4, grain: district x year x crop)
   +--> Crop-specific weather predictors (Issue 5, grain: crop)
```
Issue #7 (the integrated ML-ready dataset) already joins crop-long data, weather anomalies, yield anomalies, selected predictors and climate zones on `district_code`, `year` and `crop`. The financial data attaches to the same `district_code`.

## 2.3 Join specification

| Dataset | Grain | Key to financial data | Join type | What we carry across |
|---|---|---|---|---|
| Loan portfolio (after mapping) | account | `district_code` via pincode | many accounts to one district | Aggregates only: account count, sanctioned and outstanding amounts, overdue share, asset-class counts |
| District climate zones (`district_zone_map_v1.csv`) | district (13 rows) | `district_code` | many-to-one | Zone label (a similarity group, not a risk level) |
| Weather anomalies (`weather_anomalies_forward_chaining_v1.csv`) | district × year × window × indicator | `district_code` + period | many-to-one after aligning period | Anomaly z-scores for selected features |
| Yield anomalies and crop data (`yield_anomalies_forward_chaining_v1.csv`, `crop_long_v1.csv`) | district × year × crop | `district_code` + period (+ crop only if loan data has it) | many-to-one at district level | District-level yield-anomaly summaries |

## 2.4 Decisions

1. **Join key.** `district_code` is the only key used to connect financial data to weather and agriculture. It comes from the pincode mapping, never from names typed by hand.
2. **Grain.** Aggregate loans to district first, then join. This keeps every join many-to-one and avoids multiplying rows. Before and after each join we check key uniqueness and row counts.
3. **Unmapped loans stay unmapped.** Accounts whose pincode is ambiguous (36 of 1,243 Andhra Pradesh pincodes), outside Andhra Pradesh, missing or invalid are reported in a coverage table with their exposure. We do not allocate them to a district by proportion or by guess.
4. **Crop linkage.** The loan layout has an `agri sector` category (for example Horticulture, Dairy) but no crop name or season. So crop-level weather and yield features are **not** joined to individual loans. At district level we use yield anomalies summarised across crops, or weighted by area only if the weighting rule is agreed. Crop-level linkage becomes possible only if PNB's data identifies the financed crop (to confirm).
5. **Time alignment.** Weather and yield are annual or seasonal (windows such as kharif, rabi, annual crop year). The sample has one reporting date and an account open date, which is not the disbursement date. We will only link by period once we have multiple reporting periods and an agreed rule for mapping financial dates to agricultural seasons (to confirm: calendar year, financial year, or crop season). Until then the link is a static district profile, not a time series.
6. **Anomaly method.** For any predictive use, only forward-chaining anomalies are used, so a target period never sees later years. LOYO is not used to claim forecasting ability.
7. **District definition.** All joins use the 13 pre-2022 Andhra Pradesh districts that match our weather and yield data. If PNB reports the current 26 districts, a documented crosswalk is needed first.
8. **Zones.** Climate zones are used as a grouping for comparison. A zone is never described as a high, medium or low risk category.

## 2.5 Output we will produce

A district-level portfolio and weather table, one row per `district_code` × reporting period (a single period for now):

| Group | Columns |
|---|---|
| Key | `district_code`, `district_name`, `reporting_period` |
| Portfolio | `n_accounts`, `sanction_total`, `outstanding_total`, `overdue_total`, `n_standard`, `n_sma0`, `n_sma1`, `n_sma2`, `n_npa`, `agri_share` |
| Coverage | `accounts_mapped`, `accounts_ambiguous`, `accounts_outside_scope`, `accounts_unmapped`, `mapped_share` |
| Context | `climate_zone`, selected weather anomaly features, yield-anomaly summaries |

Every run also writes validation checks: key uniqueness, join cardinality, row counts before and after, unmatched keys and missing-value counts.

## 2.6 What we will not claim
- That a weather anomaly caused a repayment outcome, delinquency or NPA. We report associations.
- That a feature not selected has no effect.
- That a district match shows where the farm is. It shows where the loan is recorded.
- That a climate zone is a risk level.
- That any score is a validated prediction until PNB approves definitions and validation.
- Statistical findings from 25 temporary rows.

## 2.7 What this approach needs before it can go further
- Real or properly masked PNB data with a district-level view of the portfolio, and several reporting periods.
- PNB's confirmation of the asset-class definitions, currency units, whether `pincode` is the borrower's or the branch's location, and the reporting calendar.
- If crop-level analysis is wanted, a field that identifies the financed crop or activity and the season.

## 2.8 Extension for the cyclone and insurance outcomes
These need data that neither the merged weather and agriculture dataset nor the current loan layout contains:
- **Cyclone events:** a table of event dates, duration and affected districts, keyed by `district_code` and date. It joins to the same district key.
- **Crop insurance:** enrolment, sum insured and claim fields per loan or per district. They join through the same mapping and are aggregated the same way.

Until those exist, we can describe the district portfolio and its weather context, but not repayment behaviour during cyclones or the effect of insurance.

---

# Part 3: Likely questions

**Why district and not individual loan or farm?**
The loan layout has no crop, no farm location and no district, and weather and yield are only available by district. Joining below district would mean inventing links.

**Why not allocate the ambiguous pincodes to the likely district?**
Because it creates false precision. They are 36 of 1,243 Andhra Pradesh pincodes, so we report them as a separate bucket and keep the coverage figure visible.

**Is 97.1% coverage the share of loans mapped?**
No. It is the share of Andhra Pradesh pincodes that map to exactly one district. The share of loans depends on the real portfolio, and we will report that separately.

**Does the sample prove the mapping works on PNB data?**
It proves the code works end to end. The sample geography is placeholder, so accuracy on real data is not yet shown.

**Can you tell us which districts are riskier?**
Not yet. A district's weather anomalies and a climate zone are context. A risk statement needs real repayment data over several periods and PNB-approved definitions.

**Why the June 2024 pincode source?**
It is a mirror of the government directory; the live source was reported unavailable when we built it. We recorded its hash and will refresh it, and confirm the licence, before sharing outside the project.

**What about the new 26 districts?**
Our data and the reference use the 13 districts. If PNB reports 26, we add a documented crosswalk first.

**What about the January loan peak and cyclone repayment?**
Both need disbursement dates over time, repayment history, and cyclone event data. The current sample has none of these, so they are future work and not covered by what we have built.
