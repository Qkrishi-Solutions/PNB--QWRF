# PNB Financial Data - Draft Data Dictionary

**Status:** DRAFT for PNB review. Nothing here is a confirmed PNB definition.

**Source of this draft:** the 20-column field layout of the sample file `Financial_data_PNB.csv` (a temporary 25-row dataset used for mapping purposes, one reporting date, all rows in Andhra Pradesh). The field layout was received from PNB; the sample values are not treated as real portfolio data, and no sample row values are reproduced in this document.

**Conventions**
- "Observed" means seen in the 25-row sample only; it may not hold for the full extract.
- "To confirm" means PNB must define or approve it. Nothing is assumed.
- Currency, rate and unit conventions are not stated in the sample and are listed as open questions.
- Grain of the sample: one row per loan account as at a single reporting date.

## 1. Fields received (20)

| # | Field (as received) | Proposed standard name | Type (observed) | Observed format / domain | Description (proposed) | Sensitivity | Needed for analysis? |
|---|---|---|---|---|---|---|---|
| 1 | `reporting_date` | `reporting_date` | date | `YYYY-MM-DD`; single value in sample | Date as at which balances and status are reported | Low | Yes |
| 2 | `zone` | `zone` | text | 1 value in sample (Hyderabad) | PNB administrative zone (Hyderabad zone covers the Andhra Pradesh portfolio) | Low | Yes |
| 3 | `circle` | `circle` | text | 3 placeholder values in sample (Vijayawada, Visakhapatnam, Tirupati); each circle covers several districts | PNB administrative circle (one circle spans multiple districts; real circle list to be confirmed by PNB) | Low | Yes |
| 4 | `branch ID` | `branch_id` | text | Pattern `AA999`; 1:1 with branch name in sample | Branch identifier | Low | Yes |
| 5 | `branch name` | `branch_name` | text | Free text | Branch name | Low | Optional |
| 6 | `customer ID` | `customer_id` | text | Pattern `AAAA99999`; unique per account in sample | Customer identifier | **High** (personal) | Pseudonymised only |
| 7 | `customer name` | - | text | Free text | Borrower name | **High** (personal) | **No - exclude from analysis** |
| 8 | `borrowers address` | - | text | Free text | Borrower address | **High** (personal) | **No - exclude from analysis** |
| 9 | `pincode` | `pincode` | integer (6 digits) | 6-digit PIN; 14 distinct values in sample, all in the Andhra Pradesh range (515001-535001) | Postal code, presumed borrower or branch location | Medium | Yes (location) |
| 10 | `account number` | `account_ref` | integer (11 digits) | 11-digit number | Loan account number | **High** (personal) | Pseudonymised only |
| 11 | `account open date` | `account_open_date` | date | `YYYY-MM-DD` | Date the account was opened | Low | Yes (not the same as disbursement date) |
| 12 | `facility type` | `facility_type` | text | Cash Credit, Term Loan, Kisan Credit Card, Working Capital, Overdraft | Type of credit facility | Low | Yes |
| 13 | `scheme code` | `scheme_code` | text | AGRI01, AGRI09, KCC02, MSME05, TL101, WC204 | Loan scheme | Low | Yes |
| 14 | `agri sector` | `agri_sector` | text | Crop Production, Dairy & Animal Husbandry, Farm Mechanization, Horticulture, Irrigation, Poultry, Non-Agri | Agricultural activity category | Low | Yes |
| 15 | `sanction amount` | `sanction_amount` | integer | Whole numbers | Sanctioned limit or amount | Medium | Yes |
| 16 | `rate of interest` | `interest_rate` | decimal | 7.22 to 12.59 in sample | Interest rate (unit to confirm, presumably % per annum) | Low | Optional |
| 17 | `balance outstanding` | `outstanding_balance` | decimal (2 dp) | >= 0 | Amount outstanding at reporting date | Medium | Yes |
| 18 | `overdue amount` | `overdue_amount` | decimal | >= 0 | Amount overdue at reporting date | Medium | Yes |
| 19 | `days_past_due` | `days_past_due` | integer | 0 to 204 in sample | Days the account is past due at reporting date | Medium | Yes |
| 20 | `asset class` | `asset_class` | text | Standard, SMA 0, SMA 1, SMA 2, NPA | Asset classification at reporting date | Medium | Yes |

## 2. Observed consistency rules (to be confirmed by PNB)

These held for all 25 sample rows. They are candidate validation rules, not agreed definitions.

| Rule | Observed in sample |
|---|---|
| Asset class vs days past due | Standard = 0; SMA 0 = 12-30; SMA 1 = 40-60; SMA 2 = 78-90; NPA = 115-204 (consistent with the usual SMA/NPA bands, which PNB must confirm) |
| `outstanding_balance` <= `sanction_amount` | Held; likely does not hold for overdraft or cash-credit limits in general - to confirm |
| `overdue_amount` <= `outstanding_balance` | Held |
| `overdue_amount` > 0 if and only if `days_past_due` > 0 | Held |
| `account_ref` unique per row | Held |
| One account per customer | Held in sample only |

