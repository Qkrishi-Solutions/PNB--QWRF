#!/usr/bin/env python3
"""Structural and provenance checks for Issue #41 district credit benchmark."""
import csv
import math
from pathlib import Path

CSV_PATH = Path(__file__).parent / "data" / "district_allocation_benchmark_v1.csv"
EXPECTED_DISTRICTS = {
    "Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna",
    "Kurnool", "Kadapa", "Nellore", "Prakasam", "Srikakulam",
    "Visakhapatnam", "Vizianagaram", "West Godavari",
}
NABARD_STATE_TOTAL_LAKH = 10631638.07
NABARD_SOURCE = "https://www.nabard.org/auth/writereaddata/tender/1005184203sfp%202018-19%20andhrapradesh.pdf"
RBI_SOURCE = "https://data.rbi.org.in"

def num(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

errors = []
warnings = []
districts = [r.get("district_13_geography", "").strip() for r in rows]
if len(rows) != 13:
    errors.append(f"Expected 13 rows; found {len(rows)}")
if len(set(districts)) != len(districts):
    errors.append("Duplicate district names found")
if set(districts) != EXPECTED_DISTRICTS:
    errors.append(f"District set mismatch; missing={sorted(EXPECTED_DISTRICTS-set(districts))}; unexpected={sorted(set(districts)-EXPECTED_DISTRICTS)}")

nabard_vals = [num(r, "nabard_agriculture_credit_potential_lakh") for r in rows]
if any(math.isnan(v) for v in nabard_vals):
    errors.append("Missing/non-numeric NABARD values")
nabard_sum = sum(v for v in nabard_vals if not math.isnan(v))
if abs(nabard_sum - NABARD_STATE_TOTAL_LAKH) > 0.02:
    errors.append(f"NABARD sum {nabard_sum:.4f} does not match published state total {NABARD_STATE_TOTAL_LAKH:.2f}")
for r in rows:
    if r.get("nabard_source_year") != "2018-19" or r.get("nabard_source_url") != NABARD_SOURCE:
        errors.append(f"{r.get('district_13_geography')}: NABARD source year/URL missing or unexpected")

shares = [num(r, "benchmark_share") for r in rows]
share_sum = sum(v for v in shares if not math.isnan(v))
if any(math.isnan(v) for v in shares):
    errors.append("Missing/non-numeric allocation shares")
if abs(share_sum - 1.0) > 1e-8:
    errors.append(f"Allocation share sum is {share_sum:.12f}, expected 1")

rbi_vals = [num(r, "rbi_total_agriculture_outstanding_thousand") for r in rows]
if any(math.isnan(v) for v in rbi_vals):
    errors.append("Missing/non-numeric RBI agriculture totals")
for r in rows:
    d = r.get("district_13_geography", "<unknown>")
    if r.get("rbi_source_year") != "2019-03-31":
        errors.append(f"{d}: RBI reference date missing or incorrect")
    if r.get("rbi_source_url") != RBI_SOURCE:
        errors.append(f"{d}: RBI source URL missing or incorrect")
    direct = num(r, "rbi_direct_finance_outstanding_thousand")
    indirect = num(r, "rbi_indirect_finance_outstanding_thousand")
    total = num(r, "rbi_total_agriculture_outstanding_thousand")
    crore = num(r, "rbi_total_agriculture_outstanding_crore")
    if any(math.isnan(v) for v in [direct, indirect, total, crore]):
        errors.append(f"{d}: RBI direct/indirect/total/unit-converted value missing")
    elif abs((direct + indirect) - total) > 0.01:
        errors.append(f"{d}: RBI direct + indirect does not equal total")
    elif abs(total / 10000.0 - crore) > 1e-8:
        errors.append(f"{d}: RBI ₹ thousand to ₹ crore conversion mismatch")
    if "excluding Regional Rural Banks" not in r.get("rbi_coverage_note", ""):
        errors.append(f"{d}: RBI RRB coverage limitation not documented")

print("ISSUE #41 DISTRICT BENCHMARK VALIDATION")
print("-" * 48)
print(f"Rows: {len(rows)}")
print(f"Distinct districts: {len(set(districts))}")
print(f"NABARD total: {nabard_sum:,.2f} lakh")
print(f"Published NABARD state total: {NABARD_STATE_TOTAL_LAKH:,.2f} lakh")
print(f"Allocation share sum: {share_sum:.12f}")
print(f"RBI agriculture total (13 districts): {sum(v for v in rbi_vals if not math.isnan(v)):,.0f} thousand")
print(f"RBI district totals populated: {sum(not math.isnan(v) for v in rbi_vals)}")
print(f"Errors: {len(errors)}")
for e in errors:
    print("ERROR:", e)
print(f"Warnings: {len(warnings)}")
for w in warnings:
    print("WARN:", w)
if errors:
    print("RESULT: FAILED")
    raise SystemExit(1)
print("RESULT: ALL STRUCTURAL AND SOURCE-FIELD CHECKS PASSED")
print("Note: Source values are traceable to the downloaded RBI DBIE Table 3.9 workbook; keep that source workbook archived with project evidence.")
