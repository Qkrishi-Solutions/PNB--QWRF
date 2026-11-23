#!/usr/bin/env python3
"""Independent standard-library audit for the synthetic PNB loan portfolio.

This is an additional audit, not a replacement for the official pandas-based
Issue #24 validator. It writes aggregate results only; no borrower-level values
are written to the report.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

EXPECTED_COLUMNS = [
    "reporting_date", "zone", "circle", "branch ID", "branch name",
    "customer ID", "customer name", "borrowers address", "pincode",
    "account number", "account open date", "facility type", "scheme code",
    "agri sector", "sanction amount", "rate of interest",
    "balance outstanding", "overdue amount", "days_past_due", "asset class",
]

ASSET_BANDS = {
    "Standard": (0, 0),
    "SMA 0": (1, 30),
    "SMA 1": (31, 60),
    "SMA 2": (61, 90),
    "NPA": (91, 10**9),
}

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pincode-reference", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("data/private/stdlib_audit"))
    args = parser.parse_args()

    columns, rows = read_csv(args.input)
    ref_columns, ref_rows = read_csv(args.pincode_reference)
    checks = []

    def record(check_id, passed, detail, severity="ERROR"):
        checks.append({
            "check_id": check_id,
            "severity": severity,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    missing_cols = sorted(set(EXPECTED_COLUMNS) - set(columns))
    extra_cols = sorted(set(columns) - set(EXPECTED_COLUMNS))
    record("required_columns", not missing_cols,
           f"missing_count={len(missing_cols)}")
    record("unexpected_columns", not extra_cols, f"extra_count={len(extra_cols)}", "WARN")
    record("row_count_5000", len(rows) == 5000, f"rows={len(rows)}")

    ref_required = {"pincode", "district_code", "mapping_status"}
    ref_missing = sorted(ref_required - set(ref_columns))
    record("reference_columns", not ref_missing, f"missing_count={len(ref_missing)}")

    ref_map = {}
    if not ref_missing:
        for r in ref_rows:
            pin = (r.get("pincode") or "").strip()
            if pin:
                ref_map[pin] = {
                    "status": (r.get("mapping_status") or "").strip(),
                    "district": (r.get("district_code") or "").strip(),
                }

    missing_counts = Counter()
    bad_dates = 0
    bad_numbers = 0
    bad_asset_dpd = 0
    overdue_gt_outstanding = 0
    overdue_dpd_mismatch = 0
    open_after_reporting = 0
    duplicate_accounts = 0
    seen_accounts = set()
    invalid_pins = 0
    not_mapped_pins = 0
    outside_ap_pins = 0
    district_counts = Counter()

    for row in rows:
        for col in EXPECTED_COLUMNS:
            if not (row.get(col) or "").strip():
                missing_counts[col] += 1

        try:
            reporting = date.fromisoformat(row["reporting_date"].strip())
            opened = date.fromisoformat(row["account open date"].strip())
            if opened > reporting:
                open_after_reporting += 1
        except Exception:
            bad_dates += 1

        try:
            sanction = float(row["sanction amount"])
            interest = float(row["rate of interest"])
            outstanding = float(row["balance outstanding"])
            overdue = float(row["overdue amount"])
            dpd = int(row["days_past_due"])
            if sanction < 0 or interest < 0 or outstanding < 0 or overdue < 0 or dpd < 0:
                bad_numbers += 1
            if overdue > outstanding:
                overdue_gt_outstanding += 1
            if (overdue > 0) != (dpd > 0):
                overdue_dpd_mismatch += 1
            asset = row["asset class"].strip()
            lo_hi = ASSET_BANDS.get(asset)
            if lo_hi is None or not (lo_hi[0] <= dpd <= lo_hi[1]):
                bad_asset_dpd += 1
        except Exception:
            bad_numbers += 1

        account = (row.get("account number") or "").strip()
        if account:
            if account in seen_accounts:
                duplicate_accounts += 1
            seen_accounts.add(account)

        pin = (row.get("pincode") or "").strip()
        if len(pin) != 6 or not pin.isdigit() or pin[0] == "0":
            invalid_pins += 1
        else:
            mapped = ref_map.get(pin)
            if not mapped or mapped["status"] != "mapped" or not mapped["district"]:
                not_mapped_pins += 1
            else:
                district_counts[mapped["district"]] += 1
            if mapped and mapped["status"] in {"outside_project_scope", "ambiguous_state"}:
                outside_ap_pins += 1

    record("missing_required_values", sum(missing_counts.values()) == 0,
           f"blank_cells={sum(missing_counts.values())}")
    record("valid_dates", bad_dates == 0, f"invalid_date_rows={bad_dates}")
    record("open_date_not_after_reporting", open_after_reporting == 0,
           f"violations={open_after_reporting}")
    record("valid_nonnegative_numbers", bad_numbers == 0, f"invalid_rows={bad_numbers}")
    record("asset_class_dpd_bands", bad_asset_dpd == 0, f"violations={bad_asset_dpd}")
    record("overdue_not_above_outstanding", overdue_gt_outstanding == 0,
           f"violations={overdue_gt_outstanding}")
    record("overdue_dpd_consistency", overdue_dpd_mismatch == 0,
           f"violations={overdue_dpd_mismatch}")
    record("unique_account_numbers", duplicate_accounts == 0,
           f"duplicate_rows={duplicate_accounts}")
    record("valid_pincode_format", invalid_pins == 0, f"invalid_rows={invalid_pins}")
    record("pincodes_mapped_to_project_district", not_mapped_pins == 0,
           f"unmapped_or_ambiguous_rows={not_mapped_pins}")
    record("all_13_project_districts_present", len(district_counts) == 13,
           f"districts_present={len(district_counts)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    report_path = args.outdir / "stdlib_audit_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check_id", "severity", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    summary = {
        "input_rows": len(rows),
        "input_columns": len(columns),
        "pincode_reference_rows": len(ref_rows),
        "mapped_districts_present": len(district_counts),
        "district_account_counts": dict(sorted(district_counts.items())),
        "checks_passed": sum(c["status"] == "PASS" for c in checks),
        "checks_failed": sum(c["status"] == "FAIL" for c in checks),
        "note": "Independent standard-library audit only; not the official Issue #24 validator.",
    }
    summary_path = args.outdir / "stdlib_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Rows checked: {len(rows)}")
    print(f"Checks passed: {summary['checks_passed']}")
    print(f"Checks failed: {summary['checks_failed']}")
    print(f"Districts represented: {len(district_counts)}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    return 0 if summary["checks_failed"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
