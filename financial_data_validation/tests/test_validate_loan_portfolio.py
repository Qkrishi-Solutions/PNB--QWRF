"""Tests for the loan-portfolio validator (synthetic data only)."""

import pandas as pd
import pytest
import validate_loan_portfolio as vlp

SCHEMA = vlp.load_schema(vlp.DEFAULT_SCHEMA)


def good_frame() -> pd.DataFrame:
    """Two clean synthetic accounts."""
    rows = [
        [
            "2026-10-31",
            "Hyderabad",
            "Vijayawada",
            "SB001",
            "Synthetic Branch 1",
            "SYNT00001",
            "Synthetic Customer 1",
            "Plot 1, Area, Locality, Guntur",
            "522001",
            "12345678901",
            "2022-01-15",
            "Term Loan",
            "TL101",
            "Horticulture",
            "500000",
            "9.5",
            "300000",
            "0",
            "0",
            "Standard",
        ],
        [
            "2026-10-31",
            "Hyderabad",
            "Vijayawada",
            "SB001",
            "Synthetic Branch 1",
            "SYNT00002",
            "Synthetic Customer 2",
            "Plot 2, Area, Locality, Guntur",
            "522002",
            "12345678902",
            "2023-03-10",
            "Cash Credit",
            "AGRI01",
            "Dairy & Animal Husbandry",
            "250000",
            "8.2",
            "200000",
            "20000",
            "45",
            "SMA 1",
        ],
    ]
    return pd.DataFrame(rows, columns=[c["name"] for c in SCHEMA["columns"]])


def reference() -> pd.DataFrame:
    """Tiny pincode reference."""
    return pd.DataFrame(
        {
            "pincode": ["522001", "522002", "110001", "515465"],
            "mapping_status": ["mapped", "mapped", "outside_project_scope", "ambiguous_district"],
        }
    )


def run(frame: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    return vlp.to_report(vlp.validate(frame, SCHEMA, ref))


def status(report: pd.DataFrame, check_id: str, column: str | None = None) -> str:
    rows = report[report["check_id"] == check_id]
    if column is not None:
        rows = rows[rows["column"] == column]
    assert len(rows) == 1, (check_id, column, len(rows))
    return rows.iloc[0]["status"]


def test_clean_frame_has_no_failures() -> None:
    report = run(good_frame(), reference())
    assert not report["status"].isin(["FAIL", "WARN"]).any(), report[
        report["status"].isin(["FAIL", "WARN"])
    ]
    assert vlp.exit_code(vlp.validate(good_frame(), SCHEMA, reference())) == 0


def test_validation_does_not_modify_input() -> None:
    frame = good_frame()
    before = frame.copy(deep=True)
    run(frame, reference())
    pd.testing.assert_frame_equal(frame, before)


def test_missing_required_column_fails_and_skips_dependents() -> None:
    report = run(good_frame().drop(columns=["days_past_due"]))
    assert status(report, "required_columns") == "FAIL"
    assert status(report, "asset_class_vs_dpd") == "SKIPPED"


def test_unexpected_column_is_warning() -> None:
    frame = good_frame().assign(extra="x")
    assert status(run(frame), "unexpected_columns") == "WARN"


def test_missing_required_value_fails_and_is_not_imputed() -> None:
    frame = good_frame()
    frame.loc[0, "pincode"] = ""
    report = run(frame)
    assert status(report, "missing_values", "pincode") == "FAIL"
    assert frame.loc[0, "pincode"] == ""


@pytest.mark.parametrize(
    ("column", "value", "check_id"),
    [
        ("reporting_date", "31/10/2026", "valid_date"),
        ("sanction amount", "abc", "valid_number"),
        ("days_past_due", "4.5", "whole_number"),
        ("sanction amount", "-1", "minimum"),
        ("rate of interest", "150", "maximum"),
        ("pincode", "12345", "format_pattern"),
        ("facility type", "Gold Loan", "allowed_values"),
        ("asset class", "Doubtful", "allowed_values"),
    ],
)
def test_format_and_value_failures(column: str, value: str, check_id: str) -> None:
    frame = good_frame()
    frame.loc[0, column] = value
    assert status(run(frame), check_id, column) == "FAIL"


def test_id_pattern_failures_are_warnings_not_errors() -> None:
    frame = good_frame()
    frame.loc[0, "account number"] = "123"
    report = run(frame)
    assert status(report, "format_pattern", "account number") == "WARN"
    assert vlp.exit_code(vlp.validate(frame, SCHEMA)) == 0


@pytest.mark.parametrize(
    ("asset", "dpd", "overdue"),
    [("Standard", "10", "5000"), ("NPA", "45", "20000"), ("SMA 2", "20", "20000")],
)
def test_asset_class_must_match_dpd_band(asset: str, dpd: str, overdue: str) -> None:
    frame = good_frame()
    frame.loc[1, ["asset class", "days_past_due", "overdue amount"]] = [asset, dpd, overdue]
    assert status(run(frame), "asset_class_vs_dpd") == "FAIL"


def test_overdue_cannot_exceed_outstanding() -> None:
    frame = good_frame()
    frame.loc[1, "overdue amount"] = "999999"
    assert status(run(frame), "overdue_le_outstanding") == "FAIL"


def test_overdue_and_dpd_must_agree() -> None:
    frame = good_frame()
    frame.loc[1, "days_past_due"] = "0"
    assert status(run(frame), "overdue_iff_dpd") == "FAIL"


def test_outstanding_above_sanction_is_only_a_warning() -> None:
    frame = good_frame()
    frame.loc[0, "balance outstanding"] = "600000"
    assert status(run(frame), "outstanding_le_sanction") == "WARN"


def test_open_date_after_reporting_date_fails() -> None:
    frame = good_frame()
    frame.loc[0, "account open date"] = "2027-01-01"
    assert status(run(frame), "open_date_not_after_reporting") == "FAIL"


def test_duplicate_account_numbers_fail() -> None:
    frame = good_frame()
    frame.loc[1, "account number"] = frame.loc[0, "account number"]
    report = run(frame)
    assert status(report, "account_key_unique") == "FAIL"
    assert report[report["check_id"] == "account_key_unique"].iloc[0]["failing_rows"] == 2


def test_branch_with_two_names_warns() -> None:
    frame = good_frame()
    frame.loc[1, "branch name"] = "Other Name"
    assert status(run(frame), "branch_id_one_name") == "WARN"


def test_pincode_reference_checks() -> None:
    frame = good_frame()
    frame.loc[0, "pincode"] = "110001"  # outside project state
    frame.loc[1, "pincode"] = "999999"  # not in reference
    report = run(frame, reference())
    assert status(report, "pincode_in_project_state") == "FAIL"
    assert status(report, "pincode_in_reference") == "FAIL"


def test_ambiguous_pincode_is_a_warning() -> None:
    frame = good_frame()
    frame.loc[0, "pincode"] = "515465"
    assert status(run(frame, reference()), "pincode_district_ambiguous") == "WARN"


def test_report_contains_no_cell_values() -> None:
    frame = good_frame()
    frame.loc[0, "customer name"] = "Distinctive Private Name"
    frame.loc[0, "days_past_due"] = "abc"
    assert "Distinctive Private Name" not in run(frame).to_string()


def test_main_returns_nonzero_on_error_and_2_on_bad_input(tmp_path) -> None:
    bad = good_frame()
    bad.loc[0, "asset class"] = "Standard"
    bad.loc[0, "days_past_due"] = "99"
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    assert vlp.main(["--input", str(path), "--outdir", str(tmp_path / "out")]) == 1
    assert (tmp_path / "out" / "validation_report.csv").exists()
    assert (
        vlp.main(["--input", str(tmp_path / "missing.csv"), "--outdir", str(tmp_path / "o2")]) == 2
    )
