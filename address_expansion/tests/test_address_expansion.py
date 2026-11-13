"""Tests for pincode reference building and address expansion (synthetic data only)."""

import build_pincode_reference as bpr
import expand_addresses as ea
import pandas as pd
import pytest

PROJECT = {"GUNTUR": 505, "KRISHNA": 510, "YSR": 504, "ANANTAPURAMU": 502}


def raw_directory() -> pd.DataFrame:
    """Synthetic post-office directory covering each mapping status."""
    rows = [
        ("522001", "GUNTUR", "ANDHRA PRADESH"),
        ("522001", "GUNTUR", "ANDHRA PRADESH"),
        ("520001", "KRISHNA", "ANDHRA PRADESH"),
        ("515465", "ANANTAPUR", "ANDHRA PRADESH"),  # alias spelling
        ("515465", "Y.S.R.", "ANDHRA PRADESH"),  # ambiguous district
        ("505505", "GUNTUR", "ANDHRA PRADESH"),  # ambiguous state
        ("505505", "RANGAREDDY", "TELANGANA"),
        ("110001", "NEW DELHI", "DELHI"),
        ("400001", "MUMBAI", "MAHARASHTRA"),
    ]
    return pd.DataFrame(rows, columns=["Pincode", "District", "StateName"])


@pytest.fixture
def reference() -> pd.DataFrame:
    """Reference table built from the synthetic directory."""
    built, _ = bpr.build_reference(raw_directory(), PROJECT)
    return built.astype(str).replace("nan", "")


def test_normalise_name_handles_punctuation() -> None:
    assert bpr.normalise_name("Y.S.R.") == "Y S R"
    assert bpr.normalise_name("  east   godavari ") == "EAST GODAVARI"


def test_reference_has_one_row_per_pincode_and_statuses() -> None:
    built, counts = bpr.build_reference(raw_directory(), PROJECT)
    assert built["pincode"].is_unique
    status = built.set_index("pincode")["mapping_status"].to_dict()
    assert status["522001"] == "mapped"
    assert status["515465"] == "ambiguous_district"
    assert status["505505"] == "ambiguous_state"
    assert status["110001"] == "outside_project_scope"
    assert counts["raw_rows"] == 9


def test_alias_maps_to_project_code_and_ambiguity_leaves_code_blank() -> None:
    built, _ = bpr.build_reference(raw_directory(), PROJECT)
    row = built.set_index("pincode")
    assert row.loc["522001", "district_code"] == "505"
    assert row.loc["515465", "district_code"] == ""
    assert row.loc["515465", "project_district_codes"] == "502|504"


def test_invalid_pincodes_are_dropped_and_counted() -> None:
    raw = pd.concat(
        [
            raw_directory(),
            pd.DataFrame(
                [("12345", "X", "Y"), ("012345", "X", "Y")], columns=raw_directory().columns
            ),
        ]
    )
    _, counts = bpr.build_reference(raw, PROJECT)
    assert counts["invalid_pincode_rows"] == 2


def test_parse_address_four_parts() -> None:
    parsed = ea.parse_address("Plot 12, Green Park, Lakshmipuram, Guntur")
    assert parsed["address_parse_status"] == "parsed"
    assert parsed["addr_city"] == "Guntur"


def test_parse_address_labels_instead_of_guessing() -> None:
    assert ea.parse_address("Plot 12, Guntur")["address_parse_status"] == "unexpected_part_count"
    assert ea.parse_address("")["address_parse_status"] == "missing_address"
    assert ea.parse_address(None)["address_parse_status"] == "missing_address"


def test_embedded_pincode_extracted_only_when_unambiguous() -> None:
    assert ea.parse_address("A, B, C, Guntur 522001")["address_pincode"] == "522001"
    assert ea.parse_address("A 522001, B 520001, C, D")["address_pincode"] == ""


def test_clean_pincode() -> None:
    assert ea.clean_pincode("522001") == "522001"
    assert ea.clean_pincode("522001.0") == "522001"
    assert ea.clean_pincode("52200") == ""
    assert ea.clean_pincode(None) == ""


def test_city_check() -> None:
    known = {"GUNTUR", "KRISHNA"}
    assert ea.city_check("Guntur", "GUNTUR", known) == "consistent"
    assert ea.city_check("Krishna", "GUNTUR", known) == "conflict"
    assert ea.city_check("Lakshmipuram", "GUNTUR", known) == "not_comparable"
    assert ea.city_check("", "GUNTUR", known) == "no_city"


def test_expand_preserves_rows_and_labels_statuses(reference: pd.DataFrame) -> None:
    frame = pd.DataFrame(
        {
            "account number": ["1", "2", "3", "4", "5"],
            "borrowers address": ["P1, A1, L1, Guntur"] * 4 + ["only two, parts"],
            "pincode": ["522001", "515465", "999999", "12"] + ["110001"],
        }
    )
    out = ea.expand(frame, reference)
    assert len(out) == len(frame)
    status = dict(zip(out["account number"], out["mapping_status"], strict=True))
    assert status == {
        "1": "mapped",
        "2": "ambiguous_district",
        "3": "not_in_reference",
        "4": "invalid_pincode",
        "5": "outside_project_scope",
    }
    assert out.loc[out["account number"] == "1", "district_code"].iloc[0] == "505"
    assert (
        out.loc[out["account number"] == "5", "address_parse_status"].iloc[0]
        == "unexpected_part_count"
    )


def test_report_contains_no_row_level_values(reference: pd.DataFrame) -> None:
    frame = pd.DataFrame(
        {
            "account number": ["1"],
            "borrowers address": ["P1, A1, L1, Guntur"],
            "pincode": ["522001"],
        }
    )
    report = ea.build_report(ea.expand(frame, reference))
    assert set(report.columns) == {"check", "value", "count"}
    assert "P1" not in report.to_string()
