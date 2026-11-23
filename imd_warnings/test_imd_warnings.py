"""Tests for the IMD warning schema and parser (Issue 47).

Run from the repository root:
    python3 -m pytest imd_warnings/test_imd_warnings.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_imd_warnings as p  # noqa: E402

SAMPLE = Path(__file__).resolve().parent / "samples" / "synthetic_imd_advisories.txt"
ROOT = Path(__file__).resolve().parent.parent
WEATHER = ROOT / "weather_feature_engineering" / "weather_anomalies_forward_chaining_v1.csv"

GOOD = """ADVISORY-ID: T-1
ISSUED: 2025-10-27 14:30 IST
STATE: Andhra Pradesh
WARNING: Heavy rainfall
COLOUR CODE: Orange
DISTRICTS: Guntur, Krishna
VALID FROM: 2025-10-28 08:30 IST
VALID TO: 2025-10-29 08:30 IST
HEADLINE: test
"""


def parse_one(text):
    records, rejects = p.parse_advisories(text, "synthetic_sample")
    return records, rejects


def without(text, key):
    return "\n".join(l for l in text.splitlines() if not l.startswith(key + ":"))


def replace(text, key, value):
    return "\n".join(f"{key}: {value}" if l.startswith(key + ":") else l for l in text.splitlines())


# ---------- passing cases ----------

def test_valid_advisory_parses_and_matches_schema():
    records, rejects = parse_one(GOOD)
    assert len(records) == 1 and not rejects
    rec = records[0]
    assert rec["warning_type"] == "heavy_rain"
    assert rec["severity"] == "orange"
    assert rec["districts"] == ["Guntur", "Krishna"]
    assert p.validate_record(rec) == []


def test_times_are_stored_with_ist_offset_and_utc_is_converted():
    rec = parse_one(replace(GOOD, "ISSUED", "2025-10-27 09:00 UTC"))[0][0]
    assert rec["issue_time"] == "2025-10-27T14:30:00+05:30"
    assert rec["valid_from"].endswith("+05:30")


@pytest.mark.parametrize("phrase,expected", [
    ("Heavy rainfall", "heavy_rain"),
    ("Cyclonic circulation", "cyclone"),
    ("Heat wave", "heat_wave"),
    ("Cold wave", "cold_wave"),
    ("Thunderstorm with lightning", "thunderstorm"),
    ("Strong winds", "strong_wind"),
    ("Flood watch", "flood"),
])
def test_warning_type_mapping(phrase, expected):
    assert p.classify_type(phrase) == expected


def test_unknown_phrase_is_other_not_guessed():
    assert p.classify_type("Dense fog") == "other"
    assert parse_one(replace(GOOD, "WARNING", "Dense fog"))[0][0]["warning_type"] == "other"


def test_blank_headline_is_allowed():
    records, rejects = parse_one(without(GOOD, "HEADLINE"))
    assert len(records) == 1 and not rejects and records[0]["headline"] == ""


# ---------- failing cases: required fields are never filled in ----------

@pytest.mark.parametrize("key", [
    "ADVISORY-ID", "ISSUED", "STATE", "WARNING", "COLOUR CODE", "DISTRICTS", "VALID FROM", "VALID TO",
])
def test_missing_required_field_is_rejected(key):
    records, rejects = parse_one(without(GOOD, key))
    assert not records
    assert len(rejects) == 1
    assert f"missing required field: {key}" in rejects[0]["errors"]


def test_missing_valid_to_is_not_defaulted():
    records, rejects = parse_one(without(GOOD, "VALID TO"))
    assert not records and "VALID TO" in rejects[0]["errors"]


def test_invalid_colour_code_rejected():
    records, rejects = parse_one(replace(GOOD, "COLOUR CODE", "Purple"))
    assert not records and "invalid colour code" in rejects[0]["errors"]


@pytest.mark.parametrize("bad", ["2025-13-40 08:30 IST", "28/10/2025 08:30", "2025-10-28 08:30", "tomorrow"])
def test_invalid_time_rejected(bad):
    records, rejects = parse_one(replace(GOOD, "VALID FROM", bad))
    assert not records and "invalid time for VALID FROM" in rejects[0]["errors"]


def test_valid_to_must_be_after_valid_from():
    swapped = replace(replace(GOOD, "VALID FROM", "2025-10-29 08:30 IST"), "VALID TO", "2025-10-28 08:30 IST")
    records, rejects = parse_one(swapped)
    assert not records and "valid_to must be after valid_from" in rejects[0]["errors"]


def test_unknown_field_and_unparseable_line_rejected():
    assert "unknown field: EXTRA" in parse_one(GOOD + "EXTRA: x\n")[1][0]["errors"]
    assert "unparseable line" in parse_one(GOOD + "no colon here\n")[1][0]["errors"]


def test_bad_source_kind_rejected():
    with pytest.raises(ValueError):
        p.parse_advisories(GOOD, "made_up")


# ---------- schema validation ----------

def valid_record():
    return parse_one(GOOD)[0][0]


def test_schema_accepts_valid_record():
    assert p.validate_record(valid_record()) == []


@pytest.mark.parametrize("field,value,fragment", [
    ("warning_type", "tornado", "not allowed"),
    ("severity", "purple", "not allowed"),
    ("districts", [], "required"),
    ("districts", [1], "list of non-empty strings"),
    ("issue_time", "27/10/2025 14:30", "not ISO 8601"),
    ("issue_time", "2025-10-27T14:30:00", "time zone"),
    ("issue_time", "2025-10-27 14:30", "time zone"),
    ("source_kind", "other", "not allowed"),
    ("warning_id", "", "required"),
])
def test_schema_rejects_bad_values(field, value, fragment):
    rec = valid_record()
    rec[field] = value
    assert any(fragment in problem for problem in p.validate_record(rec))


def test_schema_rejects_field_not_in_schema():
    rec = valid_record()
    rec["model_risk_score"] = 0.9
    assert any("not in schema" in problem for problem in p.validate_record(rec))


# ---------- districts ----------

PROJECT = {
    p.normalise_district("ANANTAPURAMU"): (502, "ANANTAPURAMU"),
    p.normalise_district("SPR NELLORE"): (515, "SPR NELLORE"),
    p.normalise_district("YSR"): (504, "YSR"),
    p.normalise_district("GUNTUR"): (506, "GUNTUR"),
}


@pytest.mark.parametrize("raw,code", [
    ("Anantapur", 502), ("ANANTAPURAMU", 502), ("Nellore", 515), ("SPSR Nellore", 515),
    ("Kadapa", 504), ("YSR Kadapa", 504), ("  guntur ", 506), ("Y.S.R.", 504),
])
def test_district_aliases_match(raw, code):
    rows = p.to_district_rows([{**valid_record(), "districts": [raw]}], PROJECT)
    assert rows.loc[0, "imd_warning_district_match"] == "matched"
    assert int(rows.loc[0, "imd_warning_district_code"]) == code


def test_unknown_district_is_unmatched_and_kept():
    rows = p.to_district_rows([{**valid_record(), "districts": ["Hyderabad"]}], PROJECT)
    assert len(rows) == 1
    assert rows.loc[0, "imd_warning_district_match"] == "unmatched"
    assert rows.loc[0, "imd_warning_district_code"] == ""


def test_district_rows_one_per_warning_district_and_unique():
    rows = p.to_district_rows([valid_record()], PROJECT)
    assert len(rows) == 2
    assert not rows.duplicated(["imd_warning_id", "imd_warning_district_raw"]).any()


# ---------- separation from model features ----------

def test_all_output_columns_are_prefixed_and_flagged_official():
    schema = p.load_schema()
    rows = p.to_district_rows([valid_record()], PROJECT, schema)
    assert all(c.startswith(p.PREFIX) for c in rows.columns)
    assert set(rows["imd_warning_record_class"]) == {"official_warning"}


def test_feature_guard_rejects_warning_columns():
    rows = p.to_district_rows([valid_record()], PROJECT)
    with pytest.raises(ValueError, match="must not be used as model features"):
        p.assert_not_in_feature_frame(list(rows.columns))
    with pytest.raises(ValueError):
        p.assert_not_in_feature_frame(["wx_kharif__rain_sum_z", "imd_warning_severity"])


def test_feature_guard_allows_ordinary_features():
    p.assert_not_in_feature_frame(["wx_kharif__rain_sum_z", "district_code", "climate_zone"])


# ---------- end to end on the synthetic sample ----------

@pytest.mark.skipif(not WEATHER.exists(), reason="weather anomalies file not present")
def test_sample_end_to_end(tmp_path):
    p.run(SAMPLE, WEATHER, tmp_path, "synthetic_sample")
    import pandas as pd

    parsed = pd.read_csv(tmp_path / "imd_warnings_parsed_sample.csv", keep_default_na=False)
    rejected = pd.read_csv(tmp_path / "imd_warnings_rejected_sample.csv", keep_default_na=False)
    assert parsed["imd_warning_id"].nunique() == 5
    assert sorted(rejected["warning_id"]) == ["SYN-0006", "SYN-0007", "SYN-0008"]
    assert set(parsed["imd_warning_source_kind"]) == {"synthetic_sample"}
    unmatched = parsed[parsed["imd_warning_district_match"] == "unmatched"]
    assert list(unmatched["imd_warning_district_raw"]) == ["Hyderabad"]
    assert not parsed.duplicated(["imd_warning_id", "imd_warning_district_raw"]).any()
