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

PROJECT_13 = [
    (502, "ANANTAPURAMU"), (503, "CHITTOOR"), (504, "YSR"), (505, "EAST GODAVARI"), (506, "GUNTUR"),
    (510, "KRISHNA"), (511, "KURNOOL"), (515, "SPR NELLORE"), (517, "PRAKASAM"), (519, "SRIKAKULAM"),
    (520, "VISAKHAPATNAM"), (521, "VIZIANAGARAM"), (523, "WEST GODAVARI"),
]


def write_districts(path, rows=PROJECT_13):
    import pandas as pd

    pd.DataFrame(rows, columns=["district_code", "district_name"]).to_csv(path, index=False)
    return path


def test_sample_end_to_end(tmp_path):
    # Uses a small fixture districts file, so it always runs (no skip).
    districts = write_districts(tmp_path / "districts.csv")
    p.run(SAMPLE, districts, tmp_path, "synthetic_sample")
    import pandas as pd

    parsed = pd.read_csv(tmp_path / "imd_warnings_parsed_sample.csv", keep_default_na=False)
    rejected = pd.read_csv(tmp_path / "imd_warnings_rejected_sample.csv", keep_default_na=False)
    assert parsed["imd_warning_id"].nunique() == 5
    assert sorted(rejected["warning_id"]) == ["SYN-0006", "SYN-0007", "SYN-0008"]
    assert set(parsed["imd_warning_source_kind"]) == {"synthetic_sample"}
    unmatched = parsed[parsed["imd_warning_district_match"] == "unmatched"]
    assert list(unmatched["imd_warning_district_raw"]) == ["Hyderabad"]
    assert not parsed.duplicated(["imd_warning_id", "imd_warning_district_raw"]).any()


# ---------- review fixes: duplicates, merged advisories ----------

def test_repeated_key_rejects_block():
    records, rejects = parse_one(GOOD + "VALID TO: 2025-11-30 08:30 IST\n")
    assert not records and "duplicate field: VALID TO" in rejects[0]["errors"]


def test_missing_separator_does_not_silently_merge_advisories():
    second = replace(GOOD, "ADVISORY-ID", "T-2")
    records, rejects = parse_one(GOOD + "\n" + second)  # no dashes between them
    assert not records
    assert len(rejects) == 1 and "duplicate field" in rejects[0]["errors"]
    assert rejects[0]["warning_id"] == "T-1"  # the first advisory is identified, not lost


def test_duplicate_warning_id_across_blocks_rejects_later_one():
    text = GOOD + "\n----------\n" + replace(GOOD, "STATE", "Another State")
    records, rejects = parse_one(text)
    assert len(records) == 1 and records[0]["state"] == "Andhra Pradesh"
    assert len(rejects) == 1 and "duplicate warning_id: T-1" in rejects[0]["errors"]


def test_parse_block_returns_documented_three_values():
    schema = p.load_schema()
    record, errors, warning_id = p.parse_block(GOOD.strip().splitlines(), "synthetic_sample", schema)
    assert record is not None and errors == [] and warning_id == "T-1"
    record, errors, warning_id = p.parse_block(without(GOOD, "VALID TO").splitlines(), "synthetic_sample", schema)
    assert record is None and errors and warning_id == "T-1"


# ---------- review fixes: hazard classification ----------

@pytest.mark.parametrize("phrase,expected", [
    ("Thunderstorm with heavy rain", "other"),            # two hazards, not collapsed onto the first match
    ("Heavy rainfall with cyclonic circulation", "other"),
    ("No heavy rain expected", "other"),                 # negated
    ("Heavy rain warning withdrawn", "other"),
    ("Very heavy rainfall", "heavy_rain"),
    ("Extremely heavy rain", "heavy_rain"),
    ("Cyclonic circulation with strong winds", "cyclone"),  # cyclone absorbs strong wind
    ("Squall", "strong_wind"),
    ("", "other"),
])
def test_classification_never_guesses_between_hazards(phrase, expected):
    assert p.classify_type(phrase) == expected


def test_raw_phrase_is_kept_when_type_is_other():
    rec = parse_one(replace(GOOD, "WARNING", "Thunderstorm with heavy rain"))[0][0]
    assert rec["warning_type"] == "other"
    assert rec["warning_phrase"] == "Thunderstorm with heavy rain"
    assert p.validate_record(rec) == []
    rows = p.to_district_rows([rec], PROJECT)
    assert rows.loc[0, "imd_warning_phrase"] == "Thunderstorm with heavy rain"


# ---------- review fixes: times ----------

@pytest.mark.parametrize("text", [
    "2025-10-28 08:30 IST", "2025-10-28 08:30:00 IST", "2025-10-28T08:30:00+05:30",
    "2025-10-28 08:30+0530", "2025-10-28 03:00 UTC", "2025-10-28T03:00:00Z",
])
def test_accepted_time_forms_all_equal_same_instant(text):
    assert p.parse_time(text).isoformat() == "2025-10-28T08:30:00+05:30"


@pytest.mark.parametrize("text", [
    "2025-10-28 08:30", "2025-10-28T08:30:00", "2025-10-28 08:30 PST", "2025-10-28 08:61 IST",
    "2025-10-28 08:30 +25:00", "0830 IST", "",
])
def test_time_without_a_valid_zone_is_rejected_not_assumed(text):
    assert p.parse_time(text) is None


# ---------- review fixes: districts ----------

def test_same_district_under_two_spellings_is_collapsed_and_reported():
    rec = {**valid_record(), "districts": ["Kadapa", "YSR Kadapa", "Guntur"]}
    rows = p.to_district_rows([rec], PROJECT)
    assert list(rows["imd_warning_district_code"]) == [504, 506]
    assert not rows.duplicated(["imd_warning_id", "imd_warning_district_code"]).any()
    assert rows.attrs["collapsed_duplicates"] == [("T-1", "YSR Kadapa")]


def test_unmatched_districts_are_never_collapsed():
    rec = {**valid_record(), "districts": ["Hyderabad", "Warangal"]}
    rows = p.to_district_rows([rec], PROJECT)
    assert len(rows) == 2 and rows.attrs["collapsed_duplicates"] == []


def test_district_file_with_clashing_names_or_codes_is_rejected(tmp_path):
    ok = write_districts(tmp_path / "a.csv", [(502, "ANANTAPURAMU"), (503, "CHITTOOR")])
    same_name = write_districts(tmp_path / "b.csv", [(1, "GUNTUR"), (2, "Guntur")])
    same_code = write_districts(tmp_path / "c.csv", [(1, "GUNTUR"), (1, "KRISHNA")])
    assert len(p.load_project_districts(ok)) == 2
    with pytest.raises(ValueError, match="exactly one"):
        p.load_project_districts(same_name)
    with pytest.raises(ValueError, match="exactly one"):
        p.load_project_districts(same_code)


# ---------- review fixes: output naming and safety ----------

def test_output_names_never_label_real_advisories_as_samples():
    assert "sample" in p.output_names("synthetic_sample")["parsed"]
    assert all("sample" not in name for name in p.output_names("imd_advisory").values())


def test_real_advisories_cannot_be_written_into_committed_samples_folder(tmp_path):
    districts = write_districts(tmp_path / "d.csv")
    with pytest.raises(ValueError, match="Refusing to write real advisories"):
        p.run(SAMPLE, districts, p.SAMPLES_DIR, "imd_advisory")


def test_real_advisory_run_writes_non_sample_files_and_leaves_fixtures_alone(tmp_path):
    districts = write_districts(tmp_path / "d.csv")
    before = (p.SAMPLES_DIR / "imd_warnings_parsed_sample.csv").read_bytes()
    out = tmp_path / "out"
    p.run(SAMPLE, districts, out, "imd_advisory")
    assert (out / "imd_warnings_parsed_advisories.csv").exists()
    assert not (out / "imd_warnings_parsed_sample.csv").exists()
    assert (p.SAMPLES_DIR / "imd_warnings_parsed_sample.csv").read_bytes() == before


# ---------- review fixes: separation is checked on the repository's current data ----------

def test_no_repository_csv_outside_this_folder_contains_warning_columns():
    """The guard is not yet called by feature code, so check the data instead:
    no other CSV in the repository may carry imd_warning_ columns."""
    offenders = []
    for path in ROOT.rglob("*.csv"):
        if "imd_warnings" in path.parts or ".claude" in path.parts or ".git" in path.parts:
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
        if p.PREFIX in header:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
