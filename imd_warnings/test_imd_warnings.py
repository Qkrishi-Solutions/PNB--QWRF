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


def test_duplicate_warning_id_across_blocks_rejects_all_blocks_with_that_id():
    text = GOOD + "\n----------\n" + replace(GOOD, "STATE", "Another State")
    records, rejects = parse_one(text)
    assert records == []
    assert len(rejects) == 2 and all("duplicate warning_id: T-1" in r["errors"] for r in rejects)


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


@pytest.fixture
def fake_samples(monkeypatch, tmp_path):
    """Point the 'committed samples folder' at a temp copy, so a broken guard cannot write into the repo."""
    folder = tmp_path / "samples"
    folder.mkdir()
    bundled = folder / "synthetic_imd_advisories.txt"
    bundled.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(p, "SAMPLES_DIR", folder)
    monkeypatch.setattr(p, "BUNDLED_SAMPLE", bundled)
    return folder, bundled


def test_real_advisories_cannot_be_written_into_committed_samples_folder(tmp_path, fake_samples):
    folder, bundled = fake_samples
    districts = write_districts(tmp_path / "d.csv")
    # even the bundled file may not be written there under the real-advisory label
    with pytest.raises(ValueError, match="Refusing to write into the committed samples folder"):
        p.run(bundled, districts, folder, "imd_advisory")
    assert not any(folder.glob("imd_warnings_*"))


def test_real_data_labelled_synthetic_cannot_be_written_into_samples(tmp_path, fake_samples):
    folder, _ = fake_samples
    districts = write_districts(tmp_path / "d.csv")
    real = tmp_path / "real_bulletin.txt"
    real.write_text(GOOD, encoding="utf-8")                  # ids like T-1, not SYN-
    with pytest.raises(ValueError, match="Refusing to write into the committed samples folder"):
        p.run(real, districts, folder, "synthetic_sample")
    assert not any(folder.glob("imd_warnings_*"))


def test_bundled_synthetic_file_may_be_written_into_samples(tmp_path, fake_samples):
    folder, bundled = fake_samples
    districts = write_districts(tmp_path / "d.csv")
    p.run(bundled, districts, folder, "synthetic_sample")
    assert (folder / "imd_warnings_parsed_sample.csv").exists()


def test_synthetic_label_requires_syn_ids_everywhere(tmp_path):
    districts = write_districts(tmp_path / "d.csv")
    real = tmp_path / "real_bulletin.txt"
    real.write_text(GOOD, encoding="utf-8")
    with pytest.raises(ValueError, match="requires every advisory id to start with"):
        p.run(real, districts, tmp_path / "out", "synthetic_sample")
    # the same file under the honest label is fine
    p.run(real, districts, tmp_path / "out", "imd_advisory")
    assert (tmp_path / "out" / "imd_warnings_parsed_advisories.csv").exists()


def test_synthetic_label_also_checks_rejected_blocks(tmp_path):
    districts = write_districts(tmp_path / "d.csv")
    mixed = tmp_path / "mixed.txt"
    mixed.write_text(replace(GOOD, "ADVISORY-ID", "SYN-1") + "\n----------\n" + without(replace(GOOD, "ADVISORY-ID", "REAL-1"), "VALID TO"),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="REAL-1"):
        p.run(mixed, districts, tmp_path / "out", "synthetic_sample")


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
    scanned, offenders = p.find_warning_columns(ROOT)
    assert scanned > 0, "the scan looked at no CSV at all, so it proves nothing"
    assert offenders == []


def write_csv(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n1,2\n", encoding="utf-8")


def test_scan_finds_warning_columns_even_when_the_checkout_is_under_a_dot_claude_folder(tmp_path):
    root = tmp_path / ".claude" / "worktrees" / "some_worktree"      # the location that used to hide every file
    write_csv(root / "data" / "features.csv", "district_code,imd_warning_severity")
    write_csv(root / "data" / "clean.csv", "district_code,rain")
    scanned, offenders = p.find_warning_columns(root)
    assert scanned == 2 and offenders == [str(Path("data") / "features.csv")]


def test_scan_excludes_the_warning_folder_by_relative_path_only(tmp_path):
    root = tmp_path / "imd_warnings" / "repo"                         # an ancestor with that name must not hide files
    write_csv(root / "imd_warnings" / "samples" / "parsed.csv", "imd_warning_id")   # this folder is allowed
    write_csv(root / "other" / "leak.csv", "imd_warning_id")
    scanned, offenders = p.find_warning_columns(root)
    assert scanned == 1 and offenders == [str(Path("other") / "leak.csv")]


def test_scan_ignores_git_internals_and_reports_zero_for_an_empty_tree(tmp_path):
    write_csv(tmp_path / ".git" / "x.csv", "imd_warning_id")
    assert p.find_warning_columns(tmp_path) == (0, [])


# ---------- second review: duplicate ids, state, names, keywords ----------

def test_duplicate_id_outcome_does_not_depend_on_block_order():
    bad = without(GOOD, "VALID TO")                                   # invalid, same id T-1
    for text in (bad + "\n----------\n" + GOOD, GOOD + "\n----------\n" + bad):
        records, rejects = parse_one(text)
        assert records == [] and len(rejects) == 2


def test_rejected_block_still_reserves_its_id_against_a_later_valid_one():
    records, rejects = parse_one(without(GOOD, "VALID TO") + "\n----------\n" + GOOD)
    assert records == []
    assert all("duplicate warning_id: T-1" in r["errors"] for r in rejects)


def test_other_ids_are_unaffected_by_a_duplicate_pair():
    third = replace(GOOD, "ADVISORY-ID", "T-3")
    records, rejects = parse_one(GOOD + "\n----------\n" + GOOD + "\n----------\n" + third)
    assert [r["warning_id"] for r in records] == ["T-3"] and len(rejects) == 2


def test_district_in_another_state_is_never_matched():
    rec = {**valid_record(), "state": "Maharashtra", "districts": ["Guntur", "Pune"]}
    rows = p.to_district_rows([rec], PROJECT)
    assert list(rows["imd_warning_district_match"]) == ["outside_project_state", "outside_project_state"]
    assert (rows["imd_warning_district_code"] == "").all()


@pytest.mark.parametrize("state", ["Andhra Pradesh", "ANDHRA PRADESH", "andhra  pradesh", " Andhra Pradesh "])
def test_state_spelling_variants_match_the_project_state(state):
    rec = {**valid_record(), "state": state, "districts": ["Guntur"]}
    assert p.to_district_rows([rec], PROJECT).loc[0, "imd_warning_district_match"] == "matched"


def test_telangana_is_not_the_project_state():
    rec = {**valid_record(), "state": "Telangana", "districts": ["Guntur"]}
    assert p.to_district_rows([rec], PROJECT).loc[0, "imd_warning_district_match"] == "outside_project_state"


def test_repeated_unmatched_names_are_collapsed_and_reported():
    rec = {**valid_record(), "districts": ["Hyderabad", "hyderabad", "HYDERABAD.", "Guntur"]}
    rows = p.to_district_rows([rec], PROJECT)
    assert list(rows["imd_warning_district_raw"]) == ["Hyderabad", "Guntur"]
    assert [name for _, name in rows.attrs["collapsed_duplicates"]] == ["hyderabad", "HYDERABAD."]


PROJECT_FULL = {
    p.clean_name("ANANTAPURAMU"): (502, "ANANTAPURAMU"), p.clean_name("SPR NELLORE"): (515, "SPR NELLORE"),
    p.clean_name("YSR"): (504, "YSR"), p.clean_name("EAST GODAVARI"): (505, "EAST GODAVARI"),
    p.clean_name("WEST GODAVARI"): (523, "WEST GODAVARI"), p.clean_name("VISAKHAPATNAM"): (520, "VISAKHAPATNAM"),
}


@pytest.mark.parametrize("raw,code", [
    ("Y.S.R. Kadapa", 504), ("Y S R Kadapa", 504), ("YSR Kadapa", 504), ("Y.S.R.", 504),
    ("Ananthapur", 502), ("Ananthapuramu", 502), ("S.P.S.R. Nellore", 515), ("Vizag", 520),
    ("E. Godavari", 505), ("W Godavari", 523),
])
def test_more_district_spellings_match(raw, code):
    hit = p.lookup_district(raw, PROJECT_FULL)
    assert hit is not None and hit[0] == code


def test_project_name_that_equals_an_alias_key_is_not_silently_remapped(tmp_path):
    csv_path = write_districts(tmp_path / "d.csv", [(999, "NELLORE"), (515, "SPR NELLORE")])
    project = p.load_project_districts(csv_path)
    assert set(project) == {"NELLORE", "SPR NELLORE"}                     # keys are cleaned, not aliased
    assert p.lookup_district("Nellore", project)[0] == 999                # an exact project name wins over the alias
    assert p.lookup_district("SPSR Nellore", project)[0] == 515           # the alias still works for other spellings


@pytest.mark.parametrize("phrase,expected", [
    ("Strong surface winds", "strong_wind"), ("Gusty winds expected", "strong_wind"), ("Gale warning", "strong_wind"),
    ("Flash flood warning", "flood"), ("Thunder and lightning", "thunderstorm"), ("Severe heat-wave", "heat_wave"),
    ("Cold-wave conditions", "cold_wave"), ("Very heavy rainfall", "heavy_rain"),
    ("Anticyclonic circulation", "other"),            # not a cyclone: keywords match from the start of a word
    ("Light surface winds", "other"), ("Hot and humid conditions", "other"), ("Storm surge", "other"),
    ("August heavy rain", "heavy_rain"),              # 'august' contains 'gust' but is not a wind keyword
])
def test_extended_keywords_and_word_boundaries(phrase, expected):
    assert p.classify_type(phrase) == expected
