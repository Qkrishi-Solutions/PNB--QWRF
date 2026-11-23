"""Tests for the cyclone catalogue (Issue 46). All data here is synthetic.

Run from the repository root:
    python3 -m pytest cyclone_catalogue/test_cyclone_catalogue.py -v
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_cyclone_catalogue as c  # noqa: E402

KM_PER_DEG_LON = c.KM_PER_DEG * c.COS_LAT0

PROJECT = [
    (502, "ANANTAPURAMU"), (503, "CHITTOOR"), (504, "YSR"), (505, "EAST GODAVARI"), (506, "GUNTUR"),
    (510, "KRISHNA"), (511, "KURNOOL"), (515, "SPR NELLORE"), (517, "PRAKASAM"), (519, "SRIKAKULAM"),
    (520, "VISAKHAPATNAM"), (521, "VIZIANAGARAM"), (523, "WEST GODAVARI"),
]
GB_BY_PROJECT = {v: k for k, v in c.GB_TO_PROJECT.items()}
# Position of each district code in PROJECT. Test squares are laid out in this order, so tests
# look positions up by code instead of hard-coding an index.
IDX = {code: i for i, (code, _) in enumerate(PROJECT)}
PROJECT_CODES = {code for code, _ in PROJECT}


def square(lon0, lat0, size=0.1):
    return [[[lon0, lat0], [lon0 + size, lat0], [lon0 + size, lat0 + size], [lon0, lat0 + size], [lon0, lat0]]]


def district_lon0(index):
    return 77.0 + 0.6 * index


def write_inputs(tmp_path, rows):
    """Districts file, boundaries GeoJSON (13 squares at lat 15.95-16.05) and a mini IBTrACS CSV."""
    districts = tmp_path / "districts.csv"
    pd.DataFrame(PROJECT, columns=["district_code", "district_name"]).to_csv(districts, index=False)

    features = []
    for i, (_, project_name) in enumerate(PROJECT):
        features.append({
            "type": "Feature",
            "properties": {"shapeName": GB_BY_PROJECT[project_name]},
            "geometry": {"type": "Polygon", "coordinates": square(district_lon0(i), 15.95)},
        })
    boundaries = tmp_path / "boundaries.geojson"
    boundaries.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")

    header = ["SID", "SEASON", "NAME", "ISO_TIME", "SUBBASIN", "NEWDELHI_LAT", "NEWDELHI_LON",
              "NEWDELHI_GRADE", "NEWDELHI_WIND", "NEWDELHI_PRES"]
    units = [" ", " ", " ", " ", " ", "degrees_north", "degrees_east", " ", "kts", "mb"]
    lines = [",".join(header), ",".join(units)] + [",".join(str(v) for v in r) for r in rows]
    tracks = tmp_path / "tracks.csv"
    tracks.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tracks, boundaries, districts


def storm(sid, season, name, lat, lon, grade="CS", wind=40, pres=990, subbasin="BB", n=3, start_hour=0):
    """A stationary storm with n points 6 hours apart."""
    rows = []
    for k in range(n):
        hour = start_hour + 6 * k
        rows.append([sid, season, name, f"{season}-10-10 {hour % 24:02d}:00:00", subbasin,
                     lat, lon, grade, wind, pres])
    return rows


# ---------- geometry ----------

def poly(lon0, lat0, size=1.0):
    return c.polygons_from_geometry({"type": "Polygon", "coordinates": square(lon0, lat0, size)})


def test_point_inside_and_outside_square():
    polygons = poly(80.0, 16.0)
    x, y = c.to_xy([80.5, 82.0], [16.5, 16.5])
    inside = c.points_inside(x, y, polygons)
    assert list(inside) == [True, False]


def test_distance_outside_equals_planar_distance_to_edge():
    polygons = poly(80.0, 16.0)
    x, y = c.to_xy([81.1], [16.5])  # 0.1 degrees east of the east edge
    assert c.distance_to_district(x, y, polygons)[0] == pytest.approx(0.1 * KM_PER_DEG_LON, rel=1e-6)
    x, y = c.to_xy([80.5], [17.2])  # 0.2 degrees north of the north edge
    assert c.distance_to_district(x, y, polygons)[0] == pytest.approx(0.2 * c.KM_PER_DEG, rel=1e-6)


def test_distance_inside_is_zero():
    polygons = poly(80.0, 16.0)
    x, y = c.to_xy([80.5], [16.5])
    assert c.distance_to_district(x, y, polygons)[0] == 0.0


def test_hole_is_outside_and_distance_is_to_hole_edge():
    outer = [[80.0, 16.0], [82.0, 16.0], [82.0, 18.0], [80.0, 18.0], [80.0, 16.0]]
    hole = [[80.5, 16.5], [81.5, 16.5], [81.5, 17.5], [80.5, 17.5], [80.5, 16.5]]
    polygons = c.polygons_from_geometry({"type": "Polygon", "coordinates": [outer, hole]})
    x, y = c.to_xy([81.0, 80.25], [17.0, 17.0])
    assert list(c.points_inside(x, y, polygons)) == [False, True]
    d = c.distance_to_district(*c.to_xy([81.0], [17.0]), polygons)[0]
    # Nearest hole edge is the east or west one, 0.5 degrees of longitude away (shorter than 0.5 of latitude).
    assert d == pytest.approx(0.5 * KM_PER_DEG_LON, rel=1e-6)


def test_multipolygon_inside_any_part():
    geom = {"type": "MultiPolygon", "coordinates": [square(80.0, 16.0, 1.0), square(84.0, 16.0, 1.0)]}
    polygons = c.polygons_from_geometry(geom)
    x, y = c.to_xy([80.5, 84.5, 82.5], [16.5, 16.5, 16.5])
    assert list(c.points_inside(x, y, polygons)) == [True, True, False]


def test_unsupported_geometry_rejected():
    with pytest.raises(ValueError):
        c.polygons_from_geometry({"type": "Point", "coordinates": [80, 16]})


# ---------- grades ----------

@pytest.mark.parametrize("grade,expected", [
    ("CS", True), ("SCS", True), ("SCS(H)", True), ("VSCS", True), ("ESCS", True), ("SUCS", True),
    ("D", False), ("DD", False), ("L", False), ("", False),
])
def test_cs_or_above(grade, expected):
    assert (c.normalise_grade(grade) in c.CS_OR_ABOVE) is expected


def test_sucs_spellings_are_the_same_grade():
    assert c.normalise_grade("SuCS") == c.normalise_grade("SUCS") == "SUCS"


# ---------- track sampling ----------

def pts(rows):
    frame = pd.DataFrame(rows, columns=["NEWDELHI_LAT", "NEWDELHI_LON", "time", "NEWDELHI_WIND", "grade"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


def test_interpolation_between_close_cs_points_every_10_km():
    # about 106.9 km apart (1 degree of longitude at 16N), 6 hours apart
    s = c.sample_track(pts([(16.0, 80.0, "2020-01-01 00:00", 40, "CS"), (16.0, 81.0, "2020-01-01 06:00", 50, "SCS")]))
    assert s["original"].sum() == 2
    assert len(s) == 2 + (math.ceil(KM_PER_DEG_LON / c.STEP_KM) - 1)
    x, y = c.to_xy(s.sort_values("lon")["lon"], s.sort_values("lon")["lat"])
    assert np.max(np.hypot(np.diff(x), np.diff(y))) <= c.STEP_KM + 1e-6


def test_interpolated_sample_inherits_nearer_endpoint():
    s = c.sample_track(pts([(16.0, 80.0, "2020-01-01 00:00", 40, "CS"), (16.0, 81.0, "2020-01-01 06:00", 90, "VSCS")]))
    mid = s[~s["original"]].sort_values("lon")
    assert (mid[mid["lon"] < 80.5]["grade"] == "CS").all() and (mid[mid["lon"] < 80.5]["wind_kt"] == 40).all()
    assert (mid[mid["lon"] > 80.5]["grade"] == "VSCS").all()


def test_no_interpolation_across_long_gap_or_non_cs_neighbour():
    gap = c.sample_track(pts([(16.0, 80.0, "2020-01-01 00:00", 40, "CS"), (16.0, 81.0, "2020-01-02 00:00", 40, "CS")]))
    assert len(gap) == 2 and gap["original"].all()
    weak = c.sample_track(pts([(16.0, 80.0, "2020-01-01 00:00", 40, "CS"), (16.0, 81.0, "2020-01-01 06:00", 25, "DD")]))
    assert len(weak) == 1 and weak.iloc[0]["grade"] == "CS"


def test_only_cs_or_above_points_are_sampled():
    s = c.sample_track(pts([(16.0, 80.0, "2020-01-01 00:00", 25, "D"), (16.0, 80.1, "2020-01-01 06:00", 30, "DD")]))
    assert s.empty


def test_missing_wind_stays_missing_when_interpolating():
    s = c.sample_track(pts([(16.0, 80.0, "2020-01-01 00:00", np.nan, "CS"), (16.0, 80.5, "2020-01-01 06:00", np.nan, "CS")]))
    assert s["wind_kt"].isna().all()


# ---------- build: rule, tiers, unmapped, exclusions ----------

def build_from(tmp_path, rows, **kwargs):
    tracks, boundaries, districts = write_inputs(tmp_path, rows)
    return c.build(tracks, boundaries, districts, **kwargs)


def inside_point(index):
    return 16.0, district_lon0(index) + 0.05


def test_track_inside_district_is_tier_inside_with_zero_distance(tmp_path):
    lat, lon = inside_point(IDX[504])
    events, pairs, unmapped, ctx = build_from(tmp_path, storm("S1", 2015, "INSIDER", lat, lon))
    row = pairs[(pairs["event_id"] == "S1") & (pairs["district_code"] == 504)]
    assert len(row) == 1
    assert row.iloc[0]["rule_tier"] == "inside" and row.iloc[0]["min_distance_km"] == 0.0
    # The two neighbouring squares are only about 53 km away, so they are "near", never "inside".
    neighbours = {PROJECT[IDX[504] - 1][0], PROJECT[IDX[504] + 1][0]}
    others = pairs[pairs["district_code"] != 504]
    assert set(others["district_code"]) == neighbours and (others["rule_tier"] == "near").all()
    expected_codes = ";".join(str(x) for x in sorted(neighbours | {504}))
    assert events.iloc[0]["mapping_status"] == "mapped" and events.iloc[0]["affected_district_codes"] == expected_codes


def test_threshold_controls_which_districts_are_affected(tmp_path):
    # 0.54 degrees north of district index 5 (code 510): about 60 km from it, neighbours about 80 km.
    rows = storm("S2", 2016, "NEARBY", 16.05 + 0.54, district_lon0(IDX[510]) + 0.05)
    _, p100, _, _ = build_from(tmp_path, rows, distance_km=100.0)
    assert 510 in set(p100["district_code"])
    near = p100[p100["district_code"] == 510].iloc[0]
    assert near["rule_tier"] == "near" and 59.0 < near["min_distance_km"] < 61.0
    _, p50, _, _ = build_from(tmp_path, rows, distance_km=50.0)
    assert len(p50) == 0
    _, p70, _, _ = build_from(tmp_path, rows, distance_km=70.0)
    assert set(p70["district_code"]) == {510}


def test_pairs_are_one_per_event_and_district_and_checks_pass(tmp_path):
    lat, lon = inside_point(IDX[505])
    rows = storm("S1", 2015, "A", lat, lon) + storm("S3", 2017, "B", 16.0, district_lon0(IDX[505]) + 0.05)
    events, pairs, _, _ = build_from(tmp_path, rows)
    assert not pairs.duplicated(["event_id", "district_code"]).any()
    assert all(ok for _, ok, _ in c.validate(events, pairs))


def test_far_storm_is_unmapped_with_reason(tmp_path):
    events, pairs, unmapped, _ = build_from(tmp_path, storm("S4", 2015, "FAR", 18.5, 80.0))
    assert len(pairs) == 0
    assert events.iloc[0]["mapping_status"] == "unmapped_no_project_district_within_threshold"
    assert list(unmapped["mapping_status"]) == ["unmapped_no_project_district_within_threshold"]


def test_weak_storm_never_cs_is_unmapped_even_if_over_a_district(tmp_path):
    lat, lon = inside_point(IDX[504])
    events, pairs, unmapped, _ = build_from(tmp_path, storm("S5", 2015, "WEAK", lat, lon, grade="DD", wind=25))
    assert len(pairs) == 0
    assert events.iloc[0]["mapping_status"] == "unmapped_never_reached_cyclonic_storm"
    assert events.iloc[0]["n_cs_or_above_points"] == 0


def test_storm_without_imd_positions_is_listed_not_dropped(tmp_path):
    blank = [["S6", 2015, "NOPOS", f"2015-10-10 {h:02d}:00:00", "BB", "", "", "", "", ""] for h in (0, 6)]
    events, pairs, unmapped, ctx = build_from(tmp_path, blank + storm("S1", 2015, "OK", *inside_point(IDX[504])))
    assert "S6" not in set(events["event_id"])
    row = unmapped[unmapped["event_id"] == "S6"]
    assert list(row["mapping_status"]) == ["unmapped_no_imd_track_positions"]
    assert ctx["bay_of_bengal_storms_without_imd_positions"] == 1


def test_arabian_sea_only_storm_is_not_in_catalogue(tmp_path):
    events, _, unmapped, _ = build_from(
        tmp_path, storm("S7", 2015, "ARAB", 16.0, 70.0, subbasin="AS") + storm("S1", 2015, "OK", *inside_point(IDX[504])))
    assert "S7" not in set(events["event_id"]) and "S7" not in set(unmapped["event_id"])


def test_storms_before_start_year_are_excluded(tmp_path):
    rows = storm("OLD", 1985, "OLD", *inside_point(IDX[504])) + storm("NEW", 2015, "NEW", *inside_point(IDX[504]))
    events, _, _, _ = build_from(tmp_path, rows, start_year=1990)
    assert list(events["event_id"]) == ["NEW"]


def test_missing_wind_and_pressure_are_not_filled(tmp_path):
    rows = storm("S8", 2015, "NOWIND", *inside_point(IDX[504]), wind="", pres="")
    events, pairs, _, _ = build_from(tmp_path, rows)
    assert np.isnan(events.iloc[0]["max_imd_wind_kt"]) and np.isnan(events.iloc[0]["min_imd_pressure_hpa"])
    assert np.isnan(pairs.iloc[0]["max_imd_wind_kt_within"])


def test_event_summary_fields(tmp_path):
    rows = storm("S9", 2018, "SUMMARY", *inside_point(IDX[504]), grade="SCS", wind=60, pres=980, n=3)
    events, _, _, _ = build_from(tmp_path, rows)
    e = events.iloc[0]
    assert e["n_imd_points"] == 3 and e["n_cs_or_above_points"] == 3
    assert e["max_imd_wind_kt"] == 60 and e["min_imd_pressure_hpa"] == 980
    assert e["first_time_utc"] == "2018-10-10 00:00" and e["last_time_utc"] == "2018-10-10 12:00"
    assert e["grades_observed"] == "SCS" and e["subbasins_visited"] == "BB"


# ---------- input guards ----------

def test_wrong_units_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    tracks.write_text(tracks.read_text().replace("kts", "m/s"), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected source units"):
        c.build(tracks, boundaries, districts)


def test_non_numeric_value_in_numeric_column_rejected(tmp_path):
    rows = storm("S1", 2015, "A", *inside_point(IDX[504]), wind="fast")
    with pytest.raises(ValueError, match="Non-numeric"):
        build_from(tmp_path, rows)


def test_duplicate_time_rows_rejected(tmp_path):
    rows = storm("S1", 2015, "A", *inside_point(IDX[504]), n=1) * 2
    with pytest.raises(ValueError, match="Duplicate"):
        build_from(tmp_path, rows)


def test_missing_boundary_district_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    gj = json.loads(boundaries.read_text())
    gj["features"] = gj["features"][:-1]
    boundaries.write_text(json.dumps(gj))
    with pytest.raises(ValueError, match="Missing"):
        c.build(tracks, boundaries, districts)


def test_boundary_outside_andhra_pradesh_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    gj = json.loads(boundaries.read_text())
    gj["features"][0]["geometry"]["coordinates"] = square(70.0, 25.0)
    boundaries.write_text(json.dumps(gj))
    with pytest.raises(ValueError, match="sanity box"):
        c.build(tracks, boundaries, districts)


def test_district_file_with_duplicate_codes_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    pd.DataFrame([(1, "GUNTUR"), (1, "KRISHNA")], columns=["district_code", "district_name"]).to_csv(districts, index=False)
    with pytest.raises(ValueError):
        c.build(tracks, boundaries, districts)


# ---------- validation function catches problems ----------

def test_validate_detects_duplicate_pairs_and_unknown_events(tmp_path):
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    assert all(ok for _, ok, _ in c.validate(events, pairs))
    bad = pd.concat([pairs, pairs], ignore_index=True)
    assert not dict((n, ok) for n, ok, _ in c.validate(events, bad))["pair_key_unique"]
    orphan = pairs.assign(event_id="NOT_IN_CATALOGUE")
    assert not dict((n, ok) for n, ok, _ in c.validate(events, orphan))["pairs_reference_catalogue"]


def test_spot_check_table_flags_missing_and_found_events(tmp_path):
    # A storm placed in ANANTAPURAMU (502) is far from the expected VISAKHAPATNAM (520), so the check fails.
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2014, "HUDHUD", 16.0, district_lon0(IDX[502]) + 0.05))
    table = c.spot_checks(events, pairs)
    hudhud = table[table["event"] == "HUDHUD"].iloc[0]
    assert bool(hudhud["found_in_catalogue"]) and not bool(hudhud["all_expected_found"])
    # The same storm placed in VISAKHAPATNAM (520) meets the expectation.
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2014, "HUDHUD", 16.0, district_lon0(IDX[520]) + 0.05))
    hudhud = c.spot_checks(events, pairs).query("event == 'HUDHUD'").iloc[0]
    assert bool(hudhud["all_expected_found"]) and hudhud["closest_expected_distance_km"] == 0.0
    assert not table[table["event"] == "PHETHAI"].iloc[0]["found_in_catalogue"]


# ---------- review fixes: threshold, cap and input validation ----------

@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf")])
def test_non_positive_or_non_finite_threshold_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="distance_km"):
        build_from(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])), distance_km=bad)


def test_threshold_above_default_cap_is_honoured_not_truncated(tmp_path):
    # The storm sits about 272 km north of the district squares (lat 18.5 vs top edge 16.05).
    rows = storm("S2", 2016, "FAR", 18.5, district_lon0(IDX[510]) + 0.05)
    _, p_default, _, _ = build_from(tmp_path, rows)
    _, p_250, _, _ = build_from(tmp_path, rows, distance_km=250.0)
    _, p_300, _, _ = build_from(tmp_path, rows, distance_km=300.0)
    assert len(p_default) == 0 and len(p_250) == 0
    row = p_300[p_300["district_code"] == 510].iloc[0]
    assert 270 < row["min_distance_km"] < 275 and row["distance_threshold_km"] == 300.0


# ---------- review fixes: names ----------

def test_pick_name_uses_real_name_even_if_first_rows_are_unnamed():
    assert c.pick_name(pd.Series(["UNNAMED", "UNNAMED", "KAVYA", "KAVYA", "KAVYA"])) == "KAVYA"
    assert c.pick_name(pd.Series(["NOT_NAMED", ""])) == "UNNAMED"
    assert c.pick_name(pd.Series(["UNNAMED"])) == "UNNAMED"
    # a tie is broken alphabetically so that the result never depends on row order
    assert c.pick_name(pd.Series(["BETA", "ALPHA"])) == c.pick_name(pd.Series(["ALPHA", "BETA"])) == "ALPHA"


def test_storm_named_after_first_rows_gets_its_name_in_catalogue(tmp_path):
    rows = storm("S1", 2015, "UNNAMED", *inside_point(IDX[504]), n=1, start_hour=0)
    rows += storm("S1", 2015, "LATER", *inside_point(IDX[504]), n=2, start_hour=6)
    events, _, _, _ = build_from(tmp_path, rows)
    assert events.iloc[0]["name"] == "LATER"


def test_storm_id_with_two_seasons_rejected(tmp_path):
    rows = storm("S1", 2015, "A", *inside_point(IDX[504]), n=1) + storm("S1", 2016, "A", *inside_point(IDX[504]), n=1, start_hour=6)
    with pytest.raises(ValueError, match="more than one season"):
        build_from(tmp_path, rows)


# ---------- review fixes: skipped gaps and observed-only values ----------

def test_skipped_long_gaps_are_counted():
    s = c.sample_track(pts([
        (16.0, 80.0, "2020-01-01 00:00", 40, "CS"), (16.0, 81.0, "2020-01-02 00:00", 40, "CS"),   # 24 h gap, skipped
        (16.0, 81.2, "2020-01-02 06:00", 40, "CS"),                                                # 6 h, interpolated
    ]))
    assert s.attrs["skipped_long_gaps"] == 1
    ok = c.sample_track(pts([(16.0, 80.0, "2020-01-01 00:00", 40, "CS"), (16.0, 80.5, "2020-01-01 06:00", 40, "CS")]))
    assert ok.attrs["skipped_long_gaps"] == 0


def test_events_table_reports_skipped_segments(tmp_path):
    rows = (storm("S1", 2015, "GAPPY", 18.5, 80.0, n=1, start_hour=0)
            + [["S1", 2015, "GAPPY", "2015-10-12 00:00:00", "BB", 18.5, 80.5, "CS", 40, 990]])  # 48 h later
    events, _, _, ctx = build_from(tmp_path, rows)
    assert events.iloc[0]["n_cs_segments_not_interpolated"] == 1
    assert ctx["events_with_skipped_long_gaps"] == 1 and ctx["cs_segments_not_interpolated"] == 1


def test_peak_wind_and_closest_flag_use_observed_points_only(tmp_path):
    k = IDX[510]
    lon0 = district_lon0(k)
    # Two observed CS points about 0.45 degrees either side of the square: wind 40 west, 90 east.
    rows = [
        ["S1", 2015, "BORROW", "2015-10-10 00:00:00", "BB", 16.0, lon0 - 0.45, "CS", 40, 990],
        ["S1", 2015, "BORROW", "2015-10-10 06:00:00", "BB", 16.0, lon0 + 0.55, "VSCS", 90, 960],
    ]
    # With a 30 km threshold no observed point is within range; only interpolated samples reach the square.
    _, pairs, _, _ = build_from(tmp_path, rows, distance_km=30.0)
    row = pairs[pairs["district_code"] == 510].iloc[0]
    assert row["rule_tier"] == "inside" and row["n_imd_points_within"] == 0
    assert np.isnan(row["max_imd_wind_kt_within"])      # borrowed wind is not reported as observed
    assert not row["closest_point_is_observed"]
    # With a 100 km threshold both observed points count, so the observed maximum is reported.
    _, pairs100, _, _ = build_from(tmp_path, rows, distance_km=100.0)
    assert pairs100[pairs100["district_code"] == 510].iloc[0]["max_imd_wind_kt_within"] == 90.0


# ---------- review fixes: validation and spot checks ----------

def test_validate_rejects_district_code_outside_project_list(tmp_path):
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    assert all(ok for _, ok, _ in c.validate(events, pairs, project_codes=PROJECT_CODES))
    bad = pairs.assign(district_code=999)
    result = {n: ok for n, ok, _ in c.validate(events, bad, project_codes=PROJECT_CODES)}
    assert not result["district_codes_valid"]


def test_validate_runs_pair_checks_on_empty_map(tmp_path):
    events, pairs, _, _ = build_from(tmp_path, storm("S4", 2015, "FAR", 18.5, 80.0))
    assert len(pairs) == 0
    names = [n for n, _, _ in c.validate(events, pairs, project_codes=PROJECT_CODES)]
    for expected in ("pair_key_unique", "district_codes_valid", "affected_counts_agree", "affected_codes_agree"):
        assert expected in names
    # An empty map must still contradict a catalogue that claims affected districts.
    claims = events.assign(n_affected_districts=2, affected_district_codes="504;505", mapping_status="mapped")
    result = {n: ok for n, ok, _ in c.validate(claims, pairs, project_codes=PROJECT_CODES)}
    assert not result["affected_counts_agree"] and not result["affected_codes_agree"]


def test_validate_distances_check_can_fail(tmp_path):
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    too_far = pairs.assign(min_distance_km=pairs["distance_threshold_km"] + 1, rule_tier="near")
    assert not {n: ok for n, ok, _ in c.validate(events, too_far)}["distances_valid"]
    negative = pairs.assign(min_distance_km=-1.0)
    assert not {n: ok for n, ok, _ in c.validate(events, negative)}["distances_valid"]


def test_validate_detects_catalogue_codes_that_differ_from_map(tmp_path):
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    tampered = events.assign(affected_district_codes="999;998;997")
    assert not {n: ok for n, ok, _ in c.validate(tampered, pairs)}["affected_codes_agree"]


def test_spot_check_matches_compound_names_and_sets_status(tmp_path):
    rows = storm("S1", 2021, "GULAB:SHAHEEN-GU", 16.0, district_lon0(IDX[519]) + 0.05)
    events, pairs, _, _ = build_from(tmp_path, rows)
    table = c.spot_checks(events, pairs, start_year=1990).set_index("event")
    assert table.loc["GULAB", "status"] == "pass" and bool(table.loc["GULAB", "found_in_catalogue"])
    assert table.loc["HUDHUD", "status"] == "not_in_catalogue"
    # wrong district -> fail
    rows = storm("S1", 2021, "GULAB:SHAHEEN-GU", 16.0, district_lon0(IDX[502]) + 0.05)
    events, pairs, _, _ = build_from(tmp_path, rows)
    assert c.spot_checks(events, pairs).set_index("event").loc["GULAB", "status"] == "fail"


def test_spot_checks_before_start_year_are_skipped_not_failed(tmp_path):
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2023, "MICHAUNG", *inside_point(IDX[506])), start_year=2015)
    table = c.spot_checks(events, pairs, start_year=2015).set_index("event")
    assert table.loc["LAILA", "status"] == "skipped_before_start_year"     # 2010
    assert table.loc["HUDHUD", "status"] == "skipped_before_start_year"    # 2014
    assert table.loc["MICHAUNG", "status"] == "pass"
    assert not table["status"].isin(["fail"]).any()


# ---------- second review: spot checks are unverified and non-fatal by default ----------

def run_main(monkeypatch, tmp_path, rows, *extra):
    tracks, boundaries, districts = write_inputs(tmp_path, rows)
    out = tmp_path / "out"
    argv = ["build_cyclone_catalogue.py", "--tracks", str(tracks), "--boundaries", str(boundaries),
            "--districts", str(districts), "--outdir", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    c.main()
    return out


def test_spot_check_mismatch_only_warns_by_default(monkeypatch, tmp_path, capsys):
    # HUDHUD placed in ANANTAPURAMU: valid data, but not where the (unverified) expectation says.
    rows = storm("S1", 2014, "HUDHUD", 16.0, district_lon0(IDX[502]) + 0.05)
    out = run_main(monkeypatch, tmp_path, rows)           # must not raise SystemExit
    assert "WARNING: Spot check did not match" in capsys.readouterr().out
    table = pd.read_csv(out / "cyclone_spot_checks_v1.csv").set_index("event")
    assert table.loc["HUDHUD", "status"] == "fail"
    assert (out / "validation_checks.csv").exists() and (out / "cyclone_run_summary_v1.csv").exists()


def test_strict_flag_makes_spot_check_mismatch_fatal(monkeypatch, tmp_path):
    rows = storm("S1", 2014, "HUDHUD", 16.0, district_lon0(IDX[502]) + 0.05)
    with pytest.raises(SystemExit, match="Spot check did not match"):
        run_main(monkeypatch, tmp_path, rows, "--strict-spot-checks")


def test_spot_check_csv_labels_expectations_as_unverified(monkeypatch, tmp_path):
    out = run_main(monkeypatch, tmp_path, storm("S1", 2014, "HUDHUD", 16.0, district_lon0(IDX[520]) + 0.05))
    table = pd.read_csv(out / "cyclone_spot_checks_v1.csv")
    assert (table["expectation_verified"] == False).all()  # noqa: E712
    assert table["expectation_source"].str.contains("unverified").all()


def test_two_events_with_the_same_name_and_season_are_ambiguous(tmp_path):
    rows = (storm("A1", 2018, "TITLI", 16.0, district_lon0(IDX[519]) + 0.05)
            + storm("A2", 2018, "TITLI", 18.5, 80.0))           # second fragment maps to nothing
    events, pairs, _, _ = build_from(tmp_path, rows)
    titli = c.spot_checks(events, pairs).set_index("event").loc["TITLI"]
    assert titli["status"] == "ambiguous_multiple_events" and titli["n_matching_events"] == 2
    assert "SRIKAKULAM" in titli["found_districts"]               # both events are considered, not just the first


# ---------- second review: rounding, empty input, grade versus wind ----------

def test_tiny_outside_distance_is_rounded_up_not_to_zero(tmp_path):
    # 0.03 km east of the east edge of the square (lon0 + 0.1).
    k = IDX[510]
    lon = district_lon0(k) + 0.1 + 0.03 / KM_PER_DEG_LON
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2015, "EDGE", 16.0, lon, n=1), distance_km=10.0)
    row = pairs[pairs["district_code"] == 510].iloc[0]
    assert row["rule_tier"] == "near" and row["min_distance_km"] == 0.1
    assert all(ok for _, ok, _ in c.validate(events, pairs, project_codes=PROJECT_CODES))


def test_track_exactly_on_a_boundary_is_inside_with_zero_distance(tmp_path):
    k = IDX[510]
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2015, "ONEDGE", 16.0, district_lon0(k), n=1), distance_km=10.0)
    row = pairs[pairs["district_code"] == 510].iloc[0]
    assert row["rule_tier"] == "inside" and row["min_distance_km"] == 0.0
    assert all(ok for _, ok, _ in c.validate(events, pairs, project_codes=PROJECT_CODES))


def test_distance_check_allows_only_the_rounding_margin():
    events = pd.DataFrame([{"event_id": "E", "n_affected_districts": 1, "affected_district_codes": "510",
                            "mapping_status": "mapped", "subbasins_visited": "BB"}])
    pairs = pd.DataFrame([{"event_id": "E", "district_code": 510, "rule_tier": "near",
                           "min_distance_km": 50.1, "distance_threshold_km": 50.06}])
    assert {n: ok for n, ok, _ in c.validate(events, pairs)}["distances_valid"]
    assert not {n: ok for n, ok, _ in c.validate(events, pairs.assign(min_distance_km=50.3))}["distances_valid"]


def test_empty_catalogue_writes_empty_outputs_instead_of_crashing(monkeypatch, tmp_path):
    out = run_main(monkeypatch, tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])), "--start-year", "2100")
    events = pd.read_csv(out / "cyclone_events_v1.csv")
    pairs = pd.read_csv(out / "cyclone_district_map_v1.csv")
    assert list(events.columns) == c.EVENT_COLUMNS and len(events) == 0
    assert list(pairs.columns) == c.PAIR_OUTPUT_COLUMNS and len(pairs) == 0
    checks = pd.read_csv(out / "validation_checks.csv")
    assert (checks["status"] == "PASS").all()
    spot = pd.read_csv(out / "cyclone_spot_checks_v1.csv")
    assert (spot["status"] == "skipped_before_start_year").all()


def test_grade_decides_when_present_and_wind_only_when_grade_is_blank(tmp_path):
    # Two CS points, one blank-grade point at 35 kt (counts by wind), one grade D at 40 kt (grade wins,
    # reported but not used), one DD at 30 kt (below both).
    rows = (storm("S1", 2015, "GRADES", *inside_point(IDX[504]), grade="CS", wind=40, n=2)
            + [["S1", 2015, "GRADES", "2015-10-11 00:00:00", "BB", 16.0, 79.0, "D", 40, 990],
               ["S1", 2015, "GRADES", "2015-10-11 06:00:00", "BB", 16.0, 79.0, "", 35, 990],
               ["S1", 2015, "GRADES", "2015-10-11 12:00:00", "BB", 16.0, 79.0, "DD", 30, 990]])
    events, _, _, ctx = build_from(tmp_path, rows)
    assert ctx["points_cs_by_wind_only_blank_grade"] == 1 and ctx["events_with_points_cs_by_wind_only"] == 1
    assert ctx["points_wind_ge_34kt_with_grade_below_cs_not_used"] == 1
    assert ctx["events_with_wind_ge_34kt_grade_below_cs_not_used"] == 1
    assert events.iloc[0]["n_cs_or_above_points"] == 3 and events.iloc[0]["n_cs_points_by_wind_only"] == 1


@pytest.mark.parametrize("grade,wind,expected", [
    ("", 34, True), ("", 127, True), ("", 33.9, False), ("", np.nan, False),   # blank grade: wind decides, missing is not CS
    ("D", 80, False), ("DD", 50, False), ("L", 60, False),                      # a present low grade always wins
    ("CS", np.nan, True), ("CS", 20, True), ("SUCS", 130, True),                # a CS-or-above grade needs no wind
])
def test_cs_or_above_mask_rule(grade, wind, expected):
    frame = pd.DataFrame({"grade": [grade], "NEWDELHI_WIND": [wind]})
    assert bool(c.cs_or_above_mask(frame).iloc[0]) is expected


def test_blank_grade_storm_with_high_wind_is_mapped_and_labelled_by_wind(tmp_path):
    # Like the early-1990s rows of the source file: no grade at all, but a strong IMD wind.
    rows = storm("OLD1", 1991, "UNNAMED", *inside_point(IDX[510]), grade="", wind=100, n=3)
    events, pairs, unmapped, _ = build_from(tmp_path, rows)
    e = events.iloc[0]
    assert e["mapping_status"] == "mapped" and e["n_cs_points_by_wind_only"] == 3 and len(unmapped) == 0
    row = pairs[pairs["district_code"] == 510].iloc[0]
    assert row["rule_tier"] == "inside" and row["grade_at_closest"] == c.WIND_BASIS_LABEL
    assert row["max_imd_wind_kt_within"] == 100.0


def test_blank_grade_storm_with_weak_or_missing_wind_stays_unmapped(tmp_path):
    rows = (storm("W1", 1991, "WEAK", *inside_point(IDX[510]), grade="", wind=30, n=2)
            + storm("W2", 1991, "NOWIND", *inside_point(IDX[510]), grade="", wind="", n=2))
    events, pairs, unmapped, _ = build_from(tmp_path, rows)
    assert len(pairs) == 0
    assert set(events["mapping_status"]) == {"unmapped_never_reached_cyclonic_storm"}


def test_run_summary_numbers_come_from_the_outputs(tmp_path):
    rows = storm("S1", 2016, "A", *inside_point(IDX[504])) + storm("S4", 2016, "FAR", 18.5, 80.0)
    events, pairs, unmapped, ctx = build_from(tmp_path, rows)
    summary = c.run_summary(ctx, events, pairs, unmapped).set_index("metric")["value"]
    assert summary["catalogue_events"] == 2 and summary["events_mapped"] == 1
    assert summary["event_district_pairs"] == len(pairs)
    assert summary["pairs_inside"] + summary["pairs_near"] == len(pairs)
    assert summary["pairs_closest_point_observed"] + summary["pairs_closest_point_interpolated"] == len(pairs)
    assert summary["pairs_within_100_km"] == len(pairs)
    assert summary["pairs_within_50_km"] <= summary["pairs_within_100_km"] <= summary["pairs_within_150_km"]
    assert summary["mapped_events_2015_to_2019"] == 1 and summary["events_2015_to_2019"] == 2


# ---------- second review: projection and boundary guards ----------

def test_distance_uses_a_projection_centred_on_the_district(tmp_path):
    """A district near 19 N must not be measured with the 16 N scale (that over-states east-west distances by 1.7 percent)."""
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    gj = json.loads(boundaries.read_text())
    for feature in gj["features"]:
        if feature["properties"]["shapeName"] == "Srikakulam":
            feature["geometry"]["coordinates"] = square(80.0, 19.0, 0.1)   # square at about 19.05 N
    boundaries.write_text(json.dumps(gj))
    project_codes = c.load_project_districts(districts)
    name, polygons, lat0 = c.load_district_polygons(boundaries, project_codes)[519]
    assert name == "SRIKAKULAM" and lat0 == pytest.approx(19.04, abs=0.02)
    east_deg = 100.0 / (c.KM_PER_DEG * math.cos(math.radians(19.05)))       # a true 100 km east of the east edge
    x, y = c.to_xy([80.1 + east_deg], [19.05], lat0)
    assert c.distance_to_district(x, y, polygons)[0] == pytest.approx(100.0, abs=0.5)
    x16, y16 = c.to_xy([80.1 + east_deg], [19.05])                          # fixed 16 N scale, for contrast
    fixed = c.distance_to_district(x16, y16, c.polygons_from_geometry({"type": "Polygon", "coordinates": square(80.0, 19.0, 0.1)}))[0]
    assert fixed > 101.0


def test_boundary_file_with_a_duplicated_district_name_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    gj = json.loads(boundaries.read_text())
    gj["features"].append(dict(gj["features"][0]))                          # same shapeName twice
    boundaries.write_text(json.dumps(gj))
    with pytest.raises(ValueError, match="duplicated"):
        c.build(tracks, boundaries, districts)


def test_boundary_with_unsupported_geometry_type_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(IDX[504])))
    gj = json.loads(boundaries.read_text())
    gj["features"][0]["geometry"] = {"type": "Point", "coordinates": [80.0, 16.0]}
    boundaries.write_text(json.dumps(gj))
    with pytest.raises(ValueError, match="Unsupported geometry"):
        c.build(tracks, boundaries, districts)
