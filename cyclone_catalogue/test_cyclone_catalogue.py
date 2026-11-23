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
GB_NAMES = list(c.GB_TO_PROJECT)  # same order as PROJECT's districts by construction below
NAME_BY_CODE = {code: name for code, name in PROJECT}
GB_BY_PROJECT = {v: k for k, v in c.GB_TO_PROJECT.items()}


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
    lat, lon = inside_point(2)
    events, pairs, unmapped, ctx = build_from(tmp_path, storm("S1", 2015, "INSIDER", lat, lon))
    row = pairs[(pairs["event_id"] == "S1") & (pairs["district_code"] == 504)]
    assert len(row) == 1
    assert row.iloc[0]["rule_tier"] == "inside" and row.iloc[0]["min_distance_km"] == 0.0
    # The two neighbouring squares are only about 53 km away, so they are "near", never "inside".
    others = pairs[pairs["district_code"] != 504]
    assert set(others["district_code"]) == {503, 505} and (others["rule_tier"] == "near").all()
    assert events.iloc[0]["mapping_status"] == "mapped" and events.iloc[0]["affected_district_codes"] == "503;504;505"


def test_threshold_controls_which_districts_are_affected(tmp_path):
    # 0.54 degrees north of district index 5 (code 510): about 60 km from it, neighbours about 80 km.
    rows = storm("S2", 2016, "NEARBY", 16.05 + 0.54, district_lon0(5) + 0.05)
    _, p100, _, _ = build_from(tmp_path, rows, distance_km=100.0)
    assert 510 in set(p100["district_code"])
    near = p100[p100["district_code"] == 510].iloc[0]
    assert near["rule_tier"] == "near" and 59.0 < near["min_distance_km"] < 61.0
    _, p50, _, _ = build_from(tmp_path, rows, distance_km=50.0)
    assert len(p50) == 0
    _, p70, _, _ = build_from(tmp_path, rows, distance_km=70.0)
    assert set(p70["district_code"]) == {510}


def test_pairs_are_one_per_event_and_district_and_checks_pass(tmp_path):
    lat, lon = inside_point(3)
    rows = storm("S1", 2015, "A", lat, lon) + storm("S3", 2017, "B", 16.0, district_lon0(3) + 0.05)
    events, pairs, _, _ = build_from(tmp_path, rows)
    assert not pairs.duplicated(["event_id", "district_code"]).any()
    assert all(ok for _, ok, _ in c.validate(events, pairs))


def test_far_storm_is_unmapped_with_reason(tmp_path):
    events, pairs, unmapped, _ = build_from(tmp_path, storm("S4", 2015, "FAR", 18.5, 80.0))
    assert len(pairs) == 0
    assert events.iloc[0]["mapping_status"] == "unmapped_no_project_district_within_threshold"
    assert list(unmapped["mapping_status"]) == ["unmapped_no_project_district_within_threshold"]


def test_weak_storm_never_cs_is_unmapped_even_if_over_a_district(tmp_path):
    lat, lon = inside_point(2)
    events, pairs, unmapped, _ = build_from(tmp_path, storm("S5", 2015, "WEAK", lat, lon, grade="DD", wind=25))
    assert len(pairs) == 0
    assert events.iloc[0]["mapping_status"] == "unmapped_never_reached_cyclonic_storm_grade"
    assert events.iloc[0]["n_cs_or_above_points"] == 0


def test_storm_without_imd_positions_is_listed_not_dropped(tmp_path):
    blank = [["S6", 2015, "NOPOS", f"2015-10-10 {h:02d}:00:00", "BB", "", "", "", "", ""] for h in (0, 6)]
    events, pairs, unmapped, ctx = build_from(tmp_path, blank + storm("S1", 2015, "OK", *inside_point(2)))
    assert "S6" not in set(events["event_id"])
    row = unmapped[unmapped["event_id"] == "S6"]
    assert list(row["mapping_status"]) == ["unmapped_no_imd_track_positions"]
    assert ctx["bay_of_bengal_storms_without_imd_positions"] == 1


def test_arabian_sea_only_storm_is_not_in_catalogue(tmp_path):
    events, _, unmapped, _ = build_from(
        tmp_path, storm("S7", 2015, "ARAB", 16.0, 70.0, subbasin="AS") + storm("S1", 2015, "OK", *inside_point(2)))
    assert "S7" not in set(events["event_id"]) and "S7" not in set(unmapped["event_id"])


def test_storms_before_start_year_are_excluded(tmp_path):
    rows = storm("OLD", 1985, "OLD", *inside_point(2)) + storm("NEW", 2015, "NEW", *inside_point(2))
    events, _, _, _ = build_from(tmp_path, rows, start_year=1990)
    assert list(events["event_id"]) == ["NEW"]


def test_missing_wind_and_pressure_are_not_filled(tmp_path):
    rows = storm("S8", 2015, "NOWIND", *inside_point(2), wind="", pres="")
    events, pairs, _, _ = build_from(tmp_path, rows)
    assert np.isnan(events.iloc[0]["max_imd_wind_kt"]) and np.isnan(events.iloc[0]["min_imd_pressure_hpa"])
    assert np.isnan(pairs.iloc[0]["max_imd_wind_kt_within"])


def test_event_summary_fields(tmp_path):
    rows = storm("S9", 2018, "SUMMARY", *inside_point(2), grade="SCS", wind=60, pres=980, n=3)
    events, _, _, _ = build_from(tmp_path, rows)
    e = events.iloc[0]
    assert e["n_imd_points"] == 3 and e["n_cs_or_above_points"] == 3
    assert e["max_imd_wind_kt"] == 60 and e["min_imd_pressure_hpa"] == 980
    assert e["first_time_utc"] == "2018-10-10 00:00" and e["last_time_utc"] == "2018-10-10 12:00"
    assert e["grades_observed"] == "SCS" and e["subbasins_visited"] == "BB"


# ---------- input guards ----------

def test_wrong_units_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(2)))
    tracks.write_text(tracks.read_text().replace("kts", "m/s"), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected source units"):
        c.build(tracks, boundaries, districts)


def test_non_numeric_value_in_numeric_column_rejected(tmp_path):
    rows = storm("S1", 2015, "A", *inside_point(2), wind="fast")
    with pytest.raises(ValueError, match="Non-numeric"):
        build_from(tmp_path, rows)


def test_duplicate_time_rows_rejected(tmp_path):
    rows = storm("S1", 2015, "A", *inside_point(2), n=1) * 2
    with pytest.raises(ValueError, match="Duplicate"):
        build_from(tmp_path, rows)


def test_missing_boundary_district_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(2)))
    gj = json.loads(boundaries.read_text())
    gj["features"] = gj["features"][:-1]
    boundaries.write_text(json.dumps(gj))
    with pytest.raises(ValueError, match="Missing"):
        c.build(tracks, boundaries, districts)


def test_boundary_outside_andhra_pradesh_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(2)))
    gj = json.loads(boundaries.read_text())
    gj["features"][0]["geometry"]["coordinates"] = square(70.0, 25.0)
    boundaries.write_text(json.dumps(gj))
    with pytest.raises(ValueError, match="sanity box"):
        c.build(tracks, boundaries, districts)


def test_district_file_with_duplicate_codes_rejected(tmp_path):
    tracks, boundaries, districts = write_inputs(tmp_path, storm("S1", 2015, "A", *inside_point(2)))
    pd.DataFrame([(1, "GUNTUR"), (1, "KRISHNA")], columns=["district_code", "district_name"]).to_csv(districts, index=False)
    with pytest.raises(ValueError):
        c.build(tracks, boundaries, districts)


# ---------- validation function catches problems ----------

def test_validate_detects_duplicate_pairs_and_unknown_events(tmp_path):
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2015, "A", *inside_point(2)))
    assert all(ok for _, ok, _ in c.validate(events, pairs))
    bad = pd.concat([pairs, pairs], ignore_index=True)
    assert not dict((n, ok) for n, ok, _ in c.validate(events, bad))["pair_key_unique"]
    orphan = pairs.assign(event_id="NOT_IN_CATALOGUE")
    assert not dict((n, ok) for n, ok, _ in c.validate(events, orphan))["pairs_reference_catalogue"]


def test_spot_check_table_flags_missing_and_found_events(tmp_path):
    # Index 0 is ANANTAPURAMU, far from the expected VISAKHAPATNAM (index 10), so the expectation fails.
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2014, "HUDHUD", 16.0, district_lon0(0) + 0.05))
    table = c.spot_checks(events, pairs)
    hudhud = table[table["event"] == "HUDHUD"].iloc[0]
    assert bool(hudhud["found_in_catalogue"]) and not bool(hudhud["all_expected_found"])
    # Index 10 is VISAKHAPATNAM, so the same storm placed there meets the expectation.
    events, pairs, _, _ = build_from(tmp_path, storm("S1", 2014, "HUDHUD", 16.0, district_lon0(10) + 0.05))
    hudhud = c.spot_checks(events, pairs).query("event == 'HUDHUD'").iloc[0]
    assert bool(hudhud["all_expected_found"]) and hudhud["closest_expected_distance_km"] == 0.0
    assert not table[table["event"] == "PHETHAI"].iloc[0]["found_in_catalogue"]
