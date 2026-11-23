"""Cyclone event catalogue and track-to-district map (Issue 46).

Inputs (public, not committed):
  * IBTrACS v04r01, North Indian Ocean CSV. The IMD (RSMC New Delhi) fields
    NEWDELHI_LAT / NEWDELHI_LON / NEWDELHI_GRADE / NEWDELHI_WIND / NEWDELHI_PRES
    are the IMD track data; they are the only position and intensity fields used.
  * geoBoundaries IND ADM2, which still has the 13 Andhra Pradesh districts
    that the project uses (pre-2022 boundaries). Its licence and attribution
    terms must be confirmed before the outputs are distributed (see README).
  * The project district file (district_code, district_name).

AFFECTED-DISTRICT RULE (assumption, there is no authoritative IMD definition):
  A district is "affected" by an event if the IMD best track, at a time when
  IMD graded it Cyclonic Storm (CS) or higher, came within --distance-km
  (default 100 km) of the district polygon, or inside it. Tracks are linearly
  interpolated between consecutive IMD points (both CS or higher, at most 12 h
  apart) every <= 10 km so that a track passing between two reports is not
  missed. Tier `inside` means a sample point fell in the district polygon;
  tier `near` means the closest approach was within the threshold.

  This is a track-proximity rule. It is not a wind-field, surge or rainfall
  footprint, and it is not evidence of damage or loss.

Nothing is imputed: points without IMD coordinates are not used, a missing
wind or pressure stays missing, and events that cannot be mapped are listed
with a reason.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

KM_PER_DEG = 111.195          # 2 * pi * 6371 / 360
PROJECTION_LAT0 = 16.0        # centre of the Andhra Pradesh coast; planar approximation
COS_LAT0 = math.cos(math.radians(PROJECTION_LAT0))
STEP_KM = 10.0
MAX_GAP_HOURS = 12.0
DISTANCE_CAP_KM = 200.0       # distances beyond this are not computed
SENSITIVITY_KM = (50, 100, 150)

# IMD grades at or above Cyclonic Storm (>= 34 kt). D, DD and L are below it.
CS_OR_ABOVE = {"CS", "SCS", "SCS(H)", "VSCS", "ESCS", "SUCS"}

# geoBoundaries shapeName -> project district name (exact match only).
GB_TO_PROJECT = {
    "Anantapur": "ANANTAPURAMU",
    "Chittoor": "CHITTOOR",
    "Kadapa(YSR)": "YSR",
    "East Godavari": "EAST GODAVARI",
    "Guntur": "GUNTUR",
    "Krishna": "KRISHNA",
    "Kurnool": "KURNOOL",
    "Sri Potti Sriramulu Nellore": "SPR NELLORE",
    "Prakasam": "PRAKASAM",
    "Srikakulam": "SRIKAKULAM",
    "Visakhapatnam": "VISAKHAPATNAM",
    "Vizianagaram": "VIZIANAGARAM",
    "West Godavari": "WEST GODAVARI",
}
AP_BBOX = (76.0, 12.0, 86.0, 20.5)  # lon_min, lat_min, lon_max, lat_max sanity box


def normalise_grade(value):
    """IBTrACS writes the super cyclonic storm grade as both 'SuCS' and 'SUCS'."""
    text = str(value).strip()
    return "SUCS" if text.upper() == "SUCS" else text


def to_xy(lon, lat):
    """Equirectangular projection to kilometres around the AP coast."""
    return (np.asarray(lon, float) * KM_PER_DEG * COS_LAT0, np.asarray(lat, float) * KM_PER_DEG)


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

def _ring_to_xy(ring):
    arr = np.asarray(ring, float)
    x, y = to_xy(arr[:, 0], arr[:, 1])
    pts = np.column_stack([x, y])
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def polygons_from_geometry(geometry):
    """Return a list of polygons; each polygon is a list of rings (outer first)."""
    kind, coords = geometry["type"], geometry["coordinates"]
    if kind == "Polygon":
        coords = [coords]
    elif kind != "MultiPolygon":
        raise ValueError(f"Unsupported geometry type: {kind}")
    return [[_ring_to_xy(ring) for ring in polygon] for polygon in coords]


def points_inside(px, py, polygons, chunk=2000):
    """Even-odd ray casting; a point is inside if it is inside any polygon."""
    result = np.zeros(len(px), bool)
    for start in range(0, len(px), chunk):
        x, y = px[start:start + chunk], py[start:start + chunk]
        inside_any = np.zeros(len(x), bool)
        for rings in polygons:
            crossings = np.zeros(len(x), int)
            for ring in rings:
                x1, y1, x2, y2 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
                straddle = (y1[None, :] > y[:, None]) != (y2[None, :] > y[:, None])
                with np.errstate(divide="ignore", invalid="ignore"):
                    x_cross = (x2 - x1)[None, :] * (y[:, None] - y1[None, :]) / (y2 - y1)[None, :] + x1[None, :]
                crossings += (straddle & (x[:, None] < x_cross)).sum(axis=1)
            inside_any |= crossings % 2 == 1
        result[start:start + chunk] = inside_any
    return result


def distance_to_boundary(px, py, polygons, chunk=200):
    """Minimum planar distance (km) from each point to any polygon edge."""
    best = np.full(len(px), np.inf)
    for rings in polygons:
        for ring in rings:
            x1, y1, x2, y2 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
            dx, dy = x2 - x1, y2 - y1
            length2 = dx * dx + dy * dy
            for start in range(0, len(px), chunk):
                x, y = px[start:start + chunk, None], py[start:start + chunk, None]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(length2 > 0, ((x - x1) * dx + (y - y1) * dy) / length2, 0.0)
                t = np.clip(t, 0.0, 1.0)
                d = np.hypot(x - (x1 + t * dx), y - (y1 + t * dy)).min(axis=1)
                best[start:start + chunk] = np.minimum(best[start:start + chunk], d)
    return best


def distance_to_district(px, py, polygons):
    """Distance (km) to the district; 0 for points inside it."""
    dist = distance_to_boundary(px, py, polygons)
    dist[points_inside(px, py, polygons)] = 0.0
    return dist


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------

def load_project_districts(path):
    frame = pd.read_csv(path, usecols=["district_code", "district_name"]).drop_duplicates()
    if frame.district_code.duplicated().any() or frame.district_name.duplicated().any():
        raise ValueError("District file must map each name to one code.")
    # Both columns come from the same frame, so their lengths are equal.
    return {str(n).upper().strip(): int(c) for c, n in zip(frame.district_code, frame.district_name)}


def load_district_polygons(boundaries_path, project_codes):
    """{district_code: (district_name, polygons)} for the 13 project districts."""
    gj = json.loads(Path(boundaries_path).read_text(encoding="utf-8"))
    by_name = {}
    for feature in gj["features"]:
        name = feature["properties"].get("shapeName")
        if name in GB_TO_PROJECT:
            by_name.setdefault(name, []).append(feature)
    missing = sorted(set(GB_TO_PROJECT) - set(by_name))
    duplicated = sorted(n for n, f in by_name.items() if len(f) > 1)
    if missing or duplicated:
        raise ValueError(f"Boundary file problem. Missing: {missing}; duplicated: {duplicated}")
    out = {}
    for gb_name, features in by_name.items():
        project_name = GB_TO_PROJECT[gb_name]
        if project_name not in project_codes:
            raise ValueError(f"Project district file has no {project_name}")
        polygons = polygons_from_geometry(features[0]["geometry"])
        xs = np.concatenate([ring[:, 0] for p in polygons for ring in p]) / (KM_PER_DEG * COS_LAT0)
        ys = np.concatenate([ring[:, 1] for p in polygons for ring in p]) / KM_PER_DEG
        if not (AP_BBOX[0] <= xs.min() and xs.max() <= AP_BBOX[2] and AP_BBOX[1] <= ys.min() and ys.max() <= AP_BBOX[3]):
            raise ValueError(f"{gb_name} lies outside the Andhra Pradesh sanity box; wrong boundary file?")
        out[project_codes[project_name]] = (project_name, polygons)
    if len(out) != 13:
        raise ValueError(f"Expected 13 districts, found {len(out)}")
    return out


UNNAMED_VALUES = {"", "UNNAMED", "NOT_NAMED"}


def pick_name(names):
    """One name per storm id.

    IBTrACS can change the NAME of a storm while it is being tracked (for example
    from UNNAMED to its name once named). Use the most frequent real name; if the
    storm never had one, return UNNAMED.
    """
    real = [n for n in names if n.strip().upper() not in UNNAMED_VALUES]
    if not real:
        return "UNNAMED"
    return pd.Series(real).value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0]


def load_tracks(path, start_year):
    """IMD track rows. Returns (imd_points, storm_info) as DataFrames."""
    use = ["SID", "SEASON", "NAME", "ISO_TIME", "SUBBASIN", "NEWDELHI_LAT", "NEWDELHI_LON",
           "NEWDELHI_GRADE", "NEWDELHI_WIND", "NEWDELHI_PRES"]
    raw = pd.read_csv(path, skiprows=[1], usecols=use, dtype=str, keep_default_na=False)
    units = pd.read_csv(path, nrows=1, usecols=["NEWDELHI_WIND", "NEWDELHI_PRES", "NEWDELHI_LAT"], dtype=str).iloc[0]
    if units["NEWDELHI_WIND"].strip() != "kts" or units["NEWDELHI_PRES"].strip() != "mb":
        raise ValueError(f"Unexpected source units: {units.to_dict()}")
    raw = raw.apply(lambda col: col.str.strip())
    raw["SEASON"] = raw["SEASON"].astype(int)
    raw = raw[raw["SEASON"] >= start_year].copy()
    if raw.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError("Duplicate (SID, ISO_TIME) rows in the track file.")

    for col in ["NEWDELHI_LAT", "NEWDELHI_LON", "NEWDELHI_WIND", "NEWDELHI_PRES"]:
        numeric = pd.to_numeric(raw[col].replace("", np.nan), errors="coerce")
        if (numeric.isna() & raw[col].ne("")).any():
            raise ValueError(f"Non-numeric, non-blank values in {col}")
        raw[col] = numeric
    raw["time"] = pd.to_datetime(raw["ISO_TIME"], utc=True)
    raw["grade"] = raw["NEWDELHI_GRADE"].map(normalise_grade)

    if (raw.groupby("SID")["SEASON"].nunique() > 1).any():
        raise ValueError("A storm id has more than one season in the track file.")
    storm_info = (
        raw.groupby("SID")
        .agg(season=("SEASON", "first"), name=("NAME", pick_name),
             subbasins=("SUBBASIN", lambda s: ";".join(sorted(set(s) - {""}))))
        .reset_index()
    )
    imd = raw[raw["NEWDELHI_LAT"].notna() & raw["NEWDELHI_LON"].notna()].copy()
    imd = imd.sort_values(["SID", "time"]).reset_index(drop=True)
    return imd, storm_info


# ---------------------------------------------------------
# Track sampling
# ---------------------------------------------------------

def sample_track(points):
    """Samples along one storm's CS-or-above track (originals plus interpolated).

    Returns a DataFrame with lat, lon, time, wind_kt, grade, original.
    An interpolated sample inherits wind and grade from its nearer end point,
    so only rows with original=True are observed IMD values.

    `frame.attrs["skipped_long_gaps"]` counts consecutive CS-or-above points that
    were NOT joined because they are more than MAX_GAP_HOURS apart. Rows without
    IMD coordinates are removed before this function, so they widen a gap too.
    A track that crosses a district inside a skipped gap can be missed.
    """
    rows = []
    skipped = 0
    pts = points.reset_index(drop=True)
    is_cs = pts["grade"].isin(CS_OR_ABOVE).to_numpy()
    for i in range(len(pts)):
        if not is_cs[i]:
            continue
        p = pts.loc[i]
        rows.append((p.NEWDELHI_LAT, p.NEWDELHI_LON, p.time, p.NEWDELHI_WIND, p.grade, True))
        if i + 1 < len(pts) and is_cs[i + 1]:
            q = pts.loc[i + 1]
            gap_h = (q.time - p.time).total_seconds() / 3600.0
            if gap_h > MAX_GAP_HOURS:
                skipped += 1
            if gap_h <= 0 or gap_h > MAX_GAP_HOURS:
                continue
            x1, y1 = to_xy(p.NEWDELHI_LON, p.NEWDELHI_LAT)
            x2, y2 = to_xy(q.NEWDELHI_LON, q.NEWDELHI_LAT)
            steps = max(1, math.ceil(math.hypot(x2 - x1, y2 - y1) / STEP_KM))
            for k in range(1, steps):
                f = k / steps
                near = p if f < 0.5 else q
                rows.append((p.NEWDELHI_LAT + f * (q.NEWDELHI_LAT - p.NEWDELHI_LAT),
                             p.NEWDELHI_LON + f * (q.NEWDELHI_LON - p.NEWDELHI_LON),
                             p.time + (q.time - p.time) * f, near.NEWDELHI_WIND, near.grade, False))
    frame = pd.DataFrame(rows, columns=["lat", "lon", "time", "wind_kt", "grade", "original"])
    frame.attrs["skipped_long_gaps"] = skipped
    return frame


# ---------------------------------------------------------
# Build
# ---------------------------------------------------------

def build(tracks_path, boundaries_path, districts_path, distance_km=100.0, start_year=1990):
    if not (distance_km > 0 and np.isfinite(distance_km)):
        raise ValueError("distance_km must be a positive number.")
    # Distances are computed up to this cap. It follows the requested threshold, so a
    # threshold above DISTANCE_CAP_KM is honoured and never silently truncated.
    cap_km = max(DISTANCE_CAP_KM, float(distance_km))
    project_codes = load_project_districts(districts_path)
    districts = load_district_polygons(boundaries_path, project_codes)
    imd, storm_info = load_tracks(tracks_path, start_year)

    bb_storms = set(storm_info.loc[storm_info["subbasins"].str.contains("BB"), "SID"])
    imd_storms = set(imd["SID"])
    catalogue_ids = sorted(bb_storms & imd_storms)
    info = storm_info.set_index("SID")

    events, pairs = [], []
    sensitivity = {km: set() for km in SENSITIVITY_KM}
    skipped_gaps_total = 0

    for sid in catalogue_ids:
        pts = imd[imd["SID"] == sid]
        samples = sample_track(pts)
        skipped_gaps = int(samples.attrs.get("skipped_long_gaps", 0))
        skipped_gaps_total += skipped_gaps
        n_cs = int(pts["grade"].isin(CS_OR_ABOVE).sum())
        affected = []

        if len(samples):
            px, py = to_xy(samples["lon"].to_numpy(), samples["lat"].to_numpy())
            for code, (dname, polygons) in districts.items():
                xmin = min(r[:, 0].min() for p in polygons for r in p) - cap_km
                xmax = max(r[:, 0].max() for p in polygons for r in p) + cap_km
                ymin = min(r[:, 1].min() for p in polygons for r in p) - cap_km
                ymax = max(r[:, 1].max() for p in polygons for r in p) + cap_km
                near_box = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
                if not near_box.any():
                    continue
                idx = np.flatnonzero(near_box)
                dist = distance_to_district(px[idx], py[idx], polygons)
                closest = int(np.argmin(dist))
                min_d = float(dist[closest])
                if not np.isfinite(min_d) or min_d > cap_km:
                    continue
                for km in SENSITIVITY_KM:
                    if min_d <= km:
                        sensitivity[km].add((sid, code))
                if min_d > distance_km:
                    continue
                within = idx[dist <= distance_km]
                s = samples.iloc[idx[closest]]
                # Peak wind uses observed IMD points only; interpolated samples borrow
                # their wind from an end point and are not observations.
                within_samples = samples.iloc[within]
                winds = within_samples.loc[within_samples["original"], "wind_kt"].dropna()
                pairs.append({
                    "event_id": sid, "season": int(info.loc[sid, "season"]), "name": info.loc[sid, "name"],
                    "district_code": code, "district_name": dname,
                    "rule_tier": "inside" if min_d == 0.0 else "near",
                    "min_distance_km": round(min_d, 1),
                    "distance_threshold_km": distance_km,
                    "n_imd_points_within": int(samples.iloc[within]["original"].sum()),
                    "first_time_within_utc": samples.iloc[within]["time"].min().strftime("%Y-%m-%d %H:%M"),
                    "time_at_closest_utc": s["time"].strftime("%Y-%m-%d %H:%M"),
                    "grade_at_closest": s["grade"],
                    "closest_point_is_observed": bool(s["original"]),
                    "max_imd_wind_kt_within": float(winds.max()) if len(winds) else np.nan,
                })
                affected.append(code)

        if n_cs == 0:
            status = "unmapped_never_reached_cyclonic_storm_grade"
        elif affected:
            status = "mapped"
        else:
            status = "unmapped_no_project_district_within_threshold"

        wind, pres = pts["NEWDELHI_WIND"], pts["NEWDELHI_PRES"]
        events.append({
            "event_id": sid, "season": int(info.loc[sid, "season"]), "name": info.loc[sid, "name"],
            "first_time_utc": pts["time"].min().strftime("%Y-%m-%d %H:%M"),
            "last_time_utc": pts["time"].max().strftime("%Y-%m-%d %H:%M"),
            "n_imd_points": len(pts), "n_cs_or_above_points": n_cs,
            "n_cs_segments_not_interpolated": skipped_gaps,
            "grades_observed": ";".join(sorted(set(pts["grade"]) - {""})),
            "max_imd_wind_kt": float(wind.max()) if wind.notna().any() else np.nan,
            "min_imd_pressure_hpa": float(pres.min()) if pres.notna().any() else np.nan,
            "subbasins_visited": info.loc[sid, "subbasins"],
            "n_affected_districts": len(affected),
            "affected_district_codes": ";".join(str(c) for c in sorted(affected)),
            "mapping_status": status,
            "source": "IBTrACS v04r01 NI, IMD RSMC New Delhi (NEWDELHI_*) fields",
        })

    events_df = pd.DataFrame(events)
    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        pairs_df = pairs_df.sort_values(["event_id", "district_code"]).reset_index(drop=True)
    unmapped_df = events_df.loc[events_df["mapping_status"] != "mapped",
                                ["event_id", "season", "name", "n_cs_or_above_points", "max_imd_wind_kt", "mapping_status"]]
    # Bay of Bengal storms with no IMD positions cannot enter the catalogue or be mapped;
    # list them so that nothing is silently dropped.
    no_positions = storm_info[storm_info["SID"].isin(bb_storms - imd_storms)].rename(
        columns={"SID": "event_id"})[["event_id", "season", "name"]]
    no_positions = no_positions.assign(
        n_cs_or_above_points=np.nan, max_imd_wind_kt=np.nan, mapping_status="unmapped_no_imd_track_positions")
    unmapped_df = pd.concat([unmapped_df, no_positions], ignore_index=True).sort_values("event_id")
    context = {
        "storms_in_file_from_start_year": int(len(storm_info)),
        "storms_with_imd_positions": len(imd_storms),
        "bay_of_bengal_storms_without_imd_positions": len(bb_storms - imd_storms),
        "catalogue_events": len(events_df),
        "sensitivity_pairs": {km: len(v) for km, v in sensitivity.items()},
        "sensitivity_events": {km: len({e for e, _ in v}) for km, v in sensitivity.items()},
        "events_with_skipped_long_gaps": int((events_df["n_cs_segments_not_interpolated"] > 0).sum()),
        "cs_segments_not_interpolated": skipped_gaps_total,
        "distance_km": distance_km, "start_year": start_year,
    }
    return events_df, pairs_df, unmapped_df.reset_index(drop=True), context


PAIR_COLUMNS = ["event_id", "district_code", "rule_tier", "min_distance_km", "distance_threshold_km"]


def validate(events_df, pairs_df, project_codes=None, districts_n=13):
    """Return a list of (check, passed, detail).

    Every check always runs. An empty map is not skipped: the catalogue must then
    agree that no event has an affected district. `project_codes` is the set of valid
    district codes; without it only the number of distinct codes is checked.
    """
    checks = []
    n_pairs = len(pairs_df)
    if n_pairs == 0:
        pairs_df = pd.DataFrame({c: pd.Series(dtype=object) for c in PAIR_COLUMNS})
    checks.append(("event_key_unique", not events_df.duplicated("event_id").any(), f"{len(events_df)} events"))
    checks.append(("pair_key_unique", not pairs_df.duplicated(["event_id", "district_code"]).any(),
                   f"{n_pairs} event-district pairs"))
    checks.append(("pairs_reference_catalogue", set(pairs_df["event_id"]) <= set(events_df["event_id"]),
                   "every mapped event is in the catalogue"))
    codes = set(pairs_df["district_code"])
    if project_codes is None:
        codes_ok, codes_detail = len(codes) <= districts_n, f"{len(codes)} district codes appear"
    else:
        codes_ok = codes <= set(project_codes)
        codes_detail = f"{len(codes)} district codes appear, all in the {len(set(project_codes))} project codes" if codes_ok \
            else f"codes not in the project list: {sorted(codes - set(project_codes))}"
    checks.append(("district_codes_valid", codes_ok, codes_detail))
    dist = pd.to_numeric(pairs_df["min_distance_km"], errors="coerce")
    thr = pd.to_numeric(pairs_df["distance_threshold_km"], errors="coerce")
    checks.append(("distances_valid", bool((np.isfinite(dist) & (dist >= 0) & (dist <= thr)).all()),
                   "finite, non-negative and within each pair's threshold"))
    checks.append(("inside_means_zero_distance",
                   bool(((pairs_df["rule_tier"] == "inside") == (dist == 0)).all()), "tier agrees with distance"))
    counts = pairs_df.groupby("event_id").size()
    merged = events_df.set_index("event_id")["n_affected_districts"]
    checks.append(("affected_counts_agree",
                   bool((counts.reindex(merged.index).fillna(0).astype(int) == merged).all()),
                   "catalogue counts match the map"))
    by_event = pairs_df.groupby("event_id")["district_code"].apply(lambda s: {int(c) for c in s})
    listed = events_df.set_index("event_id")["affected_district_codes"].fillna("").astype(str)
    codes_agree = all(
        {int(x) for x in listed[e].split(";") if x} == by_event.get(e, set()) for e in listed.index
    )
    checks.append(("affected_codes_agree", codes_agree, "catalogue district codes match the map"))
    checks.append(("mapped_status_consistent",
                   bool(((events_df["mapping_status"] == "mapped") == (events_df["n_affected_districts"] > 0)).all()),
                   "status mapped iff at least one district"))
    checks.append(("only_bay_of_bengal_events", bool(events_df["subbasins_visited"].str.contains("BB").all()),
                   "every event has a Bay of Bengal point"))
    return checks


SPOT_CHECKS = [
    # (name, season, expected district names). Expectations come from public reports of landfall
    # locations (pre-2022 district boundaries); they are NOT derived from the data and should be
    # confirmed by a reviewer against IMD reports.
    ("HUDHUD", 2014, ["VISAKHAPATNAM"], "landfall near Visakhapatnam"),
    ("PHETHAI", 2018, ["EAST GODAVARI"], "landfall near Kakinada"),
    ("MICHAUNG", 2023, ["GUNTUR"], "landfall near Bapatla (then in Guntur)"),
    ("LAILA", 2010, ["GUNTUR"], "landfall near Bapatla"),
    ("TITLI", 2018, ["SRIKAKULAM"], "landfall near the Odisha-Srikakulam coast"),
    ("GULAB", 2021, ["SRIKAKULAM"], "landfall near Kalingapatnam (Srikakulam)"),
]


def spot_checks(events_df, pairs_df, start_year=1990):
    """Compare known events with the map.

    Status: `pass` (all expected districts found), `fail` (event found but an expected
    district is missing), `not_in_catalogue` (event absent although its season is in range)
    or `skipped_before_start_year`. Compound IBTrACS names such as `GULAB:SHAHEEN-GU` are
    matched on the part before the colon.
    """
    base = events_df["name"].str.upper().str.split(":").str[0].str.strip() if len(events_df) else pd.Series(dtype=str)
    rows = []
    for name, season, expected, note in SPOT_CHECKS:
        row = {"event": name, "season": season, "expected_districts": ";".join(expected), "note": note,
               "status": "", "found_in_catalogue": False, "found_districts": "", "all_expected_found": False,
               "closest_expected_distance_km": np.nan}
        if season < start_year:
            row["status"] = "skipped_before_start_year"
            rows.append(row)
            continue
        ev = events_df[(base == name) & (events_df["season"] == season)]
        if ev.empty:
            row["status"] = "not_in_catalogue"
            rows.append(row)
            continue
        sid = ev.iloc[0]["event_id"]
        sub = pairs_df[pairs_df["event_id"] == sid] if len(pairs_df) else pairs_df
        found = sorted(sub["district_name"]) if len(sub) else []
        dists = sub[sub["district_name"].isin(expected)]["min_distance_km"] if len(sub) else []
        row.update({"found_in_catalogue": True, "found_districts": ";".join(found),
                    "all_expected_found": all(d in found for d in expected),
                    "closest_expected_distance_km": float(min(dists)) if len(dists) else np.nan})
        row["status"] = "pass" if row["all_expected_found"] else "fail"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Issue 46: cyclone event catalogue and district map")
    parser.add_argument("--tracks", required=True, help="IBTrACS v04r01 NI CSV (ibtracs.NI.list.v04r01.csv)")
    parser.add_argument("--boundaries", required=True, help="geoBoundaries IND ADM2 GeoJSON")
    parser.add_argument("--districts", required=True, help="CSV with district_code and district_name")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--distance-km", type=float, default=100.0,
                        help="affected-district threshold in km (any positive value; distances are computed up to "
                             "max(200, this value))")
    parser.add_argument("--start-year", type=int, default=1990)
    a = parser.parse_args()

    events, pairs, unmapped, ctx = build(a.tracks, a.boundaries, a.districts, a.distance_km, a.start_year)
    checks = validate(events, pairs, project_codes=set(load_project_districts(a.districts).values()))
    failed = [c for c in checks if not c[1]]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "cyclone_events_v1.csv", index=False)
    pairs.to_csv(out / "cyclone_district_map_v1.csv", index=False)
    unmapped.to_csv(out / "cyclone_unmapped_events_v1.csv", index=False)
    pd.DataFrame(checks, columns=["check", "passed", "detail"]).assign(
        status=lambda d: np.where(d["passed"], "PASS", "FAIL"))[["check", "status", "detail"]].to_csv(
        out / "validation_checks.csv", index=False)
    checks_df = spot_checks(events, pairs, a.start_year)
    checks_df.to_csv(out / "cyclone_spot_checks_v1.csv", index=False)
    spot_failed = checks_df[checks_df["status"].isin(["fail", "not_in_catalogue"])]

    print("CYCLONE CATALOGUE COMPLETED")
    print("---------------------------")
    for k, v in ctx.items():
        print(f"{k}: {v}")
    print("mapping_status:", events["mapping_status"].value_counts().to_dict())
    print(f"event-district pairs: {len(pairs)}")
    print("\nVALIDATION CHECKS")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print("\nSPOT CHECKS (expectations from public reports, not from the data)")
    print(checks_df.to_string(index=False))
    print(f"\nOutputs saved to: {out.resolve()}")
    problems = []
    if failed:
        problems.append("Validation failed: " + ", ".join(c[0] for c in failed))
    if len(spot_failed):
        problems.append("Spot check did not match expectations for: "
                        + ", ".join(f"{r.event} {r.season} ({r.status})" for r in spot_failed.itertuples())
                        + ". Expectations come from public reports; review before relying on this run.")
    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()
