# Cyclone event catalogue and track-to-district map (Issue 46)

A catalogue of historical Bay of Bengal cyclone events built from **IMD track data**, and a map from each event to the Andhra Pradesh districts it affected under a documented rule.

**Scope: Andhra Pradesh only.** The project works with 13 Andhra Pradesh districts, so a storm is catalogued only if its IMD track (any grade) came within **300 km** of one of them (`--catalogue-radius-km`). Storms that stayed far from AP (for example ones that hit Odisha, Tamil Nadu, Bangladesh or Myanmar without coming near) are left out and only counted in the run summary. All geography is the 13 AP districts.

Numbers in this README are from the recorded run (inputs hashed below, start year 1990, 100 km). They are also written to `cyclone_run_summary_v1.csv`; if the inputs or options change, trust that file, not this text.

## Outputs

| File | Grain | Content |
|---|---|---|
| `cyclone_events_v1.csv` | one row per event | Event id, season, name, IMD time span, IMD point counts, points that count as Cyclonic Storm or above by wind only, CS-or-above segments not interpolated because of a long gap, grades seen, peak IMD wind (kt), minimum IMD pressure (hPa), sub-basins, `min_distance_any_grade_km` (closest approach of any IMD point to any AP district), number and codes of affected districts, mapping status. |
| `cyclone_district_map_v1.csv` | one row per event and affected district | Rule tier (`inside` or `near`), minimum distance (km, rounded **up** to 0.1 km), first time within the threshold, time and grade at the closest point, `closest_point_is_observed`, peak **observed** IMD wind within the threshold (empty if no observed point is that close). |
| `cyclone_unmapped_events_v1.csv` | one row per event that is not mapped | The catalogued events that map to no district, plus the storms near AP that have no IMD positions, each with the reason and the closest approach to AP (`closest_approach_km`, with `closest_approach_basis` saying whether it is from IMD points or from the IBTrACS merged track, which is not IMD data). |
| `cyclone_run_summary_v1.csv` | one row per metric | Counts, threshold sensitivity and data-quality numbers of the run. |
| `validation_checks.csv` | one row per check | 11 integrity checks. |
| `cyclone_spot_checks_v1.csv` | one row per spot-checked storm | Known events compared with the map. The expectations are **unverified** (see below). |

`event_id` is the IBTrACS storm id. `name` is the most frequent real name of the storm (IBTrACS can change a name while a storm is tracked; `UNNAMED` is used only if it never had one). Storms that crossed basins carry compound names such as `GULAB:SHAHEEN-GU`.

All times are UTC. Wind is in knots and pressure in mb (= hPa), taken from the units row of the source file (the run stops if they differ).

## Inputs (public, not committed)

| Input | Source | Size and SHA-256 used for the recorded run |
|---|---|---|
| IMD tracks | IBTrACS v04r01, North Indian Ocean CSV, NOAA NCEI: `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv` | 27,919,356 bytes, `34b6a7ead2fb30c2986eff8274aed498958eceb9664dbaf99ed117b0a70bd97b` |
| District boundaries | geoBoundaries gbOpen, India ADM2 (simplified): `geoBoundaries-IND-ADM2_simplified.geojson` | 7,979,916 bytes, `d68db39cd3e2d0892af268e2b0454166368ce3b5b8a78fcda63069ec92a641db` |
| Project districts | `weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv` (`district_code`, `district_name`) | in the repository |

IBTrACS republishes the track data of each agency. Only the **IMD RSMC New Delhi** fields (`NEWDELHI_LAT`, `NEWDELHI_LON`, `NEWDELHI_GRADE`, `NEWDELHI_WIND`, `NEWDELHI_PRES`) are used for grades, winds, pressures, mapping and the catalogue. "New Delhi" is the name of the reporting agency (IMD's Regional Specialised Meteorological Centre, the official source for North Indian Ocean cyclones), not a location: no Delhi position is used anywhere.

The one exception is the IBTrACS merged `LAT`/`LON` of storms that have **no IMD positions at all**. They are used only to tell whether such a storm came near AP, so that it can be listed. They never grade or map a storm.

The boundary file still has the **13 pre-2022 Andhra Pradesh districts** that the project uses (the 2022 reorganisation created 26). GADM 4.1 also has 13, but its licence terms appear to restrict commercial use (not verified here), so geoBoundaries was used. **The licence and attribution terms of both inputs (IBTrACS and geoBoundaries) have not been confirmed.** Confirm them, and add any required attribution, before these outputs are distributed.

## Run

From the repository root (the download takes about a minute):

```bash
mkdir -p /tmp/cyclone_inputs   # any folder outside the repository
curl -L -o /tmp/cyclone_inputs/ibtracs.NI.list.v04r01.csv \
  "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
# download geoBoundaries-IND-ADM2_simplified.geojson (gbOpen, IND, ADM2) into the same folder

python3 cyclone_catalogue/build_cyclone_catalogue.py \
  --tracks /tmp/cyclone_inputs/ibtracs.NI.list.v04r01.csv \
  --boundaries /tmp/cyclone_inputs/geoBoundaries-IND-ADM2_simplified.geojson \
  --districts weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv \
  --outdir cyclone_catalogue

python3 -m pytest cyclone_catalogue/test_cyclone_catalogue.py -v
```

Options:
- `--distance-km` (default 100): any positive value is honoured; distances are computed up to the larger of 200 km and this value.
- `--start-year` (default 1990).
- `--catalogue-radius-km` (default 300): the Andhra Pradesh focus. A storm is catalogued only if an IMD track point of any grade came within this distance of an AP district. Must be at least `--distance-km`.
- `--strict-spot-checks`: exit with an error when a spot check does not match its expectation. By default a mismatch is only printed as a warning, because the expectations are unverified and a newer IBTrACS release can legitimately change a track or a name.

The script exits with an error (after writing the outputs) if a validation check fails. Requires `pandas`, `numpy` and `pytest`; no geometry library is needed. Tested with Python 3.12 only. IBTrACS is updated regularly, so a newer download may add or change events.

## Which events are in the catalogue

An event is included if (a) it has at least one point in the IBTrACS Bay of Bengal sub-basin (`BB`), (b) it has IMD positions, (c) its season is the start year or later, and (d) an IMD point of **any grade** (depressions included) came within the catalogue radius (300 km) of an AP district. Events that visit the Arabian Sea only are not included. Storms that stayed farther away are excluded and counted in the run summary.

Bay of Bengal storms with **no IMD positions** cannot be graded or catalogued. They are handled by the IBTrACS merged track, which is not IMD data:
- within the radius of AP: listed in `cyclone_unmapped_events_v1.csv` with status `unmapped_no_imd_track_positions` and `closest_approach_basis = ibtracs_merged_track_not_imd`;
- farther than the radius: not listed, counted in the run summary;
- no usable position at all: listed with basis `none_available` (relevance unknown).

The radius is a choice. It is about three times the 100 km mapping threshold, so storms that passed offshore are kept as context. The run summary reports how many storms each radius would catalogue.

## Affected-district rule

> A district is **affected** by an event if the IMD best track, at a time when it counted as **Cyclonic Storm (CS) or higher**, came **within 100 km** of the district polygon, or inside it.

**When a point counts as CS or higher.**
- The **IMD grade decides** whenever it is present: `CS`, `SCS`, `SCS(H)`, `VSCS`, `ESCS` and `SUCS` (`SuCS` is read as `SUCS`) count; `D`, `DD` and `L` do not, whatever the wind. Grade `CS` is IMD's class for a sustained wind of at least 34 kt.
- Where the **grade is blank**, the IMD wind decides: 34 kt or more counts, and a missing wind does not. Those points are labelled `blank_grade_wind_ge_34kt` and counted per event in `n_cs_points_by_wind_only`.
- Why: in the source file the grade is blank for many early-1990s rows even when the wind is strong. A grade-only rule wrongly labelled such storms "never reached cyclonic storm" and missed four events (12 district pairs), all unnamed storms of 1990 to 1995 (one has a peak IMD wind of 127 kt). Nothing that was mapped under the grade-only rule changed, and no 2015 to 2019 result changed.
- Points with a present grade below CS but wind of 34 kt or more are **not** used and are counted in the run summary (see Results).

**Other rules.**
- The 100 km threshold is **an assumption**. There is no authoritative IMD definition of an affected district. Sensitivity is below.
- Between two consecutive CS-or-above IMD points at most 12 hours apart, the track is linearly interpolated every 10 km or less, so a track that passes between two reports is not missed. An interpolated point takes the wind and grade of its nearer end point, so it is **not an observation**: the peak wind in the map uses observed IMD points only, and `closest_point_is_observed` says whether the closest sample is an IMD point or an interpolated one.
- Consecutive CS-or-above points more than 12 hours apart are not joined, and the number of such segments is reported per event (`n_cs_segments_not_interpolated`). A track that crosses a district inside such a gap, or around an IMD row with blank coordinates, could be missed.
- `rule_tier` is `inside` when a sample point lies in the district polygon or on its boundary (distance 0) and `near` when the closest approach is within the threshold.
- Distances are planar, in a projection centred on each district's own mean latitude, so the east-west scale error stays within about 2 percent (about 2 km at 100 km) inside the search area. Polygons are simplified. Distances are rounded **up** to 0.1 km, so a positive distance never shows as 0.0.

This is a **track-proximity rule**. It is not a wind field, storm-surge or rainfall footprint, and it says nothing about damage, crop loss or repayment.

## Results (recorded run)

- IBTrACS storms from 1990: 371, of which 325 have IMD positions. 247 of those visit the Bay of Bengal; 131 came within 300 km of an AP district and form the catalogue, and 116 stayed farther away and are excluded. 39 Bay of Bengal storms have no IMD positions (see below).
- Catalogue: **131 events**. Of these, 70 never reached CS (by grade or, where blank, wind), 30 reached CS but never came within 100 km of a project district, and **31 are mapped** to at least one district (122 event-district pairs: 21 `inside`, 101 `near`). Limiting the catalogue to AP removed no mapped event: the mapped set is the same as with the whole Bay of Bengal.
- `cyclone_unmapped_events_v1.csv` has 110 rows: those 70 + 30, plus 10 storms near AP that have no IMD positions.
- 2015 to 2019 (the project's yield and weather years): 20 events, **6 mapped**: ROANU 2016, VARDAH 2016, DAYE 2018, TITLI 2018, PHETHAI 2018, FANI 2019.
- **Storms near AP that cannot be mapped (no IMD positions in IBTrACS).** Of the 39 Bay of Bengal storms without IMD positions, 10 came within 300 km of AP by their merged track, and 5 of those ran over AP districts (within 0.1 km): one unnamed 1992 storm, one unnamed 2026 storm, and the 2026 storms ARNAB, MURASU and AKVAN. IBTrACS has no IMD fields for them (the recent ones are provisional), so they are listed but **not mapped**. Until IBTrACS adds IMD positions, the map misses them. The other 29 stayed farther than 300 km and are only counted.
- 12 of the 13 districts are affected at least once. Kurnool is affected by 2 events; East Godavari by 14.
- Closest point is an observed IMD point for 68 pairs and an interpolated one for 54. Peak observed wind is empty for 3 pairs (no observed IMD point within 100 km).
- Grade versus wind: 154 points in 8 catalogued events (all in 1990 to 1995) count as CS or above by wind because their grade is blank. 1 point in 1 event has wind of 34 kt or more but a grade below CS, and is not used.
- No CS-or-above segment was longer than 12 hours, so no gap-skipping occurred.
- All 11 validation checks pass (the new one checks every event is within the catalogue radius); 97 tests pass.

### Sensitivity to the threshold

| Threshold | Event-district pairs | Events mapped |
|---|---|---|
| 50 km | 74 | 28 |
| 100 km (used) | 122 | 31 |
| 150 km | 164 | 35 |

The pair count grows faster than the event count, so any analysis that uses the map should state the threshold.

### Sensitivity to the catalogue radius

| Radius | Catalogued events (IMD storms that came this close) |
|---|---|
| 150 km | 86 |
| 200 km | 100 |
| 300 km (used) | 131 |
| 500 km | 196 |

The radius changes only how many context storms are listed; it cannot change the mapped set, because the radius must be at least the 100 km mapping threshold.

### Spot checks against known events

`cyclone_spot_checks_v1.csv` compares six storms with landfall locations as generally reported (pre-2022 districts). **These expectations are unverified: they were recalled from general public reporting, no source is attached to any of them, and they are not derived from the data.** The CSV says so in the `expectation_verified` (always `False`) and `expectation_source` columns. Each should be confirmed against IMD's published cyclone reports before a `pass` is read as evidence. A storm is matched on the part of its name before any `:`. Status is `pass`, `fail`, `not_in_catalogue`, `ambiguous_multiple_events` (more than one catalogue event with that name and season) or `skipped_before_start_year`.

| Storm | Expected district | Result |
|---|---|---|
| HUDHUD 2014 | Visakhapatnam | found, track inside (0 km) |
| PHETHAI 2018 | East Godavari | found, track inside (0 km) |
| MICHAUNG 2023 | Guntur | found, track inside (0 km) |
| LAILA 2010 | Guntur | found, track inside (0 km) |
| TITLI 2018 | Srikakulam | found, track inside (0 km) |
| GULAB 2021 (`GULAB:SHAHEEN-GU`) | Srikakulam | found, track inside (0 km); also Vizianagaram (inside) and Visakhapatnam |

All six expected districts were found. This is a small consistency check, not a validation of the rule. It does not test false positives (districts listed that were not materially affected) or events that were missed.

## Limitations

- The affected-district rule and the 100 km threshold are assumptions; the map describes track proximity, not impact. Association is not causation: an affected district is not evidence of crop loss, default or NPA.
- Only IMD fields are used for grading and mapping. Bay of Bengal storms whose IMD fields are blank in IBTrACS (39 since 1990) cannot be mapped, and 10 of them came near AP (5 over AP districts, including three 2026 storms). The map therefore under-counts AP events until IBTrACS has IMD positions for them.
- The 300 km catalogue radius is a choice; it affects only which context storms are listed, not which events are mapped.
- Using wind where the grade is blank is a choice made to avoid losing strong early-1990s storms; it relies on the IMD wind values in those rows being reliable.
- IBTrACS NI is a best-track dataset assembled from agency reports. Its coverage and quality change over time, and a re-download may change results.
- District polygons are simplified and use pre-2022 boundaries.
- Wind radius, storm size, rainfall, surge and the timing of local conditions are not considered.
- The spot-check expectations are unverified (see above). The licences of the two inputs are unconfirmed.
- Mapped events per project year: 2015 none, 2016 two, 2017 none, 2018 three, 2019 one. Any statistical use of the map on the project's 2015 to 2019 data would rest on very few events.
