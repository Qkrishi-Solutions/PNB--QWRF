# Cyclone event catalogue and track-to-district map (Issue 46)

A catalogue of historical Bay of Bengal cyclone events built from **IMD track data**, and a map from each event to the Andhra Pradesh districts it affected under a documented rule.

## Outputs

| File | Grain | Content |
|---|---|---|
| `cyclone_events_v1.csv` | one row per event (247) | Event id, season, name, IMD time span, IMD point counts, number of CS-or-above segments that were not interpolated because of a long gap, grades seen, peak IMD wind (kt), minimum IMD pressure (hPa), sub-basins, number and codes of affected districts, mapping status. |
| `cyclone_district_map_v1.csv` | one row per event and affected district (110) | Rule tier (`inside` or `near`), minimum distance (km), first time within the threshold, time and IMD grade at the closest point, `closest_point_is_observed`, peak **observed** IMD wind within the threshold (empty if no observed point is that close). |
| `cyclone_unmapped_events_v1.csv` | one row per event that is not mapped (259) | The events that map to no district, with the reason. |
| `validation_checks.csv` | one row per check | 10 integrity checks (all PASS). |
| `cyclone_spot_checks_v1.csv` | one row per spot-checked storm | Known events compared with the map, with a status (see below). |

`event_id` is the IBTrACS storm id. `name` is the most frequent real name of the storm (IBTrACS can change a name while a storm is tracked; `UNNAMED` is used only if it never had one). Storms that crossed basins carry compound names such as `GULAB:SHAHEEN-GU`.

All times are UTC. Wind is in knots and pressure in mb (= hPa), taken from the units row of the source file (the run stops if they differ).

## Inputs (public, not committed)

| Input | Source | Size and SHA-256 used for the recorded run |
|---|---|---|
| IMD tracks | IBTrACS v04r01, North Indian Ocean CSV, NOAA NCEI: `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv` | 27,919,356 bytes, `34b6a7ead2fb30c2986eff8274aed498958eceb9664dbaf99ed117b0a70bd97b` |
| District boundaries | geoBoundaries gbOpen, India ADM2 (simplified): `geoBoundaries-IND-ADM2_simplified.geojson` | 7,979,916 bytes, `d68db39cd3e2d0892af268e2b0454166368ce3b5b8a78fcda63069ec92a641db` |
| Project districts | `weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv` (`district_code`, `district_name`) | in the repository |

IBTrACS republishes the track data of each agency. Only the **IMD RSMC New Delhi** fields (`NEWDELHI_LAT`, `NEWDELHI_LON`, `NEWDELHI_GRADE`, `NEWDELHI_WIND`, `NEWDELHI_PRES`) are used. The merged `LAT`/`LON`/`WMO_*` fields are ignored.

The boundary file still has the **13 pre-2022 Andhra Pradesh districts** that the project uses (the 2022 reorganisation created 26). GADM 4.1 also has 13, but I understand its licence restricts commercial use, so geoBoundaries was used. Confirm the licence terms of both inputs (IBTrACS and geoBoundaries) before external distribution.

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

Options: `--distance-km` (default 100; any positive value is honoured, and distances are computed up to the larger of 200 km and this value) and `--start-year` (default 1990). Requires `pandas`, `numpy` and `pytest`; no geometry library is needed. Run with Python 3.12; no older version was tested. IBTrACS is updated regularly, so a newer download may add or change events.

The script exits with an error (after writing the outputs) if a validation check fails or if a spot check that applies to the chosen start year does not match its expectation.

## Which events are in the catalogue

An event is included if (a) it has at least one point in the IBTrACS Bay of Bengal sub-basin (`BB`), (b) it has IMD positions, and (c) its season is the start year or later. Events that visit the Arabian Sea only are not included. Bay of Bengal storms with **no IMD positions** cannot be catalogued; they are listed in `cyclone_unmapped_events_v1.csv` with status `unmapped_no_imd_track_positions`, not dropped silently.

## Affected-district rule

> A district is **affected** by an event if the IMD best track, at a time when IMD graded it **Cyclonic Storm (CS) or higher**, came **within 100 km** of the district polygon, or inside it.

- CS or higher means the IMD grades `CS`, `SCS`, `SCS(H)`, `VSCS`, `ESCS` and `SUCS` (`SuCS` is read as `SUCS`). Depressions (`D`, `DD`) and `L` are below the threshold. Grade `CS` is IMD's own class for a sustained wind of at least 34 kt.
- The 100 km threshold is **an assumption**. There is no authoritative IMD definition of an affected district. Sensitivity is below.
- Between two consecutive IMD points that are both CS or higher and at most 12 hours apart, the track is linearly interpolated every 10 km or less, so a track that passes between two reports is not missed. An interpolated point takes the wind and grade of its nearer end point, so it is **not an observation**: the peak wind in the map uses observed IMD points only, and `closest_point_is_observed` says whether the closest sample is an IMD point (61 of the 110 pairs) or an interpolated one (49). Consecutive CS-or-above points more than 12 hours apart are not joined, and the number of such segments is reported per event (`n_cs_segments_not_interpolated`). A track that crosses a district inside such a gap, or around an IMD row with blank coordinates, could be missed.
- `rule_tier` is `inside` when a sample point lies in the district polygon (distance 0) and `near` when the closest approach is within the threshold.
- Distances use a planar approximation (equirectangular projection around 16 degrees N) and simplified district polygons, so they are approximate, to within a few km.

This is a **track-proximity rule**. It is not a wind field, storm-surge or rainfall footprint, and it says nothing about damage, crop loss or repayment.

## Results (actual run, start year 1990, 100 km)

- IBTrACS storms from 1990: 371, of which 325 have IMD positions. 39 Bay of Bengal storms have none (listed as unmapped).
- Catalogue: **247 events**. Of these, 159 never reached CS grade, 61 reached CS or higher but never came within 100 km of a project district, and **27 are mapped** to at least one district (110 event-district pairs: 18 `inside`, 92 `near`).
- 2015 to 2019 (the project's yield and weather years): 38 events, **6 mapped**: ROANU 2016, VARDAH 2016, DAYE 2018, TITLI 2018, PHETHAI 2018, FANI 2019.
- All 10 validation checks pass; 57 tests pass.
- No CS-or-above segment in this data was longer than 12 hours, so no gap-skipping occurred (`n_cs_segments_not_interpolated` is 0 for every event).
- Peak observed wind is empty for 3 of the 110 pairs (no observed IMD point within 100 km).
- 12 of the 13 districts are affected at least once. Kurnool is affected by 1 event; East Godavari by 13.

### Sensitivity to the threshold

| Threshold | Event-district pairs | Events mapped |
|---|---|---|
| 50 km | 66 | 25 |
| 100 km (used) | 110 | 27 |
| 150 km | 149 | 31 |

The mapped set grows slowly with the threshold but the pair count grows quickly, so any analysis that uses the map should state the threshold.

### Spot checks against known events

`cyclone_spot_checks_v1.csv` compares six storms with landfall locations as I recall them from public reports (pre-2022 districts). **These expectations are from memory, not from the data, and a reviewer should confirm them against IMD reports.** A storm is matched on the part of its name before any `:`. Status is `pass`, `fail`, `not_in_catalogue`, or `skipped_before_start_year`.

| Storm | Expected district | Result |
|---|---|---|
| HUDHUD 2014 | Visakhapatnam | found, track inside (0 km) |
| PHETHAI 2018 | East Godavari | found, track inside (0 km) |
| MICHAUNG 2023 | Guntur | found, track inside (0 km) |
| LAILA 2010 | Guntur | found, track inside (0 km) |
| TITLI 2018 | Srikakulam | found, track inside (0 km) |
| GULAB 2021 (`GULAB:SHAHEEN-GU`) | Srikakulam | found, track inside (0 km); also Vizianagaram (inside) and Visakhapatnam (19.7 km) |

All six expected districts were found. This is a small, consistency-level check, not a validation of the rule. It does not test false positives (districts listed that were not materially affected) or events that were missed.

## Limitations

- The affected-district rule and the 100 km threshold are assumptions; the map describes track proximity, not impact.
- Only IMD fields are used. Events whose IMD fields are blank in IBTrACS (39 Bay of Bengal storms from 1990) cannot be mapped.
- IBTrACS NI is a best-track dataset assembled from IMD reports. Its coverage and quality change over time, and a re-download may change results.
- District polygons are simplified and use pre-2022 boundaries. The planar distance approximation is accurate to a few km at these latitudes.
- Wind radius, storm size, rainfall, surge and the timing of local conditions are not considered. The licences of the two inputs have not been confirmed here. If geoBoundaries (gbOpen) turns out to require attribution, attribute it as its documentation asks before distributing these outputs.
- Association is not causation: an affected district is not evidence of crop loss, default or NPA.
- Mapped events per project year: 2015 none, 2016 two, 2017 none, 2018 three, 2019 one. Any statistical use of the map on the project's 2015 to 2019 data would rest on very few events.
