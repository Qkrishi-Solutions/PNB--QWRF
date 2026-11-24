"""
Build Canonical Extended Meteorological Dataset (Issue #45)
Module: W1 (Phase 2: Severe-weather intelligence)

Extracts and standardizes extended atmospheric and marine variables from ERA5:
- Mean Sea Level Pressure (msl_min_hpa, msl_mean_hpa)
- Wind Speed & Gust (wind_speed_mean_kmh, wind_speed_max_kmh, wind_gust_max_kmh)
- Wind Direction (wind_direction_deg)
- Sea Surface Temperature (sst_mean_c) for coastal districts; NaN for inland

Generates:
1. weather_extended_v1.csv
2. weather_extended_coverage_report_v1.csv
3. WEATHER_EXTENDED_DATA_DICTIONARY.md
4. README.md
"""

from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import xarray as xr

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "weather_extended"
PROJECT_ROOT = REPO_ROOT.parent
RAW_DIR = PROJECT_ROOT / "dat1" / "raw" / "era5" / "historical"
BOUNDARY_FILE = PROJECT_ROOT / "dat1" / "interim" / "ap_districts_validated.geojson"

COASTAL_DISTRICTS_RAW = {
    "East Godavari", "Guntur", "Krishna", "Spsr Nellore",
    "Prakasam", "Srikakulam", "Visakhapatanam", "Vizianagaram", "West Godavari"
}

DISTRICT_MAPPING = {
    "Anantapur": (502, "ANANTAPURAMU"),
    "Chittoor": (503, "CHITTOOR"),
    "Y.S.R.": (504, "YSR"),
    "East Godavari": (505, "EAST GODAVARI"),
    "Guntur": (506, "GUNTUR"),
    "Krishna": (510, "KRISHNA"),
    "Kurnool": (511, "KURNOOL"),
    "Spsr Nellore": (515, "SPR NELLORE"),
    "Prakasam": (517, "PRAKASAM"),
    "Srikakulam": (519, "SRIKAKULAM"),
    "Visakhapatanam": (520, "VISAKHAPATNAM"),
    "Vizianagaram": (521, "VIZIANAGARAM"),
    "West Godavari": (523, "WEST GODAVARI"),
}


def load_boundaries() -> gpd.GeoDataFrame:
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"Boundary file missing: {BOUNDARY_FILE}")
    gdf = gpd.read_file(BOUNDARY_FILE).to_crs("EPSG:4326")
    gdf["district"] = gdf["district"].astype(str).str.strip()
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def compute_stream_records(inst_path: Path, max_path: Path, districts: gpd.GeoDataFrame) -> list[dict]:
    with xr.open_dataset(inst_path) as ds_i, xr.open_dataset(max_path) as ds_m:
        ds_inst = ds_i.load()
        ds_max = ds_m.load()

    for ds in (ds_inst, ds_max):
        t_key = next((c for c in ("valid_time", "time") if c in ds.coords), None)
        if t_key and t_key != "valid_time":
            ds.rename({t_key: "valid_time"}, inplace=True)

    u10 = ds_inst["u10"]
    v10 = ds_inst["v10"]
    msl = ds_inst["msl"]
    sst = ds_inst["sst"]
    fg10 = ds_max["fg10"]

    # Hourly conversions
    ws_hourly_kmh = np.sqrt(u10**2 + v10**2) * 3.6
    gust_hourly_kmh = fg10 * 3.6
    msl_hourly_hpa = msl / 100.0
    sst_hourly_c = sst - 273.15

    # Daily resampled fields
    daily_ws_mean = ws_hourly_kmh.resample(valid_time="1D").mean(skipna=True)
    daily_ws_max = ws_hourly_kmh.resample(valid_time="1D").max(skipna=True)
    daily_gust_max = gust_hourly_kmh.resample(valid_time="1D").max(skipna=True)
    daily_msl_min = msl_hourly_hpa.resample(valid_time="1D").min(skipna=True)
    daily_msl_mean = msl_hourly_hpa.resample(valid_time="1D").mean(skipna=True)
    daily_sst_mean = sst_hourly_c.resample(valid_time="1D").mean(skipna=True)

    # Daily vector components for wind direction
    daily_u_mean = u10.resample(valid_time="1D").mean(skipna=True)
    daily_v_mean = v10.resample(valid_time="1D").mean(skipna=True)

    lats = daily_ws_mean.latitude.values
    lons = daily_ws_mean.longitude.values
    xx, yy = np.meshgrid(lons, lats)
    grid_points = gpd.GeoSeries([Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())], crs="EPSG:4326")

    # Ocean points mask
    sst_ref = sst.isel(valid_time=0).values.ravel()
    is_ocean = np.isfinite(sst_ref)
    ocean_points = grid_points[is_ocean]

    districts_proj = districts.to_crs("EPSG:32644")
    ocean_points_proj = ocean_points.to_crs("EPSG:32644")

    district_land_indices = {}
    district_marine_indices = {}

    for _, row in districts.iterrows():
        d_name = row.district
        inside_land = np.asarray(grid_points.intersects(row.geometry))
        district_land_indices[d_name] = np.where(inside_land)[0]

        if d_name in COASTAL_DISTRICTS_RAW:
            row_proj = districts_proj[districts_proj.district == d_name].iloc[0]
            buffer_75km = row_proj.geometry.buffer(75000)
            nearby_ocean = np.asarray(ocean_points_proj.intersects(buffer_75km))
            marine_orig_indices = ocean_points.index[nearby_ocean].to_numpy()
            district_marine_indices[d_name] = marine_orig_indices
        else:
            district_marine_indices[d_name] = np.array([], dtype=int)

    dates = pd.to_datetime(daily_ws_mean.valid_time.values)
    records = []

    for d_idx, dt in enumerate(dates):
        date_str = dt.strftime("%Y-%m-%d")

        ws_m_vals = daily_ws_mean.isel(valid_time=d_idx).values.ravel()
        ws_x_vals = daily_ws_max.isel(valid_time=d_idx).values.ravel()
        gust_x_vals = daily_gust_max.isel(valid_time=d_idx).values.ravel()
        msl_min_vals = daily_msl_min.isel(valid_time=d_idx).values.ravel()
        msl_m_vals = daily_msl_mean.isel(valid_time=d_idx).values.ravel()
        sst_vals = daily_sst_mean.isel(valid_time=d_idx).values.ravel()
        u_vals = daily_u_mean.isel(valid_time=d_idx).values.ravel()
        v_vals = daily_v_mean.isel(valid_time=d_idx).values.ravel()

        for _, row in districts.iterrows():
            d_raw = row.district
            l_idx = district_land_indices[d_raw]
            if len(l_idx) == 0:
                continue

            d_code, d_canon = DISTRICT_MAPPING[d_raw]

            ws_m = float(np.nanmean(ws_m_vals[l_idx]))
            ws_x = float(np.nanmax(ws_x_vals[l_idx]))
            gust_x = float(np.nanmax(gust_x_vals[l_idx]))
            msl_min = float(np.nanmin(msl_min_vals[l_idx]))
            msl_m = float(np.nanmean(msl_m_vals[l_idx]))

            # Wind direction from mean u and v components
            u_bar = float(np.nanmean(u_vals[l_idx]))
            v_bar = float(np.nanmean(v_vals[l_idx]))
            # Meteorological direction from which wind blows (0=N, 90=E, 180=S, 270=W)
            wdir = float((270.0 - np.arctan2(v_bar, u_bar) * (180.0 / np.pi)) % 360.0)

            # SST for coastal vs inland
            m_idx = district_marine_indices[d_raw]
            if len(m_idx) > 0:
                sst_val = float(np.nanmean(sst_vals[m_idx]))
            else:
                sst_val = np.nan

            records.append({
                "district_code": d_code,
                "district_name": d_canon,
                "date": date_str,
                "era5_grid_cells": int(len(l_idx)),
                "msl_min_hpa": round(msl_min, 2),
                "msl_mean_hpa": round(msl_m, 2),
                "wind_speed_mean_kmh": round(ws_m, 2),
                "wind_speed_max_kmh": round(ws_x, 2),
                "wind_gust_max_kmh": round(gust_x, 2),
                "wind_direction_deg": round(wdir, 1),
                "sst_mean_c": round(sst_val, 2) if np.isfinite(sst_val) else np.nan,
            })

    return records


def build():
    MODULE_DIR.mkdir(parents=True, exist_ok=True)
    districts = load_boundaries()
    print(f"Loaded {len(districts)} districts.")

    # Identified 2015 streams
    stream_pairs = [
        (RAW_DIR / "era5_ap_2015_01_instant.nc", RAW_DIR / "era5_ap_2015_01_max.nc"),
        (RAW_DIR / "era5_ap_2015_02_instant.nc", RAW_DIR / "era5_ap_2015_02_max.nc"),
        (RAW_DIR / "era5_ap_2015_03_instant.nc", RAW_DIR / "era5_ap_2015_03_max.nc"),
        (RAW_DIR / "era5_ap_2015_04_instant.nc", RAW_DIR / "era5_ap_2015_04_max.nc"),
        (RAW_DIR / "era5_ap_2015_05_instant.nc", RAW_DIR / "era5_ap_2015_05_max.nc"),
        (RAW_DIR / "era5_ap_2015_06_instant.nc", RAW_DIR / "era5_ap_2015_06_max.nc"),
        (RAW_DIR / "era5_ap_2015_m07_m08_instant.nc", RAW_DIR / "era5_ap_2015_m07_m08_max.nc"),
        (RAW_DIR / "era5_ap_2015_m09_m10_instant.nc", RAW_DIR / "era5_ap_2015_m09_m10_max.nc"),
        (RAW_DIR / "era5_ap_2015_m11_m12_instant.nc", RAW_DIR / "era5_ap_2015_m11_m12_max.nc"),
    ]

    all_records = []
    for inst, mx in stream_pairs:
        if not inst.exists() or not mx.exists():
            print(f"Skipping missing stream pair: {inst.name}, {mx.name}")
            continue
        print(f"Processing stream pair: {inst.name} ...")
        recs = compute_stream_records(inst, mx, districts)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["district_code", "date"]).sort_values(["date", "district_code"]).reset_index(drop=True)

    # 1. Save weather_extended_v1.csv
    out_csv = MODULE_DIR / "weather_extended_v1.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows, {df['date'].nunique()} dates, {df['district_code'].nunique()} districts).")

    # 2. Save Coverage & Audit Report
    coverage_rows = []
    for d_code, g in df.groupby("district_code"):
        d_name = g["district_name"].iloc[0]
        n_days = len(g)
        is_coastal = d_name not in {"ANANTAPURAMU", "CHITTOOR", "KURNOOL", "YSR"}
        sst_nulls = int(g["sst_mean_c"].isna().sum())
        sst_pct_null = round(sst_nulls / n_days * 100.0, 1)

        coverage_rows.append({
            "district_code": d_code,
            "district_name": d_name,
            "is_coastal": is_coastal,
            "total_days": n_days,
            "start_date": g["date"].min(),
            "end_date": g["date"].max(),
            "grid_cells": int(g["era5_grid_cells"].iloc[0]),
            "msl_min_range_hpa": f"{g['msl_min_hpa'].min():.1f} - {g['msl_min_hpa'].max():.1f}",
            "wind_speed_max_kmh": round(g["wind_speed_max_kmh"].max(), 1),
            "wind_gust_max_kmh": round(g["wind_gust_max_kmh"].max(), 1),
            "sst_pct_null": sst_pct_null,
            "status": "VALID_COASTAL" if is_coastal and sst_nulls == 0 else ("VALID_INLAND_NULL" if not is_coastal and sst_pct_null == 100.0 else "INCONSISTENT"),
        })

    cov_df = pd.DataFrame(coverage_rows).sort_values("district_code").reset_index(drop=True)
    out_cov = MODULE_DIR / "weather_extended_coverage_report_v1.csv"
    cov_df.to_csv(out_cov, index=False)
    print(f"Saved coverage report: {out_cov}")


if __name__ == "__main__":
    build()
