# Data directory

This directory contains the project datasets that are intended to be shared
with collaborators. It is divided by data role so that source data, weather
data, and derived merged data are not confused with one another.

## Subfolders

| Folder | Contents | Documentation |
|---|---|---|
| `Crop/` | ICRISAT crop and district-level agricultural data | [Crop data guide](Crop/CROP_DATA_GUIDE.md) |
| `Weather/` | Daily historical weather data and the filled weather version | [Weather data guide](Weather/WEATHER_DATA_GUIDE.md) |
| `Merged/` | Versioned annual crop-weather datasets | [Merged data catalog](Merged/MERGED_DATA_CATALOG.md) |

## Dataset handling rules

- Treat each filename as a dataset version, not as an interchangeable copy.
- Use the recommended file in the relevant folder for new analysis.
- Keep older versions for traceability and comparison.
- Do not overwrite an older version when producing a new revision.
- Store processing reports and scripts in their appropriate project folders;
  this directory is primarily for datasets and their scientific guides.
- Confirm whether source or client-restricted data may be uploaded before
  publishing this directory.

## Current status

The directory is an initial curated handoff. The files are organized, but
future additions should update the relevant folder README with the source,
processing method, date/version, validation status, and recommended use.
