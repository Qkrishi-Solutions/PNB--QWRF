# Issue 49 annual sown-area benchmark

This benchmark forecasts annual sown area in hectares for each district and crop. It uses Issue 7's integrated dataset for forecasts, reconciles its area values and quality flags against Issue 1, and uses Issue 43's calendar for provisional sowing cutoffs. The calendar guides the forecast date for each annual target.

Run from the repository root with Python 3.12 and NumPy from the root requirements.

```text
python -B issue_49/forecast_area.py --out issue_49/outputs --test-years 2018 2019
python -B -m unittest discover -s issue_49/tests -v
```

The [report](outputs/REPORT.md) records mean absolute error (MAE), input hashes and the runtime. [Forecasts](outputs/area_forecasts.csv) include one row per method and test key, with reason codes when a prediction or score is unavailable. [Metrics](outputs/area_metrics.csv) report MAE and coverage by crop and year. The default run has a cohort shared by the three baselines, another shared by the baselines and area-only Ridge, and separate metrics on each method's available rows. Only the shared cohorts support direct comparisons.

To run against separately staged inputs, use `--repo-root` for a copy of the canonical directory layout or `--inputs` for a directory containing the six files by basename. The results were verified against the six input blobs on `dev` at `9bdb23e37bcf86fbe831375d12bfebe614d54073`. The [report](outputs/REPORT.md) lists their repository paths.

## Forecast timing

The default run is a retrospective comparison. It assumes strictly earlier annual area was available and uses provisional dates from Issue 43. The timing-aware run requires verified crop-year alignment, area release dates and prior Kharif rainfall release dates for the specific Issue 7 source. Keep unknown release dates blank.

Supply the verified records with `--timing`. The CSV must have one row per Issue 7 key and these fields.

```text
district_code,crop,year,forecast_date,area_available_on,lagged_kharif_available_on,source_reference
```

Use ISO dates. The rainfall date applies to the previous calendar year's June–September window. An input's release must precede the target forecast date, and rainfall can be released from the end of the window onward. Record the public evidence for the confirmed cutoff and release dates in `source_reference`. The timing-aware run adds a matched comparison of both Ridge candidates and all three baselines. The published backtest used the default mode because the canonical inputs lack verified historical release dates.
