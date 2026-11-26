# Issue 49 annual sown-area backtest

## Matched baseline results

| Crop | Scored rows | Last value MAE (ha) | Climatology MAE (ha) | Trend MAE (ha) | Flags |
| --- | ---: | ---: | ---: | ---: | --- |
| all | 265 | 4,068.91 | 4,906.60 | 4,569.15 |  |
| chickpea | 23 | 5,803.48 | 7,422.07 | 7,604.29 |  |
| finger_millet | 21 | 322.38 | 399.88 | 452.29 |  |
| groundnut | 26 | 8,293.08 | 9,576.63 | 7,687.88 |  |
| maize | 26 | 4,620.77 | 3,711.60 | 5,690.71 |  |
| pearl_millet | 22 | 1,454.09 | 1,475.42 | 1,798.79 |  |
| pigeonpea | 26 | 2,788.46 | 3,881.44 | 5,024.23 |  |
| rabi_sorghum | 21 | 3,979.05 | 3,565.52 | 4,808.40 |  |
| rice | 26 | 13,444.62 | 18,698.62 | 12,870.06 |  |
| sesamum | 25 | 1,222.00 | 1,443.13 | 1,296.33 |  |
| sugarcane | 26 | 971.92 | 1,165.71 | 1,006.79 |  |
| sunflower | 23 | 383.48 | 645.51 | 617.30 |  |

## Area-only Ridge

Across 265 matched rows, Ridge did not beat last_value (8,314.29 vs 4,068.91 ha MAE).

| Crop | Scored rows | Best naive baseline | Baseline MAE (ha) | Ridge MAE (ha) | Flags |
| --- | ---: | --- | ---: | ---: | --- |
| all | 265 | last_value | 4,068.91 | 8,314.29 |  |
| chickpea | 23 | last_value | 5,803.48 | 19,569.28 |  |
| finger_millet | 21 | last_value | 322.38 | 596.69 |  |
| groundnut | 26 | trend | 7,687.88 | 16,649.64 |  |
| maize | 26 | climatology | 3,711.60 | 12,435.26 |  |
| pearl_millet | 22 | last_value | 1,454.09 | 2,378.84 |  |
| pigeonpea | 26 | last_value | 2,788.46 | 5,788.95 |  |
| rabi_sorghum | 21 | climatology | 3,565.52 | 8,326.52 |  |
| rice | 26 | trend | 12,870.06 | 18,838.77 |  |
| sesamum | 25 | last_value | 1,222.00 | 2,175.53 |  |
| sugarcane | 26 | last_value | 971.92 | 1,729.09 |  |
| sunflower | 23 | last_value | 383.48 | 765.16 |  |

## Weather comparison

Weather candidate not evaluated. No dated records corroborate the annual-area publication dates or the historical release of the exact rainfall product in the Issue 7 export. Weather improvement over a naive baseline is unestablished.

The Issue 7 column dictionary leaves forecast-time availability unresolved. The [IMD rainfall archive](https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html) describes the historical product and yearly files, with no vintage dates for the export used here.

## Scope and assumptions

- The target is annual district-crop sown area. The 11 crops in Issue 7 define the benchmark.
- Forecast and metric files contain retrospective backtest results. Two test years provide limited evidence. Scores do not establish a validated operational model.
- Missing and negative targets and ambiguous zeros are excluded. An observed zero remains zero.
- Per-crop sample flags use fewer than 10 scored rows, fewer than five districts or fewer than two test years for pooled results.
- Ridge is fit per crop across districts using recent observed area and calendar year. Each fit uses its own training means and scales, with a fixed penalty of 1.
- Ridge requires ten training rows across at least five districts and two earlier target years.
- Forecast years 2018, 2019. Python 3.12.14, NumPy 2.3.5.
- The baselines and area-only Ridge assume strictly earlier reported annual area was available. Calendar cutoffs are provisional because crop-year alignment and historical publication dates remain unverified.

## Input SHA-256 with LF line endings

- `CropLongFormatv1/crop_long_v1.csv` 4765429e3acb7cf3eb9e64226babc9d29d910fb8deb7d0c42ac48e3baddf727d
- `crop_calendar/crop_calendar_v1.csv` 0cc1fca9aef10858ea74b991378df98f9f61cf837582055d5c4e97657f7f85e0
- `crop_calendar/district_crop_calendar_v1.csv` 427d70292b1bebcdf221977a00763f5e8bdd86727fce565b80df05378bd4b649
- `ml_ready_dataset/ml_ready_dataset_v1.csv` d0c3314865f0892a7c35d71412beaabe23092fde8eb2cbabb6127402543b47a3
- `ml_ready_dataset/ml_ready_column_dictionary_v1.csv` 1dd5d75d25ef58f07de8ab4ca704c985e4f479669a2169b4d5bace9ef90bf276
- `ml_ready_dataset/reproducibility_report_v1.json` 7f193acaaa4e7647f0866be0d2f782659aa705985f778ebea9b6ac6b9682db6d

## Reproduction

```text
python -B issue_49/forecast_area.py --out "issue_49/outputs" --test-years 2018 2019
python -B -m unittest discover -s issue_49/tests -v
```
