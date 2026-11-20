# v5 sample (100 rows)

> **SYNTHETIC. This is not PNB customer or portfolio data.** A small sample of the v5 dataset for sharing and review. Do not quote any figure here as real asset quality or insurance coverage.

| File | Content |
|---|---|
| `Financial_data_PNB_SYNTHETIC_v5_sample_100.csv` | **PNB layout, 23 columns: the 20 PNB loan fields + the 3 PNB insurance fields** (`crop_insurance_available`, `insurance_start_date`, `insurance_end_date`, last three columns) |
| `synthetic_training_v5_sample_100.csv` | The same 100 accounts, rich layout (39 columns: loan fields, weather and yield anomalies, climate zone, crop, the 3 insurance fields, targets) |

**How it was drawn:** from `../v5/` (11,000 rows), seed 2026, 20 rows per snapshot year (2015 to 2019), and within each year 6 Standard, 4 SMA 0, 4 SMA 1, 3 SMA 2, 3 NPA (the 30/20/18/16/16 mix). So the sample is stratified, not a purely random draw; it holds 60 train-split and 40 test-split rows.

**Insurance fields:** `crop_insurance_available` is Y or N. When N, the start and end dates are blank. Every date is inside 2015 to 2019.

**Limits:** 100 rows is for reading the layout, not for training or judging a model. The insured share in this sample (50%) differs from the full dataset (45.0%) by chance. Full rules and limitations: `../docs/RULESET_V5.md`.
