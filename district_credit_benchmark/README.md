# Issue #41 — District Credit Benchmark

This package contains the updated benchmark CSV, assumptions/provenance note, and validator.

## Contents
- `data/district_allocation_benchmark_v1.csv`
- `data/district_allocation_benchmark_v1_assumptions.md`
- `validate_benchmark.py`
- `source/RBI_Table_3_9_March_2019.xlsx` (original downloaded RBI workbook; retained as source evidence)

## Run validation
From this folder:
```powershell
python validate_benchmark.py
```

The CSV keeps NABARD credit-potential values and RBI outstanding-credit values in separate columns. Do not combine or compare them as if they were the same measure. RBI figures are for Scheduled Commercial Banks excluding Regional Rural Banks.
