# IMD warning ingestion (Issue 47)

Official IMD warnings are an **authoritative operational input**. This folder defines a structured warning schema and a parser that turns advisories into that schema. Warnings are kept **separate from model-derived indicators** (Issue 48) and are never used as model features.

## Files

| File | Purpose |
|---|---|
| `imd_warning_schema_v1.json` | Warning schema: fields, allowed types, IMD colour-code severities, output columns. |
| `parse_imd_warnings.py` | Parser, schema validator, district matcher and the feature-separation guard. |
| `test_imd_warnings.py` | 53 tests (passing and failing cases for each rule). |
| `samples/synthetic_imd_advisories.txt` | **Synthetic** advisories (5 valid, 3 deliberately invalid). Not real IMD data. |
| `samples/imd_warnings_parsed_sample.csv` | Parsed output, one row per warning and district. |
| `samples/imd_warnings_rejected_sample.csv` | Rejected advisories with the reasons. |
| `samples/imd_warnings_records_sample.json` | Record-level output (one record per warning, `districts` as a list). |

## Schema (v1)

One record per warning. Times are ISO 8601 with an explicit offset (IST, +05:30).

| Field | Type | Required | Notes |
|---|---|---|---|
| `warning_id` | string | yes | As given in the advisory. |
| `warning_type` | enum | yes | `heavy_rain`, `cyclone`, `heat_wave`, `cold_wave`, `thunderstorm`, `strong_wind`, `flood`, `other`. Mapped by an explicit keyword table; unknown phrases become `other`, never guessed. |
| `severity` | enum | yes | IMD colour code: `green`, `yellow`, `orange`, `red`. |
| `state` | string | yes | |
| `districts` | list of strings | yes (at least 1) | Exactly as written in the advisory. |
| `issue_time`, `valid_from`, `valid_to` | datetime | yes | `valid_to` must be after `valid_from`. A missing `valid_to` is **not** defaulted: the advisory is rejected. |
| `headline` | string | no | |
| `source_kind` | enum | yes | `imd_advisory` or `synthetic_sample`. |

The district-level CSV has one row per (warning, district). Every column starts with `imd_warning_` and `imd_warning_record_class` is always `official_warning`. A district that cannot be matched to the 13 project districts is kept with `imd_warning_district_match = unmatched` and a blank code; it is never dropped or guessed.

## Keeping warnings out of model features

1. All output columns carry the `imd_warning_` prefix and the record class `official_warning`.
2. `assert_not_in_feature_frame(columns)` raises if any such column is about to enter a feature table. Any code that builds model features (for example the Issue 48 indicators) should call it.
3. The schema validator rejects fields that are not in the schema, so model scores cannot be added to a warning record.

## Run

From the repository root:

```bash
python3 imd_warnings/parse_imd_warnings.py \
  --input imd_warnings/samples/synthetic_imd_advisories.txt \
  --districts weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv \
  --outdir imd_warnings/samples \
  --source-kind synthetic_sample

python3 -m pytest imd_warnings/test_imd_warnings.py -v
```

Requires `pandas` and `pytest`. For real advisories use `--source-kind imd_advisory` (the default).

## Results on the synthetic sample (actual)

- 8 advisories read, 5 parsed and schema-valid, 3 rejected.
- 14 district rows: 13 matched, 1 unmatched (`Hyderabad`, outside the project districts, on purpose).
- Rejections: `SYN-0006` missing `VALID TO`, `SYN-0007` invalid colour code `Purple`, `SYN-0008` `valid_to` before `valid_from`.
- `SYN-0005` (`Dense fog`) maps to type `other`.

## Limitations

- **The input format is assumed.** Advisories are read as blocks of `KEY: value` lines. This simplified format was defined for this issue; it is not confirmed to match real IMD bulletins, which may be tabular, PDF or a CAP/RSS feed. A format adapter and a check on real advisories are needed before operational use.
- The sample data is synthetic and describes no real event. No real PNB data is included.
- District aliases (for example `Anantapur`, `Nellore`, `Kadapa`) are a small explicit table. Anything not in it is reported as `unmatched`, not guessed.
- Warning types come from a keyword table and cover only the listed hazards.
- This is a data-ingestion layer. It makes no claim about forecast skill or impact.
