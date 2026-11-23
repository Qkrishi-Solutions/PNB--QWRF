# IMD warning ingestion (Issue 47)

Official IMD warnings are an **authoritative operational input**. This folder defines a structured warning schema and a parser that turns advisories into that schema. Warnings are kept **separate from model-derived indicators** (Issue 48) and are never used as model features.

## Files

| File | Purpose |
|---|---|
| `imd_warning_schema_v1.json` | Warning schema: fields, allowed types, IMD colour-code severities, output columns. |
| `parse_imd_warnings.py` | Parser, schema validator, district matcher, the feature-separation guard and the CSV-header scan. |
| `test_imd_warnings.py` | 128 tests (passing and failing cases for each rule). |
| `samples/synthetic_imd_advisories.txt` | **Synthetic** advisories (5 valid, 3 deliberately invalid). Not real IMD data. |
| `samples/imd_warnings_parsed_sample.csv` | Parsed output, one row per warning and district. |
| `samples/imd_warnings_rejected_sample.csv` | Rejected advisories with the reasons. |
| `samples/imd_warnings_records_sample.json` | Record-level output (one record per warning, `districts` as a list). |

## Schema (v1)

One record per warning. Times are ISO 8601 with an explicit offset (IST, +05:30).

| Field | Type | Required | Notes |
|---|---|---|---|
| `warning_id` | string | yes | As given in the advisory. |
| `warning_type` | enum | yes | `heavy_rain`, `cyclone`, `heat_wave`, `cold_wave`, `thunderstorm`, `strong_wind`, `flood`, `other`. Mapped by an explicit keyword table; a keyword matches from the start of a word ("heavy rain" matches "heavy rainfall"; "cyclone" does not match "anticyclone"). A phrase must resolve to exactly one hazard: unknown, negated ("no heavy rain expected", "withdrawn") and multi-hazard ("thunderstorm with heavy rain") phrases become `other`, never guessed. `cyclone` absorbs `strong_wind`, so "cyclonic circulation with strong winds" stays `cyclone`. Wordings covered include "strong surface winds", "gusty winds", "gale", "flash flood", "thunder" and "heat-wave". "Hot and humid", "hot day", "storm surge" and "depression" are different advisories and stay `other`; the table is not exhaustive. |
| `warning_phrase` | string | no | The hazard phrase exactly as written, so `other` loses no information. |
| `severity` | enum | yes | IMD colour code: `green`, `yellow`, `orange`, `red`. |
| `state` | string | yes | |
| `districts` | list of strings | yes (at least 1) | Exactly as written in the advisory. |
| `issue_time`, `valid_from`, `valid_to` | datetime | yes | `valid_to` must be after `valid_from`. A missing `valid_to` is **not** defaulted: the advisory is rejected. Accepted input forms: `YYYY-MM-DD HH:MM[:SS]` or `T` separator, followed by `IST`, `UTC`, `Z` or a numeric offset (`+05:30`, `+0530`). A time with no zone is rejected, never assumed. |
| `headline` | string | no | |
| `source_kind` | enum | yes | `imd_advisory` or `synthetic_sample`. |

The district-level CSV has one row per (warning, district). Every column starts with `imd_warning_` and `imd_warning_record_class` is always `official_warning`. `imd_warning_district_match` has three values:
- `matched`: the advisory's state is Andhra Pradesh and the district is one of the 13 project districts. The code and name are filled.
- `unmatched`: the state is Andhra Pradesh but the district name is not a project district or a known spelling. The code stays blank. It is never dropped or guessed.
- `outside_project_state`: the advisory is for another state (for example Maharashtra or Telangana). A same-named district there is **never** matched to an AP district.

Matching ignores punctuation and case, tries an exact project name first and then a small alias table of common spellings (for example `Y.S.R. Kadapa`, `Anantapur`, `Nellore`, `Vizag`). The aliases apply to advisory names only, never to the project's own names. The table is not exhaustive, so an unlisted spelling comes out `unmatched`. If one advisory lists the same district twice (the same code under two spellings, or the same unmatched name twice), one row is kept and the collapse is printed by the run, so counts per district are not inflated.

## Input safety rules

- A repeated key inside one block rejects the block. This is what a missing `----------` separator between two advisories looks like, so advisories are never silently merged.
- A `warning_id` that appears in more than one block: **every** block with that id is rejected, valid or not. The result does not depend on block order, and a rejected block still reserves its id.
- The district file must map each name to one code and each code to one name, otherwise the run stops.
- Output filenames depend on `--source-kind`: `*_sample.*` only for `synthetic_sample`, `*_advisories.*` for real advisories.
- Nothing may be written into `imd_warnings/samples/` except a run of the bundled synthetic file with `--source-kind synthetic_sample`. The label alone does not unlock it: a file labelled `synthetic_sample` must have every advisory id (rejected ones included) start with `SYN-`, otherwise the run stops. Do not commit real advisory outputs.

## Keeping warnings out of model features

1. All output columns carry the `imd_warning_` prefix and the record class `official_warning`.
2. `assert_not_in_feature_frame(columns)` raises if any such column is about to enter a feature table.
3. The schema validator rejects fields that are not in the schema, so model scores cannot be added to a warning record.
4. `find_warning_columns(root)` reads the header of every CSV under `root` and returns how many it scanned and which contain an `imd_warning_` column. A test runs it on the repository and fails if any CSV outside this folder has such a column. Paths are judged relative to the repository root, so the result is the same in a checkout under `.claude/worktrees/`, and the test also fails if the scan looked at no file. It checks the current data, not future code.

**Open work, not done in this PR:** `assert_not_in_feature_frame` is **not called by any feature-building code** (Issues 3, 4, 5, 6 or 48), and this PR does not change those scripts. Until each feature builder calls it, the separation is protected only by the header scan above and by the naming convention. Wiring it in belongs to those issues, and Issue 48 should do it first.

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
- District aliases are a small explicit table of common spellings. Anything not in it and not a project name is reported as `unmatched`, not guessed. Only Andhra Pradesh districts are matched, so a project that later adds a state must change `PROJECT_STATE`.
- Warning types come from a keyword table and cover only the listed hazards. Compact times such as `0830 IST` are not accepted.
- This is a data-ingestion layer. It makes no claim about forecast skill or impact.
