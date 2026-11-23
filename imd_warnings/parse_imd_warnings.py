"""Parse IMD warning advisories into the structured imd_warning schema (Issue 47).

Official IMD warnings are an authoritative operational input. They are kept
separate from model-derived indicators: every output column carries the
`imd_warning_` prefix, and `assert_not_in_feature_frame` lets any feature
builder refuse such columns.

Input format (ASSUMED, simplified): advisory blocks separated by a line of
dashes, each block made of `KEY: value` lines. See README.md. Real IMD
bulletins must be checked against this format before operational use.

Nothing is imputed: a missing or invalid required field rejects the advisory
(it is listed in the rejects file with the reasons), and an unrecognised
district is kept as `unmatched`, never guessed.
"""

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "imd_warning_schema_v1.json"

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

# Explicit keyword table; first match wins. Unknown phrases become "other".
TYPE_KEYWORDS = [
    ("cyclone", "cyclone"),
    ("cyclonic", "cyclone"),
    ("heavy rain", "heavy_rain"),
    ("very heavy rain", "heavy_rain"),
    ("extremely heavy rain", "heavy_rain"),
    ("heat wave", "heat_wave"),
    ("heatwave", "heat_wave"),
    ("cold wave", "cold_wave"),
    ("thunderstorm", "thunderstorm"),
    ("lightning", "thunderstorm"),
    ("strong wind", "strong_wind"),
    ("squall", "strong_wind"),
    ("flood", "flood"),
]

# IMD / project spellings that differ from the project district names
# (same four India Post mappings as address_expansion/ plus common IMD forms).
DISTRICT_ALIASES = {
    "ANANTAPUR": "ANANTAPURAMU",
    "SPSR NELLORE": "SPR NELLORE",
    "NELLORE": "SPR NELLORE",
    "SRI POTTI SRIRAMULU NELLORE": "SPR NELLORE",
    "VISAKHAPATANAM": "VISAKHAPATNAM",
    "VISAKHAPATNAM": "VISAKHAPATNAM",
    "Y S R": "YSR",
    "YSR KADAPA": "YSR",
    "KADAPA": "YSR",
    "CUDDAPAH": "YSR",
}

REQUIRED_KEYS = {
    "ADVISORY-ID": "warning_id",
    "ISSUED": "issue_time",
    "STATE": "state",
    "WARNING": "warning_phrase",
    "COLOUR CODE": "severity",
    "DISTRICTS": "districts",
    "VALID FROM": "valid_from",
    "VALID TO": "valid_to",
}
OPTIONAL_KEYS = {"HEADLINE": "headline"}

PREFIX = "imd_warning_"


def load_schema(path=SCHEMA_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------
# Separation from model features
# ---------------------------------------------------------

def assert_not_in_feature_frame(columns):
    """Raise if any official-warning column is about to enter a feature table."""
    offending = sorted(c for c in columns if str(c).startswith(PREFIX))
    if offending:
        raise ValueError(
            "Official IMD warning columns must not be used as model features: "
            + ", ".join(offending)
        )


# ---------------------------------------------------------
# Field parsing
# ---------------------------------------------------------

def parse_time(text):
    """Parse 'YYYY-MM-DD HH:MM IST|UTC' to an aware datetime; None if invalid."""
    match = re.fullmatch(
        r"\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})\s*(IST|UTC)\s*", text or ""
    )
    if not match:
        return None
    try:
        naive = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return naive.replace(tzinfo=IST if match.group(3) == "IST" else UTC).astimezone(IST)


def classify_type(phrase):
    lowered = (phrase or "").lower()
    for keyword, warning_type in TYPE_KEYWORDS:
        if keyword in lowered:
            return warning_type
    return "other"


def normalise_district(name):
    text = re.sub(r"[^A-Za-z ]", " ", str(name)).upper()
    text = re.sub(r"\s+", " ", text).strip()
    return DISTRICT_ALIASES.get(text, text)


def split_blocks(text):
    blocks, current = [], []
    for line in text.splitlines():
        if re.fullmatch(r"\s*-{3,}\s*", line):
            if current:
                blocks.append(current)
            current = []
        elif line.strip() and not line.lstrip().startswith("#"):
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def parse_block(lines, source_kind, schema):
    """Return (record or None, errors). Never fills in a missing value."""
    raw = {}
    errors = []
    for line in lines:
        if ":" not in line:
            errors.append(f"unparseable line: {line.strip()[:60]}")
            continue
        key, value = line.split(":", 1)
        raw[key.strip().upper()] = value.strip()

    unknown = set(raw) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    for key in sorted(unknown):
        errors.append(f"unknown field: {key}")

    for key in REQUIRED_KEYS:
        if not raw.get(key):
            errors.append(f"missing required field: {key}")

    record = {"warning_id": raw.get("ADVISORY-ID", "")}
    record["state"] = raw.get("STATE", "")
    record["warning_type"] = classify_type(raw.get("WARNING", "")) if raw.get("WARNING") else ""

    colour = raw.get("COLOUR CODE", "").strip().lower()
    if raw.get("COLOUR CODE") and colour not in schema["severity_levels"]:
        errors.append(f"invalid colour code: {raw['COLOUR CODE']}")
    record["severity"] = colour if colour in schema["severity_levels"] else ""

    districts = [d.strip() for d in raw.get("DISTRICTS", "").split(",") if d.strip()]
    record["districts"] = districts

    times = {}
    for key, field in (("ISSUED", "issue_time"), ("VALID FROM", "valid_from"), ("VALID TO", "valid_to")):
        if raw.get(key):
            parsed = parse_time(raw[key])
            if parsed is None:
                errors.append(f"invalid time for {key}: {raw[key]}")
            times[field] = parsed
        else:
            times[field] = None
    record.update({k: (v.isoformat() if v else "") for k, v in times.items()})

    if times.get("valid_from") and times.get("valid_to") and times["valid_to"] <= times["valid_from"]:
        errors.append("valid_to must be after valid_from")
    if times.get("issue_time") and times.get("valid_to") and times["issue_time"] > times["valid_to"]:
        errors.append("issue_time is after valid_to")

    record["headline"] = raw.get("HEADLINE", "")
    record["source_kind"] = source_kind
    return (None if errors else record), errors, record


def parse_advisories(text, source_kind="imd_advisory", schema=None):
    """Parse all blocks. Returns (records, rejects); rejects keep their reasons."""
    schema = schema or load_schema()
    if source_kind not in schema["source_kinds"]:
        raise ValueError(f"source_kind must be one of {schema['source_kinds']}")
    records, rejects = [], []
    for lines in split_blocks(text):
        record, errors, partial = parse_block(lines, source_kind, schema)
        if record is None:
            rejects.append({
                "warning_id": partial.get("warning_id", ""),
                "errors": "; ".join(errors),
                "block": " | ".join(line.strip() for line in lines),
            })
        else:
            records.append(record)
    return records, rejects


# ---------------------------------------------------------
# Schema validation of a record
# ---------------------------------------------------------

def validate_record(record, schema=None):
    """Return a list of schema violations for one parsed record (empty = valid)."""
    schema = schema or load_schema()
    problems = []
    for field in schema["fields"]:
        name, kind = field["name"], field["type"]
        value = record.get(name)
        blank = value in (None, "", [])
        if blank:
            if field["required"]:
                problems.append(f"{name}: required")
            continue
        if kind == "enum":
            if value not in schema[field["enum_ref"]]:
                problems.append(f"{name}: {value!r} not allowed")
        elif kind == "list[string]":
            if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
                problems.append(f"{name}: must be a list of non-empty strings")
        elif kind == "datetime":
            try:
                parsed = datetime.fromisoformat(value)
            except (TypeError, ValueError):
                problems.append(f"{name}: not ISO 8601")
            else:
                if parsed.tzinfo is None:
                    problems.append(f"{name}: missing time zone offset")
        elif kind == "string" and not isinstance(value, str):
            problems.append(f"{name}: must be a string")
    unexpected = set(record) - {f["name"] for f in schema["fields"]}
    for name in sorted(unexpected):
        problems.append(f"{name}: not in schema")
    return problems


# ---------------------------------------------------------
# District-level output
# ---------------------------------------------------------

def load_project_districts(path):
    frame = pd.read_csv(path, usecols=["district_code", "district_name"]).drop_duplicates()
    return {
        normalise_district(name): (int(code), str(name))
        for code, name in zip(frame.district_code, frame.district_name, strict=True)
    }


def to_district_rows(records, project_districts, schema=None):
    schema = schema or load_schema()
    rows = []
    for rec in records:
        for raw_name in rec["districts"]:
            hit = project_districts.get(normalise_district(raw_name))
            rows.append({
                "imd_warning_id": rec["warning_id"],
                "imd_warning_type": rec["warning_type"],
                "imd_warning_severity": rec["severity"],
                "imd_warning_state": rec["state"],
                "imd_warning_district_raw": raw_name,
                "imd_warning_district_code": hit[0] if hit else "",
                "imd_warning_district_name": hit[1] if hit else "",
                "imd_warning_district_match": "matched" if hit else "unmatched",
                "imd_warning_issue_time": rec["issue_time"],
                "imd_warning_valid_from": rec["valid_from"],
                "imd_warning_valid_to": rec["valid_to"],
                "imd_warning_headline": rec["headline"],
                "imd_warning_source_kind": rec["source_kind"],
                "imd_warning_record_class": schema["record_class"],
            })
    return pd.DataFrame(rows, columns=schema["district_output_columns"])


def run(input_path, districts_path, outdir, source_kind):
    schema = load_schema()
    text = Path(input_path).read_text(encoding="utf-8")
    records, rejects = parse_advisories(text, source_kind, schema)

    for rec in records:
        problems = validate_record(rec, schema)
        if problems:
            raise RuntimeError(f"Parsed record {rec['warning_id']} violates schema: {problems}")

    project = load_project_districts(districts_path)
    rows = to_district_rows(records, project, schema)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(outdir / "imd_warnings_parsed_sample.csv", index=False)
    pd.DataFrame(rejects, columns=["warning_id", "errors", "block"]).to_csv(
        outdir / "imd_warnings_rejected_sample.csv", index=False
    )
    (outdir / "imd_warnings_records_sample.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    unmatched = rows[rows["imd_warning_district_match"] == "unmatched"]
    print("IMD WARNING PARSE COMPLETED")
    print("---------------------------")
    print(f"Advisories read: {len(records) + len(rejects)}")
    print(f"Parsed and schema-valid: {len(records)}")
    print(f"Rejected: {len(rejects)}")
    print(f"District rows: {len(rows)} (matched {len(rows) - len(unmatched)}, unmatched {len(unmatched)})")
    if len(unmatched):
        print("Unmatched districts:", ", ".join(sorted(set(unmatched["imd_warning_district_raw"]))))
    print(f"Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issue 47: parse IMD advisories into the warning schema")
    parser.add_argument("--input", required=True, help="Advisory text file")
    parser.add_argument("--districts", required=True,
                        help="CSV with district_code and district_name (e.g. the Issue 3 weather anomalies file)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--source-kind", default="imd_advisory", choices=["imd_advisory", "synthetic_sample"])
    args = parser.parse_args()
    run(args.input, args.districts, args.outdir, args.source_kind)
