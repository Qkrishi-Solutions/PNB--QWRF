"""Expand loan-portfolio addresses and map pincodes to project districts.

Input grain: one row per loan account (field names follow
``financial_data_requirements/DATA_DICTIONARY.md``).
Output grain: unchanged, one row per input row (join cardinality is asserted).

Outputs
- ``address_expanded.csv``: structured address parts plus state/district mapping.
  Contains address text, so it is written to a private folder, never committed.
- ``address_expansion_report.csv``: counts only (no personal data).

Rules (no guessing): unparseable addresses, unknown or ambiguous pincodes and
out-of-scope pincodes are labelled with a status and left unmapped.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("expand_addresses")

ADDRESS_COLUMN = "borrowers address"
PINCODE_COLUMN = "pincode"
ACCOUNT_COLUMN = "account number"

EXPECTED_ADDRESS_PARTS = 4  # house/plot, area, locality, city (observed in sample)
PINCODE_PATTERN = r"[1-9]\d{5}"
EMBEDDED_PINCODE = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")
LIST_SEPARATOR = "|"

PARSED_COLUMNS = ["addr_house", "addr_area", "addr_locality", "addr_city"]


def normalise_name(value: object) -> str:
    """Upper-case a place name and collapse punctuation and whitespace."""
    cleaned = re.sub(r"[^A-Z0-9]+", " ", str(value).upper())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_address(address: object) -> dict[str, str]:
    """Split one address into parts; label it instead of guessing when it is odd."""
    empty = dict.fromkeys(PARSED_COLUMNS, "")
    if address is None or pd.isna(address) or not str(address).strip():
        return {**empty, "address_parse_status": "missing_address", "address_pincode": ""}
    text = str(address)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    embedded = EMBEDDED_PINCODE.findall(text)
    pincode = embedded[0] if len(set(embedded)) == 1 else ""
    if len(parts) != EXPECTED_ADDRESS_PARTS:
        return {
            **empty,
            "address_parse_status": "unexpected_part_count",
            "address_pincode": pincode,
        }
    parsed = dict(zip(PARSED_COLUMNS, parts, strict=True))
    return {**parsed, "address_parse_status": "parsed", "address_pincode": pincode}


def clean_pincode(value: object) -> str:
    """Return a 6-digit pincode string, or ``""`` if the value is not valid."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text if re.fullmatch(PINCODE_PATTERN, text) else ""


def city_check(city: str, source_districts: str, known_districts: set[str]) -> str:
    """Compare the address city with the pincode's district(s).

    ``consistent``: city equals one of the pincode's districts.
    ``conflict``: city is itself a known district name but not one of this pincode's.
    ``not_comparable``: city is not a district name (e.g. a locality), so no judgement.
    """
    if not city:
        return "no_city"
    city_norm = normalise_name(city)
    if not source_districts:
        return "not_comparable"
    districts = set(source_districts.split(LIST_SEPARATOR))
    if city_norm in districts:
        return "consistent"
    return "conflict" if city_norm in known_districts else "not_comparable"


def expand(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add parsed address parts and pincode-derived state/district to ``frame``."""
    parsed = pd.DataFrame([parse_address(a) for a in frame[ADDRESS_COLUMN]], index=frame.index)
    result = pd.concat([frame[[ACCOUNT_COLUMN, PINCODE_COLUMN]].copy(), parsed], axis=1)
    result["pincode_clean"] = frame[PINCODE_COLUMN].map(clean_pincode)

    ref = reference.copy()
    ref["pincode"] = ref["pincode"].astype(str)
    ref = ref.rename(columns={"pincode": "pincode_clean"})
    if ref["pincode_clean"].duplicated().any():
        raise ValueError("Reference has duplicate pincodes; cannot join one-to-one")
    rows_before = len(result)
    merged = result.merge(ref, how="left", on="pincode_clean", validate="many_to_one")
    if len(merged) != rows_before:
        raise AssertionError("Join changed the row count")

    merged["mapping_status"] = merged["mapping_status"].fillna("")
    merged.loc[merged["pincode_clean"].eq(""), "mapping_status"] = "invalid_pincode"
    unmatched = merged["mapping_status"].eq("") & merged["pincode_clean"].ne("")
    merged.loc[unmatched, "mapping_status"] = "not_in_reference"
    for column in [
        "state_name",
        "state_code",
        "source_districts",
        "project_district_codes",
        "district_code",
    ]:
        merged[column] = merged[column].fillna("")

    known = {
        d for value in reference["source_districts"].dropna() for d in value.split(LIST_SEPARATOR)
    }
    merged["city_check"] = [
        city_check(c, s, known)
        for c, s in zip(merged["addr_city"], merged["source_districts"], strict=True)
    ]
    embedded = merged["address_pincode"].ne("")
    merged["address_pincode_check"] = "no_pincode_in_address"
    merged.loc[
        embedded & merged["address_pincode"].eq(merged["pincode_clean"]), "address_pincode_check"
    ] = "consistent"
    merged.loc[
        embedded & merged["address_pincode"].ne(merged["pincode_clean"]), "address_pincode_check"
    ] = "conflict"
    return merged.drop(columns=["n_states", "n_source_districts"], errors="ignore")


def build_report(expanded: pd.DataFrame) -> pd.DataFrame:
    """Summarise counts only; contains no row-level data."""
    rows = [("total_rows", "all", len(expanded))]
    rows.append(("unique_account_keys", "all", int(expanded[ACCOUNT_COLUMN].nunique())))
    for column in ["address_parse_status", "mapping_status", "city_check", "address_pincode_check"]:
        for value, count in expanded[column].value_counts().sort_index().items():
            rows.append((column, value, int(count)))
    mapped = int(expanded["mapping_status"].eq("mapped").sum())
    rows.append(
        ("mapped_to_project_district_pct", "all", round(100 * mapped / max(len(expanded), 1), 2))
    )
    return pd.DataFrame(rows, columns=["check", "value", "count"], dtype=object)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="loan portfolio CSV (private)")
    parser.add_argument(
        "--reference", type=Path, default=Path("address_expansion/pincode_reference_v1.csv")
    )
    parser.add_argument("--outdir", type=Path, default=Path("data/private/address_expansion"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frame = pd.read_csv(args.input, dtype=str)
    missing = {ADDRESS_COLUMN, PINCODE_COLUMN, ACCOUNT_COLUMN} - set(frame.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {sorted(missing)}")
    reference = pd.read_csv(args.reference, dtype=str, keep_default_na=False)

    expanded = expand(frame, reference)
    report = build_report(expanded)
    args.outdir.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(args.outdir / "address_expanded.csv", index=False)
    report.to_csv(args.outdir / "address_expansion_report.csv", index=False)
    LOGGER.info("%s", report.to_string(index=False))


if __name__ == "__main__":
    main()
