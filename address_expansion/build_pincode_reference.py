"""Build a pincode reference table from the India Post pincode directory.

Input grain: one row per post office (many rows per pincode).
Output grain: one row per pincode, with state, source district(s) and, for
Andhra Pradesh pincodes, the project ``district_code`` where the mapping is
unique.

Rules (no guessing):
- A pincode whose offices span more than one state is ``ambiguous_state``.
- A pincode whose offices span more than one district is ``ambiguous_district``;
  all candidate project district codes are kept and ``district_code`` is blank.
- Only the 13 project districts receive a ``district_code``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("build_pincode_reference")

PROJECT_STATE_NAME = "ANDHRA PRADESH"
PROJECT_STATE_CODE = 1  # state_code used in CropLongFormatv1/crop_long_v1.csv

# India Post spelling (normalised) -> project spelling (normalised).
# Only differences observed between the two sources are listed.
DISTRICT_ALIASES = {
    "ANANTAPUR": "ANANTAPURAMU",
    "SPSR NELLORE": "SPR NELLORE",
    "VISAKHAPATANAM": "VISAKHAPATNAM",
    "Y S R": "YSR",
}

PINCODE_PATTERN = r"[1-9]\d{5}"
LIST_SEPARATOR = "|"


def normalise_name(value: object) -> str:
    """Upper-case a place name and collapse punctuation and whitespace."""
    cleaned = re.sub(r"[^A-Z0-9]+", " ", str(value).upper())
    return re.sub(r"\s+", " ", cleaned).strip()


def load_project_districts(path: Path) -> dict[str, int]:
    """Return ``{normalised district name: district_code}`` from a project CSV."""
    frame = pd.read_csv(path, usecols=["district_code", "district_name"]).drop_duplicates()
    if frame["district_code"].duplicated().any() or frame["district_name"].duplicated().any():
        raise ValueError(f"{path}: district code/name pairs are not one-to-one")
    return {
        normalise_name(n): int(c)
        for c, n in zip(frame["district_code"], frame["district_name"], strict=True)
    }


def _project_code(source_district: str, project: dict[str, int]) -> int | None:
    """Map one normalised India Post district to a project district code."""
    return project.get(DISTRICT_ALIASES.get(source_district, source_district))


def build_reference(
    raw: pd.DataFrame, project: dict[str, int]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse the post-office directory to one row per pincode.

    Returns the reference table and a dictionary of build counts.
    """
    frame = raw.loc[:, ["Pincode", "District", "StateName"]].copy()
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    valid = frame["Pincode"].str.fullmatch(PINCODE_PATTERN)
    counts = {"raw_rows": len(frame), "invalid_pincode_rows": int((~valid).sum())}
    frame = frame.loc[valid].copy()
    frame["district_norm"] = frame["District"].map(normalise_name)
    frame["state_norm"] = frame["StateName"].map(normalise_name)

    grouped = frame.groupby("Pincode", sort=True)
    reference = grouped.agg(
        state_name=("state_norm", lambda s: LIST_SEPARATOR.join(sorted(set(s)))),
        source_districts=("district_norm", lambda s: LIST_SEPARATOR.join(sorted(set(s)))),
    ).reset_index()
    reference = reference.rename(columns={"Pincode": "pincode"})
    reference["n_states"] = reference["state_name"].str.count(r"\|") + 1
    reference["n_source_districts"] = reference["source_districts"].str.count(r"\|") + 1

    def classify(row: pd.Series) -> pd.Series:
        if row["n_states"] > 1:
            return pd.Series(["", "", "ambiguous_state"])
        if row["state_name"] != PROJECT_STATE_NAME:
            return pd.Series(["", "", "outside_project_scope"])
        codes = [_project_code(d, project) for d in row["source_districts"].split(LIST_SEPARATOR)]
        if any(code is None for code in codes):
            return pd.Series(["", "", "ap_district_not_in_project"])
        joined = LIST_SEPARATOR.join(str(code) for code in sorted(codes))
        if len(codes) == 1:
            return pd.Series([str(codes[0]), joined, "mapped"])
        return pd.Series(["", joined, "ambiguous_district"])

    reference[["district_code", "project_district_codes", "mapping_status"]] = reference.apply(
        classify, axis=1
    )
    reference["state_code"] = (
        reference["state_name"]
        .eq(PROJECT_STATE_NAME)
        .map({True: str(PROJECT_STATE_CODE), False: ""})
    )
    columns = [
        "pincode",
        "state_name",
        "state_code",
        "n_states",
        "source_districts",
        "n_source_districts",
        "project_district_codes",
        "district_code",
        "mapping_status",
    ]
    reference = reference.loc[:, columns]

    counts["unique_pincodes"] = len(reference)
    for status, number in reference["mapping_status"].value_counts().sort_index().items():
        counts[f"pincodes_{status}"] = int(number)
    covered = {
        int(c)
        for value in reference["project_district_codes"]
        for c in value.split(LIST_SEPARATOR)
        if value
    }
    counts["project_districts_covered"] = len(covered & set(project.values()))
    counts["project_districts_expected"] = len(project)
    return reference, counts


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="India Post pincode directory CSV")
    parser.add_argument(
        "--districts",
        type=Path,
        required=True,
        help="project CSV with district_code and district_name columns",
    )
    parser.add_argument("--outdir", type=Path, default=Path("address_expansion"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project = load_project_districts(args.districts)
    raw = pd.read_csv(args.raw, dtype=str)
    reference, counts = build_reference(raw, project)
    if counts["project_districts_covered"] != counts["project_districts_expected"]:
        raise SystemExit(f"Project districts not all found in directory: {counts}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    reference.to_csv(args.outdir / "pincode_reference_v1.csv", index=False)
    report = {"raw_file": args.raw.name, "raw_sha256": sha256_of(args.raw), **counts}
    pd.DataFrame({"item": list(report), "value": list(report.values())}).to_csv(
        args.outdir / "pincode_reference_build_report_v1.csv", index=False
    )
    for key, value in report.items():
        LOGGER.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
