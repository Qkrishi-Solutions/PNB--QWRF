"""Validate a loan-portfolio CSV against the schema in ``loan_portfolio_schema_v1.json``.

Input grain: one row per loan account (fields as in
``financial_data_requirements/DATA_DICTIONARY.md``).

Output: ``validation_report.csv`` with one row per check (check, column, severity,
status, failing row count, first failing row numbers, detail). The report never
contains cell values, so it holds no personal data. Row numbers are data rows
counted from 1, header excluded.

Nothing is imputed or corrected. Exit code: 0 if no error-severity check failed
(warnings allowed), 1 if any error-severity check failed, 2 on usage or read errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("validate_loan_portfolio")

ERROR = "error"
WARNING = "warning"
INFO = "info"
MAX_ROWS_LISTED = 10
DEFAULT_SCHEMA = Path(__file__).with_name("loan_portfolio_schema_v1.json")

REPORT_COLUMNS = [
    "check_id",
    "column",
    "severity",
    "status",
    "failing_rows",
    "total_rows",
    "first_failing_rows",
    "detail",
]


@dataclass
class CheckResult:
    """One line of the validation report."""

    check_id: str
    column: str
    severity: str
    status: str
    failing_rows: int
    total_rows: int
    first_failing_rows: str
    detail: str


def _result(
    check_id: str,
    column: str,
    severity: str,
    failing: pd.Series | None,
    total: int,
    detail: str = "",
    status: str | None = None,
) -> CheckResult:
    """Build a result from a boolean mask of failing rows (True = failing)."""
    if failing is None:
        return CheckResult(check_id, column, severity, status or "SKIPPED", 0, total, "", detail)
    count = int(failing.sum())
    rows = [str(i + 1) for i in range(len(failing)) if bool(failing.iloc[i])][:MAX_ROWS_LISTED]
    if status is None:
        status = "PASS" if count == 0 else ("FAIL" if severity == ERROR else "WARN")
    return CheckResult(check_id, column, severity, status, count, total, ";".join(rows), detail)


def load_schema(path: Path) -> dict:
    """Read the schema JSON and check it has the keys the validator needs."""
    schema = json.loads(path.read_text(encoding="utf-8"))
    for key in ("columns", "asset_class_dpd_bands", "date_format", "account_key"):
        if key not in schema:
            raise ValueError(f"Schema {path} is missing '{key}'")
    return schema


def _is_missing(series: pd.Series) -> pd.Series:
    """Return True where a cell is empty or whitespace (strings are expected)."""
    return series.isna() | series.astype(str).str.strip().eq("")


def _column_checks(
    name: str, series: pd.Series, spec: dict, schema: dict, total: int
) -> tuple[list[CheckResult], pd.Series | None]:
    """Run missing/type/range/pattern/allowed-value checks on one column."""
    results: list[CheckResult] = []
    missing = _is_missing(series)
    required = bool(spec.get("required"))
    results.append(
        _result(
            "missing_values",
            name,
            ERROR if required else INFO,
            missing,
            total,
            "required column"
            if required
            else "optional column; missing values are reported, not imputed",
        )
    )
    present = ~missing
    text = series.astype(str).str.strip()
    parsed: pd.Series | None = None
    kind = spec["type"]
    if kind == "date":
        parsed = pd.to_datetime(text.where(present), format=schema["date_format"], errors="coerce")
        results.append(
            _result(
                "valid_date",
                name,
                ERROR,
                present & parsed.isna(),
                total,
                f"format {schema['date_format']}",
            )
        )
    elif kind in {"number", "integer"}:
        parsed = pd.to_numeric(text.where(present), errors="coerce")
        results.append(_result("valid_number", name, ERROR, present & parsed.isna(), total))
        if kind == "integer":
            results.append(
                _result(
                    "whole_number", name, ERROR, parsed.notna() & parsed.ne(parsed.round()), total
                )
            )
        if "min" in spec:
            results.append(
                _result(
                    "minimum",
                    name,
                    ERROR,
                    parsed.notna() & parsed.lt(spec["min"]),
                    total,
                    f">= {spec['min']}",
                )
            )
        if "max" in spec:
            results.append(
                _result(
                    "maximum",
                    name,
                    ERROR,
                    parsed.notna() & parsed.gt(spec["max"]),
                    total,
                    f"<= {spec['max']}",
                )
            )
    if "pattern" in spec:
        bad = present & ~text.str.fullmatch(spec["pattern"])
        severity = spec.get("pattern_severity", ERROR)
        results.append(
            _result("format_pattern", name, severity, bad, total, f"pattern {spec['pattern']}")
        )
    if "allowed" in spec:
        bad = present & ~text.isin(spec["allowed"])
        results.append(
            _result(
                "allowed_values", name, ERROR, bad, total, f"allowed: {', '.join(spec['allowed'])}"
            )
        )
    return results, parsed


def _skipped(
    check_id: str, column: str, severity: str, total: int, needs: list[str]
) -> CheckResult:
    return _result(
        check_id, column, severity, None, total, f"skipped: needs {', '.join(needs)}", "SKIPPED"
    )


def _cross_checks(
    frame: pd.DataFrame, schema: dict, parsed: dict[str, pd.Series], total: int
) -> list[CheckResult]:
    """Relationship checks between columns."""
    results: list[CheckResult] = []
    p = parsed

    def need(check_id: str, column: str, severity: str, cols: list[str]) -> bool:
        absent = [c for c in cols if c not in p]
        if absent:
            results.append(_skipped(check_id, column, severity, total, absent))
        return not absent

    if (
        need("asset_class_vs_dpd", "asset class", ERROR, ["days_past_due"])
        and "asset class" in frame
    ):
        dpd = p["days_past_due"]
        bad = pd.Series(False, index=frame.index)
        for label, (low, high) in schema["asset_class_dpd_bands"].items():
            in_class = frame["asset class"].astype(str).str.strip().eq(label) & dpd.notna()
            outside = dpd.lt(low) | (dpd.gt(high) if high is not None else False)
            bad |= in_class & outside
        results.append(
            _result(
                "asset_class_vs_dpd",
                "asset class",
                ERROR,
                bad,
                total,
                "bands "
                + json.dumps(schema["asset_class_dpd_bands"])
                + " (to be confirmed by PNB)",
            )
        )
    if need(
        "overdue_le_outstanding", "overdue amount", ERROR, ["overdue amount", "balance outstanding"]
    ):
        both = p["overdue amount"].notna() & p["balance outstanding"].notna()
        results.append(
            _result(
                "overdue_le_outstanding",
                "overdue amount",
                ERROR,
                both & p["overdue amount"].gt(p["balance outstanding"]),
                total,
            )
        )
    if need("overdue_iff_dpd", "overdue amount", ERROR, ["overdue amount", "days_past_due"]):
        both = p["overdue amount"].notna() & p["days_past_due"].notna()
        mismatch = p["overdue amount"].gt(0) != p["days_past_due"].gt(0)
        results.append(
            _result(
                "overdue_iff_dpd",
                "overdue amount",
                ERROR,
                both & mismatch,
                total,
                "overdue > 0 if and only if days past due > 0",
            )
        )
    if need(
        "outstanding_le_sanction",
        "balance outstanding",
        WARNING,
        ["balance outstanding", "sanction amount"],
    ):
        both = p["balance outstanding"].notna() & p["sanction amount"].notna()
        results.append(
            _result(
                "outstanding_le_sanction",
                "balance outstanding",
                WARNING,
                both & p["balance outstanding"].gt(p["sanction amount"]),
                total,
                "may legitimately fail for limits; PNB to confirm",
            )
        )
    rdate = schema.get("reporting_date_column", "reporting_date")
    if need(
        "open_date_not_after_reporting", "account open date", ERROR, ["account open date", rdate]
    ):
        both = p["account open date"].notna() & p[rdate].notna()
        results.append(
            _result(
                "open_date_not_after_reporting",
                "account open date",
                ERROR,
                both & p["account open date"].gt(p[rdate]),
                total,
            )
        )
    return results


def _key_checks(frame: pd.DataFrame, schema: dict, total: int) -> list[CheckResult]:
    """Uniqueness and one-to-one consistency checks."""
    results: list[CheckResult] = []
    key = schema["account_key"]
    if key in frame:
        keyed = frame[key].astype(str).str.strip()
        dup = keyed.duplicated(keep=False) & ~_is_missing(frame[key])
        results.append(
            _result(
                "account_key_unique", key, ERROR, dup, total, "every account number must be unique"
            )
        )
    else:
        results.append(_skipped("account_key_unique", key, ERROR, total, [key]))
    results.append(_result("duplicate_rows", "all", WARNING, frame.duplicated(keep=False), total))
    for child, parent, label in [
        ("branch ID", "branch name", "branch_id_one_name"),
        ("circle", "zone", "circle_one_zone"),
    ]:
        if child in frame and parent in frame:
            clash = frame.groupby(child)[parent].transform("nunique").gt(1)
            results.append(
                _result(
                    label, child, WARNING, clash, total, f"each {child} should map to one {parent}"
                )
            )
        else:
            results.append(_skipped(label, child, WARNING, total, [child, parent]))
    rdate = schema.get("reporting_date_column", "reporting_date")
    if rdate in frame:
        n = frame[rdate].nunique(dropna=True)
        results.append(
            _result(
                "reporting_dates",
                rdate,
                INFO,
                None,
                total,
                f"{n} distinct reporting date(s)",
                "INFO",
            )
        )
    return results


def _pincode_checks(
    frame: pd.DataFrame, reference: pd.DataFrame, schema: dict, total: int
) -> list[CheckResult]:
    """Check pincodes against the India Post reference (see address_expansion/)."""
    if "pincode" not in frame:
        return [_skipped("pincode_in_reference", "pincode", ERROR, total, ["pincode"])]
    pin = frame["pincode"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    valid = pin.str.fullmatch(r"[1-9]\d{5}")
    ref = reference.set_index("pincode")["mapping_status"].to_dict()
    status = pin.map(ref)
    return [
        _result(
            "pincode_in_reference",
            "pincode",
            ERROR,
            valid & status.isna(),
            total,
            "valid-format pincode not found in the India Post reference",
        ),
        _result(
            "pincode_in_project_state",
            "pincode",
            ERROR,
            status.eq("outside_project_scope"),
            total,
            f"project scope is {schema.get('project_state', 'the project state')} only",
        ),
        _result(
            "pincode_district_ambiguous",
            "pincode",
            WARNING,
            status.isin(["ambiguous_district", "ambiguous_state"]),
            total,
            "pincode spans more than one district or state; not mapped uniquely",
        ),
    ]


def validate(
    frame: pd.DataFrame, schema: dict, pincode_reference: pd.DataFrame | None = None
) -> list[CheckResult]:
    """Run all checks on ``frame`` and return the results. ``frame`` is never modified."""
    total = len(frame)
    results: list[CheckResult] = []
    expected = [c["name"] for c in schema["columns"]]
    missing_cols = [c for c in expected if c not in frame.columns]
    extra_cols = [c for c in frame.columns if c not in expected]
    results.append(
        CheckResult(
            "required_columns",
            "all",
            ERROR,
            "FAIL" if missing_cols else "PASS",
            len(missing_cols),
            total,
            "",
            f"missing: {', '.join(missing_cols)}"
            if missing_cols
            else f"{len(expected)} expected columns present",
        )
    )
    results.append(
        CheckResult(
            "unexpected_columns",
            "all",
            WARNING,
            "WARN" if extra_cols else "PASS",
            len(extra_cols),
            total,
            "",
            f"unexpected: {', '.join(extra_cols)}" if extra_cols else "none",
        )
    )
    parsed: dict[str, pd.Series] = {}
    for spec in schema["columns"]:
        name = spec["name"]
        if name not in frame.columns:
            continue
        col_results, values = _column_checks(name, frame[name], spec, schema, total)
        results.extend(col_results)
        if values is not None:
            parsed[name] = values
    results.extend(_cross_checks(frame, schema, parsed, total))
    results.extend(_key_checks(frame, schema, total))
    if pincode_reference is not None:
        results.extend(_pincode_checks(frame, pincode_reference, schema, total))
    return results


def to_report(results: list[CheckResult]) -> pd.DataFrame:
    """Convert results to the report table."""
    return pd.DataFrame([asdict(r) for r in results], columns=REPORT_COLUMNS)


def exit_code(results: list[CheckResult]) -> int:
    """Return 1 if any error-severity check failed, else 0."""
    return int(any(r.status == "FAIL" for r in results))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="loan portfolio CSV (private)")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--pincode-reference",
        type=Path,
        default=None,
        help="address_expansion/pincode_reference_v1.csv (enables pincode checks)",
    )
    parser.add_argument("--outdir", type=Path, default=Path("data/private/validation"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        schema = load_schema(args.schema)
        frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        reference = None
        if args.pincode_reference:
            reference = pd.read_csv(args.pincode_reference, dtype=str, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        LOGGER.error("Cannot read inputs: %s", error)
        return 2

    results = validate(frame, schema, reference)
    report = to_report(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.outdir / "validation_report.csv", index=False)
    counts = report["status"].value_counts().to_dict()
    LOGGER.info("rows: %d | checks: %d | %s", len(frame), len(report), counts)
    for row in report.loc[report["status"].isin(["FAIL", "WARN"])].itertuples():
        LOGGER.info(
            "%s %s [%s] %s: %d row(s) e.g. %s",
            row.status,
            row.check_id,
            row.column,
            row.severity,
            row.failing_rows,
            row.first_failing_rows,
        )
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
