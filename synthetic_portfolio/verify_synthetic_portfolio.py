"""Check a generated SYNTHETIC portfolio against SYNTHETIC_DATA_RULESET.md (T1, T4, T5, T7, T8, T9).

T2 (reproducibility), T3 (validator) and T6 (address round trip) run through
their own scripts; see the README. Output is counts only, never cell values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from generate_synthetic_portfolio import (
    apportion,
    load_avoid_values,
    load_district_codes,
)

ASSET_BANDS = {
    "Standard": (0, 0),
    "SMA 0": (1, 30),
    "SMA 1": (31, 60),
    "SMA 2": (61, 90),
    "NPA": (91, None),
}


def _band_ok(asset: str, dpd: int) -> bool:
    low, high = ASSET_BANDS[asset]
    return dpd >= low and (high is None or dpd <= high)


def run_checks(
    frame: pd.DataFrame,
    cfg: dict,
    reference: pd.DataFrame,
    district_codes: dict[str, int],
    avoid: set[str],
) -> list[tuple[str, bool, str]]:
    """Return ``(check id, passed, detail)`` tuples."""
    out: list[tuple[str, bool, str]] = []
    districts = list(cfg["district_population_2011"])
    code_of = {district_codes[d]: d for d in districts}
    mapped = reference.loc[reference["mapping_status"] == "mapped"]
    pin_district = {
        str(p): code_of.get(int(float(c)))
        for p, c in zip(mapped["pincode"], mapped["district_code"], strict=True)
    }
    frame = frame.copy()
    frame["district"] = frame["pincode"].astype(str).map(pin_district)

    expected = apportion(int(cfg["total_accounts"]), cfg["district_population_2011"])
    actual = frame["district"].value_counts().to_dict()
    out.append(
        (
            "T1",
            len(frame) == cfg["total_accounts"] and actual == expected,
            f"rows={len(frame)}; district counts match allocation={actual == expected}",
        )
    )

    sanction = frame["sanction amount"].astype(float)
    outstanding = frame["balance outstanding"].astype(float)
    overdue = frame["overdue amount"].astype(float)
    dpd = frame["days_past_due"].astype(int)
    combos = {
        (r["scheme code"], r["facility type"], r["agri sector"])
        for r in cfg["product_mix_percent"]
    }
    row_combo = list(
        zip(
            frame["scheme code"],
            frame["facility type"],
            frame["agri sector"],
            strict=True,
        )
    )
    branch_districts = frame.groupby("branch ID")["district"].nunique()
    circle_ok = frame["circle"] == frame["district"].map(cfg["circle_of_district"])
    cities_ok = frame["borrowers address"].str.rsplit(",", n=1).str[
        -1
    ].str.strip() == frame["district"].map(cfg["city_of_district"])
    x = {
        "X1": int((outstanding > sanction).sum()),
        "X2": int((overdue > outstanding).sum()),
        "X3": int(
            sum(
                not _band_ok(a, d)
                for a, d in zip(frame["asset class"], dpd, strict=True)
            )
        ),
        "X4": int(((dpd == 0) != (overdue == 0)).sum()),
        "X5": int(
            (
                pd.to_datetime(frame["account open date"])
                > pd.to_datetime(frame["reporting_date"])
            ).sum()
        ),
        "X6": int(sum(c not in combos for c in row_combo)),
        "X7": int(frame["district"].isna().sum() + (~cities_ok).sum()),
        "X8": int((branch_districts > 1).sum() + (~circle_ok).sum()),
        "X9": int(frame["account number"].duplicated().sum()),
    }
    for key, violations in x.items():
        out.append((f"T4/{key}", violations == 0, f"violations={violations}"))

    # T5: per-district share deviations, in percentage points.
    limit_rows = cfg["t5_tolerance_pp"]["small_district_rows"]
    worst = 0.0
    for dimension, global_share in (
        ("asset class", cfg["asset_class_percent"]),
        (
            "sanction amount",
            {str(k): v for k, v in cfg["sanction_amount_weights"].items()},
        ),
    ):
        total_w = float(sum(global_share.values()))
        series = frame[dimension].astype(str)
        for district, group in frame.groupby("district"):
            tol = cfg["t5_tolerance_pp"][
                "small" if len(group) < limit_rows else "large"
            ]
            shares = series[group.index].value_counts(normalize=True) * 100
            for key, weight in global_share.items():
                dev = abs(shares.get(key, 0.0) - 100 * weight / total_w)
                worst = max(worst, dev)
                if dev > tol:
                    out.append(
                        (
                            "T5",
                            False,
                            f"{dimension}/{key}/{district}: {dev:.1f}pp > {tol}",
                        )
                    )
    out.append(
        (
            "T5",
            worst <= cfg["t5_tolerance_pp"]["small"],
            f"largest deviation={worst:.2f}pp",
        )
    )

    out.append(
        (
            "T8",
            frame["pincode"].astype(str).isin(pin_district).all(),
            "all pincodes are 'mapped'",
        )
    )
    collisions = sum(
        v in avoid
        for column in ("account number", "customer ID", "customer name")
        for v in frame[column].astype(str)
    ) + sum(v[4:] in avoid for v in frame["customer ID"].astype(str))
    out.append(
        ("T7", collisions == 0, f"collisions with real sample values={collisions}")
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; exit 1 if any check fails."""
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/private/synthetic/synthetic_portfolio_v1.csv"),
    )
    parser.add_argument(
        "--config", type=Path, default=here / "synthetic_config_v1.json"
    )
    parser.add_argument(
        "--pincode-reference",
        type=Path,
        default=root / "address_expansion" / "pincode_reference_v1.csv",
    )
    parser.add_argument(
        "--project-districts",
        type=Path,
        default=root / "CropLongFormatv1" / "crop_long_v1.csv",
    )
    parser.add_argument("--avoid-from", type=Path)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input, dtype=str)
    reference = pd.read_csv(args.pincode_reference, dtype=str)
    avoid = load_avoid_values(args.avoid_from) if args.avoid_from else set()
    results = run_checks(
        frame, cfg, reference, load_district_codes(args.project_districts), avoid
    )
    for check_id, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {check_id:7s} {detail}")
    return 0 if all(p for _, p, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
