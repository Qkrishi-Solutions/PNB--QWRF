"""Generate the SYNTHETIC loan portfolio described in SYNTHETIC_DATA_RULESET.md.

The output is a development test fixture. It is NOT PNB data.

Only the number of accounts varies by district; every other distribution is the
same in every district. Product mix, asset class and sanction amount are
apportioned exactly within each district (largest remainder) and then shuffled,
so the per-district shares match the config as closely as integers allow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("generate_synthetic_portfolio")

OUTPUT_COLUMNS = [
    "reporting_date",
    "zone",
    "circle",
    "branch ID",
    "branch name",
    "customer ID",
    "customer name",
    "borrowers address",
    "pincode",
    "account number",
    "account open date",
    "facility type",
    "scheme code",
    "agri sector",
    "sanction amount",
    "rate of interest",
    "balance outstanding",
    "overdue amount",
    "days_past_due",
    "asset class",
]
ASSET_CLASSES = ["Standard", "SMA 0", "SMA 1", "SMA 2", "NPA"]
FIRST_BRANCH_NUMBER = 101
CUSTOMER_ID_RANGE = (10001, 99999)
ACCOUNT_NUMBER_RANGE = (10_000_000_000, 99_999_999_999)
OUTPUT_NAME = "synthetic_portfolio_v1.csv"
REPORT_NAME = "synthetic_generation_report_v1.csv"


def apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``total`` into integers proportional to ``weights`` (largest remainder)."""
    weight_sum = float(sum(weights.values()))
    exact = {k: total * v / weight_sum for k, v in weights.items()}
    counts = {k: int(np.floor(v)) for k, v in exact.items()}
    leftover = total - sum(counts.values())
    by_remainder = sorted(exact, key=lambda k: (exact[k] - counts[k], k), reverse=True)
    for key in by_remainder[:leftover]:
        counts[key] += 1
    return counts


def expand_counts(counts: dict[str, int], rng: np.random.Generator) -> list[str]:
    """Return a shuffled list holding each key ``counts[key]`` times."""
    items = [k for k, n in counts.items() for _ in range(n)]
    order = rng.permutation(len(items))
    return [items[i] for i in order]


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_ints(
    rng: np.random.Generator, low: int, high: int, n: int, avoid: set[int]
) -> list[int]:
    """Draw ``n`` distinct integers in ``[low, high]`` that are not in ``avoid``."""
    chosen: list[int] = []
    seen: set[int] = set(avoid)
    while len(chosen) < n:
        for value in rng.integers(low, high + 1, size=(n - len(chosen)) * 2 + 8):
            value = int(value)
            if value not in seen:
                seen.add(value)
                chosen.append(value)
                if len(chosen) == n:
                    break
    return chosen


def _branch_names(city: str, n: int) -> list[str]:
    """Placeholder branch names for one district."""
    return [f"{city} Main"] + [f"{city} Branch {i}" for i in range(2, n + 1)]


def _build_customers(
    rng: np.random.Generator,
    cfg: dict,
    district: str,
    n_accounts: int,
    pincodes: list[str],
    branch_ids: list[str],
    branch_names: list[str],
    customer_numbers: list[int],
    avoid_names: set[str],
) -> tuple[list[dict], list[int]]:
    """Create the customers of one district and the customer index of each account."""
    two_share = float(cfg["customers_with_two_accounts_percent"]) / 100
    n_double = round(n_accounts * two_share)
    n_customers = n_accounts - n_double
    lists = cfg["place_name_lists"]
    city = cfg["city_of_district"][district]
    customers: list[dict] = []
    used_names: set[str] = set()
    for i in range(n_customers):
        for _ in range(100):
            name = (
                f"{lists['given_names'][rng.integers(len(lists['given_names']))]} "
                f"{lists['surnames'][rng.integers(len(lists['surnames']))]}"
            )
            if name not in avoid_names:
                break
        used_names.add(name)
        branch = int(rng.integers(len(branch_ids)))
        address = ", ".join(
            [
                f"{lists['house'][rng.integers(len(lists['house']))]} {int(rng.integers(1, 400))}",
                lists["area"][rng.integers(len(lists["area"]))],
                lists["locality"][rng.integers(len(lists["locality"]))],
                city,
            ]
        )
        customers.append(
            {
                "customer ID": f"CUST{customer_numbers[i]:05d}",
                "customer name": name,
                "borrowers address": address,
                "pincode": pincodes[int(rng.integers(len(pincodes)))],
                "branch ID": branch_ids[branch],
                "branch name": branch_names[branch],
            }
        )
    second = (
        rng.choice(n_customers, size=n_double, replace=False).tolist()
        if n_double
        else []
    )
    account_customer = list(range(n_customers)) + second
    return customers, account_customer


def generate(
    cfg: dict,
    reference: pd.DataFrame,
    district_codes: dict[str, int],
    avoid_values: set[str] | None = None,
) -> pd.DataFrame:
    """Build the synthetic portfolio. Deterministic for a given config."""
    avoid_values = avoid_values or set()
    rng = np.random.default_rng(int(cfg["seed"]))
    total = int(cfg["total_accounts"])
    districts = list(cfg["district_population_2011"])
    per_district = apportion(total, cfg["district_population_2011"])

    mapped = reference.loc[reference["mapping_status"] == "mapped"].copy()
    mapped["district_code"] = mapped["district_code"].astype(float).astype(int)
    pincodes_of = {
        d: sorted(
            mapped.loc[mapped["district_code"] == district_codes[d], "pincode"].astype(
                str
            )
        )
        for d in districts
    }
    for district, pins in pincodes_of.items():
        if not pins:
            raise ValueError(f"no mapped pincodes for {district}")

    avoid_int = {int(v) for v in avoid_values if v.isdigit()}
    avoid_names = {v for v in avoid_values if not v.isdigit()}
    n_customers_total = sum(
        n - round(n * float(cfg["customers_with_two_accounts_percent"]) / 100)
        for n in per_district.values()
    )
    customer_pool = _unique_ints(
        rng,
        CUSTOMER_ID_RANGE[0],
        CUSTOMER_ID_RANGE[1],
        n_customers_total,
        {v for v in avoid_int if CUSTOMER_ID_RANGE[0] <= v <= CUSTOMER_ID_RANGE[1]},
    )
    account_pool = _unique_ints(
        rng, ACCOUNT_NUMBER_RANGE[0], ACCOUNT_NUMBER_RANGE[1], total, avoid_int
    )

    products = {i: row for i, row in enumerate(cfg["product_mix_percent"])}
    product_weights = {str(i): float(r["weight"]) for i, r in products.items()}
    sanction_weights = {k: float(v) for k, v in cfg["sanction_amount_weights"].items()}
    asset_weights = {k: float(v) for k, v in cfg["asset_class_percent"].items()}
    open_low = pd.Timestamp(cfg["open_date_range"][0])
    open_high = pd.Timestamp(cfg["open_date_range"][1])
    span_days = (open_high - open_low).days

    rows: list[dict] = []
    branch_counter = FIRST_BRANCH_NUMBER
    customer_cursor = 0
    account_cursor = 0
    for district in districts:
        n = per_district[district]
        city = cfg["city_of_district"][district]
        n_branches = max(1, round(n / int(cfg["accounts_per_branch"])))
        branch_ids = [f"BR{branch_counter + i}" for i in range(n_branches)]
        branch_counter += n_branches
        branch_names = _branch_names(city, n_branches)

        n_double = round(n * float(cfg["customers_with_two_accounts_percent"]) / 100)
        n_cust = n - n_double
        customers, account_customer = _build_customers(
            rng,
            cfg,
            district,
            n,
            pincodes_of[district],
            branch_ids,
            branch_names,
            customer_pool[customer_cursor : customer_cursor + n_cust],
            avoid_names,
        )
        customer_cursor += n_cust

        product_draw = expand_counts(apportion(n, product_weights), rng)
        sanction_draw = expand_counts(apportion(n, sanction_weights), rng)
        asset_draw = expand_counts(apportion(n, asset_weights), rng)

        for i in range(n):
            customer = customers[account_customer[i]]
            product = products[int(product_draw[i])]
            asset = asset_draw[i]
            sanction = int(sanction_draw[i])
            rate = round(float(rng.uniform(*cfg["interest_rate_range"])), 2)
            ratio = float(rng.uniform(*cfg["outstanding_to_sanction_range"]))
            outstanding = round(sanction * ratio)
            dpd = int(
                rng.integers(cfg["dpd_range"][asset][0], cfg["dpd_range"][asset][1] + 1)
            )
            low, high = cfg["overdue_to_outstanding_range"][asset]
            overdue = round(outstanding * float(rng.uniform(low, high))) if dpd else 0
            opened = open_low + pd.Timedelta(days=int(rng.integers(0, span_days + 1)))
            rows.append(
                {
                    "reporting_date": cfg["reporting_date"],
                    "zone": cfg["zone"],
                    "circle": cfg["circle_of_district"][district],
                    "branch ID": customer["branch ID"],
                    "branch name": customer["branch name"],
                    "customer ID": customer["customer ID"],
                    "customer name": customer["customer name"],
                    "borrowers address": customer["borrowers address"],
                    "pincode": customer["pincode"],
                    "account number": f"{account_pool[account_cursor]:011d}",
                    "account open date": opened.strftime("%Y-%m-%d"),
                    "facility type": product["facility type"],
                    "scheme code": product["scheme code"],
                    "agri sector": product["agri sector"],
                    "sanction amount": sanction,
                    "rate of interest": rate,
                    "balance outstanding": outstanding,
                    "overdue amount": overdue,
                    "days_past_due": dpd,
                    "asset class": asset,
                    "_district": district,
                }
            )
            account_cursor += 1

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["branch ID", "account number"], kind="stable"
    ).reset_index(drop=True)
    return frame


def build_report(
    frame: pd.DataFrame, cfg_hash: str, seed: int, output_hash: str
) -> pd.DataFrame:
    """Aggregate counts only: no customer-level cell values."""
    records: list[tuple[str, str, str]] = [
        ("meta", "label", "SYNTHETIC - not PNB data"),
        ("meta", "seed", str(seed)),
        ("meta", "rows", str(len(frame))),
        ("meta", "config_sha256", cfg_hash),
        ("meta", "output_sha256", output_hash),
    ]
    for key, value in frame["_district"].value_counts().sort_index().items():
        records.append(("accounts_by_district", str(key), str(value)))
    for key, value in frame["asset class"].value_counts().items():
        records.append(("accounts_by_asset_class", str(key), str(value)))
    combo = (
        frame["scheme code"]
        + " | "
        + frame["facility type"]
        + " | "
        + frame["agri sector"]
    )
    for key, value in combo.value_counts().sort_index().items():
        records.append(("accounts_by_product", str(key), str(value)))
    records.append(("meta", "branches", str(frame["branch ID"].nunique())))
    records.append(("meta", "customers", str(frame["customer ID"].nunique())))
    return pd.DataFrame(records, columns=["section", "key", "value"])


def load_district_codes(path: Path) -> dict[str, int]:
    """Return ``{DISTRICT NAME: district_code}`` from the project crop CSV."""
    frame = pd.read_csv(
        path, usecols=["district_code", "district_name"]
    ).drop_duplicates()
    return {
        re.sub(r"\s+", " ", str(n).upper()).strip(): int(c)
        for c, n in zip(frame["district_code"], frame["district_name"], strict=True)
    }


def load_avoid_values(path: Path) -> set[str]:
    """Collect identifier/name values that generated rows must not reuse."""
    frame = pd.read_csv(path, dtype=str)
    values: set[str] = set()
    for column in ("account number", "customer ID", "customer name"):
        if column in frame.columns:
            values.update(frame[column].dropna().astype(str))
    values.update(v[4:] for v in list(values) if v.startswith("CUST"))
    return values


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--avoid-from", type=Path, help="private CSV whose ids/names are not reused"
    )
    parser.add_argument("--outdir", type=Path, default=Path("data/private/synthetic"))
    parser.add_argument("--report-dir", type=Path, default=here)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    reference = pd.read_csv(args.pincode_reference, dtype=str)
    codes = load_district_codes(args.project_districts)
    avoid = load_avoid_values(args.avoid_from) if args.avoid_from else set()
    frame = generate(cfg, reference, codes, avoid)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / OUTPUT_NAME
    frame[OUTPUT_COLUMNS].to_csv(out_path, index=False)
    report = build_report(
        frame, sha256_of(args.config), int(cfg["seed"]), sha256_of(out_path)
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_dir / REPORT_NAME, index=False)
    LOGGER.info("SYNTHETIC portfolio: %d rows -> %s", len(frame), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
