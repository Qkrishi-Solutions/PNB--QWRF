"""Generate the SYNTHETIC train/test dataset for asset-class prediction (v2).

The output is a development fixture. It is NOT PNB data.

v2 differs from v1 on purpose: the asset class is derived from a documented,
config-defined risk score so that a model has something to learn. Any model
trained on this data recovers those rules; it says nothing about real credit
risk. See SYNTHETIC_TRAINING_RULESET_V2.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "synthetic_portfolio"))

from generate_synthetic_portfolio import OUTPUT_COLUMNS as PNB_LAYOUT_COLUMNS  # noqa: E402
from generate_synthetic_portfolio import apportion, generate  # noqa: E402

LOGGER = logging.getLogger("generate_training_dataset")

ASSET_ORDER = ["Standard", "SMA 0", "SMA 1", "SMA 2", "NPA"]
TRAIN_NAME = "synthetic_training_v2_train.csv"
TEST_NAME = "synthetic_training_v2_test.csv"
PNB_LAYOUT_TRAIN_NAME = "Financial_data_PNB_SYNTHETIC_train.csv"
PNB_LAYOUT_TEST_NAME = "Financial_data_PNB_SYNTHETIC_test.csv"
DICTIONARY_NAME = "synthetic_training_v2_data_dictionary.csv"
REPORT_NAME = "synthetic_training_report_v2.csv"


def file_names(version: str) -> dict[str, str]:
    """Output file names for a dataset version (``v2``, ``v3``, ...)."""
    return {
        "train": f"synthetic_training_{version}_train.csv",
        "test": f"synthetic_training_{version}_test.csv",
        "dictionary": f"synthetic_training_{version}_data_dictionary.csv",
        "report": f"synthetic_training_report_{version}.csv",
    }

INSURANCE_COLUMNS = [
    "crop_insurance_available",
    "insurance_policy_id",
    "insurance_scheme",
    "insurance_amount",
    "insurance_premium",
    "insurance_start_date",
    "insurance_end_date",
    "insurance_provider",
    "insurance_coverage_status",
    "insurance_claim_status",
    "insurance_claim_amount",
    "insurance_payout_amount",
    "insurance_payout_date",
]


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wx_name(column: str) -> str:
    """Output column name of a weather anomaly z-score."""
    return f"wx_{column}_z"


# --------------------------------------------------------------------------- inputs
def load_weather(path: Path, columns: list[str], years: list[int]) -> pd.DataFrame:
    """District-year forward-chaining weather anomaly z-scores (NaN kept as NaN)."""
    frame = pd.read_csv(path)
    frame = frame[
        frame["is_target_year"]
        & frame["weather_column"].isin(columns)
        & frame["year"].isin(years)
    ]
    wide = frame.pivot_table(
        index=["district_code", "year"],
        columns="weather_column",
        values="anomaly_z",
        aggfunc="first",
        dropna=False,
    )
    wide = wide.reindex(columns=columns).reset_index()
    wide.columns.name = None
    return wide.rename(columns={c: wx_name(c) for c in columns})


def load_district_codes(path: Path) -> dict[str, int]:
    """``{DISTRICT NAME: district_code}`` from the weather file."""
    frame = pd.read_csv(path, usecols=["district_code", "district_name"])
    frame = frame.drop_duplicates()
    return {str(n).upper(): int(c) for c, n in zip(frame.district_code, frame.district_name, strict=True)}


def load_crop_table(path: Path, years: list[int]) -> pd.DataFrame:
    """Crop rows usable for assignment, with the forward-chaining yield z-score."""
    use = [
        "district_code", "year", "crop", "area_1000_ha", "is_aggregate",
        "is_group_category", "is_seasonal_component", "anomaly_status",
        "yield_anomaly_zscore",
    ]
    frame = pd.read_csv(path, usecols=use)
    keep = (
        frame["year"].isin(years)
        & ~frame["is_aggregate"]
        & ~frame["is_group_category"]
        & ~frame["is_seasonal_component"]
        & (frame["area_1000_ha"] > 0)
    )
    frame = frame.loc[keep].copy()
    calc = frame["anomaly_status"] == "calculated"
    frame["yield_anomaly_zscore"] = frame["yield_anomaly_zscore"].where(calc)
    return frame


# ---------------------------------------------------------------------- assignment
def assign_snapshots(
    frame: pd.DataFrame, snapshots: list[dict], rng: np.random.Generator
) -> pd.Series:
    """Assign each customer (all of their accounts) to one snapshot year.

    Stratified by district so every district has the same year shares.
    """
    shares = {s["year"]: float(s["accounts"]) for s in snapshots}
    year_of: dict[str, int] = {}
    for _, part in frame.groupby("_district", sort=True):
        target = apportion(len(part), {str(y): w for y, w in shares.items()})
        remaining = {int(y): n for y, n in target.items()}
        sizes = part.groupby("customer ID").size()
        order = rng.permutation(len(sizes))
        for idx in order:
            cust, size = sizes.index[idx], int(sizes.iloc[idx])
            year = max(remaining, key=lambda y: (remaining[y], -y))
            year_of[cust] = year
            remaining[year] -= size
    return frame["customer ID"].map(year_of).astype(int)


# --------------------------------------------------------------------- insurance
def _status_draw(rng: np.random.Generator, percent: dict[str, float]) -> str:
    names = list(percent)
    probs = np.array([percent[n] for n in names], dtype=float)
    return str(rng.choice(names, p=probs / probs.sum()))


def make_insurance(
    row: pd.Series, stress: float, yield_z: float, cfg: dict, rng: np.random.Generator
) -> dict:
    """Insurance fields for one account. Dates, status and claim are rule-consistent."""
    ins = cfg["insurance"]
    blank = dict.fromkeys(INSURANCE_COLUMNS)
    sector = row["agri sector"]
    if rng.random() >= ins["availability_probability"][sector]:
        blank.update(crop_insurance_available="N", insurance_coverage_status="Not Insured",
                     insurance_claim_status="No Claim")
        return blank

    rep = pd.Timestamp(row["reporting_date"])
    opened = pd.Timestamp(row["account open date"])
    term = int(ins["term_days"])
    status = _status_draw(rng, ins["status_percent"])
    if status == "Active":
        start = rep - pd.Timedelta(days=int(rng.integers(1, term + 1)))
        start = max(start, opened)
        end = start + pd.Timedelta(days=term)
    elif status == "Expired":
        room = max(2, min(300, (rep - opened).days - 30))
        end = rep - pd.Timedelta(days=int(rng.integers(1, room)))
        start = max(opened, end - pd.Timedelta(days=term))
    else:
        start = rep + pd.Timedelta(days=int(rng.integers(1, int(ins["pending_start_within_days"]) + 1)))
        end = start + pd.Timedelta(days=term)

    lo, hi = ins["insured_to_sanction_range"]
    amount = round(float(row["sanction amount"]) * float(rng.uniform(lo, hi)))
    plo, phi = ins["premium_to_insured_range"]
    premium = round(amount * float(rng.uniform(plo, phi)))
    out = dict(
        crop_insurance_available="Y",
        insurance_policy_id=f"SYN-POL-{int(rng.integers(10**7, 10**8))}",
        insurance_scheme=ins["scheme_by_sector"].get(sector, ins["default_scheme"]),
        insurance_amount=amount,
        insurance_premium=premium,
        insurance_start_date=start.strftime("%Y-%m-%d"),
        insurance_end_date=end.strftime("%Y-%m-%d"),
        insurance_provider=str(rng.choice(ins["providers"])),
        insurance_coverage_status=status,
        insurance_claim_status="No Claim",
        insurance_claim_amount=None,
        insurance_payout_amount=None,
        insurance_payout_date=None,
    )
    covered = status in ("Active", "Expired")
    sens = cfg["sector_weather_sensitivity"][sector]
    hit = sens * stress > ins["claim_stress_threshold"] or (
        not np.isnan(yield_z) and yield_z <= -1.0
    )
    if covered and hit and rng.random() < ins["claim_probability_given_stress"]:
        claim_status = _status_draw(rng, ins["claim_status_percent"])
        clo, chi = ins["claim_to_insured_range"]
        claim = round(amount * float(rng.uniform(clo, chi)))
        out.update(insurance_claim_status=claim_status, insurance_claim_amount=claim)
        if claim_status == "Paid":
            window = (rep - start).days - int(ins["min_days_start_to_payout"])
            if window > 0:
                olo, ohi = ins["payout_to_claim_range"]
                pay_date = start + pd.Timedelta(
                    days=int(ins["min_days_start_to_payout"]) + int(rng.integers(0, window + 1))
                )
                out.update(
                    insurance_payout_amount=round(claim * float(rng.uniform(olo, ohi))),
                    insurance_payout_date=pay_date.strftime("%Y-%m-%d"),
                )
            else:
                out["insurance_claim_status"] = "Approved"
    return out


# -------------------------------------------------------------------------- build
def build_dataset(
    cfg: dict,
    base_cfg: dict,
    reference: pd.DataFrame,
    district_codes: dict[str, int],
    weather: pd.DataFrame,
    crops: pd.DataFrame,
    zones: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Build the full dataset (train and test rows) and a diagnostics dict."""
    seed = int(cfg["seed"])
    base = dict(base_cfg)
    # optional documented synthetic overrides of the v1 mixes (chosen for learnability)
    for key in ("product_mix_percent", "asset_class_percent"):
        if f"{key}_override" in cfg:
            base[key] = cfg[f"{key}_override"]
    base["seed"] = seed
    base["total_accounts"] = int(cfg["total_accounts"])
    frame = generate(base, reference, district_codes)
    rng = np.random.default_rng(seed + 1)

    snapshots = cfg["snapshots"]
    frame["snapshot_year"] = assign_snapshots(frame, snapshots, rng)
    split_of = {s["year"]: s["split"] for s in snapshots}
    frame["split"] = frame["snapshot_year"].map(split_of)
    frame["reporting_date"] = frame["snapshot_year"].astype(str) + "-" + cfg["reporting_month_day"]

    rep = pd.to_datetime(frame["reporting_date"])
    start = pd.Timestamp(cfg["open_date_start"])
    latest = rep - pd.Timedelta(days=int(cfg["min_account_age_days"]))
    span = (latest - start).dt.days.to_numpy()
    offset = np.floor(rng.random(len(frame)) * (span + 1)).astype(int)
    opened = start + pd.to_timedelta(offset, unit="D")
    frame["account open date"] = opened.strftime("%Y-%m-%d")
    frame["loan_age_days"] = (rep - opened).dt.days

    frame["district_code"] = frame["_district"].map(district_codes).astype(int)
    frame = frame.rename(columns={"_district": "district_name"})
    frame = frame.merge(weather, how="left", left_on=["district_code", "snapshot_year"],
                        right_on=["district_code", "year"]).drop(columns="year")
    frame = frame.merge(zones, how="left", on="district_code")

    clip = float(cfg["anomaly_z_clip"])
    signs = cfg["weather_stress_signs"]
    signed = np.column_stack(
        [np.clip(frame[wx_name(c)].to_numpy(dtype=float), -clip, clip) * s for c, s in signs.items()]
    )
    # mean of the available signed z-scores; 0 only inside the generative score
    # when none is available (the exposed wx_* columns keep their NaN)
    stress = np.array([np.nanmean(r) if not np.isnan(r).all() else 0.0 for r in signed])

    # crop assignment: only the "Crop Production" sector carries a crop
    frame["crop"] = None
    frame["yield_anomaly_zscore"] = np.nan
    frame["yield_anomaly_status"] = "not_applicable"
    groups = {k: g.reset_index(drop=True) for k, g in crops.groupby(["district_code", "year"])}
    for i in frame.index[frame["agri sector"] == "Crop Production"]:
        pool = groups.get((int(frame.at[i, "district_code"]), int(frame.at[i, "snapshot_year"])))
        if pool is None or pool.empty:
            frame.at[i, "yield_anomaly_status"] = "no_crop_data"
            continue
        w = pool["area_1000_ha"].to_numpy(dtype=float)
        pick = pool.iloc[int(rng.choice(len(pool), p=w / w.sum()))]
        frame.at[i, "crop"] = pick["crop"]
        frame.at[i, "yield_anomaly_status"] = pick["anomaly_status"]
        frame.at[i, "yield_anomaly_zscore"] = pick["yield_anomaly_zscore"]

    yield_z = frame["yield_anomaly_zscore"].to_numpy(dtype=float)
    ins_rows = [
        make_insurance(frame.iloc[i], float(stress[i]), float(yield_z[i]), cfg, rng)
        for i in range(len(frame))
    ]
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(ins_rows)], axis=1)

    rs = cfg["risk_score"]
    sens = frame["agri sector"].map(cfg["sector_weather_sensitivity"]).to_numpy(dtype=float)
    agri = (frame["agri sector"] != "Non-Agri").to_numpy()
    status = frame["insurance_coverage_status"]
    uncovered = agri & status.isin(["Not Insured", "Pending"]).to_numpy()
    expired = agri & (status == "Expired").to_numpy()
    outstanding = frame["balance outstanding"].to_numpy(dtype=float)
    insured_amt = pd.to_numeric(frame["insurance_amount"]).to_numpy(dtype=float)
    shortfall = np.where(
        status.isin(["Active", "Expired"]).to_numpy(),
        np.clip(1 - insured_amt / np.maximum(outstanding, 1), 0, 1),
        0.0,
    )

    def z(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / values.std()

    yshort = np.where(np.isnan(yield_z), 0.0, np.clip(-yield_z, -clip, clip))
    signal = (
        rs["weather"] * sens * stress
        + rs["yield_shortfall"] * yshort
        + rs["uninsured_agri"] * uncovered
        + rs["expired_policy"] * expired
        + rs["coverage_shortfall"] * shortfall
        + rs["log_sanction"] * z(np.log(frame["sanction amount"].to_numpy(dtype=float)))
        + rs["interest_rate"] * z(frame["rate of interest"].to_numpy(dtype=float))
        + rs["account_age"] * z(frame["loan_age_days"].to_numpy(dtype=float))
    )
    noise = rng.normal(0.0, rs["noise_sd"], len(frame))
    score = signal + noise

    # Rank by score and cut at the configured class shares. Pooled over all years
    # (v2) a harsh weather year absorbs most NPA, so train and test class mixes
    # differ a lot. With rank_within_snapshot_year every year gets the target mix;
    # weather then differentiates districts and sectors inside each year.
    shares = {k: float(v) for k, v in base["asset_class_percent"].items()}
    if cfg.get("rank_within_snapshot_year", False):
        groups = [np.flatnonzero(frame["snapshot_year"].to_numpy() == y)
                  for y in sorted(frame["snapshot_year"].unique())]
    else:
        groups = [np.arange(len(frame))]
    asset = np.empty(len(frame), dtype=object)
    for idx in groups:
        order = idx[np.argsort(score[idx], kind="stable")]
        counts = apportion(len(idx), shares)
        cursor = 0
        for cls in ASSET_ORDER:
            asset[order[cursor : cursor + counts[cls]]] = cls
            cursor += counts[cls]
    frame["asset class"] = asset
    frame["npa_flag"] = (frame["asset class"] == "NPA").astype(int)

    dpd = np.zeros(len(frame), dtype=int)
    overdue = np.zeros(len(frame), dtype=int)
    for i, cls in enumerate(asset):
        lo, hi = base_cfg["dpd_range"][cls]
        dpd[i] = int(rng.integers(lo, hi + 1))
        olo, ohi = base_cfg["overdue_to_outstanding_range"][cls]
        overdue[i] = round(outstanding[i] * float(rng.uniform(olo, ohi))) if dpd[i] else 0
    frame["days_past_due"] = dpd
    frame["overdue amount"] = overdue

    end = pd.to_datetime(frame["insurance_end_date"])
    frame["insurance_days_to_expiry"] = (end - pd.to_datetime(frame["reporting_date"])).dt.days
    frame["insurance_coverage_ratio"] = (insured_amt / np.maximum(outstanding, 1)).round(4)
    frame.loc[frame["crop_insurance_available"] != "Y", "insurance_coverage_ratio"] = np.nan
    frame["synthetic_flag"] = "SYNTHETIC"
    for col in ("insurance_amount", "insurance_premium", "insurance_claim_amount",
                "insurance_payout_amount", "insurance_days_to_expiry"):
        frame[col] = pd.to_numeric(frame[col]).astype("Int64")

    diag = {
        "signal_variance_share": float(np.var(signal) / np.var(score)),
        "noise_sd": float(rs["noise_sd"]),
    }
    return frame, diag


# ------------------------------------------------------------------ dictionary / IO
def column_roles() -> list[tuple[str, str, str]]:
    """``(column, role, description)``. Roles drive which columns a model may use."""
    r: list[tuple[str, str, str]] = [
        ("synthetic_flag", "label", "Always SYNTHETIC."),
        ("split", "split", "train or test (out-of-time: by snapshot year)."),
        ("snapshot_year", "feature", "Reporting year of the snapshot; joins that year's weather."),
        ("reporting_date", "feature_raw_date", "Snapshot reporting date."),
        ("zone", "feature", "Constant placeholder zone."),
        ("circle", "feature", "Placeholder circle grouping."),
        ("branch ID", "identifier_exclude", "Placeholder branch id; high cardinality."),
        ("branch name", "identifier_exclude", "Placeholder branch name."),
        ("customer ID", "identifier_exclude", "Synthetic customer id."),
        ("customer name", "identifier_exclude", "Synthetic name."),
        ("borrowers address", "identifier_exclude", "Synthetic address."),
        ("pincode", "geo_id", "Pincode mapped to the district."),
        ("account number", "identifier_exclude", "Synthetic account number."),
        ("account open date", "feature_raw_date", "Account open date."),
        ("facility type", "feature", "Facility type."),
        ("scheme code", "feature", "Scheme code."),
        ("agri sector", "feature", "Agricultural sector."),
        ("sanction amount", "feature", "Sanctioned amount (INR)."),
        ("rate of interest", "feature", "Interest rate (%)."),
        ("balance outstanding", "feature", "Outstanding balance (INR)."),
        ("loan_age_days", "feature", "Days from account open date to reporting date."),
        ("district_code", "feature", "Project district code."),
        ("district_name", "feature", "District name."),
        ("climate_zone", "feature", "Issue 6 climate zone: similarity group, not a risk level."),
        ("crop", "feature", "Crop for Crop Production accounts only; else blank."),
        ("yield_anomaly_zscore", "feature", "Forward-chaining yield anomaly z-score; blank when not calculated. Not imputed."),
        ("yield_anomaly_status", "feature", "Status of the yield anomaly (calculated, insufficient_history, ...)."),
        ("crop_insurance_available", "feature", "Y or N."),
        ("insurance_scheme", "feature", "Synthetic scheme code."),
        ("insurance_amount", "feature", "Insured amount (INR)."),
        ("insurance_premium", "feature", "Premium (INR)."),
        ("insurance_provider", "feature", "Synthetic insurer."),
        ("insurance_coverage_status", "feature", "Active, Expired, Pending or Not Insured."),
        ("insurance_start_date", "feature_raw_date", "Policy start."),
        ("insurance_end_date", "feature_raw_date", "Policy end."),
        ("insurance_policy_id", "identifier_exclude", "Synthetic policy id."),
        ("insurance_days_to_expiry", "feature", "Policy end minus reporting date (negative = expired)."),
        ("insurance_coverage_ratio", "feature", "Insured amount / outstanding balance."),
        ("insurance_claim_status", "post_event_exclude", "Claim outcome after the weather event; excluded for early-warning use."),
        ("insurance_claim_amount", "post_event_exclude", "Claim amount."),
        ("insurance_payout_amount", "post_event_exclude", "Payout amount."),
        ("insurance_payout_date", "post_event_exclude", "Payout date."),
        ("days_past_due", "leakage_exclude", "Determined by the asset class (target leakage)."),
        ("overdue amount", "leakage_exclude", "Determined by the asset class (target leakage)."),
        ("asset class", "target", "Standard, SMA 0, SMA 1, SMA 2, NPA (ordered)."),
        ("npa_flag", "target_binary", "1 if asset class is NPA."),
    ]
    return r


def weather_roles(columns: list[str]) -> list[tuple[str, str, str]]:
    return [
        (wx_name(c), "feature", f"Forward-chaining anomaly z-score of {c}; blank when unavailable. Not imputed.")
        for c in columns
    ]


def ordered_columns(cfg: dict) -> list[str]:
    roles = column_roles()
    keep_ins = set(cfg.get("insurance_output_fields", INSURANCE_COLUMNS))
    names = [c for c, _, _ in roles if c not in INSURANCE_COLUMNS or c in keep_ins]
    # place weather features after climate_zone
    at = names.index("climate_zone") + 1
    wx = [wx_name(c) for c in cfg["weather_feature_columns"]]
    return names[:at] + wx + names[at:]


def build_report(frame: pd.DataFrame, cfg_hash: str, diag: dict, hashes: dict[str, str]) -> pd.DataFrame:
    """Aggregate counts and hashes only: no row-level values."""
    rec: list[tuple[str, str, str]] = [
        ("meta", "label", "SYNTHETIC - not PNB data"),
        ("meta", "seed", str(frame.attrs.get("seed", ""))),
        ("meta", "config_sha256", cfg_hash),
        ("meta", "rows", str(len(frame))),
        ("meta", "signal_variance_share", f"{diag['signal_variance_share']:.4f}"),
        ("meta", "noise_sd", str(diag["noise_sd"])),
    ]
    rec += [("file_sha256", k, v) for k, v in hashes.items()]
    for k, v in frame["split"].value_counts().items():
        rec.append(("rows_by_split", str(k), str(v)))
    for (s, c), v in frame.groupby(["split", "asset class"]).size().items():
        rec.append(("asset_class_by_split", f"{s} | {c}", str(v)))
    for k, v in frame.groupby("snapshot_year").size().items():
        rec.append(("rows_by_snapshot_year", str(k), str(v)))
    for k, v in frame["insurance_coverage_status"].value_counts().items():
        rec.append(("insurance_coverage_status", str(k), str(v)))
    for k, v in frame["insurance_claim_status"].value_counts().items():
        rec.append(("insurance_claim_status", str(k), str(v)))
    for k, v in frame["yield_anomaly_status"].value_counts().items():
        rec.append(("yield_anomaly_status", str(k), str(v)))
    rec.append(("meta", "accounts_with_crop", str(int(frame["crop"].notna().sum()))))
    return pd.DataFrame(rec, columns=["section", "key", "value"])


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "synthetic_training_config_v2.json")
    parser.add_argument("--pincode-reference", type=Path,
                        default=ROOT / "address_expansion" / "pincode_reference_v1.csv")
    parser.add_argument("--weather", type=Path,
                        default=ROOT / "weather_feature_engineering" / "weather_anomalies_forward_chaining_v1.csv")
    parser.add_argument("--yield-anomalies", type=Path,
                        default=ROOT / "yield_feature_engineering" / "yield_anomalies_forward_chaining_v1.csv")
    parser.add_argument("--zone-map", type=Path,
                        default=ROOT / "District Climate Zonal Clustering" / "district_zone_map_v1.csv")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="default: financial_train_test_data/<dataset_version>")
    parser.add_argument("--report-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    names = file_names(cfg.get("dataset_version", "v2"))
    if args.outdir is None:
        args.outdir = Path("financial_train_test_data") / cfg.get("dataset_version", "v2")
    base_cfg = json.loads((args.config.parent / cfg["base_config"]).read_text(encoding="utf-8"))
    years =[s["year"] for s in cfg["snapshots"]]
    reference = pd.read_csv(args.pincode_reference, dtype=str)
    codes = load_district_codes(args.weather)
    weather = load_weather(args.weather, cfg["weather_feature_columns"], years)
    crops = load_crop_table(args.yield_anomalies, years)
    zones = pd.read_csv(args.zone_map)[["district_code", "climate_zone"]]

    frame, diag = build_dataset(cfg, base_cfg, reference, codes, weather, crops, zones)
    frame.attrs["seed"] = cfg["seed"]
    cols = ordered_columns(cfg)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"columns not built: {missing}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = {"train": args.outdir / names["train"], "test": args.outdir / names["test"]}
    for split, path in paths.items():
        frame.loc[frame["split"] == split, cols].to_csv(path, index=False)
    # Same 20 columns, same order, as Financial_data_PNB.csv (no labels or extras).
    # Rows are the identical accounts as the train/test files above.
    layout = {
        "train": args.outdir / PNB_LAYOUT_TRAIN_NAME,
        "test": args.outdir / PNB_LAYOUT_TEST_NAME,
    }
    for split, path in layout.items():
        frame.loc[frame["split"] == split, PNB_LAYOUT_COLUMNS].to_csv(path, index=False)
    roles = {c: (r, d) for c, r, d in column_roles() + weather_roles(cfg["weather_feature_columns"])}
    pd.DataFrame(
        [(c, *roles[c]) for c in cols], columns=["column", "role", "description"]
    ).to_csv(args.outdir / names["dictionary"], index=False)

    report = build_report(
        frame, sha256_of(args.config), diag,
        {p.name: sha256_of(p) for p in paths.values()},
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_dir / names["report"], index=False)
    LOGGER.info("SYNTHETIC: train=%d test=%d -> %s",
                (frame["split"] == "train").sum(), (frame["split"] == "test").sum(), args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
