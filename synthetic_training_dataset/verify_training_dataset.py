"""Verify the SYNTHETIC train/test dataset (v2) and run a baseline sanity model.

Every rule is counted over all rows. Exit code 0 only if all checks pass.
The baseline shows the label is learnable *because the rules were injected*;
it is not evidence about real credit risk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_training_dataset import (  # noqa: E402
    DICTIONARY_NAME, TEST_NAME, TRAIN_NAME, wx_name,
)

DPD_BANDS = {"Standard": (0, 0), "SMA 0": (1, 30), "SMA 1": (31, 60),
             "SMA 2": (61, 90), "NPA": (91, 365)}
CLAIM_STATUSES = {"Filed", "Approved", "Rejected", "Paid"}


def _date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def run_checks(
    train: pd.DataFrame, test: pd.DataFrame, cfg: dict, weather: pd.DataFrame | None = None
) -> list[tuple[str, bool, str]]:
    """Return ``(check, passed, detail)`` rows."""
    both = pd.concat([train, test], ignore_index=True)
    out: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, bool(ok), detail))

    add("V1 all rows labelled SYNTHETIC", (both["synthetic_flag"] == "SYNTHETIC").all())
    years_train = sorted(s["year"] for s in cfg["snapshots"] if s["split"] == "train")
    years_test = sorted(s["year"] for s in cfg["snapshots"] if s["split"] == "test")
    add("V2 split follows snapshot year (out-of-time)",
        set(train["snapshot_year"]) == set(years_train)
        and set(test["snapshot_year"]) == set(years_test)
        and max(years_train) < min(years_test),
        f"train={years_train} test={years_test}")
    add("V3 no customer shared between train and test",
        not set(train["customer ID"]) & set(test["customer ID"]))
    add("V4 account numbers unique", both["account number"].is_unique)
    add("V5 X1 outstanding <= sanction", (both["balance outstanding"] <= both["sanction amount"]).all())
    add("V6 X2 overdue <= outstanding", (both["overdue amount"] <= both["balance outstanding"]).all())
    bands_ok = all(
        both.loc[both["asset class"] == c, "days_past_due"].between(lo, hi).all()
        for c, (lo, hi) in DPD_BANDS.items()
    )
    add("V7 X3 asset class agrees with DPD band", bands_ok)
    add("V8 X4 dpd = 0 <=> overdue = 0",
        ((both["days_past_due"] == 0) == (both["overdue amount"] == 0)).all())
    rep, opened = _date(both["reporting_date"]), _date(both["account open date"])
    add("V9 X5 open date <= reporting date, min age respected",
        ((rep - opened).dt.days >= int(cfg["min_account_age_days"])).all())

    yes = both["crop_insurance_available"] == "Y"
    blank_cols = [c for c in both.columns if c.startswith("insurance_") and c != "insurance_claim_status"
                  and c != "insurance_coverage_status"]
    add("V10 uninsured rows have blank insurance fields",
        both.loc[~yes, blank_cols].isna().all().all())
    add("V11 insured rows have policy fields and unique policy ids",
        both.loc[yes, ["insurance_policy_id", "insurance_amount", "insurance_premium",
                       "insurance_start_date", "insurance_end_date"]].notna().all().all()
        and both.loc[yes, "insurance_policy_id"].is_unique)
    ins = both.loc[yes]
    start, end, irep = _date(ins["insurance_start_date"]), _date(ins["insurance_end_date"]), _date(ins["reporting_date"])
    add("V12 policy end after start", (end > start).all())
    expected = np.where(end < irep, "Expired", np.where(start > irep, "Pending", "Active"))
    add("V13 coverage status agrees with dates",
        (ins["insurance_coverage_status"].to_numpy() == expected).all())
    add("V14 premium < insured amount", (ins["insurance_premium"] < ins["insurance_amount"]).all())
    has_claim = both["insurance_claim_status"].isin(CLAIM_STATUSES)
    add("V15 claim amount present iff a claim exists",
        (both["insurance_claim_amount"].notna() == has_claim).all())
    paid = both["insurance_payout_amount"].notna()
    add("V16 payout only for Paid claims, payout <= claim, dated within policy..reporting",
        (both.loc[paid, "insurance_claim_status"] == "Paid").all()
        and (both.loc[paid, "insurance_payout_amount"] <= both.loc[paid, "insurance_claim_amount"]).all()
        and (_date(both.loc[paid, "insurance_payout_date"]) >= _date(both.loc[paid, "insurance_start_date"])).all()
        and (_date(both.loc[paid, "insurance_payout_date"]) <= _date(both.loc[paid, "reporting_date"])).all()
        and both.loc[~paid, "insurance_payout_date"].isna().all())
    add("V17 crop only for Crop Production; yield z only when calculated",
        both.loc[both["crop"].notna(), "agri sector"].eq("Crop Production").all()
        and both.loc[both["yield_anomaly_zscore"].notna(), "yield_anomaly_status"].eq("calculated").all())
    if weather is not None:
        cols = cfg["weather_feature_columns"]
        merged = both[["district_code", "snapshot_year"] + [wx_name(c) for c in cols]].drop_duplicates().merge(
            weather, how="left", left_on=["district_code", "snapshot_year"], right_on=["district_code", "year"],
            suffixes=("", "_src"))
        same = all(
            np.allclose(merged[wx_name(c)].astype(float), merged[wx_name(c) + "_src"].astype(float),
                        equal_nan=True) for c in cols
        )
        add("V18 weather features equal the source anomalies (not altered or imputed)", same)
    return out


def baseline(train: pd.DataFrame, test: pd.DataFrame, dictionary: pd.DataFrame, seed: int = 0) -> dict:
    """Gradient boosting on role==feature columns vs majority and shuffled-label controls."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score

    feats = dictionary.loc[dictionary["role"] == "feature", "column"].tolist()
    feats = [c for c in feats if c in train.columns]

    def prep(frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[feats].copy()
        for c in x.columns:
            if x[c].dtype == object or str(x[c].dtype).startswith("str"):
                x[c] = x[c].astype("category")
        return x

    xtr, xte = prep(train), prep(test)
    for c in xtr.columns:
        if str(xtr[c].dtype) == "category":
            cats = pd.api.types.union_categoricals([xtr[c], xte[c]]).categories
            xtr[c] = pd.Categorical(xtr[c], categories=cats)
            xte[c] = pd.Categorical(xte[c], categories=cats)
    ytr, yte = train["asset class"], test["asset class"]

    def fit_eval(y_fit: pd.Series) -> tuple[float, float]:
        model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed,
                                               max_iter=200, learning_rate=0.05)
        model.fit(xtr, y_fit)
        pred = model.predict(xte)
        return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")

    acc, f1 = fit_eval(ytr)
    shuffled = ytr.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled.index = ytr.index
    s_acc, s_f1 = fit_eval(shuffled)
    major = ytr.mode().iloc[0]
    return {
        "n_features": len(feats),
        "model_accuracy": acc, "model_macro_f1": f1,
        "shuffled_label_accuracy": s_acc, "shuffled_label_macro_f1": s_f1,
        "majority_accuracy": float((yte == major).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    import json

    root = HERE.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", type=Path, default=Path("data/private/synthetic_training"))
    parser.add_argument("--config", type=Path, default=HERE / "synthetic_training_config_v2.json")
    parser.add_argument("--weather", type=Path,
                        default=root / "weather_feature_engineering" / "weather_anomalies_forward_chaining_v1.csv")
    parser.add_argument("--baseline", action="store_true", help="also run the sanity model")
    args = parser.parse_args(argv)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    train = pd.read_csv(args.datadir / TRAIN_NAME)
    test = pd.read_csv(args.datadir / TEST_NAME)
    from generate_training_dataset import load_weather
    weather = load_weather(args.weather, cfg["weather_feature_columns"], [s["year"] for s in cfg["snapshots"]])

    results = run_checks(train, test, cfg, weather)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\nrows: train={len(train)} test={len(test)}; checks passed {len(results) - len(failed)}/{len(results)}")

    if args.baseline:
        dictionary = pd.read_csv(args.datadir / DICTIONARY_NAME)
        res = baseline(train, test, dictionary)
        print("\nbaseline (SYNTHETIC rules were injected; this is a pipeline check, not a finding):")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
