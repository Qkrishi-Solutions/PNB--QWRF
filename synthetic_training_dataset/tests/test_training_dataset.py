"""Tests for the SYNTHETIC training dataset generator (tiny fixtures, no real data)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from generate_training_dataset import (
    build_dataset, column_roles, ordered_columns, weather_roles, wx_name,
)
from verify_training_dataset import run_checks

HERE = Path(__file__).resolve().parents[1]
CFG = json.loads((HERE / "synthetic_training_config_v2.json").read_text())
BASE = json.loads((HERE.parent / "synthetic_portfolio" / "synthetic_config_v1.json").read_text())
CFG["total_accounts"] = 1100
CFG["snapshots"] = [
    {"year": 2017, "split": "train", "accounts": 500},
    {"year": 2018, "split": "train", "accounts": 500},
    {"year": 2019, "split": "test", "accounts": 100},
]
CODES = {d: 500 + i for i, d in enumerate(BASE["district_population_2011"])}
YEARS = [2017, 2018, 2019]


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame(
        [
            {"pincode": str(515000 + 10 * c + k), "district_code": str(c), "mapping_status": "mapped"}
            for c in CODES.values()
            for k in range(1, 5)
        ]
    )
    cols = CFG["weather_feature_columns"]
    weather = pd.DataFrame(
        [
            {"district_code": c, "year": y, **{wx_name(col): rng.normal(0, 1.5) for col in cols}}
            for c in CODES.values()
            for y in YEARS
        ]
    )
    weather.loc[0, wx_name(cols[0])] = np.nan  # missing stays missing
    crops = pd.DataFrame(
        [
            {
                "district_code": c, "year": y, "crop": crop, "area_1000_ha": 10.0 + i,
                "anomaly_status": "calculated" if i % 2 == 0 else "insufficient_history",
                "yield_anomaly_zscore": rng.normal(0, 1) if i % 2 == 0 else np.nan,
            }
            for c in CODES.values()
            for y in YEARS
            for i, crop in enumerate(["rice", "cotton", "maize"])
        ]
    )
    zones = pd.DataFrame({"district_code": list(CODES.values()), "climate_zone": "Climate Zone 1"})
    return reference, weather, crops, zones


@pytest.fixture(scope="module")
def built(inputs):
    reference, weather, crops, zones = inputs
    frame, diag = build_dataset(CFG, BASE, reference, CODES, weather, crops, zones)
    return frame, diag


def test_all_checks_pass(built, inputs) -> None:
    frame, _ = built
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    results = run_checks(train, test, CFG, inputs[1])
    failed = [n for n, ok, _ in results if not ok]
    assert not failed, failed


def test_split_sizes_close_to_config(built) -> None:
    frame, _ = built
    counts = frame["snapshot_year"].value_counts()
    for snap in CFG["snapshots"]:
        assert abs(counts[snap["year"]] - snap["accounts"]) <= 15


def test_deterministic(inputs) -> None:
    reference, weather, crops, zones = inputs
    a, _ = build_dataset(CFG, BASE, reference, CODES, weather, crops, zones)
    b, _ = build_dataset(CFG, BASE, reference, CODES, weather, crops, zones)
    pd.testing.assert_frame_equal(a, b)


def test_missing_weather_not_imputed(built) -> None:
    frame, _ = built
    col = wx_name(CFG["weather_feature_columns"][0])
    assert frame[col].isna().any()


def test_asset_mix_matches_config(built) -> None:
    frame, _ = built
    share = frame["asset class"].value_counts(normalize=True) * 100
    for cls, pct in BASE["asset_class_percent"].items():
        assert abs(share[cls] - pct) < 0.5


def test_weather_effect_is_injected(built) -> None:
    frame, diag = built
    assert 0 < diag["signal_variance_share"] < 1


def test_dictionary_covers_output_columns(built) -> None:
    frame, _ = built
    roles = {c for c, _, _ in column_roles() + weather_roles(CFG["weather_feature_columns"])}
    cols = ordered_columns(CFG)
    assert set(cols) <= roles
    assert set(cols) <= set(frame.columns)


def test_leakage_columns_are_not_features() -> None:
    roles = {c: r for c, r, _ in column_roles()}
    for col in ("days_past_due", "overdue amount"):
        assert roles[col] == "leakage_exclude"
    for col in ("insurance_claim_status", "insurance_payout_amount"):
        assert roles[col] == "post_event_exclude"
