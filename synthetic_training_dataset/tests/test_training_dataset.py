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
    failed = [n for n, status, _ in results if status == "FAIL"]
    skipped = [n for n, status, _ in results if status == "SKIP"]
    assert not failed, failed
    assert not skipped, skipped  # full frame has every column, so nothing is skipped


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


# ----------------------------------------------------------------------------- v3
CFG3 = json.loads((HERE / "synthetic_training_config_v3.json").read_text())
CFG3["total_accounts"] = 1100
CFG3["snapshots"] = CFG["snapshots"]


@pytest.fixture(scope="module")
def built3(inputs):
    reference, weather, crops, zones = inputs
    frame, _ = build_dataset(CFG3, BASE, reference, CODES, weather, crops, zones)
    return frame


def test_v3_writes_only_six_insurance_fields() -> None:
    from generate_training_dataset import INSURANCE_COLUMNS

    cols = ordered_columns(CFG3)
    written = [c for c in cols if c in INSURANCE_COLUMNS]
    assert sorted(written) == sorted(CFG3["insurance_output_fields"])
    assert len(written) == 6
    assert "insurance_coverage_ratio" in cols and "insurance_days_to_expiry" in cols


def test_v3_asset_mix_follows_override(built3) -> None:
    share = built3["asset class"].value_counts(normalize=True) * 100
    for cls, pct in CFG3["asset_class_percent_override"].items():
        assert abs(share[cls] - pct) < 0.5


def test_v3_product_mix_uses_only_v1_combinations(built3) -> None:
    allowed = {(r["scheme code"], r["facility type"], r["agri sector"]) for r in BASE["product_mix_percent"]}
    used = set(zip(built3["scheme code"], built3["facility type"], built3["agri sector"], strict=True))
    assert used <= allowed
    assert abs(sum(r["weight"] for r in CFG3["product_mix_percent_override"]) - 100) < 1e-9
    assert (built3["agri sector"] == "Crop Production").mean() > 0.25


def test_v3_checks_skip_dropped_columns_without_failing(built3, inputs) -> None:
    out = built3[ordered_columns(CFG3)]
    results = run_checks(out[out["split"] == "train"], out[out["split"] == "test"], CFG3, inputs[1])
    assert not [n for n, s, _ in results if s == "FAIL"]
    assert [n for n, s, _ in results if s == "SKIP"]  # dropped columns are reported, not passed


# ----------------------------------------------------------------------------- v4
CFG4 = json.loads((HERE / "synthetic_training_config_v4.json").read_text())
CFG4["total_accounts"] = 1100
CFG4["snapshots"] = CFG["snapshots"]
SECTORS4 = {"Crop Production", "Fisheries", "Poultry", "Dairy & Animal Husbandry"}


@pytest.fixture(scope="module")
def built4(inputs):
    reference, weather, crops, zones = inputs
    frame, _ = build_dataset(CFG4, BASE, reference, CODES, weather, crops, zones)
    return frame


def test_v4_agri_sector_only_the_four_allowed(built4) -> None:
    assert set(built4["agri sector"]) == SECTORS4
    assert set(CFG4["allowed_agri_sectors"]) == SECTORS4


def test_v4_product_mix_is_valid(built4) -> None:
    weights = [r["weight"] for r in CFG4["product_mix_percent_override"]]
    assert abs(sum(weights) - 100) < 1e-9
    assert {r["agri sector"] for r in CFG4["product_mix_percent_override"]} == SECTORS4
    # no scheme code or facility type outside the original PNB domain
    schemes = {r["scheme code"] for r in BASE["product_mix_percent"]}
    facilities = {r["facility type"] for r in BASE["product_mix_percent"]}
    assert set(built4["scheme code"]) <= schemes
    assert set(built4["facility type"]) <= facilities


def test_v4_sector_config_maps_cover_every_sector() -> None:
    for key in (CFG4["sector_weather_sensitivity"], CFG4["insurance"]["availability_probability"]):
        assert set(key) == SECTORS4


def test_v4_crop_only_for_crop_production_and_all_checks_pass(built4, inputs) -> None:
    assert built4.loc[built4["crop"].notna(), "agri sector"].eq("Crop Production").all()
    out = built4[ordered_columns(CFG4)]
    results = run_checks(out[out["split"] == "train"], out[out["split"] == "test"], CFG4, inputs[1])
    assert not [n for n, s, _ in results if s == "FAIL"]
    assert [s for n, s, _ in results if n.startswith("V19")] == ["PASS"]


def test_v4_sector_check_fails_on_an_unexpected_sector(built4, inputs) -> None:
    out = built4[ordered_columns(CFG4)].copy()
    out.loc[out.index[0], "agri sector"] = "Horticulture"
    results = run_checks(out[out["split"] == "train"], out[out["split"] == "test"], CFG4, inputs[1])
    assert [s for n, s, _ in results if n.startswith("V19")] == ["FAIL"]


def test_synthetic_flag_column_is_config_driven() -> None:
    assert "synthetic_flag" not in ordered_columns(CFG3)  # v3: column removed
    assert "synthetic_flag" in ordered_columns(CFG)  # v2 unchanged


def test_v3_label_check_is_skipped_not_passed_without_the_column(built3, inputs) -> None:
    out = built3[ordered_columns(CFG3)]
    results = run_checks(out[out["split"] == "train"], out[out["split"] == "test"], CFG3, inputs[1])
    v1 = [s for n, s, _ in results if n.startswith("V1 ")]
    assert v1 == ["SKIP"]
