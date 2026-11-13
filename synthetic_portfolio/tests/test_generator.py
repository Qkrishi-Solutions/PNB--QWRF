"""Tests for the SYNTHETIC portfolio generator (tiny config, no real data)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from generate_synthetic_portfolio import OUTPUT_COLUMNS, apportion, generate
from verify_synthetic_portfolio import run_checks

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "synthetic_config_v1.json").read_text()
)
CONFIG["total_accounts"] = 400
CODES = {d: 500 + i for i, d in enumerate(CONFIG["district_population_2011"])}


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    rows = [
        {
            "pincode": str(515000 + 10 * code + k),
            "district_code": str(code),
            "mapping_status": "mapped",
        }
        for code in CODES.values()
        for k in range(1, 5)
    ]
    rows.append(
        {
            "pincode": "515999",
            "district_code": "",
            "mapping_status": "ambiguous_district",
        }
    )
    return pd.DataFrame(rows)


def test_apportion_sums_exactly() -> None:
    counts = apportion(5000, CONFIG["district_population_2011"])
    assert sum(counts.values()) == 5000
    assert all(v >= 1 for v in counts.values())


def test_same_seed_is_identical_and_other_seed_differs(reference: pd.DataFrame) -> None:
    a = generate(CONFIG, reference, CODES)
    b = generate(CONFIG, reference, CODES)
    c = generate({**CONFIG, "seed": 7}, reference, CODES)
    pd.testing.assert_frame_equal(a, b)
    assert not a[OUTPUT_COLUMNS].equals(c[OUTPUT_COLUMNS])


def test_columns_and_row_count(reference: pd.DataFrame) -> None:
    frame = generate(CONFIG, reference, CODES)
    assert list(frame[OUTPUT_COLUMNS].columns) == OUTPUT_COLUMNS
    assert len(frame) == 400


def test_ruleset_checks_all_pass(reference: pd.DataFrame) -> None:
    frame = generate(CONFIG, reference, CODES).astype(str)
    cfg = {
        **CONFIG,
        "t5_tolerance_pp": {"small_district_rows": 150, "small": 15, "large": 15},
    }
    failures = [r for r in run_checks(frame, cfg, reference, CODES, set()) if not r[1]]
    assert failures == []


def test_ambiguous_pincode_never_used(reference: pd.DataFrame) -> None:
    frame = generate(CONFIG, reference, CODES)
    assert "515999" not in set(frame["pincode"])


def test_avoided_values_not_reused(reference: pd.DataFrame) -> None:
    first = generate(CONFIG, reference, CODES)
    avoid = set(first["account number"]) | set(first["customer name"])
    second = generate(CONFIG, reference, CODES, avoid)
    assert not set(second["account number"]) & avoid
    assert not set(second["customer name"]) & set(first["customer name"])
