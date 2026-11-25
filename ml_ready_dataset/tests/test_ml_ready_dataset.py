"""Tests for Issue #7: integrated ML-ready dataset. Run from the repo root:
    python -m pytest ml_ready_dataset/tests -v
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import ml_ready_dataset.build_ml_ready_dataset as mod
from ml_ready_dataset.build_ml_ready_dataset import DEFAULTS, KEY, DY, build

OUT = "ml_ready_dataset"
DATA = os.path.join(OUT, "ml_ready_dataset_v1.csv")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def checks():
    return pd.read_csv(os.path.join(OUT, "validation_checks_v1.csv"))


def test_deliverables_exist():
    for f in ["ml_ready_dataset_v1.csv", "ml_ready_column_dictionary_v1.csv", "join_coverage_report_v1.csv",
              "unmatched_keys_v1.csv", "exclusion_report_v1.csv", "validation_checks_v1.csv",
              "reproducibility_report_v1.json", "README.md"]:
        assert os.path.getsize(os.path.join(OUT, f)) > 0, f


def test_key_unique(df):
    assert not df.duplicated(KEY).any()


def test_row_count_reconciles_to_inputs(df):
    crop_long = pd.read_csv(DEFAULTS["crop_long"])
    elig = pd.read_csv(DEFAULTS["eligibility"])
    ok = set(elig.loc[elig["proceed_to_modelling"].astype(bool), "crop"])
    expected = int((crop_long["crop"].isin(ok) & crop_long["use_in_modelling"].astype(bool)).sum())
    assert len(df) == expected
    assert set(df["crop"]) == ok


def test_all_recorded_checks_pass(checks):
    assert checks["passed"].all(), checks[~checks["passed"]]
    assert (checks["kind"] == "leakage").sum() >= 5


def test_every_join_reports_matched_and_unmatched():
    cov = pd.read_csv(os.path.join(OUT, "join_coverage_report_v1.csv"))
    assert {"eligibility", "yield_anomaly", "weather", "climate_zone", "crop_feature_selection"} <= set(cov["join"])
    assert {"matched_base_keys", "unmatched_base_keys"} <= set(cov.columns)
    um = pd.read_csv(os.path.join(OUT, "unmatched_keys_v1.csv"))
    n_unmatched = int(cov["unmatched_base_keys"].sum())
    assert (um["side"] == "base_without_source").sum() == n_unmatched


def test_weather_is_deduplicated_to_district_year(df):
    w = [c for c in df.columns if c.startswith("wanom__")]
    per_dy = df.groupby(DY)[w].nunique(dropna=False).max().max()
    assert per_dy == 1, "weather must be identical across crops within a district-year"


def test_missing_values_not_imputed(df):
    src = pd.read_csv(DEFAULTS["yield_fc"]).merge(df[KEY], on=KEY)
    assert df["yield_anomaly_kg_per_ha"].isna().sum() == src["yield_anomaly_kg_per_ha"].isna().sum()
    early = df[df["year"].isin([2015, 2016])]
    assert early["yield_anomaly_kg_per_ha"].isna().all()
    # 2015/2016 weather anomalies need two prior years: stay missing, never zero
    w = [c for c in df.columns if c.startswith("wanom__")]
    assert df[df["year"] == 2015][w].isna().all().all()
    # A numeric kg/ha anomaly exists only where it was calculated. zero_baseline_std rows
    # keep a kg/ha anomaly but have an undefined z-score (no spread in the baseline).
    numeric = df[df["yield_anomaly_kg_per_ha"].notna()]
    assert set(numeric["anomaly_status"]) <= {"calculated", "zero_baseline_std"}
    zero_std = numeric[numeric["anomaly_status"] == "zero_baseline_std"]
    assert zero_std["yield_anomaly_zscore"].isna().all()
    # insufficient history / ineligible observation never carry a numeric anomaly
    assert df[df["anomaly_status"].isin(["insufficient_history", "ineligible_observation"])][
        "yield_anomaly_kg_per_ha"].isna().all()


def test_unmatched_issue5_crop_stays_missing(df):
    fs = pd.read_csv(DEFAULTS["feature_selection"])
    unmatched = set(df["crop"]) - set(fs["crop"])
    for c in unmatched:
        assert df.loc[df["crop"] == c, "loyo_n_stable_predictors"].isna().all()


def test_column_dictionary_covers_all_columns_and_flags_leakage(df):
    d = pd.read_csv(os.path.join(OUT, "ml_ready_column_dictionary_v1.csv"))
    assert set(d["column"]) == set(df.columns)
    assert "baseline_leakage_safe" in d.columns and "forward_chaining_safe_feature" not in d.columns
    safe = set(d.loc[d["baseline_leakage_safe"], "column"])
    assert safe
    n_weather = sum(c.startswith("wanom__") for c in df.columns)
    assert len(safe) == 3 * n_weather + 2, "3 weather forms per feature + climate_zone + crop"
    assert {"crop", "climate_zone"} <= safe
    for c in safe:
        assert c in ("crop", "climate_zone") or c.startswith(("wanom__", "wz__", "wact__")), c
    target_cols = set(d.loc[d["role"] == "target", "column"])
    assert "yield_anomaly_kg_per_ha" in target_cols
    assert not (target_cols & safe)
    loyo = d[d["role"] == "selection_info_loyo"]
    assert len(loyo) > 0 and not loyo["baseline_leakage_safe"].any()


def test_weather_features_carry_timing_caveat():
    d = pd.read_csv(os.path.join(OUT, "ml_ready_column_dictionary_v1.csv"))
    w = d[d["column"].str.startswith(("wanom__", "wz__", "wact__"))]
    assert len(w) > 0 and w["timing_caveat"].fillna("").str.contains("NOT established").all()
    readme = open(os.path.join(OUT, "README.md"), encoding="utf-8").read()
    assert "COLUMN_DICTIONARY_START" in readme and "Timing caveat" in readme
    assert "`yield_anomaly_kg_per_ha`" in readme and "`wanom__<window>__<indicator>`" in readme


def test_unmatched_keys_separate_real_gaps_from_expected_exclusions():
    um = pd.read_csv(os.path.join(OUT, "unmatched_keys_v1.csv"))
    cov = pd.read_csv(os.path.join(OUT, "join_coverage_report_v1.csv"))
    assert "reason" in um.columns
    gaps = um[um["reason"].str.startswith("GAP")]
    assert len(gaps) == int(cov["unmatched_base_keys"].sum())
    assert (gaps["side"] == "base_without_source").all()
    yields = um[(um["join"] == "yield_anomaly") & (um["side"] == "source_not_in_base")]
    assert yields["reason"].str.startswith("expected").all()


def test_forward_chaining_baselines_use_earlier_years_only():
    w = pd.read_csv(DEFAULTS["weather"]).dropna(subset=["training_years"])
    for ty, y in zip(w["training_years"], w["year"]):
        yrs = [int(float(v)) for v in str(ty).split(";") if v.strip()]
        if yrs:
            assert max(yrs) < y


def test_build_is_reproducible_to_float_tolerance(tmp_path):
    """Same values on any pandas/numpy; byte-identical only with the pinned versions."""
    build(DEFAULTS_PATHS(), str(tmp_path), "pytest")
    a = pd.read_csv(DATA)
    b = pd.read_csv(tmp_path / "ml_ready_dataset_v1.csv")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_builder_fails_on_baseline_leakage_in_inputs(tmp_path):
    """Mutation test: a weather baseline that includes the target year must stop the build."""
    w = pd.read_csv(DEFAULTS["weather"])
    i = w.index[w["training_years"].notna()][0]
    w.loc[i, "training_years"] = str(w.loc[i, "year"]) + ";" + str(w.loc[i, "training_years"])
    bad = tmp_path / "weather_bad.csv"
    w.to_csv(bad, index=False)
    paths = DEFAULTS_PATHS()
    paths["weather"] = str(bad)
    with pytest.raises(SystemExit):
        build(paths, str(tmp_path / "out"), "pytest")


def test_builder_fails_on_duplicate_key_in_inputs(tmp_path):
    y = pd.read_csv(DEFAULTS["yield_fc"])
    bad = tmp_path / "yield_bad.csv"
    pd.concat([y, y.iloc[[0]]]).to_csv(bad, index=False)
    paths = DEFAULTS_PATHS()
    paths["yield_fc"] = str(bad)
    with pytest.raises(AssertionError):
        build(paths, str(tmp_path / "out"), "pytest")


def test_builder_fails_if_a_target_column_is_marked_safe(tmp_path, monkeypatch):
    """The provenance check must be able to fail: mark a target safe and the build must stop."""
    real = mod.column_role

    def bad_role(c):
        r = list(real(c))
        if c == "yield_anomaly_kg_per_ha":
            r[3] = True  # baseline_leakage_safe
        return tuple(r)

    monkeypatch.setattr(mod, "column_role", bad_role)
    with pytest.raises(SystemExit):
        build(DEFAULTS_PATHS(), str(tmp_path), "pytest")


def test_reproducibility_report_has_required_fields():
    r = json.load(open(os.path.join(OUT, "reproducibility_report_v1.json")))
    assert {"command", "python", "packages", "input_sha256", "row_counts", "output_sha256"} <= set(r)
    assert "platform" not in r  # machine-specific strings make the file noisy
    assert {"pandas", "numpy"} <= set(r["packages"])


def DEFAULTS_PATHS():
    return {"crop_long": DEFAULTS["crop_long"], "eligibility": DEFAULTS["eligibility"],
            "weather": DEFAULTS["weather"], "yield_fc": DEFAULTS["yield_fc"],
            "feature_selection": DEFAULTS["feature_selection"], "zones": DEFAULTS["zones"]}
