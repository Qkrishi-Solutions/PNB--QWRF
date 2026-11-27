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
              "reproducibility_report_v1.json", "README.md", "missing_data_report_v1.csv",
              "excluded_crop_rows_v1.csv", "weather_loyo_anomalies_v1.csv"]:
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


# ---------------- forward-chaining crop-specific predictors (fc_*) ----------------

def _fc_summary():
    return pd.read_csv(os.path.join(OUT, "fc_crop_feature_selection_v1.csv"))


def test_fc_selection_trains_only_on_earlier_target_years():
    fc = _fc_summary()
    assert len(fc) == fc["crop"].nunique() * fc["year"].nunique()
    for r in fc.itertuples(index=False):
        if isinstance(r.fc_selection_train_years, str) and r.fc_selection_train_years:
            assert max(int(v) for v in r.fc_selection_train_years.split(";")) < r.year


def test_fc_selection_status_semantics(df):
    fc = _fc_summary()
    assert set(fc["fc_selection_status"]) <= {
        "no_prior_target_years", "crop_excluded_by_issue5_row_filter",
        "no_predictors_passed_screen", "selected", "insufficient_training_samples",
        "no_features_with_sufficient_data"}
    # No forward-chaining anomaly exists before 2017, so nothing earlier can be selected from.
    early = fc[fc["year"] <= 2017]
    assert early["fc_n_selected_predictors"].isna().all()
    # Ran-and-found-nothing (0) is distinct from could-not-run (missing)
    ran_none = fc[fc["fc_selection_status"] == "no_predictors_passed_screen"]
    assert (ran_none["fc_n_selected_predictors"] == 0).all()
    sel = fc[fc["fc_selection_status"] == "selected"]
    assert (sel["fc_n_selected_predictors"] >= 1).all() and (sel["fc_n_selected_predictors"] <= 4).all()
    assert sel["fc_selected_predictors"].notna().all()
    # attached to dataset rows by (crop, year)
    cols = ["fc_selection_status", "fc_n_selected_predictors", "fc_selected_predictors"]
    chk = df[["crop", "year"] + cols].drop_duplicates()
    assert len(chk) == len(fc[fc["crop"].isin(df["crop"])])


def test_rabi_sorghum_gap_is_explained_by_issue5_row_filter():
    fc = _fc_summary()
    rs = fc[fc["crop"] == "rabi_sorghum"]
    assert (rs["fc_selection_status"] == "crop_excluded_by_issue5_row_filter").all()
    crop_long = pd.read_csv(DEFAULTS["crop_long"])
    assert crop_long.loc[crop_long["crop"] == "rabi_sorghum", "is_seasonal_component"].astype(bool).all()


def test_fc_selection_ignores_future_target_years(df):
    i5 = mod.load_issue5(DEFAULTS["issue5_script"])
    summary, _ = mod.forward_chaining_selection(df, i5)
    assert mod.selection_ignores_future(df, i5, summary) == []
    on_disk = _fc_summary().reset_index(drop=True)
    fresh = summary.reset_index(drop=True)
    text = ["crop", "fc_selection_train_years", "fc_selection_status", "fc_selected_predictors"]
    for c in text:  # empty string and NaN are the same after a CSV round trip
        assert fresh[c].fillna("").astype(str).tolist() == on_disk[c].fillna("").astype(str).tolist(), c
    for c in ["year", "fc_selection_n_training_rows", "fc_n_selected_predictors"]:
        assert np.allclose(fresh[c].astype(float), on_disk[c].astype(float), equal_nan=True), c


def test_leakage_check_fails_if_selection_uses_the_target_year(tmp_path, monkeypatch):
    """Mutation test: a selection that also trains on year == T (look-ahead) must stop the build."""
    real = mod._select_one

    def leaky(pool, feats, i5, crop, target_year):
        return real(pool, feats, i5, crop, target_year + 1)  # trains on year <= T

    monkeypatch.setattr(mod, "_select_one", leaky)
    with pytest.raises(SystemExit):
        build(DEFAULTS_PATHS(), str(tmp_path), "pytest")


def test_dictionary_marks_fc_columns_and_points_loyo_users_to_them():
    d = pd.read_csv(os.path.join(OUT, "ml_ready_column_dictionary_v1.csv"))
    fc = d[d["column"].str.startswith("fc_")]
    assert len(fc) == 5 and (fc["role"] == "selection_info_forward_chaining").all()
    loyo = d[d["role"] == "selection_info_loyo"]
    assert loyo["leakage_note"].str.contains("fc_").all()


# ---------------- completeness: nothing missing is silent ----------------

def _crop_long():
    return pd.read_csv(DEFAULTS["crop_long"])


def test_every_crop_long_row_is_carried_or_listed_as_excluded(df):
    cl = _crop_long()
    ex = pd.read_csv(os.path.join(OUT, "excluded_crop_rows_v1.csv"))
    assert len(df) + len(ex) == len(cl)
    assert not ex.duplicated(KEY).any()
    assert len(ex[KEY].merge(df[KEY], on=KEY)) == 0
    assert set(ex["crop"]) == set(cl["crop"]) - set(df["crop"])
    assert {"exclusion_reason", "eligibility_class", "anomaly_status"} <= set(ex.columns)


def test_excluded_rows_carry_the_issue4_anomalies_from_the_source():
    ex = pd.read_csv(os.path.join(OUT, "excluded_crop_rows_v1.csv"))
    src = pd.read_csv(DEFAULTS["yield_fc"])
    m = ex.merge(src[KEY + ["yield_anomaly_kg_per_ha"]], on=KEY, suffixes=("", "_src"))
    assert len(m) == len(ex)
    assert np.allclose(m["yield_anomaly_kg_per_ha"].fillna(-1e18), m["yield_anomaly_kg_per_ha_src"].fillna(-1e18))


def test_yield_value_status_partitions_rows_and_labels_zeros_correctly(df):
    assert df["yield_value_status"].notna().all()
    assert set(df["yield_value_status"]) <= {"valid", "missing", "zero_source_filled", "zero_ambiguous", "zero_yield_only"}
    cl = _crop_long().merge(df[KEY], on=KEY)  # independent recount from the source rows that are carried
    zero = (cl["area_1000_ha"] == 0) & (cl["production_1000_tons"] == 0) & (cl["yield_kg_per_ha"] == 0)
    filled = cl["flag_nan_filled_with_zero"].astype(bool)
    assert int((df["yield_value_status"] == "zero_ambiguous").sum()) == int((zero & ~filled).sum())
    assert int((df["yield_value_status"] == "zero_source_filled").sum()) == int((zero & filled).sum())
    # source values are never altered: raw yield still equals crop-long
    m = df.merge(cl[KEY + ["yield_kg_per_ha"]], on=KEY, suffixes=("", "_src"))
    assert np.allclose(m["yield_kg_per_ha"].fillna(-1), m["yield_kg_per_ha_src"].fillna(-1))
    # a labelled ambiguous zero is never a baseline observation
    amb = df[df["yield_value_status"] == "zero_ambiguous"]
    assert not amb["eligible_for_baseline"].astype(bool).any() and amb["yield_anomaly_kg_per_ha"].isna().all()


def test_zscore_confidence_flag_and_valid_zscore_flag(df):
    has = df["yield_anomaly_zscore"].notna()
    assert (df["has_valid_yield_zscore"] == has).all()
    flagged = df[df["yield_zscore_low_confidence"]]
    assert (flagged["training_observation_count"] < 3).all() and flagged["has_valid_yield_zscore"].all()
    unflagged = df[has & ~df["yield_zscore_low_confidence"]]
    assert (unflagged["training_observation_count"] >= 3).all()
    # the unstable z-scores are exactly the ones built on the smallest baseline
    assert flagged["year"].nunique() == 1


def test_empty_columns_are_dropped_and_constant_weather_features_are_flagged_unusable(df):
    assert not [c for c in df.columns if df[c].isna().all() and not c.startswith(("wanom__", "wz__", "wact__"))]
    assert "aggregate_member_sum_area" not in df.columns and "aggregate_member_sum_production" not in df.columns
    dd = pd.read_csv(os.path.join(OUT, "ml_ready_column_dictionary_v1.csv"))
    assert {"usable_as_feature", "unusable_reason", "availability_note"} <= set(dd.columns)
    dy = df.drop_duplicates(["district_code", "year"])
    for c in [c for c in df.columns if c.startswith(("wanom__", "wz__", "wact__"))]:
        dead = dy[c].isna().all() or dy[c].dropna().nunique() <= 1
        row = dd[dd["column"] == c].iloc[0]
        assert bool(row["usable_as_feature"]) == (not dead), c
        if dead:
            assert isinstance(row["unusable_reason"], str) and row["unusable_reason"]
    assert not dd.loc[~dd["baseline_leakage_safe"], "usable_as_feature"].any()


def test_list_separators_are_semicolons_everywhere(df):
    for c in ["training_years", "fc_selection_train_years", "fc_selected_predictors", "flag_summary"]:
        assert not df[c].dropna().astype(str).str.contains(",").any(), c


def test_loyo_weather_companion_covers_every_year_and_matches_the_source():
    lo = pd.read_csv(os.path.join(OUT, "weather_loyo_anomalies_v1.csv"))
    main = pd.read_csv(DATA)
    assert not lo.duplicated(DY).any()
    assert set(map(tuple, lo[DY].values)) == set(map(tuple, main[DY].drop_duplicates().values))
    first_two = sorted(main["year"].unique())[:2]
    assert lo[lo["year"].isin(first_two)].filter(like="wloyo__").notna().any(axis=None)  # the years the main table lacks
    src = pd.read_csv(DEFAULTS["weather_loyo"])
    src = src[src["is_validation_year"].astype(bool)].dropna(subset=["anomaly"]).set_index(DY + ["weather_column"])["anomaly"]
    sample = lo.melt(id_vars=DY, var_name="f", value_name="v").dropna()
    sample["f"] = sample["f"].str.replace("wloyo__", "", regex=False)
    src.index = src.index.set_names(DY + ["f"])
    j = sample.set_index(DY + ["f"]).join(src.rename("src"), how="left")
    assert len(j) == len(sample)
    assert j["src"].notna().all() and np.allclose(j["v"], j["src"])


def test_missing_data_report_is_complete_and_reconciles():
    rep = pd.read_csv(os.path.join(OUT, "missing_data_report_v1.csv"))
    assert {"category", "item", "count", "reason", "by_design", "where_to_look"} <= set(rep.columns)
    assert set(rep["category"]) == {"rows", "columns"}
    assert rep["reason"].notna().all() and (rep["reason"].str.len() > 0).all()
    ex = pd.read_csv(os.path.join(OUT, "excluded_crop_rows_v1.csv"))
    assert int(rep.loc[rep["item"].str.startswith("crop-long rows of"), "count"].sum()) == len(ex)
    amb = rep[rep["item"].str.contains("zero_ambiguous")]
    assert len(amb) == 1 and "AMBIGUOUS" in amb["by_design"].iloc[0]
    readme = open(os.path.join(OUT, "README.md"), encoding="utf-8").read()
    assert "COMPLETENESS_START" in readme and "zero_ambiguous" in readme and "excluded_crop_rows_v1.csv" in readme


def test_row_accounting_still_holds_when_eligibility_changes(tmp_path):
    """Robustness: exclude a crop via the eligibility input and every crop-long row must still be accounted for."""
    cl = _crop_long()
    elig = pd.read_csv(DEFAULTS["eligibility"])
    bad = elig.copy()
    bad.loc[bad["crop"] == "maize", "proceed_to_modelling"] = False  # maize is now excluded -> row counts change
    p = tmp_path / "elig_bad.csv"
    bad.to_csv(p, index=False)
    paths = DEFAULTS_PATHS()
    paths["eligibility"] = str(p)
    out_df, cf = build(paths, str(tmp_path / "o"), "pytest")
    assert len(out_df) + len(pd.read_csv(tmp_path / "o" / "excluded_crop_rows_v1.csv")) == len(cl)
    assert "maize" not in set(out_df["crop"])
