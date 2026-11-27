"""
test_multi_year.py - proves the Issue #50 pipeline is not tied to the five years 2015-2019.

Part 1: the fold generator (expanding, fixed_origin, explicit) on plain year lists, including error cases.
Part 2: the whole pipeline on a SYNTHETIC 12-year dataset (2008-2019) with a planted weather signal. It must
        build the folds from the data, train only on earlier years, run all models and recover the signal.
Part 3: the five-year experiment's nested anomalies on that longer synthetic history.

All data here is synthetic and is generated in the tests. Run from the repo root:
    python -m pytest yield_forecast/tests/test_multi_year.py -v
"""
import json

import numpy as np
import pandas as pd
import pytest

from yield_forecast.folds import generate_folds, validate_folds
from yield_forecast.train_forecast import run_training_pipeline
import yield_forecast.experiment_five_year_loyo as ex

REAL_CONFIG = "yield_forecast/models/model_config.json"


# ------------------------------------------------------------------ synthetic multi-year data

def make_dataset(years, n_districts=6, seed=0, signal=2.0, noise=0.5, n_weather=6):
    """Dataset shaped like ml_ready_dataset_v1.csv (only the columns the pipeline reads).

    Weather feature w0 drives the yield anomaly (planted signal); w1..w5 are pure noise. Forward-chaining
    anomalies need two earlier years, exactly as in the real data, so the first two years have no target.
    """
    rng = np.random.default_rng(seed)
    crops = {"alpha": 1.0, "beta": 10.0, "gamma": 100.0}  # very different yield scales on purpose
    zones = ["Climate Zone 1", "Climate Zone 2", "Climate Zone 3"]
    years = list(years)
    wnames = [f"w{j}" for j in range(n_weather)]

    weather = {}  # (district, year) -> array of raw values
    for d in range(n_districts):
        for y in years:
            weather[(d, y)] = rng.normal(0, 1, n_weather)

    rows = []
    for d in range(n_districts):
        for crop, scale in crops.items():
            level = 1000.0 * scale
            ylds = {y: level + scale * (signal * weather[(d, y)][0] + noise * rng.normal()) for y in years}
            for i, y in enumerate(years):
                rec = {"district_code": 500 + d, "district_name": f"D{d}", "crop": crop, "year": y,
                       "climate_zone": zones[d % 3], "yield_kg_per_ha": ylds[y], "eligible_for_baseline": True}
                prior_y = [ylds[v] for v in years[:i]]
                rec["yield_anomaly_kg_per_ha"] = ylds[y] - np.mean(prior_y) if len(prior_y) >= 2 else np.nan
                rec["usable_for_forward_chaining_eval"] = len(prior_y) >= 2
                prior_w = np.array([weather[(d, v)] for v in years[:i]])
                for j, nm in enumerate(wnames):
                    rec[f"wact__{nm}"] = weather[(d, y)][j]
                    rec[f"wanom__{nm}"] = (weather[(d, y)][j] - prior_w[:, j].mean()) if len(prior_w) >= 2 else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def light_config(tmp_path, validation):
    """The real config with small models (fast tests) and the requested validation block."""
    with open(REAL_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["validation"] = validation
    cfg["models"]["RandomForest"]["n_estimators"] = 10
    cfg["models"]["XGBoost"]["n_estimators"] = 10
    cfg["models"]["CatBoost"]["iterations"] = 10
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ part 1: fold generator

ALL = list(range(2000, 2020))              # 20 data years
WITH_TARGETS = ALL[2:]                      # targets start in the third year


def test_expanding_builds_one_fold_per_usable_year_and_never_leaks():
    folds = generate_folds({"mode": "expanding"}, WITH_TARGETS, ALL)
    assert [f["test_year"] for f in folds] == WITH_TARGETS[1:]          # first usable year is training only
    for f in folds:
        assert f["train_years"] == [y for y in ALL if y < f["test_year"]]
        assert max(f["train_years"]) < f["test_year"]
    assert [f["fold"] for f in folds] == list(range(1, len(folds) + 1))


def test_expanding_respects_min_train_years_and_test_window():
    folds = generate_folds({"mode": "expanding", "min_train_years_with_targets": 3,
                            "first_test_year": 2008, "last_test_year": 2012}, WITH_TARGETS, ALL)
    assert [f["test_year"] for f in folds] == [2008, 2009, 2010, 2011, 2012]


def test_fixed_origin_shares_one_training_window_and_tests_every_later_year():
    folds = generate_folds({"mode": "fixed_origin", "train_end_year": 2009}, WITH_TARGETS, ALL)
    assert [f["test_year"] for f in folds] == list(range(2010, 2020))
    assert all(f["train_years"] == list(range(2000, 2010)) for f in folds)


def test_explicit_mode_returns_the_listed_folds_and_old_configs_still_work():
    listed = [{"fold": 1, "train_years": [2017], "test_year": 2018}, {"fold": 2, "train_years": [2017, 2018], "test_year": 2019}]
    assert generate_folds({"mode": "explicit", "folds": listed}, [2017, 2018, 2019], [2015, 2016, 2017, 2018, 2019]) == listed
    # a config written before modes existed (a bare "folds" list) is treated as explicit
    assert generate_folds({"folds": listed}, [2017, 2018, 2019], [2015, 2016, 2017, 2018, 2019]) == listed


def test_fold_generator_rejects_bad_configurations():
    with pytest.raises(ValueError):
        generate_folds({"mode": "bogus"}, WITH_TARGETS, ALL)
    with pytest.raises(ValueError):
        generate_folds({"mode": "fixed_origin"}, WITH_TARGETS, ALL)                       # no train_end_year
    with pytest.raises(ValueError):
        generate_folds({"mode": "fixed_origin", "train_end_year": 2019}, WITH_TARGETS, ALL)  # nothing to test
    with pytest.raises(ValueError):
        generate_folds({"mode": "expanding"}, [], ALL)                                    # no targets at all
    with pytest.raises(ValueError):
        generate_folds({"mode": "expanding", "min_train_years_with_targets": 99}, WITH_TARGETS, ALL)


def test_validate_folds_catches_leakage_and_thin_data():
    df = pd.DataFrame({"year": [2017] * 10 + [2018] * 10 + [2019] * 10})
    good = [{"fold": 1, "train_years": [2017], "test_year": 2018}]
    assert validate_folds(good, df, 8)[0]["n_train"] == 10
    with pytest.raises(ValueError, match="leaks"):
        validate_folds([{"fold": 1, "train_years": [2017, 2018], "test_year": 2018}], df, 8)
    with pytest.raises(ValueError, match="training rows"):
        validate_folds(good, df, 11)
    with pytest.raises(ValueError, match="no rows"):
        validate_folds([{"fold": 1, "train_years": [2017], "test_year": 2025}], df, 8)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_folds(good + [{"fold": 2, "train_years": [2017], "test_year": 2018}], df, 8)


def test_real_data_expanding_mode_is_the_standard_two_fold_forward_chaining():
    """For our 2015-2019 data, 'expanding' gives: train 2015-2017 -> 2018, then train 2015-2018 -> 2019."""
    df = pd.read_csv("ml_ready_dataset/ml_ready_dataset_v1.csv")
    usable = df[df["usable_for_forward_chaining_eval"]]
    folds = generate_folds({"mode": "expanding"}, sorted(usable["year"].unique()), sorted(df["year"].unique()))
    assert [(f["train_years"], f["test_year"]) for f in folds] == [([2015, 2016, 2017], 2018), ([2015, 2016, 2017, 2018], 2019)]


# ------------------------------------------------------------------ part 2: the whole pipeline on 12 years

@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("multi")
    df = make_dataset(range(2008, 2020))
    data_path = tmp / "synthetic_ml_ready.csv"
    df.to_csv(data_path, index=False)
    cfg_path = light_config(tmp, {"method": "forward_chaining", "mode": "expanding", "min_training_rows": 8})
    out = tmp / "results"
    res = run_training_pipeline(dataset_path=str(data_path), config_path=cfg_path, results_dir=str(out))
    return df, res, tmp


def test_pipeline_builds_the_expected_number_of_folds_from_the_data(synthetic_run):
    df, res, _ = synthetic_run
    usable_years = sorted(df.loc[df["usable_for_forward_chaining_eval"], "year"].unique())   # 2010..2019
    expected_tests = usable_years[1:]                                                      # 2011..2019
    fm = res["fold_metrics"]
    assert sorted(fm["test_year"].unique()) == expected_tests
    assert set(fm["model"]) == {"Climatology", "Ridge", "RandomForest", "XGBoost", "CatBoost"}


def test_every_synthetic_fold_trains_only_on_earlier_years(synthetic_run):
    _, res, _ = synthetic_run
    for _, r in res["fold_metrics"].iterrows():
        used = [int(y) for y in str(r["train_years_with_targets"]).split(";")]
        assert max(used) < r["test_year"]
        assert r["n_train"] >= 8


def test_synthetic_prediction_table_is_complete(synthetic_run):
    df, res, _ = synthetic_run
    preds = res["predictions"]
    per_year = df[df["usable_for_forward_chaining_eval"]].groupby("year").size()
    expected = sum(int(per_year[y]) for y in sorted(preds["test_year"].unique())) * 5   # five models per test row
    assert len(preds) == expected
    assert preds["predicted_yield_anomaly"].notna().all()
    assert np.isfinite(preds["scaled_abs_error"]).all()


def test_pipeline_recovers_a_planted_weather_signal(synthetic_run):
    """With a strong real signal the models must beat the climatology baseline clearly: the pipeline learns."""
    _, res, _ = synthetic_run
    agg = res["aggregate_comparison"].set_index("model")
    base = agg.loc["Climatology", "scaled_mae"]
    assert agg.loc["Ridge", "scaled_mae"] < 0.7 * base
    assert agg.loc["Ridge", "statistically_significant_improvement"] == "YES"


def test_per_crop_table_and_flags_work_on_a_long_history(synthetic_run):
    _, res, _ = synthetic_run
    crop = res["crop_metrics"]
    assert {"low_sample_flag", "skill_vs_climatology_pct", "min_fold_n_test"} <= set(crop.columns)
    assert set(crop["crop"]) == {"alpha", "beta", "gamma"}


def test_multi_year_run_is_deterministic(synthetic_run, tmp_path):
    df, res, tmp = synthetic_run
    cfg_path = light_config(tmp_path, {"method": "forward_chaining", "mode": "expanding", "min_training_rows": 8})
    data_path = tmp_path / "again.csv"
    df.to_csv(data_path, index=False)
    again = run_training_pipeline(dataset_path=str(data_path), config_path=cfg_path, results_dir=str(tmp_path / "out"))
    pd.testing.assert_frame_equal(res["predictions"], again["predictions"])


def test_fixed_origin_mode_also_runs_end_to_end(tmp_path):
    df = make_dataset(range(2008, 2020), seed=3)
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)
    cfg_path = light_config(tmp_path, {"method": "forward_chaining", "mode": "fixed_origin",
                                       "train_end_year": 2013, "min_training_rows": 8})
    res = run_training_pipeline(dataset_path=str(data_path), config_path=cfg_path, results_dir=str(tmp_path / "o"))
    fm = res["fold_metrics"]
    assert sorted(fm["test_year"].unique()) == list(range(2014, 2020))
    assert (fm["train_years"] == ";".join(str(y) for y in range(2008, 2014))).all()


def test_pipeline_fails_clearly_when_training_data_is_too_thin(tmp_path):
    df = make_dataset(range(2008, 2020), seed=4)
    data_path = tmp_path / "d.csv"
    df.to_csv(data_path, index=False)
    cfg_path = light_config(tmp_path, {"method": "forward_chaining", "mode": "expanding", "min_training_rows": 10_000})
    with pytest.raises(ValueError, match="training rows"):
        run_training_pipeline(dataset_path=str(data_path), config_path=cfg_path, results_dir=str(tmp_path / "o"))


# ------------------------------------------------------------------ part 3: the five-year experiment on 12 years

def test_nested_leave_one_year_out_works_on_a_twelve_year_history(tmp_path):
    df = make_dataset(range(2008, 2020), seed=5)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    keys, Y, dists, W, wcols, years = ex.load_arrays(str(path))
    assert years == list(range(2008, 2020)) and Y.shape[1] == 12
    t_idx = 6
    tr_a, te_a, feats = ex.fold_frames(keys, Y, dists, W, wcols, years, t_idx)
    Y2, W2 = Y.copy(), W.copy()
    Y2[:, t_idx] = Y2[:, t_idx] * 9.0 + 77.0
    W2[:, t_idx, :] = W2[:, t_idx, :] * -3.0 + 5.0
    tr_b, te_b, _ = ex.fold_frames(keys, Y2, dists, W2, wcols, years, t_idx)
    cols = ["district_code", "crop", "year", ex.TARGET] + feats
    pd.testing.assert_frame_equal(tr_a[cols].reset_index(drop=True), tr_b[cols].reset_index(drop=True))
    assert not np.allclose(te_a[ex.TARGET].values, te_b[ex.TARGET].values, equal_nan=True)
