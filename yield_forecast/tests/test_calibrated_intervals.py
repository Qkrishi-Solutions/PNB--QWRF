"""
test_calibrated_intervals.py - tests for the calibrated-interval experiment (yield_forecast/experiment_calibrated_intervals.py).

Run from the repo root:  python -m pytest yield_forecast/tests/test_calibrated_intervals.py -v
"""
import numpy as np
import pandas as pd
import pytest

import yield_forecast.experiment_calibrated_intervals as ci
from yield_forecast.tests.test_multi_year import light_config, make_dataset

EXP = "yield_forecast/results/experiments/"


# ------------------------------------------------------------------ the conformal quantile

def test_conformal_quantile_matches_the_definition():
    assert ci.conformal_quantile(np.arange(1, 11), alpha=0.2) == 9      # ceil(11 * 0.8) = 9th smallest
    assert ci.conformal_quantile(np.arange(1, 11), alpha=0.1) == 10     # ceil(11 * 0.9) = 10th smallest
    assert ci.conformal_quantile([3, 1, 2], alpha=0.1) == 3             # n too small: falls back to the maximum
    assert ci.conformal_quantile([5.0], alpha=0.2) == 5.0
    with pytest.raises(ValueError):
        ci.conformal_quantile([], alpha=0.2)


def test_conformal_quantile_gives_finite_sample_coverage_on_exchangeable_data():
    """With exchangeable scores the conformal quantile covers at least 1-alpha on average (the method's guarantee)."""
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(400):
        cal = rng.normal(size=60)
        q = ci.conformal_quantile(np.abs(cal), alpha=0.2)
        hits.append(np.abs(rng.normal(size=200)) <= q)
    assert np.mean(hits) >= 0.80 - 0.01


# ------------------------------------------------------------------ reproduces the pipeline, and is leak-free

def test_current_variant_reproduces_the_committed_pipeline_coverage():
    summ = pd.read_csv(EXP + "calibrated_intervals_summary_v1.csv")
    cur = summ[summ["variant"] == "current_in_sample_abs"].set_index("model")["coverage"]
    preds = pd.read_csv("yield_forecast/results/predictions_forward_chaining_v1.csv")
    committed = preds.groupby("model")["pi_covered_80_in_sample"].mean()
    for m in cur.index:
        assert cur[m] == pytest.approx(committed[m], abs=1e-9), m


def test_intervals_do_not_depend_on_the_test_year_targets(tmp_path):
    """Calibration uses training rows only: changing every test-year target must leave all interval bounds unchanged."""
    df = make_dataset(range(2012, 2020), seed=11)
    cfg = light_config(tmp_path, {"method": "forward_chaining", "mode": "fixed_origin",
                                  "train_end_year": 2016, "min_training_rows": 8})
    a_path = tmp_path / "a.csv"
    df.to_csv(a_path, index=False)
    b = df.copy()
    test_rows = b["year"] > 2016
    b.loc[test_rows & b["yield_anomaly_kg_per_ha"].notna(), "yield_anomaly_kg_per_ha"] *= -5.0
    b_path = tmp_path / "b.csv"
    b.to_csv(b_path, index=False)
    ra, _ = ci.run(str(a_path), cfg)
    rb, _ = ci.run(str(b_path), cfg)
    key = ["fold", "district_code", "crop", "model", "variant"]
    ra, rb = ra.sort_values(key).reset_index(drop=True), rb.sort_values(key).reset_index(drop=True)
    assert np.allclose(ra[["pred", "lo", "hi"]].values, rb[["pred", "lo", "hi"]].values)
    assert (ra["covered"] != rb["covered"]).any()   # the targets did change, so coverage differs: the test can fail


# ------------------------------------------------------------------ structure of the outputs

def test_outputs_have_the_expected_structure():
    rows = pd.read_csv(EXP + "calibrated_intervals_rows_v1.csv")
    assert set(rows["variant"]) == set(ci.VARIANTS) and set(rows["model"]) == set(ci.ALL_MODELS)
    assert (rows["hi"] >= rows["lo"]).all() and rows[["lo", "hi", "pred"]].notna().all().all()
    assert (rows["covered"] == ((rows["lo"] <= rows["actual"]) & (rows["actual"] <= rows["hi"]))).all()
    summ = pd.read_csv(EXP + "calibrated_intervals_summary_v1.csv")
    assert summ["coverage"].between(0, 1).all() and (summ["interval_score_scaled"] > 0).all()


def test_absolute_variants_share_one_width_and_scaled_variants_have_per_crop_widths():
    rows = pd.read_csv(EXP + "calibrated_intervals_rows_v1.csv")
    for (v, m, f), g in rows[rows["variant"].isin(["current_in_sample_abs", "oof_abs"])].groupby(["variant", "model", "fold"]):
        assert g["width_kg"].round(6).nunique() == 1, (v, m, f)           # one kg/ha width for every crop
    for (v, m, f), g in rows[rows["variant"].isin(["in_sample_scaled", "oof_scaled"])].groupby(["variant", "model", "fold"]):
        assert g["width_scaled"].round(6).nunique() == 1, (v, m, f)       # one width in crop-std units ...
        assert g["width_kg"].round(6).nunique() > 1                       # ... which is a different kg/ha width per crop


# ------------------------------------------------------------------ the pipeline's default interval

def test_pipeline_default_interval_equals_the_experiments_calibrated_variant():
    """One implementation: the pipeline's calibrated bounds equal the experiment's oof_scaled bounds row for row."""
    preds = pd.read_csv("yield_forecast/results/predictions_forward_chaining_v1.csv")
    ex = pd.read_csv(EXP + "calibrated_intervals_rows_v1.csv")
    ex = ex[ex["variant"] == "oof_scaled"]
    key = ["fold", "district_code", "crop", "model"]
    m = preds.merge(ex[key + ["lo", "hi"]], on=key)
    assert len(m) == len(preds)
    assert np.allclose(m["pi_lower_80"], m["lo"]) and np.allclose(m["pi_upper_80"], m["hi"])


def test_pipeline_reports_per_crop_interval_coverage_and_scores():
    crop = pd.read_csv("yield_forecast/results/crop_level_evaluation_metrics_v1.csv")
    assert {"pi_coverage_80", "pi_coverage_80_in_sample"} <= set(crop.columns)
    agg = pd.read_csv("yield_forecast/results/aggregate_model_comparison_v1.csv")
    assert {"empirical_pi_coverage_80", "empirical_pi_coverage_80_in_sample", "interval_score_scaled",
            "interval_score_scaled_in_sample"} <= set(agg.columns)
    assert agg["empirical_pi_coverage_80"].between(0, 1).all()


def _run_pipeline(tmp_path, data, method, name):
    from yield_forecast.train_forecast import run_training_pipeline
    import json
    cfg_path = light_config(tmp_path, {"method": "forward_chaining", "mode": "fixed_origin",
                                      "train_end_year": 2016, "min_training_rows": 8})
    cfg = json.loads(open(cfg_path, encoding="utf-8").read())
    cfg["prediction_intervals"]["method"] = method
    open(cfg_path, "w", encoding="utf-8").write(json.dumps(cfg))
    data_path = tmp_path / f"{name}.csv"
    data.to_csv(data_path, index=False)
    return run_training_pipeline(dataset_path=str(data_path), config_path=cfg_path, results_dir=str(tmp_path / name))


def test_pipeline_legacy_method_reproduces_the_in_sample_interval(tmp_path):
    res = _run_pipeline(tmp_path, make_dataset(range(2012, 2020), seed=21), "in_sample_residual_quantile", "legacy")
    p = res["predictions"]
    assert (p["pi_method"] == "uncalibrated_in_sample_exploratory").all()
    assert np.allclose(p["pi_lower_80"], p["pi_lower_80_in_sample"]) and np.allclose(p["pi_upper_80"], p["pi_upper_80_in_sample"])


def test_pipeline_calibrated_intervals_ignore_test_year_targets(tmp_path):
    """Pipeline-level leakage test: changing every test-year target leaves the calibrated bounds unchanged."""
    df = make_dataset(range(2012, 2020), seed=22)
    b = df.copy()
    rows = (b["year"] > 2016) & b["yield_anomaly_kg_per_ha"].notna()
    b.loc[rows, "yield_anomaly_kg_per_ha"] *= -4.0
    ra = _run_pipeline(tmp_path, df, "oof_conformal_per_crop", "a")["predictions"]
    rb = _run_pipeline(tmp_path, b, "oof_conformal_per_crop", "b")["predictions"]
    key = ["fold", "district_code", "crop", "model"]
    ra, rb = ra.sort_values(key).reset_index(drop=True), rb.sort_values(key).reset_index(drop=True)
    assert (ra["pi_method"] == "oof_conformal_per_crop_exploratory").all()
    assert np.allclose(ra[["pi_lower_80", "pi_upper_80"]].values, rb[["pi_lower_80", "pi_upper_80"]].values)
    assert (ra["pi_covered_80"] != rb["pi_covered_80"]).any()          # targets did change: the test can fail
