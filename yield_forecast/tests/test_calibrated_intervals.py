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
    committed = preds.groupby("model")["pi_covered_80"].mean()
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
