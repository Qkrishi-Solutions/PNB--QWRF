#!/usr/bin/env python3
"""
experiment_calibrated_intervals.py - Issue #50 follow-up: calibrated 80% prediction intervals (EXPLORATORY).

The pipeline's current intervals are  prediction +/- q  with q the 80th percentile of the absolute
IN-SAMPLE training residuals (kg/ha, one q for all crops). They undercover (about 70-75% against 80%) because
in-sample residuals are smaller than out-of-sample errors, and a single kg/ha width is dominated by sugarcane.

Four versions are compared on the SAME folds, features and models as the pipeline (nothing is re-tuned):

  current_in_sample_abs   the pipeline's rule (reproduced exactly; checked against the committed coverage)
  in_sample_scaled        in-sample residuals, but normalised by each crop's training std (per-crop widths)
  oof_abs                 split-conformal on OUT-OF-FOLD residuals in kg/ha
  oof_scaled              split-conformal on OUT-OF-FOLD residuals normalised by crop std (per-crop widths)

Out-of-fold residuals come from leave-one-district-out cross-validation INSIDE the training rows only
(feature selection is repeated inside each inner fold). The test year is never used to calibrate.
Conformal quantile: the ceil((n+1)(1-alpha))/n empirical quantile of the calibration scores.

What this can and cannot fix: leave-one-district-out inside one training year measures error on NEW DISTRICTS in
the SAME year. A new TEST YEAR adds a year-level shift that this cannot see, so some undercoverage is expected
and is reported (test_to_oof_error_ratio). With two test years and 13 districts the evidence is thin.

Outputs (yield_forecast/results/experiments/):
  calibrated_intervals_summary_v1.csv   coverage, width and gap by model and variant
  calibrated_intervals_by_crop_v1.csv   coverage by crop (the per-crop problem)
  calibrated_intervals_by_fold_v1.csv   coverage and the test-to-calibration error ratio per fold
  calibrated_intervals_rows_v1.csv      every test row with each variant's interval
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from yield_forecast.folds import generate_folds, validate_folds
from yield_forecast.train_forecast import (  # one shared implementation with the main pipeline
    MODEL_NAMES as MODELS,
    conformal_quantile,
    fit_crop_scales,
    oof_abs_residuals,
    predict_model,
    select_features_in_fold,
)

ALPHA = 0.2                      # 80% intervals, as in the pipeline
VARIANTS = ["current_in_sample_abs", "in_sample_scaled", "oof_abs", "oof_scaled"]
ALL_MODELS = ["Climatology"] + MODELS


def run(dataset, config, alpha=ALPHA):
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    target, fs = cfg["target_column"], cfg["feature_selection"]
    raw = pd.read_csv(dataset)
    df = raw[raw["usable_for_forward_chaining_eval"] == True].copy()  # noqa: E712
    weather = [c for c in df.columns if c.startswith("wanom__")]
    folds = generate_folds(cfg["validation"], sorted(df["year"].unique()), sorted(raw["year"].unique()))
    validate_folds(folds, df, int(cfg["validation"].get("min_training_rows", 8)))

    rows, fold_diag = [], []
    for fold in folds:
        tr = df[df["year"].isin(fold["train_years"])].copy()
        te = df[df["year"] == fold["test_year"]].copy().reset_index(drop=True)
        avail = [c for c in weather if tr[c].notna().all() and te[c].notna().all()]
        feats = select_features_in_fold(tr, avail, target, fs["max_features"], fs["max_collinearity"],
                                        fs["min_observed_fraction"])
        crop_scale, gstd = fit_crop_scales(tr, target)
        s_tr = tr["crop"].map(crop_scale).fillna(gstd).values
        s_te = te["crop"].map(crop_scale).fillna(gstd).values
        oof = oof_abs_residuals(tr, avail, cfg, target)
        y_te = te[target].values

        for m in ALL_MODELS:
            pred, pred_tr = predict_model(m, tr, te, feats, cfg, target, s_tr, s_te)
            ins = np.abs(tr[target].values - pred_tr)           # in-sample absolute residuals (kg/ha)
            o = oof[m]                                          # out-of-fold absolute residuals (kg/ha)
            half = {
                "current_in_sample_abs": np.full(len(te), np.quantile(ins, 1 - alpha)),
                "in_sample_scaled": np.quantile(ins / s_tr, 1 - alpha) * s_te,
                "oof_abs": np.full(len(te), conformal_quantile(o, alpha)),
                "oof_scaled": conformal_quantile(o / s_tr, alpha) * s_te,
            }
            for v, h in half.items():
                lo, hi = pred - h, pred + h
                for i in range(len(te)):
                    rows.append({"fold": fold["fold"], "test_year": fold["test_year"],
                                 "district_code": te.loc[i, "district_code"], "crop": te.loc[i, "crop"], "model": m,
                                 "variant": v, "actual": y_te[i], "pred": pred[i], "lo": lo[i], "hi": hi[i],
                                 "covered": bool(lo[i] <= y_te[i] <= hi[i]), "width_kg": hi[i] - lo[i],
                                 "width_scaled": (hi[i] - lo[i]) / s_te[i]})
            fold_diag.append({"fold": fold["fold"], "test_year": fold["test_year"], "model": m,
                              "n_calibration_oof": int(len(o)),
                              "oof_scaled_mae": float(np.mean(o / s_tr)),
                              "in_sample_scaled_mae": float(np.mean(ins / s_tr)),
                              "test_scaled_mae": float(np.mean(np.abs(y_te - pred) / s_te)),
                              "test_to_oof_error_ratio": float(np.mean(np.abs(y_te - pred) / s_te) / np.mean(o / s_tr)),
                              "test_to_in_sample_error_ratio": float(np.mean(np.abs(y_te - pred) / s_te) / np.mean(ins / s_tr))})
    return pd.DataFrame(rows), pd.DataFrame(fold_diag)


def summarise(r):
    out = []
    for (m, v), g in r.groupby(["model", "variant"], sort=False):
        by_crop = g.groupby("crop")["covered"].mean()
        sc = g["width_kg"] / g["width_scaled"]                      # the crop std used for this row
        miss = (2 / ALPHA) * (np.maximum(g["lo"] - g["actual"], 0) + np.maximum(g["actual"] - g["hi"], 0)) / sc
        row = {"interval_score_scaled": float((g["width_scaled"] + miss).mean()),"model": m, "variant": v, "n": len(g), "coverage": g["covered"].mean(),
               "gap_vs_80": g["covered"].mean() - 0.8, "mean_width_scaled": g["width_scaled"].mean(),
               "mean_width_kg": g["width_kg"].mean(),
               "crop_coverage_min": by_crop.min(), "crop_coverage_max": by_crop.max(),
               "crops_within_10pts_of_80": int(((by_crop - 0.8).abs() <= 0.10).sum()), "n_crops": int(len(by_crop))}
        for fo, gf in g.groupby("fold"):
            row[f"coverage_fold{fo}"] = gf["covered"].mean()
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml_ready_dataset/ml_ready_dataset_v1.csv")
    ap.add_argument("--config", default="yield_forecast/models/model_config.json")
    ap.add_argument("--outdir", default="yield_forecast/results/experiments")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    rows, diag = run(a.dataset, a.config)
    summ = summarise(rows)
    by_crop = rows.groupby(["model", "variant", "crop"], sort=False)["covered"].agg(["mean", "size"]).reset_index().rename(
        columns={"mean": "coverage", "size": "n"})
    rows.to_csv(os.path.join(a.outdir, "calibrated_intervals_rows_v1.csv"), index=False)
    summ.to_csv(os.path.join(a.outdir, "calibrated_intervals_summary_v1.csv"), index=False)
    by_crop.to_csv(os.path.join(a.outdir, "calibrated_intervals_by_crop_v1.csv"), index=False)
    diag.to_csv(os.path.join(a.outdir, "calibrated_intervals_by_fold_v1.csv"), index=False)

    pd.set_option("display.width", 250)
    print("\nCALIBRATED 80% INTERVALS: coverage on the 256 test rows (target 0.80)\n")
    pv = summ.pivot(index="model", columns="variant", values="coverage")[VARIANTS].loc[ALL_MODELS]
    print(pv.round(3).to_string())
    print("\nMEAN INTERVAL WIDTH in crop-std units (narrower = sharper at the same coverage)\n")
    print(summ.pivot(index="model", columns="variant", values="mean_width_scaled")[VARIANTS].loc[ALL_MODELS].round(2).to_string())
    print("\nINTERVAL SCORE in crop-std units (Winkler, alpha 0.2; LOWER IS BETTER; rewards narrow AND correct)\n")
    print(summ.pivot(index="model", columns="variant", values="interval_score_scaled")[VARIANTS].loc[ALL_MODELS].round(2).to_string())
    print("\nCOVERAGE BY FOLD (oof_scaled | current_in_sample_abs), target 0.80\n")
    for mdl in ALL_MODELS:
        r = summ[summ.model == mdl].set_index("variant")
        print(f"  {mdl:13s} fold1 {r.loc['oof_scaled','coverage_fold1']:.3f} | {r.loc['current_in_sample_abs','coverage_fold1']:.3f}"
              f"    fold2 {r.loc['oof_scaled','coverage_fold2']:.3f} | {r.loc['current_in_sample_abs','coverage_fold2']:.3f}")
    print("\nWORST-CROP COVERAGE (min over the 11 crops)\n")
    print(summ.pivot(index="model", columns="variant", values="crop_coverage_min")[VARIANTS].loc[ALL_MODELS].round(2).to_string())
    print("\nCROPS WITHIN 10 POINTS OF 80% (of 11)\n")
    print(summ.pivot(index="model", columns="variant", values="crops_within_10pts_of_80")[VARIANTS].loc[ALL_MODELS].to_string())
    print("\nWHY SOME UNDERCOVERAGE REMAINS: test error vs calibration error (ratio > 1 = test is harder than calibration)\n")
    print(diag.pivot(index="model", columns="fold", values="test_to_oof_error_ratio").loc[ALL_MODELS].round(2).to_string())
