#!/usr/bin/env python3
"""
experiment_scaled_shrinkage.py - Issue #50 follow-up experiment (exploratory).

Tests two changes against the validated v1 pipeline, using the same forward-chaining
folds, in-fold feature selection (no imputation) and model settings:

  A. Target scaling: models are fitted on yield anomaly divided by each crop's
     training-fold standard deviation, and predictions are converted back to kg/ha.
  B. Shrinkage blend: prediction = w * model + (1 - w) * climatology.
     The primary weight (w = 0.5) is fixed in advance. Other weights are reported only
     as a post-hoc sensitivity check and must NOT be used to select a weight, because
     only two test years exist.

Output (does not touch the v1 results):
  yield_forecast/results/experiments/scaled_shrinkage_comparison_v1.csv
  yield_forecast/results/experiments/scaled_shrinkage_sensitivity_v1.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from yield_forecast.train_forecast import (
    ClimatologyBaseline,
    classify_performance,
    select_features_in_fold,
)

PRIMARY_W = 0.5
SENSITIVITY_W = [0.25, 0.5, 0.75, 1.0]
MODELS = ["Ridge", "RandomForest", "XGBoost", "CatBoost"]
KEY = ["fold", "district_code", "crop"]


def build_models(cfg):
    m = cfg["models"]
    return {
        "Ridge": Ridge(alpha=m["Ridge"]["alpha"], fit_intercept=m["Ridge"]["fit_intercept"],
                       random_state=m["Ridge"]["random_state"]),
        "RandomForest": RandomForestRegressor(
            n_estimators=m["RandomForest"]["n_estimators"], max_depth=m["RandomForest"]["max_depth"],
            min_samples_leaf=m["RandomForest"]["min_samples_leaf"], max_features=m["RandomForest"]["max_features"],
            n_jobs=m["RandomForest"]["n_jobs"], random_state=m["RandomForest"]["random_state"]),
        "XGBoost": XGBRegressor(
            n_estimators=m["XGBoost"]["n_estimators"], max_depth=m["XGBoost"]["max_depth"],
            learning_rate=m["XGBoost"]["learning_rate"], reg_alpha=m["XGBoost"]["reg_alpha"],
            reg_lambda=m["XGBoost"]["reg_lambda"], subsample=m["XGBoost"]["subsample"],
            colsample_bytree=m["XGBoost"]["colsample_bytree"], n_jobs=m["XGBoost"]["n_jobs"],
            random_state=m["XGBoost"]["random_state"]),
        "CatBoost": CatBoostRegressor(
            iterations=m["CatBoost"]["iterations"], depth=m["CatBoost"]["depth"],
            learning_rate=m["CatBoost"]["learning_rate"], l2_leaf_reg=m["CatBoost"]["l2_leaf_reg"],
            thread_count=m["CatBoost"]["thread_count"], random_seed=m["CatBoost"]["random_seed"],
            verbose=False, allow_writing_files=False),
    }


def run(dataset, config, outdir):
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    target = cfg["target_column"]
    df = pd.read_csv(dataset)
    df = df[df["usable_for_forward_chaining_eval"] == True].copy()  # noqa: E712
    weather = [c for c in df.columns if c.startswith("wanom__")]
    fs = cfg["feature_selection"]

    recs = []
    for fold in cfg["validation"]["folds"]:
        tr = df[df["year"].isin(fold["train_years"])].copy()
        te = df[df["year"] == fold["test_year"]].copy().reset_index(drop=True)

        avail = [c for c in weather if tr[c].notna().all() and te[c].notna().all()]
        sel = select_features_in_fold(tr, avail, target, fs["max_features"],
                                      fs["max_collinearity"], fs["min_observed_fraction"])

        gstd = float(tr[target].std(ddof=0))
        cstd = tr.groupby("crop")[target].std(ddof=0).to_dict()
        scale = {c: (v if v and v > 0 else gstd) for c, v in cstd.items()}
        s_tr = tr["crop"].map(scale).fillna(gstd).values
        s_te = te["crop"].map(scale).fillna(gstd).values

        clim = ClimatologyBaseline().fit(tr, target)
        clim_te = clim.predict(te)
        y_te = te[target].values

        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        cats = ["crop", "climate_zone"]
        sc = StandardScaler()
        X_tr = np.hstack([ohe.fit_transform(tr[cats]), sc.fit_transform(tr[sel])])
        X_te = np.hstack([ohe.transform(te[cats]), sc.transform(te[sel])])
        y_tr_scaled = tr[target].values / s_tr

        def add(model, pred, variant, w):
            for i in range(len(te)):
                recs.append({
                    "fold": fold["fold"], "district_code": te.loc[i, "district_code"],
                    "crop": te.loc[i, "crop"], "model": model, "variant": variant, "w": w,
                    "actual": y_te[i], "pred": pred[i],
                    "scaled_abs_error": abs(y_te[i] - pred[i]) / s_te[i],
                    "abs_error": abs(y_te[i] - pred[i]),
                })

        add("Climatology", clim_te, "baseline", np.nan)
        for name, mdl in build_models(cfg).items():
            mdl.fit(X_tr, y_tr_scaled)
            raw = mdl.predict(X_te) * s_te  # back to kg/ha
            add(name, raw, "scaled_target", 1.0)
            for w in SENSITIVITY_W:
                add(name, w * raw + (1 - w) * clim_te, "scaled_target+shrink", w)
    return pd.DataFrame(recs)


def summarise(p, variant, w):
    base = p[p.model == "Climatology"].sort_values(KEY).reset_index(drop=True)
    rows = []
    for name in ["Climatology"] + MODELS:
        if name == "Climatology":
            g = base
        else:
            g = p[(p.model == name) & (p.variant == variant) & (p.w == w)].sort_values(KEY).reset_index(drop=True)
        assert (g[KEY].values == base[KEY].values).all()
        sdiff = g.scaled_abs_error.mean() - base.scaled_abs_error.mean()
        if name == "Climatology":
            pv = 1.0
        else:
            d = pd.Series(g.scaled_abs_error.values - base.scaled_abs_error.values).groupby(g.district_code.values).mean()
            pv = 1.0 if np.allclose(d.values, 0) else float(wilcoxon(d.values, alternative="two-sided").pvalue)
        sig, verdict = classify_performance(name, sdiff, pv)
        direction = (np.sign(g.pred) == np.sign(g.actual)).mean()
        rows.append({
            "model": name, "variant": variant if name != "Climatology" else "baseline",
            "shrink_w": w if name != "Climatology" else np.nan, "n": len(g),
            "mae_kg_ha": g.abs_error.mean(), "scaled_mae": g.scaled_abs_error.mean(),
            "skill_vs_clim_pct": 100 * (1 - g.scaled_abs_error.mean() / base.scaled_abs_error.mean()),
            "beats_clim_pct": np.nan if name == "Climatology" else 100 * (g.scaled_abs_error < base.scaled_abs_error).mean(),
            "direction_acc_pct": 100 * direction,
            "district_wilcoxon_p": pv, "significantly_better": sig, "verdict": verdict,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml_ready_dataset/ml_ready_dataset_v1.csv")
    ap.add_argument("--config", default="yield_forecast/models/model_config.json")
    ap.add_argument("--outdir", default="yield_forecast/results/experiments")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    p = run(a.dataset, a.config, a.outdir)
    scaled_only = summarise(p, "scaled_target", 1.0)
    primary = summarise(p, "scaled_target+shrink", PRIMARY_W)
    comp = pd.concat([scaled_only, primary[primary.model != "Climatology"]], ignore_index=True)
    comp.to_csv(os.path.join(a.outdir, "scaled_shrinkage_comparison_v1.csv"), index=False)

    sens = pd.concat([summarise(p, "scaled_target+shrink", w).assign(w=w) for w in SENSITIVITY_W], ignore_index=True)
    sens = sens[sens.model != "Climatology"][["model", "w", "scaled_mae", "skill_vs_clim_pct", "beats_clim_pct", "district_wilcoxon_p"]]
    sens.to_csv(os.path.join(a.outdir, "scaled_shrinkage_sensitivity_v1.csv"), index=False)

    pd.set_option("display.width", 250)
    cols = ["model", "variant", "shrink_w", "mae_kg_ha", "scaled_mae", "skill_vs_clim_pct", "beats_clim_pct",
            "direction_acc_pct", "district_wilcoxon_p", "verdict"]
    print("\nPRIMARY RESULTS (shrink weight fixed at 0.5 in advance)\n")
    print(comp[cols].round(3).to_string(index=False))
    print("\nSENSITIVITY (post-hoc, NOT for selecting w; w=1.0 is no shrinkage)\n")
    print(sens.round(3).to_string(index=False))
