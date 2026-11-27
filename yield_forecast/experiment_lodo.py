#!/usr/bin/env python3
"""
experiment_lodo.py - Issue #50 follow-up: forward-chaining leave-one-district-out check.

For every held-out district d and every test year T of the configured folds (2018 and 2019 in the current data):
  - train on the OTHER 12 districts, using only years < T (strict forward chaining)
  - feature selection runs on that training set only; no imputation
  - predict district d in year T

This tests whether the models generalise to unseen districts without letting any later
year (or any row of the held-out district) into training. Compares:
  Climatology (crop mean of training rows), and each model trained on
  (a) the raw anomaly (v1 setup) and (b) the per-crop scaled anomaly (experiment setup).

Output: yield_forecast/results/experiments/lodo_forward_chaining_comparison_v1.csv
        yield_forecast/results/experiments/lodo_forward_chaining_by_year_v1.csv
Exploratory: 13 districts, 2 test years. No causal claim.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from yield_forecast.experiment_scaled_shrinkage import MODELS, build_models
from yield_forecast.folds import generate_folds
from yield_forecast.train_forecast import (
    ClimatologyBaseline,
    classify_performance,
    select_features_in_fold,
)

KEY = ["district_code", "year", "crop"]


def run(dataset, config):
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    target = cfg["target_column"]
    fs = cfg["feature_selection"]
    raw = pd.read_csv(dataset)
    df = raw[raw["usable_for_forward_chaining_eval"] == True].copy()  # noqa: E712
    weather = [c for c in df.columns if c.startswith("wanom__")]
    folds = generate_folds(cfg["validation"], sorted(df["year"].unique()), sorted(raw["year"].unique()))
    test_years = [f["test_year"] for f in folds]

    recs = []
    for district in sorted(df["district_code"].unique()):
        for ty in test_years:
            tr = df[(df["district_code"] != district) & (df["year"] < ty)].copy()
            te = df[(df["district_code"] == district) & (df["year"] == ty)].copy().reset_index(drop=True)
            if te.empty or tr.empty:
                continue

            avail = [c for c in weather if tr[c].notna().all() and te[c].notna().all()]
            sel = select_features_in_fold(tr, avail, target, fs["max_features"],
                                          fs["max_collinearity"], fs["min_observed_fraction"])

            gstd = float(tr[target].std(ddof=0))
            cstd = tr.groupby("crop")[target].std(ddof=0).to_dict()
            scale = {c: (v if v and v > 0 else gstd) for c, v in cstd.items()}
            s_tr = tr["crop"].map(scale).fillna(gstd).values
            s_te = te["crop"].map(scale).fillna(gstd).values

            clim_te = ClimatologyBaseline().fit(tr, target).predict(te)
            y_te = te[target].values

            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            cats = ["crop", "climate_zone"]
            sc = StandardScaler()
            X_tr = np.hstack([ohe.fit_transform(tr[cats]), sc.fit_transform(tr[sel])])
            X_te = np.hstack([ohe.transform(te[cats]), sc.transform(te[sel])])

            def add(model, variant, pred):
                for i in range(len(te)):
                    recs.append({
                        "district_code": district, "year": ty, "crop": te.loc[i, "crop"],
                        "model": model, "variant": variant, "actual": y_te[i], "pred": pred[i],
                        "abs_error": abs(y_te[i] - pred[i]),
                        "scaled_abs_error": abs(y_te[i] - pred[i]) / s_te[i],
                    })

            add("Climatology", "baseline", clim_te)
            for name in MODELS:
                m = build_models(cfg)[name]
                m.fit(X_tr, tr[target].values)
                add(name, "raw_target", m.predict(X_te))
                m = build_models(cfg)[name]
                m.fit(X_tr, tr[target].values / s_tr)
                add(name, "scaled_target", m.predict(X_te) * s_te)
    return pd.DataFrame(recs)


def summarise(p, group_year=None):
    sub = p if group_year is None else p[p.year == group_year]
    base = sub[sub.model == "Climatology"].sort_values(KEY).reset_index(drop=True)
    rows = []
    for variant in ["raw_target", "scaled_target"]:
        for name in MODELS:
            g = sub[(sub.model == name) & (sub.variant == variant)].sort_values(KEY).reset_index(drop=True)
            assert (g[KEY].values == base[KEY].values).all()
            d = pd.Series(g.scaled_abs_error.values - base.scaled_abs_error.values).groupby(g.district_code.values).mean()
            pv = 1.0 if np.allclose(d.values, 0) else float(wilcoxon(d.values, alternative="two-sided").pvalue)
            sdiff = g.scaled_abs_error.mean() - base.scaled_abs_error.mean()
            _, verdict = classify_performance(name, sdiff, pv)
            rows.append({
                "model": name, "variant": variant, "n": len(g),
                "mae_kg_ha": g.abs_error.mean(), "clim_mae_kg_ha": base.abs_error.mean(),
                "scaled_mae": g.scaled_abs_error.mean(), "clim_scaled_mae": base.scaled_abs_error.mean(),
                "skill_vs_clim_pct": 100 * (1 - g.scaled_abs_error.mean() / base.scaled_abs_error.mean()),
                "beats_clim_pct": 100 * (g.scaled_abs_error.values < base.scaled_abs_error.values).mean(),
                "districts_better": int((d.values < 0).sum()), "districts_total": int(len(d)),
                "direction_acc_pct": 100 * (np.sign(g.pred) == np.sign(g.actual)).mean(),
                "district_wilcoxon_p": pv, "verdict": verdict,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml_ready_dataset/ml_ready_dataset_v1.csv")
    ap.add_argument("--config", default="yield_forecast/models/model_config.json")
    ap.add_argument("--outdir", default="yield_forecast/results/experiments")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    p = run(a.dataset, a.config)
    overall = summarise(p)
    overall.to_csv(os.path.join(a.outdir, "lodo_forward_chaining_comparison_v1.csv"), index=False)
    by_year = pd.concat([summarise(p, y).assign(test_year=y) for y in sorted(p.year.unique())], ignore_index=True)
    by_year.to_csv(os.path.join(a.outdir, "lodo_forward_chaining_by_year_v1.csv"), index=False)

    pd.set_option("display.width", 250)
    cols = ["model", "variant", "n", "mae_kg_ha", "clim_mae_kg_ha", "scaled_mae", "clim_scaled_mae",
            "skill_vs_clim_pct", "beats_clim_pct", "districts_better", "direction_acc_pct",
            "district_wilcoxon_p", "verdict"]
    print("\nFORWARD-CHAINING LEAVE-ONE-DISTRICT-OUT (all folds pooled)\n")
    print(overall[cols].round(3).to_string(index=False))
    print("\nBY TEST YEAR (skill vs climatology %, districts better of 13)\n")
    bt = by_year.pivot_table(index=["model", "variant"], columns="test_year",
                             values=["skill_vs_clim_pct", "districts_better"]).round(2)
    print(bt.to_string())
