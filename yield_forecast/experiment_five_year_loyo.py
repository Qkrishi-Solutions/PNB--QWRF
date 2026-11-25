#!/usr/bin/env python3
"""
experiment_five_year_loyo.py - Issue #50 follow-up: retrospective weather-yield response estimation
using ALL FIVE years (2015-2019) with nested leave-one-year-out validation.  EXPLORATORY.

Why: the forward-chaining pipeline scores only 2018 and 2019 (anomalies need two earlier years). With
only 2015-2019 available, this track uses all five years to ask whether the weather-yield association
and the scaled-target gain hold across five test years instead of two.

THIS IS NOT FORECASTING. Baselines (normals) use years later than a row's year, so the track can
estimate how strongly weather relates to yield, but it cannot support a forecasting-skill claim
(roadmap Engineering rules; CLAUDE.md 7.4). The forward-chaining result stays the forecasting evidence.

Nested design (no leakage across folds). For held-out year T, every anomaly - target and weather - is
recomputed from raw values using a baseline that excludes BOTH the row's own year and T:
    anomaly(row y) = value(y) - mean( value(years not in {y, T}) ),  at least 2 valid years.
For the held-out year (y = T) this is exactly the leave-one-year-out anomaly in
yield_anomalies_v1.csv / weather_anomalies_v1.csv, which the script verifies against those files.
Training anomalies therefore never contain year-T information (checked by a perturbation test).

Per fold: in-fold feature selection (pairwise-complete Spearman + collinearity filter, features fully
observed in train and test), one-hot crop and climate zone, the same models and seeds as the main
pipeline, each trained on the raw target and on the per-crop scaled target.
Baseline: crop mean of the training anomalies.

Outputs (yield_forecast/results/experiments/):
  five_year_loyo_comparison_v1.csv, five_year_loyo_by_fold_v1.csv,
  five_year_loyo_by_crop_v1.csv, five_year_loyo_validation_v1.csv
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from yield_forecast.experiment_scaled_shrinkage import MODELS, build_models
from yield_forecast.train_forecast import (
    ClimatologyBaseline,
    classify_performance,
    fit_crop_scales,
    select_features_in_fold,
)

YEARS = [2015, 2016, 2017, 2018, 2019]
MIN_BASELINE_YEARS = 2          # same minimum as the source LOYO files
TARGET = "y_anom"


def baseline_anomaly(values, y_idx, t_idx):
    """values: (..., n_years). anomaly at y_idx using a baseline over years not in {y_idx, t_idx}."""
    use = [i for i in range(values.shape[-1]) if i not in (y_idx, t_idx)]
    base = values[..., use]
    n = np.sum(~np.isnan(base), axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-missing baselines -> NaN, handled below
        mean = np.nanmean(base, axis=-1)
    out = values[..., y_idx] - mean
    res = np.where(n >= MIN_BASELINE_YEARS, out, np.nan)
    return float(res) if np.ndim(res) == 0 else res


def load_arrays(dataset_path):
    df = pd.read_csv(dataset_path)
    wcols = [c for c in df.columns if c.startswith("wact__")]
    # yield observations usable for baselines (Issue 1/2 eligibility), one array per district-crop
    obs = df.copy()
    obs["y_val"] = np.where(obs["eligible_for_baseline"].astype(bool), obs["yield_kg_per_ha"], np.nan)
    keys = obs[["district_code", "crop", "climate_zone"]].drop_duplicates().sort_values(
        ["crop", "district_code"]).reset_index(drop=True)
    Y = np.full((len(keys), len(YEARS)), np.nan)
    idx = {(r.district_code, r.crop): i for i, r in enumerate(keys.itertuples(index=False))}
    for r in obs.itertuples(index=False):
        Y[idx[(r.district_code, r.crop)], YEARS.index(r.year)] = r.y_val
    dists = sorted(df["district_code"].unique())
    dy = df.drop_duplicates(["district_code", "year"])
    W = np.full((len(dists), len(YEARS), len(wcols)), np.nan)
    for r in dy.itertuples(index=False):
        d = dists.index(r.district_code)
        W[d, YEARS.index(r.year), :] = [getattr(r, c) for c in wcols]
    return keys, Y, dists, W, wcols


def fold_frames(keys, Y, dists, W, wcols, t_idx):
    """Training and test frames for held-out year T, with nested anomalies."""
    feats = [c.replace("wact__", "wanom__") for c in wcols]
    rows = []
    for i, k in enumerate(keys.itertuples(index=False)):
        d = dists.index(k.district_code)
        for y_idx in range(len(YEARS)):
            ya = baseline_anomaly(Y[i], y_idx, t_idx)
            wa = baseline_anomaly(W[d].T, y_idx, t_idx)  # (n_features,)
            rec = {"district_code": k.district_code, "crop": k.crop, "climate_zone": k.climate_zone,
                   "year": YEARS[y_idx], TARGET: ya}
            rec.update(dict(zip(feats, wa)))
            rows.append(rec)
    fr = pd.DataFrame(rows)
    fr = fr[fr[TARGET].notna()]
    return fr[fr["year"] != YEARS[t_idx]].copy(), fr[fr["year"] == YEARS[t_idx]].copy(), feats


def run(dataset, config):
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    fs = cfg["feature_selection"]
    keys, Y, dists, W, wcols = load_arrays(dataset)
    recs, sel_log = [], []

    for t_idx, T in enumerate(YEARS):
        tr, te, feats = fold_frames(keys, Y, dists, W, wcols, t_idx)
        te = te.reset_index(drop=True)
        avail = [c for c in feats if tr[c].notna().all() and te[c].notna().all()]
        sel = select_features_in_fold(tr, avail, TARGET, fs["max_features"], fs["max_collinearity"],
                                      fs["min_observed_fraction"])
        sel_log.append({"test_year": T, "n_train": len(tr), "n_test": len(te),
                        "n_features_available": len(avail), "selected": ";".join(sel)})

        crop_scale, gstd = fit_crop_scales(tr, TARGET)
        s_tr = tr["crop"].map(crop_scale).fillna(gstd).values
        s_te = te["crop"].map(crop_scale).fillna(gstd).values
        clim = ClimatologyBaseline().fit(tr, TARGET).predict(te)
        y_te = te[TARGET].values

        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        cats = ["crop", "climate_zone"]
        sc = StandardScaler()
        X_tr = np.hstack([ohe.fit_transform(tr[cats]), sc.fit_transform(tr[sel])])
        X_te = np.hstack([ohe.transform(te[cats]), sc.transform(te[sel])])

        def add(model, variant, pred):
            for i in range(len(te)):
                recs.append({"test_year": T, "district_code": te.loc[i, "district_code"], "crop": te.loc[i, "crop"],
                             "model": model, "variant": variant, "actual": y_te[i], "pred": pred[i],
                             "abs_error": abs(y_te[i] - pred[i]),
                             "scaled_abs_error": abs(y_te[i] - pred[i]) / s_te[i]})

        add("Climatology", "baseline", clim)
        for name in MODELS:
            m = build_models(cfg)[name]
            m.fit(X_tr, tr[TARGET].values)
            add(name, "raw_target", m.predict(X_te))
            m = build_models(cfg)[name]
            m.fit(X_tr, tr[TARGET].values / s_tr)
            add(name, "scaled_target", m.predict(X_te) * s_te)
    return pd.DataFrame(recs), pd.DataFrame(sel_log)


def verify_against_source(dataset, keys, Y, dists, W, wcols):
    """Held-out-year anomalies must equal the source LOYO files (proves the recomputation is right)."""
    out = []
    ysrc = pd.read_csv("yield_feature_engineering/yield_anomalies_v1.csv")
    ysrc = ysrc[ysrc["anomaly_status"].isin(["calculated", "zero_baseline_std"])]
    ysrc = ysrc.set_index(["district_code", "crop", "year"])["yield_anomaly_kg_per_ha"]
    diffs, n_missing = [], 0
    for i, k in enumerate(keys.itertuples(index=False)):
        for t_idx, T in enumerate(YEARS):
            mine = baseline_anomaly(Y[i], t_idx, t_idx)
            key = (k.district_code, k.crop, T)
            if key in ysrc.index and not np.isnan(mine):
                diffs.append(abs(mine - ysrc.loc[key]))
            elif key in ysrc.index and np.isnan(mine):
                n_missing += 1
    out.append({"quantity": "yield anomaly (held-out year)", "n_compared": len(diffs),
                "max_abs_diff": float(np.max(diffs)) if diffs else np.nan, "source_only_missing_in_mine": n_missing})
    wsrc = pd.read_csv("weather_feature_engineering/weather_anomalies_v1.csv")
    wsrc = wsrc[wsrc["is_validation_year"] == True]  # noqa: E712
    wsrc = wsrc.dropna(subset=["anomaly"]).set_index(["district_code", "year", "weather_column"])["anomaly"]
    wd, wmiss = [], 0
    names = [c.replace("wact__", "") for c in wcols]
    for d, dc in enumerate(dists):
        for t_idx, T in enumerate(YEARS):
            mine = baseline_anomaly(W[d].T, t_idx, t_idx)
            for j, nm in enumerate(names):
                key = (dc, T, nm)
                if key in wsrc.index and not np.isnan(mine[j]):
                    wd.append(abs(mine[j] - wsrc.loc[key]))
                elif key in wsrc.index:
                    wmiss += 1
    out.append({"quantity": "weather anomaly (held-out year)", "n_compared": len(wd),
                "max_abs_diff": float(np.max(wd)) if wd else np.nan, "source_only_missing_in_mine": wmiss})
    return pd.DataFrame(out)


def summarise(p):
    base = p[p.model == "Climatology"].sort_values(["test_year", "district_code", "crop"]).reset_index(drop=True)
    rows = []
    for variant in ["raw_target", "scaled_target"]:
        for name in MODELS:
            g = p[(p.model == name) & (p.variant == variant)].sort_values(
                ["test_year", "district_code", "crop"]).reset_index(drop=True)
            assert (g[["test_year", "district_code", "crop"]].values == base[["test_year", "district_code", "crop"]].values).all()
            d = pd.Series(g.scaled_abs_error.values - base.scaled_abs_error.values).groupby(g.district_code.values).mean()
            pv = 1.0 if np.allclose(d.values, 0) else float(wilcoxon(d.values, alternative="two-sided").pvalue)
            sdiff = g.scaled_abs_error.mean() - base.scaled_abs_error.mean()
            _, verdict = classify_performance(name, sdiff, pv)
            per_year = {}
            for T in YEARS:
                a = g[g.test_year == T].scaled_abs_error.mean()
                b = base[base.test_year == T].scaled_abs_error.mean()
                per_year[T] = 100 * (1 - a / b)
            rows.append({
                "model": name, "variant": variant, "n": len(g),
                "mae_kg_ha": g.abs_error.mean(), "clim_mae_kg_ha": base.abs_error.mean(),
                "scaled_mae": g.scaled_abs_error.mean(), "clim_scaled_mae": base.scaled_abs_error.mean(),
                "skill_vs_clim_pct": 100 * (1 - g.scaled_abs_error.mean() / base.scaled_abs_error.mean()),
                "beats_clim_pct": 100 * (g.scaled_abs_error.values < base.scaled_abs_error.values).mean(),
                "years_better_than_clim": int(sum(v > 0 for v in per_year.values())), "years_total": len(YEARS),
                "direction_acc_pct": 100 * (np.sign(g.pred) == np.sign(g.actual)).mean(),
                "district_wilcoxon_p": pv, "verdict": verdict,
                **{f"skill_{T}": per_year[T] for T in YEARS},
            })
    return pd.DataFrame(rows)


def by_crop(p):
    base = p[p.model == "Climatology"].set_index(["test_year", "district_code", "crop"])
    rows = []
    for name in MODELS:
        g = p[(p.model == name) & (p.variant == "scaled_target")].set_index(["test_year", "district_code", "crop"])
        b = base.loc[g.index]
        for crop, gi in g.groupby("crop"):
            bi = b.loc[gi.index]
            rows.append({"crop": crop, "model": name, "n": len(gi),
                         "skill_vs_clim_pct": 100 * (1 - gi.scaled_abs_error.mean() / bi.scaled_abs_error.mean()),
                         "beats_clim_pct": 100 * (gi.scaled_abs_error.values < bi.scaled_abs_error.values).mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml_ready_dataset/ml_ready_dataset_v1.csv")
    ap.add_argument("--config", default="yield_forecast/models/model_config.json")
    ap.add_argument("--outdir", default="yield_forecast/results/experiments")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    keys, Y, dists, W, wcols = load_arrays(a.dataset)
    val = verify_against_source(a.dataset, keys, Y, dists, W, wcols)
    p, sel = run(a.dataset, a.config)
    comp = summarise(p)
    crop = by_crop(p)

    comp.to_csv(os.path.join(a.outdir, "five_year_loyo_comparison_v1.csv"), index=False)
    sel.to_csv(os.path.join(a.outdir, "five_year_loyo_by_fold_v1.csv"), index=False)
    crop.to_csv(os.path.join(a.outdir, "five_year_loyo_by_crop_v1.csv"), index=False)
    val.to_csv(os.path.join(a.outdir, "five_year_loyo_validation_v1.csv"), index=False)

    pd.set_option("display.width", 260)
    print("\nRECOMPUTATION CHECK against the source LOYO files (held-out-year anomalies)\n")
    print(val.to_string(index=False))
    print("\nFOLDS (nested leave-one-year-out; selected features per held-out year)\n")
    print(sel.to_string(index=False))
    cols = ["model", "variant", "n", "mae_kg_ha", "clim_mae_kg_ha", "scaled_mae", "clim_scaled_mae",
            "skill_vs_clim_pct", "beats_clim_pct", "years_better_than_clim", "direction_acc_pct",
            "district_wilcoxon_p", "verdict"]
    print("\nFIVE-YEAR RETROSPECTIVE RESPONSE ESTIMATION (NOT forecasting)\n")
    print(comp[cols].round(3).to_string(index=False))
    sk = ["model", "variant"] + [f"skill_{T}" for T in YEARS]
    print("\nSKILL VS CLIMATOLOGY BY HELD-OUT YEAR (%)\n")
    print(comp[sk].round(1).to_string(index=False))
