#!/usr/bin/env python3
"""
train_forecast.py - Issue #50: ML Yield-Anomaly Response Estimation Pipeline

Executes forward-chaining prospective evaluation for:
  1. Climatology Baseline (Crop-specific historical training mean)
  2. Ridge Regression (L2 regularized linear model)
  3. Random Forest Regressor (Conservative shallow trees)
  4. XGBoost Regressor (Gradient boosted trees with L1/L2 penalties)
  5. CatBoost Regressor (Oblivious trees with L2 regularization)

Operational Taxonomy & Timing Horizon:
  - This pipeline is an END-OF-SEASON RETROSPECTIVE WEATHER-YIELD RESPONSE ESTIMATION
    (diagnostic hindcast). It uses weather indicators from annual and seasonal windows
    (annual_cal, kharif, ne, rabi_next) contemporaneous with or postdating crop growth.
  - It does NOT provide advance pre-sowing or mid-season operational early-warning forecasts.

Key Principles & Safeguards:
  - Strict forward chaining: Models train on year(s) < test_year. No future temporal leakage.
  - In-fold feature selection: Dynamic Spearman screening & multicollinearity filter
    fitted strictly on training fold data.
  - Target: Forward-chaining yield anomaly (yield_anomaly_kg_per_ha). Models are fitted on the
    anomaly divided by each crop's training-fold std (so high-variance crops such as sugarcane
    do not dominate) and predictions are converted back to kg/ha. Use --no-scale-target for
    the raw-target variant.
  - Categorical context: One-hot encoded crop and climate_zone (Issue #6 clusters).
  - Uncertainty Intervals: Exploratory residual-quantile intervals derived from in-sample
    training residuals, honestly labeled as uncalibrated due to small sample constraints.
  - Statistical Rigor: A district-clustered, two-sided Wilcoxon signed-rank test on per-crop
    scaled absolute errors assesses whether a model differs from the climatology baseline.
    Verdict labels are derived from the computed result, never hardcoded.
  - No imputation: feature screening is pairwise-complete and only features fully observed
    in the train and test rows of a fold are eligible.
  - Reproducibility: Deterministic seeds and thread constraints (OMP_NUM_THREADS=1).
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

try:  # imported as a package (python -m ...) or run as a script (python yield_forecast/train_forecast.py)
    from yield_forecast.folds import generate_folds, validate_folds
except ModuleNotFoundError:
    from folds import generate_folds, validate_folds
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Enforce deterministic thread environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class ClimatologyBaseline:
    """Predicts historical mean anomaly per crop from training fold."""
    def __init__(self):
        self.crop_means = {}
        self.global_mean = 0.0

    def fit(self, df_train: pd.DataFrame, target_col: str):
        self.crop_means = df_train.groupby('crop')[target_col].mean().to_dict()
        self.global_mean = float(df_train[target_col].mean()) if len(df_train) > 0 else 0.0
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        return df_test['crop'].map(self.crop_means).fillna(self.global_mean).values


def _pairwise_complete(x: pd.Series, y: pd.Series):
    mask = x.notna() & y.notna()
    return x[mask].values, y[mask].values


def select_features_in_fold(
    df_train: pd.DataFrame,
    candidate_features: list,
    target_col: str,
    max_features: int = 3,
    max_collinearity: float = 0.80,
    min_observed_fraction: float = 1.0,
    min_pairs: int = 8
) -> list:
    """
    Selects top weather features strictly on the training fold.

    Spearman screening and the greedy multicollinearity filter use pairwise-complete
    observations only. Nothing is imputed. Candidates observed in fewer than
    `min_observed_fraction` of training rows, or with fewer than `min_pairs`
    complete pairs, are skipped.
    """
    valid_candidates = [
        col for col in candidate_features
        if df_train[col].notna().mean() >= min_observed_fraction
    ]
    if not valid_candidates:
        return []

    y = df_train[target_col]
    corrs = []
    for col in valid_candidates:
        x_v, y_v = _pairwise_complete(df_train[col], y)
        if len(x_v) < min_pairs or np.std(x_v) == 0 or np.std(y_v) == 0:
            continue
        rho, p_val = spearmanr(x_v, y_v)
        if not np.isnan(rho):
            corrs.append((col, abs(rho), rho, p_val))

    # Sort descending by absolute Spearman rho (column name breaks ties deterministically)
    corrs.sort(key=lambda x: (-x[1], x[0]))

    selected = []
    for col, abs_rho, rho, p_val in corrs:
        if len(selected) >= max_features:
            break
        collinear = False
        for prev in selected:
            a_v, b_v = _pairwise_complete(df_train[col], df_train[prev])
            if len(a_v) < min_pairs or np.std(a_v) == 0 or np.std(b_v) == 0:
                continue
            if abs(np.corrcoef(a_v, b_v)[0, 1]) > max_collinearity:
                collinear = True
                break
        if not collinear:
            selected.append(col)

    return selected


def fit_crop_scales(df_train: pd.DataFrame, target_col: str) -> tuple:
    """Per-crop anomaly standard deviation from the TRAINING fold only.

    Returns (crop_scale dict, global_std). A crop with zero/undefined spread falls back to
    the global training std. Used both to scale the training target and to put errors on
    a comparable footing across crops.
    """
    global_std = float(df_train[target_col].std(ddof=0))
    crop_std = df_train.groupby('crop')[target_col].std(ddof=0).to_dict()
    crop_scale = {c: (v if v and v > 0 else global_std) for c, v in crop_std.items()}
    return crop_scale, global_std


def classify_performance(m_name: str, mae_diff: float, p_val: float, alpha: float = 0.05) -> tuple:
    """Derive (significant_flag, verdict) from the computed result only."""
    if m_name == "Climatology":
        return "Reference", "Reference Climatology Baseline"
    if p_val < alpha and mae_diff < 0:
        return "YES", f"Significantly better than Climatology (p={p_val:.4f})"
    if p_val < alpha and mae_diff > 0:
        return "NO", f"Significantly worse than Climatology (p={p_val:.4f})"
    direction = "lower" if mae_diff < 0 else "higher"
    return "NO", f"Not distinguishable from Climatology; scaled MAE {direction} (p={p_val:.4f})"


MODEL_NAMES = ["Ridge", "RandomForest", "XGBoost", "CatBoost"]


def build_models(config):
    """The four models with the settings in model_config.json (single definition used by the pipeline,
    the inner calibration folds and the experiments)."""
    m = config["models"]
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
            verbose=m["CatBoost"]["verbose"], allow_writing_files=m["CatBoost"].get("allow_writing_files", False)),
    }


def conformal_quantile(scores, alpha=0.2):
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (the maximum if n is too small)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])


def predict_model(name, tr, te, feats, config, target, s_tr, s_te):
    """Fit `name` on tr (target divided by s_tr) and return predictions in kg/ha for te and, in-sample, for tr."""
    if name == "Climatology":
        base = ClimatologyBaseline().fit(tr, target)
        return base.predict(te), base.predict(tr)
    cats = ["crop", "climate_zone"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    parts_tr, parts_te = [ohe.fit_transform(tr[cats])], [ohe.transform(te[cats])]
    if feats:
        sc = StandardScaler()
        parts_tr.append(sc.fit_transform(tr[feats]))
        parts_te.append(sc.transform(te[feats]))
    X_tr, X_te = np.hstack(parts_tr), np.hstack(parts_te)
    model = build_models(config)[name]
    model.fit(X_tr, tr[target].values / s_tr)
    return model.predict(X_te) * s_te, model.predict(X_tr) * s_tr


def oof_abs_residuals(df_train, available_features, config, target, scale_target=True):
    """Out-of-fold absolute residuals (kg/ha) for the baseline and every model, from leave-one-district-out
    cross-validation INSIDE the training rows only. Feature selection is repeated inside each inner fold.
    Returns {model name: array aligned with df_train's rows}."""
    fs = config["feature_selection"]
    names = ["Climatology"] + MODEL_NAMES
    res = {m: np.full(len(df_train), np.nan) for m in names}
    pos = {ix: i for i, ix in enumerate(df_train.index)}
    for dist in sorted(df_train["district_code"].unique()):
        inner_tr = df_train[df_train["district_code"] != dist]
        inner_te = df_train[df_train["district_code"] == dist]
        feats = select_features_in_fold(inner_tr, available_features, target, fs["max_features"],
                                        fs["max_collinearity"], fs["min_observed_fraction"])
        scale, g = fit_crop_scales(inner_tr, target)
        s_in_tr = inner_tr["crop"].map(scale).fillna(g).values
        s_in_te = inner_te["crop"].map(scale).fillna(g).values
        if not scale_target:
            s_in_tr, s_in_te = np.ones_like(s_in_tr), np.ones_like(s_in_te)
        idx = [pos[i] for i in inner_te.index]
        for m in names:
            pred, _ = predict_model(m, inner_tr, inner_te, feats, config, target, s_in_tr, s_in_te)
            res[m][idx] = np.abs(inner_te[target].values - pred)
    return res


def make_intervals(name, pred_te, pred_tr, y_tr, oof, s_tr_crop, s_te_crop, confidence_level):
    """Return (lo, hi, lo_in_sample, hi_in_sample, n_calibration).

    in-sample : pred +/- the `confidence_level` quantile of the absolute IN-SAMPLE training residuals (kg/ha).
    calibrated: pred +/- q * crop_std, with q the split-conformal quantile of OUT-OF-FOLD residuals divided by the
                crop's training std, so every crop gets its own width. Used when `oof` is given, else the
                in-sample interval is returned for both."""
    q_abs = np.quantile(np.abs(y_tr - pred_tr), confidence_level)
    lo_in, hi_in = pred_te - q_abs, pred_te + q_abs
    if oof is None:
        return lo_in, hi_in, lo_in, hi_in, 0
    q = conformal_quantile(oof[name] / s_tr_crop, 1 - confidence_level)
    half = q * s_te_crop
    return pred_te - half, pred_te + half, lo_in, hi_in, int(len(oof[name]))


def interval_score(lo, hi, y, s, alpha):
    """Winkler interval score in crop-std units (lower is better): width plus a penalty for misses."""
    lo, hi, y, s = (np.asarray(a, float) for a in (lo, hi, y, s))
    return ((hi - lo) + (2.0 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))) / s


def interval_fields(i, lo, hi, lo_in, hi_in, act, s, confidence_level, status):
    a = 1 - confidence_level
    return {
        "pi_lower_80": float(lo[i]), "pi_upper_80": float(hi[i]), "pi_covered_80": bool(lo[i] <= act <= hi[i]),
        "pi_lower_80_in_sample": float(lo_in[i]), "pi_upper_80_in_sample": float(hi_in[i]),
        "pi_covered_80_in_sample": bool(lo_in[i] <= act <= hi_in[i]),
        "pi_score_scaled": float(interval_score(lo[i], hi[i], act, s, a)),
        "pi_score_scaled_in_sample": float(interval_score(lo_in[i], hi_in[i], act, s, a)),
        "pi_method": status,
    }


def pi_eval_row(fold_idx, test_year, model, status, confidence_level, lo, hi, lo_in, hi_in, y, s, n_cal):
    a = 1 - confidence_level
    cov = float(np.mean((lo <= y) & (y <= hi)))
    cov_in = float(np.mean((lo_in <= y) & (y <= hi_in)))
    return {
        "fold": fold_idx, "test_year": test_year, "model": model, "calibration_status": status,
        "nominal_target_coverage": confidence_level, "empirical_test_coverage": cov,
        "coverage_gap": cov - confidence_level, "avg_interval_width": float(np.mean(hi - lo)),
        "avg_interval_width_scaled": float(np.mean((hi - lo) / s)),
        "interval_score_scaled": float(np.mean(interval_score(lo, hi, y, s, a))),
        "empirical_test_coverage_in_sample": cov_in, "avg_interval_width_in_sample": float(np.mean(hi_in - lo_in)),
        "interval_score_scaled_in_sample": float(np.mean(interval_score(lo_in, hi_in, y, s, a))),
        "n_calibration_oof": n_cal, "n_evaluated": int(len(y)),
    }


def run_training_pipeline(
    dataset_path: str = "ml_ready_dataset/ml_ready_dataset_v1.csv",
    config_path: str = "yield_forecast/models/model_config.json",
    results_dir: str = "yield_forecast/results",
    scale_target: bool = None
):
    print("=" * 70)
    print("Issue #50: Executing ML Yield-Anomaly Response Estimation Pipeline")
    print("=" * 70)

    # 1. Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    target_col = config["target_column"]
    if scale_target is None:
        scale_target = bool(config.get("target_scaling", {}).get("enabled", True))
    print(f"Target scaling (per-crop training-fold std): {'ON' if scale_target else 'OFF (raw kg/ha target)'}")
    confidence_level = config["prediction_intervals"]["confidence_level"]
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"{dataset_path} not found. Build the single ML-ready dataset first (Issue #7): "
            "python ml_ready_dataset/build_ml_ready_dataset.py"
        )

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Filter strictly to eligible crops with valid forward-chaining anomaly target
    df_eval = df[df["usable_for_forward_chaining_eval"] == True].copy()
    print(f"Evaluated subset (eligible crops & non-null anomalies): {df_eval.shape[0]} rows")
    print("Distribution across evaluation years:")
    for yr, count in df_eval["year"].value_counts().sort_index().items():
        print(f"  Year {yr}: {count} observations")

    # Candidate weather anomaly columns
    candidate_weather_cols = [c for c in df.columns if c.startswith("wanom__")]
    print(f"Candidate weather anomaly indicators: {len(candidate_weather_cols)}")

    # 3. Initialize evaluation storage
    all_predictions = []
    fold_metrics = []
    pi_evaluations = []

    # Folds are built from the years actually present in the data (year-agnostic; see folds.py).
    all_years = sorted(int(y) for y in df["year"].unique())
    years_with_targets = sorted(int(y) for y in df_eval["year"].unique())
    folds = generate_folds(config["validation"], years_with_targets, all_years)
    fold_summary = validate_folds(folds, df_eval, int(config["validation"].get("min_training_rows", 8)))
    print(f"Validation mode: {config['validation'].get('mode', 'explicit')} | years in data: {all_years[0]}-{all_years[-1]} "
          f"({len(all_years)}) | years with targets: {years_with_targets}")
    for fs_ in fold_summary:
        print(f"  Fold {fs_['fold']}: test {fs_['test_year']} (n={fs_['n_test']}) | train rows {fs_['n_train']} "
              f"from years {fs_['train_years_with_targets']}")

    for f_info in folds:
        fold_idx = f_info["fold"]
        train_years = f_info["train_years"]
        test_year = f_info["test_year"]

        train_mask = df_eval["year"].isin(train_years)
        test_mask = df_eval["year"] == test_year

        df_train = df_eval[train_mask].copy()
        df_test = df_eval[test_mask].copy()

        n_train = len(df_train)
        n_test = len(df_test)
        # Forward-chaining targets exist only for some years (an anomaly needs two earlier years), so a
        # configured training year can contribute zero rows. Record the years that actually do.
        train_years_with_targets = ";".join(str(int(y)) for y in sorted(df_train["year"].unique()))
        assert all(y < test_year for y in df_train["year"].unique()), "training data must precede the test year"

        print("\n" + "-" * 70)
        print(f"FOLD {fold_idx}: Train {train_years} (N={n_train}) -> Test {test_year} (N={n_test})")
        print(f"Configured training years that actually have usable targets: {train_years_with_targets}")
        print("-" * 70)

        # Only features that are observed for every train AND test row are eligible
        # (availability at prediction time; no values are imputed anywhere).
        available_weather = [
            c for c in candidate_weather_cols
            if df_train[c].notna().all() and df_test[c].notna().all()
        ]
        print(f"Fully observed candidate features in Fold {fold_idx}: {len(available_weather)}/{len(candidate_weather_cols)}")

        # Dynamic feature selection on training data only
        selected_weather = select_features_in_fold(
            df_train=df_train,
            candidate_features=available_weather,
            target_col=target_col,
            max_features=config["feature_selection"]["max_features"],
            max_collinearity=config["feature_selection"]["max_collinearity"],
            min_observed_fraction=config["feature_selection"]["min_observed_fraction"]
        )
        print(f"Selected weather features for Fold {fold_idx}: {selected_weather}")

        # Per-crop scale (training-fold anomaly std) used only to put errors on a
        # comparable footing across crops; fitted on the training fold only.
        crop_scale, global_std = fit_crop_scales(df_train, target_col)
        s_train = df_train['crop'].map(crop_scale).fillna(global_std).values
        s_test = df_test['crop'].map(crop_scale).fillna(global_std).values
        if not scale_target:
            s_train = np.ones_like(s_train)
            s_test = np.ones_like(s_test)

        # Baseline: Climatology
        baseline = ClimatologyBaseline().fit(df_train, target_col)
        y_test_true = df_test[target_col].values
        base_preds = baseline.predict(df_test)

        base_mae = mean_absolute_error(y_test_true, base_preds)
        base_rmse = np.sqrt(mean_squared_error(y_test_true, base_preds))
        base_r2 = r2_score(y_test_true, base_preds)

        print(f"Climatology Baseline: MAE={base_mae:.2f}, RMSE={base_rmse:.2f}, R2={base_r2:.3f}")

        # Preprocessing setup (fitted on training data only)
        cat_cols = ["crop", "climate_zone"]
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        X_train_cat = ohe.fit_transform(df_train[cat_cols])
        X_test_cat = ohe.transform(df_test[cat_cols])

        # Selected features are fully observed by construction; nothing is imputed.
        assert not df_train[selected_weather].isna().any().any()
        assert not df_test[selected_weather].isna().any().any()
        scaler = StandardScaler()
        X_train_w = scaler.fit_transform(df_train[selected_weather])
        X_test_w = scaler.transform(df_test[selected_weather])

        X_train = np.hstack([X_train_cat, X_train_w])
        X_test = np.hstack([X_test_cat, X_test_w])
        y_train = df_train[target_col].values

        # Prediction intervals. Default: out-of-fold, per-crop-scaled split-conformal (see make_intervals);
        # the old in-sample interval is kept beside it in the *_in_sample columns for comparison.
        interval_cfg = config.get("prediction_intervals", {})
        use_oof = (interval_cfg.get("method") == "oof_conformal_per_crop"
                   and df_train["district_code"].nunique() >= 3)
        interval_status = (interval_cfg.get("calibration_status", "oof_conformal_per_crop_exploratory")
                           if use_oof else "uncalibrated_in_sample_exploratory")
        s_crop_train = df_train["crop"].map(crop_scale).fillna(global_std).values
        s_crop_test = df_test["crop"].map(crop_scale).fillna(global_std).values
        oof = oof_abs_residuals(df_train, available_weather, config, target_col, scale_target) if use_oof else None
        print(f"Prediction intervals: {interval_status}")
        train_base_preds = baseline.predict(df_train)
        lo, hi, lo_in, hi_in, n_cal = make_intervals("Climatology", base_preds, train_base_preds, y_train, oof,
                                                     s_crop_train, s_crop_test, confidence_level)

        for i, row in df_test.reset_index(drop=True).iterrows():
            pred_val = float(base_preds[i])
            act_val = float(y_test_true[i])
            all_predictions.append({
                "fold": fold_idx,
                "train_years": ";".join(str(y) for y in train_years),
                "train_years_with_targets": train_years_with_targets,
                "test_year": test_year,
                "district_code": row["district_code"],
                "district_name": row["district_name"],
                "crop": row["crop"],
                "model": "Climatology",
                "actual_yield_anomaly": act_val,
                "predicted_yield_anomaly": pred_val,
                "residual": act_val - pred_val,
                "scaled_abs_error": abs(act_val - pred_val) / crop_scale.get(row["crop"], global_std),
                **interval_fields(i, lo, hi, lo_in, hi_in, act_val, s_crop_test[i], confidence_level, interval_status),
            })

        fold_metrics.append({
            "fold": fold_idx,
            "train_years": ";".join(str(y) for y in train_years),
                "train_years_with_targets": train_years_with_targets,
            "test_year": test_year,
            "n_train": n_train,
            "n_test": n_test,
            "selected_features": ";".join(selected_weather),
            "model": "Climatology",
            "mae": base_mae,
            "rmse": base_rmse,
            "r2": base_r2,
            "mae_diff_vs_baseline": 0.0,
            "rmse_diff_vs_baseline": 0.0
        })

        pi_evaluations.append(pi_eval_row(fold_idx, test_year, "Climatology", interval_status, confidence_level,
                                          lo, hi, lo_in, hi_in, y_test_true, s_crop_test, n_cal))

        model_instances = build_models(config)

        for m_name, model in model_instances.items():
            # Fit on the (optionally per-crop scaled) target; convert predictions back to kg/ha.
            model.fit(X_train, y_train / s_train)
            test_preds = model.predict(X_test) * s_test

            mae = mean_absolute_error(y_test_true, test_preds)
            rmse = np.sqrt(mean_squared_error(y_test_true, test_preds))
            r2 = r2_score(y_test_true, test_preds)

            # Prediction intervals (calibrated default plus the in-sample interval for comparison)
            train_preds = model.predict(X_train) * s_train
            lo, hi, lo_in, hi_in, n_cal = make_intervals(m_name, test_preds, train_preds, y_train, oof,
                                                         s_crop_train, s_crop_test, confidence_level)

            # Record individual test predictions
            for i, row in df_test.reset_index(drop=True).iterrows():
                pred_val = float(test_preds[i])
                act_val = float(y_test_true[i])
                all_predictions.append({
                    "fold": fold_idx,
                    "train_years": ";".join(str(y) for y in train_years),
                "train_years_with_targets": train_years_with_targets,
                    "test_year": test_year,
                    "district_code": row["district_code"],
                    "district_name": row["district_name"],
                    "crop": row["crop"],
                    "model": m_name,
                    "actual_yield_anomaly": act_val,
                    "predicted_yield_anomaly": pred_val,
                    "residual": act_val - pred_val,
                    "scaled_abs_error": abs(act_val - pred_val) / crop_scale.get(row["crop"], global_std),
                    **interval_fields(i, lo, hi, lo_in, hi_in, act_val, s_crop_test[i], confidence_level, interval_status),
                })

            pi_row = pi_eval_row(fold_idx, test_year, m_name, interval_status, confidence_level,
                                 lo, hi, lo_in, hi_in, y_test_true, s_crop_test, n_cal)
            pi_evaluations.append(pi_row)
            emp_cov = pi_row["empirical_test_coverage"]

            fold_metrics.append({
                "fold": fold_idx,
                "train_years": ";".join(str(y) for y in train_years),
                "train_years_with_targets": train_years_with_targets,
                "test_year": test_year,
                "n_train": n_train,
                "n_test": n_test,
                "selected_features": ";".join(selected_weather),
                "model": m_name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mae_diff_vs_baseline": mae - base_mae,
                "rmse_diff_vs_baseline": rmse - base_rmse
            })

            print(f"{m_name:<15}: MAE={mae:.2f} (diff={mae - base_mae:+.2f}), RMSE={rmse:.2f}, R2={r2:.3f}, PI Cov={emp_cov*100:.1f}% (in-sample rule {pi_row['empirical_test_coverage_in_sample']*100:.1f}%)")

    # 4. Save Prediction Deliverable
    df_preds = pd.DataFrame(all_predictions)
    pred_path = os.path.join(results_dir, "predictions_forward_chaining_v1.csv")
    df_preds.to_csv(pred_path, index=False)
    print(f"\n[SAVED] Predictions: {pred_path} ({len(df_preds)} rows)")

    # 5. Save Fold-Level Deliverable
    df_fold_metrics = pd.DataFrame(fold_metrics)
    fold_metrics_path = os.path.join(results_dir, "fold_evaluation_metrics_v1.csv")
    df_fold_metrics.to_csv(fold_metrics_path, index=False)
    print(f"[SAVED] Fold-level metrics: {fold_metrics_path}")

    # 6. Save Prediction Interval Deliverable
    df_pi = pd.DataFrame(pi_evaluations)
    pi_path = os.path.join(results_dir, "prediction_interval_evaluation_v1.csv")
    df_pi.to_csv(pi_path, index=False)
    print(f"[SAVED] Prediction intervals: {pi_path}")

    # 7. Aggregate Model Comparison
    # Significance: per-district mean of (model - climatology) scaled absolute error,
    # then a two-sided Wilcoxon signed-rank test across districts. Districts are the
    # independent-ish units; residuals within a district/year/crop are clustered, so a
    # row-level t-test would overstate significance. Only 2 test years and 13 districts
    # exist, so p-values are indicative at best.
    key = ["fold", "district_code", "crop"]
    agg_rows = []
    base_sub = df_preds[df_preds["model"] == "Climatology"].sort_values(key).reset_index(drop=True)
    base_agg_mae = mean_absolute_error(base_sub["actual_yield_anomaly"], base_sub["predicted_yield_anomaly"])
    base_agg_rmse = np.sqrt(mean_squared_error(base_sub["actual_yield_anomaly"], base_sub["predicted_yield_anomaly"]))
    base_scaled = base_sub["scaled_abs_error"].values
    base_crop_scaled = base_sub.groupby("crop")["scaled_abs_error"].mean()

    for m_name in ["Climatology", "Ridge", "RandomForest", "XGBoost", "CatBoost"]:
        m_preds = df_preds[df_preds["model"] == m_name].sort_values(key).reset_index(drop=True)
        assert (m_preds[key].values == base_sub[key].values).all(), "Observation mismatch across models"
        y_act = m_preds["actual_yield_anomaly"].values
        y_pr = m_preds["predicted_yield_anomaly"].values

        m_mae = mean_absolute_error(y_act, y_pr)
        m_rmse = np.sqrt(mean_squared_error(y_act, y_pr))
        m_r2 = r2_score(y_act, y_pr)
        avg_pi_cov = m_preds["pi_covered_80"].mean()
        avg_pi_cov_in = m_preds["pi_covered_80_in_sample"].mean()
        scaled_mae = float(m_preds["scaled_abs_error"].mean())
        macro_crop_scaled = float(m_preds.groupby("crop")["scaled_abs_error"].mean().mean())

        mae_diff = m_mae - base_agg_mae
        pct_mae_diff = (mae_diff / base_agg_mae) * 100
        rmse_diff = m_rmse - base_agg_rmse
        scaled_diff = scaled_mae - float(base_scaled.mean())

        if m_name == "Climatology":
            p_val = 1.0
        else:
            d = (m_preds["scaled_abs_error"].values - base_scaled)
            per_district = pd.Series(d).groupby(m_preds["district_code"].values).mean()
            if np.allclose(per_district.values, 0):
                p_val = 1.0
            else:
                p_val = float(wilcoxon(per_district.values, alternative="two-sided").pvalue)
        # Significance decision uses the scaled (cross-crop comparable) difference
        stat_sig, perf_conclusion = classify_performance(m_name, scaled_diff, p_val)

        agg_rows.append({
            "model": m_name,
            "total_test_samples": len(m_preds),
            "aggregate_mae": m_mae,
            "aggregate_rmse": m_rmse,
            "aggregate_r2": m_r2,
            "scaled_mae": scaled_mae,
            "scaled_mae_diff_vs_baseline": scaled_diff,
            "macro_crop_scaled_mae": macro_crop_scaled,
            "mae_diff_vs_baseline": mae_diff,
            "pct_mae_diff_vs_baseline": pct_mae_diff,
            "rmse_diff_vs_baseline": rmse_diff,
            "district_cluster_wilcoxon_p_value": p_val,
            "statistically_significant_improvement": stat_sig,
            "nominal_pi_target_80": confidence_level,
            "empirical_pi_coverage_80": avg_pi_cov,
            "empirical_pi_coverage_80_in_sample": avg_pi_cov_in,
            "interval_score_scaled": float(m_preds["pi_score_scaled"].mean()),
            "interval_score_scaled_in_sample": float(m_preds["pi_score_scaled_in_sample"].mean()),
            "pi_calibration_status": m_preds["pi_method"].iloc[0],
            "performance_vs_climatology": perf_conclusion
        })

    df_agg = pd.DataFrame(agg_rows)
    agg_path = os.path.join(results_dir, "aggregate_model_comparison_v1.csv")
    df_agg.to_csv(agg_path, index=False)
    print(f"[SAVED] Aggregate model comparison: {agg_path}")
    print("\n" + "=" * 70)
    print("Aggregate Model Comparison across All Forward-Chaining Folds:")
    print("=" * 70)
    print(df_agg[["model", "total_test_samples", "aggregate_mae", "scaled_mae", "macro_crop_scaled_mae", "pct_mae_diff_vs_baseline", "district_cluster_wilcoxon_p_value", "statistically_significant_improvement", "empirical_pi_coverage_80", "performance_vs_climatology"]].to_string(index=False))

    # 8. Per-Crop Evaluation Breakdown (roadmap M5: per-crop sample-size flags)
    # Errors are scaled by each crop's training-fold std so crops are comparable. The flag marks crops
    # whose smallest fold has too few test rows for a per-crop conclusion (project threshold in config).
    min_fold_n = int(config.get("evaluation", {}).get("min_fold_test_samples_for_conclusion", 10))
    crop_rows = []
    crops = sorted(df_eval["crop"].unique())
    key = ["fold", "district_code", "crop"]
    base_all = df_preds[df_preds["model"] == "Climatology"].set_index(key)
    for crop_name in crops:
        for m_name in ["Climatology", "Ridge", "RandomForest", "XGBoost", "CatBoost"]:
            m_crop = df_preds[(df_preds["crop"] == crop_name) & (df_preds["model"] == m_name)]
            if len(m_crop) == 0:
                continue
            c_act = m_crop["actual_yield_anomaly"].values
            c_pr = m_crop["predicted_yield_anomaly"].values
            c_mae = mean_absolute_error(c_act, c_pr)
            c_rmse = np.sqrt(mean_squared_error(c_act, c_pr))
            b_crop = base_all.loc[m_crop.set_index(key).index]
            scaled = float(m_crop["scaled_abs_error"].mean())
            base_scaled = float(b_crop["scaled_abs_error"].mean())
            min_fold = int(m_crop.groupby("fold").size().min())
            is_base = m_name == "Climatology"
            crop_rows.append({
                "crop": crop_name,
                "model": m_name,
                "n_test_samples": len(m_crop),
                "n_districts": int(m_crop["district_code"].nunique()),
                "min_fold_n_test": min_fold,
                "low_sample_flag": bool(min_fold < min_fold_n),
                "mae": c_mae,
                "rmse": c_rmse,
                "scaled_mae": scaled,
                "skill_vs_climatology_pct": 0.0 if is_base else 100.0 * (1.0 - scaled / base_scaled),
                "beats_climatology_pct": np.nan if is_base else float(
                    100.0 * (m_crop["scaled_abs_error"].values < b_crop["scaled_abs_error"].values).mean()),
                "pi_coverage_80": float(m_crop["pi_covered_80"].mean()),
                "pi_coverage_80_in_sample": float(m_crop["pi_covered_80_in_sample"].mean()),
            })

    df_crop = pd.DataFrame(crop_rows)
    crop_path = os.path.join(results_dir, "crop_level_evaluation_metrics_v1.csv")
    df_crop.to_csv(crop_path, index=False)
    print(f"\n[SAVED] Per-crop evaluation metrics: {crop_path}")

    return {
        "predictions": df_preds,
        "fold_metrics": df_fold_metrics,
        "aggregate_comparison": df_agg,
        "crop_metrics": df_crop,
        "pi_metrics": df_pi
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issue #50 ML Yield Forecasting Runner")
    parser.add_argument("--dataset", default="ml_ready_dataset/ml_ready_dataset_v1.csv",
                        help="Issue #7 ML-ready dataset (single shared input)")
    parser.add_argument("--config", default="yield_forecast/models/model_config.json")
    parser.add_argument("--results-dir", default="yield_forecast/results")
    parser.add_argument("--no-scale-target", action="store_true",
                        help="Train on the raw kg/ha anomaly (original v1 behaviour)")
    args = parser.parse_args()

    run_training_pipeline(
        dataset_path=args.dataset,
        config_path=args.config,
        results_dir=args.results_dir,
        scale_target=False if args.no_scale_target else None
    )
