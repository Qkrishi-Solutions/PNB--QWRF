#!/usr/bin/env python3
"""
build_area_forecast.py - Issue #49: ML Sown Area Forecast per Crop, District & Season

Reads:
  - ml_ready_dataset/ml_ready_dataset_v1.csv (Issue #7)
  - crop_calendar/crop_calendar_v1.csv (Issue #43)

Evaluates 3 Mandatory Naive Baselines (strictly on year < test_year):
  1. LastValue (Persistence: area at t-1)
  2. Climatology (Expanding historical mean over year < test_year)
  3. LinearTrend (Per district-crop OLS slope over year < test_year, clipped >= 0)

Evaluates 4 Candidate ML Models (Strictly Pre-Sowing Operational Horizon, Zero Silent Imputation):
  4. XGBoost_L1_LogDelta (L1 objective, Log-Delta target, native NaN handling, GroupKFold-by-year GridSearchCV)
  5. HistGBR_L1_LogDelta (L1 objective, Log-Delta target, native NaN handling, GroupKFold-by-year GridSearchCV)
  6. RandomForest_L1_LogDelta (L1 criterion, Log-Delta target, 100%-observed features, GroupKFold-by-year GridSearchCV)
  7. Ridge_Delta (Issue #50 method: L2 Ridge on per-crop training-fold std scaled Delta target)

Governance & Safeguards:
  - Strict forward chaining: training and baselines never access test_year or future years.
  - Strictly causal weather horizon: candidate weather features restricted to prior-season wact__rabi_prev__*.
  - Zero silent imputation: missing lags stay NaN natively for XGBoost/HistGBR, and pure observed lags feed RF/Ridge.
  - Honest evaluation: reports full regression suite (MAE, RMSE, MedAE, WAPE_%, sMAPE_%, R2, Skill_%, Wilcoxon p).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "ml_ready_dataset" / "ml_ready_dataset_v1.csv"
DEFAULT_CALENDAR_PATH = REPO_ROOT / "crop_calendar" / "crop_calendar_v1.csv"
DEFAULT_CONFIG_PATH = REPO_ROOT / "area_forecast" / "model_config.json"
DEFAULT_OUTDIR = REPO_ROOT / "area_forecast"

PURE_LAG_COLS = [
    "lag_area_1",
    "log_lag_area_1",
    "lag_hist_mean",
    "lag_mean_reversion",
    "lag_hist_max",
    "lag_is_zero",
]

RICH_LAG_COLS = PURE_LAG_COLS + [
    "lag_area_2",
    "lag_area_diff",
    "lag_yield_1",
    "lag_prod_1",
]


def _pairwise_complete(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    mask = x.notna() & y.notna()
    return x[mask].values, y[mask].values


def select_features_in_fold(
    df_train: pd.DataFrame,
    candidate_features: list[str],
    target_col: str,
    max_features: int = 4,
    max_collinearity: float = 0.80,
    min_observed_fraction: float = 1.0,
    min_pairs: int = 8,
) -> list[str]:
    """Select top pre-sowing weather features strictly on df_train via Spearman + collinearity filter."""
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

    corrs.sort(key=lambda x: (-x[1], x[0]))

    selected: list[str] = []
    for col, _, _, _ in corrs:
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


def fit_crop_scales(df_train: pd.DataFrame, target_col: str) -> tuple[dict[str, float], float]:
    """Per-crop target standard deviation fitted strictly on df_train."""
    global_std = float(df_train[target_col].std(ddof=0))
    if not global_std or np.isnan(global_std) or global_std <= 0:
        global_std = 1.0
    crop_std = df_train.groupby("crop")[target_col].std(ddof=0).to_dict()
    crop_scale = {
        c: (float(v) if v is not None and not np.isnan(v) and v > 0 else global_std)
        for c, v in crop_std.items()
    }
    return crop_scale, global_std


def build_crop_calendar_lookup(calendar_path: Path) -> pd.DataFrame:
    """Summarize crop-calendar seasons and weather-window alignments per crop without multiplying rows."""
    if not calendar_path.exists():
        raise FileNotFoundError(f"Missing crop calendar: {calendar_path}")
    cal = pd.read_csv(calendar_path)
    win_col = "weather_window_primary" if "weather_window_primary" in cal.columns else "weather_window_alignment"
    rows = []
    for crop, grp in cal.groupby("crop"):
        seasons = "; ".join(sorted(grp["season"].dropna().astype(str).unique()))
        prim = "; ".join(sorted(grp[win_col].dropna().astype(str).unique()))
        rows.append({
            "crop": crop,
            "calendar_seasons": seasons,
            "weather_window_alignment": prim,
        })
    return pd.DataFrame(rows)


def engineer_causal_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create strictly causal (<= t-1) lag features per (district_code, crop) without imputing missing values."""
    df = df.sort_values(["district_code", "crop", "year"]).reset_index(drop=True)
    g = df.groupby(["district_code", "crop"])

    lag_1 = g["area_1000_ha"].shift(1)
    lag_2 = g["area_1000_ha"].shift(2)
    lag_yield_1 = g["yield_kg_per_ha"].shift(1)
    lag_prod_1 = g["production_1000_tons"].shift(1)

    hist_mean = g["area_1000_ha"].transform(lambda s: s.shift(1).expanding().mean())
    hist_max = g["area_1000_ha"].transform(lambda s: s.shift(1).expanding().max())

    eng = pd.DataFrame({
        "lag_area_1": lag_1,
        "lag_area_2": lag_2,
        "lag_area_diff": lag_1 - lag_2,
        "log_lag_area_1": np.log1p(lag_1),
        "lag_hist_mean": hist_mean,
        "lag_mean_reversion": lag_1 - hist_mean,
        "lag_hist_max": hist_max,
        "lag_is_zero": (lag_1 == 0).astype(float),
        "lag_yield_1": lag_yield_1,
        "lag_prod_1": lag_prod_1,
        "area_delta": df["area_1000_ha"] - lag_1,
        "area_log_delta": np.log1p(df["area_1000_ha"]) - np.log1p(lag_1),
    }, index=df.index)

    return pd.concat([df, eng], axis=1)


def compute_sample_size_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-crop observation coverage, zero counts, and suitability flags."""
    rows = []
    for crop, grp in df.groupby("crop"):
        n_total = int(len(grp))
        n_nonzero = int((grp["area_1000_ha"] > 0).sum())
        n_zero = int((grp["area_1000_ha"] == 0).sum())
        n_missing = int(grp["area_1000_ha"].isna().sum())
        mean_a = float(grp["area_1000_ha"].mean())
        std_a = float(grp["area_1000_ha"].std())
        if n_zero > 10:
            flag = "LOW_SAMPLE_ZERO_INFLATED"
        elif mean_a < 5.0:
            flag = "MINOR_ACREAGE"
        else:
            flag = "SUFFICIENT"
        rows.append({
            "crop": crop,
            "calendar_seasons": str(grp["calendar_seasons"].iloc[0]),
            "weather_window_alignment": str(grp["weather_window_alignment"].iloc[0]),
            "total_obs": n_total,
            "nonzero_obs": n_nonzero,
            "zero_obs": n_zero,
            "missing_obs": n_missing,
            "mean_area_1000_ha": round(mean_a, 4),
            "std_area_1000_ha": round(std_a, 4),
            "sample_size_flag": flag,
        })
    return pd.DataFrame(rows).sort_values("mean_area_1000_ha", ascending=False).reset_index(drop=True)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    num = 2.0 * np.abs(y_true - y_pred)
    ratio = np.zeros_like(denom, dtype=float)
    np.divide(num, denom, out=ratio, where=denom > 0.0)
    return float(np.mean(ratio) * 100.0)


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100.0) if total > 0 else np.nan


def classify_vs_baseline(name: str, mae_diff: float, p_val: float, alpha: float = 0.05) -> str:
    if name == "LastValue":
        return "Reference Persistence Baseline"
    if p_val < alpha and mae_diff < 0:
        return f"Beats LastValue significantly (p={p_val:.4f})"
    if p_val < alpha and mae_diff > 0:
        return f"Does not beat LastValue; significantly worse (p={p_val:.4f})"
    if mae_diff < 0:
        return f"Lower MAE/RMSE than LastValue, not statistically significant over 13 districts (p={p_val:.4f})"
    return f"Does not beat LastValue (p={p_val:.4f})"


def run_area_forecast(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    outdir: Path = DEFAULT_OUTDIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing ML-ready dataset: {dataset_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    assert df[["district_code", "crop", "year"]].duplicated().sum() == 0, "Duplicate keys in ml_ready_dataset!"
    assert (df["area_1000_ha"] >= 0).all(), "Negative area_1000_ha in input dataset!"

    cal_lookup = build_crop_calendar_lookup(calendar_path)
    n_before = len(df)
    df = df.merge(cal_lookup, on="crop", how="left")
    if len(df) != n_before:
        raise AssertionError(f"Crop calendar join multiplied rows: {n_before} -> {len(df)}")
    if df["calendar_seasons"].isna().any():
        missing_crops = df[df["calendar_seasons"].isna()]["crop"].unique().tolist()
        raise ValueError(f"Missing crop calendar entries for: {missing_crops}")

    df = engineer_causal_lag_features(df)
    sample_flags_df = compute_sample_size_flags(df)

    w_prefix = config["feature_selection"]["weather_prefix_allowed"]
    candidate_weather = [c for c in df.columns if c.startswith(w_prefix)]
    conf = float(config["prediction_intervals"]["confidence_level"])
    seed = int(config["random_seed"])

    all_preds = []
    fold_metrics_rows = []

    for f_info in config["validation"]["folds"]:
        fold_idx = int(f_info["fold"])
        train_years = [int(y) for y in f_info["train_years"]]
        test_year = int(f_info["test_year"])

        assert all(y < test_year for y in train_years), f"Temporal leakage detected: {train_years} vs {test_year}"

        df_hist = df[df["year"] < test_year].copy()
        df_train = df[df["year"].isin(train_years)].copy()
        df_test = df[df["year"] == test_year].copy()
        assert df_train["year"].max() < df_test["year"].min()

        # Pre-sowing weather 100% observed in both train and test (zero imputation)
        available_weather = [
            c for c in candidate_weather
            if df_train[c].notna().all() and df_test[c].notna().all()
        ]
        selected_weather = select_features_in_fold(
            df_train=df_train,
            candidate_features=available_weather,
            target_col="area_log_delta",
            max_features=config["feature_selection"]["max_features"],
            max_collinearity=config["feature_selection"]["max_collinearity"],
            min_observed_fraction=config["feature_selection"]["min_observed_fraction"],
            min_pairs=config["feature_selection"]["min_pairs"],
        )

        # Per-crop scale fitted strictly on df_train
        crop_scale, global_std = fit_crop_scales(df_train, "area_delta")
        s_train = df_train["crop"].map(crop_scale).fillna(global_std).values
        s_test = df_test["crop"].map(crop_scale).fillna(global_std).values

        # --- 1. Baselines (year < test_year) ---
        pred_last = df_test["lag_area_1"].values

        clim_map = df_hist.groupby(["district_code", "crop"])["area_1000_ha"].mean().to_dict()
        pred_clim = np.array([
            clim_map.get((r.district_code, r.crop), r.lag_area_1) for r in df_test.itertuples()
        ])

        trend_map = {}
        for (d, c), grp in df_hist.groupby(["district_code", "crop"]):
            if len(grp) >= 2 and grp["area_1000_ha"].nunique() > 1:
                lr = LinearRegression().fit(grp["year"].values.reshape(-1, 1), grp["area_1000_ha"].values)
                trend_map[(d, c)] = max(0.0, float(lr.predict([[test_year]])[0]))
            else:
                trend_map[(d, c)] = float(grp["area_1000_ha"].iloc[-1])
        pred_trend = np.array([
            trend_map.get((r.district_code, r.crop), r.lag_area_1) for r in df_test.itertuples()
        ])

        # --- 2. Preprocessors fitted strictly on df_train (Zero Imputation!) ---
        cat_cols = ["district_code", "crop", "climate_zone"]
        num_rich = RICH_LAG_COLS + selected_weather
        num_pure = PURE_LAG_COLS + selected_weather

        assert not df_train[num_pure].isna().any().any()
        assert not df_test[num_pure].isna().any().any()

        # StandardScaler preserves NaNs natively in num_rich for XGBoost & HistGBR
        prep_rich = ColumnTransformer([
            ("num", StandardScaler(), num_rich),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])
        X_tr_rich = prep_rich.fit_transform(df_train[num_rich + cat_cols])
        X_te_rich = prep_rich.transform(df_test[num_rich + cat_cols])

        prep_pure = ColumnTransformer([
            ("num", StandardScaler(), num_pure),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])
        X_tr_pure = prep_pure.fit_transform(df_train[num_pure + cat_cols])
        X_te_pure = prep_pure.transform(df_test[num_pure + cat_cols])

        inner_cv = GroupKFold(n_splits=df_train["year"].nunique())
        groups_train = df_train["year"].values

        y_tr_logdelta = df_train["area_log_delta"].values
        y_tr_delta_scaled = df_train["area_delta"].values / s_train
        log_lag_tr = np.log1p(df_train["lag_area_1"].values)
        log_lag_te = np.log1p(df_test["lag_area_1"].values)
        y_tr_area = df_train["area_1000_ha"].values
        y_te_area = df_test["area_1000_ha"].values

        fold_df = df_test[[
            "district_code", "district_name", "climate_zone", "year", "crop",
            "calendar_seasons", "weather_window_alignment", "area_1000_ha", "lag_area_1"
        ]].copy()
        fold_df["fold"] = fold_idx
        fold_df["crop_scale"] = s_test
        fold_df["pred_LastValue"] = np.clip(pred_last, 0.0, None)
        fold_df["pred_Climatology"] = np.clip(pred_clim, 0.0, None)
        fold_df["pred_LinearTrend"] = np.clip(pred_trend, 0.0, None)

        for b_name in ["LastValue", "Climatology", "LinearTrend"]:
            bp = fold_df[f"pred_{b_name}"].values
            fold_metrics_rows.append({
                "fold": fold_idx,
                "train_years": ";".join(str(y) for y in train_years),
                "test_year": test_year,
                "n_train": len(df_train),
                "n_test": len(df_test),
                "selected_weather_features": ";".join(selected_weather),
                "model": b_name,
                "best_params": "baseline",
                "mae_1000_ha": float(mean_absolute_error(y_te_area, bp)),
                "rmse_1000_ha": float(np.sqrt(mean_squared_error(y_te_area, bp))),
                "medae_1000_ha": float(median_absolute_error(y_te_area, bp)),
                "wape_percent": compute_wape(y_te_area, bp),
                "smape_percent": compute_smape(y_te_area, bp),
                "r2": float(r2_score(y_te_area, bp)),
            })

        # --- 3. Train & Tune Candidate ML Models ---
        # 3a. XGBoost_L1_LogDelta (native NaN)
        xgb_cfg = config["models"]["XGBoost_L1_LogDelta"]
        xgb_grid = GridSearchCV(
            XGBRegressor(objective=xgb_cfg["objective"], random_state=seed, n_jobs=1, verbosity=0),
            param_grid=xgb_cfg["param_grid"],
            cv=inner_cv,
            scoring=config["inner_tuning"]["scoring"],
            n_jobs=1,
        )
        xgb_grid.fit(X_tr_rich, y_tr_logdelta, groups=groups_train)
        p_xgb_tr = np.clip(np.expm1(log_lag_tr + xgb_grid.predict(X_tr_rich)), 0.0, None)
        p_xgb_te = np.clip(np.expm1(log_lag_te + xgb_grid.predict(X_te_rich)), 0.0, None)

        # 3b. HistGBR_L1_LogDelta (pure 100%-observed features so inner-fold binning never sees all-NaN col)
        hgb_cfg = config["models"]["HistGBR_L1_LogDelta"]
        hgb_grid = GridSearchCV(
            HistGradientBoostingRegressor(loss=hgb_cfg["loss"], random_state=seed),
            param_grid=hgb_cfg["param_grid"],
            cv=inner_cv,
            scoring=config["inner_tuning"]["scoring"],
            n_jobs=1,
        )
        hgb_grid.fit(X_tr_pure, y_tr_logdelta, groups=groups_train)
        p_hgb_tr = np.clip(np.expm1(log_lag_tr + hgb_grid.predict(X_tr_pure)), 0.0, None)
        p_hgb_te = np.clip(np.expm1(log_lag_te + hgb_grid.predict(X_te_pure)), 0.0, None)

        # 3c. RandomForest_L1_LogDelta (pure 100%-observed features)
        rf_cfg = config["models"]["RandomForest_L1_LogDelta"]
        rf_grid = GridSearchCV(
            RandomForestRegressor(criterion=rf_cfg["criterion"], random_state=seed, n_jobs=1),
            param_grid=rf_cfg["param_grid"],
            cv=inner_cv,
            scoring=config["inner_tuning"]["scoring"],
            n_jobs=1,
        )
        rf_grid.fit(X_tr_pure, y_tr_logdelta, groups=groups_train)
        p_rf_tr = np.clip(np.expm1(log_lag_tr + rf_grid.predict(X_tr_pure)), 0.0, None)
        p_rf_te = np.clip(np.expm1(log_lag_te + rf_grid.predict(X_te_pure)), 0.0, None)

        # 3d. Ridge_Delta (Issue #50 direct counterpart: per-crop std scaled additive delta)
        ridge_cfg = config["models"]["Ridge_Delta"]
        ridge_mod = Ridge(alpha=ridge_cfg["alpha"], fit_intercept=ridge_cfg["fit_intercept"], random_state=seed)
        ridge_mod.fit(X_tr_pure, y_tr_delta_scaled)
        p_ridge_tr = np.clip(df_train["lag_area_1"].values + ridge_mod.predict(X_tr_pure) * s_train, 0.0, None)
        p_ridge_te = np.clip(df_test["lag_area_1"].values + ridge_mod.predict(X_te_pure) * s_test, 0.0, None)

        ml_outputs = [
            ("XGBoost_L1_LogDelta", p_xgb_tr, p_xgb_te, json.dumps(xgb_grid.best_params_)),
            ("HistGBR_L1_LogDelta", p_hgb_tr, p_hgb_te, json.dumps(hgb_grid.best_params_)),
            ("RandomForest_L1_LogDelta", p_rf_tr, p_rf_te, json.dumps(rf_grid.best_params_)),
            ("Ridge_Delta", p_ridge_tr, p_ridge_te, json.dumps({"alpha": ridge_cfg["alpha"]})),
        ]

        for m_name, ptr, pte, bparams in ml_outputs:
            resid = np.abs(y_tr_area - ptr)
            q_pi = float(np.quantile(resid, conf))
            fold_df[f"pred_{m_name}"] = pte
            fold_df[f"pred_{m_name}_pi_low"] = np.clip(pte - q_pi, 0.0, None)
            fold_df[f"pred_{m_name}_pi_high"] = pte + q_pi

            fold_metrics_rows.append({
                "fold": fold_idx,
                "train_years": ";".join(str(y) for y in train_years),
                "test_year": test_year,
                "n_train": len(df_train),
                "n_test": len(df_test),
                "selected_weather_features": ";".join(selected_weather),
                "model": m_name,
                "best_params": bparams,
                "mae_1000_ha": float(mean_absolute_error(y_te_area, pte)),
                "rmse_1000_ha": float(np.sqrt(mean_squared_error(y_te_area, pte))),
                "medae_1000_ha": float(median_absolute_error(y_te_area, pte)),
                "wape_percent": compute_wape(y_te_area, pte),
                "smape_percent": compute_smape(y_te_area, pte),
                "r2": float(r2_score(y_te_area, pte)),
            })

        all_preds.append(fold_df)

    forecast_df = pd.concat(all_preds, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)

    # --- Aggregate Model Comparison ---
    y_true = forecast_df["area_1000_ha"].values
    base_err = np.abs(y_true - forecast_df["pred_LastValue"].values)
    base_mae = float(np.mean(base_err))
    base_rmse = float(np.sqrt(np.mean((y_true - forecast_df["pred_LastValue"].values) ** 2)))

    forecast_df["_base_scaled_err"] = base_err / forecast_df["crop_scale"].values
    dist_base = forecast_df.groupby("district_code")["_base_scaled_err"].mean()

    model_names = [
        "LastValue", "Climatology", "LinearTrend",
        "XGBoost_L1_LogDelta", "HistGBR_L1_LogDelta", "RandomForest_L1_LogDelta", "Ridge_Delta"
    ]
    agg_rows = []
    for m_name in model_names:
        yp = forecast_df[f"pred_{m_name}"].values
        abs_err = np.abs(y_true - yp)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean((y_true - yp) ** 2)))
        medae = float(median_absolute_error(y_true, yp))
        wape = compute_wape(y_true, yp)
        smape = compute_smape(y_true, yp)
        r2 = float(r2_score(y_true, yp))
        scaled_mae = float(np.mean(abs_err / forecast_df["crop_scale"].values))
        skill_pct = (1.0 - mae / base_mae) * 100.0 if base_mae > 0 else 0.0
        mae_diff = mae - base_mae
        rmse_diff = rmse - base_rmse

        if m_name == "LastValue":
            p_val = 1.0
        else:
            forecast_df["_m_scaled_err"] = abs_err / forecast_df["crop_scale"].values
            dist_m = forecast_df.groupby("district_code")["_m_scaled_err"].mean()
            diff = (dist_m - dist_base).values
            p_val = 1.0 if np.all(diff == 0) else float(wilcoxon(dist_m.values, dist_base.values, alternative="two-sided")[1])

        if f"pred_{m_name}_pi_low" in forecast_df.columns:
            covered = (
                (y_true >= forecast_df[f"pred_{m_name}_pi_low"].values)
                & (y_true <= forecast_df[f"pred_{m_name}_pi_high"].values)
            )
            pi_cov = round(float(np.mean(covered)), 4)
            pi_status = config["prediction_intervals"]["calibration_status"]
        else:
            pi_cov = np.nan
            pi_status = "not_applicable"

        agg_rows.append({
            "model": m_name,
            "total_test_samples": len(forecast_df),
            "mae_1000_ha": round(mae, 4),
            "rmse_1000_ha": round(rmse, 4),
            "medae_1000_ha": round(medae, 4),
            "wape_percent": round(wape, 2),
            "smape_percent": round(smape, 2),
            "r2": round(r2, 4),
            "scaled_mae": round(scaled_mae, 4),
            "mae_diff_vs_last_value": round(mae_diff, 4),
            "rmse_diff_vs_last_value": round(rmse_diff, 4),
            "skill_vs_last_value_pct": round(skill_pct, 2),
            "beats_last_value_mae": bool(mae < base_mae),
            "beats_last_value_rmse": bool(rmse < base_rmse),
            "district_wilcoxon_p_value": round(float(p_val), 6),
            "empirical_pi_coverage_80": pi_cov,
            "pi_calibration_status": pi_status,
            "verdict": classify_vs_baseline(m_name, mae_diff, float(p_val)),
        })

    forecast_df.drop(columns=["_base_scaled_err", "_m_scaled_err"], errors="ignore", inplace=True)
    agg_metrics_df = pd.DataFrame(agg_rows).sort_values("mae_1000_ha").reset_index(drop=True)

    # --- Per-Crop Metrics ---
    crop_rows = []
    for crop, grp in forecast_df.groupby("crop"):
        yt = grp["area_1000_ha"].values
        row = {
            "crop": crop,
            "calendar_seasons": str(grp["calendar_seasons"].iloc[0]),
            "n_test_samples": len(grp),
            "mean_area_1000_ha": round(float(grp["area_1000_ha"].mean()), 4),
        }
        best_name = None
        best_mae = float("inf")
        for m_name in model_names:
            m_mae = float(mean_absolute_error(yt, grp[f"pred_{m_name}"].values))
            row[f"mae_{m_name}"] = round(m_mae, 4)
            if m_mae < best_mae:
                best_mae = m_mae
                best_name = m_name
        row["best_model_by_mae"] = best_name
        crop_rows.append(row)

    crop_metrics_df = pd.DataFrame(crop_rows).sort_values("mean_area_1000_ha", ascending=False).reset_index(drop=True)

    # Write CSV outputs
    forecast_df.to_csv(outdir / "area_forecast_v1.csv", index=False)
    agg_metrics_df.to_csv(outdir / "area_forecast_metrics_v1.csv", index=False)
    fold_metrics_df.to_csv(outdir / "area_forecast_fold_metrics_v1.csv", index=False)
    crop_metrics_df.to_csv(outdir / "area_forecast_crop_metrics_v1.csv", index=False)
    sample_flags_df.to_csv(outdir / "sample_size_flags_v1.csv", index=False)

    return forecast_df, agg_metrics_df, fold_metrics_df, crop_metrics_df, sample_flags_df


def main():
    parser = argparse.ArgumentParser(description="Issue #49: Sown Area Forecast Pipeline")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    forecast_df, agg_df, _, _, _ = run_area_forecast(
        dataset_path=args.dataset,
        calendar_path=args.calendar,
        config_path=args.config,
        outdir=args.outdir,
    )
    print(f"Generated area_forecast_v1.csv ({len(forecast_df)} rows)")
    print(
        agg_df[[
            "model", "mae_1000_ha", "rmse_1000_ha", "medae_1000_ha",
            "wape_percent", "smape_percent", "skill_vs_last_value_pct",
            "beats_last_value_mae", "district_wilcoxon_p_value"
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
