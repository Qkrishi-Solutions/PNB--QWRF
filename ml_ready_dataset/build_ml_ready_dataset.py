#!/usr/bin/env python3
"""
build_ml_ready_dataset.py - Issue #7: Integrated ML-ready dataset and pipeline validation.

Builds ONE integrated table, one row per (district_code, year, crop), that is the single
input for the forecasting issues (#49 to #52). Nothing is imputed: missing stays missing.

Inputs (defaults are repo-relative):
  CropLongFormatv1/crop_long_v1.csv                                        base rows (Issue 1)
  CropModelEligibility/crop_model_eligibility_v1.csv                       crop eligibility (Issue 2)
  weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv    weather anomalies (Issue 3)
  yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv        yield anomalies (Issue 4)
  crop specific weather feature selection/crop_feature_selection_v1.csv    LOYO predictors (Issue 5)
  District Climate Zonal Clustering/district_zone_map_v1.csv               climate zones (Issue 6)

Outputs (in --outdir, default ml_ready_dataset/):
  ml_ready_dataset_v1.csv           the dataset
  ml_ready_column_dictionary_v1.csv column roles, source, leakage and timing notes (also rendered in README.md)
  fc_crop_feature_selection_v1.csv  forward-chaining crop-specific predictors per crop and target year
  fc_crop_feature_selection_detail_v1.csv  the correlation statistics behind them
  join_coverage_report_v1.csv       matched / unmatched keys for every join
  unmatched_keys_v1.csv             the actual unmatched keys
  exclusion_report_v1.csv           crop-long rows excluded and why
  validation_checks_v1.csv          every validation and leakage check with its result
  reproducibility_report_v1.json    command, row counts, package versions, SHA-256 of files
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

KEY = ["district_code", "year", "crop"]
DY = ["district_code", "year"]

DEFAULTS = {
    "crop_long": "CropLongFormatv1/crop_long_v1.csv",
    "eligibility": "CropModelEligibility/crop_model_eligibility_v1.csv",
    "weather": "weather_feature_engineering/weather_anomalies_forward_chaining_v1.csv",
    "yield_fc": "yield_feature_engineering/yield_anomalies_forward_chaining_v1.csv",
    "feature_selection": "crop specific weather feature selection/crop_feature_selection_v1.csv",
    "zones": "District Climate Zonal Clustering/district_zone_map_v1.csv",
    "issue5_script": "crop specific weather feature selection/feature_selection.py",
}

YIELD_ANOMALY_COLS = [
    "eligibility_reason", "eligible_for_baseline", "validation_year", "baseline_mean_kg_per_ha",
    "baseline_std_kg_per_ha", "training_observation_count", "training_years", "baseline_status",
    "minimum_training_observations", "baseline_method", "yield_anomaly_kg_per_ha",
    "yield_anomaly_pct", "yield_anomaly_zscore", "anomaly_status", "yield_deviation_class",
]
# Same-year raw weather columns in crop-long; identical to wact__* so they are dropped after
# verifying equality (avoids carrying every weather value twice).
RAW_WEATHER_PREFIXES = ("annual_cal__", "annual_crop__", "kharif__", "ne__", "rabi_next__", "rabi_prev__")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail, kind="validation"):
        self.rows.append({"check": name, "kind": kind, "passed": bool(passed), "detail": detail})

    def frame(self):
        return pd.DataFrame(self.rows)


def _split_years(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(float(x)) for x in str(value).replace(",", ";").split(";") if str(x).strip() != ""]


# Column provenance (used by both the dictionary and the independent leakage checks).
OUTCOME_COLS = {"area_1000_ha", "production_1000_tons", "yield_kg_per_ha", "yield_check_kg_per_ha",
                "yield_rel_diff", "yield_vs_other_years_median"}
TARGET_COLS = {"yield_anomaly_kg_per_ha", "yield_anomaly_pct", "yield_anomaly_zscore",
               "yield_deviation_class", "anomaly_status"}
BASELINE_META_COLS = {"baseline_mean_kg_per_ha", "baseline_std_kg_per_ha", "training_observation_count",
                      "training_years", "baseline_status", "minimum_training_observations", "baseline_method",
                      "validation_year", "eligible_for_baseline", "eligibility_reason"}
ELIGIBILITY_COLS = {"is_eligible_crop", "has_valid_yield_anomaly", "usable_for_forward_chaining_eval",
                    "eligibility_class", "proceed_to_modelling", "eligibility_reasons"}

TIMING_CAVEAT = ("Baseline-leakage safe only. Window dates are not defined in the source, and annual/ne/rabi_next "
                 "windows may extend past the crop's harvest, so availability at a forecast origin is NOT established. "
                 "Treat as retrospective unless the window end is confirmed to precede the forecast origin.")

README_START = "<!-- COLUMN_DICTIONARY_START (generated by build_ml_ready_dataset.py; do not edit by hand) -->"
README_END = "<!-- COLUMN_DICTIONARY_END -->"


def column_role(c):
    """(role, description, is_key, baseline_leakage_safe, timing_caveat, leakage_note)"""
    if c == "crop":
        return ("key", "Row key; usable as a categorical feature", True, True, "", "")
    if c in KEY:
        return ("key", "Row key", True, False, "", "")
    if c.startswith("wanom__"):
        return ("feature_weather_anomaly", "Issue 3 forward-chaining anomaly (actual minus normal from earlier years)",
                False, True, TIMING_CAVEAT, "")
    if c.startswith("wz__"):
        return ("feature_weather_anomaly_z", "Issue 3 forward-chaining anomaly z-score", False, True, TIMING_CAVEAT, "")
    if c.startswith("wact__"):
        return ("feature_weather_actual", "Issue 3 target-year weather value (units not defined in source)", False, True,
                TIMING_CAVEAT, "Units not fully defined in source; do not assume")
    if c == "climate_zone":
        return ("feature_context", "Issue 6 climate zone (a similarity group, not a risk level)", False, True, "", "")
    if c == "cluster_id":
        return ("feature_context", "Issue 6 cluster label", False, False, "", "A label, not a rank")
    if c in TARGET_COLS:
        return ("target", "Issue 4 forward-chaining yield anomaly outcome", False, False, "", "Outcome: never a feature")
    if c in OUTCOME_COLS:
        return ("outcome_year_measurement", "Crop outcome measured in the target year", False, False, "",
                "Target-year outcome: leakage if used as a feature")
    if c.startswith("fc_"):
        return ("selection_info_forward_chaining",
                "Per-crop weather predictors selected with Issue 5's method using only target years before this row's year",
                False, False, "",
                "Safe as this row's per-crop feature list (selection saw only earlier years). Not itself a feature.")
    if c.startswith("loyo_"):
        return ("selection_info_loyo", "Issue 5 leave-one-year-out predictor summary", False, False, "",
                "Selected using the held-out and later years: not forward-chaining safe. Use the fc_* columns instead")
    if c in BASELINE_META_COLS:
        return ("target_baseline_meta", "How the yield anomaly baseline was formed", False, False, "",
                "Derived from the target; not a feature")
    if c.startswith("flag_") or c in ("n_flags", "flag_summary"):
        return ("data_quality_flag", "Issue 1 data-quality flag", False, False, "", "Informational")
    if c in ELIGIBILITY_COLS:
        return ("eligibility_flag", "Eligibility and usability flags", False, False, "", "Use to filter rows")
    return ("metadata", "Descriptive attribute from the crop-long table", False, False, "", "Informational")


def make_dictionary(df):
    rows = []
    for c in df.columns:
        r, desc, is_key, safe, caveat, note = column_role(c)
        rows.append({"column": c, "role": r, "description": desc, "is_key": is_key,
                     "baseline_leakage_safe": bool(safe), "timing_caveat": caveat, "leakage_note": note,
                     "n_missing": int(df[c].isna().sum())})
    return pd.DataFrame(rows)


def render_dictionary_markdown(dd):
    """Compact but complete README dictionary: weather families collapsed to patterns."""
    def fam(prefix):
        cols = [c for c in dd["column"] if c.startswith(prefix)]
        wins = sorted({c.split("__")[1] for c in cols})
        inds = sorted({c.split("__", 2)[2] for c in cols})
        return cols, wins, inds

    out = [README_START, "", "### Column dictionary", "",
           f"{len(dd)} columns. `safe` = baseline-leakage safe (see the timing caveat below the table).", "",
           "| Column | Role | Safe | Missing | Description |", "|---|---|---|---|---|"]
    for pref, label in (("wanom__", "feature_weather_anomaly"), ("wz__", "feature_weather_anomaly_z"),
                        ("wact__", "feature_weather_actual")):
        cols, wins, inds = fam(pref)
        sub = dd[dd["column"].isin(cols)]
        desc = sub["description"].iloc[0] if len(sub) else ""
        out.append(f"| `{pref}<window>__<indicator>` ({len(cols)} columns) | {label} | yes | "
                   f"{int(sub['n_missing'].min())}-{int(sub['n_missing'].max())} | {desc} |")
    cols, wins, inds = fam("wanom__")
    rest = dd[~dd["column"].str.startswith(("wanom__", "wz__", "wact__"))]
    for r in rest.itertuples(index=False):
        out.append(f"| `{r.column}` | {r.role} | {'yes' if r.baseline_leakage_safe else 'no'} | {r.n_missing} | {r.description} |")
    out += ["",
            f"Weather windows ({len(wins)}): {', '.join(f'`{w}`' for w in wins)}. "
            f"Indicators ({len(inds)}): {', '.join(f'`{i}`' for i in inds)}.", "",
            f"**Timing caveat for the weather columns:** {TIMING_CAVEAT}", "",
            README_END]
    return "\n".join(out)


def update_readme(readme_path, dd):
    if not os.path.exists(readme_path):
        return False
    text = open(readme_path, encoding="utf-8").read()
    if README_START not in text or README_END not in text:
        return False
    a, b = text.index(README_START), text.index(README_END) + len(README_END)
    open(readme_path, "w", encoding="utf-8").write(text[:a] + render_dictionary_markdown(dd) + text[b:])
    return True


def load_issue5(path):
    """Load Issue 5's own selection code so the statistics, thresholds and row filter are identical."""
    spec = importlib.util.spec_from_file_location("issue5_feature_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fc_pool(df, i5):
    """Eligible training pool in Issue 5's naming: weather z-score columns without the wz__ prefix."""
    z_cols = [c for c in df.columns if c.startswith("wz__")]
    data = df.rename(columns={c: c[len("wz__"):] for c in z_cols})
    return i5.filter_eligible_crops(data), [c[len("wz__"):] for c in z_cols]


def _select_one(pool, feats, i5, crop, target_year):
    """Issue 5 selection for one crop using ONLY target years strictly before target_year."""
    crop_pool = pool[pool["crop"] == crop]
    tr = crop_pool[crop_pool["year"] < target_year]
    if crop_pool.empty:
        # e.g. rabi_sorghum: eligible in Issue 2 but a seasonal component, which Issue 5's row filter drops
        return tr, [], [], "crop_excluded_by_issue5_row_filter"
    if tr.empty:
        return tr, [], [], "no_prior_target_years"
    selected, stats, status = i5.select_features_for_fold(tr, feats, crop, target_year)
    return tr, selected, stats, status


def forward_chaining_selection(df, i5):
    """Crop-specific predictors selected with Issue 5's method but forward-chaining folds.

    For every crop and target year T the selection sees only eligible rows of years < T, so it is
    safe to attach to rows of year T. Issue 5's published selection used leave-one-year-out folds
    (the held-out year's neighbours include later years) and is therefore not forward-chaining safe.
    """
    pool, feats = _fc_pool(df, i5)
    summary, detail = [], []
    for crop in sorted(df["crop"].unique()):
        for ty in sorted(df["year"].unique()):
            tr, selected, stats, status = _select_one(pool, feats, i5, crop, ty)
            names = [item["feature"] for item in selected]
            ran = status not in ("no_prior_target_years", "crop_excluded_by_issue5_row_filter")
            summary.append({
                "crop": crop, "year": int(ty),
                "fc_selection_train_years": ";".join(str(int(y)) for y in sorted(tr["year"].unique())),
                "fc_selection_n_training_rows": int(len(tr)),
                "fc_selection_status": status,
                "fc_n_selected_predictors": len(names) if ran else np.nan,
                "fc_selected_predictors": ";".join(names),
            })
            for item in stats:
                detail.append({
                    "crop": crop, "year": int(ty), "feature": item["feature"], "paired_n": item["paired_n"],
                    "spearman_rho": item["spearman_rho"], "p_value": item["p_value"],
                    "bh_q_value": item["bh_q_value"], "passes_screen": bool(item["passes_screen"]),
                    "selected": item["feature"] in names,
                })
    summary = pd.DataFrame(summary)
    summary["fc_n_selected_predictors"] = summary["fc_n_selected_predictors"].astype("Int64")
    cols = ["crop", "year", "feature", "paired_n", "spearman_rho", "p_value", "bh_q_value", "passes_screen", "selected"]
    return summary, pd.DataFrame(detail, columns=cols)


def selection_ignores_future(df, i5, summary):
    """Leakage test: scramble the target in every year >= T; the selection for year T must not change."""
    pool, feats = _fc_pool(df, i5)
    mismatches = []
    for r in summary.itertuples(index=False):
        p2 = pool.copy()
        future = p2["year"] >= r.year
        p2.loc[future, "yield_anomaly_zscore"] = -3.0 * p2.loc[future, "yield_anomaly_zscore"] + 11.0
        _, selected, _, status = _select_one(p2, feats, i5, r.crop, r.year)
        names = ";".join(item["feature"] for item in selected)
        if status != r.fc_selection_status or names != r.fc_selected_predictors:
            mismatches.append((r.crop, r.year))
    return mismatches


def build(paths, outdir, command):
    checks = Checks()
    os.makedirs(outdir, exist_ok=True)

    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input '{name}': {p}")
    in_hash = {k: sha256(v) for k, v in paths.items()}
    issue5_path = paths.get("issue5_script", DEFAULTS["issue5_script"])
    if not os.path.exists(issue5_path):
        raise FileNotFoundError(f"Missing Issue 5 selection code: {issue5_path}")
    in_hash["issue5_script"] = sha256(issue5_path)

    crop_long = pd.read_csv(paths["crop_long"])
    elig = pd.read_csv(paths["eligibility"])
    weather = pd.read_csv(paths["weather"])
    yield_fc = pd.read_csv(paths["yield_fc"])
    fsel = pd.read_csv(paths["feature_selection"])
    zones = pd.read_csv(paths["zones"])

    # ---- 1. key uniqueness of every input at its own grain --------------------------------
    uniq = [
        ("crop_long unique on district_code+year+crop", not crop_long.duplicated(KEY).any()),
        ("yield_fc unique on district_code+year+crop", not yield_fc.duplicated(KEY).any()),
        ("weather unique on district_code+year+weather_column",
         not weather.duplicated(DY + ["weather_column"]).any()),
        ("eligibility unique on crop", not elig.duplicated("crop").any()),
        ("zones unique on district_code", not zones.duplicated("district_code").any()),
        ("feature_selection unique on crop+feature", not fsel.duplicated(["crop", "feature"]).any()),
    ]
    for n, ok in uniq:
        checks.add(n, ok, "input key uniqueness")
        assert ok, f"Key uniqueness failed: {n}"

    # ---- 2. base rows: eligible crops with use_in_modelling (Issue 1 + Issue 2) -----------
    eligible_crops = set(elig.loc[elig["proceed_to_modelling"].astype(bool), "crop"])
    base_mask = crop_long["crop"].isin(eligible_crops) & crop_long["use_in_modelling"].astype(bool)
    expected_rows = int(base_mask.sum())  # derived from the inputs, not hard-coded
    base = crop_long[base_mask].copy()

    excl = crop_long[~base_mask].copy()
    excl["exclusion_reason"] = np.where(
        ~excl["crop"].isin(eligible_crops), "crop not eligible (proceed_to_modelling is False)",
        "use_in_modelling is False")
    excl_report = (excl.groupby(["crop", "exclusion_reason"]).size().rename("rows_excluded").reset_index())
    excl_report.to_csv(os.path.join(outdir, "exclusion_report_v1.csv"), index=False)

    # ---- 3. weather: long -> wide, one row per district-year ------------------------------
    def pivot(value, prefix):
        w = weather.pivot(index=DY, columns="weather_column", values=value)
        w.columns = [f"{prefix}{c}" for c in w.columns]
        return w.reset_index()

    w_anom = pivot("anomaly", "wanom__")
    w_z = pivot("anomaly_z", "wz__")
    w_act = pivot("actual_value", "wact__")
    weather_wide = w_anom.merge(w_z, on=DY, how="outer").merge(w_act, on=DY, how="outer")
    n_district_years = weather[DY].drop_duplicates().shape[0]
    checks.add("weather deduplicated to one row per district-year",
               len(weather_wide) == n_district_years and not weather_wide.duplicated(DY).any(),
               f"{len(weather_wide)} district-years (derived from input: {n_district_years})")
    n_features = weather["weather_column"].nunique()
    checks.add("weather feature count consistent", weather_wide.filter(like="wanom__").shape[1] == n_features,
               f"{n_features} weather features (derived from input)")

    # ---- 4. joins, each reporting matched / unmatched keys --------------------------------
    coverage, unmatched_rows = [], []

    def report(name, left, right, on, right_is_extra_info=""):
        lk = left[on].drop_duplicates()
        rk = right[on].drop_duplicates()
        m = lk.merge(rk, on=on, how="left", indicator=True)
        um_left = m[m["_merge"] == "left_only"][on]
        m2 = rk.merge(lk, on=on, how="left", indicator=True)
        um_right = m2[m2["_merge"] == "left_only"][on]
        coverage.append({
            "join": name, "join_keys": "+".join(on), "base_keys": len(lk), "source_keys": len(rk),
            "matched_base_keys": len(lk) - len(um_left), "unmatched_base_keys": len(um_left),
            "source_keys_not_in_base": len(um_right), "note": right_is_extra_info,
        })
        for _, r in um_left.iterrows():
            unmatched_rows.append({"join": name, "side": "base_without_source",
                                   "reason": "GAP: base key has no source record (stays missing)",
                                   **{k: r[k] for k in on}})
        # Source-only keys can be many (e.g. ineligible crops); tag whether that is by design.
        for _, r in um_right.iterrows():
            by_design = "crop" in on and r["crop"] not in eligible_crops
            unmatched_rows.append({"join": name, "side": "source_not_in_base",
                                   "reason": ("expected: crop not eligible in Issue 2" if by_design
                                              else "source key not present in base"),
                                   **{k: r[k] for k in on}})

    elig_cols = elig[["crop", "eligibility_class", "proceed_to_modelling", "reasons"]].rename(
        columns={"reasons": "eligibility_reasons"})
    report("eligibility", base, elig_cols, ["crop"], "source also lists ineligible crops")
    df = base.merge(elig_cols, on="crop", how="left", validate="many_to_one")

    y_cols = yield_fc[KEY + YIELD_ANOMALY_COLS]
    report("yield_anomaly", df, y_cols, KEY, "source also holds ineligible-crop rows")
    df = df.merge(y_cols, on=KEY, how="left", validate="one_to_one")

    report("weather", df, weather_wide, DY, "weather repeats across crops; joined at district-year")
    df = df.merge(weather_wide, on=DY, how="left", validate="many_to_one")

    zcols = zones[["district_code", "cluster_id", "climate_zone"]]
    report("climate_zone", df, zcols, ["district_code"], "")
    df = df.merge(zcols, on="district_code", how="left", validate="many_to_one")

    # ---- 5. Issue 5 crop-specific selected predictors (LOYO; informational only) ----------
    stable = fsel[fsel["is_stable_predictor"].astype(bool)]
    agg = (stable.groupby("crop")["feature"].agg(lambda s: ";".join(sorted(s))).rename("loyo_stable_predictors"))
    cs = (fsel.groupby("crop").agg(
        loyo_crop_selection_status=("crop_selection_status", "first"),
        loyo_n_stable_predictors=("is_stable_predictor", lambda s: int(s.astype(bool).sum())),
    ).reset_index().merge(agg.reset_index(), on="crop", how="left"))
    cs["loyo_stable_predictors"] = cs["loyo_stable_predictors"].fillna("")
    report("crop_feature_selection", df, cs, ["crop"], "Issue 5 selected with leave-one-year-out folds")
    df = df.merge(cs, on="crop", how="left", validate="many_to_one")
    # Constant property of the Issue 5 method (LOYO), true for every row regardless of match.
    df["loyo_selection_forward_chaining_safe"] = False

    # ---- 6. drop duplicate raw weather columns after verifying equality -------------------
    raw_cols = [c for c in df.columns if c.startswith(RAW_WEATHER_PREFIXES)]
    equal = True
    for c in raw_cols:
        a = df[c].to_numpy(dtype=float)
        b = df["wact__" + c].to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True):
            equal = False
            break
    checks.add("crop-long raw weather columns identical to weather actual values (dropped as duplicates)",
               equal, f"{len(raw_cols)} columns compared")
    assert equal
    df = df.drop(columns=raw_cols)

    # ---- 7. derived flags (no imputation) -------------------------------------------------
    df["is_eligible_crop"] = df["proceed_to_modelling"].astype(bool)
    df["has_valid_yield_anomaly"] = df["yield_anomaly_kg_per_ha"].notna()
    df["usable_for_forward_chaining_eval"] = df["is_eligible_crop"] & df["has_valid_yield_anomaly"]

    df = df.sort_values(["crop", "district_code", "year"]).reset_index(drop=True)

    # ---- 7b. forward-chaining crop-specific predictors (leakage-safe alternative to loyo_*) ----
    i5 = load_issue5(issue5_path)
    fc_summary, fc_detail = forward_chaining_selection(df, i5)
    report("forward_chaining_selection", df, fc_summary[["crop", "year"]], ["crop", "year"],
           "per crop and target year; uses only earlier target years")
    df = df.merge(fc_summary, on=["crop", "year"], how="left", validate="many_to_one")

    # ---- 8. validation checks -------------------------------------------------------------
    checks.add("(district_code, year, crop) unique in output", not df.duplicated(KEY).any(), f"{len(df)} rows")
    checks.add("row count reconciles to eligible crop-long rows", len(df) == expected_rows,
               f"output {len(df)} == eligible crop-long rows derived from inputs {expected_rows}")
    for r in coverage:
        if r["join"] in ("eligibility", "yield_anomaly", "weather", "climate_zone"):
            checks.add(f"join '{r['join']}' leaves no unmatched base keys", r["unmatched_base_keys"] == 0,
                       f"matched {r['matched_base_keys']}/{r['base_keys']}")
    # Issue 5 coverage differs from the Issue 2 eligible set: reported, not imputed.
    fs_cov = next(r for r in coverage if r["join"] == "crop_feature_selection")
    miss = sorted(set(base["crop"]) - set(cs["crop"]))
    extra = sorted(set(cs["crop"]) - set(base["crop"]))
    checks.add("crop_feature_selection coverage reported (unmatched stay missing)", True,
               f"matched {fs_cov['matched_base_keys']}/{fs_cov['base_keys']} eligible crops; "
               f"eligible crops with no Issue 5 result: {miss}; Issue 5 crops not eligible here: {extra}",
               kind="coverage_report")
    # missing values stay missing: NaN counts of source-derived columns equal source NaN counts
    out_anom = df.drop_duplicates(DY).filter(like="wanom__").isna().sum().sum()
    expected_nan = int(weather.merge(df[DY].drop_duplicates(), on=DY)["anomaly"].isna().sum())
    checks.add("weather anomaly missingness preserved (no imputation)", int(out_anom) == expected_nan,
               f"{int(out_anom)} NaN anomaly cells in output district-years == {expected_nan} in source")
    checks.add("yield anomaly NaN count equals source NaN count for kept rows",
               int(df["yield_anomaly_kg_per_ha"].isna().sum()) ==
               int(yield_fc.merge(df[KEY], on=KEY)["yield_anomaly_kg_per_ha"].isna().sum()),
               f"{int(df['yield_anomaly_kg_per_ha'].isna().sum())} NaN targets kept as missing")
    checks.add("no yield anomaly of exactly zero created from missing",
               not ((df["yield_anomaly_kg_per_ha"] == 0) & (df["anomaly_status"] != "calculated")).any(),
               "missing, insufficient history and true zero stay distinct")

    # ---- 9. leakage tests (forward chaining) ----------------------------------------------
    # (a) weather baselines use strictly earlier years
    wv = weather.dropna(subset=["training_years"])
    w_ok = all((max(_split_years(ty)) < y) for ty, y in zip(wv["training_years"], wv["year"]) if _split_years(ty))
    checks.add("weather anomaly baselines use only years < target year", w_ok,
               f"{len(wv)} rows with a baseline checked", kind="leakage")
    # (b) yield baselines use strictly earlier years
    yv = yield_fc.dropna(subset=["training_years"])
    y_ok = all((max(_split_years(ty)) < y) for ty, y in zip(yv["training_years"], yv["year"]) if _split_years(ty))
    checks.add("yield anomaly baselines use only years < target year", y_ok,
               f"{len(yv)} rows with a baseline checked", kind="leakage")
    # (c) recompute weather normals from the listed training years and compare
    actual = {(r.district_code, r.weather_column, r.year): r.actual_value
              for r in weather[["district_code", "weather_column", "year", "actual_value"]].itertuples()}
    avail = weather[weather["normal_status"] == "available"]
    bad = 0
    for r in avail.itertuples():
        vals = [actual.get((r.district_code, r.weather_column, ty)) for ty in _split_years(r.training_years)]
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        if not vals or not np.isclose(np.mean(vals), r.normal_mean, rtol=1e-6, atol=1e-6):
            bad += 1
    checks.add("weather normals recompute from earlier years only", bad == 0,
               f"{len(avail)} available normals recomputed; {bad} mismatches", kind="leakage")
    # (d) anomaly = actual - normal for target-year rows
    chk = weather.dropna(subset=["anomaly", "normal_mean", "actual_value"])
    checks.add("weather anomaly equals actual minus normal",
               bool(np.allclose(chk["anomaly"], chk["actual_value"] - chk["normal_mean"], atol=1e-6)),
               f"{len(chk)} rows", kind="leakage")
    # (f) forward-chaining crop selection: only earlier target years, and invariant to the future
    fc_max_ok = all(
        (not tys) or max(int(v) for v in tys.split(";")) < yr
        for tys, yr in zip(fc_summary["fc_selection_train_years"], fc_summary["year"]))
    checks.add("forward-chaining crop selection trains on target years strictly before each row's year",
               fc_max_ok, f"{len(fc_summary)} crop-year selections checked", kind="leakage")
    fc_mismatch = selection_ignores_future(df, i5, fc_summary)
    checks.add("forward-chaining crop selection is unchanged when target years >= T are scrambled",
               not fc_mismatch, f"{len(fc_summary)} selections re-run; changed: {fc_mismatch}", kind="leakage")
    status_counts = fc_summary["fc_selection_status"].value_counts().to_dict()
    checks.add("forward-chaining crop selection outcome reported (none selected is a valid result)", True,
               f"statuses: {status_counts}; crop-years with >=1 predictor: "
               f"{int((fc_summary['fc_n_selected_predictors'].fillna(0) > 0).sum())}", kind="coverage_report")

    # (e) provenance-based: the safe set must come only from the weather join, climate zone and crop,
    #     and must not intersect anything sourced from the yield file, outcome columns or Issue 5.
    dd = make_dictionary(df)
    safe_cols = set(dd.loc[dd["baseline_leakage_safe"], "column"])
    allowed = (set(weather_wide.columns) - set(DY)) | {"climate_zone", "crop"}
    forbidden = (set(YIELD_ANOMALY_COLS) | OUTCOME_COLS | (set(cs.columns) - {"crop"})
                 | {c for c in df.columns if c.startswith("loyo_")})
    checks.add("baseline-leakage-safe set comes only from weather join, climate zone and crop",
               safe_cols <= allowed, f"{len(safe_cols)} safe columns; outside allowed: {sorted(safe_cols - allowed)}",
               kind="leakage")
    checks.add("no target, outcome-year or Issue 5 (LOYO) column is marked safe",
               not (safe_cols & forbidden), f"overlap: {sorted(safe_cols & forbidden)}", kind="leakage")
    checks.add("every column documented; weather features carry the timing caveat",
               set(dd["column"]) == set(df.columns)
               and (dd.loc[dd["column"].str.startswith(("wanom__", "wz__", "wact__")), "timing_caveat"] != "").all(),
               f"{len(dd)} columns documented", kind="leakage")

    # ---- 10. write outputs ----------------------------------------------------------------
    out_csv = os.path.join(outdir, "ml_ready_dataset_v1.csv")
    df.to_csv(out_csv, index=False)
    dd.to_csv(os.path.join(outdir, "ml_ready_column_dictionary_v1.csv"), index=False)
    fc_summary.to_csv(os.path.join(outdir, "fc_crop_feature_selection_v1.csv"), index=False)
    fc_detail.to_csv(os.path.join(outdir, "fc_crop_feature_selection_detail_v1.csv"), index=False)
    readme_updated = update_readme(os.path.join(outdir, "README.md"), dd)
    pd.DataFrame(coverage).to_csv(os.path.join(outdir, "join_coverage_report_v1.csv"), index=False)
    um = pd.DataFrame(unmatched_rows)
    if "year" in um.columns:
        um["year"] = um["year"].astype("Int64")
    if "district_code" in um.columns:
        um["district_code"] = um["district_code"].astype("Int64")
    um.to_csv(os.path.join(outdir, "unmatched_keys_v1.csv"), index=False)
    cf = checks.frame()
    cf.to_csv(os.path.join(outdir, "validation_checks_v1.csv"), index=False)

    repro = {
        "command": command,
        "python": sys.version.split()[0],
        "packages": {"pandas": pd.__version__, "numpy": np.__version__},
        "reproducibility_note": ("The output is byte-identical only with the pinned package versions in "
                                 "ml_ready_dataset/requirements.txt; other versions can differ in the last float digit."),
        "input_sha256": in_hash,
        "forward_chaining_selection_config": {k: getattr(i5, k) for k in
                                              ("MIN_N", "COVERAGE", "MIN_RHO", "MAX_Q", "MAX_CORR", "MAX_FEATURES")},
        "row_counts": {
            "crop_long_rows": len(crop_long), "eligible_rows_kept": len(df), "rows_excluded": int((~base_mask).sum()),
            "eligible_crops": len(eligible_crops), "weather_district_years": n_district_years,
            "weather_features": int(n_features), "output_columns": int(df.shape[1]),
            "rows_with_valid_yield_anomaly": int(df["has_valid_yield_anomaly"].sum()),
        },
        "output_sha256": sha256(out_csv),
        "checks_passed": int(cf["passed"].sum()), "checks_total": int(len(cf)),
    }
    with open(os.path.join(outdir, "reproducibility_report_v1.json"), "w", encoding="utf-8") as f:
        json.dump(repro, f, indent=2, sort_keys=True)

    failed = cf[~cf["passed"]]
    print("=" * 70)
    print("Issue #7: ML-ready dataset")
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}  Eligible crops: {len(eligible_crops)}")
    print(f"Rows with valid forward-chaining yield anomaly: {int(df['has_valid_yield_anomaly'].sum())}")
    print(f"Checks passed: {int(cf['passed'].sum())}/{len(cf)}")
    print(f"Output SHA-256: {repro['output_sha256'][:16]}...")
    if len(failed):
        print(failed.to_string(index=False))
        raise SystemExit("Validation failed")
    return df, cf


def main():
    ap = argparse.ArgumentParser(description="Issue #7 integrated ML-ready dataset builder")
    ap.add_argument("--crop-long", default=DEFAULTS["crop_long"])
    ap.add_argument("--eligibility", default=DEFAULTS["eligibility"])
    ap.add_argument("--weather", default=DEFAULTS["weather"])
    ap.add_argument("--yield-anomalies", default=DEFAULTS["yield_fc"])
    ap.add_argument("--feature-selection", default=DEFAULTS["feature_selection"])
    ap.add_argument("--zones", default=DEFAULTS["zones"])
    ap.add_argument("--outdir", default="ml_ready_dataset")
    a = ap.parse_args()
    paths = {"crop_long": a.crop_long, "eligibility": a.eligibility, "weather": a.weather,
             "yield_fc": a.yield_anomalies, "feature_selection": a.feature_selection, "zones": a.zones}
    build(paths, a.outdir, "python " + " ".join(sys.argv))


if __name__ == "__main__":
    main()
