#!/usr/bin/env python3
"""
folds.py - year-agnostic forward-chaining fold generation for the Issue #50 pipeline.

The pipeline used to carry hand-written folds for the five years 2015-2019. This module builds the folds
from whatever years the ML-ready dataset contains, so the same code works on a longer history.

Terms
  all_years           every year present in the dataset
  years_with_targets  years that have at least one usable forward-chaining yield anomaly. An anomaly needs
                      two earlier years of history, so the first years of any dataset have no target.
                      Such years can be listed as training years but contribute no training rows.

Modes (set in model_config.json -> validation.mode)
  expanding     (recommended for a long history) every year with targets that has at least
                `min_train_years_with_targets` earlier years with targets is a test year. Each fold trains on
                ALL years before its test year. With N usable years this gives about N-1 folds.
  fixed_origin  one training window, every later year is a test year:
                train = all years <= `train_end_year`; test = each year with targets after it.
                (This is the design used for the current 2015-2019 data: train_end_year 2017.)
  explicit      the folds are listed by hand under validation.folds.

Strict forward chaining is enforced for every mode: every training year is earlier than the test year.
"""

MODES = ("expanding", "fixed_origin", "explicit")


def _resolve_mode(cfg):
    mode = cfg.get("mode")
    if mode is None:
        mode = "explicit" if "folds" in cfg else "expanding"
    if mode not in MODES:
        raise ValueError(f"validation.mode must be one of {MODES}, got {mode!r}")
    return mode


def generate_folds(validation_cfg, years_with_targets, all_years):
    """Return a list of {"fold", "train_years", "test_year"} dicts built from the data's own years."""
    all_years = sorted(int(y) for y in all_years)
    years_with_targets = sorted(int(y) for y in years_with_targets)
    if not years_with_targets:
        raise ValueError("No year has a usable forward-chaining target; cannot build folds.")
    mode = _resolve_mode(validation_cfg)

    if mode == "explicit":
        raw = validation_cfg.get("folds")
        if not raw:
            raise ValueError("mode 'explicit' needs a non-empty validation.folds list")
        folds = []
        for i, f in enumerate(raw, start=1):
            folds.append({"fold": int(f.get("fold", i)),
                          "train_years": sorted(int(y) for y in f["train_years"]),
                          "test_year": int(f["test_year"])})
        return folds

    if mode == "fixed_origin":
        if "train_end_year" not in validation_cfg:
            raise ValueError("mode 'fixed_origin' needs validation.train_end_year")
        train_end = int(validation_cfg["train_end_year"])
        train_years = [y for y in all_years if y <= train_end]
        test_years = [y for y in years_with_targets if y > train_end]
        if not train_years:
            raise ValueError(f"No data years on or before train_end_year={train_end}")
        if not test_years:
            raise ValueError(f"No test year with targets after train_end_year={train_end}")
        return [{"fold": i, "train_years": list(train_years), "test_year": ty}
                for i, ty in enumerate(test_years, start=1)]

    # expanding
    min_train = int(validation_cfg.get("min_train_years_with_targets", 1))
    first_test = validation_cfg.get("first_test_year")
    last_test = validation_cfg.get("last_test_year")
    folds = []
    for idx, ty in enumerate(years_with_targets):
        if idx < min_train:
            continue  # not enough earlier years with targets to train on
        if first_test is not None and ty < int(first_test):
            continue
        if last_test is not None and ty > int(last_test):
            continue
        folds.append({"fold": len(folds) + 1, "train_years": [y for y in all_years if y < ty], "test_year": ty})
    if not folds:
        raise ValueError("The expanding mode produced no folds; check min_train_years_with_targets and the year range")
    return folds


def validate_folds(folds, df_eval, min_training_rows=8):
    """Raise ValueError unless the folds are leak-free and usable. df_eval = rows that have a target."""
    if not folds:
        raise ValueError("No folds to validate")
    test_years = [f["test_year"] for f in folds]
    if len(set(test_years)) != len(test_years):
        raise ValueError(f"Duplicate test years in folds: {test_years}")
    summary = []
    for f in folds:
        ty = f["test_year"]
        if not f["train_years"]:
            raise ValueError(f"Fold {f['fold']} has no training years")
        if max(f["train_years"]) >= ty:
            raise ValueError(f"Fold {f['fold']} leaks: training year {max(f['train_years'])} is not before test year {ty}")
        n_test = int((df_eval["year"] == ty).sum())
        if n_test == 0:
            raise ValueError(f"Fold {f['fold']}: test year {ty} has no rows with a target")
        train = df_eval[df_eval["year"].isin(f["train_years"])]
        if len(train) < min_training_rows:
            raise ValueError(
                f"Fold {f['fold']}: only {len(train)} training rows with targets (minimum {min_training_rows}). "
                f"Configured training years {f['train_years']} contribute rows only for "
                f"{sorted(int(y) for y in train['year'].unique())}.")
        summary.append({"fold": f["fold"], "test_year": ty, "n_test": n_test, "n_train": int(len(train)),
                        "train_years_with_targets": sorted(int(y) for y in train["year"].unique())})
    return summary
