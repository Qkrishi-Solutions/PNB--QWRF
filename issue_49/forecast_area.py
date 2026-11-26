"""Issue 49 annual sown-area backtest."""

import argparse
import csv
import hashlib
import math
import platform
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import linear_regression, mean


HERE = Path(__file__).resolve().parent
CROP_LONG = Path("CropLongFormatv1/crop_long_v1.csv")
STATE_CALENDAR = Path("crop_calendar/crop_calendar_v1.csv")
DISTRICT_CALENDAR = Path("crop_calendar/district_crop_calendar_v1.csv")
ML_READY = Path("ml_ready_dataset/ml_ready_dataset_v1.csv")
DICTIONARY = Path("ml_ready_dataset/ml_ready_column_dictionary_v1.csv")
MANIFEST = Path("ml_ready_dataset/reproducibility_report_v1.json")
INPUTS = (CROP_LONG, STATE_CALENDAR, DISTRICT_CALENDAR, ML_READY, DICTIONARY, MANIFEST)
BASELINES = {"last_value": 1, "climatology": 2, "trend": 3}
MODELS = ("ridge_area_trend", "ridge_area_trend_weather")
MONTH = {name: number for number, name in enumerate(
    "January February March April May June July August September October November December".split(), 1
)}
RAIN = "wact__kharif__rain_sum"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["district_code"], row["crop"], int(row["year"])


def unique_rows(records, make_key):
    indexed = {}
    for record in records:
        ident = make_key(record)
        if ident in indexed:
            raise ValueError(f"Duplicate input key {ident}")
        indexed[ident] = record
    return indexed


def number(value):
    if not value.strip():
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def area(row):
    value = number(row["area_1000_ha"])
    if value is None:
        return None, "missing_area"
    if value < 0:
        return None, "negative_area"
    if value == 0 and row["flag_nan_filled_with_zero"].casefold() == "true":
        return None, "ambiguous_zero"
    return value * 1000, ""


def calendars(inputs):
    state = unique_rows(
        (r for r in rows(inputs[STATE_CALENDAR]) if r["status"] == "verified"),
        lambda r: (r["crop"], r["season"]),
    )
    district = unique_rows(
        (r for r in rows(inputs[DISTRICT_CALENDAR]) if r["status"] == "verified"),
        lambda r: (r["district_code"], r["crop"], r["season"]),
    )
    return state, district


def provisional_cutoff(ident, state, district):
    code, crop, year = ident
    dates = []
    for (name, season), statewide in state.items():
        if name != crop:
            continue
        override = district.get((code, crop, season))
        if override:
            month, day = override["typical_sowing_window"].split(" - ", 1)[0].split()
            dates.append(date(year, MONTH[month], int(day)))
        else:
            dates.append(date(year, MONTH[statewide["sowing_window_start"]], 1))
    return min(dates) - timedelta(days=1) if dates else None


def load_inputs(inputs):
    crop_long = unique_rows(rows(inputs[CROP_LONG]), key)
    ml = unique_rows(rows(inputs[ML_READY]), key)
    dictionary = unique_rows(rows(inputs[DICTIONARY]), lambda r: r["column"])
    if dictionary[RAIN]["role"] != "feature_weather_actual":
        raise ValueError(f"Unexpected Issue 7 role for {RAIN}")
    series = defaultdict(dict)
    rainfall = {}
    for ident, record in ml.items():
        source = crop_long[ident]
        for column in ("area_1000_ha", "flag_area_zero", "flag_nan_filled_with_zero"):
            parse = number if column == "area_1000_ha" else str.casefold
            if parse(source[column]) != parse(record[column]):
                raise ValueError(f"Area input mismatch for {ident} in {column}")
        code, crop, year = ident
        series[code, crop][year] = area(record)
        value = number(record[RAIN])
        if (code, year) in rainfall and rainfall[code, year] != value:
            raise ValueError(f"Conflicting district-year weather for {(code, year)}")
        rainfall[code, year] = value
    return series, rainfall, calendars(inputs)


def load_timing(path, series):
    if path is None:
        return {}
    timing = unique_rows(rows(path), key)
    expected = {(code, crop, year) for (code, crop), history in series.items() for year in history}
    if set(timing) != expected:
        raise ValueError("Timing evidence must cover exactly the benchmark keys")
    for row in timing.values():
        if not row["source_reference"].strip():
            raise ValueError(f"Missing timing evidence reference for {key(row)}")
        for field in ("forecast_date", "area_available_on", "lagged_kharif_available_on"):
            row[field] = date.fromisoformat(row[field]) if row[field].strip() else None
        if row["forecast_date"] is None:
            raise ValueError(f"Missing confirmed forecast date for {key(row)}")
        published = row["area_available_on"]
        if published is not None and published <= row["forecast_date"]:
            raise ValueError(f"Annual area release must follow the forecast for {key(row)}")
        weather_release = row["lagged_kharif_available_on"]
        if weather_release is not None and weather_release < date(key(row)[2] - 1, 9, 30):
            raise ValueError(f"Kharif rainfall release predates the window end for {key(row)}")
    return timing


def history_before(series, ident, timing):
    code, crop, year = ident
    cutoff = timing[ident]["forecast_date"] if timing else None
    available = []
    for earlier, (value, _) in sorted(series[code, crop].items()):
        if earlier >= year or value is None:
            continue
        if cutoff is not None:
            released = timing[code, crop, earlier]["area_available_on"]
            if released is None or released >= cutoff:
                continue
        available.append((earlier, value))
    return available


def baseline_forecast(method, history, year):
    if len(history) < BASELINES[method]:
        return None
    if method == "last_value":
        return history[-1][1]
    years, values = zip(*history)
    if method == "climatology":
        return mean(values)
    slope, intercept = linear_regression(years, values)
    return max(0.0, slope * year + intercept)


def weather_features(ident, rainfall, timing, history):
    code, _, year = ident
    if not timing or not history:
        return None
    release = timing[ident]["lagged_kharif_available_on"]
    rain = rainfall.get((code, year - 1))
    if release is None or release >= timing[ident]["forecast_date"] or rain is None:
        return None
    return history[-1][1], year, rain


def fit_ridge(training, target):
    import numpy as np

    matrix = np.array([features for _, features, _ in training], dtype=float)
    outcome = np.array([value for _, _, value in training], dtype=float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1
    standard = (matrix - center) / scale
    coefficients = np.linalg.solve(
        standard.T @ standard + np.eye(matrix.shape[1]), standard.T @ (outcome - outcome.mean())
    )
    return max(0.0, float(outcome.mean() + ((target - center) / scale) @ coefficients))


def candidate_training(features, series, timing, ident):
    training = []
    for earlier, predictors in features.items():
        code, crop, year = earlier
        if crop != ident[1] or year >= ident[2] or predictors is None:
            continue
        actual, _ = series[code, crop][year]
        if actual is None:
            continue
        if timing:
            release = timing[earlier]["area_available_on"]
            if release is None or release >= timing[ident]["forecast_date"]:
                continue
        training.append((earlier, predictors, actual))
    return training


def forecast(series, rainfall, calendar, timing, test_years, methods):
    features = {method: {} for method in MODELS if method in methods}
    if features:
        for (code, crop), observations in series.items():
            for year in observations:
                ident = code, crop, year
                history = history_before(series, ident, timing)
                if MODELS[0] in features:
                    features[MODELS[0]][ident] = (history[-1][1], year) if history else None
                if MODELS[1] in features:
                    features[MODELS[1]][ident] = weather_features(ident, rainfall, timing, history)
    for (code, crop), observations in sorted(series.items()):
        for year in test_years:
            if year not in observations:
                continue
            ident = code, crop, year
            history = history_before(series, ident, timing)
            actual, actual_issue = observations[year]
            cutoff = timing[ident]["forecast_date"] if timing else provisional_cutoff(ident, *calendar)
            for method in methods:
                train_years = [period for period, _ in history]
                train_rows = len(history)
                estimate = None
                if method in BASELINES:
                    estimate = baseline_forecast(method, history, year)
                    reason = "" if estimate is not None else "insufficient_area_history"
                else:
                    reason = "unverified_or_missing_weather" if method == MODELS[1] else "insufficient_area_history"
                    train_years, train_rows = [], 0
                    predictors = features[method][ident]
                    if predictors is not None:
                        training = candidate_training(features[method], series, timing, ident)
                        train_years = sorted({sample[0][2] for sample in training})
                        train_rows = len(training)
                        districts = {sample[0][0] for sample in training}
                        reason = "insufficient_candidate_training"
                        if train_rows >= 10 and len(districts) >= 5 and len(train_years) >= 2:
                            estimate = fit_ridge(training, predictors)
                            reason = ""
                yield {
                    "district_code": code, "crop": crop, "year": year,
                    "forecast_date": cutoff.isoformat() if cutoff else "",
                    "timing_status": ("confirmed" if timing else
                                      "provisional_calendar" if cutoff else "calendar_unavailable"),
                    "method": method, "observed_area_ha": actual,
                    "predicted_area_ha": round(estimate, 6) if estimate is not None else None,
                    "status": "scored" if estimate is not None and actual is not None else "unscored",
                    "reason": ";".join(filter(None, (actual_issue, reason))),
                    "training_years": ";".join(map(str, train_years)), "training_rows": train_rows,
                }


def score(predictions, methods, cohort, matched=True):
    groups = defaultdict(dict)
    for record in predictions:
        if record["method"] in methods:
            groups[key(record)][record["method"]] = record
    scopes = defaultdict(list)
    for ident, records in groups.items():
        scored = {method for method, row in records.items() if row["status"] == "scored"}
        for method in methods:
            row = records[method]
            for crop, year in ((ident[1], ident[2]), (ident[1], "all"), ("all", ident[2]), ("all", "all")):
                scopes[crop, year, method].append((row, method in scored and (
                    not matched or len(scored) == len(methods)
                )))
    for (crop, year, method), entries in sorted(scopes.items(), key=lambda item: str(item[0])):
        scored = [row for row, included in entries if included]
        districts = len({row["district_code"] for row in scored})
        years = len({row["year"] for row in scored})
        flags = []
        if len(scored) < 10 or districts < 5:
            flags.append("low_sample")
        if year == "all" and years < 2:
            flags.append("low_temporal_coverage")
        yield {
            "cohort": cohort, "crop": crop, "year": year, "method": method,
            "mae_ha": round(mean(abs(row["predicted_area_ha"] - row["observed_area_ha"])
                                 for row in scored), 6) if scored else None,
            "n_scored": len(scored), "n_targets": len(entries), "n_districts": districts,
            "n_test_years": years, "coverage": round(len(scored) / len(entries), 6),
            "sample_flags": ";".join(flags),
        }


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def candidate_summary(metrics, cohort, candidate):
    totals = defaultdict(dict)
    for row in metrics:
        if row["year"] == "all" and row["cohort"] == cohort:
            totals[row["crop"]][row["method"]] = row
    lines = [
        "| Crop | Scored rows | Best naive baseline | Baseline MAE (ha) | Ridge MAE (ha) | Flags |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for crop, methods in totals.items():
        result = methods[candidate]
        flags = result["sample_flags"]
        count = result["n_scored"]
        if not count:
            lines.append(f"| {crop} | 0 | — | — | — | {flags} |")
            continue
        best = min(BASELINES, key=lambda name: methods[name]["mae_ha"])
        reference = methods[best]["mae_ha"]
        error = result["mae_ha"]
        lines.append(f"| {crop} | {count} | {best} | {reference:,.2f} | {error:,.2f} | {flags} |")
        if crop == "all":
            verdict = "beat" if error < reference else "did not beat"
            lines[:0] = [f"Across {count} matched rows, Ridge {verdict} {best} "
                         f"({error:,.2f} vs {reference:,.2f} ha MAE).", ""]
    return lines


def write_report(path, metrics, inputs, args, timing_count):
    lines = ["# Issue 49 annual sown-area backtest", "",
             "## Matched baseline results", "",
             "| Crop | Scored rows | Last value MAE (ha) | Climatology MAE (ha) | Trend MAE (ha) | Flags |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    totals = defaultdict(dict)
    for row in metrics:
        if row["year"] == "all" and row["cohort"] == "matched_baselines":
            totals[row["crop"]][row["method"]] = row
    for crop, methods in totals.items():
        first = methods["last_value"]
        values = [f"{methods[name]['mae_ha']:,.2f}" if methods[name]["mae_ha"] is not None else "—"
                  for name in BASELINES]
        lines.append(f"| {crop} | {first['n_scored']} | {' | '.join(values)} | {first['sample_flags']} |")
    lines.extend(["", "## Area-only Ridge", ""])
    lines.extend(candidate_summary(metrics, "matched_area_ridge", MODELS[0]))
    lines.extend(["", "## Weather comparison", ""])
    if not timing_count:
        lines.append("Weather candidate not evaluated. No dated records corroborate the annual-area publication dates or the historical release of the exact rainfall product in the Issue 7 export. Weather improvement over a naive baseline is unestablished.")
        lines.append("")
        lines.append("The Issue 7 column dictionary leaves forecast-time availability unresolved. The [IMD rainfall archive](https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html) describes the historical product and yearly files, with no vintage dates for the export used here.")
    elif not any(r["n_scored"] for r in metrics if r["cohort"] == "matched_weather"):
        lines.append("No rows met the matched candidate coverage and training rules. Weather improvement over a naive baseline is unestablished.")
    else:
        lines.extend(candidate_summary(metrics, "matched_weather", MODELS[1]))
        lines.extend(["", "The metrics CSV also gives the area-only Ridge's MAE on these same held-out rows. "
                      "The two Ridge models may use different training rows when historical weather is missing."])
    import numpy as np

    lines.extend(["", "## Scope and assumptions", "",
                  "- The target is annual district-crop sown area. The 11 crops in Issue 7 define the benchmark.",
                  "- Forecast and metric files contain retrospective backtest results. Two test years provide limited evidence. Scores do not establish a validated operational model.",
                  "- Missing and negative targets and ambiguous zeros are excluded. An observed zero remains zero.",
                  "- Per-crop sample flags use fewer than 10 scored rows, fewer than five districts or fewer than two test years for pooled results.",
                  "- Ridge is fit per crop across districts using recent observed area and calendar year. "
                  "Each fit uses its own training means and scales, with a fixed penalty of 1.",
                  "- Ridge requires ten training rows across at least five districts and two earlier target years.",
                  f"- Forecast years {', '.join(map(str, args.test_years))}. "
                  f"Python {platform.python_version()}, NumPy {np.__version__}."])
    if not timing_count:
        lines.append("- The baselines and area-only Ridge assume strictly earlier reported annual area was "
                     "available. Calendar cutoffs are provisional because crop-year alignment and historical "
                     "publication dates remain unverified.")
    lines.extend(["", "## Input SHA-256 with LF line endings", ""])
    for name, source in inputs.items():
        digest = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        lines.append(f"- `{name.as_posix()}` {digest}")
    if args.timing:
        digest = hashlib.sha256(args.timing.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        lines.append(f"- `{args.timing.name}` {digest}")
    lines.extend(["", "## Reproduction", "", "```text",
                  args.command, "python -B -m unittest discover -s issue_49/tests -v", "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--repo-root", type=Path, default=HERE.parent, help="Root of the canonical repository inputs")
    source.add_argument("--inputs", type=Path, help="Directory containing flat copies of canonical inputs")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    parser.add_argument("--test-years", type=int, nargs="+", default=[2018, 2019])
    parser.add_argument("--timing", type=Path, help="Verified forecast and historical release dates")
    args = parser.parse_args()
    if len(set(args.test_years)) != len(args.test_years) or args.test_years != sorted(args.test_years):
        parser.error("Test years must be unique and in ascending order")
    inputs = {name: args.inputs / name.name if args.inputs else args.repo_root / name for name in INPUTS}
    series, rainfall, calendar = load_inputs(inputs)
    if any(not any(year in history for history in series.values()) for year in args.test_years):
        parser.error("All test years must occur in Issue 7")
    timing = load_timing(args.timing, series)
    candidates = MODELS if timing else MODELS[:1]
    predictions = list(forecast(series, rainfall, calendar, timing, args.test_years, (*BASELINES, *candidates)))
    metrics = list(score(predictions, BASELINES, "matched_baselines"))
    metrics.extend(score(predictions, BASELINES, "available_baselines", matched=False))
    metrics.extend(score(predictions, (*BASELINES, MODELS[0]), "matched_area_ridge"))
    metrics.extend(score(predictions, candidates, "available_candidates", matched=False))
    if timing:
        metrics.extend(score(predictions, (*BASELINES, *MODELS), "matched_weather"))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "area_forecasts.csv", predictions)
    write_csv(args.out / "area_metrics.csv", metrics)
    args.command = "python -B issue_49/forecast_area.py"
    if args.inputs:
        args.command += f' --inputs "{args.inputs.as_posix()}"'
    elif args.repo_root != HERE.parent:
        args.command += f' --repo-root "{args.repo_root.as_posix()}"'
    args.command += f' --out "{args.out.as_posix()}"'
    args.command += " --test-years " + " ".join(map(str, args.test_years))
    if args.timing:
        args.command += f' --timing "{args.timing}"'
    write_report(args.out / "REPORT.md", metrics, inputs, args, len(timing))
    print(f"Wrote {len(predictions)} forecast rows and {len(metrics)} metric rows to {args.out}")


if __name__ == "__main__":
    main()
