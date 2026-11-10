import argparse
import hashlib
import platform
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


# ============================================================
# 1. CONFIGURATION
# ============================================================

WINDOWS = (
    "annual_cal",
    "annual_crop",
    "kharif",
    "ne",
    "rabi_next",
    "rabi_prev",
)

CLIMATE_INDICATORS = (
    "rain_sum",
    "rainy_days",
    "longest_dry",
    "max_5day_rain",
    "tmax_mean",
    "tmin_mean",
    "hot_days",
    "cold_days",
    "dtr_mean",
)

CLUSTER_COUNTS = (3, 4)

FINAL_K = 3

RANDOM_STATE = 42

N_INIT = 50

STABILITY_SEEDS = tuple(range(10))

EXPECTED_DISTRICTS = 13


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

def validate_source(source_df):
    """Validate required identifiers and source data."""

    required_columns = {
        "district_code",
        "district_name",
        "year",
    }

    missing = required_columns - set(source_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if source_df.empty:
        raise ValueError("The input CSV contains no records.")

    for column in required_columns:
        if source_df[column].isna().any():
            raise ValueError(
                f"Missing values found in identifier column: {column}"
            )

    # A district code must always refer to one district name.
    names_per_code = (
        source_df.groupby("district_code")["district_name"]
        .nunique()
    )

    if (names_per_code > 1).any():
        raise ValueError(
            "A district_code is associated with multiple district names."
        )

    # A district name must not refer to multiple district codes.
    codes_per_name = (
        source_df.groupby("district_name")["district_code"]
        .nunique()
    )

    if (codes_per_name > 1).any():
        raise ValueError(
            "A district_name is associated with multiple district codes."
        )

    return source_df


def select_climate_columns(columns):

    selected = [
        f"{window}__{indicator}"
        for window in WINDOWS
        for indicator in CLIMATE_INDICATORS
    ]

    missing = [
        column for column in selected
        if column not in columns
    ]

    if missing:
        raise ValueError(
            "The source CSV is missing expected climate features:\n"
            + "\n".join(missing)
        )

    if len(selected) != 54:
        raise ValueError(
            f"Expected 54 climate features, found {len(selected)}."
        )

    return selected


def validate_numeric_features(df, climate_columns):
    result = df.copy()

    for column in climate_columns:
        converted = pd.to_numeric(
            result[column],
            errors="coerce"
        )

        invalid = result[column].notna() & converted.isna()

        if invalid.any():
            raise ValueError(
                f"Non-numeric values found in climate feature: {column}"
            )

        result[column] = converted

    values = result[climate_columns].to_numpy(dtype=float)

    if np.isinf(values).any():
        raise ValueError(
            "Infinite values found in the climate features."
        )

    return result


# ============================================================
# 3. VALIDATION-FOLD CONSISTENCY
# ============================================================

def check_fold_consistency(df, climate_columns):

    grouped = df.groupby(
        ["district_code", "year"],
        dropna=False
    )[climate_columns]

    unique_counts = grouped.nunique(dropna=False)

    inconsistent = unique_counts > 1

    if inconsistent.any().any():

        inconsistent_features = (
            inconsistent.sum()
            .loc[lambda x: x > 0]
            .to_dict()
        )

        raise ValueError(
            "Climate feature inconsistency detected across folds.\n"
            f"Affected features: {inconsistent_features}"
        )

    print("Fold consistency check: PASSED")


# ============================================================
# 4. DISTRICT-YEAR DATASET
# ============================================================

def build_district_year_dataset(
    source_df,
    climate_columns
):

    selected_columns = [
        "district_code",
        "district_name",
        "year",
    ] + climate_columns

    district_year_df = (
        source_df[selected_columns]
        .sort_values(
            ["district_code", "year"]
        )
        .drop_duplicates(
            subset=["district_code", "year"]
        )
        .reset_index(drop=True)
    )

    if district_year_df.duplicated(
        subset=["district_code", "year"]
    ).any():
        raise ValueError(
            "Duplicate district-year records remain after deduplication."
        )

    return district_year_df


# ============================================================
# 5. DISTRICT CLIMATE PROFILES
# ============================================================

def build_district_profiles(
    district_year_df,
    climate_columns
):

    district_profiles = (
        district_year_df
        .groupby(
            ["district_code", "district_name"],
            as_index=False
        )[climate_columns]
        .mean()
        .sort_values("district_code")
        .reset_index(drop=True)
    )

    if district_profiles.empty:
        raise ValueError(
            "No district climate profiles were generated."
        )
    missing_counts = (
        district_profiles[climate_columns]
        .isna()
        .sum()
    )

    missing_counts = missing_counts[
        missing_counts > 0
    ]

    if not missing_counts.empty:
        raise ValueError(
            "Missing values found in district profiles. "
            "No automatic imputation has been performed.\n"
            f"{missing_counts.to_dict()}"
        )

    values = district_profiles[
        climate_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite values found in district profiles."
        )

    return district_profiles


# ============================================================
# 6. FEATURE STANDARDIZATION
# ============================================================

def standardize_profiles(
    district_profiles,
    climate_columns
):

    X = district_profiles[
        climate_columns
    ].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Standardization produced non-finite values."
        )

    constant_features = [
        column
        for column in climate_columns
        if district_profiles[column].nunique() <= 1
    ]

    if constant_features:
        print(
            "Warning: Constant climate features detected "
            "(retained to match the agreed feature set):"
        )
        print(constant_features)

    return X_scaled


# ============================================================
# 7. INITIALIZATION STABILITY
# ============================================================

def calculate_stability(X_scaled, k):
    labels_by_seed = []

    for seed in STABILITY_SEEDS:

        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=N_INIT,
        )

        labels = model.fit_predict(X_scaled)

        labels_by_seed.append(labels)

    pairwise_ari = [
        adjusted_rand_score(
            labels_by_seed[i],
            labels_by_seed[j]
        )
        for i, j in combinations(
            range(len(labels_by_seed)),
            2
        )
    ]

    return {
        "mean_ari": float(np.mean(pairwise_ari)),
        "minimum_ari": float(np.min(pairwise_ari)),
        "maximum_ari": float(np.max(pairwise_ari)),
        "pairwise_comparisons": len(pairwise_ari),
    }


# ============================================================
# 8. K-MEANS EVALUATION
# ============================================================

def evaluate_clustering(
    X_scaled,
    source_hash,
    district_count,
    observation_count,
    feature_count,
):

    evaluation_rows = []
    fitted_models = {}

    for k in CLUSTER_COUNTS:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )

        labels = model.fit_predict(X_scaled)

        unique_labels = np.unique(labels)

        if len(unique_labels) != k:
            raise ValueError(
                f"K={k} generated an empty cluster."
            )

        distance_matrix = cdist( 
            X_scaled, 
            X_scaled,
            metric="euclidean"
        )
        silhouette = silhouette_score(
            distance_matrix,
            labels,
            metric="precomputed"
        )  

        cluster_sizes = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )

        stability = calculate_stability(
            X_scaled,
            k
        )

        evaluation_rows.append({

            "n_clusters": k,

            "silhouette_score": silhouette,

            "inertia": float(model.inertia_),

            "cluster_sizes": ";".join(
                f"{int(cluster_id)}:{int(size)}"
                for cluster_id, size
                in cluster_sizes.items()
            ),

            "mean_ari": stability["mean_ari"],

            "minimum_ari": stability["minimum_ari"],

            "maximum_ari": stability["maximum_ari"],

            "pairwise_ari_comparisons":
                stability["pairwise_comparisons"],

            "selected": k == FINAL_K,

            "selection_reason": (
                "Perfect initialization stability in the "
                "notebook evaluation; selected over K=4 "
                "despite its marginally higher silhouette."
                if k == FINAL_K
                else
                "Not selected: lower initialization stability."
            ),

            "algorithm": "KMeans",

            "scaler": "StandardScaler",

            "random_state": RANDOM_STATE,

            "n_init": N_INIT,

            "stability_seeds": ",".join(
                map(str, STABILITY_SEEDS)
            ),

            "district_count": district_count,

            "district_year_observations": observation_count,

            "feature_count": feature_count,

            "input_sha256": source_hash,

            "python_version": platform.python_version(),

            "pandas_version": pd.__version__,

            "numpy_version": np.__version__,

            "sklearn_version": sklearn.__version__,

        })

        fitted_models[k] = (
            model,
            labels
        )

    return (
        pd.DataFrame(evaluation_rows),
        fitted_models
    )


# ============================================================
# 9. FINAL DISTRICT-TO-ZONE MAPPING
# ============================================================

def generate_zone_mapping(
    district_profiles,
    final_labels
):
    mapping = district_profiles[
        ["district_code", "district_name"]
    ].copy()

    mapping["cluster_id"] = final_labels

    mapping["climate_zone"] = [
        f"Climate Zone {int(label) + 1}"
        for label in final_labels
    ]

    mapping["selected_k"] = FINAL_K

    mapping = (
        mapping
        .sort_values("district_code")
        .reset_index(drop=True)
    )

    if mapping["district_code"].duplicated().any():
        raise ValueError(
            "Duplicate districts found in final zone mapping."
        )

    if mapping["climate_zone"].isna().any():
        raise ValueError(
            "One or more districts have no climate-zone assignment."
        )

    return mapping


# ============================================================
# 10. DATASET FINGERPRINT
# ============================================================

def calculate_file_hash(input_path):

    digest = hashlib.sha256()

    with input_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 11. EXPORT
# ============================================================

def export_outputs(
    district_profiles,
    district_zone_map,
    evaluation_df,
    output_dir,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    profiles_path = (
        output_dir /
        "district_climate_profiles_v1.csv"
    )

    mapping_path = (
        output_dir /
        "district_zone_map_v1.csv"
    )

    evaluation_path = (
        output_dir /
        "clustering_evaluation_v1.csv"
    )

    district_profiles.to_csv(
        profiles_path,
        index=False
    )

    district_zone_map.to_csv(
        mapping_path,
        index=False
    )

    evaluation_df.to_csv(
        evaluation_path,
        index=False
    )

    return (
        profiles_path,
        mapping_path,
        evaluation_path
    )


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible Issue #6 district climate zoning."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to weather_stress_features_v1.csv"
    )

    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory for the three output CSV files."
    )

    parser.add_argument(
        "--expected-districts",
        type=int,
        default=EXPECTED_DISTRICTS,
        help=(
            "Expected number of districts. "
            "Use 0 to disable this check."
        )
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    ).expanduser().resolve()

    output_dir = Path(
        args.outdir
    ).expanduser().resolve()

    print("\n========================================")
    print(" ISSUE #6: DISTRICT CLIMATE ZONING")
    print("========================================\n")

    # --------------------------------------------------------
    # Step 1: Validate input file
    # --------------------------------------------------------

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    source_hash = calculate_file_hash(
        input_path
    )

    source_df = pd.read_csv(
        input_path
    )

    validate_source(
        source_df
    )

    print(f"Source file: {input_path}")
    print(f"Source rows: {len(source_df)}")

    # --------------------------------------------------------
    # Step 2: Select the 54 agreed climate features
    # --------------------------------------------------------

    climate_columns = select_climate_columns(
        source_df.columns
    )

    source_df = validate_numeric_features(
        source_df,
        climate_columns
    )

    print(
        f"Selected climate features: {len(climate_columns)}"
    )

    # --------------------------------------------------------
    # Step 3: Verify fold consistency
    # --------------------------------------------------------

    check_fold_consistency(
        source_df,
        climate_columns
    )

    # --------------------------------------------------------
    # Step 4: Deduplicate district-year records
    # --------------------------------------------------------

    district_year_df = build_district_year_dataset(
        source_df,
        climate_columns
    )

    print(
        "Unique district-year observations:",
        len(district_year_df)
    )

    # --------------------------------------------------------
    # Step 5: Generate district profiles
    # --------------------------------------------------------

    district_profiles = build_district_profiles(
        district_year_df,
        climate_columns
    )

    district_count = len(district_profiles)

    print(
        f"District climate profiles generated: {district_count}"
    )

    if (
        args.expected_districts > 0
        and district_count != args.expected_districts
    ):
        raise ValueError(
            f"Expected {args.expected_districts} districts, "
            f"but generated {district_count}. "
            "Check the input dataset or use "
            "--expected-districts to change the expected count."
        )

    # --------------------------------------------------------
    # Step 6: Standardize climate features
    # --------------------------------------------------------

    X_scaled = standardize_profiles(
        district_profiles,
        climate_columns
    )

    print("Standardization: PASSED")

    # --------------------------------------------------------
    # Step 7: Evaluate K=3 and K=4
    # --------------------------------------------------------

    evaluation_df, fitted_models = evaluate_clustering(
        X_scaled=X_scaled,
        source_hash=source_hash,
        district_count=district_count,
        observation_count=len(district_year_df),
        feature_count=len(climate_columns),
    )

    print("\nCLUSTERING EVALUATION")
    print("----------------------------------------")

    print(
        evaluation_df[
            [
                "n_clusters",
                "silhouette_score",
                "inertia",
                "mean_ari",
                "minimum_ari",
                "maximum_ari",
                "selected",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Step 8: Apply the agreed K=3 solution
    # --------------------------------------------------------

    if FINAL_K not in fitted_models:
        raise ValueError(
            "The final K configuration was not evaluated."
        )

    final_model, final_labels = fitted_models[
        FINAL_K
    ]

    district_zone_map = generate_zone_mapping(
        district_profiles,
        final_labels
    )

    # --------------------------------------------------------
    # Step 9: Validate final district assignments
    # --------------------------------------------------------

    if len(district_zone_map) != district_count:
        raise ValueError(
            "The number of zone assignments does not "
            "match the number of districts."
        )

    if district_zone_map["cluster_id"].isna().any():
        raise ValueError(
            "Missing cluster assignments detected."
        )

    # --------------------------------------------------------
    # Step 10: Export deliverables
    # --------------------------------------------------------

    output_paths = export_outputs(
        district_profiles=district_profiles,
        district_zone_map=district_zone_map,
        evaluation_df=evaluation_df,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # Step 11: Final report
    # --------------------------------------------------------

    print("\n========================================")
    print(" EXECUTION COMPLETED SUCCESSFULLY")
    print("========================================")

    print(f"\nDistricts processed: {district_count}")
    print(f"Climate features: {len(climate_columns)}")
    print(f"District-year records: {len(district_year_df)}")
    print(f"Selected clustering configuration: K={FINAL_K}")

    print("\nFINAL DISTRICT-TO-CLIMATE-ZONE MAPPING")
    print("----------------------------------------")

    print(
        district_zone_map.to_string(index=False)
    )

    print("\nGENERATED DELIVERABLES")
    print("----------------------------------------")

    for path in output_paths:
        print(f"[OK] {path}")

    print("\nInput SHA-256:")
    print(source_hash)

    print("\nNote:")
    print(
        "Climate-zone numbers are categorical labels, "
        "not rankings of agricultural risk or climate severity."
    )

    print(
        "ARI measures stability across random initializations; "
        "it does not establish robustness to new datasets or years."
    )

    print(
        "\nAll three Issue #6 CSV deliverables have been generated."
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nEXECUTION FAILED")
        print("----------------------------------------")
        print(f"{type(error).__name__}: {error}")

        sys.exit(1)
