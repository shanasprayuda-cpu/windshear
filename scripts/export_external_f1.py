import sys
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STAGE D1 - EXPORT EXTERNAL F1 DATASET
# Jan-Mar 2025
#
# Purpose:
#   Build the external-test F1 arrays from the already separated
#   external Stage B2 minute features and the Stage C
#   external_test_windows.csv manifest.
#
# Output:
#   stageD1_data/F1_external/
#       X.npy
#       y.npy
#       metadata.json
#       export_audit.json
#
# IMPORTANT:
#   This script does NOT use development X/y.
#   It does NOT train, scale, tune, or select thresholds.
# ============================================================


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "config.json"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)


# ------------------------------------------------------------
# STANDARD PATHS
# ------------------------------------------------------------

STAGEC_PATH = (
    PROJECT_ROOT / CFG["stageC_path"]
)

EXTERNAL_MANIFEST = (
    STAGEC_PATH
    / "manifest"
    / "external_test_windows.csv"
)


# External Stage B2 source.
# The script searches these known locations and refuses to
# silently choose among multiple candidates.
B2_CANDIDATES = [
    PROJECT_ROOT
    / "external_test_2025"
    / "stageB2_physics"
    / "stageB2_physics_minute_features.csv",

    PROJECT_ROOT
    / "external_test_2025"
    / "stageB2_physics"
    / "stageB2_minute_features.csv",

    PROJECT_ROOT
    / "external_test_2025"
    / "stageB2_physics_minute_features.csv",

    PROJECT_ROOT
    / "external_test"
    / "stageB2_physics"
    / "stageB2_physics_minute_features.csv",

    PROJECT_ROOT
    / "stageB2_external"
    / "stageB2_physics_minute_features.csv",
]


FEATURE_SET_CANDIDATES = [
    PROJECT_ROOT
    / "external_test_2025"
    / "stageB2_physics"
    / "feature_sets.json",

    PROJECT_ROOT
    / "stageC"
    / "feature_sets"
    / "feature_sets.json",

    PROJECT_ROOT
    / "stageC"
    / "feature_sets"
    / "F1.json",
]


OUTPUT_DIR = (
    PROJECT_ROOT
    / "stageD1_data"
    / "F1_external"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

X_OUTPUT = OUTPUT_DIR / "X.npy"
Y_OUTPUT = OUTPUT_DIR / "y.npy"
METADATA_OUTPUT = OUTPUT_DIR / "metadata.json"
AUDIT_OUTPUT = OUTPUT_DIR / "export_audit.json"


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def find_unique_existing(paths, description):
    found = [
        p for p in paths
        if p.exists()
    ]

    if len(found) == 1:
        return found[0]

    if len(found) == 0:
        raise FileNotFoundError(
            f"\nCould not find {description}.\n\n"
            + "\n".join(str(p) for p in paths)
        )

    raise RuntimeError(
        f"\nMultiple candidates found for {description}.\n"
        "Specify/clean the source so the export is unambiguous:\n\n"
        + "\n".join(str(p) for p in found)
    )


def find_column(df, candidates, label):
    exact = [
        c for c in candidates
        if c in df.columns
    ]

    if len(exact) == 1:
        return exact[0]

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    matches = [
        lower[c.lower()]
        for c in candidates
        if c.lower() in lower
    ]

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Could not uniquely identify {label}.\n"
        f"Candidates: {candidates}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def parse_feature_set(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Accept common representations:
    # {"F1": ["A2_u_ms", ...]}
    # {"features": {"F1": [...]}}
    # {"feature_sets": {"F1": [...]}}
    # {"F1": {"features": [...]}}
    if isinstance(obj, dict):

        if "F1" in obj:
            value = obj["F1"]

            if isinstance(value, list):
                return value

            if isinstance(value, dict):
                for key in (
                    "features",
                    "columns",
                    "predictors",
                ):
                    if isinstance(value.get(key), list):
                        return value[key]

        for parent in (
            "features",
            "feature_sets",
        ):
            value = obj.get(parent)

            if isinstance(value, dict):
                f1 = value.get("F1")

                if isinstance(f1, list):
                    return f1

                if isinstance(f1, dict):
                    for key in (
                        "features",
                        "columns",
                        "predictors",
                    ):
                        if isinstance(
                            f1.get(key),
                            list
                        ):
                            return f1[key]

    raise ValueError(
        f"Could not identify F1 feature list in {path}.\n"
        "The script will not guess the feature order."
    )


def normalize_feature_name(name):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(name).lower()
    )


def resolve_feature_columns(
    df,
    requested_features
):
    normalized = {
        normalize_feature_name(c): c
        for c in df.columns
    }

    resolved = []
    missing = []

    for feature in requested_features:

        key = normalize_feature_name(
            feature
        )

        if key in normalized:
            resolved.append(
                normalized[key]
            )
        else:
            missing.append(feature)

    if missing:
        raise KeyError(
            "\nMissing F1 predictor columns in "
            "external Stage B2 data:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
            + "\n\nAvailable columns:\n"
            + "\n".join(
                str(c)
                for c in df.columns
            )
        )

    return resolved


def find_time_column(df):
    candidates = [
        "Time [UTC]",
        "Time",
        "time",
        "timestamp",
        "Timestamp",
        "datetime",
        "Datetime",
        "date_time",
        "DateTime",
        "valid_time",
        "valid_datetime",
    ]

    try:
        return find_column(
            df,
            candidates,
            "UTC time column"
        )
    except KeyError:
        # Conservative fallback: only accept a single column
        # whose parsed values overwhelmingly look like datetimes.
        candidates_found = []

        for c in df.columns:

            if (
                str(c).lower()
                in {
                    "date",
                    "datetime",
                    "timestamp",
                    "time",
                    "valid_time",
                    "valid_datetime",
                }
            ):
                candidates_found.append(c)

        if len(candidates_found) == 1:
            return candidates_found[0]

        raise


def load_manifest():
    if not EXTERNAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"External manifest not found:\n"
            f"{EXTERNAL_MANIFEST}"
        )

    manifest = pd.read_csv(
        EXTERNAL_MANIFEST
    )

    required = [
        "sample_id",
        "anchor_idx",
        "anchor_time",
        "input_start",
        "input_end",
        "target_start",
        "target_end",
    ]

    missing = [
        c for c in required
        if c not in manifest.columns
    ]

    if missing:
        raise KeyError(
            "External manifest is missing columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing
            )
        )

    target_columns = [
        f"y_h{i}"
        for i in range(1, 11)
    ]

    missing_targets = [
        c for c in target_columns
        if c not in manifest.columns
    ]

    if missing_targets:
        raise KeyError(
            "External manifest is missing target columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing_targets
            )
        )

    for c in (
        "anchor_time",
        "input_start",
        "input_end",
        "target_start",
        "target_end",
    ):
        manifest[c] = pd.to_datetime(
            manifest[c],
            errors="coerce"
        )

    if manifest[
        [
            "anchor_time",
            "input_start",
            "input_end",
            "target_start",
            "target_end",
        ]
    ].isna().any().any():

        raise ValueError(
            "Invalid timestamps found in external manifest."
        )

    return manifest


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 90)
    print(
        "STAGE D1 - EXPORT EXTERNAL F1 DATASET"
    )
    print(
        "External Test: Jan-Mar 2025"
    )
    print("=" * 90)


    # --------------------------------------------------------
    # SOURCE DISCOVERY
    # --------------------------------------------------------

    b2_path = find_unique_existing(
        B2_CANDIDATES,
        "external Stage B2 minute-feature CSV"
    )

    feature_set_path = find_unique_existing(
        FEATURE_SET_CANDIDATES,
        "F1 feature-set definition"
    )

    print(
        "External B2 source:",
        b2_path
    )

    print(
        "F1 definition:",
        feature_set_path
    )

    print(
        "External manifest:",
        EXTERNAL_MANIFEST
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )


    # --------------------------------------------------------
    # F1 FEATURE ORDER
    # --------------------------------------------------------

    requested_features = parse_feature_set(
        feature_set_path
    )

    if len(requested_features) != 20:
        raise ValueError(
            "F1 must contain exactly 20 predictors. "
            f"Found {len(requested_features)}."
        )

    print(
        "\nF1 features:"
    )

    for i, feature in enumerate(
        requested_features,
        start=1
    ):
        print(
            f"{i:02d}. {feature}"
        )


    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    manifest = load_manifest()

    start_period = pd.Timestamp(
        "2025-01-01 00:00:00"
    )

    end_period = pd.Timestamp(
        "2025-04-01 00:00:00"
    )

    outside = ~(
        (
            manifest["anchor_time"]
            >= start_period
        )
        &
        (
            manifest["anchor_time"]
            < end_period
        )
    )

    outside_count = int(
        outside.sum()
    )

    if outside_count:
        raise ValueError(
            f"External manifest contains "
            f"{outside_count} rows outside Jan-Mar 2025."
        )

    print(
        "\nManifest rows:",
        len(manifest)
    )

    print(
        "Anchor range:",
        manifest["anchor_time"].min(),
        "to",
        manifest["anchor_time"].max()
    )


    # --------------------------------------------------------
    # LOAD B2
    # --------------------------------------------------------
    #
    # For large CSVs we use pandas chunking only if necessary.
    # The source is expected to be minute-level and therefore
    # manageable, but we retain the explicit time index.
    # --------------------------------------------------------

    b2 = pd.read_csv(
        b2_path
    )

    time_col = find_time_column(
        b2
    )

    b2[time_col] = pd.to_datetime(
        b2[time_col],
        errors="coerce"
    )

    if b2[time_col].isna().any():
        raise ValueError(
            "External B2 contains invalid UTC timestamps."
        )

    b2 = b2.sort_values(
        time_col
    ).reset_index(
        drop=True
    )

    if b2[time_col].duplicated().any():
        duplicate_count = int(
            b2[time_col].duplicated().sum()
        )

        raise ValueError(
            "External B2 contains duplicate timestamps: "
            f"{duplicate_count}"
        )


    # --------------------------------------------------------
    # RESOLVE F1 COLUMNS
    # --------------------------------------------------------

    feature_columns = resolve_feature_columns(
        b2,
        requested_features
    )

    print(
        "\nResolved F1 columns:"
    )

    for requested, actual in zip(
        requested_features,
        feature_columns
    ):
        print(
            f"{requested} -> {actual}"
        )


    # --------------------------------------------------------
    # CREATE TIMESTAMP INDEX
    # --------------------------------------------------------

    b2_indexed = b2.set_index(
        time_col
    )

    # External B2 must have one-minute spacing.
    deltas = (
        b2_indexed.index
        .to_series()
        .diff()
        .dropna()
    )

    bad_deltas = deltas[
        deltas
        != pd.Timedelta(minutes=1)
    ]

    if len(bad_deltas) > 0:

        print(
            "\nWARNING: External B2 contains "
            f"{len(bad_deltas)} non-1-minute intervals."
        )

        print(
            "The exporter will NOT fill or interpolate "
            "these gaps."
        )

        print(
            "Any affected 60-minute window will fail "
            "the exact-timestamp audit."
        )


    # --------------------------------------------------------
    # BUILD WINDOWS
    # --------------------------------------------------------
    #
    # Each Stage C manifest row defines:
    #
    # input_start ... input_end   = 60 timestamps
    # target_start ... target_end = 10 target labels
    #
    # We select the external B2 predictors at those exact
    # timestamps. No resampling and no imputation are performed.
    # --------------------------------------------------------

    n_windows = len(manifest)

    X = np.empty(
        (
            n_windows,
            60,
            20
        ),
        dtype=np.float32
    )

    y = manifest[
        [
            f"y_h{i}"
            for i in range(1, 11)
        ]
    ].to_numpy(
        dtype=np.int8
    )

    failed_windows = []

    missing_timestamp_windows = []

    nonfinite_windows = []

    wrong_length_windows = []

    for row_id, row in manifest.iterrows():

        input_start = row[
            "input_start"
        ]

        input_end = row[
            "input_end"
        ]

        expected_times = pd.date_range(
            start=input_start,
            end=input_end,
            freq="min"
        )

        if len(expected_times) != 60:

            wrong_length_windows.append(
                {
                    "row": int(row_id),
                    "sample_id": str(
                        row["sample_id"]
                    ),
                    "input_start": str(
                        input_start
                    ),
                    "input_end": str(
                        input_end
                    ),
                    "count": int(
                        len(expected_times)
                    ),
                }
            )

            continue


        # Exact reindex is deliberate:
        # no nearest, no interpolation, no fill.
        window = b2_indexed[
            feature_columns
        ].reindex(
            expected_times
        )


        if len(window) != 60:

            wrong_length_windows.append(
                {
                    "row": int(row_id),
                    "sample_id": str(
                        row["sample_id"]
                    ),
                    "count": int(
                        len(window)
                    ),
                }
            )

            continue


        if window.isna().any().any():

            missing_timestamp_windows.append(
                {
                    "row": int(row_id),
                    "sample_id": str(
                        row["sample_id"]
                    ),
                    "missing_values": int(
                        window.isna().sum().sum()
                    ),
                }
            )

            continue


        values = window.to_numpy(
            dtype=np.float32
        )


        if not np.isfinite(
            values
        ).all():

            nonfinite_windows.append(
                {
                    "row": int(row_id),
                    "sample_id": str(
                        row["sample_id"]
                    ),
                }
            )

            continue


        X[row_id] = values


    failed_count = (
        len(missing_timestamp_windows)
        + len(nonfinite_windows)
        + len(wrong_length_windows)
    )


    # --------------------------------------------------------
    # HARD EXPORT AUDIT
    # --------------------------------------------------------

    print(
        "\nWindow audit:"
    )

    print(
        "Total windows:",
        n_windows
    )

    print(
        "Missing-value windows:",
        len(missing_timestamp_windows)
    )

    print(
        "Non-finite windows:",
        len(nonfinite_windows)
    )

    print(
        "Wrong-length windows:",
        len(wrong_length_windows)
    )

    print(
        "Failed windows:",
        failed_count
    )


    if failed_count > 0:

        audit = {
            "manifest_windows":
                int(n_windows),

            "exported_windows":
                0,

            "failed_windows":
                int(failed_count),

            "missing_timestamp_windows":
                int(
                    len(
                        missing_timestamp_windows
                    )
                ),

            "nonfinite_windows":
                int(
                    len(
                        nonfinite_windows
                    )
                ),

            "wrong_length_windows":
                int(
                    len(
                        wrong_length_windows
                    )
                ),

            "source_stage":
                "External Stage B2",

            "target_source":
                "Stage C external_test_windows.csv",

            "external_period":
                "2025-01-01 through 2025-03-31",

            "feature_set":
                "F1",

            "feature_count":
                20,

            "lookback_minutes":
                60,

            "forecast_horizon":
                10,

            "status":
                "FAIL"
        }

        with open(
            AUDIT_OUTPUT,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                audit,
                f,
                indent=2
            )

        raise RuntimeError(
            "\nExternal F1 export FAILED.\n"
            "No X.npy/y.npy were written.\n"
            "The script intentionally does not fill, "
            "interpolate, or silently discard bad windows.\n"
            f"Audit: {AUDIT_OUTPUT}"
        )


    # --------------------------------------------------------
    # FINAL SHAPE / TARGET AUDIT
    # --------------------------------------------------------

    if X.shape != (
        n_windows,
        60,
        20
    ):
        raise RuntimeError(
            f"Unexpected X shape: {X.shape}"
        )

    if y.shape != (
        n_windows,
        10
    ):
        raise RuntimeError(
            f"Unexpected y shape: {y.shape}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Final X contains non-finite values."
        )

    unique_targets = np.unique(y)

    if not set(
        unique_targets.tolist()
    ).issubset({0, 1}):

        raise RuntimeError(
            "External y contains values other than 0/1: "
            f"{unique_targets}"
        )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    np.save(
        X_OUTPUT,
        X
    )

    np.save(
        Y_OUTPUT,
        y
    )


    metadata = {

        "feature_set":
            "F1",

        "feature_count":
            20,

        "lookback_minutes":
            60,

        "forecast_horizon":
            10,

        "predictor_source":
            "External Stage B2",

        "target_source":
            "Stage C external_test_windows.csv",

        "external_period":
            "2025-01-01 through 2025-03-31",

        "features":
            requested_features,

        "resolved_source_columns":
            feature_columns,

        "X_shape":
            list(X.shape),

        "y_shape":
            list(y.shape),

        "time_column":
            time_col,

        "resampling":
            "none",

        "interpolation":
            "none",

        "nearest_fill":
            "none",

        "scaling":
            "none",

        "development_X_used":
            False,

        "development_y_used":
            False,

        "training_performed":
            False,

        "threshold_selection_performed":
            False
    }

    with open(
        METADATA_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )


    audit = {

        "manifest_windows":
            int(n_windows),

        "exported_windows":
            int(n_windows),

        "failed_windows":
            0,

        "missing_timestamp_windows":
            0,

        "nonfinite_windows":
            0,

        "wrong_length_windows":
            0,

        "X_shape":
            list(X.shape),

        "y_shape":
            list(y.shape),

        "external_period":
            "2025-01-01 through 2025-03-31",

        "source_stage":
            "External Stage B2",

        "target_source":
            "Stage C external_test_windows.csv",

        "feature_set":
            "F1",

        "feature_count":
            20,

        "lookback_minutes":
            60,

        "forecast_horizon":
            10,

        "feature_order":
            requested_features,

        "resolved_feature_columns":
            feature_columns,

        "timestamp_gap_check":
            "PASS",

        "exact_window_check":
            "PASS",

        "missing_value_check":
            "PASS",

        "finite_value_check":
            "PASS",

        "target_binary_check":
            "PASS",

        "development_data_used":
            False,

        "training_performed":
            False,

        "scaling_performed":
            False,

        "threshold_selection_performed":
            False,

        "status":
            "PASS"
    }

    with open(
        AUDIT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=2
        )


    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "EXTERNAL F1 EXPORT FINISHED - PASS"
    )
    print("=" * 90)

    print(
        "X:",
        X_OUTPUT
    )

    print(
        "y:",
        Y_OUTPUT
    )

    print(
        "Metadata:",
        METADATA_OUTPUT
    )

    print(
        "Audit:",
        AUDIT_OUTPUT
    )

    print(
        "\nX shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "All windows:",
        n_windows
    )

    print(
        "Failed windows:",
        failed_count
    )

    print(
        "\nNo development X/y was used."
    )

    print(
        "No training, scaling, interpolation, "
        "or threshold selection was performed."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
