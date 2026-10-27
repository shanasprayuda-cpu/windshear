import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STAGE D1 - EXPORT DEVELOPMENT F2 AND F3
#
# F1 = 20 u/v
# F2 = F1 + 17 spatial shear/gradient features = 37
# F3 = F2 + 8 divergence features = 45
#
# Uses:
#   - Development Stage B2 minute features
#   - Stage C development_windows.csv
#   - Stage C target labels
#
# Does NOT use:
#   - External Test
#   - development F1 X.npy as predictor source
#   - interpolation / nearest fill
#   - scaling
#   - training
#
# Output:
#   ..\stageD1_data\F2\
#   ..\stageD1_data\F3\
# ============================================================


ENGINE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ENGINE_ROOT.parent

if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

B2_DIR = PROJECT_ROOT / "stageB2_physics"
B2_CSV = B2_DIR / "stageB2_physics_minute_features.csv"
FEATURE_SETS = B2_DIR / "feature_sets.json"

STAGEC_DIR = PROJECT_ROOT / "stageC"
MANIFEST = STAGEC_DIR / "manifest" / "development_windows.csv"

STAGED1_DIR = PROJECT_ROOT / "stageD1_data"
OUTPUTS = {
    "F2": STAGED1_DIR / "F2",
    "F3": STAGED1_DIR / "F3",
}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def die(msg):
    raise RuntimeError(msg)


def normalize_name(x):
    return "".join(
        ch.lower()
        for ch in str(x)
        if ch.isalnum()
    )


def load_feature_sets():
    if not FEATURE_SETS.exists():
        die(f"feature_sets.json not found:\n{FEATURE_SETS}")

    with open(FEATURE_SETS, "r", encoding="utf-8") as f:
        obj = json.load(f)

    result = {}

    for name in ("F1", "F2", "F3"):
        if name not in obj:
            die(f"{name} missing from {FEATURE_SETS}")

        value = obj[name]

        if not isinstance(value, dict):
            die(f"{name} definition is not an object.")

        columns = value.get("columns")

        if not isinstance(columns, list):
            die(f"{name}.columns is missing or invalid.")

        result[name] = columns

    if len(result["F1"]) != 20:
        die(
            f"F1 must contain 20 columns; "
            f"found {len(result['F1'])}"
        )

    if len(result["F2"]) != 37:
        die(
            f"F2 must contain 37 columns; "
            f"found {len(result['F2'])}"
        )

    if len(result["F3"]) != 45:
        die(
            f"F3 must contain 45 columns; "
            f"found {len(result['F3'])}"
        )

    if result["F2"][:20] != result["F1"]:
        die(
            "F2 is not an exact extension of F1. "
            "Feature order is inconsistent."
        )

    if result["F3"][:37] != result["F2"]:
        die(
            "F3 is not an exact extension of F2. "
            "Feature order is inconsistent."
        )

    return result


def resolve_columns(df, requested):
    normalized = {}

    for c in df.columns:
        key = normalize_name(c)

        if key in normalized:
            die(
                "Ambiguous normalized column name: "
                f"{c}"
            )

        normalized[key] = c

    resolved = []

    missing = []

    for feature in requested:
        key = normalize_name(feature)

        if key not in normalized:
            missing.append(feature)
        else:
            resolved.append(normalized[key])

    if missing:
        die(
            "Missing requested feature columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
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
        "valid_time",
        "valid_datetime",
    ]

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    matches = [
        lower[c.lower()]
        for c in candidates
        if c.lower() in lower
    ]

    if len(matches) != 1:
        die(
            "Could not uniquely identify UTC time column.\n"
            f"Candidates found: {matches}\n"
            f"Columns: {df.columns.tolist()}"
        )

    return matches[0]


def load_manifest():
    if not MANIFEST.exists():
        die(
            f"Development manifest not found:\n{MANIFEST}"
        )

    df = pd.read_csv(MANIFEST)

    required = [
        "sample_id",
        "dataset_role",
        "anchor_idx",
        "anchor_time",
        "input_start",
        "input_end",
        "target_start",
        "target_end",
    ] + [
        f"y_h{i}"
        for i in range(1, 11)
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        die(
            "Development manifest missing columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )

    for c in [
        "anchor_time",
        "input_start",
        "input_end",
        "target_start",
        "target_end",
    ]:
        df[c] = pd.to_datetime(
            df[c],
            errors="coerce"
        )

    if df[
        [
            "anchor_time",
            "input_start",
            "input_end",
            "target_start",
            "target_end",
        ]
    ].isna().any().any():
        die(
            "Development manifest contains invalid timestamps."
        )

    if df["sample_id"].duplicated().any():
        die(
            "Development manifest contains duplicate sample_id."
        )

    if df["anchor_idx"].duplicated().any():
        die(
            "Development manifest contains duplicate anchor_idx."
        )

    # The manifest is the authoritative row order.
    # Do not assume anchor_idx == row position.
    return df.reset_index(drop=True)


def audit_target_columns(manifest):
    target_cols = [
        f"y_h{i}"
        for i in range(1, 11)
    ]

    y = manifest[target_cols].to_numpy(
        dtype=np.int8
    )

    if not np.isin(y, [0, 1]).all():
        values = np.unique(y)
        die(
            f"Target contains values other than 0/1: {values}"
        )

    return y


def exact_window_from_b2(
    b2_index,
    columns,
    start,
    end,
):
    expected = pd.date_range(
        start=start,
        end=end,
        freq="min",
    )

    if len(expected) != 60:
        return None, "wrong_window_length"

    window = b2_index[
        columns
    ].reindex(expected)

    if len(window) != 60:
        return None, "wrong_reindexed_length"

    if window.isna().any().any():
        return None, "missing_values"

    values = window.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(values).all():
        return None, "nonfinite_values"

    return values, None


def export_feature_set(
    name,
    manifest,
    b2_index,
    resolved_columns,
    y,
    requested_columns,
):
    output_dir = OUTPUTS[name]
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    n = len(manifest)
    p = len(requested_columns)

    X = np.empty(
        (n, 60, p),
        dtype=np.float32
    )

    failures = []

    for i, row in manifest.iterrows():

        values, reason = exact_window_from_b2(
            b2_index,
            resolved_columns,
            row["input_start"],
            row["input_end"],
        )

        if reason is not None:
            failures.append({
                "row": int(i),
                "sample_id": str(
                    row["sample_id"]
                ),
                "anchor_idx": int(
                    row["anchor_idx"]
                ),
                "reason": reason,
            })
            continue

        X[i] = values

    print(
        f"\n{name} window audit:"
    )
    print(
        f"  expected windows : {n}"
    )
    print(
        f"  expected shape   : ({n}, 60, {p})"
    )
    print(
        f"  failed windows   : {len(failures)}"
    )

    if failures:
        failure_path = (
            output_dir / "failed_windows.csv"
        )

        pd.DataFrame(
            failures
        ).to_csv(
            failure_path,
            index=False
        )

        die(
            f"{name} export failed. "
            f"See {failure_path}"
        )

    if X.shape != (n, 60, p):
        die(
            f"{name} shape mismatch: {X.shape}"
        )

    if not np.isfinite(X).all():
        die(
            f"{name} contains non-finite values."
        )

    x_path = output_dir / "X.npy"
    y_path = output_dir / "y.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    metadata = {
        "feature_set": name,
        "feature_count": p,
        "lookback_minutes": 60,
        "forecast_horizon": 10,
        "predictor_source": str(B2_CSV),
        "target_source": str(MANIFEST),
        "window_count": int(n),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "features": requested_columns,
        "resolved_source_columns": resolved_columns,
        "resampling": "none",
        "interpolation": "none",
        "nearest_fill": "none",
        "scaling": "none",
        "training_performed": False,
        "external_test_used": False,
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    audit = {
        "status": "PASS",
        "feature_set": name,
        "manifest_windows": int(n),
        "exported_windows": int(n),
        "failed_windows": 0,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "feature_count": p,
        "lookback_minutes": 60,
        "forecast_horizon": 10,
        "exact_window_check": "PASS",
        "missing_value_check": "PASS",
        "finite_value_check": "PASS",
        "target_binary_check": "PASS",
        "development_manifest_used": True,
        "external_test_used": False,
        "training_performed": False,
        "scaling_performed": False,
        "feature_order_source": str(FEATURE_SETS),
        "feature_order": requested_columns,
    }

    with open(
        output_dir / "export_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=2
        )

    return X


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 90)
    print(
        "STAGE D1 - EXPORT DEVELOPMENT F2 AND F3"
    )
    print("=" * 90)

    print(
        "Stage B2:",
        B2_CSV
    )
    print(
        "Feature definition:",
        FEATURE_SETS
    )
    print(
        "Development manifest:",
        MANIFEST
    )
    print(
        "Output root:",
        STAGED1_DIR
    )

    # --------------------------------------------------------
    # LOAD FEATURE DEFINITIONS
    # --------------------------------------------------------

    feature_sets = load_feature_sets()

    print("\nFeature set definitions:")
    for name in ("F1", "F2", "F3"):
        print(
            f"  {name}: "
            f"{len(feature_sets[name])} features"
        )

    # --------------------------------------------------------
    # LOAD DEVELOPMENT MANIFEST
    # --------------------------------------------------------

    manifest = load_manifest()

    print(
        "\nDevelopment windows:",
        len(manifest)
    )

    print(
        "Anchor range:",
        manifest["anchor_time"].min(),
        "to",
        manifest["anchor_time"].max()
    )

    if len(manifest) != 847513:
        print(
            "\nWARNING: Expected historical development "
            "window count is 847513, but manifest contains "
            f"{len(manifest)}."
        )

    y = audit_target_columns(
        manifest
    )

    # --------------------------------------------------------
    # LOAD STAGE B2
    # --------------------------------------------------------

    if not B2_CSV.exists():
        die(
            f"Stage B2 CSV not found:\n{B2_CSV}"
        )

    b2 = pd.read_csv(
        B2_CSV
    )

    print(
        "\nStage B2 rows:",
        len(b2)
    )

    time_col = find_time_column(
        b2
    )

    b2[time_col] = pd.to_datetime(
        b2[time_col],
        errors="coerce"
    )

    if b2[time_col].isna().any():
        die(
            "Stage B2 contains invalid timestamps."
        )

    b2 = b2.sort_values(
        time_col
    ).reset_index(
        drop=True
    )

    if b2[time_col].duplicated().any():
        count = int(
            b2[time_col].duplicated().sum()
        )
        die(
            f"Stage B2 contains {count} duplicate timestamps."
        )

    # --------------------------------------------------------
    # EXACT FEATURE COLUMN RESOLUTION
    # --------------------------------------------------------

    resolved = {}

    for name in ("F2", "F3"):
        resolved[name] = resolve_columns(
            b2,
            feature_sets[name]
        )

        print(
            f"\n{name} source columns resolved: "
            f"{len(resolved[name])}"
        )

    # --------------------------------------------------------
    # TIMESTAMP INDEX
    # --------------------------------------------------------

    b2_index = b2.set_index(
        time_col
    )

    # Check the source is actually minute data.
    deltas = (
        b2_index.index
        .to_series()
        .diff()
        .dropna()
    )

    bad = deltas[
        deltas != pd.Timedelta(minutes=1)
    ]

    print(
        "\nStage B2 non-1-minute intervals:",
        len(bad)
    )

    if len(bad):
        print(
            "No gap filling/interpolation will be performed."
        )

    # --------------------------------------------------------
    # EXPORT F2
    # --------------------------------------------------------

    X2 = export_feature_set(
        "F2",
        manifest,
        b2_index,
        resolved["F2"],
        y,
        feature_sets["F2"],
    )

    # --------------------------------------------------------
    # EXPORT F3
    # --------------------------------------------------------

    X3 = export_feature_set(
        "F3",
        manifest,
        b2_index,
        resolved["F3"],
        y,
        feature_sets["F3"],
    )

    # --------------------------------------------------------
    # CROSS-SET TARGET / WINDOW AUDIT
    # --------------------------------------------------------

    if X2.shape[0] != X3.shape[0]:
        die(
            "F2 and F3 do not contain the same number "
            "of windows."
        )

    if X2.shape[0] != len(manifest):
        die(
            "Exported window count does not match manifest."
        )

    y2 = np.load(
        OUTPUTS["F2"] / "y.npy"
    )

    y3 = np.load(
        OUTPUTS["F3"] / "y.npy"
    )

    if not np.array_equal(y2, y):
        die(
            "F2 y.npy does not exactly match "
            "development manifest targets."
        )

    if not np.array_equal(y3, y):
        die(
            "F3 y.npy does not exactly match "
            "development manifest targets."
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "F2/F3 DEVELOPMENT EXPORT FINISHED - PASS"
    )
    print("=" * 90)

    print(
        f"F2 X shape: {X2.shape}"
    )
    print(
        f"F2 y shape: {y.shape}"
    )
    print(
        f"F3 X shape: {X3.shape}"
    )
    print(
        f"F3 y shape: {y.shape}"
    )

    print(
        "\nF2:",
        OUTPUTS["F2"]
    )

    print(
        "F3:",
        OUTPUTS["F3"]
    )

    print(
        "\nDevelopment manifest:",
        len(manifest),
        "windows"
    )

    print(
        "External test used: NO"
    )

    print(
        "Training performed: NO"
    )

    print(
        "Scaling performed: NO"
    )

    print(
        "Interpolation performed: NO"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
