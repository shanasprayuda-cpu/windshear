import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.tcn_bilstm import TCNBiLSTM


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Stage D1 final external test for "
        "TCN-BiLSTM on Jan-Mar 2025."
    )
)

parser.add_argument(
    "--external_data_path",
    default=None,
    help=(
        "Directory containing external X.npy. "
        "If omitted, the script searches standard Stage D1 paths."
    )
)

parser.add_argument(
    "--external_manifest",
    default=None,
    help=(
        "Path to external_test_windows.csv. "
        "Default: ../stageC/manifest/external_test_windows.csv"
    )
)

parser.add_argument(
    "--checkpoint",
    default=None,
    help=(
        "Final full-development checkpoint. "
        "Default: results/final_full/FINAL_TCN_BiLSTM_TRIAL26_FULL/final_model.pt "
        "if present, otherwise search results/final_full."
    )
)

parser.add_argument(
    "--thresholds",
    default=None,
    help=(
        "Locked threshold JSON. "
        "Default: results/final_thresholds/thresholds.json"
    )
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=256
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=2
)

parser.add_argument(
    "--run_name",
    default="FINAL_TCN_BiLSTM_EXTERNAL_JAN_MAR_2025"
)

parser.add_argument(
    "--allow_outside_date_range",
    action="store_true",
    help=(
        "Allow external manifest rows outside the Jan-Mar 2025 "
        "anchor-time range. Use only after manual audit."
    )
)

args = parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "config.json"
)

with open(
    CONFIG_PATH,
    "r"
) as f:
    CFG = json.load(f)


# Locked final model hyperparameters.
# These correspond to Trial 26 / final full retraining.
FINAL_CONFIG_CANDIDATES = [
    PROJECT_ROOT
    / "results"
    / "final_full"
    / "FINAL_TCN_BiLSTM_TRIAL26_FULL"
    / "final_config.json",

    PROJECT_ROOT
    / "results"
    / "final_full"
    / "final_config.json",
]


def find_existing(candidates):

    for p in candidates:
        if p.exists():
            return p

    return None


final_config_path = find_existing(
    FINAL_CONFIG_CANDIDATES
)

if final_config_path is not None:

    with open(
        final_config_path,
        "r"
    ) as f:
        FINAL_CFG = json.load(f)

else:

    FINAL_CFG = {}


# ============================================================
# PATHS
# ============================================================

if args.external_manifest is None:

    EXTERNAL_MANIFEST = (
        PROJECT_ROOT
        / CFG["stageC_path"]
        / "manifest"
        / "external_test_windows.csv"
    )

else:

    EXTERNAL_MANIFEST = Path(
        args.external_manifest
    )

    if not EXTERNAL_MANIFEST.is_absolute():
        EXTERNAL_MANIFEST = (
            PROJECT_ROOT / EXTERNAL_MANIFEST
        )


if args.thresholds is None:

    THRESHOLD_PATH = (
        PROJECT_ROOT
        / "results"
        / "final_thresholds"
        / "thresholds.json"
    )

else:

    THRESHOLD_PATH = Path(
        args.thresholds
    )

    if not THRESHOLD_PATH.is_absolute():
        THRESHOLD_PATH = (
            PROJECT_ROOT / THRESHOLD_PATH
        )


# ============================================================
# EXTERNAL DATA PATH DISCOVERY
# ============================================================

def discover_external_data():

    if args.external_data_path is not None:

        p = Path(
            args.external_data_path
        )

        if not p.is_absolute():
            p = PROJECT_ROOT / p

        if not (
            p / "X.npy"
        ).exists():

            raise FileNotFoundError(
                f"X.npy not found in:\n{p}"
            )

        return p


    candidates = [

        PROJECT_ROOT
        / "stageD1_data"
        / "F1_external",

        PROJECT_ROOT
        / "stageD1_data"
        / "F1"
        / "external",

        PROJECT_ROOT
        / "stageD1_data"
        / "external"
        / "F1",

        PROJECT_ROOT
        / "stageD1_external"
        / "F1",

        PROJECT_ROOT
        / "external_test_data"
        / "F1",
    ]

    found = [
        p for p in candidates
        if (p / "X.npy").exists()
    ]

    if len(found) == 1:
        return found[0]

    if len(found) > 1:

        raise RuntimeError(
            "More than one external X.npy candidate was found.\n"
            "Specify --external_data_path explicitly:\n\n"
            + "\n".join(
                str(p) for p in found
            )
        )

    raise FileNotFoundError(
        "\nCould not automatically locate external X.npy.\n\n"
        "Please specify the actual external Stage D1 data directory, "
        "for example:\n\n"
        "python scripts/external_test_tcn_bilstm.py "
        "--external_data_path ..\\stageD1_data\\F1_EXTERNAL\n\n"
        "The directory must contain X.npy and should be aligned "
        "row-for-row with external_test_windows.csv."
    )


EXTERNAL_DATA_PATH = discover_external_data()


# ============================================================
# FINAL CHECKPOINT DISCOVERY
# ============================================================

def discover_checkpoint():

    if args.checkpoint is not None:

        p = Path(
            args.checkpoint
        )

        if not p.is_absolute():
            p = PROJECT_ROOT / p

        return p


    candidates = [

        PROJECT_ROOT
        / "results"
        / "final_full"
        / "FINAL_TCN_BiLSTM_TRIAL26_FULL"
        / "final_model.pt",

        PROJECT_ROOT
        / "results"
        / "final_full"
        / "final_model.pt",
    ]

    # Also search final_full recursively, but do not silently
    # choose an arbitrary checkpoint if multiple exist.
    if not any(
        p.exists()
        for p in candidates
    ):

        recursive = list(
            (
                PROJECT_ROOT
                / "results"
                / "final_full"
            ).glob(
                "**/final_model.pt"
            )
        )

        if len(recursive) == 1:
            return recursive[0]

        if len(recursive) > 1:

            raise RuntimeError(
                "Multiple final_model.pt files found. "
                "Specify --checkpoint explicitly:\n\n"
                + "\n".join(
                    str(p)
                    for p in recursive
                )
            )

    p = find_existing(
        candidates
    )

    if p is None:

        raise FileNotFoundError(
            "Final full-development checkpoint was not found.\n"
            "Specify it with --checkpoint."
        )

    return p


CHECKPOINT_PATH = discover_checkpoint()


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "external_test"
    / args.run_name
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "external_predictions.csv"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "external_metrics.csv"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "external_test_audit.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "external_summary.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = int(
    FINAL_CFG.get(
        "seed",
        2026
    )
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL CONFIG
# ============================================================

HIDDEN_SIZE = int(
    FINAL_CFG.get(
        "hidden_size",
        32
    )
)

NUM_LAYERS = int(
    FINAL_CFG.get(
        "num_layers",
        1
    )
)

DROPOUT = float(
    FINAL_CFG.get(
        "dropout",
        0.25
    )
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class ExternalDataset(Dataset):

    def __init__(
        self,
        X
    ):

        self.X = X

    def __len__(self):

        return len(self.X)

    def __getitem__(
        self,
        idx
    ):

        x = np.asarray(
            self.X[idx],
            dtype=np.float32
        )

        return torch.from_numpy(x)


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_model_checkpoint(
    model,
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    elif all(
        isinstance(
            v,
            torch.Tensor
        )
        for v in checkpoint.values()
    ):

        state_dict = checkpoint

    else:

        raise RuntimeError(
            "Could not find model state_dict in checkpoint."
        )

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[7:]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True
    )

    model.eval()

    return checkpoint


# ============================================================
# METRICS
# ============================================================

def binary_metrics(
    y_true,
    y_prob,
    threshold
):

    y_true = np.asarray(
        y_true
    ).astype(
        np.int8
    )

    y_prob = np.asarray(
        y_prob
    ).astype(
        np.float64
    )

    y_pred = (
        y_prob >= threshold
    ).astype(
        np.int8
    )

    tp = int(
        np.sum(
            (y_pred == 1)
            & (y_true == 1)
        )
    )

    fp = int(
        np.sum(
            (y_pred == 1)
            & (y_true == 0)
        )
    )

    fn = int(
        np.sum(
            (y_pred == 0)
            & (y_true == 1)
        )
    )

    tn = int(
        np.sum(
            (y_pred == 0)
            & (y_true == 0)
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    pod = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    far = (
        fp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    csi = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * pod
        / (precision + pod)
        if precision + pod > 0
        else 0.0
    )

    f2 = (
        5.0
        * precision
        * pod
        / (
            4.0 * precision
            + pod
        )
        if (
            4.0 * precision
            + pod
        ) > 0
        else 0.0
    )

    # AUPRC is meaningful only if both classes are represented.
    if (
        np.unique(y_true).size == 2
    ):

        auprc = float(
            average_precision_score(
                y_true,
                y_prob
            )
        )

    else:

        auprc = float("nan")


    # ROC AUC is provided as an auxiliary diagnostic only.
    if (
        np.unique(y_true).size == 2
    ):

        roc_auc = float(
            roc_auc_score(
                y_true,
                y_prob
            )
        )

    else:

        roc_auc = float("nan")


    return {
        "threshold": float(threshold),

        "AUPRC": auprc,
        "ROC_AUC": roc_auc,

        "CSI": float(csi),
        "POD": float(pod),
        "Precision": float(precision),
        "FAR": float(far),

        "F1": float(f1),
        "F2": float(f2),

        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,

        "N": int(len(y_true)),
        "events": int(
            np.sum(y_true == 1)
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "STAGE D1 - FINAL EXTERNAL TEST"
    )
    print(
        "TCN-BiLSTM / Jan-Mar 2025"
    )
    print("=" * 90)

    print(
        "Device          :",
        DEVICE
    )

    print(
        "Checkpoint      :",
        CHECKPOINT_PATH
    )

    print(
        "External data   :",
        EXTERNAL_DATA_PATH
    )

    print(
        "External manifest:",
        EXTERNAL_MANIFEST
    )

    print(
        "Locked thresholds:",
        THRESHOLD_PATH
    )

    print(
        "Output          :",
        OUTPUT_DIR
    )

    print("=" * 90)


    # ========================================================
    # SAFETY CHECK: EXTERNAL MANIFEST
    # ========================================================

    if not EXTERNAL_MANIFEST.exists():

        raise FileNotFoundError(
            f"External manifest not found:\n"
            f"{EXTERNAL_MANIFEST}"
        )


    manifest = pd.read_csv(
        EXTERNAL_MANIFEST
    )

    if "anchor_time" not in manifest.columns:

        raise RuntimeError(
            "external_test_windows.csv must contain "
            "'anchor_time'."
        )


    # Convert to datetime for audit/filtering.
    manifest[
        "anchor_time_parsed"
    ] = pd.to_datetime(
        manifest["anchor_time"],
        errors="coerce"
    )

    if manifest[
        "anchor_time_parsed"
    ].isna().any():

        raise RuntimeError(
            "Some external anchor_time values could not "
            "be parsed."
        )


    print(
        "External manifest rows:",
        len(manifest)
    )

    print(
        "External anchor range:",
        manifest[
            "anchor_time_parsed"
        ].min(),
        "to",
        manifest[
            "anchor_time_parsed"
        ].max()
    )


    # --------------------------------------------------------
    # Jan 1 00:00 through Apr 1 00:00 exclusive.
    # --------------------------------------------------------

    start = pd.Timestamp(
        "2025-01-01 00:00:00"
    )

    end = pd.Timestamp(
        "2025-04-01 00:00:00"
    )

    in_period = (
        (
            manifest[
                "anchor_time_parsed"
            ] >= start
        )
        &
        (
            manifest[
                "anchor_time_parsed"
            ] < end
        )
    )

    outside_count = int(
        (~in_period).sum()
    )

    if (
        outside_count > 0
        and not args.allow_outside_date_range
    ):

        raise RuntimeError(
            "\nExternal manifest contains "
            f"{outside_count} rows outside "
            "Jan-Mar 2025 based on anchor_time.\n\n"
            "This script refuses to silently filter the "
            "external test set.\n"
            "Audit the manifest first, or explicitly use "
            "--allow_outside_date_range if the Stage C "
            "definition intentionally includes those rows."
        )


    if (
        args.allow_outside_date_range
        and outside_count > 0
    ):

        manifest = manifest[
            in_period
        ].copy()

        print(
            "Rows retained after explicit Jan-Mar filter:",
            len(manifest)
        )

    elif outside_count == 0:

        print(
            "Date audit: PASS"
        )

    else:

        # Defensive branch.
        manifest = manifest[
            in_period
        ].copy()


    if len(manifest) == 0:

        raise RuntimeError(
            "No external-test rows remain."
        )


    # ========================================================
    # LOAD EXTERNAL X
    # ========================================================

    x_path = (
        EXTERNAL_DATA_PATH
        / "X.npy"
    )

    X = np.load(
        x_path,
        mmap_mode="r"
    )

    print(
        "External X shape:",
        X.shape
    )

    if X.ndim != 3:

        raise RuntimeError(
            f"Expected X to be 3-D, got {X.shape}"
        )

    if X.shape[1:] != (
        60,
        20
    ):

        raise RuntimeError(
            "External X must have shape "
            "(N, 60, 20). "
            f"Got {X.shape}"
        )

    if len(X) != len(manifest):

        raise RuntimeError(
            "\nExternal X row count does not match "
            "the filtered external manifest.\n"
            f"X rows       : {len(X)}\n"
            f"Manifest rows: {len(manifest)}\n\n"
            "The script requires row-for-row alignment."
        )


    # ========================================================
    # LOAD EXTERNAL TARGETS
    # ========================================================
    #
    # Preferred source:
    #   y.npy aligned row-for-row with X.npy.
    #
    # Fallback:
    #   y_h1 ... y_h10 columns in external manifest.
    # ========================================================

    y_path = (
        EXTERNAL_DATA_PATH
        / "y.npy"
    )

    if y_path.exists():

        y = np.load(
            y_path,
            mmap_mode="r"
        )

        if len(y) != len(manifest):

            raise RuntimeError(
                "External y.npy row count does not "
                "match external X/manifest."
            )

        if y.ndim != 2 or y.shape[1] != 10:

            raise RuntimeError(
                "External y.npy must have shape "
                "(N, 10). "
                f"Got {y.shape}"
            )

        y_external = np.asarray(
            y,
            dtype=np.int8
        )

        target_source = (
            "external_data_path/y.npy"
        )

    else:

        target_columns = [
            f"y_h{h}"
            for h in range(1, 11)
        ]

        missing = [
            c for c in target_columns
            if c not in manifest.columns
        ]

        if missing:

            raise FileNotFoundError(
                "External y.npy is absent and the "
                "manifest does not contain all target "
                f"columns. Missing: {missing}"
            )

        y_external = manifest[
            target_columns
        ].to_numpy(
            dtype=np.int8
        )

        target_source = (
            "external_test_windows.csv target columns"
        )


    # ========================================================
    # LOAD LOCKED THRESHOLDS
    # ========================================================

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            f"Locked threshold file not found:\n"
            f"{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r"
    ) as f:

        threshold_config = json.load(f)


    if threshold_config.get(
        "external_test_used",
        True
    ):

        raise RuntimeError(
            "Threshold artifact is marked as having "
            "used external test data. "
            "Refusing to proceed."
        )


    if threshold_config.get(
        "method"
    ) != "pooled_OOF_validation":

        raise RuntimeError(
            "Threshold artifact is not marked as "
            "pooled OOF validation."
        )


    horizons_cfg = threshold_config.get(
        "horizons",
        {}
    )

    thresholds = []

    for h in range(
        1,
        11
    ):

        key = f"H{h}"

        if key not in horizons_cfg:

            raise RuntimeError(
                f"Missing locked threshold for {key}."
            )

        threshold = float(
            horizons_cfg[key][
                "threshold"
            ]
        )

        if not (
            0.0 < threshold < 1.0
        ):

            raise RuntimeError(
                f"Invalid threshold for {key}: "
                f"{threshold}"
            )

        thresholds.append(
            threshold
        )


    print(
        "\nLocked thresholds:"
    )

    for h, threshold in enumerate(
        thresholds,
        start=1
    ):

        print(
            f"H{h:02d}: {threshold:.3f}"
        )


    # ========================================================
    # LOAD FINAL FULL-DEVELOPMENT MODEL
    # ========================================================

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Final checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    model = TCNBiLSTM(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)


    checkpoint = load_model_checkpoint(
        model,
        CHECKPOINT_PATH
    )


    print(
        "\nCheckpoint loaded."
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "epoch" in checkpoint:
            print(
                "Checkpoint epoch:",
                checkpoint["epoch"]
            )

        if "architecture" in checkpoint:
            print(
                "Checkpoint architecture:",
                checkpoint["architecture"]
            )


    # ========================================================
    # EXTERNAL PREDICTION
    # ========================================================

    dataset = ExternalDataset(
        X
    )

    loader_args = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True
    }

    if args.num_workers > 0:

        loader_args[
            "persistent_workers"
        ] = True

        loader_args[
            "prefetch_factor"
        ] = 4


    data_loader = DataLoader(
        **loader_args
    )


    probabilities = np.empty(
        (
            len(X),
            10
        ),
        dtype=np.float32
    )

    cursor = 0

    use_amp = (
        DEVICE.type == "cuda"
    )

    model.eval()

    with torch.no_grad():

        for xb in data_loader:

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp,
                dtype=torch.float16
            ):

                logits = model(xb)

            prob = (
                torch.sigmoid(
                    logits
                )
                .float()
                .cpu()
                .numpy()
            )

            n = len(prob)

            probabilities[
                cursor:cursor+n
            ] = prob

            cursor += n


    if cursor != len(X):

        raise RuntimeError(
            "Prediction count mismatch."
        )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    metric_rows = []

    for h in range(
        1,
        11
    ):

        result = binary_metrics(
            y_external[:, h-1],
            probabilities[:, h-1],
            thresholds[h-1]
        )

        result[
            "horizon"
        ] = f"H{h}"

        metric_rows.append(
            result
        )


    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df = metrics_df[
        [
            "horizon",
            "threshold",
            "AUPRC",
            "ROC_AUC",
            "CSI",
            "POD",
            "Precision",
            "FAR",
            "F1",
            "F2",
            "TP",
            "FP",
            "FN",
            "TN",
            "N",
            "events"
        ]
    ]


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_output = manifest.copy()

    if "anchor_time_parsed" in prediction_output.columns:

        prediction_output[
            "anchor_time_parsed"
        ] = prediction_output[
            "anchor_time_parsed"
        ].astype(str)


    for h in range(
        1,
        11
    ):

        prediction_output[
            f"y_h{h}"
        ] = y_external[
            :,
            h-1
        ]

        prediction_output[
            f"p_h{h}"
        ] = probabilities[
            :,
            h-1
        ]

        prediction_output[
            f"pred_h{h}"
        ] = (
            probabilities[
                :,
                h-1
            ]
            >= thresholds[h-1]
        ).astype(
            np.int8
        )


    prediction_output.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "stage":
            "D1 External Test",

        "architecture":
            "TCN-BiLSTM",

        "optuna_trial":
            26,

        "final_retraining":
            "full development",

        "external_period":
            "2025-01-01 through 2025-03-31",

        "external_samples":
            int(len(X)),

        "external_events_by_horizon":
            {
                f"H{h}":
                    int(
                        np.sum(
                            y_external[:, h-1] == 1
                        )
                    )
                for h in range(1, 11)
            },

        "threshold_source":
            str(THRESHOLD_PATH),

        "threshold_method":
            threshold_config.get(
                "method"
            ),

        "external_test_used_for_threshold":
            False,

        "target_source":
            target_source,

        "checkpoint":
            str(CHECKPOINT_PATH),

        "metrics_file":
            str(METRICS_PATH),

        "predictions_file":
            str(PREDICTIONS_PATH)
    }

    with open(
        SUMMARY_PATH,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


    # ========================================================
    # AUDIT
    # ========================================================

    audit = {

        "stage":
            "D1 final external test",

        "architecture":
            "TCN-BiLSTM",

        "optuna_trial":
            26,

        "seed":
            SEED,

        "device":
            str(DEVICE),

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_source":
            "final full-development retraining",

        "external_data_path":
            str(EXTERNAL_DATA_PATH),

        "external_manifest":
            str(EXTERNAL_MANIFEST),

        "external_samples":
            int(len(X)),

        "external_date_start":
            str(
                manifest[
                    "anchor_time_parsed"
                ].min()
            ),

        "external_date_end":
            str(
                manifest[
                    "anchor_time_parsed"
                ].max()
            ),

        "external_period_filter":
            "2025-01-01 <= anchor_time < 2025-04-01",

        "outside_date_rows_before_filter":
            outside_count,

        "target_source":
            target_source,

        "threshold_file":
            str(THRESHOLD_PATH),

        "threshold_external_test_used":
            threshold_config.get(
                "external_test_used"
            ),

        "threshold_method":
            threshold_config.get(
                "method"
            ),

        "threshold_objective":
            threshold_config.get(
                "objective"
            ),

        "thresholds_locked":
            thresholds,

        "external_data_used_for_training":
            False,

        "external_data_used_for_threshold_selection":
            False,

        "external_data_used_only_for_final_evaluation":
            True,

        "metrics_file":
            str(METRICS_PATH),

        "predictions_file":
            str(PREDICTIONS_PATH)
    }

    with open(
        AUDIT_PATH,
        "w"
    ) as f:

        json.dump(
            audit,
            f,
            indent=2
        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 90)
    print(
        "EXTERNAL TEST RESULTS - JAN-MAR 2025"
    )
    print("=" * 90)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.5f}"
            )
        )
    )

    print("\n" + "=" * 90)
    print(
        "EXTERNAL TEST FINISHED"
    )
    print("=" * 90)

    print(
        "Predictions:",
        PREDICTIONS_PATH
    )

    print(
        "Metrics    :",
        METRICS_PATH
    )

    print(
        "Summary    :",
        SUMMARY_PATH
    )

    print(
        "Audit      :",
        AUDIT_PATH
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
