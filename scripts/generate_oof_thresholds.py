import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Make project root importable when script is executed as:
# python scripts\generate_oof_thresholds.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tcn_bilstm import TCNBiLSTM


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

with open(CONFIG_PATH, "r") as f:
    CFG = json.load(f)

DATA_PATH = PROJECT_ROOT / CFG["data_path"]
STAGEC_PATH = PROJECT_ROOT / CFG["stageC_path"]

ARCHITECTURE = "TCN-BiLSTM"
TRIAL = 26
SEED = 2026

BATCH_SIZE = 256
NUM_WORKERS = int(CFG.get("num_workers", 2))

HORIZONS = 10

# Trial-26 architecture hyperparameters.
# These are used to reconstruct the exact model architecture.
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.25

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_thresholds"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OOF_PREDICTIONS_FILE = (
    OUTPUT_DIR / "oof_predictions.csv"
)

THRESHOLD_FILE = (
    OUTPUT_DIR / "thresholds.json"
)

THRESHOLD_SUMMARY_FILE = (
    OUTPUT_DIR / "threshold_summary.csv"
)

OOF_AUDIT_FILE = (
    OUTPUT_DIR / "oof_audit.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class IndexedMemmapDataset(Dataset):

    def __init__(
        self,
        X,
        indices
    ):
        self.X = X
        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        idx = int(self.indices[i])

        x = np.asarray(
            self.X[idx],
            dtype=np.float32
        )

        return (
            torch.from_numpy(x),
            idx
        )


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def find_checkpoint(fold):

    candidates = [

        PROJECT_ROOT
        / "results"
        / "optuna"
        / ARCHITECTURE
        / f"trial_{TRIAL}"
        / f"fold_{fold}.pt",

        PROJECT_ROOT
        / "results"
        / "optuna"
        / "checkpoints"
        / ARCHITECTURE
        / f"trial_{TRIAL}"
        / f"fold_{fold}.pt",

        PROJECT_ROOT
        / "results"
        / "optuna"
        / ARCHITECTURE
        / f"trial_{TRIAL}"
        / f"fold_{fold}.pth",

        PROJECT_ROOT
        / "results"
        / "optuna"
        / "checkpoints"
        / ARCHITECTURE
        / f"trial_{TRIAL}"
        / f"fold_{fold}.pth",
    ]

    for p in candidates:

        if p.exists():
            return p

    raise FileNotFoundError(
        "\nCould not find checkpoint for fold "
        f"{fold}.\n\n"
        "Checked:\n"
        + "\n".join(
            str(p) for p in candidates
        )
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Case 1: plain state_dict
    if isinstance(
        checkpoint,
        dict
    ):

        if all(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):

            state_dict = checkpoint

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            raise RuntimeError(
                "Checkpoint does not contain a "
                "recognized model state dictionary:\n"
                f"{checkpoint_path}"
            )

    else:

        raise RuntimeError(
            "Unsupported checkpoint format:\n"
            f"{checkpoint_path}"
        )

    # Remove DataParallel prefix if present.
    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

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

def confusion_counts(
    y_true,
    y_prob,
    threshold
):

    pred = (
        y_prob >= threshold
    )

    truth = (
        y_true.astype(bool)
    )

    tp = np.sum(
        pred & truth
    )

    fp = np.sum(
        pred & ~truth
    )

    fn = np.sum(
        ~pred & truth
    )

    tn = np.sum(
        ~pred & ~truth
    )

    return (
        int(tp),
        int(fp),
        int(fn),
        int(tn)
    )


def csi_from_counts(
    tp,
    fp,
    fn
):

    denominator = (
        tp + fp + fn
    )

    if denominator == 0:
        return 0.0

    return tp / denominator


def pod_from_counts(
    tp,
    fn
):

    denominator = (
        tp + fn
    )

    if denominator == 0:
        return 0.0

    return tp / denominator


def far_from_counts(
    tp,
    fp
):

    denominator = (
        tp + fp
    )

    if denominator == 0:
        return 0.0

    return fp / denominator


def precision_from_counts(
    tp,
    fp
):

    denominator = (
        tp + fp
    )

    if denominator == 0:
        return 0.0

    return tp / denominator


def f1_from_counts(
    tp,
    fp,
    fn
):

    precision = precision_from_counts(
        tp,
        fp
    )

    recall = pod_from_counts(
        tp,
        fn
    )

    denominator = (
        precision + recall
    )

    if denominator == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / denominator
    )


def f2_from_counts(
    tp,
    fp,
    fn
):

    precision = precision_from_counts(
        tp,
        fp
    )

    recall = pod_from_counts(
        tp,
        fn
    )

    beta2 = 4.0

    denominator = (
        beta2 * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1.0 + beta2)
        * precision
        * recall
        / denominator
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================
#
# Primary objective:
#   maximize CSI
#
# Tie breaker 1:
#   higher POD
#
# Tie breaker 2:
#   lower FAR
#
# Thresholds are derived ONLY from pooled OOF
# development validation predictions.
#
# External test data are never read here.
# ============================================================

def threshold_candidates(
    probabilities
):

    # Fixed grid guarantees reproducibility and makes
    # the threshold-search procedure explicit.
    grid = np.arange(
        0.001,
        1.000,
        0.001,
        dtype=np.float64
    )

    # Add observed probability values around the best
    # grid points only if desired; here we intentionally
    # keep a fixed 0.001 grid for reproducibility.
    return grid


def select_threshold(
    y_true,
    y_prob
):

    candidates = threshold_candidates(
        y_prob
    )

    best = None

    for threshold in candidates:

        tp, fp, fn, tn = confusion_counts(
            y_true,
            y_prob,
            threshold
        )

        csi = csi_from_counts(
            tp,
            fp,
            fn
        )

        pod = pod_from_counts(
            tp,
            fn
        )

        far = far_from_counts(
            tp,
            fp
        )

        precision = precision_from_counts(
            tp,
            fp
        )

        f1 = f1_from_counts(
            tp,
            fp,
            fn
        )

        f2 = f2_from_counts(
            tp,
            fp,
            fn
        )

        candidate = {
            "threshold": float(threshold),
            "csi": float(csi),
            "pod": float(pod),
            "far": float(far),
            "precision": float(precision),
            "f1": float(f1),
            "f2": float(f2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

        if best is None:

            best = candidate

        else:

            # CSI primary
            if candidate["csi"] > best["csi"]:

                best = candidate

            elif np.isclose(
                candidate["csi"],
                best["csi"],
                rtol=0.0,
                atol=1e-12
            ):

                # POD first tie breaker
                if candidate["pod"] > best["pod"]:

                    best = candidate

                elif np.isclose(
                    candidate["pod"],
                    best["pod"],
                    rtol=0.0,
                    atol=1e-12
                ):

                    # FAR second tie breaker
                    if candidate["far"] < best["far"]:

                        best = candidate

    return best


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print(
        "STAGE D1 - OOF VALIDATION PREDICTIONS "
        "AND THRESHOLD GENERATION"
    )
    print("=" * 90)

    print(
        "Architecture :",
        ARCHITECTURE
    )

    print(
        "Optuna trial :",
        TRIAL
    )

    print(
        "Device       :",
        device
    )

    print(
        "Data         : DEVELOPMENT ONLY"
    )

    print(
        "External test: NOT USED"
    )

    print("=" * 90)


    # ========================================================
    # LOAD DEVELOPMENT X/Y
    # ========================================================

    X = np.load(
        DATA_PATH / "X.npy",
        mmap_mode="r"
    )

    y = np.load(
        DATA_PATH / "y.npy",
        mmap_mode="r"
    )

    print(
        "X shape      :",
        X.shape
    )

    print(
        "y shape      :",
        y.shape
    )


    if tuple(X.shape) != (
        847513,
        60,
        20
    ):

        raise RuntimeError(
            f"Unexpected X shape: {X.shape}"
        )

    if tuple(y.shape) != (
        847513,
        10
    ):

        raise RuntimeError(
            f"Unexpected y shape: {y.shape}"
        )


    # ========================================================
    # OOF PREDICTION STORAGE
    # ========================================================
    #
    # One prediction per development sample.
    # Validation samples from the five folds are expected
    # to collectively cover the development set according
    # to the Stage C fold manifest.
    # ========================================================

    oof_prob = np.full(
        (len(X), HORIZONS),
        np.nan,
        dtype=np.float32
    )

    oof_seen = np.zeros(
        len(X),
        dtype=np.int8
    )


    fold_audit = []


    # ========================================================
    # PROCESS EACH VALIDATION FOLD
    # ========================================================

    for fold in range(1, 6):

        print("\n" + "-" * 90)
        print(
            f"FOLD {fold}/5"
        )
        print("-" * 90)


        val_path = (
            STAGEC_PATH
            / "folds"
            / f"fold_{fold}_val.csv"
        )

        if not val_path.exists():

            raise FileNotFoundError(
                f"Missing validation manifest:\n"
                f"{val_path}"
            )


        val_df = pd.read_csv(
            val_path
        )

        if "anchor_idx" not in val_df.columns:

            raise RuntimeError(
                f"'anchor_idx' missing from:\n"
                f"{val_path}"
            )


        val_idx = (
            val_df["anchor_idx"]
            .astype(np.int64)
            .to_numpy()
        )


        if len(val_idx) == 0:

            raise RuntimeError(
                f"Fold {fold} has zero validation samples."
            )


        if val_idx.min() < 0 or val_idx.max() >= len(X):

            raise IndexError(
                f"Fold {fold} validation index outside X range: "
                f"min={val_idx.min()}, "
                f"max={val_idx.max()}, "
                f"X={len(X)}"
            )


        # Detect duplicate validation membership.
        duplicate_count = int(
            np.sum(
                oof_seen[val_idx] > 0
            )
        )

        if duplicate_count > 0:

            raise RuntimeError(
                f"OOF leakage/duplicate membership detected "
                f"at fold {fold}: "
                f"{duplicate_count} samples already predicted."
            )


        dataset = IndexedMemmapDataset(
            X,
            val_idx
        )


        loader_kwargs = {
            "dataset": dataset,
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "num_workers": NUM_WORKERS,
            "pin_memory": True
        }

        if NUM_WORKERS > 0:

            loader_kwargs[
                "persistent_workers"
            ] = True

            loader_kwargs[
                "prefetch_factor"
            ] = 4


        val_loader = DataLoader(
            **loader_kwargs
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = TCNBiLSTM(
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)


        checkpoint_path = find_checkpoint(
            fold
        )

        print(
            "Checkpoint  :",
            checkpoint_path
        )

        checkpoint = load_checkpoint(
            model,
            checkpoint_path,
            device
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        model.eval()

        fold_probabilities = np.empty(
            (len(val_idx), HORIZONS),
            dtype=np.float32
        )

        cursor = 0

        use_amp = (
            device.type == "cuda"
        )

        with torch.no_grad():

            for xb, batch_indices in val_loader:

                xb = xb.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=use_amp,
                    dtype=torch.float16
                ):

                    logits = model(xb)

                probabilities = (
                    torch.sigmoid(logits)
                    .float()
                    .cpu()
                    .numpy()
                )

                n = len(
                    batch_indices
                )

                fold_probabilities[
                    cursor:cursor+n
                ] = probabilities

                cursor += n


        if cursor != len(val_idx):

            raise RuntimeError(
                f"Fold {fold}: prediction count mismatch. "
                f"Expected {len(val_idx)}, got {cursor}"
            )


        # Put fold predictions into global OOF array.
        oof_prob[
            val_idx
        ] = fold_probabilities

        oof_seen[
            val_idx
        ] = 1


        fold_audit.append(
            {
                "fold": fold,
                "validation_samples": int(
                    len(val_idx)
                ),
                "checkpoint": str(
                    checkpoint_path
                ),
                "checkpoint_epoch":
                    checkpoint.get(
                        "epoch"
                    )
                    if isinstance(
                        checkpoint,
                        dict
                    )
                    else None
            }
        )


        print(
            "Validation samples:",
            len(val_idx)
        )

        print(
            "Predictions stored:",
            len(val_idx)
        )


    # ========================================================
    # OOF COVERAGE AUDIT
    # ========================================================

    seen_count = int(
        np.sum(
            oof_seen
        )
    )

    missing_count = int(
        np.sum(
            oof_seen == 0
        )
    )

    duplicate_count = int(
        np.sum(
            oof_seen > 1
        )
    )


    print("\n" + "=" * 90)
    print(
        "OOF COVERAGE AUDIT"
    )
    print("=" * 90)

    print(
        "Development samples:",
        len(X)
    )

    print(
        "OOF predictions     :",
        seen_count
    )

    print(
        "Missing predictions:",
        missing_count
    )

    print(
        "Duplicate samples  :",
        duplicate_count
    )


    # We require complete one-time OOF coverage.
    if missing_count != 0:

        raise RuntimeError(
            "\nOOF coverage is incomplete.\n"
            f"Missing samples: {missing_count}\n"
            "Do not calculate thresholds until the "
            "fold coverage is understood."
        )

    if duplicate_count != 0:

        raise RuntimeError(
            "\nDuplicate OOF predictions detected."
        )


    # ========================================================
    # POOLED OOF TARGETS
    # ========================================================

    # y is already aligned to anchor_idx in Stage D1 data.
    y_oof = np.asarray(
        y,
        dtype=np.float32
    )


    if np.isnan(
        oof_prob
    ).any():

        raise RuntimeError(
            "NaN remains in OOF predictions."
        )


    # ========================================================
    # SAVE OOF PREDICTIONS
    # ========================================================
    #
    # Keep sample-level OOF predictions together with
    # targets and anchor_idx.
    # ========================================================

    development_manifest = (
        STAGEC_PATH
        / "manifest"
        / "development_windows.csv"
    )

    if not development_manifest.exists():

        raise FileNotFoundError(
            f"Missing development manifest:\n"
            f"{development_manifest}"
        )


    manifest_df = pd.read_csv(
        development_manifest,
        usecols=[
            "sample_id",
            "anchor_idx",
            "anchor_time"
        ]
    )


    if len(manifest_df) != len(X):

        raise RuntimeError(
            "Development manifest length does not "
            "match X/y length."
        )


    # Ensure manifest is in anchor_idx order.
    if not np.array_equal(
        manifest_df["anchor_idx"]
        .astype(np.int64)
        .to_numpy(),
        np.arange(
            len(X),
            dtype=np.int64
        )
    ):

        raise RuntimeError(
            "development_windows.csv is not aligned "
            "to anchor_idx=0..N-1. "
            "Refusing to write potentially misaligned OOF data."
        )


    oof_output = manifest_df.copy()


    for h in range(
        1,
        HORIZONS + 1
    ):

        oof_output[
            f"y_h{h}"
        ] = y_oof[:, h-1].astype(
            np.int8
        )

        oof_output[
            f"p_h{h}"
        ] = oof_prob[:, h-1]


    oof_output.to_csv(
        OOF_PREDICTIONS_FILE,
        index=False
    )


    # ========================================================
    # THRESHOLD SELECTION
    # ========================================================

    print("\n" + "=" * 90)
    print(
        "THRESHOLD SELECTION FROM POOLED OOF"
    )
    print("=" * 90)

    threshold_results = []

    threshold_dict = {
        "method":
            "pooled_OOF_validation",

        "objective":
            "maximize_CSI",

        "tie_breaker_1":
            "higher_POD",

        "tie_breaker_2":
            "lower_FAR",

        "external_test_used":
            False,

        "horizons": {}
    }


    for h in range(
        1,
        HORIZONS + 1
    ):

        y_true = y_oof[
            :, h-1
        ]

        y_prob = oof_prob[
            :, h-1
        ]


        result = select_threshold(
            y_true,
            y_prob
        )


        threshold_dict[
            "horizons"
        ][
            f"H{h}"
        ] = result


        threshold_results.append(
            {
                "horizon":
                    f"H{h}",

                "threshold":
                    result["threshold"],

                "CSI":
                    result["csi"],

                "POD":
                    result["pod"],

                "FAR":
                    result["far"],

                "Precision":
                    result["precision"],

                "F1":
                    result["f1"],

                "F2":
                    result["f2"],

                "TP":
                    result["tp"],

                "FP":
                    result["fp"],

                "FN":
                    result["fn"],

                "TN":
                    result["tn"]
            }
        )


        print(
            f"H{h:02d}  "
            f"Threshold={result['threshold']:.3f}  "
            f"CSI={result['csi']:.5f}  "
            f"POD={result['pod']:.5f}  "
            f"FAR={result['far']:.5f}"
        )


    # ========================================================
    # SAVE THRESHOLDS
    # ========================================================

    with open(
        THRESHOLD_FILE,
        "w"
    ) as f:

        json.dump(
            threshold_dict,
            f,
            indent=2
        )


    pd.DataFrame(
        threshold_results
    ).to_csv(
        THRESHOLD_SUMMARY_FILE,
        index=False
    )


    # ========================================================
    # AUDIT ARTIFACT
    # ========================================================

    audit = {

        "stage":
            "D1 OOF threshold generation",

        "architecture":
            ARCHITECTURE,

        "optuna_trial":
            TRIAL,

        "seed":
            SEED,

        "development_samples":
            int(len(X)),

        "oof_samples":
            seen_count,

        "missing_oof_samples":
            missing_count,

        "duplicate_oof_samples":
            duplicate_count,

        "folds":
            fold_audit,

        "external_test_used":
            False,

        "external_test_path_read":
            False,

        "threshold_method":
            "pooled OOF validation",

        "threshold_objective":
            "maximum CSI",

        "tie_breaker_1":
            "higher POD",

        "tie_breaker_2":
            "lower FAR",

        "threshold_grid":
            "0.001 to 0.999, step 0.001",

        "oof_predictions_file":
            str(
                OOF_PREDICTIONS_FILE
            ),

        "threshold_file":
            str(
                THRESHOLD_FILE
            ),

        "threshold_summary_file":
            str(
                THRESHOLD_SUMMARY_FILE
            )
    }


    with open(
        OOF_AUDIT_FILE,
        "w"
    ) as f:

        json.dump(
            audit,
            f,
            indent=2
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 90)
    print(
        "OOF THRESHOLD GENERATION FINISHED"
    )
    print("=" * 90)

    print(
        "OOF predictions :",
        OOF_PREDICTIONS_FILE
    )

    print(
        "Thresholds      :",
        THRESHOLD_FILE
    )

    print(
        "Summary         :",
        THRESHOLD_SUMMARY_FILE
    )

    print(
        "Audit           :",
        OOF_AUDIT_FILE
    )

    print(
        "\nExternal test was NOT read."
    )

    print(
        "Thresholds are now development-derived "
        "and can be frozen before external testing."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
