import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.tcn_bilstm import TCNBiLSTM


# ============================================================
# CONFIGURATION
# ============================================================

with open(
    PROJECT_ROOT / "config" / "config.json",
    "r"
) as f:
    CFG = json.load(f)

DATA_PATH = PROJECT_ROOT / CFG["data_path"]
STAGEC_PATH = PROJECT_ROOT / CFG["stageC_path"]

ARCHITECTURE = "TCN-BiLSTM"
TRIAL = 26
SEED = 2026

BATCH_SIZE = 256
NUM_WORKERS = int(CFG.get("num_workers", 2))

HORIZONS = 10

# Trial-26 architecture
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
# MEMMAP DATASET
# ============================================================

class IndexedMemmapDataset(Dataset):

    def __init__(
        self,
        X,
        row_indices
    ):
        self.X = X
        self.row_indices = np.asarray(
            row_indices,
            dtype=np.int64
        )

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, i):

        row = int(
            self.row_indices[i]
        )

        x = np.asarray(
            self.X[row],
            dtype=np.float32
        )

        return torch.from_numpy(x)


# ============================================================
# CHECKPOINT
# ============================================================

def find_checkpoint(fold):

    candidates = [

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
        / f"fold_{fold}.pt",

        PROJECT_ROOT
        / "results"
        / "optuna"
        / "checkpoints"
        / ARCHITECTURE
        / f"trial_{TRIAL}"
        / f"fold_{fold}.pth",

        PROJECT_ROOT
        / "results"
        / "optuna"
        / ARCHITECTURE
        / f"trial_{TRIAL}"
        / f"fold_{fold}.pth"
    ]

    for path in candidates:

        if path.exists():
            return path

    raise FileNotFoundError(
        f"\nCheckpoint for fold {fold} "
        "was not found.\n\n"
        + "\n".join(
            str(p)
            for p in candidates
        )
    )


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

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            f"Unsupported checkpoint format:\n"
            f"{checkpoint_path}"
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
        isinstance(v, torch.Tensor)
        for v in checkpoint.values()
    ):

        state_dict = checkpoint

    else:

        raise RuntimeError(
            "Could not identify model state_dict "
            f"in:\n{checkpoint_path}"
        )

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

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

    tp = int(
        np.sum(pred & truth)
    )

    fp = int(
        np.sum(pred & ~truth)
    )

    fn = int(
        np.sum(~pred & truth)
    )

    tn = int(
        np.sum(~pred & ~truth)
    )

    return tp, fp, fn, tn


def csi(tp, fp, fn):

    denominator = (
        tp + fp + fn
    )

    return (
        tp / denominator
        if denominator > 0
        else 0.0
    )


def pod(tp, fn):

    denominator = (
        tp + fn
    )

    return (
        tp / denominator
        if denominator > 0
        else 0.0
    )


def far(tp, fp):

    denominator = (
        tp + fp
    )

    return (
        fp / denominator
        if denominator > 0
        else 0.0
    )


def precision(tp, fp):

    denominator = (
        tp + fp
    )

    return (
        tp / denominator
        if denominator > 0
        else 0.0
    )


def f1_score(
    tp,
    fp,
    fn
):

    p = precision(
        tp,
        fp
    )

    r = pod(
        tp,
        fn
    )

    denominator = p + r

    return (
        2.0 * p * r / denominator
        if denominator > 0
        else 0.0
    )


def f2_score(
    tp,
    fp,
    fn
):

    p = precision(
        tp,
        fp
    )

    r = pod(
        tp,
        fn
    )

    beta2 = 4.0

    denominator = (
        beta2 * p + r
    )

    return (
        5.0 * p * r / denominator
        if denominator > 0
        else 0.0
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def select_threshold(
    y_true,
    y_prob
):

    best = None

    # Fixed reproducible threshold grid.
    for threshold in np.arange(
        0.001,
        1.000,
        0.001
    ):

        tp, fp, fn, tn = (
            confusion_counts(
                y_true,
                y_prob,
                threshold
            )
        )

        score_csi = csi(
            tp,
            fp,
            fn
        )

        score_pod = pod(
            tp,
            fn
        )

        score_far = far(
            tp,
            fp
        )

        result = {
            "threshold": float(threshold),
            "CSI": float(score_csi),
            "POD": float(score_pod),
            "FAR": float(score_far),
            "Precision": float(
                precision(tp, fp)
            ),
            "F1": float(
                f1_score(tp, fp, fn)
            ),
            "F2": float(
                f2_score(tp, fp, fn)
            ),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        }

        if best is None:

            best = result
            continue

        # Primary: maximum CSI
        if result["CSI"] > best["CSI"] + 1e-12:

            best = result
            continue

        # Tie 1: higher POD
        if np.isclose(
            result["CSI"],
            best["CSI"],
            atol=1e-12,
            rtol=0.0
        ):

            if result["POD"] > best["POD"] + 1e-12:

                best = result
                continue

            # Tie 2: lower FAR
            if np.isclose(
                result["POD"],
                best["POD"],
                atol=1e-12,
                rtol=0.0
            ):

                if result["FAR"] < best["FAR"]:

                    best = result

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

    print("Architecture :", ARCHITECTURE)
    print("Optuna trial :", TRIAL)
    print("Device       :", device)
    print("Data         : DEVELOPMENT ONLY")
    print("External test: NOT USED")
    print("=" * 90)


    # ========================================================
    # LOAD X / Y
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
    # DEVELOPMENT MANIFEST
    # ========================================================
    #
    # CRITICAL:
    # Stage C anchor_idx is a GLOBAL timeline index.
    # X/y are COMPACT Stage D1 development-window arrays.
    #
    # Therefore we MUST map:
    #
    #   global anchor_idx
    #          ↓
    #   compact X/y row position
    #
    # We must NOT use X[anchor_idx].
    # ========================================================

    manifest_path = (
        STAGEC_PATH
        / "manifest"
        / "development_windows.csv"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Missing:\n{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        usecols=[
            "sample_id",
            "anchor_idx",
            "anchor_time"
        ]
    )

    if len(manifest) != len(X):

        raise RuntimeError(
            "Development manifest length mismatch: "
            f"manifest={len(manifest)}, "
            f"X={len(X)}"
        )

    anchor_array = (
        manifest["anchor_idx"]
        .astype(np.int64)
        .to_numpy()
    )

    if len(
        np.unique(anchor_array)
    ) != len(anchor_array):

        raise RuntimeError(
            "Duplicate anchor_idx values in "
            "development_windows.csv."
        )

    anchor_to_row = {
        int(anchor): int(row)
        for row, anchor
        in enumerate(anchor_array)
    }

    print(
        "Manifest rows:",
        len(manifest)
    )

    print(
        "Global anchor range:",
        int(anchor_array.min()),
        "to",
        int(anchor_array.max())
    )


    # ========================================================
    # FOLD PROCESSING
    # ========================================================

    pooled_oof = []

    fold_audit = []

    for fold in range(
        1,
        6
    ):

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
                f"Missing:\n{val_path}"
            )

        val_df = pd.read_csv(
            val_path
        )

        if "anchor_idx" not in val_df.columns:

            raise RuntimeError(
                f"anchor_idx missing from {val_path}"
            )

        val_anchor = (
            val_df["anchor_idx"]
            .astype(np.int64)
            .to_numpy()
        )

        if len(val_anchor) == 0:

            raise RuntimeError(
                f"Fold {fold} has no validation rows."
            )


        # Map global anchors to compact Stage D1 rows.
        missing = [
            int(a)
            for a in val_anchor
            if int(a) not in anchor_to_row
        ]

        if missing:

            raise KeyError(
                f"Fold {fold}: "
                f"{len(missing)} anchor_idx values "
                "are absent from development_windows.csv. "
                f"Examples: {missing[:10]}"
            )

        val_rows = np.asarray(
            [
                anchor_to_row[int(a)]
                for a in val_anchor
            ],
            dtype=np.int64
        )


        # ----------------------------------------------------
        # DATA LOADER
        # ----------------------------------------------------

        dataset = IndexedMemmapDataset(
            X,
            val_rows
        )

        loader_args = {
            "dataset": dataset,
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "num_workers": NUM_WORKERS,
            "pin_memory": True
        }

        if NUM_WORKERS > 0:

            loader_args[
                "persistent_workers"
            ] = True

            loader_args[
                "prefetch_factor"
            ] = 4

        val_loader = DataLoader(
            **loader_args
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
        # PREDICT
        # ----------------------------------------------------

        probabilities = np.empty(
            (
                len(val_rows),
                HORIZONS
            ),
            dtype=np.float32
        )

        cursor = 0

        use_amp = (
            device.type == "cuda"
        )

        with torch.no_grad():

            for xb in val_loader:

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

                batch_prob = (
                    torch.sigmoid(logits)
                    .float()
                    .cpu()
                    .numpy()
                )

                n = batch_prob.shape[0]

                probabilities[
                    cursor:cursor+n
                ] = batch_prob

                cursor += n


        if cursor != len(val_rows):

            raise RuntimeError(
                f"Fold {fold} prediction count mismatch: "
                f"{cursor} vs {len(val_rows)}"
            )


        # ----------------------------------------------------
        # BUILD FOLD OOF TABLE
        # ----------------------------------------------------

        out = pd.DataFrame(
            {
                "sample_id":
                    val_df["sample_id"].to_numpy()
                    if "sample_id" in val_df.columns
                    else manifest.iloc[
                        val_rows
                    ]["sample_id"].to_numpy(),

                "anchor_idx":
                    val_anchor,

                "anchor_time":
                    val_df["anchor_time"].to_numpy()
                    if "anchor_time" in val_df.columns
                    else manifest.iloc[
                        val_rows
                    ]["anchor_time"].to_numpy(),

                "fold":
                    np.full(
                        len(val_rows),
                        fold,
                        dtype=np.int8
                    )
            }
        )


        for h in range(
            1,
            HORIZONS + 1
        ):

            out[
                f"y_h{h}"
            ] = y[
                val_rows,
                h - 1
            ].astype(
                np.int8
            )

            out[
                f"p_h{h}"
            ] = probabilities[
                :,
                h - 1
            ]


        pooled_oof.append(
            out
        )


        fold_audit.append(
            {
                "fold": fold,

                "validation_samples":
                    int(len(val_rows)),

                "min_global_anchor_idx":
                    int(val_anchor.min()),

                "max_global_anchor_idx":
                    int(val_anchor.max()),

                "min_compact_row":
                    int(val_rows.min()),

                "max_compact_row":
                    int(val_rows.max()),

                "checkpoint":
                    str(checkpoint_path),

                "checkpoint_epoch":
                    checkpoint.get("epoch")
                    if isinstance(
                        checkpoint,
                        dict
                    )
                    else None
            }
        )


        print(
            "Validation samples:",
            len(val_rows)
        )

        print(
            "Compact X rows:",
            int(val_rows.min()),
            "to",
            int(val_rows.max())
        )

        print(
            "Predictions stored:",
            len(val_rows)
        )


    # ========================================================
    # POOLED OOF AUDIT
    # ========================================================

    oof = pd.concat(
        pooled_oof,
        ignore_index=True
    )

    duplicate_count = int(
        oof["anchor_idx"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        raise RuntimeError(
            "Duplicate OOF anchor_idx detected: "
            f"{duplicate_count}"
        )

    expected_oof = sum(
        item["validation_samples"]
        for item in fold_audit
    )

    if len(oof) != expected_oof:

        raise RuntimeError(
            "Pooled OOF count mismatch: "
            f"expected={expected_oof}, "
            f"actual={len(oof)}"
        )


    print("\n" + "=" * 90)
    print(
        "OOF COVERAGE AUDIT"
    )
    print("=" * 90)

    print(
        "Development windows :",
        len(X)
    )

    print(
        "Pooled OOF samples  :",
        len(oof)
    )

    print(
        "OOF coverage ratio  :",
        f"{len(oof) / len(X):.6f}"
    )

    print(
        "Duplicate anchors   :",
        duplicate_count
    )

    print(
        "\nNOTE:"
    )

    print(
        "Expanding-window validation folds are pooled."
    )

    print(
        "Complete coverage of all development windows "
        "is NOT required for OOF threshold estimation."
    )


    # ========================================================
    # SAVE OOF PREDICTIONS
    # ========================================================

    ordered_columns = [
        "sample_id",
        "anchor_idx",
        "anchor_time",
        "fold"
    ]

    ordered_columns += [
        f"y_h{h}"
        for h in range(1, HORIZONS + 1)
    ]

    ordered_columns += [
        f"p_h{h}"
        for h in range(1, HORIZONS + 1)
    ]

    oof = oof[
        ordered_columns
    ]

    oof.to_csv(
        OOF_PREDICTIONS_FILE,
        index=False
    )


    # ========================================================
    # THRESHOLD SEARCH
    # ========================================================

    print("\n" + "=" * 90)
    print(
        "THRESHOLD SELECTION FROM POOLED OOF"
    )
    print("=" * 90)

    threshold_summary = []

    threshold_json = {
        "method":
            "pooled_OOF_validation",

        "objective":
            "maximum_CSI",

        "tie_breaker_1":
            "higher_POD",

        "tie_breaker_2":
            "lower_FAR",

        "threshold_grid":
            "0.001 to 0.999 step 0.001",

        "external_test_used":
            False,

        "horizons": {}
    }


    for h in range(
        1,
        HORIZONS + 1
    ):

        y_true = oof[
            f"y_h{h}"
        ].to_numpy(
            dtype=np.float32
        )

        y_prob = oof[
            f"p_h{h}"
        ].to_numpy(
            dtype=np.float32
        )

        result = select_threshold(
            y_true,
            y_prob
        )

        threshold_json[
            "horizons"
        ][
            f"H{h}"
        ] = result

        threshold_summary.append(
            {
                "horizon":
                    f"H{h}",

                "threshold":
                    result["threshold"],

                "CSI":
                    result["CSI"],

                "POD":
                    result["POD"],

                "FAR":
                    result["FAR"],

                "Precision":
                    result["Precision"],

                "F1":
                    result["F1"],

                "F2":
                    result["F2"],

                "TP":
                    result["TP"],

                "FP":
                    result["FP"],

                "FN":
                    result["FN"],

                "TN":
                    result["TN"]
            }
        )

        print(
            f"H{h:02d}  "
            f"Threshold={result['threshold']:.3f}  "
            f"CSI={result['CSI']:.5f}  "
            f"POD={result['POD']:.5f}  "
            f"FAR={result['FAR']:.5f}"
        )


    # ========================================================
    # SAVE THRESHOLDS
    # ========================================================

    with open(
        THRESHOLD_FILE,
        "w"
    ) as f:

        json.dump(
            threshold_json,
            f,
            indent=2
        )

    pd.DataFrame(
        threshold_summary
    ).to_csv(
        THRESHOLD_SUMMARY_FILE,
        index=False
    )


    # ========================================================
    # AUDIT
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

        "pooled_oof_samples":
            int(len(oof)),

        "oof_coverage_ratio":
            float(len(oof) / len(X)),

        "duplicate_oof_samples":
            duplicate_count,

        "folds":
            fold_audit,

        "anchor_mapping":
            "Stage C global anchor_idx -> "
            "Stage D1 compact X/y row",

        "threshold_method":
            "pooled OOF validation",

        "threshold_objective":
            "maximum CSI",

        "tie_breaker_1":
            "higher POD",

        "tie_breaker_2":
            "lower FAR",

        "threshold_grid":
            "0.001 to 0.999 step 0.001",

        "external_test_used":
            False,

        "external_test_path_read":
            False,

        "oof_predictions_file":
            str(OOF_PREDICTIONS_FILE),

        "threshold_file":
            str(THRESHOLD_FILE),

        "threshold_summary_file":
            str(THRESHOLD_SUMMARY_FILE)
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
        "Thresholds are development-derived."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
