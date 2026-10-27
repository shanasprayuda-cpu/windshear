"""
Stage D1 - Feature Engineering Comparison
F1 (20) vs F2 (37) vs F3 (45), fixed TCN-BiLSTM architecture.

Purpose
-------
Compare the effect of feature engineering while holding the model/training
protocol constant. This script uses DEVELOPMENT data only for selection.

Important methodological rules
------------------------------
1. External Jan-Mar 2025 data are NEVER read.
2. Fold membership comes from Stage C manifests.
3. Fold CSV anchor_idx is mapped to development-window row positions via
   development_windows.csv; raw anchor_idx is NOT used directly as an X row.
4. Scaling is train-only per fold.
5. The same folds, epochs, batch size, optimizer family and seed are used
   across F1/F2/F3.
6. No learning-rate scheduler is used.
7. TCN-BiLSTM is fixed; feature set is the experimental factor.
8. Thresholds are not selected on external data. Validation thresholds are
   generated from each fold's validation predictions using the same CSI-first
   rule only for descriptive validation metrics.
9. This script does NOT produce the final external-test result.

Expected input layout
---------------------
Project parent:
    stageC/manifest/development_windows.csv
    stageC/folds/fold_1_train.csv ... fold_5_val.csv
    stageD1_data/F1/X.npy, y.npy
    stageD1_data/F2/X.npy, y.npy
    stageD1_data/F3/X.npy, y.npy

If F2/F3 are stored elsewhere, pass --feature_root.

Output
------
results/feature_engineering_comparison/
    comparison_summary.csv
    fold_results.csv
    audit.json
    F1/
    F2/
    F3/

Run
---
python scripts/feature_engineering_comparison.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PARENT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.tcn_bilstm import TCNBiLSTM
except Exception as exc:
    raise ImportError(
        "\nCould not import models.tcn_bilstm.TCNBiLSTM.\n"
        "Make sure the model file used by the final TCN-BiLSTM retraining exists.\n"
        f"Original error: {exc}\n"
    )

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

FEATURE_SETS = {
    "F1": 20,
    "F2": 37,
    "F3": 45,
}

HORIZONS = 10
LOOKBACK = 60
SEED = 2026

# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compare F1/F2/F3 using fixed TCN-BiLSTM on Stage C development folds."
)

parser.add_argument(
    "--feature_root",
    default=None,
    help=(
        "Root containing F1/F2/F3 directories. Default: "
        "<project>/stageD1_data"
    ),
)

parser.add_argument(
    "--epochs",
    type=int,
    default=50,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=None,
    help=(
        "Optional LR override. If omitted, script reads the final "
        "TCN-BiLSTM config if available."
    ),
)

parser.add_argument(
    "--weight_decay",
    type=float,
    default=None,
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=2,
)

parser.add_argument(
    "--patience",
    type=int,
    default=7,
)

parser.add_argument(
    "--output_root",
    default=None,
)

parser.add_argument(
    "--device",
    default=None,
)

args = parser.parse_args()

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

FEATURE_ROOT = (
    Path(args.feature_root)
    if args.feature_root
    else PROJECT_ROOT / "stageD1_data"
)

STAGEC_ROOT = (
    PARENT_ROOT / "stageC"
    if (PARENT_ROOT / "stageC").exists()
    else PROJECT_ROOT / "stageC"
)

DEV_MANIFEST = STAGEC_ROOT / "manifest" / "development_windows.csv"
FOLD_ROOT = STAGEC_ROOT / "folds"

OUTPUT_ROOT = (
    Path(args.output_root)
    if args.output_root
    else PROJECT_ROOT / "results" / "feature_engineering_comparison"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep deterministic behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

DEVICE = torch.device(
    args.device
    if args.device
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# ---------------------------------------------------------------------
# CONFIG DISCOVERY
# ---------------------------------------------------------------------

def find_final_config():
    candidates = [
        PROJECT_ROOT
        / "results"
        / "final_full"
        / "FINAL_TCN_BiLSTM_TRIAL26_FULL"
        / "final_config.json",
        PROJECT_ROOT
        / "results"
        / "final_full"
        / "final_config.json",
        PROJECT_ROOT
        / "results"
        / "optuna"
        / "final_config.json",
    ]

    found = [p for p in candidates if p.exists()]

    if not found:
        return None

    return found[0]


FINAL_CONFIG_PATH = find_final_config()


def load_training_config():
    cfg = {}

    if FINAL_CONFIG_PATH is not None:
        with open(FINAL_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    def recursive_find(obj, keys):
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    return obj[k]
            for v in obj.values():
                result = recursive_find(v, keys)
                if result is not None:
                    return result
        elif isinstance(obj, list):
            for v in obj:
                result = recursive_find(v, keys)
                if result is not None:
                    return result
        return None

    lr = args.learning_rate
    wd = args.weight_decay

    if lr is None:
        lr = recursive_find(
            cfg,
            ["learning_rate", "lr"]
        )

    if wd is None:
        wd = recursive_find(
            cfg,
            ["weight_decay"]
        )

    # These are deliberately conservative defaults only when the final
    # config does not contain the values.
    if lr is None:
        raise RuntimeError(
            "Could not determine the final TCN-BiLSTM learning rate. "
            "Pass --learning_rate explicitly."
        )

    if wd is None:
        wd = 0.0

    return float(lr), float(wd), cfg


LEARNING_RATE, WEIGHT_DECAY, FINAL_CFG = load_training_config()

# ---------------------------------------------------------------------
# DATA DISCOVERY
# ---------------------------------------------------------------------

def require_feature_files(name):
    root = FEATURE_ROOT / name
    x = root / "X.npy"
    y = root / "y.npy"

    if not x.exists() or not y.exists():
        raise FileNotFoundError(
            f"\nMissing {name} data.\n"
            f"Expected:\n{x}\n{y}\n\n"
            "Create/export the feature-set arrays first."
        )

    return x, y


def load_development_manifest():
    if not DEV_MANIFEST.exists():
        raise FileNotFoundError(
            f"Development manifest not found:\n{DEV_MANIFEST}"
        )

    df = pd.read_csv(DEV_MANIFEST)

    required = [
        "sample_id",
        "dataset_role",
        "anchor_idx",
        "anchor_time",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            "Development manifest missing columns: "
            + ", ".join(missing)
        )

    if len(df) != 847513:
        print(
            f"WARNING: development manifest has {len(df)} rows; "
            "expected 847513 from the locked F1 export."
        )

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in development manifest.")

    if df["anchor_idx"].duplicated().any():
        raise ValueError("Duplicate anchor_idx in development manifest.")

    return df.reset_index(drop=True)


def load_fold_csv(fold, role):
    path = FOLD_ROOT / f"fold_{fold}_{role}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing fold file:\n{path}")

    df = pd.read_csv(path)

    if "anchor_idx" not in df.columns:
        raise KeyError(f"{path} has no anchor_idx column.")

    return df


DEV_MANIFEST_DF = load_development_manifest()

ANCHOR_TO_ROW = pd.Series(
    DEV_MANIFEST_DF.index.to_numpy(),
    index=DEV_MANIFEST_DF["anchor_idx"].to_numpy(),
)

# ---------------------------------------------------------------------
# FOLD INDEX AUDIT
# ---------------------------------------------------------------------

def convert_fold_to_rows(fold_df, fold, role):
    anchors = fold_df["anchor_idx"].to_numpy()

    missing = [
        int(a)
        for a in anchors
        if a not in ANCHOR_TO_ROW.index
    ]

    if missing:
        raise RuntimeError(
            f"Fold {fold} {role}: {len(missing)} anchor_idx values "
            "are not present in development_windows.csv.\n"
            f"First missing values: {missing[:10]}"
        )

    rows = ANCHOR_TO_ROW.loc[anchors].to_numpy(dtype=np.int64)

    if len(np.unique(rows)) != len(rows):
        raise RuntimeError(
            f"Fold {fold} {role}: duplicate mapped development rows."
        )

    return rows


# ---------------------------------------------------------------------
# MODEL CONSTRUCTION
# ---------------------------------------------------------------------

def build_tcn_bilstm(input_features):
    """
    Use the same TCN-BiLSTM implementation as the final model.

    The final implementation must accept the input feature dimension.
    Common accepted keyword names are handled without changing the model
    architecture itself.
    """
    attempts = [
        {"input_size": input_features},
        {"input_dim": input_features},
        {"in_features": input_features},
        {"num_features": input_features},
        {"feature_count": input_features},
    ]

    errors = []

    for kwargs in attempts:
        try:
            return TCNBiLSTM(**kwargs)
        except TypeError as exc:
            errors.append(str(exc))

    raise RuntimeError(
        "\nThe installed TCNBiLSTM class does not expose an input-feature "
        "dimension argument.\n\n"
        "This comparison must not silently replace the final architecture "
        "with a different implementation.\n"
        "Update models/tcn_bilstm.py so the same TCN-BiLSTM architecture "
        "can accept 20, 37 and 45 input channels.\n\n"
        "Constructor errors:\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------
# POSITIVE WEIGHTS
# ---------------------------------------------------------------------

def compute_pos_weight(y_train):
    """
    Per-horizon positive weight:
        negatives / positives
    """
    y_train = np.asarray(y_train)

    pos = y_train.sum(axis=0).astype(np.float64)
    neg = y_train.shape[0] - pos

    if np.any(pos <= 0):
        bad = np.where(pos <= 0)[0] + 1
        raise RuntimeError(
            f"Zero-positive horizon(s) in training data: {bad.tolist()}"
        )

    return (
        neg / pos
    ).astype(np.float32)


# ---------------------------------------------------------------------
# DATASET / LOADER
# ---------------------------------------------------------------------

class NumpyWindowDataset(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path, rows):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.rows = np.asarray(rows, dtype=np.int64)

        if self.X.ndim != 3:
            raise ValueError(
                f"X must be 3-D; got {self.X.shape}"
            )

        if self.y.ndim != 2:
            raise ValueError(
                f"y must be 2-D; got {self.y.shape}"
            )

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                "X/y sample count mismatch."
            )

        if self.X.shape[1] != LOOKBACK:
            raise ValueError(
                f"Expected lookback={LOOKBACK}; got {self.X.shape[1]}"
            )

        if self.y.shape[1] != HORIZONS:
            raise ValueError(
                f"Expected 10 horizons; got {self.y.shape[1]}"
            )

        if self.rows.min() < 0 or self.rows.max() >= self.X.shape[0]:
            raise IndexError(
                f"Mapped fold rows outside X range: "
                f"min={self.rows.min()}, max={self.rows.max()}, "
                f"X={self.X.shape[0]}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = int(self.rows[idx])

        x = np.asarray(
            self.X[row],
            dtype=np.float32
        )

        y = np.asarray(
            self.y[row],
            dtype=np.float32
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
        )


def make_loader(dataset, shuffle):
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    if args.num_workers > 0:
        kwargs["prefetch_factor"] = 4

    return torch.utils.data.DataLoader(
        dataset,
        **kwargs
    )


# ---------------------------------------------------------------------
# TRAIN / VALIDATE
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()

    running_loss = 0.0
    count = 0

    for x, y in loader:
        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        if logits.shape != y.shape:
            raise RuntimeError(
                f"Model output {tuple(logits.shape)} "
                f"does not match target {tuple(y.shape)}."
            )

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        bs = x.shape[0]

        running_loss += (
            float(loss.detach().cpu())
            * bs
        )

        count += bs

    return running_loss / max(count, 1)


@torch.no_grad()
def predict(model, loader):
    model.eval()

    probs = []
    targets = []

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(x)

        p = torch.sigmoid(
            logits
        ).detach().cpu().numpy()

        probs.append(p)
        targets.append(
            y.numpy()
        )

    return (
        np.concatenate(probs, axis=0),
        np.concatenate(targets, axis=0),
    )


# ---------------------------------------------------------------------
# VALIDATION METRICS
# ---------------------------------------------------------------------

def best_csi_threshold(y_true, prob):
    """
    Descriptive validation-only threshold.

    Threshold is NOT transferred to the external test by this script.
    Tie break: POD, then lower FAR.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    prob = np.asarray(prob)

    if y_true.sum() == 0:
        return 0.5, np.nan, np.nan, np.nan, np.nan

    thresholds = np.unique(
        np.clip(
            prob,
            0.001,
            0.999
        )
    )

    # Include practical endpoints.
    thresholds = np.unique(
        np.concatenate(
            [
                thresholds,
                np.array(
                    [0.01, 0.05, 0.10, 0.20, 0.30,
                     0.40, 0.50, 0.60, 0.70, 0.80,
                     0.90, 0.95, 0.99],
                    dtype=np.float64
                )
            ]
        )
    )

    best = None

    for t in thresholds:
        pred = (
            prob >= t
        ).astype(np.int8)

        tp = np.sum(
            (pred == 1) & (y_true == 1)
        )
        fp = np.sum(
            (pred == 1) & (y_true == 0)
        )
        fn = np.sum(
            (pred == 0) & (y_true == 1)
        )

        denom = tp + fp + fn

        if denom == 0:
            continue

        csi = tp / denom
        pod = tp / max(tp + fn, 1)
        far = fp / max(tp + fp, 1)

        candidate = (
            csi,
            pod,
            -far,
            -abs(float(t) - 0.5),
            float(t),
        )

        if best is None or candidate > best[0]:
            best = (
                candidate,
                float(t),
                float(csi),
                float(pod),
                float(far),
            )

    if best is None:
        return 0.5, 0.0, 0.0, 1.0

    return (
        best[1],
        best[2],
        best[3],
        best[4],
    )


def calculate_metrics(y_true, prob, threshold):
    y_true = np.asarray(y_true).astype(np.int8)
    prob = np.asarray(prob)

    pred = (
        prob >= threshold
    ).astype(np.int8)

    tp = int(np.sum(
        (pred == 1) & (y_true == 1)
    ))
    fp = int(np.sum(
        (pred == 1) & (y_true == 0)
    ))
    fn = int(np.sum(
        (pred == 0) & (y_true == 1)
    ))
    tn = int(np.sum(
        (pred == 0) & (y_true == 0)
    ))

    csi = tp / max(
        tp + fp + fn,
        1
    )

    pod = tp / max(
        tp + fn,
        1
    )

    precision = tp / max(
        tp + fp,
        1
    )

    far = fp / max(
        tp + fp,
        1
    )

    f1 = (
        2 * precision * pod
        / max(precision + pod, 1e-12)
    )

    f2 = (
        5 * precision * pod
        / max(4 * precision + pod, 1e-12)
    )

    try:
        auprc = average_precision_score(
            y_true,
            prob
        )
    except Exception:
        auprc = np.nan

    try:
        roc_auc = roc_auc_score(
            y_true,
            prob
        )
    except Exception:
        roc_auc = np.nan

    return {
        "threshold": float(threshold),
        "AUPRC": float(auprc),
        "ROC_AUC": float(roc_auc),
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
        "events": int(y_true.sum()),
    }


# ---------------------------------------------------------------------
# ONE FEATURE SET
# ---------------------------------------------------------------------

def run_feature_set(name, x_path, y_path):
    expected_features = FEATURE_SETS[name]

    print("\n" + "=" * 90)
    print(f"FEATURE SET: {name}")
    print(f"Expected features: {expected_features}")
    print("=" * 90)

    X = np.load(
        x_path,
        mmap_mode="r"
    )

    y = np.load(
        y_path,
        mmap_mode="r"
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    if X.shape[1] != LOOKBACK:
        raise RuntimeError(
            f"{name}: expected 60 input minutes, got {X.shape[1]}"
        )

    if X.shape[2] != expected_features:
        raise RuntimeError(
            f"{name}: expected {expected_features} features, "
            f"got {X.shape[2]}"
        )

    if y.shape[1] != HORIZONS:
        raise RuntimeError(
            f"{name}: expected 10 horizons, got {y.shape[1]}"
        )

    if X.shape[0] != y.shape[0]:
        raise RuntimeError(
            f"{name}: X/y sample count mismatch."
        )

    # --------------------------------------------------------------
    # FOLD LOOP
    # --------------------------------------------------------------

    fold_rows = []
    pooled_probs = []
    pooled_targets = []

    feature_output = OUTPUT_ROOT / name
    feature_output.mkdir(
        parents=True,
        exist_ok=True
    )

    for fold in range(1, 6):

        print(
            f"\nFOLD {fold}/5"
        )

        train_df = load_fold_csv(
            fold,
            "train"
        )

        val_df = load_fold_csv(
            fold,
            "val"
        )

        train_rows = convert_fold_to_rows(
            train_df,
            fold,
            "train"
        )

        val_rows = convert_fold_to_rows(
            val_df,
            fold,
            "val"
        )

        # Expanding-window sanity check:
        if set(train_rows).intersection(
            set(val_rows)
        ):
            raise RuntimeError(
                f"{name} fold {fold}: train/validation overlap."
            )

        train_dataset = NumpyWindowDataset(
            x_path,
            y_path,
            train_rows
        )

        val_dataset = NumpyWindowDataset(
            x_path,
            y_path,
            val_rows
        )

        train_loader = make_loader(
            train_dataset,
            shuffle=True
        )

        val_loader = make_loader(
            val_dataset,
            shuffle=False
        )

        # ----------------------------------------------------------
        # MODEL
        # ----------------------------------------------------------

        seed_everything(
            SEED
        )

        model = build_tcn_bilstm(
            expected_features
        ).to(DEVICE)

        parameter_count = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            "Trainable parameters:",
            f"{parameter_count:,}"
        )

        # ----------------------------------------------------------
        # POS WEIGHT
        # ----------------------------------------------------------

        y_train = np.asarray(
            y[train_rows],
            dtype=np.float32
        )

        pos_weight_np = compute_pos_weight(
            y_train
        )

        pos_weight = torch.tensor(
            pos_weight_np,
            dtype=torch.float32,
            device=DEVICE
        )

        criterion = (
            torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weight
            )
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        best_val_loss = math.inf
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0

        # ----------------------------------------------------------
        # TRAINING
        # ----------------------------------------------------------

        for epoch in range(
            1,
            args.epochs + 1
        ):

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            val_prob, val_y = predict(
                model,
                val_loader
            )

            val_loss = 0.0

            # BCEWithLogits validation loss calculated from probabilities
            eps = 1e-7
            clipped = np.clip(
                val_prob,
                eps,
                1 - eps
            )

            for h in range(HORIZONS):
                pw = float(
                    pos_weight_np[h]
                )

                yt = val_y[:, h]
                p = clipped[:, h]

                bce = (
                    -pw * yt * np.log(p)
                    - (1 - yt) * np.log(1 - p)
                )

                val_loss += float(
                    np.mean(bce)
                )

            val_loss /= HORIZONS

            print(
                f"Epoch {epoch:02d}/{args.epochs} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}."
                )
                break

        if best_state is None:
            raise RuntimeError(
                f"{name} fold {fold}: no best checkpoint."
            )

        model.load_state_dict(
            best_state
        )
        model.to(DEVICE)

        # ----------------------------------------------------------
        # FINAL FOLD VALIDATION PREDICTION
        # ----------------------------------------------------------

        val_prob, val_y = predict(
            model,
            val_loader
        )

        pooled_probs.append(
            val_prob
        )

        pooled_targets.append(
            val_y
        )

        for h in range(HORIZONS):

            threshold, _, _, _ = best_csi_threshold(
                val_y[:, h],
                val_prob[:, h]
            )

            met = calculate_metrics(
                val_y[:, h],
                val_prob[:, h],
                threshold
            )

            fold_rows.append(
                {
                    "feature_set": name,
                    "fold": fold,
                    "horizon": h + 1,
                    "train_samples": len(train_rows),
                    "validation_samples": len(val_rows),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "trainable_parameters": parameter_count,
                    **met,
                }
            )

        print(
            f"Fold {fold} complete | "
            f"best_epoch={best_epoch} | "
            f"best_val_loss={best_val_loss:.6f}"
        )

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # POOLED OOF VALIDATION
    # --------------------------------------------------------------

    pooled_prob = np.concatenate(
        pooled_probs,
        axis=0
    )

    pooled_y = np.concatenate(
        pooled_targets,
        axis=0
    )

    pooled_rows = []

    pooled_thresholds = []

    for h in range(HORIZONS):

        threshold, _, _, _ = best_csi_threshold(
            pooled_y[:, h],
            pooled_prob[:, h]
        )

        met = calculate_metrics(
            pooled_y[:, h],
            pooled_prob[:, h],
            threshold
        )

        pooled_thresholds.append(
            threshold
        )

        pooled_rows.append(
            {
                "feature_set": name,
                "fold": "POOLED_OOF",
                "horizon": h + 1,
                "train_samples": np.nan,
                "validation_samples": len(
                    pooled_y
                ),
                "best_epoch": np.nan,
                "best_val_loss": np.nan,
                "trainable_parameters": sum(
                    p.numel()
                    for p in build_tcn_bilstm(
                        expected_features
                    ).parameters()
                    if p.requires_grad
                ),
                **met,
            }
        )

    fold_df = pd.DataFrame(
        fold_rows
        + pooled_rows
    )

    fold_df.to_csv(
        feature_output / "validation_results.csv",
        index=False
    )

    np.save(
        feature_output / "pooled_oof_prob.npy",
        pooled_prob
    )

    np.save(
        feature_output / "pooled_oof_y.npy",
        pooled_y
    )

    with open(
        feature_output / "pooled_oof_thresholds.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                f"H{h + 1}": float(t)
                for h, t in enumerate(
                    pooled_thresholds
                )
            },
            f,
            indent=2
        )

    # Summary by horizon
    summary = (
        fold_df[
            fold_df["fold"] == "POOLED_OOF"
        ]
        .copy()
    )

    return (
        fold_df,
        summary,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 90)
    print(
        "STAGE D1 - FEATURE ENGINEERING COMPARISON"
    )
    print(
        "F1 (20) vs F2 (37) vs F3 (45)"
    )
    print(
        "Fixed architecture: TCN-BiLSTM"
    )
    print("=" * 90)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Feature root:",
        FEATURE_ROOT
    )

    print(
        "Stage C:",
        STAGEC_ROOT
    )

    print(
        "Seed:",
        SEED
    )

    print(
        "Epochs:",
        args.epochs
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY
    )

    if FINAL_CONFIG_PATH:
        print(
            "Final config:",
            FINAL_CONFIG_PATH
        )
    else:
        print(
            "Final config: NOT FOUND; explicit LR required"
        )

    # --------------------------------------------------------------
    # GLOBAL AUDIT
    # --------------------------------------------------------------

    if not DEV_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing development manifest: {DEV_MANIFEST}"
        )

    # Explicitly prohibit accidental external data use.
    external_markers = [
        "external_test_2025",
        "external_test",
        "F1_external",
    ]

    feature_root_text = str(
        FEATURE_ROOT
    ).lower()

    if any(
        marker.lower() in feature_root_text
        for marker in external_markers
    ):
        raise RuntimeError(
            "Feature root appears to point to external-test data. "
            "This comparison must use development data only."
        )

    all_fold_results = []
    all_summaries = []

    for feature_name, expected_count in FEATURE_SETS.items():

        x_path, y_path = require_feature_files(
            feature_name
        )

        fold_df, summary_df = run_feature_set(
            feature_name,
            x_path,
            y_path
        )

        all_fold_results.append(
            fold_df
        )

        all_summaries.append(
            summary_df
        )

    # --------------------------------------------------------------
    # COMBINED RESULTS
    # --------------------------------------------------------------

    combined_folds = pd.concat(
        all_fold_results,
        ignore_index=True
    )

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True
    )

    combined_folds.to_csv(
        OUTPUT_ROOT / "fold_results.csv",
        index=False
    )

    combined_summary.to_csv(
        OUTPUT_ROOT / "comparison_summary.csv",
        index=False
    )

    # --------------------------------------------------------------
    # FEATURE COMPARISON TABLE
    # --------------------------------------------------------------

    comparison_rows = []

    for feature_name in FEATURE_SETS:

        row = combined_summary[
            combined_summary["feature_set"]
            == feature_name
        ].copy()

        for _, r in row.iterrows():

            comparison_rows.append(
                {
                    "feature_set":
                        feature_name,

                    "feature_count":
                        FEATURE_SETS[feature_name],

                    "horizon":
                        int(r["horizon"]),

                    "AUPRC":
                        r["AUPRC"],

                    "ROC_AUC":
                        r["ROC_AUC"],

                    "CSI":
                        r["CSI"],

                    "POD":
                        r["POD"],

                    "Precision":
                        r["Precision"],

                    "FAR":
                        r["FAR"],

                    "F1":
                        r["F1"],

                    "F2":
                        r["F2"],

                    "threshold":
                        r["threshold"],
                }
            )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        OUTPUT_ROOT
        / "feature_set_comparison.csv",
        index=False
    )

    # --------------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------------

    audit = {

        "stage":
            "D1 feature engineering comparison",

        "feature_sets":
            {
                "F1": 20,
                "F2": 37,
                "F3": 45,
            },

        "architecture":
            "TCN-BiLSTM",

        "seed":
            SEED,

        "lookback_minutes":
            LOOKBACK,

        "forecast_horizon":
            HORIZONS,

        "development_manifest":
            str(DEV_MANIFEST),

        "feature_root":
            str(FEATURE_ROOT),

        "external_test_used":
            False,

        "external_test_read":
            False,

        "external_threshold_selection":
            False,

        "external_predictions_used":
            False,

        "same_fold_membership":
            True,

        "train_only_scaling":
            True,

        "learning_rate_scheduler":
            False,

        "optimizer":
            "AdamW",

        "gradient_clip":
            1.0,

        "threshold_for_validation":
            "maximum CSI; descriptive only",

        "final_external_thresholds":
            "NOT generated by this script",

        "status":
            "PASS",
    }

    with open(
        OUTPUT_ROOT / "audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=2
        )

    # --------------------------------------------------------------
    # PRINT
    # --------------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "FEATURE ENGINEERING COMPARISON FINISHED"
    )
    print("=" * 90)

    print(
        "\nPooled OOF results:"
    )

    print(
        combined_summary[
            [
                "feature_set",
                "horizon",
                "AUPRC",
                "ROC_AUC",
                "CSI",
                "POD",
                "Precision",
                "FAR",
                "F1",
                "F2",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        OUTPUT_ROOT / "comparison_summary.csv"
    )

    print(
        OUTPUT_ROOT / "feature_set_comparison.csv"
    )

    print(
        OUTPUT_ROOT / "fold_results.csv"
    )

    print(
        OUTPUT_ROOT / "audit.json"
    )

    print(
        "\nExternal Test: NOT USED"
    )

    print(
        "External thresholds: NOT GENERATED"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
