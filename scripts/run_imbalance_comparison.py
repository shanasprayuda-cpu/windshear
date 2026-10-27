"""
Stage D1 - Imbalance Comparison v3
==================================

Purpose
-------
Compare four loss/imbalance strategies for the fixed Stage D1
TCN-BiLSTM + F1 configuration:

1. BCE
2. Weighted_BCE
3. Focal
4. Weighted_Focal

IMPORTANT v3 corrections
------------------------
- Uses the authoritative Stage C anchor_idx -> compact Stage D1 row mapping
  through development_windows.csv.
- Never indexes X/y directly with global anchor_idx.
- Uses 5-fold expanding-window validation from fold_membership.csv.
- External test data are NOT loaded or used.
- Thresholds are selected ONLY after pooling all validation predictions
  across the 5 folds (OOF validation), separately for H+1 ... H+10.
- Threshold grid is 0.001 ... 0.999.
- Objective: maximum CSI; tie-break 1 = higher POD; tie-break 2 = lower FAR.
- Focal: alpha=0.6, gamma=2.0.
- Weighted BCE: positive-class weight = N_negative / N_positive,
  computed separately for each horizon using TRAINING rows of each fold.
- Weighted Focal: focal(alpha=0.6, gamma=2.0) with the same
  horizon-specific positive-class weight applied to the positive class.
- AdamW with fixed learning rate; no scheduler.
- Gradient clipping at 1.0.
- AMP FP16 on CUDA.
- Memory-mapped X/y with DataLoader pin_memory=True,
  persistent_workers=True and prefetch_factor=4.

Run from project root:
    python -m scripts.run_imbalance_comparison

Direct execution is also supported:
    python scripts/run_imbalance_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------
# Project import path
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tcn_bilstm import TCNBiLSTM


# ---------------------------------------------------------------------
# Fixed experiment configuration
# ---------------------------------------------------------------------
DEFAULT_SEED = 2026
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 0.00036954127204552557

INPUT_LENGTH = 60
INPUT_SIZE = 20
OUTPUT_SIZE = 10

# The model file supplied for this project uses tcn_channels=64.
TCN_CHANNELS = 64
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.25

FOCAL_ALPHA = 0.6
FOCAL_GAMMA = 2.0
GRAD_CLIP = 1.0

THRESHOLD_GRID = np.linspace(
    0.001, 0.999, 999, dtype=np.float64
)

STRATEGIES = [
    "BCE",
    "Weighted_BCE",
    "Focal",
    "Weighted_Focal",
]

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT.parent

X_PATH = DATA_ROOT / "stageD1_data" / "F1" / "X.npy"
Y_PATH = DATA_ROOT / "stageD1_data" / "F1" / "y.npy"

STAGE_C_ROOT = DATA_ROOT / "stageC"
DEV_MANIFEST_PATH = STAGE_C_ROOT / "manifest" / "development_windows.csv"
FOLD_MEMBERSHIP_PATH = STAGE_C_ROOT / "manifest" / "fold_membership.csv"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "imbalance_v3"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve reproducibility. Benchmarking is
    # disabled because exact reproducibility is more important here.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class MemmapWindowDataset(Dataset):
    def __init__(
        self,
        X_memmap: np.ndarray,
        y_memmap: np.ndarray,
        rows: np.ndarray,
    ) -> None:
        self.X = X_memmap
        self.y = y_memmap
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = int(self.rows[idx])

        x = np.asarray(self.X[row], dtype=np.float32)
        y = np.asarray(self.y[row], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    return DataLoader(**kwargs)


# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Binary focal loss.

    Unweighted:
        alpha_t = alpha for positive, 1-alpha for negative.

    Weighted focal:
        additionally multiplies the positive class by
        pos_weight = N_negative / N_positive.

    This makes Weighted_Focal a direct extension of Focal rather than
    replacing alpha with a prevalence-derived value.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    focal_factor = torch.pow(
        torch.clamp(1.0 - p_t, min=1e-8),
        gamma,
    )

    loss = alpha_t * focal_factor * bce

    if pos_weight is not None:
        # Positive samples receive pos_weight; negative samples receive 1.
        class_weight = 1.0 + targets * (pos_weight - 1.0)
        loss = loss * class_weight

    return loss.mean()


def compute_pos_weight(
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute horizon-specific positive-class weights.

    pos_weight[h] = N_negative[h] / N_positive[h]
    """
    y_train = np.asarray(y_train)

    positives = y_train.sum(axis=0).astype(np.float64)
    total = np.full(positives.shape, y_train.shape[0], dtype=np.float64)
    negatives = total - positives

    pos_weight = np.ones_like(positives, dtype=np.float64)

    valid = positives > 0
    pos_weight[valid] = negatives[valid] / positives[valid]

    return positives, negatives, pos_weight


# ---------------------------------------------------------------------
# Anchor mapping
# ---------------------------------------------------------------------
def build_anchor_mapper(
    development_manifest: pd.DataFrame,
    n_x: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an exact global anchor_idx -> compact X/y row mapping.

    development_windows.csv row position is the compact Stage D1 row.
    anchor_idx is a global timeline index and MUST NOT be used directly
    as X/y indexing.
    """
    if "anchor_idx" not in development_manifest.columns:
        raise ValueError(
            "development_windows.csv must contain 'anchor_idx'."
        )

    if len(development_manifest) != n_x:
        raise ValueError(
            "development_windows.csv row count does not match X.npy/y.npy: "
            f"manifest={len(development_manifest)}, X={n_x}"
        )

    anchors = development_manifest["anchor_idx"].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    if np.any(pd.isna(development_manifest["anchor_idx"])):
        raise ValueError("development_windows.csv contains NaN anchor_idx.")

    if len(np.unique(anchors)) != len(anchors):
        raise ValueError(
            "development_windows.csv contains duplicate anchor_idx values."
        )

    # Robust mapping even if manifest rows are not sorted.
    order = np.argsort(anchors)
    sorted_anchors = anchors[order]

    return sorted_anchors, order


def anchors_to_compact_rows(
    anchor_values: np.ndarray,
    sorted_anchors: np.ndarray,
    row_order: np.ndarray,
) -> np.ndarray:
    """
    Map global anchor_idx values to compact Stage D1 rows.
    """
    anchors = np.asarray(anchor_values, dtype=np.int64)

    positions = np.searchsorted(
        sorted_anchors,
        anchors,
        side="left",
    )

    valid = (
        (positions >= 0)
        & (positions < len(sorted_anchors))
    )

    if not np.all(valid):
        bad = anchors[~valid][:10]
        raise ValueError(
            f"Some anchor_idx values are outside the development manifest. "
            f"Examples: {bad.tolist()}"
        )

    matched = sorted_anchors[positions] == anchors
    if not np.all(matched):
        bad = anchors[~matched][:10]
        raise ValueError(
            f"Some anchor_idx values were not found in development_windows.csv. "
            f"Examples: {bad.tolist()}"
        )

    compact_rows = row_order[positions].astype(np.int64, copy=False)

    if len(compact_rows) > 0:
        if compact_rows.min() < 0 or compact_rows.max() >= len(row_order):
            raise IndexError(
                "Mapped compact row is outside X/y bounds."
            )

    return compact_rows


# ---------------------------------------------------------------------
# Fold loading
# ---------------------------------------------------------------------
def load_fold_membership(
    path: Path,
    sorted_anchors: np.ndarray,
    row_order: np.ndarray,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Read Stage C fold membership and map anchor_idx to compact rows.

    fold_membership.csv intentionally contains repeated rows because each
    sample can participate in train/validation roles across expanding folds.
    """
    print("=" * 80)
    print("LOAD STAGE C FOLD MEMBERSHIP")
    print("=" * 80)

    df = pd.read_csv(path)

    required = {
        "sample_id",
        "fold",
        "split_role",
        "anchor_idx",
        "anchor_time",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"fold_membership.csv missing columns: {sorted(missing)}"
        )

    print(df.head())
    print("Rows:", len(df))

    df["fold"] = df["fold"].astype(np.int16)
    df["anchor_idx"] = df["anchor_idx"].astype(np.int64)

    fold_data: Dict[int, Dict[str, np.ndarray]] = {}

    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]

        train_anchor = sub.loc[
            sub["split_role"].eq("train"),
            "anchor_idx",
        ].to_numpy(dtype=np.int64)

        val_anchor = sub.loc[
            sub["split_role"].eq("val"),
            "anchor_idx",
        ].to_numpy(dtype=np.int64)

        train_rows = anchors_to_compact_rows(
            train_anchor,
            sorted_anchors,
            row_order,
        )

        val_rows = anchors_to_compact_rows(
            val_anchor,
            sorted_anchors,
            row_order,
        )

        # Explicit bounds check prevents the previous v2 failure:
        # index 847513 is invalid for an array of size 847513.
        if len(train_rows):
            assert train_rows.min() >= 0
            assert train_rows.max() < len(row_order)

        if len(val_rows):
            assert val_rows.min() >= 0
            assert val_rows.max() < len(row_order)

        fold_data[int(fold)] = {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_anchor": train_anchor,
            "val_anchor": val_anchor,
        }

        print(
            f"Fold {int(fold)}: "
            f"train={len(train_rows):7d} "
            f"val={len(val_rows):7d}"
        )

    return fold_data


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    pred = y_prob >= threshold

    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))

    pod = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    f1_den = 2.0 * precision + pod
    f1 = (
        2.0 * precision * pod / f1_den
        if f1_den > 0
        else 0.0
    )

    f2_den = 4.0 * precision + pod
    f2 = (
        5.0 * precision * pod / f2_den
        if f2_den > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "CSI": float(csi),
        "POD": float(pod),
        "FAR": float(far),
        "Precision": float(precision),
        "F1": float(f1),
        "F2": float(f2),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def pooled_threshold_search(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_grid: np.ndarray = THRESHOLD_GRID,
) -> Dict[str, float]:
    """
    Select threshold on pooled OOF validation only.

    Efficient implementation:
    sort probabilities once and evaluate each threshold through cumulative
    TP/FP counts. This avoids constructing a huge [999 x N] boolean matrix.
    """
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("pooled_threshold_search expects 1-D arrays.")

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob lengths differ.")

    order = np.argsort(-y_prob, kind="mergesort")
    p_sorted = y_prob[order]
    y_sorted = y_true[order]

    cumulative_tp = np.cumsum(y_sorted, dtype=np.int64)
    counts = np.searchsorted(
        -p_sorted,
        -threshold_grid,
        side="right",
    )

    total_positive = int(y_true.sum())

    best = None

    for t, n_pred in zip(threshold_grid, counts):
        n_pred = int(n_pred)

        if n_pred > 0:
            tp = int(cumulative_tp[n_pred - 1])
        else:
            tp = 0

        fp = n_pred - tp
        fn = total_positive - tp

        csi_den = tp + fp + fn
        csi = tp / csi_den if csi_den else 0.0

        pod_den = tp + fn
        pod = tp / pod_den if pod_den else 0.0

        far_den = tp + fp
        far = fp / far_den if far_den else 0.0

        candidate = {
            "threshold": float(t),
            "CSI": float(csi),
            "POD": float(pod),
            "FAR": float(far),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": int(len(y_true) - tp - fp - fn),
        }

        if best is None:
            best = candidate
            continue

        # Primary objective: maximum CSI.
        if candidate["CSI"] > best["CSI"] + 1e-12:
            best = candidate
            continue

        # Tie-break 1: higher POD.
        if abs(candidate["CSI"] - best["CSI"]) <= 1e-12:
            if candidate["POD"] > best["POD"] + 1e-12:
                best = candidate
                continue

            # Tie-break 2: lower FAR.
            if abs(candidate["POD"] - best["POD"]) <= 1e-12:
                if candidate["FAR"] < best["FAR"] - 1e-12:
                    best = candidate

    if best is None:
        raise RuntimeError("Threshold search produced no candidate.")

    # Add complete metrics at the selected threshold.
    final = binary_metrics(
        y_true,
        y_prob,
        best["threshold"],
    )

    return final


def safe_average_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    if np.sum(y_true) == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def safe_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


# ---------------------------------------------------------------------
# Model / loss helpers
# ---------------------------------------------------------------------
def create_model() -> nn.Module:
    return TCNBiLSTM(
        input_size=INPUT_SIZE,
        tcn_channels=TCN_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_size=OUTPUT_SIZE,
    )


def compute_loss(
    strategy: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    bce_loss: nn.Module | None,
) -> torch.Tensor:
    if strategy == "BCE":
        assert bce_loss is not None
        return bce_loss(logits, targets)

    if strategy == "Weighted_BCE":
        assert bce_loss is not None
        return bce_loss(logits, targets)

    if strategy == "Focal":
        return focal_loss(
            logits,
            targets,
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            pos_weight=None,
        )

    if strategy == "Weighted_Focal":
        return focal_loss(
            logits,
            targets,
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            pos_weight=pos_weight,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------
def train_one_fold(
    strategy: str,
    fold: int,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    output_root: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    print("\n")
    print("-" * 80)
    print(f"Strategy : {strategy}")
    print(f"Fold     : {fold}")
    print("-" * 80)

    # Same seed for each strategy within a fold gives comparable
    # initialization and minibatch ordering.
    set_seed(seed)

    y_train = np.asarray(y[train_rows], dtype=np.float32)

    positives, negatives, pos_weight_np = compute_pos_weight(y_train)

    print("Positive counts:")
    print(np.array2string(positives, precision=0))
    print("Negative counts:")
    print(np.array2string(negatives, precision=0))
    print("Positive weights:")
    print(np.array2string(pos_weight_np, precision=4))

    model = create_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    pos_weight = torch.tensor(
        pos_weight_np,
        dtype=torch.float32,
        device=device,
    )

    if strategy == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif strategy == "Weighted_BCE":
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
        )
    else:
        criterion = None

    train_dataset = MemmapWindowDataset(
        X,
        y,
        train_rows,
    )
    val_dataset = MemmapWindowDataset(
        X,
        y,
        val_rows,
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )

    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    amp_enabled = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    history_rows: List[Dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(
                device,
                non_blocking=amp_enabled,
            )
            yb = yb.to(
                device,
                non_blocking=amp_enabled,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(xb)

                loss = compute_loss(
                    strategy,
                    logits,
                    yb,
                    pos_weight,
                    criterion,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(optimizer)
            scaler.update()

            batch_n = xb.shape[0]
            running_loss += float(loss.detach().item()) * batch_n
            n_seen += batch_n

        epoch_loss = (
            running_loss / n_seen
            if n_seen
            else float("nan")
        )

        history_rows.append(
            {
                "strategy": strategy,
                "fold": fold,
                "epoch": epoch,
                "train_loss": epoch_loss,
            }
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:2d}/{epochs} "
                f"Loss={epoch_loss:.6f}"
            )

    # Validation inference only. No validation labels are used to update
    # weights or select an epoch.
    model.eval()

    prob_chunks: List[np.ndarray] = []
    true_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(
                device,
                non_blocking=amp_enabled,
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(xb)
                probs = torch.sigmoid(logits)

            prob_chunks.append(
                probs.float().cpu().numpy()
            )
            true_chunks.append(
                yb.numpy().astype(np.uint8, copy=False)
            )

    y_prob = np.concatenate(prob_chunks, axis=0)
    y_true = np.concatenate(true_chunks, axis=0)

    # Save compact fold history.
    history_dir = output_root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    history_path = (
        history_dir
        / f"{strategy}_fold_{fold}_history.csv"
    )

    pd.DataFrame(history_rows).to_csv(
        history_path,
        index=False,
    )

    # Save final fold model for audit/reproducibility.
    checkpoint_dir = output_root / "checkpoints" / strategy
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir
        / f"fold_{fold}.pt"
    )

    torch.save(
        {
            "strategy": strategy,
            "fold": fold,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "architecture": "TCN-BiLSTM",
            "feature": "F1",
            "input_length": INPUT_LENGTH,
            "input_size": INPUT_SIZE,
            "output_size": OUTPUT_SIZE,
            "tcn_channels": TCN_CHANNELS,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "focal_alpha": FOCAL_ALPHA,
            "focal_gamma": FOCAL_GAMMA,
            "grad_clip": GRAD_CLIP,
            "pos_weight": pos_weight_np,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    # Release model before moving to the next fold.
    del model
    del optimizer
    del train_loader
    del val_loader

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fold_info = {
        "strategy": strategy,
        "fold": fold,
        "train_samples": int(len(train_rows)),
        "validation_samples": int(len(val_rows)),
        "positive_counts": positives.tolist(),
        "negative_counts": negatives.tolist(),
        "pos_weight": pos_weight_np.tolist(),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }

    return y_true, y_prob, fold_info


# ---------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------
def evaluate_strategy(
    strategy: str,
    fold_data: Dict[int, Dict[str, np.ndarray]],
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    output_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Train all 5 folds, pool validation predictions, then select thresholds
    on pooled OOF validation.
    """
    fold_order = sorted(fold_data)

    oof_true_chunks: List[np.ndarray] = []
    oof_prob_chunks: List[np.ndarray] = []
    oof_fold_chunks: List[np.ndarray] = []
    oof_row_chunks: List[np.ndarray] = []

    fold_audit: List[Dict[str, object]] = []

    for fold in fold_order:
        train_rows = fold_data[fold]["train_rows"]
        val_rows = fold_data[fold]["val_rows"]

        y_true, y_prob, info = train_one_fold(
            strategy=strategy,
            fold=fold,
            train_rows=train_rows,
            val_rows=val_rows,
            X=X,
            y=y,
            device=device,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            output_root=output_root,
        )

        oof_true_chunks.append(y_true)
        oof_prob_chunks.append(y_prob)
        oof_fold_chunks.append(
            np.full(
                len(val_rows),
                fold,
                dtype=np.int16,
            )
        )
        oof_row_chunks.append(
            val_rows.astype(np.int64, copy=False)
        )

        fold_audit.append(info)

    oof_true = np.concatenate(
        oof_true_chunks,
        axis=0,
    )
    oof_prob = np.concatenate(
        oof_prob_chunks,
        axis=0,
    )
    oof_fold = np.concatenate(
        oof_fold_chunks,
        axis=0,
    )
    oof_rows = np.concatenate(
        oof_row_chunks,
        axis=0,
    )

    # -------------------------------------------------------------
    # Save compact OOF artifact.
    # -------------------------------------------------------------
    oof_dir = output_root / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    oof_npz_path = (
        oof_dir
        / f"oof_{strategy}.npz"
    )

    np.savez_compressed(
        oof_npz_path,
        y_true=oof_true.astype(np.uint8),
        y_prob=oof_prob.astype(np.float32),
        fold=oof_fold.astype(np.int16),
        compact_row=oof_rows.astype(np.int64),
    )

    # -------------------------------------------------------------
    # Pooled OOF threshold + metrics for each horizon.
    # -------------------------------------------------------------
    summary_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []

    for h in range(OUTPUT_SIZE):
        yt = oof_true[:, h]
        yp = oof_prob[:, h]

        selected = pooled_threshold_search(
            yt,
            yp,
            THRESHOLD_GRID,
        )

        auprc = safe_average_precision(yt, yp)
        roc_auc = safe_roc_auc(yt, yp)

        row = {
            "strategy": strategy,
            "horizon": f"H{h + 1}",
            "threshold": selected["threshold"],
            "CSI": selected["CSI"],
            "POD": selected["POD"],
            "FAR": selected["FAR"],
            "Precision": selected["Precision"],
            "F1": selected["F1"],
            "F2": selected["F2"],
            "AUPRC": auprc,
            "ROC_AUC": roc_auc,
            "TP": selected["TP"],
            "FP": selected["FP"],
            "FN": selected["FN"],
            "TN": selected["TN"],
            "validation_samples": int(len(yt)),
            "positive_events": int(np.sum(yt)),
            "threshold_method": "pooled_OOF_validation",
            "threshold_objective": "maximum_CSI",
            "tie_breaker_1": "higher_POD",
            "tie_breaker_2": "lower_FAR",
        }

        summary_rows.append(row)

        threshold_rows.append(
            {
                "strategy": strategy,
                "horizon": f"H{h + 1}",
                "threshold": selected["threshold"],
                "CSI": selected["CSI"],
                "POD": selected["POD"],
                "FAR": selected["FAR"],
                "Precision": selected["Precision"],
                "F1": selected["F1"],
                "F2": selected["F2"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------
    # Per-fold metrics using the POOLED threshold.
    # This is descriptive only; it does not tune thresholds per fold.
    # -------------------------------------------------------------
    fold_rows: List[Dict[str, object]] = []

    threshold_map = {
        row["horizon"]: row["threshold"]
        for row in threshold_rows
    }

    for fold in fold_order:
        mask = oof_fold == fold

        for h in range(OUTPUT_SIZE):
            horizon = f"H{h + 1}"
            yt = oof_true[mask, h]
            yp = oof_prob[mask, h]
            threshold = float(threshold_map[horizon])

            metrics = binary_metrics(
                yt,
                yp,
                threshold,
            )

            fold_rows.append(
                {
                    "strategy": strategy,
                    "fold": fold,
                    "horizon": horizon,
                    "threshold_from_pooled_OOF": threshold,
                    "AUPRC": safe_average_precision(yt, yp),
                    "ROC_AUC": safe_roc_auc(yt, yp),
                    **{
                        k: v
                        for k, v in metrics.items()
                        if k != "threshold"
                    },
                    "validation_samples": int(len(yt)),
                    "positive_events": int(np.sum(yt)),
                }
            )

    fold_df = pd.DataFrame(fold_rows)

    # -------------------------------------------------------------
    # Threshold JSON artifact.
    # -------------------------------------------------------------
    threshold_dir = output_root / "thresholds"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    threshold_json = {
        "strategy": strategy,
        "method": "pooled_OOF_validation",
        "objective": "maximum_CSI",
        "tie_breaker_1": "higher_POD",
        "tie_breaker_2": "lower_FAR",
        "threshold_grid": "0.001 to 0.999 step 0.001",
        "external_test_used": False,
        "horizons": {
            row["horizon"]: {
                k: v
                for k, v in row.items()
                if k not in {"strategy", "horizon"}
            }
            for row in threshold_rows
        },
    }

    with open(
        threshold_dir / f"{strategy}_thresholds.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            threshold_json,
            f,
            indent=2,
        )

    audit = {
        "strategy": strategy,
        "oof_samples": int(len(oof_true)),
        "oof_path": str(oof_npz_path),
        "folds": fold_audit,
        "thresholds": threshold_json,
    }

    return summary_df, fold_df, audit


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage D1 imbalance comparison v3"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers. Default=4.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=STRATEGIES,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_ROOT),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("STAGE D1 IMBALANCE COMPARISON v3")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Project root:", PROJECT_ROOT)
    print("Data root:", DATA_ROOT)
    print("Output:", output_root)

    # -------------------------------------------------------------
    # Configuration audit
    # -------------------------------------------------------------
    config = {
        "version": "v3",
        "architecture": "TCN-BiLSTM",
        "feature": "F1",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "input_length": INPUT_LENGTH,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "tcn_channels": TCN_CHANNELS,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "scheduler": None,
        "grad_clip": GRAD_CLIP,
        "amp": bool(device.type == "cuda"),
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "strategies": args.strategies,
        "threshold": {
            "method": "pooled_OOF_validation",
            "grid": "0.001 to 0.999 step 0.001",
            "objective": "maximum_CSI",
            "tie_breaker_1": "higher_POD",
            "tie_breaker_2": "lower_FAR",
        },
        "external_test_used": False,
        "x_path": str(X_PATH),
        "y_path": str(Y_PATH),
        "development_manifest_path": str(DEV_MANIFEST_PATH),
        "fold_membership_path": str(FOLD_MEMBERSHIP_PATH),
    }

    with open(
        output_root / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Load arrays as memmaps.
    # -------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    if not X_PATH.exists():
        raise FileNotFoundError(f"X.npy not found: {X_PATH}")

    if not Y_PATH.exists():
        raise FileNotFoundError(f"y.npy not found: {Y_PATH}")

    if not DEV_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"development_windows.csv not found: {DEV_MANIFEST_PATH}"
        )

    if not FOLD_MEMBERSHIP_PATH.exists():
        raise FileNotFoundError(
            f"fold_membership.csv not found: {FOLD_MEMBERSHIP_PATH}"
        )

    X = np.load(
        X_PATH,
        mmap_mode="r",
    )
    y = np.load(
        Y_PATH,
        mmap_mode="r",
    )

    print("X shape :", X.shape)
    print("y shape :", y.shape)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to be 3-D [N,60,20], got {X.shape}"
        )

    if y.ndim != 2:
        raise ValueError(
            f"Expected y to be 2-D [N,10], got {y.shape}"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X/y row counts differ."
        )

    if X.shape[1] != INPUT_LENGTH:
        raise ValueError(
            f"Expected input length {INPUT_LENGTH}, got {X.shape[1]}"
        )

    if X.shape[2] != INPUT_SIZE:
        raise ValueError(
            f"Expected input size {INPUT_SIZE}, got {X.shape[2]}"
        )

    if y.shape[1] != OUTPUT_SIZE:
        raise ValueError(
            f"Expected output size {OUTPUT_SIZE}, got {y.shape[1]}"
        )

    # -------------------------------------------------------------
    # Load compact development manifest.
    # -------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("LOAD DEVELOPMENT MANIFEST")
    print("=" * 80)

    dev_manifest = pd.read_csv(
        DEV_MANIFEST_PATH,
        usecols=["sample_id", "anchor_idx", "anchor_time"],
    )

    print("Rows:", len(dev_manifest))
    print(dev_manifest.head())

    sorted_anchors, row_order = build_anchor_mapper(
        dev_manifest,
        n_x=X.shape[0],
    )

    print(
        "Anchor mapping: global anchor_idx -> compact X/y row"
    )
    print(
        "Compact rows:",
        len(row_order),
    )

    # -------------------------------------------------------------
    # Load fold membership and resolve compact rows.
    # -------------------------------------------------------------
    fold_data = load_fold_membership(
        FOLD_MEMBERSHIP_PATH,
        sorted_anchors,
        row_order,
    )

    if len(fold_data) != 5:
        print(
            f"WARNING: expected 5 folds, found {len(fold_data)}."
        )

    # -------------------------------------------------------------
    # Run strategies.
    # -------------------------------------------------------------
    all_summary: List[pd.DataFrame] = []
    all_fold_results: List[pd.DataFrame] = []
    strategy_audits: List[Dict[str, object]] = []

    overall_start = time.time()

    for strategy in args.strategies:
        start = time.time()

        summary_df, fold_df, audit = evaluate_strategy(
            strategy=strategy,
            fold_data=fold_data,
            X=X,
            y=y,
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            output_root=output_root,
        )

        all_summary.append(summary_df)
        all_fold_results.append(fold_df)
        strategy_audits.append(audit)

        elapsed = time.time() - start

        print("\n")
        print("-" * 80)
        print(f"COMPLETED STRATEGY: {strategy}")
        print(f"Elapsed: {elapsed / 60.0:.2f} minutes")
        print("-" * 80)

        print(
            summary_df[
                [
                    "horizon",
                    "threshold",
                    "CSI",
                    "POD",
                    "FAR",
                    "Precision",
                    "F1",
                    "F2",
                    "AUPRC",
                ]
            ].to_string(index=False)
        )

    # -------------------------------------------------------------
    # Combined outputs.
    # -------------------------------------------------------------
    summary = pd.concat(
        all_summary,
        ignore_index=True,
    )

    fold_results = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    summary.to_csv(
        output_root / "summary.csv",
        index=False,
    )

    fold_results.to_csv(
        output_root / "fold_results.csv",
        index=False,
    )

    # Mean horizon metrics by strategy.
    mean_rows = []

    metric_cols = [
        "CSI",
        "POD",
        "FAR",
        "Precision",
        "F1",
        "F2",
        "AUPRC",
        "ROC_AUC",
    ]

    for strategy in args.strategies:
        sub = summary[
            summary["strategy"] == strategy
        ]

        row = {
            "strategy": strategy,
            "n_horizons": len(sub),
        }

        for metric in metric_cols:
            row[f"mean_{metric}"] = float(
                sub[metric].mean()
            )

        mean_rows.append(row)

    mean_summary = pd.DataFrame(mean_rows)

    mean_summary.to_csv(
        output_root / "strategy_mean_summary.csv",
        index=False,
    )

    # Audit artifact.
    audit = {
        "version": "v3",
        "architecture": "TCN-BiLSTM",
        "feature": "F1",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "strategies": args.strategies,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "weighted_loss_definition": (
            "positive_class_weight = N_negative / N_positive "
            "computed from TRAINING windows separately for each horizon "
            "and fold"
        ),
        "weighted_focal_definition": (
            "Focal(alpha=0.6,gamma=2.0) multiplied by positive-class "
            "weight for positive targets; negative targets weight=1"
        ),
        "threshold_method": "pooled_OOF_validation",
        "threshold_grid": "0.001 to 0.999 step 0.001",
        "threshold_objective": "maximum_CSI",
        "threshold_tie_breaker_1": "higher_POD",
        "threshold_tie_breaker_2": "lower_FAR",
        "external_test_used": False,
        "x_shape": list(X.shape),
        "y_shape": list(y.shape),
        "development_manifest_rows": int(len(dev_manifest)),
        "fold_membership_rows": int(
            sum(
                len(v["train_rows"]) + len(v["val_rows"])
                for v in fold_data.values()
            )
        ),
        "strategies": strategy_audits,
        "total_elapsed_minutes": (
            (time.time() - overall_start) / 60.0
        ),
    }

    with open(
        output_root / "audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
        )

    print("\n")
    print("=" * 80)
    print("STAGE D1 IMBALANCE COMPARISON v3 COMPLETE")
    print("=" * 80)

    print("\nMean metrics:")
    print(
        mean_summary.to_string(index=False)
    )

    print("\nOutput directory:")
    print(output_root)

    print("\nFiles:")
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            try:
                rel = path.relative_to(output_root)
            except ValueError:
                rel = path
            print(" -", rel)


if __name__ == "__main__":
    main()
