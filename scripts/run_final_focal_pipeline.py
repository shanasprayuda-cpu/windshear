"""
Stage D1 - Final F1 + Focal pipeline
====================================

Purpose
-------
Produce a methodologically consistent final TCN-BiLSTM model using:
    Feature set : F1 (20 predictors)
    Loss        : unweighted Focal Loss, alpha=0.6, gamma=2.0
    Architecture: TCN-BiLSTM
    Trial       : Optuna Trial 26 hyperparameters

IMPORTANT CONSISTENCY RULE
---------------------------
The earlier imbalance-v3 comparison used a different optimizer configuration
(batch_size=256 and AdamW default weight_decay). Those OOF thresholds MUST NOT
be mixed with the Trial-26 final model because thresholds are model/calibration
dependent.

Therefore this pipeline deliberately performs:

  1) 5-fold expanding OOF Focal retraining using the exact Trial-26
     hyperparameters;
  2) pooled-OOF threshold selection (maximum CSI, POD/FAR tie-breaks);
  3) full-development retraining on all 847,513 windows using the same
     hyperparameters and Focal loss;
  4) independent external evaluation on Jan-Mar 2025 using ONLY the locked
     pooled-OOF thresholds.

No external data are loaded during OOF training, threshold selection, or
full-development training.

Training rules locked for Stage D1
-----------------------------------
- Seed = 2026
- Lookback = 60 min
- Output horizons = H+1 ... H+10
- F1 = 20 predictors
- AdamW
- fixed learning rate; NO scheduler
- Trial-26 learning rate = 0.00036954127204552557
- Trial-26 batch size = 128
- Trial-26 weight decay = 4.98413619067967e-05
- hidden_size = 32
- num_layers = 1
- dropout = 0.25
- max epochs = 50
- gradient clipping = 1.0
- AMP FP16 on CUDA
- memory-mapped X/y
- DataLoader pin_memory=True, persistent_workers=True,
  prefetch_factor=4
- Focal alpha = 0.6
- Focal gamma = 2.0
- threshold grid = 0.001 ... 0.999
- threshold objective = maximum CSI
- tie-break 1 = higher POD
- tie-break 2 = lower FAR

Run from the StageD1 project root:
    python scripts/run_final_focal_pipeline.py --stage all

Or run individual stages:
    python scripts/run_final_focal_pipeline.py --stage oof
    python scripts/run_final_focal_pipeline.py --stage full
    python scripts/run_final_focal_pipeline.py --stage external
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT.parent

try:
    from models.tcn_bilstm import TCNBiLSTM
except Exception as exc:
    raise ImportError(
        "Could not import models.tcn_bilstm.TCNBiLSTM. "
        "Run this script from the StageD1 project tree."
    ) from exc

# -----------------------------------------------------------------------------
# Locked configuration
# -----------------------------------------------------------------------------
SEED = 2026
INPUT_LENGTH = 60
INPUT_SIZE = 20
OUTPUT_SIZE = 10
TCN_CHANNELS = 64  # model implementation uses the fixed 64-channel TCN blocks
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.25
EPOCHS = 50
BATCH_SIZE = 128
NUM_WORKERS = 4
LR = 0.00036954127204552557
WEIGHT_DECAY = 4.98413619067967e-05
GRAD_CLIP = 1.0
FOCAL_ALPHA = 0.6
FOCAL_GAMMA = 2.0
TRIAL = 26
THRESHOLD_GRID = np.linspace(0.001, 0.999, 999, dtype=np.float64)

X_PATH = DATA_ROOT / "stageD1_data" / "F1" / "X.npy"
Y_PATH = DATA_ROOT / "stageD1_data" / "F1" / "y.npy"
STAGEC_ROOT = DATA_ROOT / "stageC"
DEV_MANIFEST_PATH = STAGEC_ROOT / "manifest" / "development_windows.csv"
FOLD_MEMBERSHIP_PATH = STAGEC_ROOT / "manifest" / "fold_membership.csv"
EXTERNAL_DATA_PATH = DATA_ROOT / "stageD1_data" / "F1_external"
EXTERNAL_MANIFEST_PATH = STAGEC_ROOT / "manifest" / "external_test_windows.csv"

RESULT_ROOT = PROJECT_ROOT / "results"
OOF_ROOT = RESULT_ROOT / "final_focal_oof_trial26"
FULL_ROOT = RESULT_ROOT / "final_full" / "FINAL_TCN_BiLSTM_TRIAL26_FULL_FOCAL"
EXTERNAL_ROOT = RESULT_ROOT / "external_test" / "FINAL_TCN_BiLSTM_FOCAL_EXTERNAL_JAN_MAR_2025"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Dataset / loader
# -----------------------------------------------------------------------------
class MemmapWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, rows: np.ndarray):
        self.X = X
        self.y = y
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = int(self.rows[idx])
        x = np.asarray(self.X[row], dtype=np.float32)
        y = np.asarray(self.y[row], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

class ExternalDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))

def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)

# -----------------------------------------------------------------------------
# Model / loss
# -----------------------------------------------------------------------------
def build_model() -> nn.Module:
    return TCNBiLSTM(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

def focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = FOCAL_ALPHA * targets + (1.0 - FOCAL_ALPHA) * (1.0 - targets)
    focal_factor = torch.pow(torch.clamp(1.0 - p_t, min=1e-8), FOCAL_GAMMA)
    return (alpha_t * focal_factor * bce).mean()

# -----------------------------------------------------------------------------
# Stage C mapping
# -----------------------------------------------------------------------------
def load_anchor_mapping() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DEV_MANIFEST_PATH, usecols=["anchor_idx"])
    anchors = df["anchor_idx"].to_numpy(dtype=np.int64)
    if len(np.unique(anchors)) != len(anchors):
        raise RuntimeError("development_windows.csv contains duplicate anchor_idx values.")
    order = np.argsort(anchors)
    return anchors[order], order

def anchors_to_rows(anchor_values: np.ndarray, sorted_anchors: np.ndarray, row_order: np.ndarray) -> np.ndarray:
    anchors = np.asarray(anchor_values, dtype=np.int64)
    pos = np.searchsorted(sorted_anchors, anchors, side="left")
    valid = (pos >= 0) & (pos < len(sorted_anchors))
    if not np.all(valid):
        raise RuntimeError("Some fold anchor_idx values are outside development_windows.csv.")
    matched = sorted_anchors[pos] == anchors
    if not np.all(matched):
        raise RuntimeError("Some fold anchor_idx values were not found in development_windows.csv.")
    return row_order[pos].astype(np.int64)

def load_folds(sorted_anchors: np.ndarray, row_order: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
    df = pd.read_csv(FOLD_MEMBERSHIP_PATH)
    required = {"fold", "split_role", "anchor_idx"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"fold_membership.csv missing columns: {sorted(missing)}")
    out = {}
    for fold in sorted(df["fold"].astype(int).unique()):
        sub = df[df["fold"].astype(int) == fold]
        train_anchor = sub.loc[sub["split_role"].eq("train"), "anchor_idx"].to_numpy(dtype=np.int64)
        val_anchor = sub.loc[sub["split_role"].eq("val"), "anchor_idx"].to_numpy(dtype=np.int64)
        train_rows = anchors_to_rows(train_anchor, sorted_anchors, row_order)
        val_rows = anchors_to_rows(val_anchor, sorted_anchors, row_order)
        out[int(fold)] = {"train_rows": train_rows, "val_rows": val_rows}
    if len(out) != 5:
        raise RuntimeError(f"Expected 5 folds, found {len(out)}.")
    return out

# -----------------------------------------------------------------------------
# Threshold selection
# -----------------------------------------------------------------------------
def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = y_prob >= threshold
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    pod = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    far = fp / (tp + fp) if tp + fp else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * precision * pod / (precision + pod) if precision + pod else 0.0
    f2 = 5 * precision * pod / (4 * precision + pod) if 4 * precision + pod else 0.0
    return {"threshold": float(threshold), "CSI": float(csi), "POD": float(pod), "FAR": float(far),
            "Precision": float(precision), "F1": float(f1), "F2": float(f2),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}

def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    best = None
    for t in THRESHOLD_GRID:
        m = threshold_metrics(y_true, y_prob, float(t))
        if best is None:
            best = m
            continue
        if m["CSI"] > best["CSI"] + 1e-12:
            best = m
        elif abs(m["CSI"] - best["CSI"]) <= 1e-12:
            if m["POD"] > best["POD"] + 1e-12:
                best = m
            elif abs(m["POD"] - best["POD"]) <= 1e-12 and m["FAR"] < best["FAR"] - 1e-12:
                best = m
    return best

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def predict_loader(model, loader, device: torch.device, with_targets: bool = True):
    amp = device.type == "cuda"
    probs = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if with_targets:
                xb, yb = batch
            else:
                xb = batch
                yb = None
            xb = xb.to(device, non_blocking=amp)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(xb)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            if yb is not None:
                targets.append(yb.numpy().astype(np.uint8, copy=False))
    p = np.concatenate(probs, axis=0)
    y = np.concatenate(targets, axis=0) if targets else None
    return y, p

# -----------------------------------------------------------------------------
# Training one fold / full
# -----------------------------------------------------------------------------
def train_model(X, y, rows: np.ndarray, device: torch.device, out_dir: Path, fold: int | None = None):
    set_seed(SEED)
    ds = MemmapWindowDataset(X, y, rows)
    loader = make_loader(ds, BATCH_SIZE, True, SEED)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(xb)
                loss = focal_loss(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            n = xb.shape[0]
            running += float(loss.detach().item()) * n
            seen += n
        epoch_loss = running / seen
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            label = f"Fold {fold}" if fold is not None else "FULL"
            print(f"{label} | Epoch {epoch:02d}/{EPOCHS} | loss={epoch_loss:.8f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": "D1_FINAL_FOCAL" if fold is None else "D1_OOF_FOCAL",
        "architecture": "TCN-BiLSTM",
        "feature": "F1",
        "optuna_trial": TRIAL,
        "seed": SEED,
        "epoch": EPOCHS,
        "strategy": "Focal",
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_train_loss": history[-1]["train_loss"],
        "hyperparameters": {
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "gradient_clip": GRAD_CLIP,
        },
        "train_samples": int(len(rows)),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "validation_used": fold is not None,
        "external_test_used": False,
        "lr_scheduler": None,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    torch.save(checkpoint, out_dir / (f"fold_{fold}.pt" if fold is not None else "final_model.pt"))
    pd.DataFrame(history).to_csv(out_dir / (f"fold_{fold}_training_history.csv" if fold is not None else "training_history.csv"), index=False)
    del loader, optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checkpoint

# -----------------------------------------------------------------------------
# Stage 1: Trial-26-consistent Focal OOF and locked thresholds
# -----------------------------------------------------------------------------
def run_oof() -> Path:
    print("=" * 90)
    print("STAGE D1 - FOCAL OOF THRESHOLD LOCK")
    print("=" * 90)
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError(f"Missing development F1 data: {X_PATH} / {Y_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH, mmap_mode="r")
    if X.shape != (847513, 60, 20) or y.shape != (847513, 10):
        raise RuntimeError(f"Unexpected development shapes: X={X.shape}, y={y.shape}")
    sorted_anchors, row_order = load_anchor_mapping()
    folds = load_folds(sorted_anchors, row_order)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OOF_ROOT.mkdir(parents=True, exist_ok=True)
    all_y, all_p, all_fold = [], [], []
    fold_audit = []
    for fold in sorted(folds):
        print(f"\n--- OOF FOLD {fold} ---")
        train_rows = folds[fold]["train_rows"]
        val_rows = folds[fold]["val_rows"]
        fold_dir = OOF_ROOT / "folds"
        train_model(X, y, train_rows, device, fold_dir, fold=fold)
        ck = torch.load(fold_dir / f"fold_{fold}.pt", map_location=device, weights_only=False)
        model = build_model().to(device)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        val_ds = MemmapWindowDataset(X, y, val_rows)
        val_loader = make_loader(val_ds, BATCH_SIZE, False, SEED)
        y_val, p_val = predict_loader(model, val_loader, device, with_targets=True)
        all_y.append(y_val)
        all_p.append(p_val)
        all_fold.append(np.full(len(val_rows), fold, dtype=np.int16))
        fold_audit.append({"fold": fold, "train_samples": int(len(train_rows)), "validation_samples": int(len(val_rows))})
        del model, val_loader
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    y_oof = np.concatenate(all_y, axis=0)
    p_oof = np.concatenate(all_p, axis=0)
    folds_oof = np.concatenate(all_fold, axis=0)
    np.savez_compressed(OOF_ROOT / "oof_predictions.npz", y_true=y_oof, y_prob=p_oof, fold=folds_oof)
    rows = []
    threshold_json = {
        "strategy": "Focal",
        "architecture": "TCN-BiLSTM",
        "feature": "F1",
        "optuna_trial": TRIAL,
        "method": "pooled_OOF_validation",
        "objective": "maximum_CSI",
        "tie_breaker_1": "higher_POD",
        "tie_breaker_2": "lower_FAR",
        "threshold_grid": "0.001 to 0.999 step 0.001",
        "external_test_used": False,
        "hyperparameters": {
            "learning_rate": LR, "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            "epochs": EPOCHS, "gradient_clip": GRAD_CLIP,
            "focal_alpha": FOCAL_ALPHA, "focal_gamma": FOCAL_GAMMA,
        },
        "horizons": {}
    }
    for h in range(10):
        m = select_threshold(y_oof[:, h], p_oof[:, h])
        threshold_json["horizons"][f"H{h+1}"] = m
        rows.append({"horizon": f"H{h+1}", **m})
    pd.DataFrame(rows).to_csv(OOF_ROOT / "oof_threshold_summary.csv", index=False)
    with open(OOF_ROOT / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(threshold_json, f, indent=2)
    audit = {
        "stage": "D1 Trial-26-consistent Focal OOF",
        "architecture": "TCN-BiLSTM",
        "feature": "F1",
        "seed": SEED,
        "folds": fold_audit,
        "oof_samples": int(len(y_oof)),
        "development_samples": int(len(X)),
        "external_test_used": False,
        "threshold_external_test_used": False,
        "threshold_method": "pooled_OOF_validation",
        "optimizer": "AdamW",
        "lr_scheduler": "NONE",
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "hyperparameters": threshold_json["hyperparameters"],
        "note": "Recomputed because the earlier imbalance-v3 Focal comparison used a different batch size/weight-decay configuration; its thresholds are not reused.",
    }
    with open(OOF_ROOT / "oof_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print("\nLocked Focal thresholds:")
    print(pd.DataFrame(rows)[["horizon", "threshold", "CSI", "POD", "FAR"]].to_string(index=False))
    return OOF_ROOT / "thresholds.json"

# -----------------------------------------------------------------------------
# Stage 2: Full development retraining
# -----------------------------------------------------------------------------
def run_full() -> Path:
    print("=" * 90)
    print("STAGE D1 - FINAL FULL DEVELOPMENT RETRAINING: TCN-BiLSTM + FOCAL")
    print("=" * 90)
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError(f"Missing development F1 data: {X_PATH} / {Y_PATH}")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH, mmap_mode="r")
    if X.shape != (847513, 60, 20) or y.shape != (847513, 10):
        raise RuntimeError(f"Unexpected development shapes: X={X.shape}, y={y.shape}")
    FULL_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = np.arange(len(X), dtype=np.int64)
    ck = train_model(X, y, rows, device, FULL_ROOT, fold=None)
    final_config = {
        "stage": "D1 final full development retraining - Focal",
        "architecture": "TCN-BiLSTM",
        "feature": "F1",
        "optuna_trial": TRIAL,
        "seed": SEED,
        "device": str(device),
        "data_path": str(X_PATH.parent),
        "development_samples": int(len(X)),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "validation_used": False,
        "external_test_used": False,
        "optimizer": "AdamW",
        "lr_scheduler": "NONE",
        "loss": "Focal",
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "hyperparameters": ck["hyperparameters"],
        "trainable_parameters": ck["trainable_parameters"],
        "final_checkpoint": str(FULL_ROOT / "final_model.pt"),
        "training_history": str(FULL_ROOT / "training_history.csv"),
    }
    with open(FULL_ROOT / "final_config.json", "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)
    return FULL_ROOT / "final_model.pt"

# -----------------------------------------------------------------------------
# Stage 3: External test
# -----------------------------------------------------------------------------
def run_external() -> Path:
    print("=" * 90)
    print("STAGE D1 - INDEPENDENT EXTERNAL TEST: JAN-MAR 2025")
    print("=" * 90)
    threshold_path = OOF_ROOT / "thresholds.json"
    checkpoint_path = FULL_ROOT / "final_model.pt"
    if not threshold_path.exists():
        raise FileNotFoundError("Locked Focal threshold artifact not found. Run --stage oof first.")
    if not checkpoint_path.exists():
        raise FileNotFoundError("Focal final checkpoint not found. Run --stage full first.")
    if not (EXTERNAL_DATA_PATH / "X.npy").exists():
        raise FileNotFoundError(f"External X.npy not found: {EXTERNAL_DATA_PATH}")
    if not (EXTERNAL_DATA_PATH / "y.npy").exists():
        raise FileNotFoundError(f"External y.npy not found: {EXTERNAL_DATA_PATH}")
    if not EXTERNAL_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"External manifest not found: {EXTERNAL_MANIFEST_PATH}")
    manifest = pd.read_csv(EXTERNAL_MANIFEST_PATH)
    if "anchor_time" not in manifest.columns:
        raise RuntimeError("external_test_windows.csv must contain anchor_time.")
    manifest["anchor_time_parsed"] = pd.to_datetime(manifest["anchor_time"], errors="coerce")
    if manifest["anchor_time_parsed"].isna().any():
        raise RuntimeError("Invalid external anchor_time values found.")
    start = pd.Timestamp("2025-01-01 00:00:00")
    end = pd.Timestamp("2025-04-01 00:00:00")
    outside = ~((manifest["anchor_time_parsed"] >= start) & (manifest["anchor_time_parsed"] < end))
    if outside.any():
        raise RuntimeError(f"External manifest contains {int(outside.sum())} rows outside Jan-Mar 2025.")
    X = np.load(EXTERNAL_DATA_PATH / "X.npy", mmap_mode="r")
    y = np.load(EXTERNAL_DATA_PATH / "y.npy", mmap_mode="r")
    if X.shape[1:] != (60, 20):
        raise RuntimeError(f"Unexpected external X shape: {X.shape}")
    if y.shape[1:] != (10,):
        raise RuntimeError(f"Unexpected external y shape: {y.shape}")
    if len(X) != len(manifest) or len(y) != len(manifest):
        raise RuntimeError(f"External X/y/manifest misalignment: X={len(X)}, y={len(y)}, manifest={len(manifest)}")
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_cfg = json.load(f)
    if threshold_cfg.get("external_test_used", True) is not False:
        raise RuntimeError("Threshold artifact is not locked as external-test independent.")
    if threshold_cfg.get("method") != "pooled_OOF_validation":
        raise RuntimeError("Threshold artifact is not pooled-OOF based.")
    thresholds = [float(threshold_cfg["horizons"][f"H{h}"]["threshold"]) for h in range(1, 11)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    loader_kwargs = {
        "dataset": ExternalDataset(X),
        "batch_size": 256,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)
    _, probabilities = predict_loader(model, loader, device, with_targets=False)
    metrics = []
    pred = {}
    for h in range(10):
        yt = np.asarray(y[:, h], dtype=np.uint8)
        yp = probabilities[:, h].astype(np.float64)
        thr = thresholds[h]
        m = threshold_metrics(yt, yp, thr)
        if len(np.unique(yt)) == 2:
            m["AUPRC"] = float(average_precision_score(yt, yp))
            m["ROC_AUC"] = float(roc_auc_score(yt, yp))
        else:
            m["AUPRC"] = float("nan")
            m["ROC_AUC"] = float("nan")
        m["horizon"] = f"H{h+1}"
        m["N"] = int(len(yt))
        m["events"] = int(yt.sum())
        metrics.append(m)
        pred[f"y_h{h+1}"] = yt
        pred[f"p_h{h+1}"] = yp.astype(np.float32)
        pred[f"pred_h{h+1}"] = (yp >= thr).astype(np.int8)
    metrics_df = pd.DataFrame(metrics)[["horizon","threshold","AUPRC","ROC_AUC","CSI","POD","Precision","FAR","F1","F2","TP","FP","FN","TN","N","events"]]
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_output = manifest.copy()
    for k, v in pred.items(): prediction_output[k] = v
    prediction_output.to_csv(EXTERNAL_ROOT / "external_predictions.csv", index=False)
    metrics_df.to_csv(EXTERNAL_ROOT / "external_metrics.csv", index=False)
    summary = {
        "stage": "D1 External Test - Focal",
        "architecture": "TCN-BiLSTM",
        "feature": "F1",
        "optuna_trial": TRIAL,
        "loss": "Focal",
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "final_retraining": "full development",
        "external_period": "2025-01-01 through 2025-03-31",
        "external_samples": int(len(X)),
        "external_events_by_horizon": {f"H{h}": int(np.sum(y[:,h-1] == 1)) for h in range(1,11)},
        "threshold_source": str(threshold_path),
        "threshold_method": "pooled_OOF_validation",
        "external_test_used_for_threshold": False,
        "checkpoint": str(checkpoint_path),
        "metrics_file": str(EXTERNAL_ROOT / "external_metrics.csv"),
        "predictions_file": str(EXTERNAL_ROOT / "external_predictions.csv"),
    }
    with open(EXTERNAL_ROOT / "external_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    audit = {
        "stage": "D1 final external test - Focal",
        "architecture": "TCN-BiLSTM",
        "feature": "F1",
        "seed": SEED,
        "checkpoint": str(checkpoint_path),
        "checkpoint_source": "full-development Focal retraining",
        "external_data_path": str(EXTERNAL_DATA_PATH),
        "external_manifest": str(EXTERNAL_MANIFEST_PATH),
        "external_samples": int(len(X)),
        "external_date_start": str(manifest["anchor_time_parsed"].min()),
        "external_date_end": str(manifest["anchor_time_parsed"].max()),
        "external_period_filter": "2025-01-01 <= anchor_time < 2025-04-01",
        "outside_date_rows_before_filter": int(outside.sum()),
        "threshold_file": str(threshold_path),
        "threshold_method": "pooled_OOF_validation",
        "threshold_objective": "maximum_CSI",
        "thresholds_locked": thresholds,
        "threshold_external_test_used": False,
        "external_data_used_for_training": False,
        "external_data_used_for_threshold_selection": False,
        "external_data_used_only_for_final_evaluation": True,
        "loss": "Focal",
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "metrics_file": str(EXTERNAL_ROOT / "external_metrics.csv"),
        "predictions_file": str(EXTERNAL_ROOT / "external_predictions.csv"),
    }
    with open(EXTERNAL_ROOT / "external_test_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print("\nEXTERNAL TEST RESULTS")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    return EXTERNAL_ROOT / "external_metrics.csv"

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["oof", "full", "external", "all"], default="all")
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in {"oof", "all"}:
        run_oof()
    if args.stage in {"full", "all"}:
        run_full()
    if args.stage in {"external", "all"}:
        run_external()
    print(f"\nPipeline complete in {(time.time()-t0)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
