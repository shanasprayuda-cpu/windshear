from pathlib import Path
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import TensorDataset, DataLoader


# ==========================================================
# DEVELOPMENT WINDOW MAPPING
# ==========================================================

def load_development_mapping(stagec_path):

    stagec_path = Path(stagec_path)

    manifest_path = (
        stagec_path
        / "manifest"
        / "development_windows.csv"
    )

    manifest = pd.read_csv(
        manifest_path,
        usecols=["anchor_idx"]
    )

    # Row position in X.npy -> Stage C anchor_idx
    anchor_to_row = {
        int(anchor): row
        for row, anchor
        in enumerate(manifest["anchor_idx"].values)
    }

    return anchor_to_row


# ==========================================================
# LOAD FOLD DATA
# ==========================================================

def load_fold_data(
    data_path,
    stagec_path,
    fold
):

    data_path = Path(data_path)
    stagec_path = Path(stagec_path)


    # ------------------------------------------------------
    # Load Stage D1 arrays
    # ------------------------------------------------------

    X = np.load(
        data_path / "X.npy"
    )

    y = np.load(
        data_path / "y.npy"
    )


    # ------------------------------------------------------
    # Load fold manifests
    # ------------------------------------------------------

    train = pd.read_csv(
        stagec_path
        / "folds"
        / f"fold_{fold}_train.csv"
    )

    val = pd.read_csv(
        stagec_path
        / "folds"
        / f"fold_{fold}_val.csv"
    )


    train_anchor = (
        train["anchor_idx"]
        .astype(int)
        .values
    )

    val_anchor = (
        val["anchor_idx"]
        .astype(int)
        .values
    )


    # ------------------------------------------------------
    # Build Stage C anchor -> X.npy row mapping
    # ------------------------------------------------------

    anchor_to_row = load_development_mapping(
        stagec_path
    )


    # ------------------------------------------------------
    # Validate all anchors exist
    # ------------------------------------------------------

    missing_train = [
        a for a in train_anchor
        if a not in anchor_to_row
    ]

    missing_val = [
        a for a in val_anchor
        if a not in anchor_to_row
    ]


    if missing_train:

        raise ValueError(
            f"Fold {fold}: "
            f"{len(missing_train)} training anchors "
            f"are missing from development_windows.csv. "
            f"Examples: {missing_train[:10]}"
        )


    if missing_val:

        raise ValueError(
            f"Fold {fold}: "
            f"{len(missing_val)} validation anchors "
            f"are missing from development_windows.csv. "
            f"Examples: {missing_val[:10]}"
        )


    # ------------------------------------------------------
    # Convert Stage C anchor_idx -> X.npy row
    # ------------------------------------------------------

    train_idx = np.array(
        [
            anchor_to_row[a]
            for a in train_anchor
        ],
        dtype=np.int64
    )


    val_idx = np.array(
        [
            anchor_to_row[a]
            for a in val_anchor
        ],
        dtype=np.int64
    )


    # ------------------------------------------------------
    # Safety check
    # ------------------------------------------------------

    if train_idx.max() >= len(X):

        raise IndexError(
            f"Fold {fold}: mapped training index "
            f"{train_idx.max()} >= X length {len(X)}"
        )


    if val_idx.max() >= len(X):

        raise IndexError(
            f"Fold {fold}: mapped validation index "
            f"{val_idx.max()} >= X length {len(X)}"
        )


    # ------------------------------------------------------
    # Load fold class weights
    # ------------------------------------------------------

    with open(
        stagec_path
        / "class_weights"
        / f"fold_{fold}_pos_weights.json"
    ) as f:

        weights = json.load(f)


    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        weights
    )


# ==========================================================
# LOAD FULL DEVELOPMENT DATA
# ==========================================================

def load_full_data(
    data_path,
    stagec_path
):

    data_path = Path(data_path)
    stagec_path = Path(stagec_path)


    X = np.load(
        data_path / "X.npy"
    )

    y = np.load(
        data_path / "y.npy"
    )


    # Full development dataset:
    # X.npy/y.npy already contain exactly
    # the 847,513 development windows.

    with open(
        stagec_path
        / "class_weights"
        / "full_dataset_pos_weights.json"
    ) as f:

        weights = json.load(f)


    return (
        X,
        y,
        weights
    )


# ==========================================================
# DATALOADER
# ==========================================================

def loader(
    X,
    y,
    batch,
    shuffle
):

    ds = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        ),
        torch.tensor(
            y,
            dtype=torch.float32
        )
    )


    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0
    )