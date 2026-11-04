import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.tcn_bilstm import TCNBiLSTM
# ============================================================
# STAGE D1 - FINAL FULL DEVELOPMENT RETRAINING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(
    PROJECT_ROOT / "config" / "config.json",
    "r"
) as f:
    CFG = json.load(f)

DATA_PATH = PROJECT_ROOT / CFG["data_path"]
STAGEC_PATH = PROJECT_ROOT / CFG["stageC_path"]





# ============================================================
# LOCKED OPTUNA TRIAL 26
# ============================================================

ARCHITECTURE = "TCN-BiLSTM"
OPTUNA_TRIAL = 26

SEED = 2026

EPOCHS = 50
BATCH_SIZE = 128

LEARNING_RATE = 0.0003695413
WEIGHT_DECAY = 0.0000498414

HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.25

GRAD_CLIP = 1.0

NUM_WORKERS = int(
    CFG.get("num_workers", 2)
)


# ============================================================
# OUTPUT
# ============================================================

RUN_NAME = "FINAL_TCN_BiLSTM_TRIAL26_FULL"

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_full"
    / RUN_NAME
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_CHECKPOINT = RESULT_DIR / "final_model.pt"
HISTORY_FILE = RESULT_DIR / "training_history.csv"
CONFIG_FILE = RESULT_DIR / "final_config.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SEED
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

class FullDevelopmentDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return self.X.shape[0]

    def __getitem__(self, index):

        x = np.asarray(
            self.X[index],
            dtype=np.float32
        )

        target = np.asarray(
            self.y[index],
            dtype=np.float32
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(target)
        )


# ============================================================
# DATA LOADER
# ============================================================

def make_loader(dataset):

    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": True
    }

    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    return DataLoader(**kwargs)


# ============================================================
# FULL DEVELOPMENT POS WEIGHTS
# ============================================================

def load_full_pos_weights():

    path = (
        STAGEC_PATH
        / "class_weights"
        / "full_dataset_pos_weights.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing:\n{path}"
        )

    with open(path, "r") as f:
        obj = json.load(f)

    if "horizons" in obj:

        weights = [
            obj["horizons"][f"H{i}"]["pos_weight"]
            for i in range(1, 11)
        ]

    else:

        weights = obj

    if len(weights) != 10:

        raise RuntimeError(
            f"Expected 10 weights, got {len(weights)}"
        )

    return path, weights


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    print("=" * 90)
    print("STAGE D1 - FINAL FULL DEVELOPMENT RETRAINING")
    print("=" * 90)

    print("Device          :", DEVICE)
    print("Architecture    :", ARCHITECTURE)
    print("Optuna trial    :", OPTUNA_TRIAL)
    print("Mode            : FULL DEVELOPMENT")
    print("Validation      : NONE")
    print("External test   : NOT USED")

    if DEVICE.type == "cuda":
        print(
            "GPU             :",
            torch.cuda.get_device_name(0)
        )

    print("-" * 90)

    print("Locked hyperparameters:")
    print("learning_rate   :", LEARNING_RATE)
    print("batch_size      :", BATCH_SIZE)
    print("weight_decay    :", WEIGHT_DECAY)
    print("hidden_size     :", HIDDEN_SIZE)
    print("num_layers      :", NUM_LAYERS)
    print("dropout         :", DROPOUT)
    print("epochs          :", EPOCHS)
    print("gradient_clip   :", GRAD_CLIP)
    print("num_workers     :", NUM_WORKERS)

    print("=" * 90)


    # ========================================================
    # LOAD DEVELOPMENT DATA
    # ========================================================

    X = np.load(
        DATA_PATH / "X.npy",
        mmap_mode="r"
    )

    y = np.load(
        DATA_PATH / "y.npy",
        mmap_mode="r"
    )

    print("\nDataset:")
    print("X shape         :", X.shape)
    print("y shape         :", y.shape)


    # ========================================================
    # INTEGRITY CHECK
    # ========================================================

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
    # DATASET
    # ========================================================

    dataset = FullDevelopmentDataset(
        X,
        y
    )

    train_loader = make_loader(
        dataset
    )

    print(
        "Train samples    :",
        len(dataset)
    )

    print(
        "Train batches    :",
        len(train_loader)
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = TCNBiLSTM(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable params :",
        f"{parameter_count:,}"
    )


    # ========================================================
    # POSITIVE WEIGHTS
    # ========================================================

    pos_weight_path, weights = (
        load_full_pos_weights()
    )

    pos_weight = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    print("\nFull-development positive weights:")

    for i, value in enumerate(
        weights,
        start=1
    ):

        print(
            f"H{i:02d}: {value:.6f}"
        )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # IMPORTANT:
    # No learning-rate scheduler.


    # ========================================================
    # AMP
    # ========================================================

    use_amp = (
        DEVICE.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )


    # ========================================================
    # TRAINING
    # ========================================================

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        loss_sum = 0.0
        batch_count = 0

        for xb, yb in train_loader:

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            yb = yb.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp,
                dtype=torch.float16
            ):

                logits = model(xb)

                loss = criterion(
                    logits,
                    yb
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            loss_sum += float(
                loss.detach().item()
            )

            batch_count += 1


        avg_loss = (
            loss_sum
            / max(batch_count, 1)
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "architecture": ARCHITECTURE,
            "optuna_trial": OPTUNA_TRIAL,
            "mode": "FULL_DEVELOPMENT"
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"Loss={avg_loss:.6f}",
            flush=True
        )


    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================
    #
    # There is no validation score here.
    # Therefore there is no "best validation epoch".
    #
    # The final checkpoint is the model state after
    # completing all 50 epochs.
    # ========================================================

    torch.save(
        {
            "stage":
                "D1_FINAL_FULL_DEVELOPMENT",

            "architecture":
                ARCHITECTURE,

            "optuna_trial":
                OPTUNA_TRIAL,

            "seed":
                SEED,

            "epoch":
                EPOCHS,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "final_train_loss":
                history[-1]["train_loss"],

            "hyperparameters":
                {
                    "learning_rate":
                        LEARNING_RATE,

                    "batch_size":
                        BATCH_SIZE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "num_layers":
                        NUM_LAYERS,

                    "dropout":
                        DROPOUT,

                    "epochs":
                        EPOCHS,

                    "gradient_clip":
                        GRAD_CLIP
                },

            "train_samples":
                len(dataset),

            "X_shape":
                list(X.shape),

            "y_shape":
                list(y.shape),

            "validation_used":
                False,

            "external_test_used":
                False,

            "lr_scheduler":
                None,

            "trainable_parameters":
                parameter_count
        },
        FINAL_CHECKPOINT
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_FILE,
        index=False
    )


    # ========================================================
    # SAVE FINAL CONFIG
    # ========================================================

    final_config = {

        "stage":
            "D1 final full development retraining",

        "architecture":
            ARCHITECTURE,

        "optuna_trial":
            OPTUNA_TRIAL,

        "seed":
            SEED,

        "device":
            str(DEVICE),

        "data_path":
            str(DATA_PATH),

        "development_samples":
            len(dataset),

        "X_shape":
            list(X.shape),

        "y_shape":
            list(y.shape),

        "validation_used":
            False,

        "external_test_used":
            False,

        "optimizer":
            "AdamW",

        "lr_scheduler":
            "NONE",

        "amp_fp16":
            use_amp,

        "gradient_clip":
            GRAD_CLIP,

        "hyperparameters":
            {
                "learning_rate":
                    LEARNING_RATE,

                "batch_size":
                    BATCH_SIZE,

                "weight_decay":
                    WEIGHT_DECAY,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "dropout":
                    DROPOUT,

                "epochs":
                    EPOCHS
            },

        "trainable_parameters":
            parameter_count,

        "pos_weight_source":
            str(pos_weight_path),

        "final_checkpoint":
            str(FINAL_CHECKPOINT),

        "training_history":
            str(HISTORY_FILE)
    }

    with open(
        CONFIG_FILE,
        "w"
    ) as f:

        json.dump(
            final_config,
            f,
            indent=2
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 90)
    print(
        "FINAL FULL DEVELOPMENT RETRAINING FINISHED"
    )
    print("=" * 90)

    print(
        "Final checkpoint:"
    )

    print(
        FINAL_CHECKPOINT
    )

    print(
        "\nTraining history:"
    )

    print(
        HISTORY_FILE
    )

    print(
        "\nFinal config:"
    )

    print(
        CONFIG_FILE
    )

    print(
        "\nNEXT STAGE:"
    )

    print(
        "EXTERNAL TEST JAN-MAR 2025"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()