import argparse
import json
import torch
import pandas as pd

from data.stagec_loader import (
    load_fold_data,
    load_full_data,
    loader
)

from models.factory import build_model
from training.trainer import train


# ======================================================
# ARGUMENT
# ======================================================

parser = argparse.ArgumentParser(
    description="StageD1 v6.4.4 Complete Real Training Engine"
)


parser.add_argument(
    "--architecture",
    default="TCN",
    help="Model architecture"
)


parser.add_argument(
    "--mode",
    default="fold",
    choices=[
        "fold",
        "full"
    ],
    help="Training mode"
)


parser.add_argument(
    "--run_name",
    default="default",
    help="Experiment name"
)


args = parser.parse_args()



# ======================================================
# CONFIG
# ======================================================

with open(
    "config/config.json"
) as f:
    cfg = json.load(f)



# ======================================================
# DEVICE
# ======================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("="*80)
print("StageD1 v6.4.4 Complete Real Training Engine")
print("="*80)

print("Device:", device)
print("Architecture:", args.architecture)
print("Mode:", args.mode)
print("Run name:", args.run_name)



# ======================================================
# LOAD DATA
# ======================================================

if args.mode == "fold":

    print("CROSS VALIDATION MODE")

    Xtr, ytr, Xv, yv, pos = load_fold_data(
        cfg["data_path"],
        cfg["stageC_path"],
        cfg["fold"]
    )


else:

    print("FULL DEVELOPMENT TRAINING MODE")


    Xtr, ytr, pos = load_full_data(
        cfg["data_path"],
        cfg["stageC_path"]
    )


    Xv = None
    yv = None



print(
    "Train samples:",
    len(Xtr)
)


if Xv is not None:

    print(
        "Validation samples:",
        len(Xv)
    )

else:

    print(
        "Validation samples: NONE"
    )



# ======================================================
# DATALOADER
# ======================================================

train_loader = loader(
    Xtr,
    ytr,
    cfg["batch_size"],
    True
)



if Xv is not None:

    val_loader = loader(
        Xv,
        yv,
        cfg["batch_size"],
        False
    )

else:

    val_loader = None



# ======================================================
# MODEL
# ======================================================

model = build_model(
    args.architecture
).to(device)



# ======================================================
# POS WEIGHT
# ======================================================

if (
    isinstance(pos, dict)
    and "horizons" in pos
):

    weights = [
        pos["horizons"][f"H{i}"]["pos_weight"]
        for i in range(1,11)
    ]

else:

    weights = pos



pos_weight = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)


print("\nPositive weights:")
print(pos_weight)



criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)



# ======================================================
# OPTIMIZER
# ======================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["learning_rate"]
)



# ======================================================
# TRAIN
# ======================================================

best, history = train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    cfg["epochs"],
    args.run_name
)



# ======================================================
# SAVE HISTORY
# ======================================================

pd.DataFrame(
    history
).to_csv(
    f"results/{args.run_name}_training_history.csv",
    index=False
)



print("\nTraining finished")

print(
    "Best Score:",
    best
)