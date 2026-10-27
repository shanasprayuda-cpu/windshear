import json
import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import pandas as pd
import torch

from models.factory import build_model


# ============================================================
# STAGE D1 MODEL SELECTION
# ============================================================
#
# Architecture candidates:
#   TCN
#   BiLSTM
#   Transformer
#   TCN-BiLSTM
#   TCN-Transformer
#
# Selection criterion:
#   1. Highest mean validation AUPRC
#   2. Lower SD of validation AUPRC
#   3. Fewer trainable parameters
#
# External test data is NOT used here.
#
# ============================================================


ARCHITECTURES = [
    "TCN",
    "BiLSTM",
    "Transformer",
    "TCN-BiLSTM",
    "TCN-Transformer",
]

FOLDS = [1, 2, 3, 4, 5]

CONFIG_PATH = Path("config/config.json")

RESULT_ROOT = Path("results/model_selection")

CHECKPOINT_ROOT = Path("checkpoints")

HISTORY_ROOT = RESULT_ROOT / "histories"

SUMMARY_FILE = RESULT_ROOT / "model_selection_summary.csv"

BEST_FILE = RESULT_ROOT / "best_architecture.json"


# ============================================================
# DIRECTORIES
# ============================================================

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

HISTORY_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)


SEED = int(
    cfg.get("seed", 2026)
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPER: PARAMETER COUNT
# ============================================================

def count_parameters(
    architecture
):

    model = build_model(
        architecture
    )

    total = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return int(total)


# ============================================================
# HELPER: UPDATE FOLD
# ============================================================

def set_fold(
    fold
):

    with open(CONFIG_PATH, "r") as f:
        current_cfg = json.load(f)


    current_cfg["fold"] = fold


    with open(CONFIG_PATH, "w") as f:
        json.dump(
            current_cfg,
            f,
            indent=2
        )


# ============================================================
# HELPER: RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    architecture,
    fold
):

    run_name = (
        f"{architecture}_fold{fold}"
    )


    print("\n")
    print("=" * 90)
    print(
        f"ARCHITECTURE : {architecture}"
    )
    print(
        f"FOLD         : {fold}"
    )
    print(
        f"RUN NAME     : {run_name}"
    )
    print("=" * 90)


    set_fold(fold)


    command = [
        sys.executable,
        "-m",
        "runner.run_training",

        "--architecture",
        architecture,

        "--mode",
        "fold",

        "--run_name",
        run_name,
    ]


    subprocess.run(
        command,
        check=True
    )


    history_file = (
        Path("results")
        / f"{run_name}_training_history.csv"
    )


    if not history_file.exists():

        raise FileNotFoundError(
            f"Training history not found:\n"
            f"{history_file}"
        )


    history = pd.read_csv(
        history_file
    )


    if "auprc" not in history.columns:

        raise ValueError(
            f"'auprc' column not found in "
            f"{history_file}"
        )


    auprc = pd.to_numeric(
        history["auprc"],
        errors="coerce"
    ).dropna()


    if len(auprc) == 0:

        raise ValueError(
            f"No validation AUPRC values found "
            f"in {history_file}"
        )


    # Best validation score for this fold
    best_auprc = float(
        auprc.max()
    )


    best_epoch = int(
        history.loc[
            history["auprc"].idxmax(),
            "epoch"
        ]
    )


    # Copy history into model-selection directory
    destination = (
        HISTORY_ROOT
        / f"{run_name}_training_history.csv"
    )


    history.to_csv(
        destination,
        index=False
    )


    print(
        f"\nFold {fold} best validation "
        f"Mean AUPRC = {best_auprc:.6f}"
    )

    print(
        f"Best epoch = {best_epoch}"
    )


    return {
        "architecture": architecture,
        "fold": fold,
        "best_AUPRC": best_auprc,
        "best_epoch": best_epoch,
        "run_name": run_name,
    }


# ============================================================
# PRE-FLIGHT CHECK
# ============================================================

print("=" * 90)
print("Stage D1 v6.4.4 - AUTOMATED MODEL SELECTION")
print("=" * 90)

print(
    f"Seed           : {SEED}"
)

print(
    f"Architectures  : {len(ARCHITECTURES)}"
)

print(
    f"Folds          : {len(FOLDS)}"
)

print(
    f"Total runs     : "
    f"{len(ARCHITECTURES) * len(FOLDS)}"
)

print(
    "Criterion      : Mean validation AUPRC"
)

print(
    "Tie breaker 1  : Lower AUPRC SD"
)

print(
    "Tie breaker 2  : Fewer parameters"
)

print(
    "External test  : NOT USED"
)

print("=" * 90)


# ============================================================
# PARAMETER AUDIT
# ============================================================

print("\nParameter count audit:")

parameter_counts = {}


for architecture in ARCHITECTURES:

    count = count_parameters(
        architecture
    )

    parameter_counts[
        architecture
    ] = count

    print(
        f"{architecture:20s} "
        f"{count:,} trainable parameters"
    )


# ============================================================
# RUN ALL ARCHITECTURES × FOLDS
# ============================================================

fold_results = []


for architecture in ARCHITECTURES:

    for fold in FOLDS:

        result = run_experiment(
            architecture,
            fold
        )

        fold_results.append(
            result
        )


# ============================================================
# FOLD RESULT TABLE
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)


fold_result_file = (
    RESULT_ROOT
    / "fold_results.csv"
)


fold_df.to_csv(
    fold_result_file,
    index=False
)


# ============================================================
# AGGREGATE ARCHITECTURE RESULTS
# ============================================================

summary_rows = []


for architecture in ARCHITECTURES:

    subset = fold_df[
        fold_df["architecture"]
        == architecture
    ]


    scores = subset[
        "best_AUPRC"
    ].to_numpy(
        dtype=float
    )


    mean_auprc = float(
        np.mean(scores)
    )


    std_auprc = float(
        np.std(
            scores,
            ddof=0
        )
    )


    summary_rows.append(
        {
            "architecture":
                architecture,

            "fold1_AUPRC":
                float(scores[0]),

            "fold2_AUPRC":
                float(scores[1]),

            "fold3_AUPRC":
                float(scores[2]),

            "fold4_AUPRC":
                float(scores[3]),

            "fold5_AUPRC":
                float(scores[4]),

            "mean_AUPRC":
                mean_auprc,

            "std_AUPRC":
                std_auprc,

            "trainable_parameters":
                parameter_counts[
                    architecture
                ],
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# MODEL SELECTION RULE
# ============================================================
#
# 1. Highest mean AUPRC
# 2. Lower SD
# 3. Fewer parameters
#
# ============================================================

summary_df = summary_df.sort_values(
    by=[
        "mean_AUPRC",
        "std_AUPRC",
        "trainable_parameters",
    ],
    ascending=[
        False,
        True,
        True,
    ],
    kind="mergesort"
).reset_index(
    drop=True
)


# ============================================================
# ADD RANK
# ============================================================

summary_df.insert(
    0,
    "rank",
    np.arange(
        1,
        len(summary_df) + 1
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# BEST ARCHITECTURE
# ============================================================

best = summary_df.iloc[0]


best_architecture = {
    "selection_stage":
        "Stage D1 architecture selection",

    "criterion":
        "highest mean validation AUPRC",

    "tie_breaker_1":
        "lower standard deviation of fold AUPRC",

    "tie_breaker_2":
        "fewer trainable parameters",

    "selected_architecture":
        best["architecture"],

    "mean_AUPRC":
        float(best["mean_AUPRC"]),

    "std_AUPRC":
        float(best["std_AUPRC"]),

    "trainable_parameters":
        int(best["trainable_parameters"]),

    "fold_AUPRC": {
        "fold1":
            float(best["fold1_AUPRC"]),

        "fold2":
            float(best["fold2_AUPRC"]),

        "fold3":
            float(best["fold3_AUPRC"]),

        "fold4":
            float(best["fold4_AUPRC"]),

        "fold5":
            float(best["fold5_AUPRC"]),
    },

    "seed":
        SEED,

    "external_test_used":
        False,
}


with open(
    BEST_FILE,
    "w"
) as f:

    json.dump(
        best_architecture,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 90)
print("MODEL SELECTION COMPLETE")
print("=" * 90)

print(
    summary_df.to_string(
        index=False
    )
)

print("\n")
print(
    f"SELECTED ARCHITECTURE: "
    f"{best['architecture']}"
)

print(
    f"Mean validation AUPRC: "
    f"{best['mean_AUPRC']:.6f}"
)

print(
    f"SD validation AUPRC: "
    f"{best['std_AUPRC']:.6f}"
)

print(
    f"Trainable parameters: "
    f"{int(best['trainable_parameters']):,}"
)

print("\nSaved:")
print(
    f"  {fold_result_file}"
)
print(
    f"  {SUMMARY_FILE}"
)
print(
    f"  {BEST_FILE}"
)
print(
    f"  {HISTORY_ROOT}"
)

print("=" * 90)