import json
import random
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna

from models.tcn import TCN
from models.bilstm import BiLSTM
from models.transformer import Transformer
from models.tcn_bilstm import TCNBiLSTM
from models.tcn_transformer import TCNTransformer


# ============================================================
# D1 CONFIGURATION
# ============================================================

CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

with open(CONFIG_PATH, "r") as f:
    CFG = json.load(f)

DATA_PATH = PROJECT_ROOT / CFG["data_path"]
STAGEC_PATH = PROJECT_ROOT / CFG["stageC_path"]

RESULT_ROOT = PROJECT_ROOT / "results" / "optuna"
CHECKPOINT_ROOT = RESULT_ROOT / "checkpoints"
TRIAL_ROOT = RESULT_ROOT / "trials"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
TRIAL_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_TRIALS = 30
MAX_EPOCHS = 50
PATIENCE = 7
CLIP_NORM = 1.0
FOLDS = [1, 2, 3, 4, 5]

ARCHITECTURES = [
    "TCN",
    "BiLSTM",
    "Transformer",
    "TCN-BiLSTM",
    "TCN-Transformer",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Set this in config.json if desired; otherwise 2.
NUM_WORKERS = int(CFG.get("num_workers", 2))


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int):
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

    def __init__(self, x_memmap, y_memmap, row_indices):
        self.X = x_memmap
        self.y = y_memmap
        self.rows = np.asarray(
            row_indices,
            dtype=np.int64
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):

        row = int(self.rows[i])

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
            torch.from_numpy(y)
        )


# ============================================================
# STAGE C ANCHOR -> X.NPY ROW MAPPING
# ============================================================

def load_anchor_mapping():

    manifest = pd.read_csv(
        STAGEC_PATH
        / "manifest"
        / "development_windows.csv",
        usecols=["anchor_idx"]
    )

    return {
        int(anchor): row
        for row, anchor
        in enumerate(
            manifest["anchor_idx"].to_numpy()
        )
    }


ANCHOR_TO_ROW = load_anchor_mapping()


def load_fold_indices(fold):

    train = pd.read_csv(
        STAGEC_PATH
        / "folds"
        / f"fold_{fold}_train.csv",
        usecols=["anchor_idx"]
    )

    val = pd.read_csv(
        STAGEC_PATH
        / "folds"
        / f"fold_{fold}_val.csv",
        usecols=["anchor_idx"]
    )

    train_anchor = (
        train["anchor_idx"]
        .astype(np.int64)
        .to_numpy()
    )

    val_anchor = (
        val["anchor_idx"]
        .astype(np.int64)
        .to_numpy()
    )

    missing_train = [
        int(a)
        for a in train_anchor
        if int(a) not in ANCHOR_TO_ROW
    ]

    missing_val = [
        int(a)
        for a in val_anchor
        if int(a) not in ANCHOR_TO_ROW
    ]

    if missing_train:
        raise RuntimeError(
            f"Fold {fold}: missing training anchors. "
            f"Examples: {missing_train[:10]}"
        )

    if missing_val:
        raise RuntimeError(
            f"Fold {fold}: missing validation anchors. "
            f"Examples: {missing_val[:10]}"
        )

    train_rows = np.asarray(
        [
            ANCHOR_TO_ROW[int(a)]
            for a in train_anchor
        ],
        dtype=np.int64
    )

    val_rows = np.asarray(
        [
            ANCHOR_TO_ROW[int(a)]
            for a in val_anchor
        ],
        dtype=np.int64
    )

    return train_rows, val_rows


# ============================================================
# POSITIVE WEIGHTS
# ============================================================

def load_pos_weight(fold):

    with open(
        STAGEC_PATH
        / "class_weights"
        / f"fold_{fold}_pos_weights.json",
        "r"
    ) as f:
        obj = json.load(f)

    if "horizons" in obj:

        weights = [
            obj["horizons"][f"H{i}"]["pos_weight"]
            for i in range(1, 11)
        ]

    else:

        weights = obj

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )


# ============================================================
# MODEL FACTORY FOR OPTUNA TRIALS
# ============================================================

def build_tuned_model(
    architecture,
    trial
):

    if architecture == "TCN":

        # The current TCN implementation has fixed
        # 64-channel convolution blocks.
        return TCN()


    if architecture == "BiLSTM":

        hidden_size = trial.suggest_categorical(
            "hidden_size",
            [32, 64, 96, 128]
        )

        num_layers = trial.suggest_int(
            "num_layers",
            1,
            3
        )

        dropout = trial.suggest_float(
            "dropout",
            0.0,
            0.30,
            step=0.05
        )

        return BiLSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )


    if architecture == "Transformer":

        d_model = trial.suggest_categorical(
            "d_model",
            [32, 64, 96, 128]
        )

        # All four d_model choices are divisible by
        # 2, 4 and 8, so this categorical space is valid
        # across all trials.
        nhead = trial.suggest_categorical(
            "nhead",
            [2, 4, 8]
        )

        num_layers = trial.suggest_int(
            "num_layers",
            1,
            3
        )

        dim_feedforward = trial.suggest_categorical(
            "dim_feedforward",
            [64, 128, 256]
        )

        dropout = trial.suggest_float(
            "dropout",
            0.0,
            0.30,
            step=0.05
        )

        return Transformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )


    if architecture == "TCN-BiLSTM":

        hidden_size = trial.suggest_categorical(
            "hidden_size",
            [32, 64, 96, 128]
        )

        num_layers = trial.suggest_int(
            "num_layers",
            1,
            3
        )

        dropout = trial.suggest_float(
            "dropout",
            0.0,
            0.30,
            step=0.05
        )

        return TCNBiLSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )


    if architecture == "TCN-Transformer":

        d_model = trial.suggest_categorical(
            "d_model",
            [32, 64, 96, 128]
        )

        nhead = trial.suggest_categorical(
            "nhead",
            [2, 4, 8]
        )

        num_layers = trial.suggest_int(
            "num_layers",
            1,
            3
        )

        dim_feedforward = trial.suggest_categorical(
            "dim_feedforward",
            [64, 128, 256]
        )

        dropout = trial.suggest_float(
            "dropout",
            0.0,
            0.30,
            step=0.05
        )

        return TCNTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )


    raise ValueError(
        f"Unknown architecture: {architecture}"
    )


# ============================================================
# AUPRC
# ============================================================

def mean_auprc_numpy(
    y_true,
    y_prob
):

    from sklearn.metrics import average_precision_score

    scores = []

    for h in range(
        y_true.shape[1]
    ):

        scores.append(
            average_precision_score(
                y_true[:, h],
                y_prob[:, h]
            )
        )

    return float(
        np.mean(scores)
    )


# ============================================================
# DATALOADER
# ============================================================

def make_loader(
    dataset,
    batch_size,
    shuffle
):

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": True
    }

    if NUM_WORKERS > 0:

        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    return DataLoader(
        **kwargs
    )


# ============================================================
# ONE FOLD / ONE TRIAL
# ============================================================

def run_fold(
    architecture,
    trial,
    fold,
    X,
    y,
    batch_size,
    learning_rate,
    weight_decay
):

    train_rows, val_rows = (
        load_fold_indices(fold)
    )

    train_ds = IndexedMemmapDataset(
        X,
        y,
        train_rows
    )

    val_ds = IndexedMemmapDataset(
        X,
        y,
        val_rows
    )

    train_loader = make_loader(
        train_ds,
        batch_size,
        True
    )

    val_loader = make_loader(
        val_ds,
        batch_size,
        False
    )

    model = build_tuned_model(
        architecture,
        trial
    ).to(DEVICE)

    pos_weight = load_pos_weight(
        fold
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    use_amp = (
        DEVICE.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_score = -np.inf
    best_epoch = 0
    patience_count = 0

    history = []

    checkpoint = (
        CHECKPOINT_ROOT
        / architecture
        / f"trial_{trial.number}"
        / f"fold_{fold}.pt"
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

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
                CLIP_NORM
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


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        yt = []
        yp = []

        with torch.no_grad():

            for xb, yb in val_loader:

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

                probabilities = (
                    torch.sigmoid(
                        logits
                    )
                )

                yt.append(
                    yb.numpy()
                )

                yp.append(
                    probabilities
                    .float()
                    .cpu()
                    .numpy()
                )

        y_true = np.concatenate(
            yt,
            axis=0
        )

        y_prob = np.concatenate(
            yp,
            axis=0
        )

        score = mean_auprc_numpy(
            y_true,
            y_prob
        )

        history.append(
            {
                "architecture": architecture,
                "trial": trial.number,
                "fold": fold,
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_mean_AUPRC": score
            }
        )

        print(
            f"[{architecture}] "
            f"Trial {trial.number:02d} "
            f"Fold {fold} "
            f"Epoch {epoch:02d}/{MAX_EPOCHS} "
            f"Loss={avg_loss:.5f} "
            f"ValAUPRC={score:.6f}",
            flush=True
        )


        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if score > best_score:

            best_score = score
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "architecture":
                        architecture,
                    "trial":
                        trial.number,
                    "fold":
                        fold,
                    "epoch":
                        epoch,
                    "val_AUPRC":
                        score,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "params":
                        trial.params
                },
                checkpoint
            )

        else:

            patience_count += 1


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_count >= PATIENCE:
            break


    history_file = (
        TRIAL_ROOT
        / architecture
        / f"trial_{trial.number}"
        / f"fold_{fold}_history.csv"
    )

    history_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_file,
        index=False
    )

    del model
    del train_loader
    del val_loader

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return (
        best_score,
        best_epoch
    )


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def make_objective(
    architecture,
    X,
    y
):

    def objective(trial):

        seed_everything(
            SEED
        )

        learning_rate = (
            trial.suggest_float(
                "learning_rate",
                1e-5,
                5e-4,
                log=True
            )
        )

        batch_size = (
            trial.suggest_categorical(
                "batch_size",
                [128, 256, 512]
            )
        )

        weight_decay = (
            trial.suggest_float(
                "weight_decay",
                1e-6,
                1e-2,
                log=True
            )
        )

        fold_scores = []
        fold_epochs = []

        for fold in FOLDS:

            seed_everything(
                SEED
            )

            score, best_epoch = (
                run_fold(
                    architecture=architecture,
                    trial=trial,
                    fold=fold,
                    X=X,
                    y=y,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay
                )
            )

            fold_scores.append(
                float(score)
            )

            fold_epochs.append(
                int(best_epoch)
            )

            cumulative_mean = float(
                np.mean(fold_scores)
            )

            trial.set_user_attr(
                f"fold_{fold}_AUPRC",
                float(score)
            )

            trial.set_user_attr(
                f"fold_{fold}_best_epoch",
                int(best_epoch)
            )

            print(
                f"[{architecture}] "
                f"Trial {trial.number:02d} "
                f"completed Fold {fold}: "
                f"AUPRC={score:.6f}; "
                f"cumulative mean={cumulative_mean:.6f}",
                flush=True
            )


        mean_score = float(
            np.mean(fold_scores)
        )

        std_score = float(
            np.std(
                fold_scores,
                ddof=0
            )
        )

        trial.set_user_attr(
            "fold_scores",
            fold_scores
        )

        trial.set_user_attr(
            "fold_best_epochs",
            fold_epochs
        )

        trial.set_user_attr(
            "mean_AUPRC",
            mean_score
        )

        trial.set_user_attr(
            "std_AUPRC",
            std_score
        )

        return mean_score

    return objective


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "Stage D1 - OPTUNA TPE MODEL SELECTION"
    )
    print("=" * 90)

    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"Seed          : {SEED}"
    )

    print(
        f"Trials/model  : {N_TRIALS}"
    )

    print(
        f"Folds         : {FOLDS}"
    )

    print(
        f"Max epochs    : {MAX_EPOCHS}"
    )

    print(
        f"Patience      : {PATIENCE}"
    )

    print(
        "Optimizer     : AdamW"
    )

    print(
        "LR scheduler  : NONE"
    )

    print(
        "AMP FP16      : "
        f"{DEVICE.type == 'cuda'}"
    )

    print(
        "Grad clip     : 1.0"
    )

    print(
        "Objective     : "
        "Mean validation AUPRC"
    )

    print(
        "External test : NOT USED"
    )

    print("=" * 90)


    # --------------------------------------------------------
    # Dataset integrity checks
    # --------------------------------------------------------

    X = np.load(
        DATA_PATH / "X.npy",
        mmap_mode="r"
    )

    y = np.load(
        DATA_PATH / "y.npy",
        mmap_mode="r"
    )

    print(
        f"X shape      : {X.shape}"
    )

    print(
        f"y shape      : {y.shape}"
    )

    if X.shape[0] != 847513:
        raise RuntimeError(
            f"Unexpected X row count: "
            f"{X.shape[0]}"
        )

    if y.shape[0] != 847513:
        raise RuntimeError(
            f"Unexpected y row count: "
            f"{y.shape[0]}"
        )

    if X.shape[1:] != (60, 20):
        raise RuntimeError(
            f"Unexpected X shape: "
            f"{X.shape}"
        )

    if y.shape[1:] != (10,):
        raise RuntimeError(
            f"Unexpected y shape: "
            f"{y.shape}"
        )


    # --------------------------------------------------------
    # Run each architecture separately
    # --------------------------------------------------------

    architecture_rows = []

    for architecture in ARCHITECTURES:

        print("\n" + "#" * 90)
        print(
            f"OPTUNA STUDY: {architecture}"
        )
        print("#" * 90)

        study_dir = (
            RESULT_ROOT / architecture
        )

        study_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        storage_path = (
            study_dir / "study.db"
        )

        storage = (
            "sqlite:///"
            + storage_path.as_posix()
        )

        study = optuna.create_study(
            study_name=(
                f"StageD1_{architecture}"
            ),
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=SEED
            )
        )

        completed_trials = len(
            study.trials
        )

        remaining_trials = max(
            0,
            N_TRIALS - completed_trials
        )

        print(
            f"Existing trials : "
            f"{completed_trials}"
        )

        print(
            f"New trials      : "
            f"{remaining_trials}"
        )

        if remaining_trials > 0:

            study.optimize(
                make_objective(
                    architecture,
                    X,
                    y
                ),
                n_trials=remaining_trials,
                gc_after_trial=True
            )


        best_trial = (
            study.best_trial
        )

        fold_scores = (
            best_trial.user_attrs
            .get(
                "fold_scores",
                []
            )
        )

        if fold_scores:

            std_score = float(
                np.std(
                    fold_scores,
                    ddof=0
                )
            )

        else:

            std_score = float("nan")


        architecture_rows.append(
            {
                "architecture":
                    architecture,

                "best_trial":
                    best_trial.number,

                "mean_AUPRC":
                    float(
                        best_trial.value
                    ),

                "std_AUPRC":
                    std_score,

                "best_params_json":
                    json.dumps(
                        best_trial.params
                    ),

                "fold_scores_json":
                    json.dumps(
                        fold_scores
                    )
            }
        )


        with open(
            study_dir / "best_trial.json",
            "w"
        ) as f:

            json.dump(
                {
                    "architecture":
                        architecture,

                    "best_trial":
                        best_trial.number,

                    "mean_AUPRC":
                        float(
                            best_trial.value
                        ),

                    "std_AUPRC":
                        std_score,

                    "params":
                        best_trial.params,

                    "fold_scores":
                        fold_scores,

                    "fold_best_epochs":
                        best_trial.user_attrs.get(
                            "fold_best_epochs",
                            []
                        )
                },
                f,
                indent=2
            )


    # --------------------------------------------------------
    # Architecture comparison
    # --------------------------------------------------------

    summary = pd.DataFrame(
        architecture_rows
    )

    summary = summary.sort_values(
        by=[
            "mean_AUPRC",
            "std_AUPRC"
        ],
        ascending=[
            False,
            True
        ],
        kind="mergesort"
    ).reset_index(
        drop=True
    )

    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1
        )
    )

    summary_file = (
        RESULT_ROOT
        / "architecture_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )


    best = summary.iloc[0]

    best_architecture = {
        "selected_architecture":
            str(
                best["architecture"]
            ),

        "best_trial":
            int(
                best["best_trial"]
            ),

        "mean_AUPRC":
            float(
                best["mean_AUPRC"]
            ),

        "std_AUPRC":
            float(
                best["std_AUPRC"]
            ),

        "params":
            json.loads(
                best["best_params_json"]
            ),

        "fold_scores":
            json.loads(
                best["fold_scores_json"]
            ),

        "selection_rule":
            [
                "highest mean validation AUPRC",
                "lower SD as tie-breaker",
                "parameter count as secondary audit"
            ],

        "seed":
            SEED,

        "n_trials_per_architecture":
            N_TRIALS,

        "external_test_used":
            False
    }


    with open(
        RESULT_ROOT
        / "best_architecture.json",
        "w"
    ) as f:

        json.dump(
            best_architecture,
            f,
            indent=2
        )


    print("\n" + "=" * 90)
    print(
        "OPTUNA MODEL SELECTION COMPLETE"
    )
    print("=" * 90)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nSelected architecture:")
    print(
        best_architecture[
            "selected_architecture"
        ]
    )

    print(
        f"Mean validation AUPRC: "
        f"{best_architecture['mean_AUPRC']:.6f}"
    )

    print(
        f"SD validation AUPRC: "
        f"{best_architecture['std_AUPRC']:.6f}"
    )

    print("\nSaved:")
    print(
        summary_file
    )

    print(
        RESULT_ROOT
        / "best_architecture.json"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
