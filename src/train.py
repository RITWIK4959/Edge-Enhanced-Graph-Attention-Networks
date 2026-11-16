import copy
import time

import torch

import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


from .config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    GRADIENT_CLIP,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    MIN_LEARNING_RATE,
    LABEL_SMOOTHING,
)


# ============================================================
# METRICS
# ============================================================

def classification_metrics(
    y_true,
    y_pred,
):

    return {

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "Macro Precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "Macro Recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "Macro F1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "Weighted F1":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data,
    mask,
    edge_attr=None,
):

    model.eval()

    logits = model(
        data.x,
        data.edge_index,
        edge_attr,
    )

    predictions = (
        logits.argmax(
            dim=1
        )
    )

    y_true = (
        data.y[mask]
        .detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        predictions[mask]
        .detach()
        .cpu()
        .numpy()
    )

    return classification_metrics(
        y_true,
        y_pred,
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    data,
    device,
    edge_attr=None,
):

    model = model.to(
        device
    )

    data = data.to(
        device
    )

    if edge_attr is not None:

        edge_attr = edge_attr.to(
            device
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
        )
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss(
            label_smoothing=
                LABEL_SMOOTHING
        )
    )

    # --------------------------------------------------------
    # Best checkpoint
    #
    # IMPORTANT:
    # Macro-F1 on validation set is used.
    # --------------------------------------------------------

    best_val_f1 = -1.0

    best_val_accuracy = -1.0

    best_state = None

    best_epoch = 0

    patience_counter = 0

    start_time = time.time()

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            data.x,
            data.edge_index,
            edge_attr,
        )

        loss = criterion(
            logits[
                data.train_mask
            ],
            data.y[
                data.train_mask
            ],
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Training accuracy
        # ----------------------------------------------------

        train_predictions = (
            logits.argmax(
                dim=1
            )
        )

        train_accuracy = (
            (
                train_predictions[
                    data.train_mask
                ]
                ==
                data.y[
                    data.train_mask
                ]
            )
            .float()
            .mean()
            .item()
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            data,
            data.val_mask,
            edge_attr,
        )

        val_accuracy = (
            val_metrics[
                "Accuracy"
            ]
        )

        val_f1 = (
            val_metrics[
                "Macro F1"
            ]
        )

        # ----------------------------------------------------
        # Scheduler uses validation Macro-F1
        # ----------------------------------------------------

        scheduler.step(
            val_f1
        )

        # ----------------------------------------------------
        # Checkpoint selection
        #
        # Primary:
        #     Macro F1
        #
        # Tie breaker:
        #     Accuracy
        # ----------------------------------------------------

        is_better = (
            val_f1 > best_val_f1
            or (
                val_f1 == best_val_f1
                and
                val_accuracy >
                best_val_accuracy
            )
        )

        if is_better:

            best_val_f1 = val_f1

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = epoch

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            patience_counter = 0

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 25 == 0
        ):

            current_lr = (
                optimizer
                .param_groups[0]
                ["lr"]
            )

            print(
                f"Epoch {epoch:4d} | "
                f"Loss {loss.item():.4f} | "
                f"Train {train_accuracy:.4f} | "
                f"Val Acc {val_accuracy:.4f} | "
                f"Val F1 {val_f1:.4f} | "
                f"LR {current_lr:.6f}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch}"
            )

            break

    # ========================================================
    # RESTORE BEST VALIDATION CHECKPOINT
    # ========================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # ========================================================
    # FINAL TEST
    #
    # This is performed ONLY after
    # selecting the validation checkpoint.
    # ========================================================

    test_metrics = evaluate(
        model,
        data,
        data.test_mask,
        edge_attr,
    )

    training_time = (
        time.time()
        - start_time
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    return {

        "Validation Accuracy":
            best_val_accuracy,

        "Validation Macro F1":
            best_val_f1,

        "Test Accuracy":
            test_metrics[
                "Accuracy"
            ],

        "Macro Precision":
            test_metrics[
                "Macro Precision"
            ],

        "Macro Recall":
            test_metrics[
                "Macro Recall"
            ],

        "Macro F1":
            test_metrics[
                "Macro F1"
            ],

        "Weighted F1":
            test_metrics[
                "Weighted F1"
            ],

        "Parameters":
            parameters,

        "Best Epoch":
            best_epoch,

        "Training Time":
            training_time,
    }