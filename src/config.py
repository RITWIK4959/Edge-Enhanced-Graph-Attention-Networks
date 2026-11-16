import torch


# ============================================================
# REPRODUCIBILITY
# ============================================================

# Final paper experiments
SEEDS = (42, 43, 44, 45, 46)

# True = run all five seeds
# False = run only the first seed
RUN_ALL_SEEDS = True


# ============================================================
# DATA
# ============================================================

DATASET_ROOT = "./data"

DATASETS = (
    "Cora",
    "CiteSeer",
    "PubMed",
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

CPU_NUM_THREADS = 4


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

NODE_DIM = 64
HIDDEN_CHANNELS = 64

NUM_HEADS = 4
NUM_LAYERS = 2


# ============================================================
# EDGE CONFIGURATION
# ============================================================

EDGE_DIM = 16
EDGE_ENCODER_HIDDEN = 16


# ============================================================
# REGULARIZATION
# ============================================================

FEATURE_DROPOUT = 0.45
ATTENTION_DROPOUT = 0.35
EDGE_DROPOUT = 0.10
CLASSIFIER_DROPOUT = 0.45


# ============================================================
# OPTIMIZATION
# ============================================================

LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.005


# ============================================================
# LABEL SMOOTHING
# ============================================================

LABEL_SMOOTHING = 0.02


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 400
PATIENCE = 50
GRADIENT_CLIP = 1.0


# ============================================================
# LR SCHEDULER
# ============================================================

LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5


# ============================================================
# CHECKPOINT SELECTION
# ============================================================

# Model selection is based on validation Macro-F1.
# Test set is evaluated only after restoring the best checkpoint.
CHECKPOINT_METRIC = "Macro F1"