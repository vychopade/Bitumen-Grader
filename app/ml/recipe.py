"""Prince & Prasad training recipe (Table 2) for froth-image models.

Shared across the trainer, dataset split, and Train page so the desktop app
follows the same optimisation protocol as the study rather than a mix of
schedules, penalties, and early-stop heuristics.
"""

from __future__ import annotations

# Table 2 — training configuration shared across all strategies.
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE_FT = 1e-4  # baseline CNN and fine-tuning
LEARNING_RATE_FE = 1e-3  # frozen feature extraction
WEIGHT_DECAY = 0.0
# Paper Case 1 is an 80/20 image split. The 20% hold-out is the test set;
# 20% of the remaining 80% is carved out as validation for checkpointing.
TEST_FRACTION = 0.20
VAL_FRACTION = 0.16
# Prefer holding out whole flotation campaigns (Case 2). That is closer to
# grading a new plant run than shuffling images. Falls back to a random
# image split when fewer than two campaigns are found.
DEFAULT_SPLIT_MODE = "experiment"
CLS_BINS = 3  # equal-frequency bins (classification endpoint in the paper)


def learning_rate_for_adaptation(adaptation: str) -> float:
    """Single Adam LR: 1e-3 when the backbone is frozen, otherwise 1e-4."""
    return LEARNING_RATE_FE if adaptation == "fe" else LEARNING_RATE_FT
