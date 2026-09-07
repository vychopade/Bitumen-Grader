"""Default training numbers. Epochs, batch size, and learning rates match the Prince and Prasad study so runs stay comparable. The rest (AdamW, Smooth L1, letterboxing, one network per grade) is what made the app usable on this data."""

from __future__ import annotations

# Same recipe for every strategy so we can compare runs fairly.
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE_FT = 1e-4  # scratch training and fine-tuning
LEARNING_RATE_FE = 1e-3  # frozen backbone, train the head only
WEIGHT_DECAY = 1e-4  # AdamW L2; 0.0 overfit the train photos in practice
# Hold out 20 percent as test. Of the rest, 20 percent of that 80 is
# validation, which is about 16 percent of everything, so train is 64.
TEST_FRACTION = 0.20
VAL_FRACTION = 0.16
# Prefer holding out whole flotation campaigns. That is closer to grading a
# new plant run than shuffling photos. If we only find one campaign we fall
# back to a random image split.
DEFAULT_SPLIT_MODE = "experiment"
CLS_BINS = 3  # three equal-frequency bins, the paper's classification check


def learning_rate_for_adaptation(adaptation: str) -> float:
    """Learning rate for this run. Frozen-backbone (fe) uses 1e-3; everything else uses 1e-4."""
    return LEARNING_RATE_FE if adaptation == "fe" else LEARNING_RATE_FT
