"""Default training settings used by the Train page.

These are the numbers that worked best on the froth photos in this project.
The dataset is small (about 160 lab measurements for a thousand photos), so
the recipe stays simple: freeze a pretrained backbone and train a small head
on top of it. Training a CNN from scratch on this much data barely beat
guessing the average bitumen percent.

When we split the data we keep whole flotation campaigns together, so photos
of the same pan never leak from train into test. If there is only one
campaign, the dataset falls back to a random image split.
"""

from __future__ import annotations

IMAGE_SIZE = 256
# Transfer models are resized to 384. The source photos are already 256, so
# this does not add extra detail; it just gives the last ResNet stage a
# slightly larger feature map to pool over, which helped a bit in testing.
# The from-scratch CNN stays at 256.
TRANSFER_IMAGE_SIZE = 384
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE_FT = 1e-4  # from-scratch training and fine-tuning
# Frozen backbone, head only. The head is tiny and starts at zero, so it can
# take a higher learning rate than the rest of the network.
LEARNING_RATE_FE = 3e-3
WEIGHT_DECAY = 1e-4  # AdamW L2 on backbone weights when they are being tuned
# One linear layer on a few hundred features needs more shrinkage than a
# full backbone, otherwise it memorizes the training pans.
HEAD_WEIGHT_DECAY = 0.1
# Fine-tuning: keep the backbone slow so ImageNet filters are not wiped out,
# and let the new head learn faster.
HEAD_LR_MULTIPLIER = 20.0
WARMUP_EPOCHS = 3  # ease in before the cosine decay starts
MIN_LR_FRACTION = 0.02  # cosine floor, as a fraction of the starting rate
# Smooth L1 (Huber) is a bit more forgiving than MSE when a few pans have
# unusually high bitumen.
SMOOTH_L1_BETA = 1.0
# Freeze the backbone by default. Fine-tuning was slower and less accurate.
DEFAULT_ADAPTATION = "fe"
# 20% test, then 20% of what is left for validation (~16% of everything),
# so training gets about 64%.
TEST_FRACTION = 0.20
VAL_FRACTION = 0.16
# Prefer holding out whole flotation campaigns. That is closer to grading a
# new plant run than shuffling photos.
DEFAULT_SPLIT_MODE = "experiment"
CLS_BINS = 3  # low / mid / high bins for a quick classification check


def learning_rate_for_adaptation(adaptation: str) -> float:
    """Learning rate for this run. Frozen-backbone (fe) uses 3e-3; everything else uses 1e-4."""
    return LEARNING_RATE_FE if adaptation == "fe" else LEARNING_RATE_FT


def image_size_for_architecture(architecture: str) -> int:
    """Input size this architecture was trained at.

    Training and grading both use this, so a saved model is always graded at
    the same size it saw during training.
    """
    return (
        TRANSFER_IMAGE_SIZE
        if architecture == "resnet18_tap"
        else IMAGE_SIZE
    )
