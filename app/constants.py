"""Shared names and cutoffs used by training, grading, and saved-model files."""

from __future__ import annotations

OUTPUT_NAMES = ("Water", "Solids", "Bitumen")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
LABEL_EXTENSIONS = (".csv", ".txt", ".xlsx", ".xls")

# The three grades should add up to about 100. Under TIGHT we paint it green;
# over OK we paint it red so you notice a bad photo.
SUM_DEVIATION_TIGHT = 2.0
SUM_DEVIATION_OK = 5.0
