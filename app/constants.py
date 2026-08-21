"""Domain constants shared by dataset, training, grading, and model I/O."""
from __future__ import annotations

OUTPUT_NAMES = ("Water", "Solids", "Bitumen")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif")
LABEL_EXTENSIONS = (".csv", ".txt", ".xlsx", ".xls")
EDITED_IMAGES_DIR_NAME = "bitumengrader_edited_images"

# Water + Solids + Bitumen should be near 100%. Under TIGHT is green; over OK is red.
SUM_DEVIATION_TIGHT = 2.0
SUM_DEVIATION_OK = 5.0
